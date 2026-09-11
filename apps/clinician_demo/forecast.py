"""Run the model over one patient and shape its outputs for the replay view.

:func:`trace_patient` is the only function here that touches the model:
it tokenizes a patient's record exactly as training did and streams it
through the model once (:func:`~odyssey.inference.patient_stream.stream_patient`,
the run's own ``chunk_size``), keeping per position the risk of every
displayed event at every horizon, the 29 concept beliefs and the top
next-event predictions. Everything else is a pure function of that trace:

- the replay shows moments at the END of a same-time bundle of events (a
  lab panel is one moment, not ten), where the model has seen the whole
  bundle;
- risk is hidden at and after an event's onset, where "will it happen"
  has no meaning (the published metrics exclude those moments too);
- the alert crossing is the first bundle end whose 24 h risk reaches the
  event's alert line (:mod:`apps.clinician_demo.thresholds`).
"""

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from apps.clinician_demo.codebook import MARKER_CATEGORIES, Codebook
from apps.clinician_demo.schemas import (
    AlertCrossing,
    BankedPoint,
    NextEvent,
    OperatingPoint,
    TimelineEntry,
    VisitSummary,
    VisitTrace,
)
from apps.clinician_demo.showcase import VISIT_STATS_SCHEMA
from apps.clinician_demo.thresholds import horizon_key
from odyssey.data.alert_events import AlertEvent, all_event_times
from odyssey.data.sequences import build_patient_sequence, ordered_sequence_rows
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.patient_stream import risk_within, stream_patient
from odyssey.models.sequence_model import SequenceModel


FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

TOP_K = 5
MAX_POINTS = 1500
MAX_TIMELINE = 5000
ALERT_HORIZON_HOURS = 24.0
UNKNOWN_TOKEN = "[UNK]"
HIDDEN_TIMELINE_CATEGORIES = frozenset({"order", "billing", "demographic"})


@dataclass(frozen=True)
class RunContext:
    """The loaded run: model, tokenization artifacts and what to display."""

    model: SequenceModel
    vocab: Vocabulary
    binner: QuantileBinner | None
    source: str
    task_set: str
    chunk_size: int
    device: str
    concept_names: tuple[str, ...]
    alerts: tuple[AlertEvent, ...]
    """Displayed alert events, in display order."""
    head_index: tuple[int, ...]
    """Index of each displayed event in ``model.event_heads.event_names``."""
    horizons: tuple[float, ...]

    @property
    def events(self) -> tuple[str, ...]:
        """Displayed event names, in display order."""
        return tuple(a.name for a in self.alerts)


def displayed_alerts(
    alerts: Sequence[AlertEvent],
    head_names: Sequence[str],
    order: Sequence[str] = (),
) -> tuple[tuple[AlertEvent, ...], tuple[int, ...]]:
    """Select the events worth showing, with their hazard-head indices.

    Keeps events the model has a head for, drops next-visit events
    (30-day readmission, which is scored only at discharge and is the
    model's weakest head), and sorts by ``order`` where given (unlisted
    events keep their registry order after the listed ones).
    """
    names = list(head_names)
    kept = [a for a in alerts if not a.next_visit and a.name in names]
    rank = {name: i for i, name in enumerate(order)}
    kept.sort(key=lambda a: rank.get(a.name, len(rank)))
    return tuple(kept), tuple(names.index(a.name) for a in kept)


@dataclass(frozen=True)
class PatientTrace:
    """The model's outputs over one patient's record, one row per position."""

    subject_id: int
    tokens: list[str]
    """The value-binned token at each position (before vocabulary lookup)."""
    values: list[float | None]
    times: list[float]
    """Hours since the patient's first recorded event."""
    timestamps: list[datetime]
    visit_ids: list[int]
    n_static: int
    risk: FloatArray
    """``(N, events, horizons)`` P(event within horizon)."""
    concepts: FloatArray
    """``(N, concepts)`` running concept beliefs."""
    top_ids: IntArray
    top_probs: FloatArray
    n_unknown: int
    """Positions whose token the model's vocabulary does not know."""

    @property
    def n_positions(self) -> int:
        """Positions traced."""
        return int(self.risk.shape[0])


def trace_patient(ctx: RunContext, raw_events: pl.DataFrame) -> PatientTrace:
    """Stream one patient's normalized record through the model.

    ``raw_events`` must already be normalized the way the run's training
    was (:class:`~apps.clinician_demo.patient_store.PatientStore` does
    this); binning uses the run's own train-fit binner.

    Raises
    ------
    ValueError
        If the record is empty or the model has no concept bottleneck or
        hazard heads.
    """
    event_heads = getattr(ctx.model, "event_heads", None)
    if event_heads is None:
        raise ValueError("the demo needs a model with event hazard heads")
    binned = add_value_tokens(raw_events, ctx.binner, source=ctx.source)
    ordered = ordered_sequence_rows(binned)
    seq = build_patient_sequence(binned, ctx.vocab)
    if len(seq) == 0:
        raise ValueError("subject has no timed events to trace")
    if len(seq) != ordered.rows.height:  # the alignment invariant
        raise RuntimeError("sequence positions and raw rows are misaligned")

    idx = torch.tensor(ctx.head_index, dtype=torch.long, device=ctx.device)
    risk_parts: list[torch.Tensor] = []
    concept_parts: list[torch.Tensor] = []
    id_parts: list[torch.Tensor] = []
    prob_parts: list[torch.Tensor] = []
    ctx.model.eval()
    with torch.no_grad():
        for span in stream_patient(
            ctx.model, seq, device=ctx.device, chunk_size=ctx.chunk_size
        ):
            n, fwd = span.n_real, span.fwd
            if fwd.bottleneck is None:
                raise ValueError("the demo needs a concept-bottleneck model")
            hazards = event_heads(fwd.features[0, :n]).index_select(-2, idx)
            risk_parts.append(
                risk_within(hazards, event_heads.edges, ctx.horizons).float().cpu()
            )
            concept_parts.append(fwd.bottleneck.concept_probs[0, :n].float().cpu())
            probs = torch.softmax(fwd.logits[0, :n].float(), dim=-1)
            top_p, top_i = probs.topk(min(TOP_K, probs.shape[-1]), dim=-1)
            id_parts.append(top_i.cpu())
            prob_parts.append(top_p.cpu())

    rows = ordered.rows
    values = (
        rows["numeric_value"].to_list()
        if "numeric_value" in rows.columns
        else [None] * rows.height
    )
    unknown_id = ctx.vocab.token_to_id.get(UNKNOWN_TOKEN)
    return PatientTrace(
        subject_id=seq.subject_id,
        tokens=rows["code"].to_list(),
        values=[None if v is None or math.isnan(v) else float(v) for v in values],
        times=list(seq.time_stamps),
        timestamps=rows["time"].to_list(),
        visit_ids=list(seq.visit_ids),
        n_static=ordered.n_static,
        risk=torch.cat(risk_parts).numpy(),
        concepts=torch.cat(concept_parts).numpy(),
        top_ids=torch.cat(id_parts).numpy().astype(np.int64),
        top_probs=torch.cat(prob_parts).numpy(),
        n_unknown=sum(1 for c in seq.concept_ids if c == unknown_id),
    )


# ---------------------------------------------------------------------------
# Pure helpers over a trace
# ---------------------------------------------------------------------------


def bundle_ends(times: Sequence[float]) -> list[int]:
    """Return the positions that end a same-timestamp bundle (last of each run)."""
    n = len(times)
    return [i for i in range(n) if i == n - 1 or times[i + 1] != times[i]]


def visit_window(
    visit_ids: Sequence[int], visit_id: int, n_static: int
) -> tuple[int, int]:
    """Return the first and last non-static position carrying ``visit_id``.

    Raises
    ------
    LookupError
        If no timed position belongs to the visit.
    """
    hits = [i for i, v in enumerate(visit_ids) if v == visit_id and i >= n_static]
    if not hits:
        raise LookupError(f"visit {visit_id} has no events in this record")
    return hits[0], hits[-1]


def mask_after_onset(
    values: Sequence[float], times: Sequence[float], onset: float | None
) -> list[float | None]:
    """Replace ``values`` with ``None`` wherever ``time >= onset`` (if any)."""
    if onset is None:
        return [float(v) for v in values]
    return [None if t >= onset else float(v) for v, t in zip(values, times)]


def first_crossing(
    times: Sequence[float], values: Sequence[float | None], threshold: float
) -> float | None:
    """Return the first time a (non-masked) value reaches ``threshold``."""
    for t, v in zip(times, values):
        if v is not None and v >= threshold:
            return t
    return None


def alert_episode_start(
    times: Sequence[float],
    values: Sequence[float | None],
    threshold: float,
    end: float | None = None,
) -> float | None:
    """Return when the alert that was still on at ``end`` switched on.

    Looks at the moments before ``end`` (all moments when ``end`` is
    ``None``) that carry a value. If the last of them is at or above
    ``threshold``, walks back while the value stays there and returns the
    time the run began; otherwise no alert was on at ``end`` and the
    answer is ``None``. This is the clinically meaningful lead time: how
    long the alarm had been sounding when the event happened, not when the
    risk first brushed the line days earlier.
    """
    seen = [
        (t, v)
        for t, v in zip(times, values)
        if v is not None and (end is None or t < end)
    ]
    if not seen or seen[-1][1] < threshold:
        return None
    start = seen[-1][0]
    for t, v in reversed(seen):
        if v < threshold:
            break
        start = t
    return start


def downsample(
    times: Sequence[float],
    max_points: int,
    keep: Collection[int] = (),
) -> list[int]:
    """Return ~``max_points`` evenly spread moment indices, plus ``keep``.

    Splits the time span into ``max_points`` equal bins and keeps the last
    moment of each (the most up-to-date forecast in that bin); indices in
    ``keep`` (onsets, alert crossings, care transitions) always survive.
    """
    n = len(times)
    if n <= max_points:
        return list(range(n))
    if max_points < 1:
        return sorted(i for i in keep if 0 <= i < n)
    t0, t1 = times[0], times[-1]
    width = (t1 - t0) / max_points or 1.0
    last_in_bin: dict[int, int] = {}
    for i, t in enumerate(times):
        last_in_bin[min(int((t - t0) / width), max_points - 1)] = i
    return sorted(set(last_in_bin.values()) | {i for i in keep if 0 <= i < n})


def onsets_for(
    raw_events: pl.DataFrame,
    alerts: Sequence[AlertEvent],
    *,
    source: str,
    task_set: str,
    subject_id: int,
    visit_id: int,
) -> dict[str, float | None]:
    """Return each event's onset for this visit, in hours since the first event.

    Subject-scoped events (death) have one onset for the whole record; it
    is returned even when it falls after this visit, so the caller can say
    "not during this admission" rather than silently dropping it.
    """
    times = all_event_times(raw_events, list(alerts), source, task_set=task_set)
    out: dict[str, float | None] = {}
    for alert in alerts:
        et = times[alert.name]
        key = (subject_id, -1) if et.subject_scoped else (subject_id, visit_id)
        out[alert.name] = et.onset.get(key)
    return out


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _pct_fine(value: float) -> str:
    return f"{100 * value:.1f}%" if value < 0.1 else f"{100 * value:.0f}%"


def callout_text(
    name: str,
    *,
    cross: float | None,
    alert_start: float | None,
    onset: float | None,
) -> str:
    """Say, in plain words, what the alert line did on this visit.

    All times are visit-relative hours. ``alert_start`` is when the alert
    that was still on at the event's onset came on (see
    :func:`alert_episode_start`); ``cross`` is the first time the risk ever
    reached the line; ``onset`` is ``None`` when the event did not happen
    during the visit.
    """
    if onset is not None:
        began = f"{name} began at hour {onset:.0f}."
        if alert_start is not None:
            lead = onset - alert_start
            if lead < 1:
                return f"{began} The alert came on just before it."
            return (
                f"{began} The alert had been on since hour {alert_start:.0f}: "
                f"{lead:.0f} h of warning."
            )
        if cross is not None:
            return (
                f"{began} The alert came on at hour {cross:.0f} "
                "but was off again by then."
            )
        return f"{began} The risk never reached the alert line: a miss."
    if cross is not None:
        return (
            f"No {name} during this stay, but the alert came on at hour "
            f"{cross:.0f}: a false alarm."
        )
    return f"No {name} during this stay, and the risk stayed below the alert line."


def alert_detail(name: str, point: OperatingPoint | None) -> str:
    """Summarize how the alert line performs across held-out patients."""
    if point is None:
        return ""
    return (
        f"Alert line: {_pct_fine(point.threshold)} risk within "
        f"{ALERT_HORIZON_HOURS:g} h. Across held-out patients it is on for "
        f"{_pct(point.alert_rate)} of moments, covers {_pct(point.sensitivity)} "
        f"of the moments in the day before {name}, and {_pct(point.ppv)} of the "
        f"moments it is on are followed by {name} within a day."
    )


def _alert_index(horizons: Sequence[float]) -> int:
    return (
        list(horizons).index(ALERT_HORIZON_HOURS)
        if ALERT_HORIZON_HOURS in horizons
        else 0
    )


def _thin_timeline(entries: list[TimelineEntry], limit: int) -> list[TimelineEntry]:
    """Keep every flagged/non-vital entry; thin routine vitals evenly to fit."""
    if len(entries) <= limit:
        return entries
    keep = [e for e in entries if e.flag is not None or e.category != "vital"]
    routine = [e for e in entries if e.flag is None and e.category == "vital"]
    room = max(limit - len(keep), 0)
    step = max(len(routine) // room, 1) if room else len(routine) + 1
    kept = keep + routine[::step][:room]
    return sorted(kept, key=lambda e: e.t)


def visit_view(  # noqa: PLR0913 -- the view composes many independent inputs
    trace: PatientTrace,
    visit: VisitSummary,
    *,
    events: Sequence[str],
    horizons: Sequence[float],
    onsets: Mapping[str, float | None],
    points: Mapping[str, OperatingPoint],
    codebook: Codebook,
    decode: Callable[[int], str],
    display: Mapping[str, str],
    seen_in_training: bool,
    banked: Sequence[BankedPoint] = (),
    max_points: int = MAX_POINTS,
) -> VisitTrace:
    """Shape one visit of a trace for the replay view (visit-relative hours).

    ``points`` maps each event to its 24 h operating point (alert line).
    """
    first, last = visit_window(trace.visit_ids, visit.visit_id, trace.n_static)
    start = visit.start_hours
    ends = [i for i in bundle_ends(trace.times) if first <= i <= last]
    end_times = [trace.times[i] for i in ends]
    h_alert = _alert_index(horizons)

    alerts: list[AlertCrossing] = []
    keep: set[int] = set()
    onsets_rel: dict[str, float | None] = {}
    for j, event in enumerate(events):
        onset = onsets.get(event)
        in_visit = (
            onset is not None and trace.times[first] <= onset <= trace.times[last]
        )
        onsets_rel[event] = onset - start if in_visit and onset is not None else None
        series = mask_after_onset(
            trace.risk[ends, j, h_alert].tolist(), end_times, onset
        )
        point = points.get(event)
        cross = first_crossing(end_times, series, point.threshold) if point else None
        alert_start = (
            alert_episode_start(end_times, series, point.threshold, onset)
            if point and in_visit
            else None
        )
        for moment in (cross, alert_start):
            if moment is not None:
                keep.add(end_times.index(moment))
        if in_visit and onset is not None:
            keep.add(max(0, int(np.searchsorted(end_times, onset)) - 1))
        name = display.get(event, event)
        alerts.append(
            AlertCrossing(
                event=event,
                horizon_hours=horizons[h_alert],
                threshold=point.threshold if point else float("nan"),
                first_cross_hours=None if cross is None else cross - start,
                onset_hours=onsets_rel[event],
                lead_hours=(
                    onset - alert_start
                    if onset is not None and alert_start is not None
                    else None
                ),
                callout=callout_text(
                    name,
                    cross=None if cross is None else cross - start,
                    alert_start=None if alert_start is None else alert_start - start,
                    onset=onsets_rel[event],
                ),
                alert_start_hours=(
                    None if alert_start is None else alert_start - start
                ),
                detail=alert_detail(name, point),
            )
        )

    timeline: list[TimelineEntry] = []
    for i in range(first, last + 1):
        entry = codebook.entry(trace.tokens[i], trace.times[i] - start, trace.values[i])
        if entry.category not in HIDDEN_TIMELINE_CATEGORIES:
            timeline.append(entry)
    markers = [e for e in timeline if e.category in MARKER_CATEGORIES]
    marker_times = {m.t + start for m in markers}
    keep |= {k for k, t in enumerate(end_times) if t in marker_times}

    chosen = downsample(end_times, max_points, keep)
    positions = [ends[k] for k in chosen]
    times = [trace.times[p] for p in positions]
    risk: dict[str, dict[str, list[float | None]]] = {}
    for j, event in enumerate(events):
        risk[event] = {
            horizon_key(h): mask_after_onset(
                trace.risk[positions, j, hi].tolist(), times, onsets.get(event)
            )
            for hi, h in enumerate(horizons)
        }
    return VisitTrace(
        subject_id=trace.subject_id,
        visit=visit,
        seen_in_training=seen_in_training,
        times=[t - start for t in times],
        risk=risk,
        concepts=trace.concepts[positions].tolist(),
        top_next=[
            [
                NextEvent(
                    label=codebook.token_label(decode(int(tid))), probability=float(p)
                )
                for tid, p in zip(trace.top_ids[pos], trace.top_probs[pos])
            ]
            for pos in positions
        ],
        timeline=_thin_timeline(timeline, MAX_TIMELINE),
        markers=markers,
        onsets=onsets_rel,
        alerts=alerts,
        banked=list(banked),
    )


def visit_stats_from_trace(
    trace: PatientTrace,
    visits: Sequence[VisitSummary],
    onsets_by_visit: Mapping[int, Mapping[str, float | None]],
    *,
    events: Sequence[str],
    horizons: Sequence[float],
    thresholds: Mapping[str, float],
) -> pl.DataFrame:
    """Build gallery stats (``VISIT_STATS_SCHEMA`` rows) from a trace.

    Exact onsets, so lead times are exact (unlike the landmark-row
    estimate). Visits with no timed events are skipped.
    """
    h_alert = _alert_index(horizons)
    ends_all = bundle_ends(trace.times)
    records: list[dict[str, object]] = []
    for visit in visits:
        try:
            first, last = visit_window(trace.visit_ids, visit.visit_id, trace.n_static)
        except LookupError:
            continue
        ends = [i for i in ends_all if first <= i <= last]
        end_times = [trace.times[i] for i in ends]
        for j, event in enumerate(events):
            if event not in thresholds:
                continue
            onset = onsets_by_visit.get(visit.visit_id, {}).get(event)
            positive = (
                onset is not None and trace.times[first] <= onset <= trace.times[last]
            )
            series = mask_after_onset(
                trace.risk[ends, j, h_alert].tolist(), end_times, onset
            )
            seen = [v for v in series if v is not None]
            if not seen:
                continue  # onset at the visit's first moment: never at risk
            records.append(
                {
                    "subject_id": trace.subject_id,
                    "visit_id": visit.visit_id,
                    "event": event,
                    "positive": positive,
                    "first_cross_hours": first_crossing(
                        end_times, series, thresholds[event]
                    ),
                    "alert_start_hours": (
                        alert_episode_start(end_times, series, thresholds[event], onset)
                        if positive
                        else None
                    ),
                    "end_hours": onset if positive else end_times[-1],
                    "start_hours": trace.times[first],
                    "max_risk": max(seen),
                    "threshold": thresholds[event],
                }
            )
    return pl.DataFrame(records, schema=VISIT_STATS_SCHEMA)


__all__ = [
    "ALERT_HORIZON_HOURS",
    "MAX_POINTS",
    "PatientTrace",
    "RunContext",
    "alert_detail",
    "alert_episode_start",
    "bundle_ends",
    "callout_text",
    "displayed_alerts",
    "downsample",
    "first_crossing",
    "mask_after_onset",
    "onsets_for",
    "trace_patient",
    "visit_stats_from_trace",
    "visit_view",
    "visit_window",
]
