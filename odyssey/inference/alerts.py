"""Alert evaluation: does the general model forecast specific events in time.

The project's claim is that one sequence forecaster replaces bespoke
single-task models. This module tests that claim the way those models
are judged: for a handful of events that matter clinically (vasopressor
start, ICU admission, acute kidney injury, death), score the probability
that the event happens within a horizon (8h, 24h, 72h) from many index
times inside each visit, on held-out patients, with time-dependent
discrimination (AUROC), Brier score and calibration -- against a bespoke
gradient-boosted classifier trained on hand-built features for exactly
that event and horizon.

Design, briefly:

- **Events** are defined once (:data:`ALERT_EVENTS`): either the first
  trigger of a concept from the same rule registry the bottleneck is
  supervised with (vasopressor start, AKI), or the first occurrence of a
  code family (ICU admission, death). Death is subject-scoped (it is not
  tied to a visit); the others are visit-scoped.
- **Index times** are landmark positions: within each visit, the first
  event of every ``landmark_hours`` bucket while the patient is still at
  risk (event not yet happened) and observation continues. Positions
  after the event are not "at risk" and are excluded; positions whose
  observation ends before ``t + horizon`` without the event are
  right-censored and excluded from that horizon (administrative
  censoring; reported as ``n_censored`` so it is never hidden).
- **Model scores** at index times come from one streaming pass: for
  concept-defined events the bottleneck's concept probability (the
  model's running belief), and for every event the next-event
  probability mass on the event's own tokens -- neither a calibrated
  "within h" probability, so scored on AUROC only -- and, for models
  trained with per-event hazard heads
  (:class:`~odyssey.models.time_to_event.EventHazardHeads`), the head's
  own ``P(event within h)``, a calibrated probability scored on AUROC,
  Brier and calibration exactly like the baseline.
- **Baseline** (:func:`fit_baselines`): per event and horizon, a
  ``HistGradientBoostingClassifier`` fitted on training shards and scored
  on the same held-out index times. Two feature sets: ``basic`` (latest
  clinical bin of each curated vital/lab, 24h family counts, hours into
  the visit, age; the original) and ``strong`` (default; the best-effort
  panel of :mod:`odyssey.inference.baseline_features`: raw values with
  window statistics and trends for ~50 vitals/labs, drug-class exposure,
  ICU/visit context), with a small hyper-parameter search on a
  subject-grouped validation split. The paper comparison uses ``strong``.

Everything patient-level stays in memory or under gitignored paths.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from odyssey.data.alert_events import (
    ALERT_EVENTS,
    ALERT_TASK_SETS,
    AlertEvent,
    EventTimes,
    alert_events_for,
    all_event_times,
    hours_since_origin,
    origin_hours,
)
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.sequences import BIRTH_CODE
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import (
    QuantileBinner,
    add_value_tokens,
    clinical_ranges_for_source,
)
from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.inference.baseline_features import StrongFeatureBuilder
from odyssey.inference.baseline_features import feature_names as strong_feature_names
from odyssey.inference.run_inference import load_run, refuse_existing_output
from odyssey.models.sequence_model import SequenceModel
from odyssey.models.time_to_event import probability_within
from odyssey.training.data import (
    iter_patient_sequences,
    load_meds_shard,
    load_meds_shards,
)
from odyssey.training.shard_stream import (
    Preparer,
    make_preparer,
    merge_event_times,
    shard_paths,
)
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

#: Version of the landmark-selection protocol used to build the model-scored
#: IndexRow sets :func:`collect_model_scores` produces (and, downstream,
#: whatever gets dumped/reported from them). Bump this and extend the
#: comment below whenever landmark selection itself changes -- not for
#: unrelated changes to scoring, features, or baselines.
#:
#: v1 (every dump/results.json written before 2026-08-21): _landmark_mask
#: was called fresh per streaming chunk with no state carried across chunk
#: boundaries -- a patient whose sequence spanned more than one chunk got a
#: spurious extra landmark row at the boundary (confirmed ~23% over-count
#: at eICU scale; review findings 8/19).
#: v2 (2026-08-21 to 2026-08-23): LandmarkState threaded across chunks,
#: fixing the above -- but still decided "new landmark" by comparing only
#: to the immediately-preceding token position (``~same_visit``). A
#: patient's own token order can legitimately interleave two different
#: visits at one shared timestamp (e.g. a discharge instant that stops
#: medication orders under both an ending and a starting admission id) --
#: each interleave step toggled ``~same_visit`` and re-triggered a
#: landmark even though the bucket had already been landmarked for that
#: visit, an artifact confirmed at ~1.4% of rows on a real eICU repro.
#: Affected every backbone (the row-construction path in
#: collect_model_scores is unconditional on ``packed``), not just
#: backbone="transformer".
#: v3 (current): landmark selection tracks, per lane, the last bucket
#: already emitted per visit (cleared at subject boundaries) and requires
#: a genuinely new bucket for THAT visit -- matching
#: _index_rows_from_events' per-(subject, visit, bucket) group-by
#: semantics exactly, order-of-arrival no longer matters.
#:
#: _index_rows_from_events (the model-free path used for baseline fitting)
#: was never affected -- it has no chunking at all -- so this version only
#: describes collect_model_scores' output, and only that output carries
#: the tag (see _stamp_landmark_protocol_version, load_index_row_table).
LANDMARK_PROTOCOL_VERSION = 3

HORIZONS_HOURS: Tuple[float, ...] = (8.0, 24.0, 72.0)

# How index rows are chosen. "landmark": every ``landmark_hours`` bucket
# within each visit (the alerts protocol). "visit_end": one row per visit at
# its last event -- the discharge-anchored scheme the 30-day readmission
# task uses (horizons in days, e.g. 168h / 720h).
INDEX_MODES: Tuple[str, ...] = ("landmark", "visit_end")
READMISSION_HORIZONS_HOURS: Tuple[float, ...] = (168.0, 720.0)


def _check_index_mode(index_mode: str) -> None:
    if index_mode not in INDEX_MODES:
        raise ValueError(f"index_mode must be one of {INDEX_MODES}, got {index_mode!r}")


# ---------------------------------------------------------------------------
# Index times and outcomes
# ---------------------------------------------------------------------------


@dataclass
class IndexRow:
    """One (subject, visit, time) at which risk is assessed."""

    subject_id: int
    visit_id: int
    time_hours: float
    scores: Dict[str, float] = field(default_factory=dict)
    """scorer name -> risk score for the event under evaluation."""
    is_tail: bool = False
    """True if this subject's history was truncated to fit max_context
    (backbone="transformer" only; see PackedContextSampler.truncated_
    subject_ids). Lets score_alerts report a separate slice for
    truncated patients instead of pooling them into the headline
    numbers -- see the module's tail-slice reporting."""


def outcome_at_horizon(
    row: IndexRow, times: EventTimes, horizon: float
) -> Optional[int]:
    """Return the horizon outcome for one index row.

    1 if the event occurs in ``(t, t+h]``, 0 if observation reaches
    ``t+h`` without it, None if censored (observation ends first) or the
    row is not at risk (the event already happened).
    """
    key = (row.subject_id, -1 if times.subject_scoped else row.visit_id)
    onset = times.onset.get(key)
    if onset is not None and onset <= row.time_hours:
        return None  # already happened: not at risk
    if onset is not None and onset <= row.time_hours + horizon:
        return 1
    censor = times.censor.get(key)
    if censor is None or censor < row.time_hours + horizon:
        return None  # follow-up ends before the horizon
    return 0


@dataclass
class LandmarkState:
    """Per-lane last-emitted-bucket-per-visit state, threaded across chunks.

    ``last_bucket_by_lane[lane]`` maps ``visit_id -> the last bucket a
    landmark was already emitted for``, scoped to that lane's CURRENT
    subject -- cleared the moment a lane's subject_id changes.
    ``subject_by_lane[lane]`` is only that lane's most recently seen real
    subject_id (``NO_SUBJECT`` if the lane has never seen a real position
    yet), used to detect that transition. Threaded across streaming
    chunks the same role the backbone's own recurrent state plays for
    model weights (see :func:`collect_model_scores`).
    """

    last_bucket_by_lane: List[Dict[int, int]]
    subject_by_lane: List[int]


def _landmark_mask(
    time_hours: torch.Tensor,
    subject_ids: torch.Tensor,
    visit_ids: torch.Tensor,
    landmark_hours: float,
    visit_start_hours: torch.Tensor,
    *,
    state: Optional[LandmarkState] = None,
) -> Tuple[torch.Tensor, LandmarkState]:
    """First position of each (subject, visit, bucket) triple, across chunks.

    Matches :func:`_index_rows_from_events`' per-(subject, visit, bucket)
    group-by semantics exactly: a position is a landmark iff its bucket
    has not already been landmarked for that visit (within the lane's
    current subject), regardless of what token immediately precedes it in
    the stream.

    Two real bugs this has fixed, in order:

    - v1->v2 (2026-08-21): an earlier version had no ``state`` parameter
      at all -- called fresh per chunk, the first position of every chunk
      was unconditionally treated as a new landmark, so any patient whose
      sequence spans more than one streaming chunk got a spurious extra
      landmark row at every chunk boundary (~23% over-count at eICU
      scale; ``tests/odyssey/inference/test_alerts.py``'s regression
      test).
    - v2->v3 (2026-08-23): even with state carried across chunks, v2
      compared each position only to the *immediately preceding* one
      (``~same_visit``) -- so a patient whose own token order legitimately
      interleaves two different visits at one shared timestamp (a
      discharge instant stopping medication orders under both an ending
      and a starting admission id is a real, observed pattern) got a
      fresh spurious landmark on every interleave step, even though that
      visit's bucket had already been landmarked. Confirmed at ~1.4% of
      rows on a real eICU repro; affected every backbone, not just
      backbone="transformer" (the row-construction path in
      :func:`collect_model_scores` is unconditional on ``packed``).

    ``state`` (``None`` on the first call) carries each lane's per-visit
    bucket map forward; without it, every lane starts with no visits
    known yet (matching the original, single-chunk-only behavior) rather
    than crashing.

    Returns
    -------
    tuple[torch.Tensor, LandmarkState]
        The boolean landmark mask, and the updated state to pass into the
        next call.
    """
    lanes, chunk_len = time_hours.shape
    bucket = torch.floor((time_hours - visit_start_hours) / landmark_hours)

    if state is None:
        last_bucket_by_lane: List[Dict[int, int]] = [{} for _ in range(lanes)]
        subject_by_lane: List[int] = [NO_SUBJECT] * lanes
    else:
        last_bucket_by_lane = state.last_bucket_by_lane
        subject_by_lane = list(state.subject_by_lane)

    bucket_rows = bucket.tolist()
    subject_rows = subject_ids.tolist()
    visit_rows = visit_ids.tolist()
    mask = torch.zeros((lanes, chunk_len), dtype=torch.bool)

    for lane in range(lanes):
        lane_buckets = last_bucket_by_lane[lane]
        current_subject = subject_by_lane[lane]
        for pos in range(chunk_len):
            subject_id = subject_rows[lane][pos]
            visit_id = visit_rows[lane][pos]
            if subject_id == NO_SUBJECT or visit_id < 0:
                continue  # padding, or a static/demographic token
            if subject_id != current_subject:
                lane_buckets = {}
                last_bucket_by_lane[lane] = lane_buckets
                current_subject = subject_id
            this_bucket = int(bucket_rows[lane][pos])
            if lane_buckets.get(visit_id) != this_bucket:
                mask[lane, pos] = True
                lane_buckets[visit_id] = this_bucket
        subject_by_lane[lane] = current_subject

    new_state = LandmarkState(
        last_bucket_by_lane=last_bucket_by_lane, subject_by_lane=subject_by_lane
    )
    return mask, new_state


# ---------------------------------------------------------------------------
# Model scores at index times
# ---------------------------------------------------------------------------


def _event_token_mask(
    vocab: Vocabulary, alert: AlertEvent, device: str
) -> torch.Tensor:
    pattern = alert.token_regex or (
        f"^{re.escape(alert.code_prefix)}" if alert.code_prefix else None
    )
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    if pattern is None:
        return mask.to(device)
    rx = re.compile(pattern, re.IGNORECASE)
    for token_id, token in vocab.id_to_token.items():
        if rx.search(token):
            mask[token_id] = True
    return mask.to(device)


def _select_index_positions(
    index_mode: str,
    chunk: StreamingChunk,
    *,
    times: torch.Tensor,
    sids: torch.Tensor,
    vids: torch.Tensor,
    landmark_hours: float,
    starts: torch.Tensor,
    landmark_state: Optional[LandmarkState],
) -> Tuple[torch.Tensor, Optional[LandmarkState]]:
    """Positions to score in this chunk, per :data:`INDEX_MODES`.

    ``visit_end``: the last position of each real (hadm-bearing) visit, as
    marked by the tokenizer (``PatientSequence.visit_ends``) and carried
    per chunk -- the same flag visit-scoped concept supervision uses, so
    "discharge instant" has one definition. ``landmark``: the per-bucket
    first positions via :func:`_landmark_mask`, state threaded across
    chunks.
    """
    if index_mode == "visit_end":
        return chunk.visit_end & (vids >= 0), landmark_state
    return _landmark_mask(
        times, sids, vids, landmark_hours, starts, state=landmark_state
    )


def collect_model_scores(
    model: SequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_names: Sequence[str],
    alerts: Sequence[AlertEvent],
    *,
    visit_start: Dict[Tuple[int, int], float],
    landmark_hours: float = 4.0,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    horizons: Sequence[float] = HORIZONS_HOURS,
    backbone: str = "hybrid",
    max_context: int = 4096,
    truncation_boundaries_out: Optional[Dict[int, float]] = None,
    index_mode: str = "landmark",
) -> Dict[str, List[IndexRow]]:
    """One streaming pass; per alert, index rows with model risk scores.

    ``index_mode="visit_end"`` selects each visit's last position instead
    of landmark buckets (see :data:`INDEX_MODES`); ``landmark_hours`` is
    then unused.

    ``truncation_boundaries_out``, if given, is populated in place with
    :attr:`~odyssey.data.packed_context.PackedContextSampler.truncation_boundaries`
    once streaming finishes -- captured from the sampler directly, not
    re-derived from the returned rows later. :func:`verify_packed_landmark_rows`
    needs this: deriving a truncated subject's boundary from ``IndexRow``
    values already in hand is circular the moment those rows are
    themselves what is being checked for correctness (confirmed the hard
    way -- an earlier version of this function's own caller re-derived
    the boundary from the row set, so removing the very row a bug should
    have dropped also silently moved what "the boundary" was computed to
    be, hiding exactly the bug being tested for).

    Scores per row: ``concept`` (the alert's concept probability, if it
    has one), ``next_mass`` (softmax mass on the alert's tokens), and,
    when the model has per-event hazard heads covering the alert,
    ``hazard@{h}h`` for each horizon in ``horizons``: the head's
    ``P(event within h)``, a calibrated probability.

    ``backbone="transformer"`` streams through
    :class:`~odyssey.data.packed_context.PackedContextSampler` instead of
    the TBTT :class:`~odyssey.data.streaming.PackedLaneSampler`.
    Landmark selection (:func:`_landmark_mask`) is reset fresh every
    chunk in that case (``landmark_state`` never carried across calls):
    ``PackedLaneSampler``'s lanes are persistent streams where the same
    lane index continues the same patient across chunks, so
    ``landmark_state`` has to carry that patient's last position forward
    (see :data:`LANDMARK_PROTOCOL_VERSION`'s v1->v2 fix). A
    ``PackedContextSampler`` row is a self-contained, complete (or
    head-truncated) set of whole patients with no continuation into the
    next chunk at all -- carrying state across such a boundary would
    incorrectly compare one chunk's last row to an unrelated next row.
    Every row already contains each of its patients' *whole* visible
    history in one pass, so resetting per chunk still sees every
    landmark within it; this is what makes the packed path's own
    landmark set provably equal to :func:`_index_rows_from_events`'
    model-free selection (see ``tests/odyssey/inference/test_alerts.py``).
    Rows belonging to a subject the sampler truncated are marked
    ``IndexRow.is_tail=True``.
    """
    _check_index_mode(index_mode)
    model.eval()
    concept_index = {name: i for i, name in enumerate(concept_names)}
    token_masks = {a.name: _event_token_mask(vocab, a, device) for a in alerts}
    event_heads = getattr(model, "event_heads", None)
    head_index = (
        {name: i for i, name in enumerate(event_heads.event_names)}
        if event_heads is not None
        else {}
    )
    rows: Dict[str, List[IndexRow]] = {a.name: [] for a in alerts}
    patients = iter_patient_sequences(
        events_binned, vocab, signal_panel=getattr(model, "signal_panel", None)
    )
    packed = backbone == "transformer"
    sampler: Union[PackedLaneSampler, PackedContextSampler] = (
        PackedContextSampler(patients, batch_size=num_lanes, max_context=max_context)
        if packed
        else PackedLaneSampler(
            patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
        )
    )

    state = None
    landmark_state: Optional[LandmarkState] = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            if packed:
                landmark_state = None  # see the docstring: never carried
            fwd = model.forward_with_features(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            logits, state = fwd.logits, fwd.state
            sids = chunk.subject_ids
            vids = chunk.visit_ids
            times = chunk.batch.aux.time_stamps
            # Packed-path timestamps are already in the true frame (see
            # packed_context._truncate_head): no un-rebasing.
            # visit start per position, via the unique (subject, visit)
            # keys in this chunk (a few hundred lookups, not one per token)
            keys = torch.stack([sids, vids], dim=-1).reshape(-1, 2)
            unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
            unique_starts = torch.tensor(
                [
                    visit_start.get((int(s), int(v)), 0.0)
                    for s, v in unique_keys.tolist()
                ],
                dtype=times.dtype,
                device=times.device,
            )
            starts = unique_starts[inverse].view_as(times)
            # Static/demographic tokens build_patient_sequence prepends
            # (GENDER etc., stamped with the first real event's time) carry
            # visit_id=-1, not a real encounter -- _landmark_mask's own
            # `visit_ids >= 0` guard already excludes them from ever being
            # selected as a landmark row, so they don't corrupt this set on
            # their own; ruled out as a cause when tracking down the
            # cross-chunk divergence _landmark_mask's own docstring
            # describes. For the lane path, landmark_state is threaded
            # across chunks the same way the model's own recurrent `state`
            # is, immediately above -- without it, a patient's sequence
            # spanning more than one chunk would get a spurious extra
            # landmark at the chunk boundary. For the packed path it was
            # just reset to None above instead (see this function's
            # docstring for why that is the correct choice there, not a
            # gap).
            keep, landmark_state = _select_index_positions(
                index_mode,
                chunk,
                times=times,
                sids=sids,
                vids=vids,
                landmark_hours=landmark_hours,
                starts=starts,
                landmark_state=landmark_state,
            )
            if not keep.any():
                continue
            probs = torch.softmax(logits[keep], dim=-1)
            hazards = (
                event_heads(fwd.features[keep]) if event_heads is not None else None
            )
            kept_sids = sids[keep].tolist()
            kept_vids = vids[keep].tolist()
            # times are in the one shared "hours since this subject's true
            # first event" frame everything else here (outcome_at_horizon,
            # EventTimes, _index_rows_from_events) assumes -- for both
            # samplers; no adjustment.
            kept_times = times[keep].tolist()
            truncated_ids = (
                set(sampler.truncation_boundaries)
                if packed and isinstance(sampler, PackedContextSampler)
                else set()
            )
            for alert in alerts:
                mass = probs[:, token_masks[alert.name]].sum(dim=-1).tolist()
                concept_p = (
                    fwd.bottleneck.concept_probs[keep][
                        :, concept_index[alert.concept]
                    ].tolist()
                    if fwd.bottleneck is not None and alert.concept in concept_index
                    else None
                )
                within: Dict[str, List[float]] = {}
                if (
                    hazards is not None
                    and event_heads is not None
                    and alert.name in head_index
                ):
                    head_logits = hazards[:, head_index[alert.name]]
                    for h in horizons:
                        within[f"hazard@{h:g}h"] = probability_within(
                            head_logits, event_heads.edges, h
                        ).tolist()
                for k, (s, v, t) in enumerate(zip(kept_sids, kept_vids, kept_times)):
                    scores = {"next_mass": float(mass[k])}
                    if concept_p is not None:
                        scores["concept"] = float(concept_p[k])
                    for label, values in within.items():
                        scores[label] = float(values[k])
                    rows[alert.name].append(
                        IndexRow(
                            int(s), int(v), float(t), scores, is_tail=s in truncated_ids
                        )
                    )
    _export_truncation_boundaries(sampler, truncation_boundaries_out)
    return rows


@dataclass
class _LaneCarry:
    """Last real position of a lane's continuing subject (previous chunk)."""

    subject_id: int
    time: float
    logits: torch.Tensor
    features: torch.Tensor
    concept_probs: Optional[torch.Tensor]


def _scores_at(
    logits: torch.Tensor,
    features: torch.Tensor,
    concept_probs: Optional[torch.Tensor],
    *,
    alerts: Sequence[AlertEvent],
    token_masks: Dict[str, torch.Tensor],
    concept_index: Dict[str, int],
    event_heads: Optional[Any],
    head_index: Dict[str, int],
    horizons: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Per-alert score dict for one position (``logits``/``features`` are 1-D)."""
    probs = torch.softmax(logits, dim=-1)
    hazards = event_heads(features.unsqueeze(0)) if event_heads is not None else None
    out: Dict[str, Dict[str, float]] = {}
    for alert in alerts:
        scores = {"next_mass": float(probs[token_masks[alert.name]].sum())}
        if concept_probs is not None and alert.concept in concept_index:
            scores["concept"] = float(concept_probs[concept_index[alert.concept]])
        if hazards is not None and event_heads is not None and alert.name in head_index:
            head_logits = hazards[:, head_index[alert.name]]
            for h in horizons:
                scores[f"hazard@{h:g}h"] = float(
                    probability_within(head_logits, event_heads.edges, h)[0]
                )
        out[alert.name] = scores
    return out


def collect_model_scores_at_rows(  # noqa: PLR0912, PLR0915
    model: SequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_names: Sequence[str],
    alerts: Sequence[AlertEvent],
    *,
    index_rows: Sequence[IndexRow],
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    horizons: Sequence[float] = HORIZONS_HOURS,
    backbone: str = "hybrid",
    max_context: int = 4096,
    truncation_boundaries_out: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[str, List[IndexRow]], int]:
    """Score the model at GIVEN index rows (the missingness protocol's path).

    ``index_rows`` are (subject, visit, time) triples fixed by the CLEAN
    record (a dump); the model is run over ``events_binned`` -- possibly a
    degraded copy -- and each row is scored at the first token charted AT
    the row's time when one exists (the landmark convention), else at the
    last visible token BEFORE it: what the model knows as of that instant,
    given this record. Rows with no visible token at or before their time
    are unscoreable and counted (second return value); the returned rows
    keep the given keys exactly, so the row set is identical across
    degradation cells by construction (docs/missingness_protocol.md,
    Principle 3), whatever the degraded record's own landmark grid would be.
    For the lane sampler the last position of a continuing subject is
    carried across chunks so "the last visible token before t" is exact
    across chunk boundaries.
    """
    model.eval()
    concept_index = {name: i for i, name in enumerate(concept_names)}
    token_masks = {a.name: _event_token_mask(vocab, a, device) for a in alerts}
    event_heads = getattr(model, "event_heads", None)
    head_index = (
        {name: i for i, name in enumerate(event_heads.event_names)}
        if event_heads is not None
        else {}
    )
    # targets per subject, ascending time (dedupe identical keys)
    targets: Dict[int, List[Tuple[int, float]]] = {}
    seen_keys: set[Tuple[int, int, float]] = set()
    for r in index_rows:
        key = (r.subject_id, r.visit_id, r.time_hours)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        targets.setdefault(r.subject_id, []).append((r.visit_id, r.time_hours))
    for target_list in targets.values():
        target_list.sort(key=lambda x: x[1])
    pointer: Dict[int, int] = dict.fromkeys(targets, 0)
    rows: Dict[str, List[IndexRow]] = {a.name: [] for a in alerts}
    unscoreable = 0

    patients = iter_patient_sequences(
        events_binned, vocab, signal_panel=getattr(model, "signal_panel", None)
    )
    packed = backbone == "transformer"
    sampler: Union[PackedLaneSampler, PackedContextSampler] = (
        PackedContextSampler(patients, batch_size=num_lanes, max_context=max_context)
        if packed
        else PackedLaneSampler(
            patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
        )
    )
    carry: Dict[int, _LaneCarry] = {}

    def emit(
        sid: int,
        vid: int,
        t: float,
        outputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        *,
        is_tail: bool,
    ) -> None:
        per_alert = _scores_at(
            *outputs,
            alerts=alerts,
            token_masks=token_masks,
            concept_index=concept_index,
            event_heads=event_heads,
            head_index=head_index,
            horizons=horizons,
        )
        for alert in alerts:
            rows[alert.name].append(
                IndexRow(sid, vid, t, per_alert[alert.name], is_tail=is_tail)
            )

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            fwd = model.forward_with_features(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            state = fwd.state
            sids_all = chunk.subject_ids
            times_all = chunk.batch.aux.time_stamps
            ends_all = chunk.patient_end
            cps_all = (
                fwd.bottleneck.concept_probs if fwd.bottleneck is not None else None
            )
            truncated_ids = (
                set(sampler.truncation_boundaries)
                if packed and isinstance(sampler, PackedContextSampler)
                else set()
            )
            for lane in range(sids_all.shape[0]):
                sids = sids_all[lane].tolist()
                times = times_all[lane].tolist()
                ends = ends_all[lane].tolist()
                prev: Optional[Tuple[int, float, int]] = (
                    None  # (sid, time, pos) in chunk
                )
                for i, sid in enumerate(sids):
                    if sid == NO_SUBJECT:
                        break
                    t_i = float(times[i])
                    lst: Optional[List[Tuple[int, float]]] = targets.get(sid)
                    if lst is None:
                        prev = (sid, t_i, i)
                        continue
                    # previous visible token of THIS subject: in-chunk or carried
                    prev_time: Optional[float] = None
                    prev_out: Optional[
                        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
                    ] = None
                    if prev is not None and prev[0] == sid:
                        prev_time = prev[1]
                        prev_out = (
                            fwd.logits[lane, prev[2]],
                            fwd.features[lane, prev[2]],
                            cps_all[lane, prev[2]] if cps_all is not None else None,
                        )
                    elif i == 0 and lane in carry and carry[lane].subject_id == sid:
                        c = carry[lane]
                        prev_time = c.time
                        prev_out = (c.logits, c.features, c.concept_probs)
                    here = (
                        fwd.logits[lane, i],
                        fwd.features[lane, i],
                        cps_all[lane, i] if cps_all is not None else None,
                    )
                    ptr = pointer[sid]
                    # targets strictly before this token: score at the previous token
                    while ptr < len(lst) and lst[ptr][1] < t_i:
                        if prev_out is None:
                            unscoreable += 1
                        else:
                            emit(
                                sid,
                                lst[ptr][0],
                                lst[ptr][1],
                                prev_out,
                                is_tail=sid in truncated_ids,
                            )
                        ptr += 1
                    # targets AT this token's time: the first token of the instant
                    if prev_time is None or prev_time != t_i:
                        while ptr < len(lst) and lst[ptr][1] == t_i:
                            emit(
                                sid,
                                lst[ptr][0],
                                t_i,
                                here,
                                is_tail=sid in truncated_ids,
                            )
                            ptr += 1
                    if ends[i]:
                        # subject over: remaining targets are at/after the last token
                        while ptr < len(lst):
                            emit(
                                sid,
                                lst[ptr][0],
                                lst[ptr][1],
                                here,
                                is_tail=sid in truncated_ids,
                            )
                            ptr += 1
                    pointer[sid] = ptr
                    prev = (sid, t_i, i)
                # carry the lane's last real position for a continuing subject
                if prev is not None and not bool(ends[prev[2]]):
                    carry[lane] = _LaneCarry(
                        prev[0],
                        prev[1],
                        fwd.logits[lane, prev[2]].clone(),
                        fwd.features[lane, prev[2]].clone(),
                        cps_all[lane, prev[2]].clone() if cps_all is not None else None,
                    )
                elif lane in carry:
                    del carry[lane]
    # subjects never seen at all (absent from the record): all their rows unscoreable
    for sid, lst in targets.items():
        unscoreable += len(lst) - pointer[sid]
    _export_truncation_boundaries(sampler, truncation_boundaries_out)
    return rows, unscoreable


def _export_truncation_boundaries(
    sampler: Union[PackedLaneSampler, PackedContextSampler],
    truncation_boundaries_out: Optional[Dict[int, float]],
) -> None:
    """Copy a finished sampler's truncation boundaries into the caller's dict."""
    if truncation_boundaries_out is not None and isinstance(
        sampler, PackedContextSampler
    ):
        truncation_boundaries_out.update(sampler.truncation_boundaries)


# ---------------------------------------------------------------------------
# Baseline features and models
# ---------------------------------------------------------------------------

_BIN_ORDINAL = {"LOW": -1.0, "NORMAL": 0.0, "HIGH": 1.0, "CRITICAL": 2.0}
_FAMILY_IDS = (
    1,
    2,
    3,
    4,
    5,
    7,
)  # diagnosis, medication, procedure, lab, visit, billing


class _SubjectHistory:
    """One subject's timed events, preprocessed for O(log n) feature lookups."""

    def __init__(
        self,
        hours: np.ndarray,
        codes: List[str],
        hadms: List[Optional[int]],
        prefixes: Sequence[str],
    ) -> None:
        self.hours = hours
        self.hadms = np.array([-1 if h is None else int(h) for h in hadms])
        family_index = {f: i for i, f in enumerate(_FAMILY_IDS)}
        one_hot = np.zeros((len(codes), len(_FAMILY_IDS)), dtype=np.int32)
        # per curated prefix: hours and ordinal bins of matching events
        prefix_hours: List[List[float]] = [[] for _ in prefixes]
        prefix_bins: List[List[float]] = [[] for _ in prefixes]
        for i, code in enumerate(codes):
            f_idx = family_index.get(code_type(code))
            if f_idx is not None:
                one_hot[i, f_idx] = 1
            if "::" not in code:
                continue
            base, bin_ = code.rsplit("::", 1)
            ordinal = _BIN_ORDINAL.get(bin_)
            if ordinal is None:
                continue
            for p_idx, prefix in enumerate(prefixes):
                if base.startswith(prefix):
                    prefix_hours[p_idx].append(float(hours[i]))
                    prefix_bins[p_idx].append(ordinal)
        self.cum_counts = np.vstack(
            [np.zeros((1, len(_FAMILY_IDS)), dtype=np.int32), one_hot.cumsum(axis=0)]
        )
        self.prefix_hours = [np.array(h, dtype=np.float64) for h in prefix_hours]
        self.prefix_bins = [np.array(b, dtype=np.float32) for b in prefix_bins]

    def visit_start(self, visit_id: int, fallback: float) -> float:
        mask = self.hadms == visit_id
        return float(self.hours[mask].min()) if mask.any() else fallback

    def features_at(self, now: float, out: np.ndarray, offset: int) -> None:
        """Fill latest-bin and trailing-24h-count features into ``out``."""
        for p_idx, (ph, pb) in enumerate(zip(self.prefix_hours, self.prefix_bins)):
            k = int(np.searchsorted(ph, now, side="left"))
            if k > 0:
                out[offset + p_idx] = pb[k - 1]
        n_prefix = len(self.prefix_hours)
        hi = int(np.searchsorted(self.hours, now, side="left"))
        lo = int(np.searchsorted(self.hours, now - 24.0, side="left"))
        out[offset + n_prefix :] = self.cum_counts[hi] - self.cum_counts[lo]


def baseline_features(
    events_binned: pl.DataFrame,
    index_rows: Sequence[IndexRow],
    *,
    source: str = "mimic_iv",
) -> np.ndarray:
    """Hand-built features at each index row, from events strictly before it.

    Columns: hours since visit start, age (years, if birth known), latest
    ordinal clinical bin per curated vital/lab prefix (NaN if never), and
    trailing-24h event counts per code family. All computable by hand
    from the same record; nothing from the model. Each subject's history
    is preprocessed once (cumulative family counts, per-prefix bin
    arrays), so each row costs a few binary searches.
    """
    ranges, _ = clinical_ranges_for_source(source)
    prefixes = sorted(ranges)
    origins = origin_hours(events_binned)
    timed = events_binned.filter(pl.col("time").is_not_null()).join(
        origins, on="subject_id", how="left"
    )
    timed = timed.with_columns(
        ((pl.col("time") - pl.col("_origin")).dt.total_seconds() / 3600.0).alias(
            "_hours"
        )
    )
    birth = (
        events_binned.filter(pl.col("code") == BIRTH_CODE)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_birth"))
    )
    birth_map = dict(zip(birth["subject_id"].to_list(), birth["_birth"].to_list()))
    origin_map = dict(
        zip(origins["subject_id"].to_list(), origins["_origin"].to_list())
    )
    needed = {row.subject_id for row in index_rows}
    histories: Dict[int, _SubjectHistory] = {}
    for key, frame in timed.select("subject_id", "_hours", "code", "hadm_id").group_by(
        "subject_id", maintain_order=True
    ):
        sid = int(key[0])
        if sid not in needed:
            continue
        ordered = frame.sort("_hours")
        histories[sid] = _SubjectHistory(
            ordered["_hours"].to_numpy(),
            ordered["code"].to_list(),
            ordered["hadm_id"].to_list(),
            prefixes,
        )

    n_feat = 2 + len(prefixes) + len(_FAMILY_IDS)
    out = np.full((len(index_rows), n_feat), np.nan, dtype=np.float32)
    visit_start_cache: Dict[Tuple[int, int], float] = {}
    for r, row in enumerate(index_rows):
        history = histories.get(row.subject_id)
        if history is None:
            continue
        key = (row.subject_id, row.visit_id)
        if key not in visit_start_cache:
            visit_start_cache[key] = history.visit_start(row.visit_id, row.time_hours)
        out[r, 0] = row.time_hours - visit_start_cache[key]
        b = birth_map.get(row.subject_id)
        o = origin_map.get(row.subject_id)
        if b is not None and o is not None:
            out[r, 1] = ((o - b).total_seconds() / 3600.0 + row.time_hours) / (
                24 * 365.25
            )
        history.features_at(row.time_hours, out[r], 2)
    return out


# ``basic`` = the original hand features (latest clinical bin per curated
# signal, 24h family counts, hours into visit, age); ``strong`` = the
# best-effort panel of :mod:`odyssey.inference.baseline_features`.
# "strong_text" = the strong panel plus note-embedding features from the
# active note_embeddings sidecar (odyssey.text.note_features): the
# text-modality probe (Track A item 7).
BASELINE_FEATURE_SETS: Tuple[str, ...] = ("basic", "strong", "strong_text")


def strong_baseline_features(
    events_binned: pl.DataFrame,
    index_rows: Sequence[IndexRow],
    *,
    source: str = "mimic_iv",
) -> np.ndarray:
    """Build the best-effort feature matrix (see ``baseline_features``)."""
    builder = StrongFeatureBuilder(events_binned, source=source)
    return builder.features(
        [r.subject_id for r in index_rows],
        [r.visit_id for r in index_rows],
        [r.time_hours for r in index_rows],
    )


def strong_text_baseline_features(
    events_binned: pl.DataFrame,
    index_rows: Sequence[IndexRow],
    *,
    source: str = "mimic_iv",
) -> np.ndarray:
    """Strong panel + note-embedding block (needs the note_embeddings sidecar)."""
    from odyssey.text.note_features import note_features_for_rows  # noqa: PLC0415

    strong = strong_baseline_features(events_binned, index_rows, source=source)
    starts = _visit_starts(events_binned)
    notes, _ = note_features_for_rows(
        events_binned,
        [r.subject_id for r in index_rows],
        [starts.get((r.subject_id, r.visit_id), r.time_hours) for r in index_rows],
        [r.time_hours for r in index_rows],
    )
    return np.concatenate([strong, notes], axis=1)


def features_for_events(
    events_binned: pl.DataFrame,
    rows: Dict[str, List[IndexRow]],
    *,
    source: str = "mimic_iv",
    feature_set: str = "strong",
) -> Dict[str, np.ndarray]:
    """Baseline features per event, computed once over the union of index rows.

    Index rows are the same landmarks for every event (only the outcome
    differs), so features are computed once and re-indexed per event.
    ``feature_set`` is one of :data:`BASELINE_FEATURE_SETS`.
    """
    if feature_set not in BASELINE_FEATURE_SETS:
        raise ValueError(f"unknown baseline feature set {feature_set!r}")
    unique: Dict[Tuple[int, int, float], int] = {}
    union: List[IndexRow] = []
    for event_rows in rows.values():
        for r in event_rows:
            k = (r.subject_id, r.visit_id, r.time_hours)
            if k not in unique:
                unique[k] = len(union)
                union.append(r)
    if not union:
        return {}
    if feature_set == "strong":
        feats = strong_baseline_features(events_binned, union, source=source)
    elif feature_set == "strong_text":
        feats = strong_text_baseline_features(events_binned, union, source=source)
    else:
        feats = baseline_features(events_binned, union, source=source)
    return {
        name: feats[
            [unique[(r.subject_id, r.visit_id, r.time_hours)] for r in event_rows]
        ]
        for name, event_rows in rows.items()
        if event_rows
    }


def _positive_class_proba(clf: object, proba: np.ndarray) -> np.ndarray:
    """Positive-class (label ``1``) column of a classifier's ``predict_proba`` output.

    Shared by :class:`~odyssey.inference.ebm_baseline.EBMBaselineModel` and
    :class:`~odyssey.inference.tabicl_baseline.TabICLBaselineModel` (both
    previously duplicated this exact block) -- looks up label ``1``'s
    actual column index via ``clf.classes_`` rather than assuming it's
    column 1, since a classifier fit on a fold where only one class was
    observed can return ``classes_`` with fewer than 2 entries in an order
    that doesn't match the label values.

    Parameters
    ----------
    clf : object
        A fitted classifier exposing ``classes_`` (typed ``object``, not a
        specific sklearn/interpret protocol, so callers referencing an
        optional dependency's classifier type don't force it on this
        module).
    proba : numpy.ndarray
        ``clf.predict_proba(x)``'s full ``(n, n_classes)`` output.

    Returns
    -------
    numpy.ndarray
        ``(n,)``, the probability of label ``1``.
    """
    classes = np.asarray(clf.classes_)  # type: ignore[attr-defined]
    pos_idx = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else 1
    result: np.ndarray = proba[:, pos_idx]
    return result


class BaselineModel:
    """A fitted GBM plus the all-missing columns it had to fill at fit time.

    ``HistGradientBoostingClassifier`` handles missing values natively,
    which is what we want (missingness is informative), but its binner
    cannot fit a column that is missing everywhere in the rows it bins
    (it bins on a 200,000-row subsample, so a column with only a handful
    of observations can vanish from it). Columns with fewer than
    :data:`GBM_MIN_OBSERVED` observed values carry no learnable signal, so
    they are filled with 0 at fit and, for consistency, at prediction.
    """

    def __init__(
        self,
        clf: HistGradientBoostingClassifier,
        fill_columns: np.ndarray,
        *,
        feature_set: str = "basic",
        params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.clf = clf
        self.fill_columns = fill_columns
        self.feature_set = feature_set
        self.n_features = int(len(fill_columns))
        self.params = dict(params or {})

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float32, copy=True)
        x[:, self.fill_columns] = np.nan_to_num(x[:, self.fill_columns], nan=0.0)
        return x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Positive-class probabilities, ``(n,)``."""
        proba: np.ndarray = self.clf.predict_proba(self._prepare(x))[:, 1]
        return proba


# Small, honest search for the bespoke GBM: learning rate x tree size x leaf
# size, each run to 400 rounds with the best round picked on a subject-grouped
# validation split, then refit on everything at that round count.
GBM_GRID: Tuple[Dict[str, float], ...] = (
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 100},
    {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20},
    {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 100},
)
GBM_MAX_ITER = 400
GBM_TUNE_MAX_ROWS = 200_000

# Row cap on the final refit (after tuning picks params/rounds). A
# gradient-boosted tree with at most 63 leaves and 400 rounds has long
# since converged well inside this many rows; at full-scale corpora
# (millions of landmark rows for a 292-shard run) fitting on everything
# kept both a full feature-matrix copy and HistGradientBoostingClassifier's
# own internal working set (binned representation, gradients, per-tree node
# arrays) large enough to matter, so both this and the tuning step above
# use the same seeded-subsample pattern rather than the whole kept set.
GBM_FIT_MAX_ROWS = 1_000_000


# HistGradientBoosting bins features on a random subsample of 200,000 rows;
# a column observed in fewer rows than this can be entirely missing inside
# that subsample (or inside a tuning fold), and the binner cannot fit an
# empty column. Such columns carry no learnable signal at this scale, so
# they are filled with 0 wherever they occur.
GBM_MIN_OBSERVED = 200


def sparse_columns(x: np.ndarray) -> np.ndarray:
    """Boolean mask of columns with fewer than :data:`GBM_MIN_OBSERVED` values."""
    observed = (~np.isnan(x)).sum(axis=0)
    return np.asarray(observed < GBM_MIN_OBSERVED)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _tune_gbm(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int
) -> Tuple[Dict[str, float], int]:
    """Best (params, n_rounds) on a subject-grouped 10% validation split."""
    rng = np.random.default_rng(seed)
    if len(y) > GBM_TUNE_MAX_ROWS:
        pick = rng.choice(len(y), GBM_TUNE_MAX_ROWS, replace=False)
        x, y, groups = x[pick], y[pick], groups[pick]
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(0.1 * len(unique_groups))))
    val_groups = set(unique_groups[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    if is_val.all() or (~is_val).all() or len(np.unique(y[is_val])) < 2:
        return dict(GBM_GRID[0]), 200
    x_tr, y_tr = x[~is_val], y[~is_val]
    x_val, y_val = x[is_val], y[is_val]
    # A column can be (nearly) all-missing inside the training fold even
    # when it is not over the full fit set (a rare exposure seen only in the
    # held-out subjects, or dropped by the row subsample); fill those with 0
    # in both folds, exactly as BaselineModel does for the full fit.
    fold_fill = sparse_columns(x_tr)
    if fold_fill.any():
        x_tr = np.array(x_tr, copy=True)
        x_val = np.array(x_val, copy=True)
        x_tr[:, fold_fill] = 0.0
        x_val[:, fold_fill] = np.nan_to_num(x_val[:, fold_fill], nan=0.0)
    best: Tuple[float, Dict[str, float], int] = (np.inf, dict(GBM_GRID[0]), 200)
    for params in GBM_GRID:
        clf = HistGradientBoostingClassifier(
            random_state=seed, max_iter=GBM_MAX_ITER, early_stopping=False, **params
        )
        clf.fit(x_tr, y_tr)
        losses = [
            _log_loss(y_val, proba[:, 1]) for proba in clf.staged_predict_proba(x_val)
        ]
        k = int(np.argmin(losses))
        if losses[k] < best[0]:
            best = (losses[k], dict(params), k + 1)
    return best[1], best[2]


def _fit_baseline_grid(
    x_all: np.ndarray,
    rows: Sequence[IndexRow],
    times: EventTimes,
    *,
    horizons: Sequence[float],
    feature_set: str,
    seed: int,
    tune: bool,
    event_name: str,
) -> Dict[float, BaselineModel]:
    """Fit one GBM per horizon for a single event, given a pre-built feature matrix.

    ``x_all`` and ``rows`` are already aligned (row ``i`` of ``x_all`` is
    ``rows[i]``'s feature vector); everything upstream of this -- how the
    features were built, whether from one in-memory frame
    (:func:`fit_baselines`) or shard by shard
    (:func:`fit_baselines_streaming`) -- is irrelevant here. Shared by both
    so the two paths fit identically once the feature matrix exists.

    Rows are capped at :data:`GBM_FIT_MAX_ROWS` (same reasoning as
    :data:`GBM_TUNE_MAX_ROWS` for tuning, see there) *before* indexing into
    ``x_all``, not after: at full corpus scale (millions of rows) building
    the uncapped feature-matrix slice first and subsampling afterward would
    still pay for the full-size copy every horizon, on top of ``x_all``
    itself.
    """
    groups_all = np.array([r.subject_id for r in rows])
    rng = np.random.default_rng(seed)
    out: Dict[float, BaselineModel] = {}
    for h in horizons:
        y = np.array(
            [outcome_at_horizon(r, times, h) for r in rows],
            dtype=object,
        )
        keep = np.flatnonzero([v is not None for v in y])
        if len(keep) < 50 or len({int(y[i]) for i in keep}) < 2:
            continue
        if len(keep) > GBM_FIT_MAX_ROWS:
            keep = rng.choice(keep, GBM_FIT_MAX_ROWS, replace=False)
        x_fit = x_all[keep]
        y_fit = y[keep].astype(int)
        fill_columns = sparse_columns(x_fit)
        x_prep = np.array(x_fit, dtype=np.float32, copy=True)
        x_prep[:, fill_columns] = np.nan_to_num(x_prep[:, fill_columns], nan=0.0)
        if tune:
            params, n_rounds = _tune_gbm(x_prep, y_fit, groups_all[keep], seed=seed)
        else:
            params, n_rounds = {}, 200
        clf = HistGradientBoostingClassifier(
            random_state=seed, max_iter=n_rounds, early_stopping=False, **params
        )
        clf.fit(x_prep, y_fit)
        out[h] = BaselineModel(
            clf,
            fill_columns,
            feature_set=feature_set,
            params={**params, "n_rounds": float(n_rounds)},
        )
        logger.info(
            "[alerts] GBM %s@%gh: %s features, rounds=%d, params=%s",
            event_name,
            h,
            feature_set,
            n_rounds,
            params,
        )
    return out


def fit_baselines(
    train_events_binned: pl.DataFrame,
    train_rows: Dict[str, List[IndexRow]],
    train_times: Dict[str, EventTimes],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    source: str = "mimic_iv",
    seed: int = 0,
    feature_set: str = "strong",
    tune: bool = True,
) -> Dict[Tuple[str, float], BaselineModel]:
    """One gradient-boosted classifier per (event, horizon) on baseline features.

    ``train_events_binned`` is held whole in memory, so this scales to
    however many baseline shards fit in RAM; past that,
    :func:`fit_baselines_streaming` fits the same models one shard at a
    time. With ``tune`` the hyper-parameters and round count come from
    :func:`_tune_gbm` per (event, horizon); otherwise a fixed 200-round
    default is fitted (fast path for tests and smoke runs).
    """
    models: Dict[Tuple[str, float], BaselineModel] = {}
    features = features_for_events(
        train_events_binned, train_rows, source=source, feature_set=feature_set
    )
    for name, rows in train_rows.items():
        if not rows:
            continue
        per_horizon = _fit_baseline_grid(
            features[name],
            rows,
            train_times[name],
            horizons=horizons,
            feature_set=feature_set,
            seed=seed,
            tune=tune,
            event_name=name,
        )
        for h, model in per_horizon.items():
            models[(name, h)] = model
    return models


def fit_baselines_streaming(
    paths: Sequence[Path],
    prepare: Preparer,
    binner: Optional[QuantileBinner],
    *,
    alerts: Sequence[AlertEvent],
    horizons: Sequence[float] = HORIZONS_HOURS,
    source: str = "mimic_iv",
    landmark_hours: float = 4.0,
    seed: int = 0,
    feature_set: str = "strong",
    tune: bool = True,
    task_set: str = "v1",
    index_mode: str = "landmark",
) -> Dict[Tuple[str, float], BaselineModel]:
    """Fit the same models as :func:`fit_baselines`, but shard by shard.

    The full-scale baseline shard set (hundreds of shards, hundreds of
    millions of events) does not fit in memory as one frame -- the same
    problem :mod:`odyssey.training.shard_stream` solves for the training
    corpus. It applies here too: baseline features
    (:mod:`odyssey.inference.baseline_features`) and landmark index rows
    are entirely per-subject, and subjects never span shards, so both can
    be built one shard at a time. Only the resulting feature matrix (one
    row per landmark, not per raw event) and index rows are kept across
    shards; the raw per-shard frame is dropped once its features are
    extracted. Every alert shares the same landmarks (the buckets are not
    alert-specific), so features are built once per shard and reused
    across events, exactly like :func:`features_for_events` does for the
    in-memory path. GBM fitting itself is unchanged, delegated to
    :func:`_fit_baseline_grid`.
    """
    event_times: Dict[str, EventTimes] = {}
    all_rows: List[IndexRow] = []
    feature_chunks: List[np.ndarray] = []
    for path in paths:
        raw = prepare(load_meds_shard(path))
        binned = add_value_tokens(raw, binner, source=source)
        merge_event_times(
            event_times, all_event_times(raw, alerts, source, task_set=task_set)
        )
        shard_rows = _index_rows_from_events(
            binned, alerts, landmark_hours=landmark_hours, index_mode=index_mode
        )[alerts[0].name]
        if not shard_rows:
            continue
        if feature_set == "strong":
            feats = strong_baseline_features(binned, shard_rows, source=source)
        elif feature_set == "strong_text":
            feats = strong_text_baseline_features(binned, shard_rows, source=source)
        else:
            feats = baseline_features(binned, shard_rows, source=source)
        all_rows.extend(shard_rows)
        feature_chunks.append(feats)
    models: Dict[Tuple[str, float], BaselineModel] = {}
    if not all_rows:
        return models
    x_all = np.concatenate(feature_chunks, axis=0)
    del feature_chunks  # concatenated: the per-shard copies are dead weight now
    for alert in alerts:
        if alert.name not in event_times:
            continue
        per_horizon = _fit_baseline_grid(
            x_all,
            all_rows,
            event_times[alert.name],
            horizons=horizons,
            feature_set=feature_set,
            seed=seed,
            tune=tune,
            event_name=alert.name,
        )
        for h, model in per_horizon.items():
            models[(alert.name, h)] = model
    return models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class AlertMetrics:
    """Time-dependent metrics for one (event, horizon, scorer)."""

    event: str
    horizon_hours: float
    scorer: str
    n_at_risk: int
    n_positive: int
    n_censored: int
    auroc: Optional[float]
    brier: Optional[float] = None
    """Only for scorers that output horizon probabilities (the baseline)."""
    calibration: Optional[List[Dict[str, float]]] = None
    """Decile bins of predicted probability: mean predicted vs observed."""
    baseline_feature_set: Optional[str] = None
    """For ``baseline_gbm`` rows: which feature set the GBM was given."""
    baseline_n_features: Optional[int] = None
    baseline_params: Optional[Dict[str, float]] = None


def _calibration(
    pred: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    order = np.argsort(pred)
    bins = np.array_split(order, n_bins)
    return [
        {
            "predicted": float(pred[b].mean()),
            "observed": float(y[b].mean()),
            "n": int(len(b)),
        }
        for b in bins
        if len(b) > 0
    ]


class _ScoredBaseline(Protocol):
    """Structural interface a fitted baseline model must satisfy to be scored.

    :class:`BaselineModel` (the GBM) and any other baseline family (e.g.
    :class:`~odyssey.inference.tabicl_baseline.TabICLBaselineModel`)
    satisfy this without inheriting from anything -- ``score_alerts``
    only ever reads these four attributes.
    """

    feature_set: str
    n_features: int
    params: dict[str, float]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Positive-class probabilities, ``(n,)``."""
        ...  # noqa: PIE790


def _score_named_baseline(
    scorer_name: str,
    event: str,
    horizon: float,
    model: _ScoredBaseline,
    x: np.ndarray,
    *,
    y: np.ndarray,
    n_censored: int,
) -> AlertMetrics:
    """One :class:`AlertMetrics` row for one fitted baseline at one horizon.

    Shared by the built-in GBM baseline and any additional named baseline
    family passed via ``extra_baselines`` -- every baseline is scored the
    same way (AUROC, Brier, calibration; it always outputs a probability),
    so this is the one place that logic lives.
    """
    p = model.predict_proba(x)
    return AlertMetrics(
        event=event,
        horizon_hours=horizon,
        scorer=scorer_name,
        n_at_risk=int(len(y)),
        n_positive=int(y.sum()),
        n_censored=n_censored,
        auroc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        calibration=_calibration(p, y),
        baseline_feature_set=model.feature_set,
        baseline_n_features=model.n_features,
        baseline_params=model.params or None,
    )


def score_alerts(
    rows: Dict[str, List[IndexRow]],
    times: Dict[str, EventTimes],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    baselines: Optional[Dict[Tuple[str, float], BaselineModel]] = None,
    baseline_features_by_event: Optional[Dict[str, np.ndarray]] = None,
    extra_baselines: dict[
        str, tuple[dict[tuple[str, float], _ScoredBaseline], dict[str, np.ndarray]]
    ]
    | None = None,
) -> List[AlertMetrics]:
    """Score every (event, horizon, scorer) present in ``rows``.

    ``extra_baselines`` scores additional baseline families beyond the
    built-in GBM, keyed by scorer name (e.g. ``"baseline_tabicl"``) ->
    ``(models, features_by_event)``, each shaped exactly like
    ``baselines``/``baseline_features_by_event``. Lets a new baseline
    family (see :mod:`odyssey.inference.tabicl_baseline`) be compared
    without this function needing to know it exists beyond this one
    generic hook.
    """
    results: List[AlertMetrics] = []
    for name, event_rows in rows.items():
        if not event_rows:
            continue
        for h in horizons:
            outcomes = [outcome_at_horizon(r, times[name], h) for r in event_rows]
            n_censored = sum(1 for o in outcomes if o is None)
            keep = [i for i, o in enumerate(outcomes) if o is not None]
            y = np.array([outcomes[i] for i in keep], dtype=int)
            if len(keep) == 0 or y.min() == y.max():
                continue
            scorer_names = sorted({k for r in event_rows for k in r.scores})
            for scorer in scorer_names:
                if scorer.startswith("hazard@") and scorer != f"hazard@{h:g}h":
                    continue  # a horizon-specific probability scores its own horizon
                pred = np.array(
                    [event_rows[i].scores.get(scorer, np.nan) for i in keep],
                    dtype=float,
                )
                ok = ~np.isnan(pred)
                if ok.sum() == 0 or y[ok].min() == y[ok].max():
                    continue
                is_probability = scorer.startswith("hazard@")
                results.append(
                    AlertMetrics(
                        event=name,
                        horizon_hours=h,
                        scorer="hazard" if is_probability else scorer,
                        n_at_risk=int(ok.sum()),
                        n_positive=int(y[ok].sum()),
                        n_censored=n_censored,
                        auroc=float(roc_auc_score(y[ok], pred[ok])),
                        brier=(
                            float(brier_score_loss(y[ok], pred[ok]))
                            if is_probability
                            else None
                        ),
                        calibration=(
                            _calibration(pred[ok], y[ok]) if is_probability else None
                        ),
                    )
                )
            if (
                baselines is not None
                and (name, h) in baselines
                and baseline_features_by_event
            ):
                x = baseline_features_by_event[name][keep]
                results.append(
                    _score_named_baseline(
                        "baseline_gbm",
                        name,
                        h,
                        baselines[(name, h)],
                        x,
                        y=y,
                        n_censored=n_censored,
                    )
                )
            for scorer_name, (models, features_by_event) in (
                extra_baselines or {}
            ).items():
                if (name, h) not in models or name not in features_by_event:
                    continue
                x = features_by_event[name][keep]
                results.append(
                    _score_named_baseline(
                        scorer_name,
                        name,
                        h,
                        models[(name, h)],
                        x,
                        y=y,
                        n_censored=n_censored,
                    )
                )
    return results


def index_row_table(
    rows: Dict[str, List[IndexRow]],
    times: Dict[str, EventTimes],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    baselines: Optional[Dict[Tuple[str, float], BaselineModel]] = None,
    baseline_features_by_event: Optional[Dict[str, np.ndarray]] = None,
    context_columns: Optional[Dict[str, np.ndarray]] = None,
    context_names: Optional[Sequence[str]] = None,
    extra_baselines: dict[
        str, tuple[dict[tuple[str, float], _ScoredBaseline], dict[str, np.ndarray]]
    ]
    | None = None,
) -> pl.DataFrame:
    """One row per (event, index time) with every score and outcome, for error analysis.

    Columns: ``event, subject_id, visit_id, time_hours``, one column per
    model scorer (``concept``, ``next_mass``, ``hazard@{h}h``), per horizon
    ``y@{h}h`` (1/0, null if censored or not at risk) and, when baselines
    are given, ``gbm@{h}h``; plus any ``context_names`` columns taken from
    the baseline feature matrix (e.g. hours into the visit, whether a
    creatinine was measured in the last 24h). Patient-level: keep it under
    the run directory, never in git.

    ``extra_baselines`` adds one column per horizon per named family (e.g.
    ``tabicl@{h}h`` for a
    :func:`~odyssey.inference.tabicl_baseline.fit_tabicl_baselines` result
    passed as ``{"tabicl": (models, features_by_event)}``), shaped exactly
    like ``baselines``/``baseline_features_by_event``. Once present, a
    third baseline family participates in the same stratified error
    analysis as ``gbm@{h}h`` already does (group by any ``ctx.*`` column,
    e.g. ``ctx.hours_into_visit`` as a proxy for sequence length so far,
    exactly as entry 22 of the research journal stratifies the GBM) --
    without this function needing any more baseline-specific logic than
    a name and a column prefix.
    """
    frames: List[pl.DataFrame] = []
    for name, event_rows in rows.items():
        if not event_rows:
            continue
        data: Dict[str, List[Optional[float]]] = {
            "subject_id": [float(r.subject_id) for r in event_rows],
            "visit_id": [float(r.visit_id) for r in event_rows],
            "time_hours": [r.time_hours for r in event_rows],
        }
        scorers = sorted({k for r in event_rows for k in r.scores})
        for s in scorers:
            data[s] = [r.scores.get(s) for r in event_rows]
        for h in horizons:
            outcomes = [outcome_at_horizon(r, times[name], h) for r in event_rows]
            data[f"y@{h:g}h"] = [None if o is None else float(o) for o in outcomes]
            if (
                baselines is not None
                and (name, h) in baselines
                and baseline_features_by_event
                and name in baseline_features_by_event
            ):
                p = baselines[(name, h)].predict_proba(baseline_features_by_event[name])
                data[f"gbm@{h:g}h"] = [float(v) for v in p]
            for prefix, (models, features_by_event) in (extra_baselines or {}).items():
                if (name, h) not in models or name not in features_by_event:
                    continue
                p = models[(name, h)].predict_proba(features_by_event[name])
                data[f"{prefix}@{h:g}h"] = [float(v) for v in p]
        if context_columns is not None and name in context_columns and context_names:
            ctx = context_columns[name]
            for j, cname in enumerate(context_names):
                data[f"ctx.{cname}"] = [float(v) for v in ctx[:, j]]
        frame = pl.DataFrame(data).with_columns(pl.lit(name).alias("event"))
        frames.append(frame)
    if not frames:
        return pl.DataFrame({"event": []})
    return pl.concat(frames, how="diagonal")


def _stamp_landmark_protocol_version(table: pl.DataFrame) -> pl.DataFrame:
    """Attach the current :data:`LANDMARK_PROTOCOL_VERSION` as a constant column.

    Only meaningful for a table built from :func:`collect_model_scores`'
    rows (the only path :data:`LANDMARK_PROTOCOL_VERSION` describes) --
    see :func:`evaluate_alerts`'s dump-rows call site, the only caller.
    """
    return table.with_columns(
        pl.lit(LANDMARK_PROTOCOL_VERSION).alias("landmark_protocol_version")
    )


def load_index_row_table(path: Union[str, Path]) -> pl.DataFrame:
    """Read a dumped index-row table, always logging its protocol version.

    ``landmark_protocol_version`` absent entirely means the dump predates
    the column -- protocol v1 (see :data:`LANDMARK_PROTOCOL_VERSION`).
    Always logs the version found; emits a loud ``WARNING`` (not an error)
    when it differs from the code's current
    :data:`LANDMARK_PROTOCOL_VERSION` -- a v1 dump is still a perfectly
    valid, internally consistent comparison set for anything scored
    against it (e.g. a baseline fitted and evaluated entirely against v1
    rows), it just isn't comparable row-for-row against a fresh run under
    a different protocol version without knowing that. Callers that need
    to block on a mismatch (rather than just warn) should check the
    returned column themselves.
    """
    table = pl.read_parquet(path)
    if "landmark_protocol_version" in table.columns:
        versions = table["landmark_protocol_version"].unique().to_list()
        version = versions[0] if len(versions) == 1 else None
        if version is None:
            logger.warning(
                "[alerts] %s has mixed landmark_protocol_version values %s -- "
                "was this concatenated from runs on different protocol versions?",
                path,
                versions,
            )
    else:
        version = 1
    logger.info("[alerts] %s: landmark_protocol_version=%s", path, version)
    if version != LANDMARK_PROTOCOL_VERSION:
        logger.warning(
            "[alerts] %s was written with landmark_protocol_version=%s, but this "
            "code is on version %s -- scores from this dump remain a valid, "
            "internally consistent comparison set among themselves, but are NOT "
            "directly comparable row-for-row against a fresh run under the "
            "current protocol.",
            path,
            version,
            LANDMARK_PROTOCOL_VERSION,
        )
    return table


def _index_rows_from_dump(
    dump_path: Union[str, Path], alerts: Sequence[AlertEvent]
) -> List[IndexRow]:
    """Return the distinct (subject, visit, time) rows of a dump for ``alerts``."""
    dump = load_index_row_table(dump_path)
    names = {a.name for a in alerts}
    sub = (
        dump.filter(pl.col("event").is_in(sorted(names)))
        if "event" in dump.columns
        else dump
    )
    keys = sub.select("subject_id", "visit_id", "time_hours").unique()
    return [
        IndexRow(int(s), int(v), float(t))
        for s, v, t in zip(keys["subject_id"], keys["visit_id"], keys["time_hours"])
    ]


def verify_rows_match_dump(
    rows: Dict[str, List[IndexRow]],
    times: Dict[str, EventTimes],
    dump_path: Union[str, Path],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
) -> None:
    """Assert a freshly-reconstructed row set exactly matches a saved dump.

    For baseline scripts that keep their own row/feature source (e.g.
    ``collect_model_scores``'s live forward pass) unchanged across protocol
    versions, but need proof -- not an assumption -- that the cohort they
    just scored is the same one a saved ``index_row_table`` dump (an
    ``alerts_rows.parquet``-style file) represents. Verifies two things,
    per event, loud failure on either:

    1. Row identity: the multiset of ``(subject_id, visit_id, time_hours)``
       triples matches exactly. A multiset, not a set -- real dumps can
       have duplicate keys (a lane/chunk streaming walk can revisit the
       same landmark position more than once), and a plain set comparison
       would silently collapse those and hide a real count mismatch.
    2. Label agreement: for every horizon, the multiset of
       ``(subject_id, visit_id, time_hours, y@h)`` matches exactly --
       :func:`outcome_at_horizon` computed fresh here must agree with the
       dump's own ``y@{h:g}h`` column for every row, not just share the
       same row keys.

    Uses :func:`load_index_row_table` to read the dump, so a
    ``landmark_protocol_version`` mismatch is still logged as a warning
    even when every row and label otherwise agrees (protocol version is
    orthogonal to row/label identity -- a dump can be on a different
    protocol and still, by coincidence or by design, describe the same
    cohort).
    """
    dump = load_index_row_table(dump_path)
    for event_name, event_rows in rows.items():
        dump_ev = dump.filter(pl.col("event") == event_name)
        own_keys = sorted((r.subject_id, r.visit_id, r.time_hours) for r in event_rows)
        dump_keys = sorted(
            zip(
                (int(s) for s in dump_ev["subject_id"].to_list()),
                (int(v) for v in dump_ev["visit_id"].to_list()),
                dump_ev["time_hours"].to_list(),
            )
        )
        if own_keys != dump_keys:
            own_set, dump_set = set(own_keys), set(dump_keys)
            raise AssertionError(
                f"{event_name}: reconstructed rows do not match dump {dump_path} -- "
                f"{len(own_keys)} reconstructed vs {len(dump_keys)} in dump; "
                f"only in reconstruction (up to 5): {sorted(own_set - dump_set)[:5]}; "
                f"only in dump (up to 5): {sorted(dump_set - own_set)[:5]}"
            )

        for h in horizons:
            y_col = f"y@{h:g}h"
            # -1.0 is a sentinel for None/censored -- outcome_at_horizon only
            # ever returns None, 0, or 1, so this is unambiguous and, unlike
            # a bare None, sorts safely against the float values it's mixed
            # with (Python raises comparing None to float when tuple keys tie
            # on subject/visit/time and only the outcome differs).
            own_y = sorted(
                (
                    r.subject_id,
                    r.visit_id,
                    r.time_hours,
                    -1.0
                    if (o := outcome_at_horizon(r, times[event_name], h)) is None
                    else float(o),
                )
                for r in event_rows
            )
            dump_y = sorted(
                (s, v, t, -1.0 if y is None else float(y))
                for s, v, t, y in zip(
                    (int(s) for s in dump_ev["subject_id"].to_list()),
                    (int(v) for v in dump_ev["visit_id"].to_list()),
                    dump_ev["time_hours"].to_list(),
                    dump_ev[y_col].to_list(),
                )
            )
            if own_y != dump_y:
                own_y_set, dump_y_set = set(own_y), set(dump_y)
                raise AssertionError(
                    f"{event_name}@{h:g}h: y@h disagrees with dump {dump_path} for "
                    f"{len(own_y_set.symmetric_difference(dump_y_set))} row(s) (up to 5 "
                    f"reconstructed): {sorted(own_y_set - dump_y_set)[:5]}; "
                    f"(up to 5 dump): {sorted(dump_y_set - own_y_set)[:5]}"
                )
    logger.info(
        "[alerts] verified against dump %s: %s",
        dump_path,
        ", ".join(f"{name}={len(r)} rows" for name, r in rows.items()),
    )


# Context columns dumped with the per-row table (names from
# odyssey.inference.baseline_features.feature_names when the strong set is used).
ROW_DUMP_CONTEXT: Tuple[str, ...] = (
    "hours_into_visit",
    "hours_since_origin",
    "age_years",
    "in_icu",
    "n_events_visit",
    "creatinine.hours_since_last",
    "creatinine.last",
    "creatinine.ratio_visit_min",
    "lactate.hours_since_last",
    "map_arterial.hours_since_last",
    "map_noninvasive.hours_since_last",
    "heart_rate.hours_since_last",
    "family.lab.n_24h",
    "family.medication.n_24h",
    "drug.vasopressor.ever_visit",
)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _visit_starts(events: pl.DataFrame) -> Dict[Tuple[int, int], float]:
    origins = origin_hours(events)
    starts = (
        events.filter(pl.col("time").is_not_null() & pl.col("hadm_id").is_not_null())
        .group_by("subject_id", "hadm_id")
        .agg(pl.col("time").min().alias("_start"))
    )
    starts = hours_since_origin(starts, "_start", origins)
    return {
        (int(s), int(v)): float(t)
        for s, v, t in zip(starts["subject_id"], starts["hadm_id"], starts["_start"])
    }


def _index_rows_from_events(
    events_binned: pl.DataFrame,
    alerts: Sequence[AlertEvent],
    *,
    landmark_hours: float,
    index_mode: str = "landmark",
    origins: Optional[pl.DataFrame] = None,
    visit_starts: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[str, List[IndexRow]]:
    """Index rows straight from events (no model), for baseline fitting.

    Landmark buckets by default; ``index_mode="visit_end"`` gives one row
    per (subject, visit) at the visit's last event. ``origins`` (subject ->
    first timed event, :func:`origin_hours`) and ``visit_starts``
    ((subject, visit) -> hours of the visit's first event) default to the
    frame's own; pass the FULL record's when ``events_binned`` is a filtered
    view (the verifier's kept-window truth) so hours and buckets stay in
    the true frame.
    """
    _check_index_mode(index_mode)
    if origins is None:
        origins = origin_hours(events_binned)
    timed = (
        events_binned.filter(
            pl.col("time").is_not_null() & pl.col("hadm_id").is_not_null()
        )
        .join(origins, on="subject_id", how="left")
        .with_columns(
            ((pl.col("time") - pl.col("_origin")).dt.total_seconds() / 3600.0).alias(
                "_hours"
            )
        )
        .sort("subject_id", "_hours")
    )
    if index_mode == "visit_end":
        firsts = timed.group_by("subject_id", "hadm_id").agg(
            pl.col("_hours").max().alias("_t")
        )
        rows = [
            IndexRow(int(s), int(v), float(t))
            for s, v, t in zip(firsts["subject_id"], firsts["hadm_id"], firsts["_t"])
        ]
        return {a.name: list(rows) for a in alerts}
    if visit_starts is None:
        starts = timed.group_by("subject_id", "hadm_id").agg(
            pl.col("_hours").min().alias("_start")
        )
    else:
        starts = pl.DataFrame(
            {
                "subject_id": [k[0] for k in visit_starts],
                "hadm_id": [k[1] for k in visit_starts],
                "_start": [float(v) for v in visit_starts.values()],
            }
        ).cast(
            {
                "subject_id": timed.schema["subject_id"],
                "hadm_id": timed.schema["hadm_id"],
            }
        )
    timed = timed.join(starts, on=["subject_id", "hadm_id"], how="left").with_columns(
        ((pl.col("_hours") - pl.col("_start")) // landmark_hours).alias("_bucket")
    )
    firsts = timed.group_by("subject_id", "hadm_id", "_bucket").agg(
        pl.col("_hours").min().alias("_t")
    )
    rows = [
        IndexRow(int(s), int(v), float(t))
        for s, v, t in zip(firsts["subject_id"], firsts["hadm_id"], firsts["_t"])
    ]
    return {a.name: list(rows) for a in alerts}


def _landmark_key_set(rows: Sequence[IndexRow]) -> set[Tuple[int, int, float]]:
    """(subject, visit, time rounded to microseconds) for set comparison.

    Rounding guards against float noise between two independently
    computed time bases that should agree exactly (both "hours since
    this subject's first timed non-birth event" -- see
    :func:`collect_model_scores`'s docstring) but arrive at that value
    through different code paths (tokenization vs. a polars expression).
    """
    return {(row.subject_id, row.visit_id, round(row.time_hours, 6)) for row in rows}


def _visible_after_truncation(
    events_binned: pl.DataFrame, truncated: Set[int], max_context: int
) -> pl.DataFrame:
    """Return the rows a truncated subject's packed window actually kept.

    Mirrors :func:`odyssey.data.sequences.build_patient_sequence`'s token
    order (timed non-birth rows sorted by time, ties in frame order; static
    rows lead and are the first to be truncated away) and
    :class:`~odyssey.data.packed_context.PackedContextSampler`'s tail
    truncation: the last ``max_context`` timed rows per truncated subject.
    Other subjects are returned untouched.
    """
    is_trunc = pl.col("subject_id").is_in(sorted(truncated))
    untouched = events_binned.filter(~is_trunc)
    timed = (
        events_binned.filter(
            is_trunc & pl.col("time").is_not_null() & (pl.col("code") != BIRTH_CODE)
        )
        .with_row_index("_row")
        .sort(["subject_id", "time", "_row"], maintain_order=True)
        .with_columns(pl.col("_row").cum_count().over("subject_id").alias("_pos"))
        .with_columns(pl.col("_pos").max().over("subject_id").alias("_n"))
        .filter(pl.col("_pos") > pl.col("_n") - max_context)
        .drop("_row", "_pos", "_n")
    )
    return pl.concat([untouched, timed.select(untouched.columns)])


def verify_packed_landmark_rows(
    model_rows: Dict[str, List[IndexRow]],
    events_binned: pl.DataFrame,
    alerts: Sequence[AlertEvent],
    *,
    landmark_hours: float,
    truncation_boundaries: Dict[int, float],
    index_mode: str = "landmark",
    max_context: Optional[int] = None,
) -> List[str]:
    """Compare collect_model_scores' index-row set against the model-free truth.

    ``max_context`` is the packed sampler's window (needed whenever
    ``truncation_boundaries`` is non-empty): a truncated subject's visible
    record is the last ``max_context`` timed tokens in the tokenizer's
    order, which is a token-count tail, not a time cut -- a same-instant
    bundle at the boundary can be split between kept and dropped rows.

    ``index_mode="visit_end"`` compares visit-end rows instead: exact set
    equality for non-truncated subjects; for truncated subjects, no row
    the truth lacks and every missing visit end strictly before the
    subject's truncation boundary (bucket logic does not apply).

    Two independent implementations of "which (subject, visit, time)
    triples are landmarks" (:func:`collect_model_scores`, either backbone,
    and :func:`_index_rows_from_events`) only get to coexist once they are
    shown to agree, not assumed to -- this is that proof, run for real
    every time a run is scored (:func:`evaluate_alerts`), not only in
    tests. Originally written and gated to backbone="transformer" only;
    generalized 2026-08-23 after the v2->v3 landmark-selection bug turned
    out to affect backbone="hybrid" too and nothing had ever verified it
    (``truncation_boundaries`` stays empty for that backbone, so every
    subject goes through the exact-match branch below -- the
    truncated-subject branches only ever trigger for
    ``PackedContextSampler``, which never gets used with backbone="hybrid").

    Truncated subjects (``IndexRow.is_tail``) are compared differently
    from everyone else, or a real evaluation at ``max_context=4096``
    would warn on every single run for entirely expected behavior: a
    truncated subject's pre-truncation history is legitimately absent
    from the packed path. Two distinct effects, both expected, both
    excluded from being flagged:

    - Whole missing buckets: any landmark bucket entirely before the
      truncation boundary has no packed-path row at all -- the ordinary
      "the packed set is a subset of the ground truth" case.
    - The one bucket straddling the boundary: if that bucket's true
      first event was itself truncated away, the packed path's own
      window-start (still real, still the first visible event of that
      bucket) becomes its landmark, at a *later* time than the ground
      truth's ("first event of the bucket over the *whole* sequence")
      -- same bucket, different exact time, not a disagreement. Proven
      necessary, not just observed: fixed here after
      ``test_verify_packed_landmark_rows_tail_aware_all_three_arms``'s
      "arm 0" (a correctly shrunk tail should report zero problems)
      failed on exact-time comparison alone. Comparing at the bucket
      level for truncated subjects absorbs this without weakening the
      non-truncated check at all -- that one stays exact-time, where
      there is no boundary effect to absorb.

    Three checks, per subject:

    - Not truncated: exact (subject, visit, time) set equality (as for
      every subject before this distinction existed).
    - Truncated: the packed set's (subject, visit, bucket) triples must
      not contain anything absent from the ground truth's (an invented
      bucket is still always a bug -- the packed path only ever sees a
      subset of the true timeline, so it can never legitimately surface
      a bucket the true timeline doesn't have).
    - Truncated: every ground-truth bucket the packed set is missing
      entirely must have its own (ground-truth) landmark time strictly
      before that subject's truncation boundary -- a whole bucket
      missing at or after the boundary means the packed path dropped a
      landmark it should have kept, a real bug, not truncation working
      as intended.

    ``truncation_boundaries`` (subject_id -> kept-window start, in the
    shared original-subject-origin time frame) must come from
    :func:`collect_model_scores`'s own sampler state
    (``truncation_boundaries_out``), captured at collection time -- NOT
    re-derived from ``model_rows`` here. An earlier version of this
    function derived the boundary from ``model_rows`` itself (each
    truncated subject's earliest ``is_tail`` row time); that is circular
    the moment ``model_rows`` is the very thing being checked for
    correctness -- confirmed the hard way, by an adversarial test that
    deleted a boundary row to simulate a dropped landmark and found the
    re-derived boundary silently shifted to match, hiding the deletion
    instead of catching it
    (``test_verify_packed_landmark_rows_tail_aware_all_three_arms``'s
    "arm 2").

    Returns human-readable mismatch descriptions (empty = agrees exactly
    once truncation is accounted for); logs a warning for any it finds
    rather than raising, matching
    :func:`~odyssey.utils.env_fingerprint.check_canary`'s house style for
    a real-time self-check that must not abort a running evaluation.
    """
    # Exact truth for what the packed path can see: every subject's full
    # record, except truncated subjects, whose events before their kept
    # window start are removed (origins kept from the full record so hours
    # stay in the true frame). The packed path's index rows must equal this
    # set exactly -- for both backbones, both index modes -- with no
    # bucket-level heuristics: the earlier three-arm bucket comparison
    # flagged a straddling bucket whose kept tokens all belonged to another
    # visit (entry 44's CPU integration pass, 17 false positives per shard).
    origins = origin_hours(events_binned)
    visible = events_binned
    if truncation_boundaries:
        if max_context is None:
            raise ValueError(
                "verify_packed_landmark_rows needs max_context when subjects were "
                "truncated (truncation_boundaries is non-empty)"
            )
        visible = _visible_after_truncation(
            events_binned, set(truncation_boundaries), max_context
        )
    expected = _index_rows_from_events(
        visible,
        alerts,
        landmark_hours=landmark_hours,
        index_mode=index_mode,
        origins=origins,
        visit_starts=_visit_starts(events_binned),
    )
    problems: List[str] = []
    for alert in alerts:
        got = _landmark_key_set(model_rows.get(alert.name, []))
        want = _landmark_key_set(expected.get(alert.name, []))
        if got != want:
            missing = len(want - got)
            extra = len(got - want)
            problems.append(
                f"{alert.name}: model index rows disagree with the model-free truth "
                f"on the visible record (missing {missing}, extra {extra}; "
                f"index_mode={index_mode}, {len(truncation_boundaries)} truncated "
                "subjects accounted for)"
            )
    for p in problems:
        logger.warning(
            "[alerts] index-row set mismatch: %s -- the model path's selection "
            "and the model-free truth must agree exactly (see "
            "verify_packed_landmark_rows' docstring); treat these scores as "
            "suspect until this is understood",
            p,
        )
    return problems


def _load_prepared_raw(
    shard_dir: Union[str, Path], max_shards: Optional[int], config: object, source: str
) -> pl.DataFrame:
    """Load one MEDS split and apply the run's own normalize/history-recap prep.

    Shared by :func:`evaluate_alerts` for both the clean held-out split and
    (when given) a missingness-protocol degraded shard directory -- same
    prep either way, only the shard directory differs.
    """
    raw = load_meds_shards(shard_dir, max_shards=max_shards)
    raw = maybe_normalize(
        raw, enabled=getattr(config, "normalize_medications", False), source=source
    )
    return maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))


def _fit_and_score_gbm_baselines(
    baseline_shard_dir: Optional[Union[str, Path]],
    *,
    stream_baseline: bool,
    max_baseline_shards: Optional[int],
    config: object,
    source: str,
    binner: Optional[QuantileBinner],
    alerts: Sequence[AlertEvent],
    horizons: Sequence[float],
    landmark_hours: float,
    baseline_feature_set: str,
    tune_baselines: bool,
    binned: pl.DataFrame,
    rows: Dict[str, List[IndexRow]],
    task_set: str = "v1",
    index_mode: str = "landmark",
    prefit_baselines: Optional[Dict[Tuple[str, float], BaselineModel]] = None,
) -> Tuple[
    Optional[Dict[Tuple[str, float], BaselineModel]], Optional[Dict[str, np.ndarray]]
]:
    """Fit and score the GBM baselines for one alerts pass.

    Fits on ``baseline_shard_dir`` (always clean -- Principle 1, never
    retrained/refit on degraded data) and scores against ``binned``/``rows``
    (the held-out split's own frame -- degraded when :func:`evaluate_alerts`
    was given ``degraded_shard_dir``). Returns ``(None, None)`` if neither
    ``prefit_baselines`` nor ``baseline_shard_dir`` was given.

    ``prefit_baselines``, when given, is used as-is -- no fit happens at
    all. This is the missingness sweep's reuse path
    (scripts/missingness_sweep.py): fitting is expensive and, per
    Principle 1, must be the SAME frozen model scored against every
    degradation cell, not independently refit 8 times (which would also
    confound the degradation signal with fit-to-fit variance from the
    hyperparameter search). Takes priority over ``baseline_shard_dir`` when
    both are given.
    """
    if prefit_baselines is not None:
        features_by_event = features_for_events(
            binned, rows, source=source, feature_set=baseline_feature_set
        )
        return prefit_baselines, features_by_event
    if baseline_shard_dir is None:
        return None, None
    if stream_baseline:
        logger.info(
            "[alerts] fitting GBM baselines on %s (streaming)", baseline_shard_dir
        )
        baseline_paths = shard_paths(baseline_shard_dir, max_shards=max_baseline_shards)
        prepare = make_preparer(
            normalize_medications=getattr(config, "normalize_medications", False),
            history_recap=getattr(config, "history_recap", False),
            source=source,
        )
        baselines = fit_baselines_streaming(
            baseline_paths,
            prepare,
            binner,
            alerts=alerts,
            horizons=horizons,
            source=source,
            landmark_hours=landmark_hours,
            feature_set=baseline_feature_set,
            tune=tune_baselines,
            task_set=task_set,
            index_mode=index_mode,
        )
    else:
        logger.info("[alerts] fitting GBM baselines on %s", baseline_shard_dir)
        train_raw = _load_prepared_raw(
            baseline_shard_dir, max_baseline_shards, config, source
        )
        train_times = all_event_times(train_raw, alerts, source, task_set=task_set)
        train_binned = add_value_tokens(train_raw, binner, source=source)
        del train_raw
        train_rows = _index_rows_from_events(
            train_binned, alerts, landmark_hours=landmark_hours, index_mode=index_mode
        )
        baselines = fit_baselines(
            train_binned,
            train_rows,
            train_times,
            horizons=horizons,
            source=source,
            feature_set=baseline_feature_set,
            tune=tune_baselines,
        )
    features_by_event = features_for_events(
        binned, rows, source=source, feature_set=baseline_feature_set
    )
    return baselines, features_by_event


def _score_degraded_at_clean_rows(
    model: SequenceModel,
    binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_names: Sequence[str],
    alerts: Sequence[AlertEvent],
    *,
    verify_against_dump: Optional[Union[str, Path]],
    num_lanes: int,
    chunk_size: int,
    device: str,
    horizons: Sequence[float],
    backbone: str,
    max_context: int,
    truncation_boundaries_out: Dict[int, float],
) -> Dict[str, List[IndexRow]]:
    """Missingness protocol: score the CLEAN dump's rows on the degraded record.

    The degraded record's own landmark grid is never used (lab lag shifts
    it, dropped visit starts re-bucket it); the model is scored at the last
    visible token at/before each clean row's time instead
    (:func:`collect_model_scores_at_rows`).
    """
    if verify_against_dump is None:
        raise ValueError(
            "degraded_shard_dir needs verify_against_dump (the clean dump whose "
            "rows are scored on the degraded record)"
        )
    dump_rows = _index_rows_from_dump(verify_against_dump, alerts)
    rows, unscoreable = collect_model_scores_at_rows(
        model,
        binned,
        vocab,
        concept_names,
        alerts,
        index_rows=dump_rows,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
        horizons=horizons,
        backbone=backbone,
        max_context=max_context,
        truncation_boundaries_out=truncation_boundaries_out,
    )
    logger.info(
        "[alerts] scored %d clean rows on the degraded record (%d unscoreable: no "
        "visible token at/before the row time)",
        len(dump_rows),
        unscoreable,
    )
    return rows


def evaluate_alerts(  # noqa: PLR0912, PLR0915
    run_dir: Union[str, Path],
    held_out_shard_dir: Union[str, Path],
    *,
    baseline_shard_dir: Optional[Union[str, Path]] = None,
    prefit_baselines: Optional[Dict[Tuple[str, float], BaselineModel]] = None,
    fitted_baselines_out: Optional[Dict[Tuple[str, float], BaselineModel]] = None,
    degraded_shard_dir: Optional[Union[str, Path]] = None,
    verify_against_dump: Optional[Union[str, Path]] = None,
    max_shards: Optional[int] = None,
    max_baseline_shards: Optional[int] = None,
    alerts: Optional[Sequence[AlertEvent]] = None,
    horizons: Sequence[float] = HORIZONS_HOURS,
    landmark_hours: float = 4.0,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    baseline_feature_set: str = "strong",
    tune_baselines: bool = True,
    stream_baseline: bool = False,
    dump_rows_path: Optional[Union[str, Path]] = None,
    index_mode: str = "landmark",
) -> List[AlertMetrics]:
    """End to end: model scores + optional GBM baselines, scored on held-out.

    ``dump_rows_path`` writes the per-index-row table of
    :func:`index_row_table` as parquet (patient-level; keep it with the run).
    ``stream_baseline`` fits the GBM baselines shard by shard
    (:func:`fit_baselines_streaming`) instead of loading
    ``baseline_shard_dir`` whole into memory (:func:`fit_baselines`); use it
    once ``max_baseline_shards`` is large enough that the whole-frame path
    risks OOM (full-scale runs, hundreds of shards).

    ``degraded_shard_dir`` is the missingness stress protocol's hook
    (docs/missingness_protocol.md; shards produced by
    :mod:`odyssey.data.degrade`): when given, everything that determines
    *labels* (``times``) and the *visit envelope* (``visit_start``, itself
    always identical across cells since anchor rows are never touched by
    any degrade.py transform) still comes from ``held_out_shard_dir`` (the
    clean v3 dump), but ``binned`` -- what both the model
    (:func:`collect_model_scores`) and the GBM baselines'
    :func:`features_for_events` actually score against -- is loaded from
    ``degraded_shard_dir`` instead. ``baseline_shard_dir`` (baseline
    *fitting*) is untouched either way: Principle 1 is frozen models AND
    frozen-fit baselines, degraded inputs only at scoring time, never at
    fit time. No change to :func:`collect_model_scores` or the streaming/
    tokenization path itself is needed for this -- a degraded shard is
    just a shard, scored the same way; landmark selection naturally
    reproduces the clean cohort as long as the degrade.py transform
    preserved it (anchor rows, and for axis C the subject's time origin).
    That is exactly what ``verify_against_dump`` checks: if given, asserts
    (via :func:`verify_rows_match_dump`) that this run's row set and
    labels exactly match a saved clean dump -- the acceptance criterion
    for a degraded cell's alerts pass -- immediately after the existing
    :func:`verify_packed_landmark_rows` self-consistency check.

    ``prefit_baselines``/``fitted_baselines_out`` are the sweep's other
    hook (see :func:`_fit_and_score_gbm_baselines`): pass the dict a
    CLEAN call populated via ``fitted_baselines_out`` back in as
    ``prefit_baselines`` on every degraded call, so the GBM is fit once
    and scored -- never refit -- against every cell, matching Principle 1
    for the baseline family too.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    activate_sidecars(held_out_shard_dir)
    concept_names = [c.name for c in concepts_for_source(source, task_set=task_set)]
    _check_index_mode(index_mode)
    if alerts is None:
        # The run's own task set: landmark mode scores the within-visit
        # events; visit_end mode scores the discharge-anchored ones.
        alerts = [
            a
            for a in alert_events_for(task_set)
            if a.next_visit == (index_mode == "visit_end")
        ]
        if not alerts:
            raise ValueError(
                f"task_set {task_set!r} has no events for index_mode={index_mode!r}"
            )

    raw = _load_prepared_raw(held_out_shard_dir, max_shards, config, source)
    times = all_event_times(raw, alerts, source, task_set=task_set)
    visit_start = _visit_starts(raw)
    if degraded_shard_dir is not None:
        logger.info(
            "[alerts] scoring against degraded shards from %s (labels/visit "
            "envelope from the clean %s)",
            degraded_shard_dir,
            held_out_shard_dir,
        )
        degraded_raw = _load_prepared_raw(
            degraded_shard_dir, max_shards, config, source
        )
        # The clean rows are scored on this record in the clean time frame:
        # every subject's origin (first timed event) must be identical
        # (degrade.py guarantees it; checked, not trusted).
        from odyssey.inference.baseline_prep import (  # noqa: PLC0415
            _verify_matching_origins,
        )

        _verify_matching_origins(raw, degraded_raw, context=str(degraded_shard_dir))
        binned = add_value_tokens(degraded_raw, binner, source=source)
        del degraded_raw
    else:
        binned = add_value_tokens(raw, binner, source=source)
    del raw

    backbone = getattr(config, "backbone", "hybrid")
    if index_mode == "visit_end":
        logger.info("[alerts] collecting model scores at visit ends (discharge)")
    else:
        logger.info(
            "[alerts] collecting model scores at %.0fh landmarks", landmark_hours
        )
    truncation_boundaries: Dict[int, float] = {}
    if degraded_shard_dir is not None:
        rows = _score_degraded_at_clean_rows(
            model,
            binned,
            vocab,
            concept_names,
            alerts,
            verify_against_dump=verify_against_dump,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
            horizons=horizons,
            backbone=backbone,
            max_context=getattr(config, "max_context", 4096),
            truncation_boundaries_out=truncation_boundaries,
        )
    else:
        rows = collect_model_scores(
            model,
            binned,
            vocab,
            concept_names,
            alerts,
            visit_start=visit_start,
            landmark_hours=landmark_hours,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
            horizons=horizons,
            backbone=backbone,
            max_context=getattr(config, "max_context", 4096),
            truncation_boundaries_out=truncation_boundaries,
            index_mode=index_mode,
        )
    # Runs for every backbone, not just "transformer": collect_model_scores'
    # row-construction path is unconditional on `packed` (see its own
    # docstring), so a landmark-selection bug isn't backbone-specific --
    # confirmed the hard way when the v2->v3 interleaved-visit bug turned
    # out to affect backbone="hybrid" dumps too, silently, because nothing
    # verified them. truncation_boundaries stays {} for backbone="hybrid"
    # (only PackedContextSampler ever populates it), so every subject goes
    # through the exact-match branch -- the truncated-subject branches are
    # simply never triggered, not skipped by a backbone check.
    if degraded_shard_dir is None:
        verify_packed_landmark_rows(
            rows,
            binned,
            alerts,
            landmark_hours=landmark_hours,
            truncation_boundaries=truncation_boundaries,
            index_mode=index_mode,
            max_context=getattr(config, "max_context", 4096),
        )
    if verify_against_dump is not None:
        verify_rows_match_dump(rows, times, verify_against_dump, horizons=horizons)

    baselines, features_by_event = _fit_and_score_gbm_baselines(
        baseline_shard_dir,
        stream_baseline=stream_baseline,
        max_baseline_shards=max_baseline_shards,
        config=config,
        source=source,
        binner=binner,
        alerts=alerts,
        horizons=horizons,
        landmark_hours=landmark_hours,
        baseline_feature_set=baseline_feature_set,
        tune_baselines=tune_baselines,
        binned=binned,
        rows=rows,
        task_set=task_set,
        index_mode=index_mode,
        prefit_baselines=prefit_baselines,
    )
    if fitted_baselines_out is not None and baselines is not None:
        fitted_baselines_out.update(baselines)

    if dump_rows_path is not None:
        context_cols = None
        context_names: Optional[List[str]] = None
        if features_by_event is not None and baseline_feature_set in (
            "strong",
            "strong_text",
        ):
            all_names = strong_feature_names()  # context columns lead either set
            keep_idx = [all_names.index(c) for c in ROW_DUMP_CONTEXT if c in all_names]
            context_names = [all_names[i] for i in keep_idx]
            context_cols = {k: v[:, keep_idx] for k, v in features_by_event.items()}
        table = index_row_table(
            rows,
            times,
            horizons=horizons,
            baselines=baselines,
            baseline_features_by_event=features_by_event,
            context_columns=context_cols,
            context_names=context_names,
        )
        table = _stamp_landmark_protocol_version(table)
        Path(dump_rows_path).parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(dump_rows_path)
        logger.info("[alerts] wrote %d index rows to %s", table.height, dump_rows_path)

    return score_alerts(
        rows,
        times,
        horizons=horizons,
        baselines=baselines,
        baseline_features_by_event=features_by_event,
    )


def _main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--baseline-shard-dir", default=None)
    parser.add_argument(
        "--degraded-shard-dir",
        default=None,
        help=(
            "score against this shard dir instead of --held-out-shard-dir "
            "(missingness stress protocol, docs/missingness_protocol.md; "
            "shards from python -m odyssey.data.degrade) -- labels and the "
            "visit envelope still come from --held-out-shard-dir"
        ),
    )
    parser.add_argument(
        "--verify-against-dump",
        default=None,
        help=(
            "assert this run's row set/labels exactly match a saved clean "
            "--dump-rows parquet before scoring (acceptance check for a "
            "degraded cell's alerts pass)"
        ),
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-baseline-shards", type=int, default=None)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument(
        "--index-mode",
        choices=INDEX_MODES,
        default="landmark",
        help=(
            "landmark (default): every --landmark-hours bucket within a visit; "
            "visit_end: one row per visit at discharge (30-day readmission)"
        ),
    )
    parser.add_argument(
        "--alerts",
        nargs="+",
        default=None,
        help=(
            "event names to score (default: the run's task set, filtered by "
            "--index-mode: within-visit events for landmark, next-visit events "
            "for visit_end)"
        ),
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=None,
        help=(
            "horizons in hours (default: 8 24 72 for landmark, 168 720 for visit_end)"
        ),
    )
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--baseline-features",
        choices=BASELINE_FEATURE_SETS,
        default="strong",
        help="feature set for the bespoke GBM (default: strong, best effort)",
    )
    parser.add_argument(
        "--no-tune-baselines",
        action="store_true",
        help="skip the GBM hyper-parameter search (fixed 200 rounds)",
    )
    parser.add_argument(
        "--stream-baseline-shards",
        action="store_true",
        help=(
            "fit the GBM baselines shard by shard instead of loading "
            "--baseline-shard-dir whole into memory; use for large "
            "--max-baseline-shards (full-scale runs, hundreds of shards)"
        ),
    )
    parser.add_argument(
        "--dump-rows",
        default=None,
        help="write the per-index-row score/outcome table as parquet (patient-level)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "allow clobbering an existing --output-json/--dump-rows file. "
            "Protocol-versioned science outputs are append-only by "
            "default -- a real, irreplaceable alerts_rows.parquet was "
            "lost to a silent overwrite on 2026-08-22. Pass this only "
            "when re-running the same run/protocol intentionally."
        ),
    )
    args = parser.parse_args()
    out = Path(args.output_json)
    refuse_existing_output(out, overwrite=args.overwrite, kind="alerts")
    if args.dump_rows is not None:
        refuse_existing_output(
            Path(args.dump_rows), overwrite=args.overwrite, kind="alerts rows"
        )
    run_dir = Path(args.run_dir)
    if args.horizons is not None:
        horizons: Sequence[float] = tuple(args.horizons)
    elif args.index_mode == "visit_end":
        horizons = READMISSION_HORIZONS_HOURS
    else:
        horizons = HORIZONS_HOURS
    chosen_alerts: Optional[List[AlertEvent]] = None
    if args.alerts is not None:
        by_name = {a.name: a for ts in ALERT_TASK_SETS.values() for a in ts}
        unknown = [n for n in args.alerts if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown --alerts {unknown}; known: {sorted(by_name)}")
        chosen_alerts = [by_name[n] for n in args.alerts]
    results = evaluate_alerts(
        run_dir,
        args.held_out_shard_dir,
        baseline_shard_dir=args.baseline_shard_dir,
        degraded_shard_dir=args.degraded_shard_dir,
        verify_against_dump=args.verify_against_dump,
        max_shards=args.max_shards,
        max_baseline_shards=args.max_baseline_shards,
        alerts=chosen_alerts,
        horizons=horizons,
        index_mode=args.index_mode,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        baseline_feature_set=args.baseline_features,
        tune_baselines=not args.no_tune_baselines,
        stream_baseline=args.stream_baseline_shards,
        dump_rows_path=args.dump_rows,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-record field, not a top-level wrapper: build_alert_finding (and
    # anything else reading this file) expects a bare list of records --
    # keep it dumb, don't change the top-level shape.
    records = [
        {**asdict(r), "landmark_protocol_version": LANDMARK_PROTOCOL_VERSION}
        for r in results
    ]
    out.write_text(json.dumps(records, indent=2))
    for r in results:
        logger.info(
            "[alerts] %-22s %5.0fh %-12s auroc=%.3f brier=%s n=%d pos=%d cens=%d",
            r.event,
            r.horizon_hours,
            r.scorer,
            r.auroc if r.auroc is not None else float("nan"),
            f"{r.brier:.4f}" if r.brier is not None else "-",
            r.n_at_risk,
            r.n_positive,
            r.n_censored,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _main()


__all__ = [
    "ALERT_EVENTS",
    "HORIZONS_HOURS",
    "AlertEvent",
    "AlertMetrics",
    "EventTimes",
    "IndexRow",
    "outcome_at_horizon",
    "collect_model_scores",
    "baseline_features",
    "strong_baseline_features",
    "BASELINE_FEATURE_SETS",
    "fit_baselines",
    "score_alerts",
    "index_row_table",
    "load_index_row_table",
    "verify_rows_match_dump",
    "evaluate_alerts",
]
