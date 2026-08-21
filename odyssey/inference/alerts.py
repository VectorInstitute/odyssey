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
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from odyssey.data.alert_events import (
    ALERT_EVENTS,
    AlertEvent,
    EventTimes,
    all_event_times,
    hours_since_origin,
    origin_hours,
)
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.sequences import BIRTH_CODE
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler
from odyssey.data.value_binning import (
    QuantileBinner,
    add_value_tokens,
    clinical_ranges_for_source,
)
from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.inference.baseline_features import StrongFeatureBuilder
from odyssey.inference.baseline_features import feature_names as strong_feature_names
from odyssey.inference.run_inference import load_run
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
#: v2 (current): LandmarkState is threaded across chunks, fixing that.
#:
#: _index_rows_from_events (the model-free path used for baseline fitting)
#: was never affected -- it has no chunking at all -- so this version only
#: describes collect_model_scores' output, and only that output carries
#: the tag (see _stamp_landmark_protocol_version, load_index_row_table).
LANDMARK_PROTOCOL_VERSION = 2

HORIZONS_HOURS: Tuple[float, ...] = (8.0, 24.0, 72.0)


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
    """Per-lane last-real-position state, threaded across streaming chunks.

    ``bucket``/``subject_id``/``visit_id`` are each ``(lanes,)``: the
    landmark bucket, subject, and visit of the last *real* (non-padding)
    position :func:`_landmark_mask` saw in that lane, across every call so
    far -- the same role the backbone's own recurrent state plays for
    model weights, carried by the caller from one chunk to the next
    (see :func:`collect_model_scores`).
    """

    bucket: torch.Tensor
    subject_id: torch.Tensor
    visit_id: torch.Tensor


def _landmark_mask(
    time_hours: torch.Tensor,
    subject_ids: torch.Tensor,
    visit_ids: torch.Tensor,
    landmark_hours: float,
    visit_start_hours: torch.Tensor,
    *,
    state: Optional[LandmarkState] = None,
) -> Tuple[torch.Tensor, LandmarkState]:
    """First real position of each visit's ``landmark_hours`` bucket, across chunks.

    Real bug this fixes: an earlier version had no ``state`` parameter at
    all -- called fresh per chunk, the first position of every chunk was
    unconditionally treated as a new landmark (nothing to compare it
    against), so any patient whose sequence spans more than one streaming
    chunk got a spurious extra landmark row at every chunk boundary, even
    when that position was still inside the same bucket as the position
    before it. Confirmed via
    ``tests/odyssey/inference/test_alerts.py``'s regression test: at
    eICU scale this over-counted landmark rows by ~23%.

    ``state`` (``None`` on the first call) carries the previous call's
    last real position per lane forward; without it, position 0 of every
    lane falls back to "nothing before it" (matching the original,
    single-chunk-only behavior) rather than crashing.

    Returns
    -------
    tuple[torch.Tensor, LandmarkState]
        The boolean landmark mask, and the updated state to pass into the
        next call.
    """
    lanes, chunk_len = time_hours.shape
    bucket = torch.floor((time_hours - visit_start_hours) / landmark_hours)

    prev_bucket = torch.full_like(bucket, -1.0)
    prev_bucket[:, 1:] = bucket[:, :-1]
    prev_subject = torch.full_like(subject_ids, NO_SUBJECT)
    prev_subject[:, 1:] = subject_ids[:, :-1]
    prev_visit = torch.full_like(visit_ids, -1)
    prev_visit[:, 1:] = visit_ids[:, :-1]
    if state is not None:
        prev_bucket[:, 0] = state.bucket
        prev_subject[:, 0] = state.subject_id
        prev_visit[:, 0] = state.visit_id

    same_visit = (subject_ids == prev_subject) & (visit_ids == prev_visit)
    new_bucket = (bucket != prev_bucket) | ~same_visit
    mask = new_bucket & (subject_ids != NO_SUBJECT) & (visit_ids >= 0)

    # Carry forward each lane's LAST real (non-padding) position for the
    # next call -- if a lane has no real position in this chunk at all
    # (a fully-padded chunk for that lane), keep whatever state it already
    # had rather than resetting to "nothing before it".
    real = subject_ids != NO_SUBJECT
    positions = torch.arange(chunk_len, device=time_hours.device).unsqueeze(0)
    real_positions = torch.where(real, positions, torch.full_like(positions, -1))
    last_idx = real_positions.max(dim=1).values
    has_real = last_idx >= 0
    gather_idx = last_idx.clamp(min=0).unsqueeze(1)
    chunk_last_bucket = bucket.gather(1, gather_idx).squeeze(1)
    chunk_last_subject = subject_ids.gather(1, gather_idx).squeeze(1)
    chunk_last_visit = visit_ids.gather(1, gather_idx).squeeze(1)
    if state is None:
        prev_state_bucket = torch.full(
            (lanes,), -1.0, dtype=bucket.dtype, device=time_hours.device
        )
        prev_state_subject = torch.full(
            (lanes,), NO_SUBJECT, dtype=subject_ids.dtype, device=time_hours.device
        )
        prev_state_visit = torch.full(
            (lanes,), -1, dtype=visit_ids.dtype, device=time_hours.device
        )
    else:
        prev_state_bucket = state.bucket
        prev_state_subject = state.subject_id
        prev_state_visit = state.visit_id
    new_state = LandmarkState(
        bucket=torch.where(has_real, chunk_last_bucket, prev_state_bucket),
        subject_id=torch.where(has_real, chunk_last_subject, prev_state_subject),
        visit_id=torch.where(has_real, chunk_last_visit, prev_state_visit),
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
) -> Dict[str, List[IndexRow]]:
    """One streaming pass; per alert, index rows with model risk scores.

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
    patients = iter_patient_sequences(events_binned, vocab)
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
            keep, landmark_state = _landmark_mask(
                times, sids, vids, landmark_hours, starts, state=landmark_state
            )
            if not keep.any():
                continue
            probs = torch.softmax(logits[keep], dim=-1)
            hazards = (
                event_heads(fwd.features[keep]) if event_heads is not None else None
            )
            kept_sids = sids[keep].tolist()
            kept_vids = vids[keep].tolist()
            kept_times = times[keep].tolist()
            truncated_ids = (
                set(sampler.truncated_subject_ids)
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
    return rows


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
BASELINE_FEATURE_SETS: Tuple[str, ...] = ("basic", "strong")


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
        merge_event_times(event_times, all_event_times(raw, alerts, source))
        shard_rows = _index_rows_from_events(
            binned, alerts, landmark_hours=landmark_hours
        )[alerts[0].name]
        if not shard_rows:
            continue
        if feature_set == "strong":
            feats = strong_baseline_features(binned, shard_rows, source=source)
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
) -> Dict[str, List[IndexRow]]:
    """Landmark index rows straight from events (no model), for baseline fitting."""
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
    starts = timed.group_by("subject_id", "hadm_id").agg(
        pl.col("_hours").min().alias("_start")
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


def verify_packed_landmark_rows(
    model_rows: Dict[str, List[IndexRow]],
    events_binned: pl.DataFrame,
    alerts: Sequence[AlertEvent],
    *,
    landmark_hours: float,
) -> List[str]:
    """Compare backbone="transformer"'s landmark set against the model-free truth.

    Two independent implementations of "which (subject, visit, time)
    triples are landmarks" (:func:`collect_model_scores`'s packed path
    and :func:`_index_rows_from_events`) only get to coexist once they
    are shown to agree, not assumed to -- this is that proof, run for
    real every time backbone="transformer" scores a run
    (:func:`evaluate_alerts`), not only in tests. Returns human-readable
    mismatch descriptions (empty = agrees exactly); logs a warning for
    any it finds rather than raising, matching
    :func:`~odyssey.utils.env_fingerprint.check_canary`'s house style for
    a real-time self-check that must not abort a running evaluation.
    """
    ground_truth = _index_rows_from_events(
        events_binned, alerts, landmark_hours=landmark_hours
    )
    problems = []
    for alert in alerts:
        got = _landmark_key_set(model_rows.get(alert.name, []))
        want = _landmark_key_set(ground_truth.get(alert.name, []))
        if got != want:
            missing = len(want - got)
            extra = len(got - want)
            problems.append(
                f"{alert.name}: packed landmark set disagrees with "
                f"_index_rows_from_events (missing {missing}, extra {extra})"
            )
    for p in problems:
        logger.warning(
            "[alerts] landmark row-set mismatch, backbone='transformer': %s -- "
            "the packed-path selection and the model-free ground truth should "
            "be provably identical (see collect_model_scores' docstring); "
            "treat these scores as suspect until this is understood",
            p,
        )
    return problems


def evaluate_alerts(
    run_dir: Union[str, Path],
    held_out_shard_dir: Union[str, Path],
    *,
    baseline_shard_dir: Optional[Union[str, Path]] = None,
    max_shards: Optional[int] = None,
    max_baseline_shards: Optional[int] = None,
    alerts: Sequence[AlertEvent] = ALERT_EVENTS,
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
) -> List[AlertMetrics]:
    """End to end: model scores + optional GBM baselines, scored on held-out.

    ``dump_rows_path`` writes the per-index-row table of
    :func:`index_row_table` as parquet (patient-level; keep it with the run).
    ``stream_baseline`` fits the GBM baselines shard by shard
    (:func:`fit_baselines_streaming`) instead of loading
    ``baseline_shard_dir`` whole into memory (:func:`fit_baselines`); use it
    once ``max_baseline_shards`` is large enough that the whole-frame path
    risks OOM (full-scale runs, hundreds of shards).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    source = getattr(config, "source", "mimic_iv")
    concept_names = [c.name for c in concepts_for_source(source)]

    raw = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw = maybe_normalize(
        raw, enabled=getattr(config, "normalize_medications", False), source=source
    )
    raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
    times = all_event_times(raw, alerts, source)
    visit_start = _visit_starts(raw)
    binned = add_value_tokens(raw, binner, source=source)
    del raw

    backbone = getattr(config, "backbone", "hybrid")
    logger.info("[alerts] collecting model scores at %.0fh landmarks", landmark_hours)
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
    )
    if backbone == "transformer":
        verify_packed_landmark_rows(rows, binned, alerts, landmark_hours=landmark_hours)

    baselines = None
    features_by_event = None
    if baseline_shard_dir is not None and stream_baseline:
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
        )
        features_by_event = features_for_events(
            binned, rows, source=source, feature_set=baseline_feature_set
        )
    elif baseline_shard_dir is not None:
        logger.info("[alerts] fitting GBM baselines on %s", baseline_shard_dir)
        train_raw = load_meds_shards(baseline_shard_dir, max_shards=max_baseline_shards)
        train_raw = maybe_normalize(
            train_raw,
            enabled=getattr(config, "normalize_medications", False),
            source=source,
        )
        train_raw = maybe_history_recap(
            train_raw, enabled=getattr(config, "history_recap", False)
        )
        train_times = all_event_times(train_raw, alerts, source)
        train_binned = add_value_tokens(train_raw, binner, source=source)
        del train_raw
        train_rows = _index_rows_from_events(
            train_binned, alerts, landmark_hours=landmark_hours
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

    if dump_rows_path is not None:
        context_cols = None
        context_names: Optional[List[str]] = None
        if features_by_event is not None and baseline_feature_set == "strong":
            all_names = strong_feature_names()
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
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-baseline-shards", type=int, default=None)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
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
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    results = evaluate_alerts(
        run_dir,
        args.held_out_shard_dir,
        baseline_shard_dir=args.baseline_shard_dir,
        max_shards=args.max_shards,
        max_baseline_shards=args.max_baseline_shards,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        baseline_feature_set=args.baseline_features,
        tune_baselines=not args.no_tune_baselines,
        stream_baseline=args.stream_baseline_shards,
        dump_rows_path=args.dump_rows,
    )
    out = Path(args.output_json)
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
    "evaluate_alerts",
]
