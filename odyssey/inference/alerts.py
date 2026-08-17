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
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
from odyssey.data.sequences import BIRTH_CODE
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler
from odyssey.data.value_binning import add_value_tokens, clinical_ranges_for_source
from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.inference.baseline_features import StrongFeatureBuilder
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import SequenceModel
from odyssey.models.time_to_event import probability_within
from odyssey.training.data import iter_patient_sequences, load_meds_shards
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

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


def _landmark_mask(
    time_hours: torch.Tensor,
    subject_ids: torch.Tensor,
    visit_ids: torch.Tensor,
    landmark_hours: float,
    visit_start_hours: torch.Tensor,
) -> torch.Tensor:
    """First real position of each visit's ``landmark_hours`` bucket, per lane."""
    bucket = torch.floor((time_hours - visit_start_hours) / landmark_hours)
    prev_bucket = torch.full_like(bucket, -1.0)
    prev_bucket[:, 1:] = bucket[:, :-1]
    same_visit = torch.zeros_like(subject_ids, dtype=torch.bool)
    same_visit[:, 1:] = (subject_ids[:, 1:] == subject_ids[:, :-1]) & (
        visit_ids[:, 1:] == visit_ids[:, :-1]
    )
    new_bucket = (bucket != prev_bucket) | ~same_visit
    return new_bucket & (subject_ids != NO_SUBJECT) & (visit_ids >= 0)


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
) -> Dict[str, List[IndexRow]]:
    """One streaming pass; per alert, index rows with model risk scores.

    Scores per row: ``concept`` (the alert's concept probability, if it
    has one), ``next_mass`` (softmax mass on the alert's tokens), and,
    when the model has per-event hazard heads covering the alert,
    ``hazard@{h}h`` for each horizon in ``horizons``: the head's
    ``P(event within h)``, a calibrated probability.
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
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
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
            keep = _landmark_mask(times, sids, vids, landmark_hours, starts)
            if not keep.any():
                continue
            probs = torch.softmax(logits[keep], dim=-1)
            hazards = (
                event_heads(fwd.features[keep]) if event_heads is not None else None
            )
            kept_sids = sids[keep].tolist()
            kept_vids = vids[keep].tolist()
            kept_times = times[keep].tolist()
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
                    rows[alert.name].append(IndexRow(int(s), int(v), float(t), scores))
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


class BaselineModel:
    """A fitted GBM plus the all-missing columns it had to fill at fit time.

    ``HistGradientBoostingClassifier`` handles missing values natively,
    which is what we want (missingness is informative), but its binner
    cannot fit a column that is missing everywhere (a curated lab that
    never appears in the fitting shards). Such a column carries no
    information, so it is filled with 0 at fit and, for consistency, at
    prediction.
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
    best: Tuple[float, Dict[str, float], int] = (np.inf, dict(GBM_GRID[0]), 200)
    for params in GBM_GRID:
        clf = HistGradientBoostingClassifier(
            random_state=seed, max_iter=GBM_MAX_ITER, early_stopping=False, **params
        )
        clf.fit(x[~is_val], y[~is_val])
        losses = [
            _log_loss(y[is_val], proba[:, 1])
            for proba in clf.staged_predict_proba(x[is_val])
        ]
        k = int(np.argmin(losses))
        if losses[k] < best[0]:
            best = (losses[k], dict(params), k + 1)
    return best[1], best[2]


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

    With ``tune`` the hyper-parameters and round count come from
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
        x_all = features[name]
        groups_all = np.array([r.subject_id for r in rows])
        for h in horizons:
            y = np.array(
                [outcome_at_horizon(r, train_times[name], h) for r in rows],
                dtype=object,
            )
            keep = np.array([v is not None for v in y])
            if keep.sum() < 50 or len({int(v) for v in y[keep]}) < 2:
                continue
            x_fit = x_all[keep]
            y_fit = y[keep].astype(int)
            fill_columns = np.isnan(x_fit).all(axis=0)
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
            models[(name, h)] = BaselineModel(
                clf,
                fill_columns,
                feature_set=feature_set,
                params={**params, "n_rounds": float(n_rounds)},
            )
            logger.info(
                "[alerts] GBM %s@%gh: %s features, rounds=%d, params=%s",
                name,
                h,
                feature_set,
                n_rounds,
                params,
            )
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


def score_alerts(
    rows: Dict[str, List[IndexRow]],
    times: Dict[str, EventTimes],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    baselines: Optional[Dict[Tuple[str, float], BaselineModel]] = None,
    baseline_features_by_event: Optional[Dict[str, np.ndarray]] = None,
) -> List[AlertMetrics]:
    """Score every (event, horizon, scorer) present in ``rows``."""
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
                model = baselines[(name, h)]
                p = model.predict_proba(x)
                results.append(
                    AlertMetrics(
                        event=name,
                        horizon_hours=h,
                        scorer="baseline_gbm",
                        n_at_risk=int(len(keep)),
                        n_positive=int(y.sum()),
                        n_censored=n_censored,
                        auroc=float(roc_auc_score(y, p)),
                        brier=float(brier_score_loss(y, p)),
                        calibration=_calibration(p, y),
                        baseline_feature_set=model.feature_set,
                        baseline_n_features=model.n_features,
                        baseline_params=model.params or None,
                    )
                )
    return results


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
) -> List[AlertMetrics]:
    """End to end: model scores + optional GBM baselines, scored on held-out."""
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
    )

    baselines = None
    features_by_event = None
    if baseline_shard_dir is not None:
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
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
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
    "evaluate_alerts",
]
