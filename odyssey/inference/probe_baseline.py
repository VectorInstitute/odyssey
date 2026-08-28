"""EHRSHOT-style frozen-probe benchmark: representation quality, not more heads.

Direct response to Amrit's 2026-08-28 redirect: don't keep adding trained
alert heads for every new clinical task (multi-task loss competition for
shared-backbone gradient is real and already documented -- e.g.
eicu_subset_v9's recency channel cost -2 to -9pp on other forecast
families). Instead measure whether the GENERAL representation already
supports a task under a FROZEN linear probe, the same evaluation
philosophy EHRSHOT itself uses (Wornow et al., "EHRSHOT: An EHR Benchmark
for Few-Shot Evaluation of Foundation Models", NeurIPS Datasets &
Benchmarks 2023): score many diverse downstream tasks against a frozen
representation, with no dedicated trained head per task.

Two task shapes, because they need genuinely different label machinery:

- The 5 "lab test value" tasks (:data:`odyssey.data.alert_events.PROBE_EVENTS`)
  are ordinary :class:`~odyssey.data.alert_events.AlertEvent`\\ s over
  concepts :mod:`odyssey.data.concepts` already defines (task_set v3) --
  "does this concept trigger within horizon h", scored the same way
  vasopressor_start/acute_kidney_injury already are. They plug into the
  EXISTING :func:`~odyssey.inference.alerts.fit_baselines` (a real GBM
  comparator, for free) and :func:`~odyssey.inference.alerts.score_alerts`'s
  ``extra_baselines`` hook -- the same generic mechanism TabICL/EBM/
  SurvivalPFN/MEDS-Tab already use, not a new dispatch path.
- ``long_los`` (EHRSHOT's "long length of stay", > 7 days) is a STATIC
  per-visit label, not a first-trigger-within-horizon event -- predicting
  "will this stay be long" from a landmark taken on day 6 is nearly
  trivial, so it is scored from a single EARLY snapshot instead of a
  landmark sweep. This does not fit
  :func:`~odyssey.inference.alerts.outcome_at_horizon`'s onset/censor
  shape (see :func:`long_los_task`'s docstring), so it gets its own small,
  honest path and its own GBM comparator (:func:`fit_gbm_for_binary_label`)
  rather than being shoehorned into ``EventTimes``.

Deliberately out of scope for this pass (see the plan this module was
built from): EHRSHOT's "Assignment of New Diagnoses" tasks (need vetted
ICD/SNOMED code lists not yet in this repo) and "Chest X-ray Findings"
(needs structured labels this project doesn't have). :data:`PROBE_EVENTS`
and :func:`run_probe_benchmark` are built so adding either later is one
more registry entry, not a redesign.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, cast

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import (
    PROBE_EVENTS,
    AlertEvent,
    EventTimes,
    all_event_times,
    visit_envelope,
)
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import (
    HORIZONS_HOURS,
    AlertMetrics,  # re-export, see __all__
    IndexRow,
    _load_prepared_raw,
    _positive_class_proba,
    _ScoredBaseline,
    _visit_starts,
    features_for_events,
    fit_baselines,
    outcome_at_horizon,
    score_alerts,
)
from odyssey.inference.embedding_probe import Key, collect_embeddings
from odyssey.inference.run_inference import load_run
from odyssey.inference.uncertainty import BootstrapAUROC, bootstrap_auroc
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


logger = logging.getLogger(__name__)

#: Minimum kept rows and minimum positives for a probe/GBM fit to run at
#: all -- mirrors odyssey.inference.alerts._fit_baseline_grid's own guard
#: (`len(keep) < 50 or len(unique(y)) < 2`), so a task with too few events
#: is silently absent from the report rather than fit on noise.
MIN_FIT_ROWS = 50

#: task_set version PROBE_EVENTS' onset labels are computed against.
#: DELIBERATELY independent of the checkpoint's own task_set: these are
#: ground-truth concept triggers read straight off the raw record via
#: odyssey.data.concepts (task_set v3 is where anemia/hyperkalemia/
#: hypoglycemia/hyponatremia/thrombocytopenia are defined), not the
#: model's own bottleneck outputs -- the whole point of this benchmark is
#: to probe a run's REPRESENTATION against tasks it may never have been
#: trained to predict. A v1-task_set checkpoint (or any other) is exactly
#: as probeable as a v3 one; only the labels' source concepts need v3.
PROBE_LABEL_TASK_SET = "v3"

#: EHRSHOT scores "long length of stay" from an early admission snapshot,
#: not a landmark sweep (predicting it late in a long stay is trivial).
#: This is the snapshot window in hours-into-visit; the earliest landmark
#: row inside it is used, one row per visit.
LONG_LOS_SNAPSHOT_BAND_HOURS: tuple[float, float] = (20.0, 28.0)
LONG_LOS_THRESHOLD_HOURS = 168.0  # 7 days


class ProbeBaselineModel:
    """A frozen StandardScaler + LogisticRegression probe on backbone embeddings.

    Duck-typed to satisfy
    :class:`odyssey.inference.alerts._ScoredBaseline` (the same generic
    interface :class:`~odyssey.inference.tabicl_baseline.TabICLBaselineModel`
    and the built-in GBM satisfy) without inheriting from anything --
    ``score_alerts`` only ever reads ``feature_set``/``n_features``/``params``/
    ``predict_proba``.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        clf: LogisticRegression,
        *,
        feature_set: str,
    ) -> None:
        self.scaler = scaler
        self.clf = clf
        self.feature_set = feature_set
        self.n_features = int(scaler.n_features_in_)
        self.params: dict[str, float] = {"C": float(clf.C)}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Positive-class probabilities, ``(n,)``."""
        proba = self.clf.predict_proba(self.scaler.transform(x))
        result: np.ndarray = _positive_class_proba(self.clf, proba)
        return result


def fit_binary_probe(
    x_train: np.ndarray, y_train: np.ndarray, *, feature_set: str
) -> ProbeBaselineModel:
    """Fit one frozen StandardScaler + LogisticRegression probe."""
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(x_train), y_train)
    return ProbeBaselineModel(scaler, clf, feature_set=feature_set)


def fit_probe_baselines(
    train_rows: dict[str, list[IndexRow]],
    train_embeddings_by_event: dict[str, np.ndarray],
    train_times: dict[str, EventTimes],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    embedding_kind: str = "post_bottleneck",
) -> dict[tuple[str, float], ProbeBaselineModel]:
    """One probe per (event, horizon), on embeddings instead of GBM features.

    Signature deliberately mirrors
    :func:`odyssey.inference.alerts.fit_baselines` (same
    ``train_rows``/``train_times``/``horizons`` shape) so a probe and a
    GBM can be fit from the same rows and compared like-for-like via
    :func:`~odyssey.inference.alerts.score_alerts`'s ``extra_baselines``
    hook. Unlike the GBM path there is no missing-value handling or
    hyperparameter search: embeddings are dense floats with no
    missingness by construction, and a plain frozen linear probe is the
    whole point (it is the ceiling this benchmark is measuring, not a
    tuned classifier).
    """
    models: dict[tuple[str, float], ProbeBaselineModel] = {}
    for name, rows in train_rows.items():
        if not rows or name not in train_embeddings_by_event:
            continue
        x_all = train_embeddings_by_event[name]
        for h in horizons:
            y = np.array(
                [outcome_at_horizon(r, train_times[name], h) for r in rows],
                dtype=object,
            )
            keep = np.flatnonzero([v is not None for v in y])
            if len(keep) < MIN_FIT_ROWS or len({int(y[i]) for i in keep}) < 2:
                continue
            model = fit_binary_probe(
                x_all[keep], y[keep].astype(int), feature_set=embedding_kind
            )
            models[(name, h)] = model
    return models


def probe_features_by_event(
    alerts: Sequence[AlertEvent], keys: Sequence[Key], embeddings: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the ``features_by_event`` shape ``score_alerts``'s hook wants.

    Every :data:`~odyssey.data.alert_events.PROBE_EVENTS` task shares the
    same landmark rows (landmark selection is event-independent, see
    :func:`~odyssey.inference.embedding_probe.collect_embeddings`), so the
    same embedding array is reused for every event name -- mirroring how
    :func:`~odyssey.inference.alerts.features_for_events` reuses one
    feature matrix across events for the same reason.
    """
    if len(keys) != embeddings.shape[0]:
        raise ValueError(
            f"keys ({len(keys)}) and embeddings ({embeddings.shape[0]}) length mismatch"
        )
    return {a.name: embeddings for a in alerts}


def long_los_task(
    keys: Sequence[Key],
    embeddings: np.ndarray,
    envelope: dict[tuple[int, int], tuple[float, float]],
    *,
    threshold_hours: float = LONG_LOS_THRESHOLD_HOURS,
    snapshot_band_hours: tuple[float, float] = LONG_LOS_SNAPSHOT_BAND_HOURS,
) -> tuple[list[Key], np.ndarray, np.ndarray]:
    """One row per visit, from the earliest landmark inside the snapshot band.

    ``long_los`` does not fit :func:`~odyssey.inference.alerts.outcome_at_horizon`'s
    onset/censor shape: that machinery answers "does the onset occur within
    h hours of THIS landmark", but "will this visit's TOTAL length of stay
    exceed 7 days" is a single static fact about the whole visit, knowable
    only once discharge is observed, being predicted from a partial-
    observation snapshot -- forcing it through ``EventTimes`` would either
    make short-stay (negative) visits read as censored (wrong: they are a
    real, known negative) or require treating "discharge" itself as the
    forecast target (a different, already-covered question). So this
    builds the label directly from :func:`~odyssey.data.alert_events.visit_envelope`
    instead.

    Only one row per visit is kept (the earliest inside
    ``snapshot_band_hours``, matching EHRSHOT's own single early-snapshot
    scoring), not a landmark sweep: predicting a stay is long from a
    landmark taken on day 6 is nearly trivial and would inflate AUROC for
    reasons unrelated to representation quality.
    """
    best_for_visit: dict[tuple[int, int], int] = {}
    for i, (s, v, t) in enumerate(keys):
        env = envelope.get((s, v))
        if env is None:
            continue
        start, _end = env
        hours_into_visit = t - start
        if not (snapshot_band_hours[0] <= hours_into_visit < snapshot_band_hours[1]):
            continue
        existing = best_for_visit.get((s, v))
        if existing is None or keys[existing][2] > t:
            best_for_visit[(s, v)] = i
    idx = sorted(best_for_visit.values())
    filtered_keys = [keys[i] for i in idx]
    filtered_embeddings = embeddings[idx]
    y = np.array(
        [
            1.0
            if (envelope[(s, v)][1] - envelope[(s, v)][0]) > threshold_hours
            else 0.0
            for (s, v, _t) in filtered_keys
        ],
        dtype=np.float64,
    )
    return filtered_keys, filtered_embeddings, y


def fit_gbm_for_binary_label(
    x_train: np.ndarray, y_train: np.ndarray, *, seed: int = 0
) -> HistGradientBoostingClassifier:
    """Fit a plain GBM comparator for a static binary label (e.g. long_los).

    Not :func:`~odyssey.inference.alerts._fit_baseline_grid`: that
    function's censoring/horizon machinery is for ``EventTimes``-shaped
    time-to-event targets, which long_los deliberately isn't (see
    :func:`long_los_task`). A fixed 200-round fit, matching
    ``fit_baselines(..., tune=False)``'s fast-path convention -- this is a
    comparator for a benchmark, not the paper-grade tuned GBM.
    """
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(x_train, y_train)
    return clf


@dataclass
class ProbeCell:
    """One task's probe/GBM comparison, with subject-clustered bootstrap CIs."""

    task: str
    horizon_hours: Optional[float]
    n_at_risk: int
    n_positive: int
    probe_pre_auroc: Optional[float] = None
    probe_post_auroc: Optional[float] = None
    gbm_auroc: Optional[float] = None
    probe_post_ci: Optional[BootstrapAUROC] = None
    gbm_ci: Optional[BootstrapAUROC] = None


@dataclass
class ProbeBenchmarkResult:
    """Full report: the concept-trigger cells, then long_los, then raw AlertMetrics."""

    cells: list[ProbeCell] = field(default_factory=list)
    alert_metrics: list[AlertMetrics] = field(default_factory=list)


def _load_split(
    shard_dir: str, max_shards: int, config: object, source: str
) -> tuple[
    pl.DataFrame,
    dict[tuple[int, int], float],
    dict[tuple[int, int], tuple[float, float]],
]:
    activate_sidecars(shard_dir)
    raw = _load_prepared_raw(shard_dir, max_shards, config, source)
    visit_start = _visit_starts(raw)
    envelope = visit_envelope(raw)
    return raw, visit_start, envelope


def run_probe_benchmark(
    run_dir: str,
    train_shard_dir: str,
    held_out_shard_dir: str,
    *,
    max_train_shards: int = 5,
    max_held_out_shards: int = 4,
    landmark_hours: float = 4.0,
    num_lanes: int = 64,
    chunk_size: int = 512,
    horizons: Sequence[float] = HORIZONS_HOURS,
    device: Optional[str] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> ProbeBenchmarkResult:
    """Fit and score the EHRSHOT-style probe benchmark against one checkpoint.

    Orchestrates: load the run and both data splits; collect pre/post-
    bottleneck embeddings once per split
    (:func:`~odyssey.inference.embedding_probe.collect_embeddings`); for
    the 5 concept-trigger tasks, build one canonical landmark row list
    directly from the collected keys (so embeddings, the real GBM baseline,
    and the probe are all aligned by construction -- no cross-pass
    reordering), fit the real tuned-feature-set GBM
    (:func:`~odyssey.inference.alerts.fit_baselines`, reused unmodified)
    and both pre- and post-bottleneck probes, then score everything through
    :func:`~odyssey.inference.alerts.score_alerts`'s existing
    ``extra_baselines`` hook; for long_los, run its own snapshot-based path.
    Every cell also gets a subject-clustered bootstrap AUROC
    (:func:`~odyssey.inference.uncertainty.bootstrap_auroc`) for the probe
    and the GBM, so "probe beats/loses to GBM" is read against a CI, not a
    point estimate -- the same discipline `scripts/probe_ci_check.py`
    already established for hazard-vs-probe comparisons.
    """
    resolved_device = device or ("cuda" if _cuda_available() else "cpu")
    model, vocab, binner, config = load_run(run_dir, device=resolved_device)
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            f"{run_dir} is not a concept-bottleneck run (model_kind must be "
            "'cbm'): this benchmark probes pre/post-bottleneck embeddings, "
            "which only exist on ConceptBottleneckSequenceModel."
        )
    source = getattr(config, "source", "mimic_iv")

    logger.info("loading %d train shard(s) from %s", max_train_shards, train_shard_dir)
    train_raw, train_visit_start, train_envelope = _load_split(
        train_shard_dir, max_train_shards, config, source
    )
    train_binned = add_value_tokens(train_raw, binner, source=source)
    logger.info(
        "loading %d held-out shard(s) from %s", max_held_out_shards, held_out_shard_dir
    )
    held_raw, held_visit_start, held_envelope = _load_split(
        held_out_shard_dir, max_held_out_shards, config, source
    )
    held_binned = add_value_tokens(held_raw, binner, source=source)

    train_keys, train_pre, train_post, _, _, _ = collect_embeddings(
        model,
        train_binned,
        vocab,
        landmark_alerts=PROBE_EVENTS,
        visit_end_alerts=[],
        visit_start=train_visit_start,
        landmark_hours=landmark_hours,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=resolved_device,
    )
    logger.info("train: %d landmark rows", len(train_keys))
    held_keys, held_pre, held_post, _, _, _ = collect_embeddings(
        model,
        held_binned,
        vocab,
        landmark_alerts=PROBE_EVENTS,
        visit_end_alerts=[],
        visit_start=held_visit_start,
        landmark_hours=landmark_hours,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=resolved_device,
    )
    logger.info("held-out: %d landmark rows", len(held_keys))

    result = ProbeBenchmarkResult()
    result.alert_metrics.extend(
        _score_concept_trigger_tasks(
            train_raw=train_raw,
            held_raw=held_raw,
            train_binned=train_binned,
            held_binned=held_binned,
            train_keys=train_keys,
            train_pre=train_pre,
            train_post=train_post,
            held_keys=held_keys,
            held_pre=held_pre,
            held_post=held_post,
            source=source,
            horizons=horizons,
            n_boot=n_boot,
            seed=seed,
            cells_out=result.cells,
        )
    )
    _score_long_los(
        train_keys=train_keys,
        train_post=train_post,
        train_envelope=train_envelope,
        held_keys=held_keys,
        held_post=held_post,
        held_envelope=held_envelope,
        n_boot=n_boot,
        seed=seed,
        cells_out=result.cells,
    )
    return result


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _score_concept_trigger_tasks(
    *,
    train_raw: pl.DataFrame,
    held_raw: pl.DataFrame,
    train_binned: pl.DataFrame,
    held_binned: pl.DataFrame,
    train_keys: list[Key],
    train_pre: np.ndarray,
    train_post: np.ndarray,
    held_keys: list[Key],
    held_pre: np.ndarray,
    held_post: np.ndarray,
    source: str,
    horizons: Sequence[float],
    n_boot: int,
    seed: int,
    cells_out: list[ProbeCell],
) -> list[AlertMetrics]:
    train_rows = {a.name: _rows_from_keys(train_keys) for a in PROBE_EVENTS}
    held_rows = {a.name: _rows_from_keys(held_keys) for a in PROBE_EVENTS}
    train_times = all_event_times(
        train_raw, PROBE_EVENTS, source, task_set=PROBE_LABEL_TASK_SET
    )
    held_times = all_event_times(
        held_raw, PROBE_EVENTS, source, task_set=PROBE_LABEL_TASK_SET
    )

    gbm_models = fit_baselines(
        train_binned, train_rows, train_times, horizons=horizons, source=source
    )
    held_gbm_features = features_for_events(held_binned, held_rows, source=source)

    probe_pre_models = fit_probe_baselines(
        train_rows,
        probe_features_by_event(PROBE_EVENTS, train_keys, train_pre),
        train_times,
        horizons=horizons,
        embedding_kind="pre_bottleneck",
    )
    probe_post_models = fit_probe_baselines(
        train_rows,
        probe_features_by_event(PROBE_EVENTS, train_keys, train_post),
        train_times,
        horizons=horizons,
        embedding_kind="post_bottleneck",
    )
    held_pre_features = probe_features_by_event(PROBE_EVENTS, held_keys, held_pre)
    held_post_features = probe_features_by_event(PROBE_EVENTS, held_keys, held_post)

    metrics = score_alerts(
        held_rows,
        held_times,
        horizons=horizons,
        baselines=gbm_models,
        baseline_features_by_event=held_gbm_features,
        # ProbeBaselineModel structurally satisfies _ScoredBaseline, but
        # dict is invariant in its value type, so the cast is needed even
        # though this is sound (nothing here mutates the dict as a
        # different concrete model type).
        extra_baselines=cast(
            "dict[str, tuple[dict[tuple[str, float], _ScoredBaseline], dict[str, np.ndarray]]]",
            {
                "probe_pre": (probe_pre_models, held_pre_features),
                "probe_post": (probe_post_models, held_post_features),
            },
        ),
    )

    by_task: dict[tuple[str, float], dict[str, AlertMetrics]] = {}
    for m in metrics:
        by_task.setdefault((m.event, m.horizon_hours), {})[m.scorer] = m
    subject_ids = np.array([k[0] for k in held_keys])
    for (name, h), scorers in sorted(by_task.items()):
        gbm = scorers.get("baseline_gbm")
        pre = scorers.get("probe_pre")
        post = scorers.get("probe_post")
        any_scorer = post or gbm or pre
        cell = ProbeCell(
            task=name,
            horizon_hours=h,
            n_at_risk=any_scorer.n_at_risk if any_scorer else 0,
            n_positive=any_scorer.n_positive if any_scorer else 0,
            probe_pre_auroc=pre.auroc if pre else None,
            probe_post_auroc=post.auroc if post else None,
            gbm_auroc=gbm.auroc if gbm else None,
        )
        labels = _cell_labels(held_rows[name], held_times[name], h)
        if labels is not None:
            y, mask = labels
            if post is not None and (name, h) in probe_post_models:
                cell.probe_post_ci = _bootstrap_ci(
                    probe_post_models[(name, h)],
                    held_post_features[name][mask],
                    y,
                    subject_ids[mask],
                    n_boot=n_boot,
                    seed=seed,
                )
            if gbm is not None and (name, h) in gbm_models:
                cell.gbm_ci = _bootstrap_ci(
                    gbm_models[(name, h)],
                    held_gbm_features[name][mask],
                    y,
                    subject_ids[mask],
                    n_boot=n_boot,
                    seed=seed,
                )
        cells_out.append(cell)
    return metrics


def _rows_from_keys(keys: Sequence[Key]) -> list[IndexRow]:
    return [IndexRow(s, v, t) for s, v, t in keys]


def _cell_labels(
    rows: list[IndexRow], times: EventTimes, horizon: float
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """At-risk (non-censored) labels and the row mask they came from, or None."""
    outcomes = [outcome_at_horizon(r, times, horizon) for r in rows]
    mask = np.array([o is not None for o in outcomes])
    if not mask.any():
        return None
    y = np.array([o for o in outcomes if o is not None], dtype=int)
    if y.min() == y.max():
        return None
    return y, mask


def _bootstrap_ci(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Optional[BootstrapAUROC]:
    """Subject-clustered bootstrap AUROC for a fitted ``_ScoredBaseline``-like model."""
    p = model.predict_proba(x)  # type: ignore[attr-defined]
    return bootstrap_auroc(y, p, subject_ids, n_boot=n_boot, seed=seed)


def _score_long_los(
    *,
    train_keys: list[Key],
    train_post: np.ndarray,
    train_envelope: dict[tuple[int, int], tuple[float, float]],
    held_keys: list[Key],
    held_post: np.ndarray,
    held_envelope: dict[tuple[int, int], tuple[float, float]],
    n_boot: int,
    seed: int,
    cells_out: list[ProbeCell],
) -> None:
    train_los_keys, train_los_x, train_los_y = long_los_task(
        train_keys, train_post, train_envelope
    )
    held_los_keys, held_los_x, held_los_y = long_los_task(
        held_keys, held_post, held_envelope
    )
    if (
        len(train_los_keys) < MIN_FIT_ROWS
        or len(np.unique(train_los_y)) < 2
        or len(held_los_keys) < MIN_FIT_ROWS
        or len(np.unique(held_los_y)) < 2
    ):
        logger.info(
            "long_los: too few rows or single-class after snapshot-band "
            "filtering (train=%d, held=%d) -- skipped",
            len(train_los_keys),
            len(held_los_keys),
        )
        return

    probe = fit_binary_probe(train_los_x, train_los_y, feature_set="post_bottleneck")
    gbm = fit_gbm_for_binary_label(train_los_x, train_los_y, seed=seed)
    held_subjects = np.array([k[0] for k in held_los_keys])

    cell = ProbeCell(
        task="long_los",
        horizon_hours=None,
        n_at_risk=len(held_los_y),
        n_positive=int(held_los_y.sum()),
        probe_post_auroc=float(
            roc_auc_score(held_los_y, probe.predict_proba(held_los_x))
        ),
        gbm_auroc=float(
            roc_auc_score(
                held_los_y, _positive_class_proba(gbm, gbm.predict_proba(held_los_x))
            )
        ),
    )
    cell.probe_post_ci = _bootstrap_ci(
        probe, held_los_x, held_los_y, held_subjects, n_boot=n_boot, seed=seed
    )
    cell.gbm_ci = bootstrap_auroc(
        held_los_y,
        _positive_class_proba(gbm, gbm.predict_proba(held_los_x)),
        held_subjects,
        n_boot=n_boot,
        seed=seed,
    )
    cells_out.append(cell)


__all__ = [
    "LONG_LOS_SNAPSHOT_BAND_HOURS",
    "LONG_LOS_THRESHOLD_HOURS",
    "MIN_FIT_ROWS",
    "AlertMetrics",
    "ProbeBaselineModel",
    "ProbeBenchmarkResult",
    "ProbeCell",
    "fit_binary_probe",
    "fit_gbm_for_binary_label",
    "fit_probe_baselines",
    "long_los_task",
    "probe_features_by_event",
    "run_probe_benchmark",
]
