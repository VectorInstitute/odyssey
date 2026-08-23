"""InterpretML's ExplainableBoostingClassifier: an optional third baseline.

Alongside the tuned ``HistGradientBoostingClassifier`` baseline
(:mod:`odyssey.inference.alerts`) and the zero-shot TabICLv2 baseline
(:mod:`odyssey.inference.tabicl_baseline`), this module fits an
`ExplainableBoostingClassifier <https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html>`_
(InterpretML's package ``interpret``, MIT-licensed), a modern GA2M-style
generalized additive model: bagging plus gradient boosting plus automatic
pairwise-interaction detection, trained the same way as the GBM (real
gradient descent on our features, not TabICL's in-context, zero-shot
approach). Where it differs from the GBM baseline is what it yields:
per-feature shape functions and pairwise interaction terms are genuinely
intrinsic, training-time interpretability artifacts, not a post-hoc
explainer fitted after the fact -- entry 29's comparator plan names this
as the gap it fills.

Optional dependency, not installed by default -- ``uv sync --extra ebm``
(or ``pip install interpret``). The import is deferred to call time (see
:func:`_load_ebm_classifier`), mirroring how
:mod:`odyssey.inference.tabicl_baseline` defers its own optional import,
so importing this module never requires the package.

Missing values are handled natively by ``ExplainableBoostingClassifier``
itself (``missing="separate"`` by default, giving missing values their own
leaf/bin), verified against the current API reference rather than assumed
-- unlike the GBM baseline, there is no fill-up front needed here.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from odyssey.inference.alerts import (
    EventTimes,
    IndexRow,
    _positive_class_proba,
    features_for_events,
    outcome_at_horizon,
)
from odyssey.inference.fit_cache import FitCache


logger = logging.getLogger(__name__)

# Below this many at-risk rows, a GA2M fit has too little signal to be
# meaningful; mirrors odyssey.inference.alerts._fit_baseline_grid's own
# floor (50) for the GBM baseline, same reasoning.
EBM_MIN_ROWS = 50

# Default row cap on the fit (overridable per call via fit_ebm_baselines'
# max_rows). Subject-grouped (whole subjects are kept or dropped, never
# split) unlike the GBM's final-fit cap, which subsamples individual rows --
# a subject straddling the cap boundary would otherwise contribute some but
# not all of their at-risk positions, a subtle, avoidable leakage-adjacent
# inconsistency for a baseline meant to be directly comparable to the GBM.
#
# 100k, not the 500k first tried: at full project scale (30 shards, 3
# horizons), EBM's *default* quality settings (outer_bags=14, each bag
# boosting many rounds) took over an hour on a single (event, horizon) fit
# at the 500k cap and still hadn't finished, dramatically slower than EBM's
# own stated large-scale numbers on this host's core count specifically --
# entry 30's method note has the measured wall-clock. This default plus
# DEFAULT_OUTER_BAGS is the speed-tuned configuration chosen to get a full
# 12-cell table in minutes rather than hours; a slower, uncapped,
# default-quality confirmation run is scheduled only if a speed-tuned
# number turns out close enough to the GBM or the hazard head to matter.
EBM_MAX_ROWS = 100_000

# EBM's own default (14) trades wall-clock for a small ensemble-averaging
# accuracy gain; halved here for the speed-tuned pass. Left as an explicit
# parameter (not hardcoded) so the later uncapped confirmation run can pass
# the real default back in without another code change.
DEFAULT_OUTER_BAGS = 4


def _load_ebm_classifier() -> Any:
    """Import and return ``interpret.glassbox.ExplainableBoostingClassifier``.

    Deferred so nothing in this module requires ``interpret`` to be
    installed except the call path that actually fits one -- see the
    module docstring.
    """
    try:
        from interpret.glassbox import ExplainableBoostingClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "EBM baseline requires the optional `interpret` package: "
            "`uv sync --extra ebm` (or `pip install interpret`). See "
            "odyssey.inference.ebm_baseline's module docstring for what "
            "it is and why it was added."
        ) from exc
    return ExplainableBoostingClassifier


@dataclass
class EBMBaselineModel:
    """A fitted ``ExplainableBoostingClassifier`` for one (event, horizon).

    Duck-typed to satisfy :class:`odyssey.inference.alerts._ScoredBaseline`
    (``predict_proba``, ``feature_set``, ``n_features``, ``params``),
    exactly like :class:`~odyssey.inference.tabicl_baseline.TabICLBaselineModel`,
    so it drops into :func:`~odyssey.inference.alerts.score_alerts`'s
    ``extra_baselines`` hook alongside the GBM and TabICL baselines.
    """

    clf: object
    """A fitted ``interpret.glassbox.ExplainableBoostingClassifier``
    (typed ``object`` so this module's public dataclass doesn't force an
    unconditional import of the optional ``interpret`` package just to
    reference its type)."""

    feature_set: str = "strong"
    n_features: int = 0
    params: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Positive-class (label ``1``) probabilities, ``(n,)``."""
        proba = self.clf.predict_proba(x)  # type: ignore[attr-defined]
        return _positive_class_proba(self.clf, proba)


def _grouped_subsample(
    keep: np.ndarray, groups: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Shrink ``keep`` to at most ``cap`` rows, dropping whole subjects only.

    Shuffles subjects, then adds each subject's full row set in turn until
    the cap would be exceeded, so no subject contributes a partial set of
    their own at-risk positions.
    """
    subjects, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(subjects))
    cum = np.cumsum(counts[order])
    # side="right" already gives the count of prefix subjects whose
    # cumulative row count stays at or under cap; max(..., 1) guarantees at
    # least one subject even if their row count alone exceeds cap.
    n_subjects = max(1, int(np.searchsorted(cum, cap, side="right")))
    n_subjects = min(n_subjects, len(subjects))
    selected = set(subjects[order[:n_subjects]].tolist())
    mask = np.array([g in selected for g in groups])
    result: np.ndarray = keep[mask]
    return result


def _fit_one_ebm(
    x_all: np.ndarray,
    rows: Sequence[IndexRow],
    times: EventTimes,
    *,
    horizons: Sequence[float],
    feature_set: str,
    seed: int,
    event_name: str,
    n_jobs: int,
    max_rows: int,
    outer_bags: int,
    cache: Optional[FitCache] = None,
) -> dict[float, EBMBaselineModel]:
    """Fit one EBM per horizon for a single event, given a feature matrix.

    Structurally mirrors :func:`odyssey.inference.alerts._fit_baseline_grid`
    (same row selection, same skip condition) so the GBM, TabICL, and EBM
    baselines are all trained on directly comparable data. Rows are capped
    at ``max_rows``, subject-grouped (see :func:`_grouped_subsample`), and
    ``outer_bags`` controls the ensemble size EBM's own default (14) would
    otherwise use -- see :data:`EBM_MAX_ROWS` and :data:`DEFAULT_OUTER_BAGS`
    for why the speed-tuned defaults differ from EBM's own.

    ``cache``, if given, is checked per horizon before fitting and written
    to immediately after -- individual EBM fits here have run past an
    hour each at full project scale (see :data:`EBM_MAX_ROWS`'s note), so
    this is where losing an in-progress run costs the most -- see
    :mod:`odyssey.inference.fit_cache`.  ``features``, if given, is the
    precomputed per-event feature dict for this function's
    ``feature_set`` (see
    :func:`odyssey.inference.baseline_prep.prepare_baseline_data`);
    ``train_events_binned`` is then unused and may be empty.
    """
    ebm_classifier_cls = None
    groups_all = np.array([r.subject_id for r in rows])
    rng = np.random.default_rng(seed)
    out: dict[float, EBMBaselineModel] = {}
    for h in horizons:
        # Keyed by feature set too: see tabicl_baseline's note.
        cache_key = f"ebm/{feature_set}/{event_name}/{h:g}h"
        if cache is not None:
            cached = cache.load_for_feature_set(cache_key, feature_set)
            if cached is not None:
                out[h] = cached
                continue

        y = np.array(
            [outcome_at_horizon(r, times, h) for r in rows],
            dtype=object,
        )
        keep = np.flatnonzero([v is not None for v in y])
        if len(keep) < EBM_MIN_ROWS or len({int(y[i]) for i in keep}) < 2:
            continue
        capped = len(keep) > max_rows
        if capped:
            keep = _grouped_subsample(keep, groups_all[keep], max_rows, rng)
        x_fit = np.array(x_all[keep], dtype=np.float64, copy=True)
        y_fit = y[keep].astype(int)

        logger.info(
            "[ebm] %s@%gh: starting fit, %s features, %d rows%s, outer_bags=%d, n_jobs=%d",
            event_name,
            h,
            feature_set,
            len(keep),
            " (capped)" if capped else "",
            outer_bags,
            n_jobs,
        )
        t0 = time.time()
        if ebm_classifier_cls is None:
            ebm_classifier_cls = _load_ebm_classifier()
        clf = ebm_classifier_cls(
            random_state=seed, n_jobs=n_jobs, outer_bags=outer_bags
        )
        clf.fit(x_fit, y_fit)
        elapsed = time.time() - t0
        out[h] = EBMBaselineModel(
            clf,
            feature_set=feature_set,
            n_features=int(x_all.shape[1]),
            params={
                "n_rows": float(len(keep)),
                "row_capped": float(capped),
                "outer_bags": float(outer_bags),
                "fit_seconds": elapsed,
            },
        )
        if cache is not None:
            cache.save(cache_key, out[h])
        logger.info(
            "[ebm] %s@%gh: done in %.1fs",
            event_name,
            h,
            elapsed,
        )
    return out


def fit_ebm_baselines(
    train_events_binned: pl.DataFrame,
    train_rows: dict[str, list[IndexRow]],
    train_times: dict[str, EventTimes],
    *,
    horizons: Sequence[float] = (8.0, 24.0, 72.0),
    source: str = "mimic_iv",
    seed: int = 0,
    feature_set: str = "strong",
    n_jobs: int = 12,
    max_rows: int | None = None,
    outer_bags: int | None = None,
    cache: Optional[FitCache] = None,
    features: Optional[dict[str, np.ndarray]] = None,
) -> dict[tuple[str, float], EBMBaselineModel]:
    """One ``ExplainableBoostingClassifier`` per (event, horizon).

    Signature deliberately mirrors
    :func:`odyssey.inference.alerts.fit_baselines` and
    :func:`odyssey.inference.tabicl_baseline.fit_tabicl_baselines` (same
    ``train_events_binned``/``train_rows``/``train_times`` shape, same
    ``feature_set``) so all three baselines fit from the same prepared
    data and compare like-for-like.

    ``n_jobs`` is passed straight to ``ExplainableBoostingClassifier``
    (default matches this project's GPU VMs' 12 vCPUs); left unset, EBM's
    own default parallelizes past the actual core count, oversubscribing
    and slowing every fit down rather than speeding it up. ``max_rows``
    and ``outer_bags`` default to the speed-tuned values (see
    :data:`EBM_MAX_ROWS`, :data:`DEFAULT_OUTER_BAGS`); pass EBM's own
    defaults (an uncapped row count, ``outer_bags=14``) for a slower,
    full-quality confirmation run.

    Requires the optional ``interpret`` package (see the module
    docstring); raises ``ImportError`` with install instructions if it is
    not installed, the first time a model would actually be fit. ``cache``,
    if given, is consulted/updated per (event, horizon) -- see
    :mod:`odyssey.inference.fit_cache`.
    """
    resolved_max_rows = EBM_MAX_ROWS if max_rows is None else max_rows
    resolved_outer_bags = DEFAULT_OUTER_BAGS if outer_bags is None else outer_bags
    models: dict[tuple[str, float], EBMBaselineModel] = {}
    if features is None:
        features = features_for_events(
            train_events_binned, train_rows, source=source, feature_set=feature_set
        )
    for name, rows in train_rows.items():
        if not rows:
            continue
        per_horizon = _fit_one_ebm(
            features[name],
            rows,
            train_times[name],
            horizons=horizons,
            feature_set=feature_set,
            seed=seed,
            event_name=name,
            n_jobs=n_jobs,
            max_rows=resolved_max_rows,
            outer_bags=resolved_outer_bags,
            cache=cache,
        )
        for h, model in per_horizon.items():
            models[(name, h)] = model
    return models


__all__ = [
    "DEFAULT_OUTER_BAGS",
    "EBM_MAX_ROWS",
    "EBM_MIN_ROWS",
    "EBMBaselineModel",
    "fit_ebm_baselines",
]
