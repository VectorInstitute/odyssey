"""TabICLv2 in-context-learning baseline: an optional second comparator.

Alongside the tuned ``HistGradientBoostingClassifier`` baseline
(:mod:`odyssey.inference.alerts`), this module fits
`TabICLv2 <https://github.com/soda-inria/tabicl>`_
(Ismail Fawaz et al., ICML 2025/2026), a pretrained tabular foundation
model that classifies via in-context learning: no gradient-descent
training on our data at all, just one forward pass conditioned on the
(subsampled) training rows as context, ``y_pred = model(X_train, y_train,
X_test)``. It is a different *kind* of strong baseline than the GBM: not
tuned per event/horizon, but pretrained once on millions of synthetic
tabular datasets and applied zero-shot.

Optional dependency, not installed by default -- ``uv sync --extra
tabicl`` (or ``pip install tabicl``). Every function here defers the
``tabicl`` import to call time (see :func:`_load_tabicl`), so importing
this module, or importing :mod:`odyssey.inference.alerts` (which does not
import this module), never requires the package.

Known scope mismatch, stated plainly rather than glossed over: TabICLv2
is pretrained on 300 to 48K rows and 2 to 100 columns, with documented
generalization beyond that (the authors report good results to roughly
600K rows and note columns beyond 100 are "observed to generalize" but
not upper-bounded). Our ``strong`` feature set has ~600 columns -- well
past the pretraining range on the column axis specifically. Treat any
TabICL result here as a real empirical data point about how it behaves
outside its documented envelope, not a claim that it is being used as
intended by its authors.

Row count is capped independently (:data:`TABICL_MAX_ROWS`) for a
different reason: unlike a GBM's cheap tree traversal at predict time,
TabICL's in-context forward pass reads the *entire* (sampled) training
set as context on every call and costs roughly ``O(n^2 + n*m^2)`` in the
number of context rows ``n`` and feature columns ``m`` -- so, unlike
:data:`~odyssey.inference.alerts.GBM_FIT_MAX_ROWS`, this cap is not just
a memory-saving convenience but the actual accuracy/cost regime the
authors validated it in.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from odyssey.inference.alerts import (
    EventTimes,
    IndexRow,
    features_for_events,
    outcome_at_horizon,
)


logger = logging.getLogger(__name__)

# Cap on the in-context training set TabICL actually sees, per (event,
# horizon). Well inside the ~100K-row range the authors report strong,
# GPU-timed results for (under 10s for 50K rows / 100 features on an H100);
# generalizes to more with accuracy that may degrade and CPU/disk
# offloading that changes the cost profile entirely, neither of which this
# module opts into. Same seeded-subsample pattern as
# odyssey.inference.alerts.GBM_TUNE_MAX_ROWS, but there is no larger
# "final fit" step to cap separately the way the GBM has
# odyssey.inference.alerts.GBM_FIT_MAX_ROWS: the sampled context *is* the
# model for that (event, horizon), TabICL has no other parameters this
# project trains.
TABICL_MAX_ROWS = 50_000

# Below the authors' documented minimum ("not tested below 300 training
# samples"); an (event, horizon) with fewer at-risk rows than this is
# skipped rather than fit, the same way
# odyssey.inference.alerts._fit_baseline_grid skips a GBM fit under 50
# rows, just at TabICL's own, higher, stated floor.
TABICL_MIN_ROWS = 300


def _load_tabicl_classifier() -> Any:
    """Import and return ``tabicl.TabICLClassifier``, or raise a clear error.

    Deferred so nothing in this module requires ``tabicl`` to be
    installed except the call path that actually fits or predicts with
    it -- mirrors how
    :class:`~odyssey.models.backbones.hybrid.EHRHybridBackbone` defers
    its ``mamba-ssm`` import for the same reason (an optional, heavy,
    platform-specific dependency).
    """
    try:
        from tabicl import TabICLClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TabICL baseline requires the optional `tabicl` package: "
            "`uv sync --extra tabicl` (or `pip install tabicl`). See "
            "odyssey.inference.tabicl_baseline's module docstring for "
            "what it is and its documented scope limits."
        ) from exc
    return TabICLClassifier


@dataclass
class TabICLBaselineModel:
    """A fitted (context-holding) TabICLv2 classifier for one (event, horizon).

    Duck-typed to satisfy :class:`odyssey.inference.alerts._ScoredBaseline`
    (``predict_proba``, ``feature_set``, ``n_features``, ``params``) so it
    drops directly into :func:`~odyssey.inference.alerts.score_alerts`'s
    ``extra_baselines`` hook alongside the GBM baseline, scored the same
    way (AUROC, Brier, calibration).

    Unlike :class:`~odyssey.inference.alerts.BaselineModel`, there is no
    missing-value fill-up front: TabICL's own preprocessing pipeline
    imputes NaN internally (``sklearn.impute.SimpleImputer`` per its
    source), so raw feature matrices -- the same ones
    :func:`~odyssey.inference.alerts.features_for_events` already
    produces for the GBM -- pass straight through.
    """

    clf: object
    """A fitted ``tabicl.TabICLClassifier`` (typed ``object`` so this
    module's public dataclass doesn't force an unconditional import of
    the optional ``tabicl`` package just to reference its type)."""

    feature_set: str = "strong"
    n_features: int = 0
    params: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Positive-class (label ``1``) probabilities, ``(n,)``."""
        proba = self.clf.predict_proba(x)  # type: ignore[attr-defined]
        classes = np.asarray(self.clf.classes_)  # type: ignore[attr-defined]
        pos_idx = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else 1
        result: np.ndarray = proba[:, pos_idx]
        return result


def _fit_one_tabicl(
    x_all: np.ndarray,
    rows: Sequence[IndexRow],
    times: EventTimes,
    *,
    horizons: Sequence[float],
    feature_set: str,
    seed: int,
    event_name: str,
    n_estimators: int,
    device: str | None,
) -> dict[float, TabICLBaselineModel]:
    """Fit one TabICL context per horizon for a single event.

    Structurally mirrors
    :func:`odyssey.inference.alerts._fit_baseline_grid` (same row
    selection, same skip conditions) so the two baselines are trained on
    directly comparable data; the model-specific part -- what "fitting"
    means -- is the only thing that differs, since TabICL's ``fit`` just
    stores the (capped, seeded-subsampled) context rather than running
    gradient descent.
    """
    tabicl_classifier_cls = _load_tabicl_classifier()
    rng = np.random.default_rng(seed)
    out: dict[float, TabICLBaselineModel] = {}
    for h in horizons:
        y = np.array(
            [outcome_at_horizon(r, times, h) for r in rows],
            dtype=object,
        )
        keep = np.flatnonzero([v is not None for v in y])
        if len(keep) < TABICL_MIN_ROWS or len({int(y[i]) for i in keep}) < 2:
            continue
        if len(keep) > TABICL_MAX_ROWS:
            keep = rng.choice(keep, TABICL_MAX_ROWS, replace=False)
        x_fit = np.array(x_all[keep], dtype=np.float32, copy=True)
        y_fit = y[keep].astype(int)

        clf = tabicl_classifier_cls(
            n_estimators=n_estimators,
            device=device,
            random_state=seed,
        )
        clf.fit(x_fit, y_fit)
        out[h] = TabICLBaselineModel(
            clf,
            feature_set=feature_set,
            n_features=int(x_all.shape[1]),
            params={
                "n_context_rows": float(len(keep)),
                "n_estimators": float(n_estimators),
            },
        )
        logger.info(
            "[tabicl] %s@%gh: %s features, %d context rows, n_estimators=%d",
            event_name,
            h,
            feature_set,
            len(keep),
            n_estimators,
        )
    return out


def fit_tabicl_baselines(
    train_events_binned: pl.DataFrame,
    train_rows: dict[str, list[IndexRow]],
    train_times: dict[str, EventTimes],
    *,
    horizons: Sequence[float] = (8.0, 24.0, 72.0),
    source: str = "mimic_iv",
    seed: int = 0,
    feature_set: str = "strong",
    n_estimators: int = 8,
    device: str | None = None,
) -> dict[tuple[str, float], TabICLBaselineModel]:
    """One TabICLv2 context per (event, horizon), on the same features as the GBM.

    Signature deliberately mirrors
    :func:`odyssey.inference.alerts.fit_baselines` (same
    ``train_events_binned``/``train_rows``/``train_times`` shape, same
    ``feature_set``) so the two can be fit from the same prepared data and
    compared like-for-like. ``n_estimators`` is TabICL's own ensemble-size
    knob (default matches ``TabICLClassifier``'s own default of 8): more
    members trade inference cost for averaging noise down, no training
    cost either way since none of them are gradient-trained.

    Requires the optional ``tabicl`` package
    (see the module docstring); raises ``ImportError`` with install
    instructions if it is not installed, the first time a context would
    actually be fit -- not merely on import of this module.
    """
    models: dict[tuple[str, float], TabICLBaselineModel] = {}
    features = features_for_events(
        train_events_binned, train_rows, source=source, feature_set=feature_set
    )
    for name, rows in train_rows.items():
        if not rows:
            continue
        per_horizon = _fit_one_tabicl(
            features[name],
            rows,
            train_times[name],
            horizons=horizons,
            feature_set=feature_set,
            seed=seed,
            event_name=name,
            n_estimators=n_estimators,
            device=device,
        )
        for h, model in per_horizon.items():
            models[(name, h)] = model
    return models


__all__ = [
    "TABICL_MAX_ROWS",
    "TABICL_MIN_ROWS",
    "TabICLBaselineModel",
    "fit_tabicl_baselines",
]
