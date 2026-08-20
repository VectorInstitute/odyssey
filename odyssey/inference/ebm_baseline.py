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

# Below this many at-risk rows, a GA2M fit has too little signal to be
# meaningful; mirrors odyssey.inference.alerts._fit_baseline_grid's own
# floor (50) for the GBM baseline, same reasoning.
EBM_MIN_ROWS = 50


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
        classes = np.asarray(self.clf.classes_)  # type: ignore[attr-defined]
        pos_idx = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else 1
        result: np.ndarray = proba[:, pos_idx]
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
) -> dict[float, EBMBaselineModel]:
    """Fit one EBM per horizon for a single event, given a feature matrix.

    Structurally mirrors :func:`odyssey.inference.alerts._fit_baseline_grid`
    (same row selection, same skip condition) so the GBM, TabICL, and EBM
    baselines are all trained on directly comparable data. No row cap:
    EBM's own documentation reports fitting datasets with 100 million
    samples in several hours, CPU-only, well past anything this project's
    landmark tables reach.
    """
    ebm_classifier_cls = _load_ebm_classifier()
    out: dict[float, EBMBaselineModel] = {}
    for h in horizons:
        y = np.array(
            [outcome_at_horizon(r, times, h) for r in rows],
            dtype=object,
        )
        keep = np.flatnonzero([v is not None for v in y])
        if len(keep) < EBM_MIN_ROWS or len({int(y[i]) for i in keep}) < 2:
            continue
        x_fit = np.array(x_all[keep], dtype=np.float64, copy=True)
        y_fit = y[keep].astype(int)

        clf = ebm_classifier_cls(random_state=seed)
        clf.fit(x_fit, y_fit)
        out[h] = EBMBaselineModel(
            clf,
            feature_set=feature_set,
            n_features=int(x_all.shape[1]),
            params={"n_rows": float(len(keep))},
        )
        logger.info(
            "[ebm] %s@%gh: %s features, %d rows",
            event_name,
            h,
            feature_set,
            len(keep),
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
) -> dict[tuple[str, float], EBMBaselineModel]:
    """One ``ExplainableBoostingClassifier`` per (event, horizon).

    Signature deliberately mirrors
    :func:`odyssey.inference.alerts.fit_baselines` and
    :func:`odyssey.inference.tabicl_baseline.fit_tabicl_baselines` (same
    ``train_events_binned``/``train_rows``/``train_times`` shape, same
    ``feature_set``) so all three baselines fit from the same prepared
    data and compare like-for-like.

    Requires the optional ``interpret`` package (see the module
    docstring); raises ``ImportError`` with install instructions if it is
    not installed, the first time a model would actually be fit.
    """
    models: dict[tuple[str, float], EBMBaselineModel] = {}
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
        )
        for h, model in per_horizon.items():
            models[(name, h)] = model
    return models


__all__ = [
    "EBM_MIN_ROWS",
    "EBMBaselineModel",
    "fit_ebm_baselines",
]
