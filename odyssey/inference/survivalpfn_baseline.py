"""SurvivalPFN in-context survival baseline: a third, survival-native comparator.

Alongside the tuned ``HistGradientBoostingClassifier`` baseline
(:mod:`odyssey.inference.alerts`) and the in-context tabular classifiers
(:mod:`odyssey.inference.tabicl_baseline`,
:mod:`odyssey.inference.ebm_baseline`), this module fits
`SurvivalPFN <https://github.com/rgklab/SurvivalPFN>`_ (Qi, Balazadeh,
Cooper, Greiner, Krishnan, "Amortizing Survival Prediction via In-Context
Bayesian Inference", arXiv:2605.15488, 2026), a prior-data fitted network
for right-censored survival analysis: no gradient-descent training on our
data, one forward pass conditioned on context rows, ``SurvivalEstimator
.fit(X, delta, T).predict_event_distribution(X_query)``.

This is the scientifically interesting property of this baseline, not
just another classifier to compare against: it is fit survival-natively,
the same way the project's own per-event hazard heads are trained --
``T`` is hours from the index time to the event's onset (observed) or to
the end of observed follow-up (censored), ``delta`` marks which -- not a
per-horizon binary "did it happen by h" outcome the way the GBM/TabICL/
EBM baselines are fit. One :class:`SurvivalEstimator` context per *event*
(not per event-horizon pair) then answers every horizon via its own
``HistogramDistribution.survival_at(h)`` method, evaluated at query time:
``P(event within h) = 1 - survival_at(h)``. Evaluation stays comparable to
every other baseline in :func:`~odyssey.inference.alerts.score_alerts`
regardless -- the same per-horizon at-risk/censoring keep-masks are
applied to every scorer's predictions identically there, this module only
changes what "fitting" means, not what gets scored.

Optional dependency, not on PyPI -- installed from git
(``uv sync --extra survivalpfn``, see ``pyproject.toml``). Every function
here defers the ``survivalpfn`` import to call time (see
:func:`_load_survival_estimator`), mirroring
:mod:`odyssey.inference.tabicl_baseline`'s deferred-import pattern, so
importing this module never requires the package until a context is
actually fit. Their own ``pyproject.toml`` has a packaging bug worth
knowing about: ``omegaconf`` is a hard runtime import
(``survivalpfn/models/loading.py``) but only declared under their
``train`` optional-extra, not their base dependencies -- our own
``survivalpfn`` extra pins it explicitly alongside the git dependency to
route around that, not because we need their training/config machinery.

Hard architectural limit, confirmed by loading the released checkpoint
(``survivalpfn_v0.1.pt``, ``shi-ang/SurvivalPFN``) and reading its
``model_config``: ``max_num_features: 100``. Unlike TabICL's documented-
but-not-upper-bounded column range, this is a fixed input width the
pretrained transformer was built for (``pad_x`` pads narrower inputs up
to it; there is no path for wider ones). This project's ``strong``
feature set (~609 columns) is well over that cap and is not supported by
this module; ``basic`` (16 columns) is. :func:`fit_survivalpfn_baselines`
defaults to ``feature_set="basic"`` for this reason (the other two
baseline families default to ``"strong"``) and :func:`_fit_one_survivalpfn`
raises a clear error rather than let an oversized matrix fail deep inside
the model's own padding code if a caller passes ``"strong"`` anyway.

No documented row/context-size envelope (unlike TabICL's stated ~50K-row
regime): the paper's released benchmark datasets are all "hundreds to low
thousands of rows" per the repo's own README, and the model is a
transformer whose context cost scales with row count, but no explicit
cap is stated anywhere checked. :data:`SURVIVALPFN_MAX_ROWS` starts at
the released-benchmark scale rather than assuming TabICL's larger,
authors-validated regime transfers; raise it only after a direct
wall-clock/memory measurement at the higher row count, the same
discipline entry 30's EBM speed-tuning used.

NaN handling: confirmed empirically (not documented) that ``fit``/
``predict_event_distribution`` accept NaN-containing input, including an
entirely-NaN column, and return finite, non-NaN output -- no
TabICL-style all-NaN-column workaround is needed here.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from odyssey.inference.alerts import EventTimes, IndexRow, features_for_events
from odyssey.inference.fit_cache import FitCache


logger = logging.getLogger(__name__)

# See the module docstring: a hard input-width limit read from the
# released checkpoint's own model_config, not a soft/documented range.
SURVIVALPFN_MAX_FEATURES = 100

# Conservative starting cap on the in-context set SurvivalPFN sees per
# event (not per event-horizon pair -- one context serves every horizon).
# See the module docstring for why this starts at the released-benchmark
# scale rather than TabICL's larger, authors-validated row regime.
SURVIVALPFN_MAX_ROWS = 5_000

# Floor matching the GBM's/EBM's own minimum; the authors do not state a
# minimum context size the way TabICL's "not tested below 300" does.
SURVIVALPFN_MIN_ROWS = 50

# Query-side batch size for predict_proba. SurvivalPFN shares TabICL's
# in-context-transformer shape and has the identical O(context/query)
# blowup with no CPU/disk offload knob at all -- a real MIMIC rescore hit
# this exact wall (~216GB attempted for one unbatched call on 552,000
# query rows, immediate OOM, no partial results) via TabICL first, and
# this module has the same structural exposure. Chunking the query
# dimension changes only how many rows are scored per forward pass, not
# any individual row's prediction: the fitted context is fixed, and each
# query row is scored against it independently of whatever other query
# rows share its batch.
_PREDICT_BATCH_SIZE = 8192


def _load_survival_estimator() -> Any:
    """Import and return ``survivalpfn.SurvivalEstimator``, or raise a clear error.

    Deferred so nothing in this module requires ``survivalpfn`` to be
    installed except the call path that actually fits or predicts with
    it -- mirrors :func:`odyssey.inference.tabicl_baseline._load_tabicl_classifier`.
    """
    try:
        from survivalpfn import SurvivalEstimator  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "SurvivalPFN baseline requires the optional `survivalpfn` "
            "package: `uv sync --extra survivalpfn` (a git dependency, "
            "not on PyPI -- see pyproject.toml). See "
            "odyssey.inference.survivalpfn_baseline's module docstring "
            "for what it is and its documented scope limits."
        ) from exc
    return SurvivalEstimator


def _grouped_subsample(
    keep: np.ndarray, groups: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Shrink ``keep`` to at most ``cap`` rows, dropping whole subjects only.

    Same subject-grouped-cap pattern as
    :func:`odyssey.inference.ebm_baseline._grouped_subsample`, duplicated
    here rather than imported across modules (each baseline module is
    self-contained, matching how :mod:`tabicl_baseline` also does not
    import from :mod:`ebm_baseline`).
    """
    subjects, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(subjects))
    cum = np.cumsum(counts[order])
    n_subjects = max(1, int(np.searchsorted(cum, cap, side="right")))
    n_subjects = min(n_subjects, len(subjects))
    selected = set(subjects[order[:n_subjects]].tolist())
    mask = np.array([g in selected for g in groups])
    result: np.ndarray = keep[mask]
    return result


class _CapAtMaxHorizon:
    """Sentinel: cap follow-up at the largest scored horizon (the default)."""


_CAP_AT_MAX_HORIZON: Any = _CapAtMaxHorizon()


def _survival_targets(
    rows: Sequence[IndexRow],
    times: EventTimes,
    followup_cap_hours: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row ``(T, delta, keep_idx)`` for a survival-native fit.

    ``T`` is hours from the index time to the event's onset (observed) or
    to the end of observed follow-up (censored); ``delta`` is 1 for
    observed, 0 for censored. Rows where the event has already happened
    by the index time are excluded (not at risk) -- the identical rule
    and identical key construction
    :func:`odyssey.inference.alerts.outcome_at_horizon` uses, so the same
    rows are "at risk" here as for every horizon-binned baseline; this
    just does not further split them by horizon.

    ``followup_cap_hours`` administratively censors follow-up at that
    horizon: any row whose event or censoring time lies beyond it becomes
    censored AT the cap. This is not a convenience -- without it, a
    subject-scoped event's fit is on a different question than the one the
    alerts are scored on. Measured on MIMIC-IV (2026-08-23): death's
    time-to-event has a median of 6,915 h and a maximum of 131,272 h,
    with only 3.3% of events inside 72 h, so an uncapped fit puts the
    entire 8/24/72 h alert window at the very start of the survival curve
    and ``1 - S(h)`` came back as a literal constant 0.0 at every horizon
    (a degenerate column, AUROC exactly 0.5). The visit-scoped events, whose
    follow-up is a stay rather than a record, have medians of 60-100 h and
    were unaffected -- as were the eICU fits, whose records are ICU stays.
    Capping at the largest scored horizon asks the survival model the same
    question the horizon-binned baselines answer.
    """
    t_list: list[float] = []
    delta_list: list[float] = []
    keep: list[int] = []
    for i, r in enumerate(rows):
        key = (r.subject_id, -1 if times.subject_scoped else r.visit_id)
        onset = times.onset.get(key)
        if onset is not None and onset <= r.time_hours:
            continue  # already happened: not at risk
        if onset is not None:
            gap, observed = onset - r.time_hours, 1.0
        else:
            censor = times.censor.get(key)
            if censor is None or censor <= r.time_hours:
                continue  # no observed follow-up past the index time
            gap, observed = censor - r.time_hours, 0.0
        if followup_cap_hours is not None and gap > followup_cap_hours:
            gap, observed = followup_cap_hours, 0.0  # administratively censored
        t_list.append(gap)
        delta_list.append(observed)
        keep.append(i)
    return (
        np.array(t_list, dtype=np.float32),
        np.array(delta_list, dtype=np.float32),
        np.array(keep, dtype=np.int64),
    )


@dataclass
class SurvivalPFNBaselineModel:
    """A fitted SurvivalPFN context, evaluated at one horizon.

    Duck-typed to satisfy :class:`odyssey.inference.alerts._ScoredBaseline`
    (``predict_proba``, ``feature_set``, ``n_features``, ``params``) so it
    drops directly into :func:`~odyssey.inference.alerts.score_alerts`'s
    ``extra_baselines`` hook. Unlike :class:`TabICLBaselineModel`/
    ``EBMBaselineModel``, several instances of this class -- one per
    horizon -- share the same fitted ``estimator``: the context is fit
    once per event (survival-native, see the module docstring), and
    ``horizon_hours`` only selects which query time this instance's
    ``predict_proba`` evaluates ``survival_at`` at.
    """

    estimator: object
    """A fitted ``survivalpfn.SurvivalEstimator`` (typed ``object`` for the
    same reason ``TabICLBaselineModel.clf`` is: avoids an unconditional
    import of the optional ``survivalpfn`` package just to reference its
    type)."""

    horizon_hours: float
    feature_set: str = "basic"
    n_features: int = 0
    params: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """``P(event within horizon_hours)``, ``(n,)`` -- ``1 - survival_at(h)``.

        Batched over the query dimension in chunks of
        :data:`_PREDICT_BATCH_SIZE` -- see that constant's comment for why
        (a real, confirmed OOM crash on this exact model shape via
        TabICL's identical exposure, not a hypothetical). Asserts the
        batched output length matches the input length.
        """
        import torch  # noqa: PLC0415

        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        chunks = []
        for start in range(0, x.shape[0], _PREDICT_BATCH_SIZE):
            chunk = x[start : start + _PREDICT_BATCH_SIZE]
            dist = self.estimator.predict_event_distribution(chunk)  # type: ignore[attr-defined]
            h = torch.full(
                (chunk.shape[0],), float(self.horizon_hours), dtype=torch.float32
            )
            # float64 before subtracting: float32's eps is ~1.2e-7, so for a
            # rare event at a short horizon -- where survival genuinely is
            # ~1 -- ``1 - S(h)`` rounds to exactly 0 for every row and the
            # column comes out a constant (measured 2026-08-23: death@8h and
            # @24h, and vasopressor@8h/@24h down to 175 distinct values at
            # magnitudes ~1e-4). Widening to float64 buys ~1e-16 of
            # resolution. If the estimator's own survival_at saturates to
            # exactly 1.0 internally, no cast here can recover it -- that is
            # what the degeneracy check below detects and says out loud,
            # rather than letting a constant column reach a results table.
            survival = dist.survival_at(h).double()
            chunks.append((1.0 - survival).detach().cpu().numpy())
        result: np.ndarray = np.concatenate(chunks)
        if result.size and float(result.max()) <= 0.0:
            logger.warning(
                "[survivalpfn] horizon %.0fh: every predicted probability is 0 -- "
                "the estimator's own survival curve saturates at 1 for this "
                "(event, horizon), so 1 - S(h) carries no information even in "
                "float64. This column is a numerical artifact, NOT a "
                "measurement; do not report it as AUROC 0.5.",
                self.horizon_hours,
            )
        elif result.size:
            # Grade two. A column can pass the all-zero check and still be
            # resolution-limited rather than measured: vasopressor@8h/@24h
            # came back with 175 and 242 distinct probabilities across
            # 111,450 and 98,722 rows at magnitudes ~1e-4, giving AUROCs of
            # 0.522 and 0.537 that are largely tie-breaking among a handful
            # of levels. Those passed the constant-column rule and were
            # nearly written into the comparator table as measurements
            # (2026-08-24). The healthy cells are two to three orders of
            # magnitude better resolved on the same run (AKI and ICU
            # admission at 0.8-0.97 distinct values per row, vasopressor@72h
            # at 0.17), so the threshold below separates them cleanly rather
            # than being tuned to a boundary.
            distinct = int(np.unique(result).size)
            if distinct < _MIN_DISTINCT_FRACTION * result.size:
                logger.warning(
                    "[survivalpfn] horizon %.0fh: only %d distinct predicted "
                    "probabilities across %d rows (max %.3g). This column is "
                    "resolution-limited, not constant: it will produce a "
                    "near-chance AUROC driven largely by ties. Report it only "
                    "with the distinct-value count attached; do not quote it "
                    "as a measurement of discrimination.",
                    self.horizon_hours,
                    distinct,
                    result.size,
                    float(result.max()),
                )
        if len(result) != x.shape[0]:
            raise AssertionError(
                f"batched predict_proba returned {len(result)} rows for "
                f"{x.shape[0]} input rows -- a row was silently dropped or "
                "duplicated across batch chunks"
            )
        return result


#: Below this many distinct predicted probabilities per row, a column is
#: resolution-limited rather than measured, and its AUROC is mostly ties.
#: Set from measured separation, not taste: the degenerate cells sat at
#: 0.0016-0.0025 distinct values per row and the healthy ones at 0.17-0.97,
#: two orders of magnitude apart, so anything in that gap works.
_MIN_DISTINCT_FRACTION = 0.01


def _fit_one_survivalpfn(
    x_all: np.ndarray,
    rows: Sequence[IndexRow],
    times: EventTimes,
    *,
    horizons: Sequence[float],
    feature_set: str,
    seed: int,
    event_name: str,
    device: str,
    max_rows: int,
    followup_cap_hours: Optional[float],
    cache: Optional[FitCache] = None,
) -> dict[float, SurvivalPFNBaselineModel]:
    """Fit one SurvivalPFN context for a single event, shared across every horizon.

    Structurally mirrors
    :func:`odyssey.inference.tabicl_baseline._fit_one_tabicl` (row
    selection, row cap, logging), except the fit target is
    :func:`_survival_targets`'s ``(T, delta)`` pair rather than a
    per-horizon binary outcome, and one fit produces every horizon's
    wrapper at once instead of looping ``horizons`` to fit separately.

    ``cache``, if given, is checked/updated per event (this baseline's
    natural fit granularity -- one context serves every horizon) --
    caches the fitted estimator plus its fit metadata, then rebuilds the
    per-horizon wrappers below on every call (cheap, no re-fit) so a
    cache hit still honors whatever ``horizons`` the caller asks for. See
    :mod:`odyssey.inference.fit_cache`.
    """
    if x_all.shape[1] > SURVIVALPFN_MAX_FEATURES:
        raise ValueError(
            f"SurvivalPFN's checkpoint has a hard max_num_features="
            f"{SURVIVALPFN_MAX_FEATURES} (see the module docstring); "
            f"got {x_all.shape[1]} columns (feature_set={feature_set!r}). "
            "Use feature_set='basic' (16 columns), not 'strong'."
        )
    # The cap is part of the key: a fit made against a different follow-up
    # window answers a different question (see _survival_targets).
    cap_tag = (
        "uncapped" if followup_cap_hours is None else f"cap{followup_cap_hours:g}h"
    )
    cache_key = f"survivalpfn/{cap_tag}/{event_name}"
    cached = cache.load(cache_key) if cache is not None else None
    if cached is not None:
        estimator, n_context_rows, elapsed, row_capped = cached
    else:
        t_all, delta_all, keep_idx = _survival_targets(rows, times, followup_cap_hours)
        if (
            len(keep_idx) < SURVIVALPFN_MIN_ROWS
            or delta_all.sum() < 1
            or (delta_all == 0).sum() < 1
        ):
            logger.info(
                "[survivalpfn] %s: skipped, %d at-risk rows (need >= %d, "
                "both event and censored rows present)",
                event_name,
                len(keep_idx),
                SURVIVALPFN_MIN_ROWS,
            )
            return {}
        groups = np.array([rows[i].subject_id for i in keep_idx])
        rng = np.random.default_rng(seed)
        n_at_risk = len(keep_idx)
        if n_at_risk > max_rows:
            sub = _grouped_subsample(np.arange(n_at_risk), groups, max_rows, rng)
            keep_idx = keep_idx[sub]
            t_all = t_all[sub]
            delta_all = delta_all[sub]
        x_fit = np.array(x_all[keep_idx], dtype=np.float32, copy=False)

        estimator_cls = _load_survival_estimator()
        estimator = estimator_cls(device=device)
        t0 = time.time()
        estimator.fit(X=x_fit, delta=delta_all, T=t_all)
        elapsed = time.time() - t0
        n_context_rows = len(keep_idx)
        row_capped = len(keep_idx) < n_at_risk
        logger.info(
            "[survivalpfn] %s: fit on %d rows (%.1f%% event, %.1f%% censored) "
            "in %.1fs, serves %d horizons, follow-up %s",
            event_name,
            n_context_rows,
            100 * float(delta_all.mean()),
            100 * (1 - float(delta_all.mean())),
            elapsed,
            len(horizons),
            cap_tag,
        )
        if cache is not None:
            cache.save(cache_key, (estimator, n_context_rows, elapsed, row_capped))

    out: dict[float, SurvivalPFNBaselineModel] = {}
    for h in horizons:
        out[h] = SurvivalPFNBaselineModel(
            estimator,
            horizon_hours=h,
            feature_set=feature_set,
            n_features=int(x_all.shape[1]),
            params={
                "n_context_rows": float(n_context_rows),
                "fit_seconds": elapsed,
                "row_capped": float(row_capped),
            },
        )
    return out


def fit_survivalpfn_baselines(
    train_events_binned: pl.DataFrame,
    train_rows: dict[str, list[IndexRow]],
    train_times: dict[str, EventTimes],
    *,
    horizons: Sequence[float] = (8.0, 24.0, 72.0),
    source: str = "mimic_iv",
    seed: int = 0,
    feature_set: str = "basic",
    device: str = "cpu",
    max_rows: int | None = None,
    followup_cap_hours: Optional[float] = _CAP_AT_MAX_HORIZON,
    cache: Optional[FitCache] = None,
    features: Optional[dict[str, np.ndarray]] = None,
) -> dict[tuple[str, float], SurvivalPFNBaselineModel]:
    """One SurvivalPFN context per event, evaluated at every horizon.

    Signature deliberately mirrors
    :func:`odyssey.inference.tabicl_baseline.fit_tabicl_baselines`/
    :func:`odyssey.inference.ebm_baseline.fit_ebm_baselines` (same
    ``train_events_binned``/``train_rows``/``train_times`` shape) so all
    three can be fit from the same prepared data and compared like-for-
    like, but returns one fitted context reused across every ``(event,
    h)`` key for a given event rather than one context per key -- see the
    module docstring for why.

    ``followup_cap_hours`` administratively censors every fit's follow-up
    at that horizon; the default caps at ``max(horizons)`` so the survival
    fit answers the same question the horizon-binned baselines do. Pass
    ``None`` for the uncapped fit, which produced a degenerate (constant)
    death column on MIMIC-IV -- see :func:`_survival_targets`.

    Requires the optional ``survivalpfn`` package (see the module
    docstring); raises ``ImportError`` with install instructions if it is
    not installed, the first time a context would actually be fit -- not
    merely on import of this module. ``cache``, if given, is
    consulted/updated per event -- see :mod:`odyssey.inference.fit_cache`.
    ``features``, if given, is the precomputed per-event feature dict for
    this function's ``feature_set`` (see
    :func:`odyssey.inference.baseline_prep.prepare_baseline_data`);
    ``train_events_binned`` is then unused and may be empty.
    """
    resolved_max_rows = SURVIVALPFN_MAX_ROWS if max_rows is None else max_rows
    models: dict[tuple[str, float], SurvivalPFNBaselineModel] = {}
    resolved_cap = (
        max(horizons)
        if followup_cap_hours is _CAP_AT_MAX_HORIZON
        else followup_cap_hours
    )
    if features is None:
        features = features_for_events(
            train_events_binned, train_rows, source=source, feature_set=feature_set
        )
    for name, rows in train_rows.items():
        if not rows:
            continue
        per_horizon = _fit_one_survivalpfn(
            features[name],
            rows,
            train_times[name],
            horizons=horizons,
            feature_set=feature_set,
            seed=seed,
            event_name=name,
            device=device,
            max_rows=resolved_max_rows,
            followup_cap_hours=resolved_cap,
            cache=cache,
        )
        for h, model in per_horizon.items():
            models[(name, h)] = model
    return models


__all__ = [
    "SURVIVALPFN_MAX_FEATURES",
    "SURVIVALPFN_MAX_ROWS",
    "SURVIVALPFN_MIN_ROWS",
    "SurvivalPFNBaselineModel",
    "fit_survivalpfn_baselines",
]
