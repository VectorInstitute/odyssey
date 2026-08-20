"""Three-way error analysis: hazard heads vs. tuned GBM vs. TabICL.

Two questions this module exists to answer, given a dumped
:func:`~odyssey.inference.alerts.index_row_table` (optionally extended
with a TabICL column via that function's ``extra_baselines``, see
:mod:`odyssey.inference.tabicl_baseline`):

1. **Which scorer wins where, overall and stratified?**
   :func:`scorer_auroc_table` computes AUROC per ``(event, horizon,
   scorer)``, optionally grouped by a stratifying condition (any boolean
   Polars expression over the dumped table's ``ctx.*`` columns) --
   exactly odyssey-6e's entry 22 methodology
   (``research_journal/experiments/22_eicu_alerts_error_analysis.html``),
   generalized from a one-off script to reusable, tested library code so
   a third scorer participates for free. ``ctx.hours_into_visit`` is the
   direct proxy for "how much sequence has the model actually seen by
   this point" -- stratifying on it is how to check whether the hazard
   head's relative advantage grows with sequence length, the question
   this module was built to make answerable, not just askable.
2. **Which scorer is more interpretable?** Not a number to compute --
   :data:`INTERPRETABILITY_COMPARISON` is a small, explicit, factual
   comparison table instead of a metric, because "more interpretable"
   collapses three genuinely different capabilities (causal-style
   test-time intervention, calibrated time-to-event output, post-hoc
   attribution) that a single score would hide.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl


HORIZONS_HOURS: tuple[float, ...] = (8.0, 24.0, 72.0)


@dataclass(frozen=True)
class ScorerAUROC:
    """One (event, horizon, scorer[, stratum]) AUROC cell."""

    event: str
    horizon_hours: float
    scorer: str
    stratum: str
    n: int
    n_positive: int
    auroc: float | None
    """None when the stratum has too few rows or only one outcome class
    present (AUROC undefined) -- reported explicitly rather than
    silently dropped, so a caller can tell "not computable" apart from
    "not present in the table at all"."""


def _scorer_column_prefixes(columns: Sequence[str]) -> dict[str, str]:
    """Map scorer name -> its horizon-column prefix, from a dumped table's columns.

    ``hazard@8h`` -> ``{"hazard": "hazard"}``, ``gbm@8h`` -> ``{"gbm":
    "gbm"}``, ``tabicl@8h`` -> ``{"tabicl": "tabicl"}``, and so on for any
    future baseline family: driven entirely by whatever ``@{h}h`` columns
    are actually present, so a new baseline needs no change here, only a
    new column in the dumped table (see
    :func:`~odyssey.inference.alerts.index_row_table`'s ``extra_baselines``).
    """
    prefixes: dict[str, str] = {}
    for col in columns:
        if "@" not in col or not col.endswith("h"):
            continue
        prefix = col.split("@", 1)[0]
        if prefix == "y":
            continue  # the outcome column, y@{h}h -- not a scorer
        prefixes[prefix] = prefix
    return prefixes


def scorer_auroc_table(
    dumped: pl.DataFrame,
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    strata: dict[str, pl.Expr] | None = None,
    min_rows: int = 50,
) -> list[ScorerAUROC]:
    """AUROC per (event, horizon, scorer), optionally split by ``strata``.

    ``dumped`` is a table from
    :func:`~odyssey.inference.alerts.index_row_table` (one row per index
    time, ``y@{h}h`` outcome columns, one score column per scorer per
    horizon). ``strata`` maps a stratum name to a boolean Polars
    expression evaluated against ``dumped`` (e.g. ``{"long_sequence":
    pl.col("ctx.hours_into_visit") >= 72}``); omitted, or in addition,
    an ``"all"`` stratum (the whole table, no condition) is always
    included so the unstratified comparison and any stratified cut are
    both available from one call.

    A cell with fewer than ``min_rows`` non-null (score, outcome) pairs,
    or only one outcome class present, gets ``auroc=None`` rather than
    being silently omitted or raising -- the same "reported, not hidden"
    principle :func:`~odyssey.inference.alerts.score_alerts` uses for
    censoring.
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    prefixes = _scorer_column_prefixes(dumped.columns)
    all_strata: dict[str, pl.Expr | None] = {"all": None}
    all_strata.update(strata or {})

    results: list[ScorerAUROC] = []
    for event in sorted(dumped["event"].unique().to_list()):
        event_frame = dumped.filter(pl.col("event") == event)
        for h in horizons:
            y_col = f"y@{h:g}h"
            if y_col not in event_frame.columns:
                continue
            for scorer, prefix in sorted(prefixes.items()):
                score_col = f"{prefix}@{h:g}h"
                if score_col not in event_frame.columns:
                    continue
                for stratum_name, condition in all_strata.items():
                    stratum_frame = (
                        event_frame
                        if condition is None
                        else event_frame.filter(condition)
                    )
                    valid = stratum_frame.filter(
                        pl.col(y_col).is_not_null() & pl.col(score_col).is_not_null()
                    )
                    n = valid.height
                    y = valid[y_col].to_numpy()
                    n_positive = int(y.sum()) if n else 0
                    auroc = None
                    if n >= min_rows and 0 < n_positive < n:
                        auroc = float(roc_auc_score(y, valid[score_col].to_numpy()))
                    results.append(
                        ScorerAUROC(
                            event=event,
                            horizon_hours=h,
                            scorer=scorer,
                            stratum=stratum_name,
                            n=n,
                            n_positive=n_positive,
                            auroc=auroc,
                        )
                    )
    return results


def best_scorer_per_cell(
    rows: Sequence[ScorerAUROC], *, stratum: str = "all"
) -> dict[tuple[str, float], str]:
    """``(event, horizon) -> name of the scorer with the highest AUROC``.

    Restricted to one ``stratum`` at a time (default ``"all"``, the
    unstratified comparison) since comparing across strata isn't
    meaningful -- answers question 1's "which model is best overall"
    directly; call once per stratum (including a stratified one from
    :func:`scorer_auroc_table`) to answer "and where does that change".
    Cells where every scorer's AUROC is ``None`` are omitted.
    """
    by_cell: dict[tuple[str, float], list[ScorerAUROC]] = {}
    for row in rows:
        if row.stratum != stratum:
            continue
        by_cell.setdefault((row.event, row.horizon_hours), []).append(row)
    best: dict[tuple[str, float], str] = {}
    for cell, cell_rows in by_cell.items():
        scored = [r for r in cell_rows if r.auroc is not None]
        if scored:
            best[cell] = max(scored, key=lambda r: r.auroc or 0.0).scorer
    return best


@dataclass(frozen=True)
class InterpretabilityRow:
    """One capability, compared across scorers -- a fact, not a score."""

    capability: str
    hazard_head: str
    gbm: str
    tabicl: str
    note: str


INTERPRETABILITY_COMPARISON: tuple[InterpretabilityRow, ...] = (
    InterpretabilityRow(
        capability="Test-time causal-style intervention",
        hazard_head="Yes (BottleneckIntervention: force a concept to a "
        "value, zero a channel; see odyssey.models.concept_bottleneck)",
        gbm="No (a fitted tree ensemble has no concept layer to "
        "intervene on; SHAP explains a prediction, it cannot edit one)",
        tabicl="No (in-context learning has no analogous concept layer "
        "either; nothing here to intervene on)",
        note="This is the one capability specific to the concept "
        "bottleneck architecture; the point of this whole project's "
        "leakage investigation (research_journal/experiments/"
        "23_concept_lever_leakage_investigation.html) is that HAVING "
        "the mechanism and it working causally are separate claims, "
        "verified separately, not implied by each other.",
    ),
    InterpretabilityRow(
        capability="Calibrated time-to-event output",
        hazard_head="Yes, first-class: EventHazardHeads produce a full "
        "discrete survival curve per event, not just a fixed-horizon "
        "probability",
        gbm="Partial: one binary classifier per (event, horizon) -- a "
        "calibrated P(event within h) at the horizons it was fit for, "
        "no curve between them",
        tabicl="Partial, same shape as the GBM: one in-context fit per "
        "(event, horizon), no continuous survival curve",
        note="Both baselines answer 'will this happen within 8h/24h/72h', "
        "not 'when'; only the hazard head answers the second question.",
    ),
    InterpretabilityRow(
        capability="Post-hoc feature attribution",
        hazard_head="Not built in (would need a separate attribution "
        "method over the bottleneck output, not implemented here)",
        gbm="Yes, standard: SHAP/feature importances work directly on "
        "a HistGradientBoostingClassifier",
        tabicl="Yes, via the optional tabicl[shap] extra -- a real "
        "capability, not absent, just not installed by default here "
        "(this project's tabicl extra does not pull it in; see "
        "odyssey.inference.tabicl_baseline's module docstring)",
        note="Attribution answers 'what did the model weigh', a "
        "different and weaker question than intervention's 'what would "
        "change the answer' -- both baselines have an attribution story, "
        "neither has a causal-intervention one.",
    ),
)


__all__ = [
    "HORIZONS_HOURS",
    "INTERPRETABILITY_COMPARISON",
    "InterpretabilityRow",
    "ScorerAUROC",
    "best_scorer_per_cell",
    "scorer_auroc_table",
]
