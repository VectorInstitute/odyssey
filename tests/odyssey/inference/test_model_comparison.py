"""Tests for the three-way (hazard head / GBM / TabICL) error-analysis comparison."""

import polars as pl
import pytest

from odyssey.inference.model_comparison import (
    INTERPRETABILITY_COMPARISON,
    best_scorer_per_cell,
    scorer_auroc_table,
)


def _dumped_table() -> pl.DataFrame:
    """Build a small, hand-built stand-in for alerts.index_row_table's output.

    hazard is a strictly better predictor of ``y`` than gbm everywhere
    (score equals ``y`` itself vs. a noisier, weaker correlate), so the
    unstratified ("all") comparison has an unambiguous winner: exactly
    what the overall-AUROC tests below check, without needing to reason
    about how per-stratum AUROCs would pool together.
    """
    n = 200
    y = [1 if (i % 3 == 0) else 0 for i in range(n)]
    hazard = [0.95 if v == 1 else 0.05 for v in y]
    # gbm: right direction, but with 1 in 4 positions flipped -- weaker,
    # not perfect, and not a monotonic function of hazard's own score.
    gbm = [
        (0.4 if v == 1 else 0.6) if i % 4 == 0 else (0.6 if v == 1 else 0.4)
        for i, v in enumerate(y)
    ]
    return pl.DataFrame(
        {
            "event": ["vasopressor_start"] * n,
            "subject_id": list(range(n)),
            "visit_id": [0] * n,
            "time_hours": [float(i) for i in range(n)],
            "y@8h": [float(v) for v in y],
            "hazard@8h": hazard,
            "gbm@8h": gbm,
            "ctx.hours_into_visit": [72.0 if i % 2 == 0 else 12.0 for i in range(n)],
        }
    )


def _stratified_table() -> pl.DataFrame:
    """Hazard is perfect in "long", perfectly *wrong* in "short"; gbm is the reverse.

    Each stratum's AUROC is computed only over that stratum's own rows
    (scorer_auroc_table filters before scoring), so within one stratum
    there is no cross-stratum pooling to reason about: hazard's "long"
    AUROC is exactly 1.0 and its "short" AUROC is exactly 0.0 by
    construction, and the reverse for gbm -- a deliberately unambiguous
    case for testing that stratification changes which scorer wins.
    """
    n = 200
    long_seq = [i % 2 == 0 for i in range(n)]
    y = [1 if (i % 3 == 0) else 0 for i in range(n)]
    hazard = [(v if is_long else 1 - v) for v, is_long in zip(y, long_seq, strict=True)]
    gbm = [(1 - v if is_long else v) for v, is_long in zip(y, long_seq, strict=True)]
    return pl.DataFrame(
        {
            "event": ["vasopressor_start"] * n,
            "subject_id": list(range(n)),
            "visit_id": [0] * n,
            "time_hours": [float(i) for i in range(n)],
            "y@8h": [float(v) for v in y],
            "hazard@8h": [float(v) for v in hazard],
            "gbm@8h": [float(v) for v in gbm],
            "ctx.hours_into_visit": [72.0 if s else 12.0 for s in long_seq],
        }
    )


def test_scorer_auroc_table_finds_both_scorer_columns() -> None:
    table = _dumped_table()
    rows = scorer_auroc_table(table, horizons=(8.0,))
    scorers = {r.scorer for r in rows}
    assert scorers == {"hazard", "gbm"}
    all_rows = [r for r in rows if r.stratum == "all"]
    assert len(all_rows) == 2  # one per scorer, one event/horizon cell


def test_scorer_auroc_table_all_stratum_always_present() -> None:
    table = _dumped_table()
    rows = scorer_auroc_table(table, horizons=(8.0,))
    assert any(r.stratum == "all" for r in rows)


def test_scorer_auroc_table_stratifies_on_a_context_column() -> None:
    table = _stratified_table()
    rows = scorer_auroc_table(
        table,
        horizons=(8.0,),
        strata={
            "long": pl.col("ctx.hours_into_visit") >= 48,
            "short": pl.col("ctx.hours_into_visit") < 48,
        },
    )
    by = {(r.scorer, r.stratum): r for r in rows}
    # by construction: hazard is perfect in "long", perfectly wrong in
    # "short"; gbm is the exact reverse.
    assert by[("hazard", "long")].auroc == pytest.approx(1.0)
    assert by[("hazard", "short")].auroc == pytest.approx(0.0)
    assert by[("gbm", "long")].auroc == pytest.approx(0.0)
    assert by[("gbm", "short")].auroc == pytest.approx(1.0)


def test_scorer_auroc_table_reports_none_below_min_rows_rather_than_omitting() -> None:
    table = _dumped_table()
    rows = scorer_auroc_table(
        table,
        horizons=(8.0,),
        strata={"tiny": pl.col("subject_id") < 5},
        min_rows=50,
    )
    tiny_rows = [r for r in rows if r.stratum == "tiny"]
    assert tiny_rows  # present, not silently dropped
    assert all(r.auroc is None for r in tiny_rows)
    assert all(r.n < 50 for r in tiny_rows)


def test_scorer_auroc_table_handles_missing_horizon_column_gracefully() -> None:
    table = _dumped_table()
    rows = scorer_auroc_table(table, horizons=(8.0, 24.0))
    # 24h has no y@24h/hazard@24h/gbm@24h columns in this fixture: skipped,
    # not an error.
    assert all(r.horizon_hours == 8.0 for r in rows)


def test_best_scorer_per_cell_picks_the_higher_auroc() -> None:
    table = _dumped_table()
    rows = scorer_auroc_table(table, horizons=(8.0,))
    best = best_scorer_per_cell(rows)
    assert best[("vasopressor_start", 8.0)] == "hazard"


def test_best_scorer_per_cell_respects_the_stratum_argument() -> None:
    table = _stratified_table()
    rows = scorer_auroc_table(
        table,
        horizons=(8.0,),
        strata={
            "long": pl.col("ctx.hours_into_visit") >= 48,
            "short": pl.col("ctx.hours_into_visit") < 48,
        },
    )
    assert (
        best_scorer_per_cell(rows, stratum="long")[("vasopressor_start", 8.0)]
        == "hazard"
    )
    assert (
        best_scorer_per_cell(rows, stratum="short")[("vasopressor_start", 8.0)] == "gbm"
    )


def test_best_scorer_per_cell_omits_a_cell_with_no_computable_auroc() -> None:
    rows = scorer_auroc_table(
        _dumped_table(), horizons=(8.0,), strata={"tiny": pl.col("subject_id") < 5}
    )
    assert ("vasopressor_start", 8.0) not in best_scorer_per_cell(rows, stratum="tiny")


@pytest.mark.parametrize("row", INTERPRETABILITY_COMPARISON)
def test_interpretability_comparison_rows_are_fully_populated(row: object) -> None:
    # every field is a real, non-empty statement -- a placeholder here
    # would silently misrepresent a capability rather than compute a
    # wrong number, which is exactly what this table exists to avoid.
    for field_name in ("capability", "hazard_head", "gbm", "tabicl", "note"):
        value = getattr(row, field_name)
        assert isinstance(value, str) and value.strip()


def test_interpretability_comparison_covers_the_three_stated_capabilities() -> None:
    capabilities = {row.capability for row in INTERPRETABILITY_COMPARISON}
    assert any("intervention" in c.lower() for c in capabilities)
    assert any(
        "time-to-event" in c.lower() or "survival" in c.lower() for c in capabilities
    )
    assert any("attribution" in c.lower() for c in capabilities)
