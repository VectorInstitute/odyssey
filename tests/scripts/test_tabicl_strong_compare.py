"""The matched-row contract behind the three-column comparator table.

TabICL costs about half an hour of prediction per cell, so it is scored on
4 of the 37 held-out shards while the eval chain's alerts.json covers all
37. A table that mixes the two is comparing different samples. These pin
the two guarantees that make the columns comparable: the join refuses a
partial overlap, and every scorer at a horizon is scored on one shared row
mask.
"""

import numpy as np
import polars as pl
import pytest

from scripts.tabicl_strong_compare import (
    SCORERS,
    _dumped,
    join_matched_rows,
    score_horizon,
)


def _dump(n: int = 40, *, with_hazard: bool = True) -> pl.DataFrame:
    """Build a dump-side frame: keys, labels, and the chain's two scorers."""
    rng = np.random.default_rng(0)
    y = (np.arange(n) % 3 == 0).astype(float)
    frame = pl.DataFrame(
        {
            "subject_id": [float(i // 2) for i in range(n)],
            "visit_id": [float(i // 2) for i in range(n)],
            "time_hours": [float(i) + 0.0000001 for i in range(n)],
            "event": ["death"] * n,
            "y@8h": y,
            "gbm@8h": y * 0.5 + rng.random(n) * 0.4,
        }
    )
    if with_hazard:
        frame = frame.with_columns(
            pl.Series("hazard@8h", y * 0.4 + rng.random(n) * 0.5)
        )
    return frame


def _scored(dump: pl.DataFrame, *, keep: int | None = None) -> pl.DataFrame:
    """Build the freshly-scored side: the same keys plus a tabicl column."""
    n = dump.height if keep is None else keep
    rng = np.random.default_rng(1)
    return pl.DataFrame(
        {
            "subject_id": dump["subject_id"].head(n),
            "visit_id": dump["visit_id"].head(n),
            "time_hours": dump["time_hours"].head(n),
            "tabicl@8h": rng.random(n),
        }
    )


def test_join_keeps_every_freshly_scored_row() -> None:
    dump = _dump()
    joined = join_matched_rows(dump, _scored(dump), "death")
    assert joined.height == dump.height
    assert "tabicl@8h" in joined.columns and "hazard@8h" in joined.columns


def test_join_tolerates_float_drift_in_the_time_key() -> None:
    """The two sides build time_hours through different float paths."""
    dump = _dump()
    scored = _scored(dump).with_columns(pl.col("time_hours") + 1e-9)
    assert join_matched_rows(dump, scored, "death").height == dump.height


def test_join_refuses_a_partial_overlap() -> None:
    """A shard/protocol mismatch must fail loudly, not shrink n quietly."""
    dump = _dump()
    scored = _scored(dump).with_columns(pl.col("time_hours") + 500.0)
    with pytest.raises(RuntimeError, match="row sets disagree"):
        join_matched_rows(dump, scored, "death")


def test_join_refuses_duplicate_keys_that_would_fan_out() -> None:
    dump = _dump()
    doubled = pl.concat([dump, dump.head(1)])
    with pytest.raises(RuntimeError, match="duplicate"):
        join_matched_rows(doubled, _scored(dump), "death")


def test_every_scorer_present_gets_scored() -> None:
    dump = _dump()
    cell = score_horizon(join_matched_rows(dump, _scored(dump), "death"), 8.0)
    assert cell is not None
    assert all(cell[name] is not None for name in SCORERS)
    assert cell["n"] == dump.height


def test_a_null_in_one_scorer_drops_the_row_from_all_of_them() -> None:
    """Otherwise the columns of one table row describe different samples."""
    dump = _dump()
    holed = dump.with_columns(
        pl.when(pl.arange(0, dump.height) < 4)
        .then(None)
        .otherwise(pl.col("hazard@8h"))
        .alias("hazard@8h")
    )
    cell = score_horizon(join_matched_rows(holed, _scored(dump), "death"), 8.0)
    assert cell is not None
    assert cell["n"] == dump.height - 4
    # The rows left with hazard, so GBM and TabICL lost them too: dropping
    # the hazard column instead returns all of them, which is what makes
    # the shrinkage attributable to the shared mask rather than the data.
    without = score_horizon(
        join_matched_rows(holed.drop("hazard@8h"), _scored(dump), "death"), 8.0
    )
    assert without is not None and without["n"] == dump.height


def test_a_missing_scorer_is_simply_absent_not_an_error() -> None:
    """--skip-tabicl and pre-hazard dumps both land here."""
    dump = _dump(with_hazard=False)
    cell = score_horizon(join_matched_rows(dump, _scored(dump), "death"), 8.0)
    assert cell is not None
    assert cell["gbm"] is not None and cell["tabicl"] is not None
    assert "hazard" not in cell


def test_absent_horizon_and_single_class_both_return_none() -> None:
    dump = _dump()
    joined = join_matched_rows(dump, _scored(dump), "death")
    assert score_horizon(joined, 24.0) is None
    all_negative = joined.with_columns(pl.lit(0.0).alias("y@8h"))
    assert score_horizon(all_negative, 8.0) is None


@pytest.mark.parametrize(
    ("column", "keep"),
    [
        ("subject_id", True),
        ("event", True),
        ("y@8h", True),
        ("hazard@72h", True),
        ("tabicl@24h", True),
        ("gbm@8h", True),
        ("next_mass", False),
        ("ctx.age_years", False),
        ("f_creatinine_last", False),
        ("concept", False),
    ],
)
def test_dump_keeps_scores_and_keys_and_drops_the_feature_panel(
    column: str, keep: bool
) -> None:
    """The dump feeds alerts_cis.py; 600 feature columns would bloat it."""
    assert _dumped(column) is keep


def test_hazard_is_the_reference_scorer() -> None:
    """alerts_cis.py pairs later scorers against the first one."""
    assert SCORERS[0] == "hazard"
