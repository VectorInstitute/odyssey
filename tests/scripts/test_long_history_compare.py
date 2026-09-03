"""Long-history comparison: truncation tags, dump joins, stratified paired AUROC."""

import numpy as np
import polars as pl

from scripts.long_history_compare import compare, join_dumps, tag_truncation


def test_tag_truncation_marks_subjects_whose_second_dump_starts_later() -> None:
    a = pl.DataFrame(
        {
            "subject_id": [1.0, 1.0, 2.0, 2.0, 3.0],
            "time_hours": [0.0, 4.0, 0.0, 8.0, 4.0],
        }
    )
    b = pl.DataFrame(
        {"subject_id": [1.0, 2.0, 2.0, 3.0], "time_hours": [4.0, 0.0, 8.0, 4.0]}
    )
    tags = tag_truncation(a, b).sort("subject_id")
    assert tags["subject_id"].to_list() == [1.0, 2.0, 3.0]
    assert tags["truncated"].to_list() == [True, False, False]


def _dump(prefix_scores: float, n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = rng.integers(0, 40, size=n)
    y = rng.integers(0, 2, size=n).astype(float)
    return pl.DataFrame(
        {
            "subject_id": subjects.astype(float),
            "visit_id": subjects.astype(float),
            "time_hours": np.arange(n, dtype=float),
            "event": ["death"] * n,
            "y@24h": y,
            "hazard@24h": prefix_scores * y + rng.normal(scale=0.5, size=n),
            "gbm@24h": rng.normal(size=n),
        }
    )


def test_join_and_compare_split_by_history_and_report_paired_delta() -> None:
    a = _dump(2.0)  # strong scorer
    b = _dump(0.2)  # weak scorer, same rows and labels (same seed)
    joined = join_dumps(a, b, "hybrid", "transformer")
    assert joined.height == a.height
    assert set(joined.columns) >= {"y@24h", "hybrid@24h", "transformer@24h"}
    # half the rows from subjects the transformer saw whole, half truncated
    joined = joined.with_columns(pl.Series("truncated", [False] * 200 + [True] * 200))
    cells = compare(joined, label_a="hybrid", label_b="transformer", n_boot=50, seed=0)
    strata = {c["stratum"]: c for c in cells}
    assert set(strata) == {"whole", "truncated"}
    for c in cells:
        assert c["auroc_hybrid"] > c["auroc_transformer"]
        assert c["delta_a_minus_b"]["point_estimate"] > 0
        assert c["n_rows"] == 200
