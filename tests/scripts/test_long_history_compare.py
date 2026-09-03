"""Long-history comparison: history tags, dump joins, stratified paired AUROC."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from scripts.long_history_compare import (
    compare,
    history_lengths,
    join_dumps,
    tag_history,
)


def _events() -> pl.DataFrame:
    t0 = datetime(2020, 1, 1)
    rows = []
    # subject 1: 6 timed events at hours 0..5 (+ a birth row with null time)
    rows.append({"subject_id": 1, "time": None, "code": "MEDS_BIRTH"})
    rows += [
        {"subject_id": 1, "time": t0 + timedelta(hours=h), "code": f"LAB//{h}"}
        for h in range(6)
    ]
    # subject 2: 3 events at hours 0, 2, 10
    rows += [
        {"subject_id": 2, "time": t0 + timedelta(hours=h), "code": "VITALS//x"}
        for h in (0, 2, 10)
    ]
    return pl.DataFrame(rows)


def test_history_lengths_count_timed_events_since_the_origin() -> None:
    lengths = history_lengths(_events())
    assert lengths[1].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert lengths[2].tolist() == [0.0, 2.0, 10.0]
    frame = pl.DataFrame(
        {"subject_id": [1, 1, 2, 2, 3], "time_hours": [0.5, 4.0, 1.0, 10.0, 3.0]}
    )
    tagged = tag_history(frame, lengths)
    # events with time <= landmark: subj 1 @0.5 -> 1, @4.0 -> 5; subj 2 @1.0 -> 1,
    # @10 -> 3; an unseen subject counts 0.
    assert tagged["history_len"].to_list() == [1, 5, 1, 3, 0]


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
    # half the rows "beyond the window": first 200 landmarks short history, rest long
    joined = joined.with_columns(pl.Series("history_len", [10] * 200 + [5000] * 200))
    cells = compare(
        joined, label_a="hybrid", label_b="transformer", window=2048, n_boot=50, seed=0
    )
    strata = {c["stratum"]: c for c in cells}
    assert set(strata) == {"within_window", "beyond_window"}
    for c in cells:
        assert c["auroc_hybrid"] > c["auroc_transformer"]
        assert c["delta_a_minus_b"]["point_estimate"] > 0
        assert c["n_rows"] == 200
