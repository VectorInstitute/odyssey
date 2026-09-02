"""Tests for the alerts CI post-processor (scripts/alerts_cis.py)."""

import numpy as np
import polars as pl

from scripts.alerts_cis import (
    SCORER_ALIASES,
    horizons_in,
    score_cell,
    subsample_subjects,
)


def _dump(n_subjects: int = 60, rows_per: int = 4, seed: int = 0) -> pl.DataFrame:
    """Synthetic dump: correlated rows per subject, hazard better than gbm."""
    rng = np.random.default_rng(seed)
    sids = np.repeat(np.arange(n_subjects), rows_per)
    y = np.repeat((np.arange(n_subjects) % 3 == 0).astype(float), rows_per)
    noise = rng.uniform(0, 1, len(sids))
    hazard = np.where(y == 1, 0.35 + 0.5 * noise, 0.15 + 0.5 * noise)
    gbm = np.where(y == 1, 0.25 + 0.6 * noise, 0.15 + 0.6 * noise)
    return pl.DataFrame(
        {
            "event": ["death"] * len(sids),
            "subject_id": sids,
            "visit_id": sids,
            "time_hours": np.arange(len(sids), dtype=float),
            "y@8h": y,
            "hazard@8h": hazard,
            "gbm@8h": gbm,
        }
    )


def test_horizons_detected_from_columns() -> None:
    assert horizons_in(_dump(), "hazard") == [8.0]
    assert horizons_in(_dump(), "tabicl") == []


def test_score_cell_reports_cis_and_paired_delta_on_shared_rows() -> None:
    cell = score_cell(_dump(), ["hazard", "gbm"], 8.0, n_boot=200, seed=0)
    assert cell is not None and "unscoreable" not in cell
    assert cell["n"] == 240 and cell["n_positive"] == 80
    for s in ("hazard", "gbm"):
        for metric in ("auroc", "auprc"):
            m = cell["scorers"][s][metric]
            assert m["ci_low"] is not None and m["ci_low"] <= m["point"] <= m["ci_high"]
    delta = cell["paired_deltas"]["hazard_minus_gbm"]["auroc"]
    assert delta["point"] > 0  # hazard constructed better
    assert "separated" in delta


def test_rows_with_any_missing_scorer_are_excluded_from_both_arms() -> None:
    frame = _dump()
    # null out gbm on half the rows: the intersection must shrink for BOTH
    frame = frame.with_columns(
        pl.when(pl.col("time_hours") < 120)
        .then(None)
        .otherwise(pl.col("gbm@8h"))
        .alias("gbm@8h")
    )
    cell = score_cell(frame, ["hazard", "gbm"], 8.0, n_boot=50, seed=0)
    assert cell is not None
    assert cell["n"] == 120  # not 240: hazard is scored on the intersection too


def test_single_class_cell_is_marked_unscoreable() -> None:
    frame = _dump().with_columns(pl.lit(0.0).alias("y@8h"))
    cell = score_cell(frame, ["hazard"], 8.0, n_boot=10, seed=0)
    assert cell is not None and cell.get("unscoreable") is True


def test_subsample_keeps_whole_subjects_and_is_seeded() -> None:
    frame = _dump(n_subjects=60, rows_per=4)
    small = subsample_subjects(frame, max_subjects=10, seed=0)
    assert small["subject_id"].n_unique() == 10
    assert small.height == 40  # every row of each kept subject
    again = subsample_subjects(frame, max_subjects=10, seed=0)
    assert small["subject_id"].to_list() == again["subject_id"].to_list()


def test_subsample_is_a_noop_when_not_smaller() -> None:
    frame = _dump(n_subjects=12)
    assert subsample_subjects(frame, max_subjects=None, seed=0).height == frame.height
    assert subsample_subjects(frame, max_subjects=12, seed=0).height == frame.height


def test_alerts_json_scorer_name_maps_to_dump_column() -> None:
    assert SCORER_ALIASES["baseline_gbm"] == "gbm"
