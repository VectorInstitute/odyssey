"""Tests for the paired two-run comparison (scripts/compare_runs_paired.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts import compare_runs_paired as crp
from scripts.compare_runs_paired import (
    RowSetMismatchError,
    compare,
    inference_block,
    join_event,
    markdown_table,
    score_pair,
    summarise,
)
from tests.scripts.test_alerts_cis import _dump


def _pair(shift: float = 0.0, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two dumps in the alerts writer format on identical rows.

    Dump B's hazard is A's plus ``shift`` on the positives, so a positive
    shift makes B strictly better separated; ``shift == 0`` gives
    identical scores. B's GBM is a different refit (re-drawn noise).
    """
    a = _dump(seed=seed)
    b = _dump(seed=seed + 1)  # same rows and labels, different gbm noise
    hazard_b = np.clip(a["hazard@8h"].to_numpy() + shift * a["y@8h"].to_numpy(), 0, 1)
    b = b.with_columns(pl.Series("hazard@8h", hazard_b))
    assert a.select("subject_id", "visit_id", "time_hours").equals(
        b.select("subject_id", "visit_id", "time_hours")
    )
    return a, b


def _joined(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    joined, counts = join_event(a, b, "death", label_a="a", label_b="b")
    assert counts["n_unmatched_a"] == 0 and counts["n_unmatched_b"] == 0
    return joined


def test_shifted_scores_give_an_interval_that_excludes_zero() -> None:
    a, b = _pair(shift=0.3)
    cell = score_pair(_joined(a, b), "hazard", 8.0, n_boot=200, seed=0)
    assert cell is not None and "unscoreable" not in cell
    assert cell["n_at_risk"] == 240 and cell["n_positive"] == 80
    assert cell["n_subjects"] == 60
    assert cell["auroc_b"] > cell["auroc_a"]
    delta = cell["delta_b_minus_a"]
    assert delta["point"] == pytest.approx(cell["auroc_b"] - cell["auroc_a"])
    assert delta["ci_low"] > 0 and delta["separated"] is True
    assert summarise({"death@8h": cell}) == {
        "b_beats_a": ["death@8h"],
        "a_beats_b": [],
        "ties": [],
    }


def test_identical_scores_give_a_zero_point_and_zero_width_interval() -> None:
    a, b = _pair(shift=0.0)
    cell = score_pair(_joined(a, b), "hazard", 8.0, n_boot=100, seed=0)
    assert cell is not None
    assert cell["auroc_a"] == cell["auroc_b"]
    delta = cell["delta_b_minus_a"]
    assert delta["point"] == 0.0
    assert delta["ci_low"] == 0.0 and delta["ci_high"] == 0.0
    assert delta["separated"] is False
    assert summarise({"death@8h": cell})["ties"] == ["death@8h"]


def test_dropped_rows_are_refused_with_both_counts_in_the_message() -> None:
    a, b = _pair()
    b_short = b.head(228)  # 5% of 240 rows dropped
    with pytest.raises(RowSetMismatchError) as excinfo:
        join_event(a, b_short, "death", label_a="bottleneck", label_b="baseline")
    msg = str(excinfo.value)
    assert "bottleneck has 240 rows" in msg
    assert "baseline has 228 rows" in msg
    assert "the join keeps 228" in msg
    assert "12 of bottleneck's 240 rows are unmatched" in msg


def test_one_extra_row_in_a_large_dump_is_within_tolerance() -> None:
    a = _dump(n_subjects=300, rows_per=4)  # 1200 rows
    b = a.head(1199)  # one row missing: 0.08%, under the 0.1% limit
    joined, counts = join_event(a, b, "death", label_a="a", label_b="b")
    assert joined.height == 1199
    assert counts["n_unmatched_a"] == 1 and counts["n_unmatched_b"] == 0


def test_duplicate_keys_are_refused() -> None:
    a, b = _pair()
    with pytest.raises(RowSetMismatchError, match="duplicate row keys"):
        join_event(pl.concat([a, a.head(1)]), b, "death", label_a="a", label_b="b")


def test_label_disagreement_is_refused() -> None:
    a, b = _pair()
    b = b.with_columns((1.0 - pl.col("y@8h")).alias("y@8h"))
    with pytest.raises(RowSetMismatchError, match="different labels"):
        score_pair(_joined(a, b), "hazard", 8.0, n_boot=10, seed=0)


def test_two_subject_fixture_shows_subject_clustering() -> None:
    """Skipped single-class resamples prove subjects are the resampling unit.

    With one all-positive and one all-negative subject, a resample that
    draws the same subject twice is single-class and must be skipped;
    that only happens when subjects, not rows, are resampled.
    """
    rng = np.random.default_rng(0)
    y = np.array([1.0] * 20 + [0.0] * 20)
    frame = pl.DataFrame(
        {
            "event": ["death"] * 40,
            "subject_id": np.repeat([1, 2], 20),
            "visit_id": np.repeat([1, 2], 20),
            "time_hours": np.arange(40, dtype=float),
            "y@8h": y,
            "hazard@8h": np.where(y == 1, 0.6, 0.3) + rng.uniform(0, 0.2, 40),
        }
    )
    b = frame.with_columns(
        (pl.col("hazard@8h") + 0.1 * pl.col("y@8h")).alias("hazard@8h")
    )
    cell = score_pair(_joined(frame, b), "hazard", 8.0, n_boot=200, seed=0)
    assert cell is not None
    assert cell["n_subjects"] == 2
    delta = cell["delta_b_minus_a"]
    assert delta["n_boot_used"] + delta["n_boot_skipped"] == 200
    # P(both draws are the same subject) = 1/2, so about half are skipped
    assert 60 <= delta["n_boot_skipped"] <= 140


def test_compare_reports_gbm_refits_without_a_bootstrap_and_rejects_missing_events() -> (
    None
):
    a, b = _pair(shift=0.2)
    out = compare(
        a,
        b,
        label_a="a",
        label_b="b",
        scorer="hazard",
        events=None,
        horizons=None,
        n_boot=50,
        seed=0,
    )
    assert list(out["cells"]) == ["death@8h"]
    assert out["events"]["death"]["n_rows_joined"] == 240
    g = out["gbm"]["death@8h"]
    assert g["auroc_a"] != g["auroc_b"]  # two refits
    assert g["delta_b_minus_a"] == pytest.approx(g["auroc_b"] - g["auroc_a"])
    assert "ci_low" not in g
    table = markdown_table(out["cells"], out["gbm"], label_a="a", label_b="b")
    assert "| death | 8h | 240 | 80 | 60 |" in table
    with pytest.raises(RowSetMismatchError, match="not in both dumps"):
        compare(
            a,
            b,
            label_a="a",
            label_b="b",
            scorer="hazard",
            events=["sepsis"],
            horizons=None,
            n_boot=10,
            seed=0,
        )


def test_inference_block_reports_side_by_side_deltas() -> None:
    inf_a = {
        "set_top1_accuracy": 0.80,
        "top1_accuracy": 0.37,
        "cross_entropy": 3.5,
        "top5_accuracy": 0.7,
        "n_predictions": 100,
        "n_set_predictions": 99,
    }
    inf_b = {**inf_a, "top1_accuracy": 0.39}
    block = inference_block(inf_a, inf_b)
    assert block is not None
    assert block["delta_b_minus_a"]["top1_accuracy"] == pytest.approx(0.02)
    assert block["delta_b_minus_a"]["n_predictions"] is None
    assert inference_block(None, None) is None


def test_main_writes_json_and_refuses_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = _pair(shift=0.3)
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    a.with_columns(pl.lit(4).alias("landmark_protocol_version")).write_parquet(pa)
    b.with_columns(pl.lit(4).alias("landmark_protocol_version")).write_parquet(pb)
    inf = tmp_path / "inference_results.json"
    inf.write_text(
        json.dumps(
            {
                "task_metrics": {
                    "top1_accuracy": 0.37,
                    "set_top1_accuracy": 0.8,
                    "cross_entropy": 3.5,
                }
            }
        )
    )
    out = tmp_path / "paired.json"
    argv = [
        "compare_runs_paired",
        "--dump-a",
        str(pa),
        "--dump-b",
        str(pb),
        "--label-a",
        "bottleneck",
        "--label-b",
        "baseline",
        "--inference-a",
        str(inf),
        "--inference-b",
        str(inf),
        "--horizons",
        "8",
        "--n-boot",
        "20",
        "--seed",
        "0",
        "--output-json",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    crp.main()
    payload = json.loads(out.read_text())
    assert payload["row_key"] == ["event", "subject_id", "visit_id", "time_hours"]
    assert payload["summary"]["b_beats_a"] == ["death@8h"]
    assert payload["landmark_protocol_version"] == {"a": 4, "b": 4}
    assert payload["inference"]["delta_b_minus_a"]["top1_accuracy"] == 0.0
    assert payload["gbm"]["death@8h"]["auroc_a"] is not None

    b.head(200).write_parquet(pb)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        crp.main()
    assert excinfo.value.code == 2
