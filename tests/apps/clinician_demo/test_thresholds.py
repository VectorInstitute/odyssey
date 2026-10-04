"""Alert lines: threshold, sensitivity, PPV and the like-for-like GBM line."""

import json
from pathlib import Path

import polars as pl
import pytest

from apps.clinician_demo.export_thresholds import main as export_main
from apps.clinician_demo.thresholds import (
    AGGREGATE_THRESHOLDS_FILENAME,
    ALERTS_ROWS_FILENAME,
    compute_operating_points,
    export_operating_points,
    flag_threshold,
    horizon_key,
    load_aggregate_operating_points,
    load_or_compute_operating_points,
    operating_point,
    operating_points_for_run,
)


def _rows() -> pl.DataFrame:
    # 20 at-risk rows for AKI at 24 h: hazard 0.00..0.19, the top 4 are positives
    # except one; the GBM ranks perfectly. Two rows are not at risk (y null).
    hazard = [i / 100 for i in range(20)]
    outcome: list[float | None] = [0.0] * 20
    for i in (19, 18, 17, 5):
        outcome[i] = 1.0
    gbm = [1.0 if y == 1.0 else 0.0 for y in outcome]
    frame = pl.DataFrame(
        {
            "event": ["acute_kidney_injury"] * 20,
            "hazard@24h": hazard,
            "y@24h": outcome,
            "gbm@24h": gbm,
        }
    )
    extra = pl.DataFrame(
        {
            "event": ["acute_kidney_injury", "death"],
            "hazard@24h": [0.99, 0.5],
            "y@24h": [None, 1.0],
            "gbm@24h": [0.5, 0.5],
        },
        schema=frame.schema,
    )
    return pl.concat([frame, extra])


def test_horizon_key_formats_like_the_banked_files() -> None:
    assert [horizon_key(h) for h in (8.0, 24.0, 72.0, 0.5)] == [
        "8h",
        "24h",
        "72h",
        "0.5h",
    ]


def test_threshold_is_an_observed_upper_quantile() -> None:
    scores = pl.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert flag_threshold(scores, 0.2) == 0.9
    assert flag_threshold(scores, 0.05) == 1.0
    with pytest.raises(ValueError, match="empty"):
        flag_threshold(pl.Series([], dtype=pl.Float64), 0.1)


def test_operating_point_counts_only_at_risk_rows_and_scores_the_gbm_alike() -> None:
    point = operating_point(_rows(), "acute_kidney_injury", 24.0, 0.2)
    assert point is not None
    assert point.n_rows == 20  # the null-outcome row is excluded
    assert point.threshold == pytest.approx(0.16)  # 4 of 20 rows flagged
    assert point.alert_rate == pytest.approx(0.2)
    assert point.sensitivity == pytest.approx(3 / 4)  # rows 17-19 flagged, row 5 missed
    assert point.ppv == pytest.approx(3 / 4)  # row 16 is a false flag
    assert point.base_rate == pytest.approx(4 / 20)
    assert point.gbm_sensitivity == pytest.approx(
        1.0
    ) and point.gbm_ppv == pytest.approx(1.0)


def test_ties_are_reported_as_the_measured_flag_rate() -> None:
    rows = pl.DataFrame(
        {"event": ["e"] * 4, "hazard@8h": [0.5] * 4, "y@8h": [0.0, 1.0, 0.0, 0.0]}
    )
    point = operating_point(rows, "e", 8.0, 0.25)
    assert point is not None
    assert point.alert_rate == 1.0 and point.sensitivity == 1.0 and point.ppv == 0.25
    assert point.gbm_sensitivity is None and point.gbm_ppv is None  # no gbm column


def test_no_positives_gives_undefined_sensitivity_and_empty_event_gives_none() -> None:
    rows = pl.DataFrame(
        {"event": ["e"] * 3, "hazard@8h": [0.1, 0.2, 0.3], "y@8h": [0.0, 0.0, 0.0]}
    )
    point = operating_point(rows, "e", 8.0, 0.34)
    assert point is not None and point.sensitivity is None and point.ppv == 0.0
    assert operating_point(rows, "other", 8.0, 0.1) is None


def test_gbm_nulls_are_dropped_from_the_gbm_line_only() -> None:
    rows = _rows().with_columns(
        pl.when(pl.col("hazard@24h") < 0.1)
        .then(None)
        .otherwise(pl.col("gbm@24h"))
        .alias("gbm@24h")
    )
    point = operating_point(rows, "acute_kidney_injury", 24.0, 0.2)
    assert point is not None and point.n_rows == 20


def test_compute_reads_only_available_horizons_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "alerts_rows.parquet"
    _rows().write_parquet(path)
    points = compute_operating_points(
        path, ["acute_kidney_injury", "death", "absent"], [24.0, 72.0], 0.2
    )
    assert [(p.event, p.horizon_hours) for p in points] == [
        ("acute_kidney_injury", 24.0),
        ("death", 24.0),
    ]


def test_cache_is_reused_only_for_an_identical_request(tmp_path: Path) -> None:
    rows_path, cache = (
        tmp_path / "alerts_rows.parquet",
        tmp_path / "c" / "thresholds.json",
    )
    _rows().write_parquet(rows_path)
    first = load_or_compute_operating_points(
        rows_path, cache, ["acute_kidney_injury"], [24.0], 0.2
    )
    assert cache.exists()
    # tamper with the cached numbers: an identical request must return them
    doc = json.loads(cache.read_text())
    doc["points"][0]["threshold"] = 0.123
    cache.write_text(json.dumps(doc))
    again = load_or_compute_operating_points(
        rows_path, cache, ["acute_kidney_injury"], [24.0], 0.2
    )
    assert again[0].threshold == 0.123
    # a different alert rate invalidates the cache
    other = load_or_compute_operating_points(
        rows_path, cache, ["acute_kidney_injury"], [24.0], 0.1
    )
    assert other[0].threshold != 0.123 and first[0].threshold == pytest.approx(0.16)


def test_unreadable_cache_is_ignored_and_rewritten(tmp_path: Path) -> None:
    rows_path, cache = tmp_path / "alerts_rows.parquet", tmp_path / "thresholds.json"
    _rows().write_parquet(rows_path)
    cache.write_text("{not json")
    points = load_or_compute_operating_points(
        rows_path, cache, ["acute_kidney_injury"], [24.0], 0.2
    )
    assert len(points) == 1 and json.loads(cache.read_text())["points"]


def test_export_then_load_round_trips_the_alert_lines(tmp_path: Path) -> None:
    rows_path, out = tmp_path / "alerts_rows.parquet", tmp_path / "agg.json"
    _rows().write_parquet(rows_path)
    exported = export_operating_points(rows_path, out, [24.0, 72.0], 0.2)
    payload = json.loads(out.read_text())
    assert payload["alert_rate"] == 0.2
    assert {p["event"] for p in payload["points"]} == {"acute_kidney_injury", "death"}
    # aggregates only: nothing row-level travels with the file
    assert all("subject_id" not in p and "visit_id" not in p for p in payload["points"])
    loaded = load_aggregate_operating_points(
        out, ["acute_kidney_injury", "death"], [24.0], 0.2
    )
    assert loaded == exported


def test_loaded_alert_lines_are_filtered_to_the_request(tmp_path: Path) -> None:
    rows_path, out = tmp_path / "alerts_rows.parquet", tmp_path / "agg.json"
    _rows().write_parquet(rows_path)
    export_operating_points(rows_path, out, [24.0], 0.2)
    only_death = load_aggregate_operating_points(out, ["death"], [24.0], 0.2)
    assert [(p.event, p.horizon_hours) for p in only_death] == [("death", 24.0)]
    assert load_aggregate_operating_points(out, ["death"], [8.0], 0.2) == []


def test_aggregate_file_absent_or_at_another_rate_gives_no_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "agg.json"
    assert load_aggregate_operating_points(out, ["death"], [24.0], 0.05) == []
    rows_path = tmp_path / "alerts_rows.parquet"
    _rows().write_parquet(rows_path)
    export_operating_points(rows_path, out, [24.0], 0.2)
    assert load_aggregate_operating_points(out, ["death"], [24.0], 0.05) == []
    assert "alert rate" in caplog.text


def test_a_run_uses_its_rows_when_present_else_its_export(tmp_path: Path) -> None:
    gpu_run, laptop_run = tmp_path / "gpu", tmp_path / "laptop"
    gpu_run.mkdir()
    laptop_run.mkdir()
    _rows().write_parquet(gpu_run / ALERTS_ROWS_FILENAME)
    cache = tmp_path / "cache" / "thresholds.json"
    from_rows = operating_points_for_run(gpu_run, cache, ["death"], [24.0], 0.2)
    assert cache.exists() and [p.event for p in from_rows] == ["death"]

    assert operating_points_for_run(laptop_run, cache, ["death"], [24.0], 0.2) == []
    export_operating_points(
        gpu_run / ALERTS_ROWS_FILENAME,
        laptop_run / AGGREGATE_THRESHOLDS_FILENAME,
        [24.0],
        0.2,
    )
    from_export = operating_points_for_run(laptop_run, cache, ["death"], [24.0], 0.2)
    assert from_export == from_rows


def test_export_command_writes_next_to_the_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _rows().write_parquet(tmp_path / ALERTS_ROWS_FILENAME)
    assert export_main(["--run-dir", str(tmp_path), "--alert-rate", "0.2"]) == 0
    payload = json.loads((tmp_path / AGGREGATE_THRESHOLDS_FILENAME).read_text())
    assert payload["alert_rate"] == 0.2
    # the demo's horizons: rows here carry 24 h only
    assert {p["horizon_hours"] for p in payload["points"]} == {24.0}
    assert "wrote 2 alert lines" in capsys.readouterr().out


def test_export_command_refuses_a_run_without_rows(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        export_main(["--run-dir", str(tmp_path)])
