"""Tests for the missingness stress protocol's degradation-table aggregator.

docs/missingness_protocol.md. Pure-computation module: no model, no shard
dir -- everything here is built from small, hand-constructed AlertMetrics-
shaped records and (for AUPRC) synthetic row-dump parquet files.
"""

from pathlib import Path

import polars as pl

from odyssey.reporting.missingness_report import (
    CLEAN_CELL,
    CellMetricRow,
    auprc_from_rows,
    build_degradation_table,
    ece_from_calibration,
    load_cell_metrics,
    render_markdown,
    write_json,
    write_markdown,
)


def _calibration(pairs: list[tuple]) -> list[dict[str, float]]:
    return [{"predicted": p, "observed": o, "n": n} for p, o, n in pairs]


def test_ece_from_calibration_weighted_mean_gap() -> None:
    # gap 0.1 over 10 rows, gap 0.3 over 30 rows -> weighted mean 0.25
    calibration = _calibration([(0.5, 0.4, 10), (0.5, 0.2, 30)])
    ece = ece_from_calibration(calibration)
    assert ece is not None
    assert abs(ece - 0.25) < 1e-9


def test_ece_from_calibration_none_when_missing_or_empty() -> None:
    assert ece_from_calibration(None) is None
    assert ece_from_calibration([]) is None


def test_auprc_from_rows_computes_from_a_real_dump(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.parquet"
    pl.DataFrame(
        {
            "hazard@8h": [0.9, 0.8, 0.2, 0.1, 0.7],
            "y@8h": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
    ).write_parquet(rows_path)
    auprc = auprc_from_rows(rows_path, scorer="hazard", horizon_hours=8.0)
    assert auprc is not None
    assert 0.0 < auprc <= 1.0


def test_auprc_from_rows_none_when_no_dump() -> None:
    assert auprc_from_rows(None, scorer="hazard", horizon_hours=8.0) is None


def test_auprc_from_rows_none_for_an_unrecognized_scorer(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.parquet"
    pl.DataFrame({"concept": [0.5], "y@8h": [1.0]}).write_parquet(rows_path)
    assert auprc_from_rows(rows_path, scorer="concept", horizon_hours=8.0) is None


def test_auprc_from_rows_none_when_columns_missing(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.parquet"
    pl.DataFrame({"other_col": [1.0]}).write_parquet(rows_path)
    assert auprc_from_rows(rows_path, scorer="hazard", horizon_hours=8.0) is None


def test_load_cell_metrics_computes_every_field(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.parquet"
    pl.DataFrame({"hazard@8h": [0.9, 0.1], "y@8h": [1.0, 0.0]}).write_parquet(rows_path)
    metrics = [
        {
            "event": "icu_admission",
            "horizon_hours": 8.0,
            "scorer": "hazard",
            "n_at_risk": 2,
            "n_positive": 1,
            "n_censored": 0,
            "auroc": 0.9,
            "brier": 0.1,
            "calibration": _calibration([(0.5, 0.4, 2)]),
        }
    ]
    rows = load_cell_metrics(
        "blackout_labs", metrics, transform="family_blackout", rows_path=rows_path
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.cell == "blackout_labs"
    assert r.transform == "family_blackout"
    assert r.scorer == "hazard"
    assert r.event == "icu_admission"
    assert r.horizon_hours == 8.0
    assert r.auroc == 0.9
    assert r.ece is not None and abs(r.ece - 0.1) < 1e-9
    assert r.auprc is not None
    assert r.n_unscoreable == 0  # default when the caller doesn't pass one


def test_load_cell_metrics_stamps_n_unscoreable_on_every_row() -> None:
    metrics = [
        {
            "event": "icu_admission",
            "horizon_hours": 8.0,
            "scorer": "hazard",
            "n_at_risk": 20,
            "n_positive": 5,
            "n_censored": 0,
            "auroc": 0.8,
            "calibration": None,
        },
        {
            "event": "icu_admission",
            "horizon_hours": 24.0,
            "scorer": "hazard",
            "n_at_risk": 20,
            "n_positive": 5,
            "n_censored": 0,
            "auroc": 0.7,
            "calibration": None,
        },
    ]
    rows = load_cell_metrics(
        "lag_4h", metrics, transform="lab_lag", rows_path=None, n_unscoreable=129
    )
    assert [r.n_unscoreable for r in rows] == [129, 129]


def test_build_degradation_table_computes_deltas_against_matching_clean_row() -> None:
    clean = [
        CellMetricRow(
            cell=CLEAN_CELL,
            transform=None,
            scorer="hazard",
            event="icu_admission",
            horizon_hours=8.0,
            n_at_risk=100,
            auroc=0.90,
            auprc=0.60,
            ece=0.05,
        )
    ]
    degraded = [
        CellMetricRow(
            cell="mcar_0_5",
            transform="mcar",
            scorer="hazard",
            event="icu_admission",
            horizon_hours=8.0,
            n_at_risk=100,
            auroc=0.75,
            auprc=0.50,
            ece=0.12,
        )
    ]
    table = build_degradation_table(clean, {"mcar_0_5": degraded})
    assert len(table) == 1
    row = table[0]
    assert row["cell"] == "mcar_0_5"
    assert abs(row["auroc_delta"] - (-0.15)) < 1e-9
    assert abs(row["auprc_delta"] - (-0.10)) < 1e-9
    assert abs(row["ece_delta"] - 0.07) < 1e-9


def test_build_degradation_table_carries_n_unscoreable_through() -> None:
    clean = [
        CellMetricRow(
            cell=CLEAN_CELL,
            transform=None,
            scorer="hazard",
            event="icu_admission",
            horizon_hours=8.0,
            n_at_risk=100,
            auroc=0.90,
            auprc=0.60,
            ece=0.05,
        )
    ]
    degraded = [
        CellMetricRow(
            cell="lag_4h",
            transform="lab_lag",
            scorer="hazard",
            event="icu_admission",
            horizon_hours=8.0,
            n_at_risk=100,
            auroc=0.80,
            auprc=0.55,
            ece=0.08,
            n_unscoreable=129,
        )
    ]
    table = build_degradation_table(clean, {"lag_4h": degraded})
    assert table[0]["n_unscoreable"] == 129


def test_build_degradation_table_none_delta_when_no_matching_clean_row() -> None:
    degraded = [
        CellMetricRow(
            cell="lag_4h",
            transform="lab_lag",
            scorer="hazard",
            event="acute_kidney_injury",  # no clean row for this event below
            horizon_hours=8.0,
            n_at_risk=10,
            auroc=0.8,
            auprc=0.4,
            ece=0.1,
        )
    ]
    clean: list[CellMetricRow] = []
    table = build_degradation_table(clean, {"lag_4h": degraded})
    assert table[0]["auroc_delta"] is None
    assert table[0]["auprc_delta"] is None
    assert table[0]["ece_delta"] is None
    # the cell's own values still land even without a clean match
    assert table[0]["auroc"] == 0.8


def test_write_json_and_markdown_round_trip(tmp_path: Path) -> None:
    table = [
        {
            "cell": "mcar_0_1",
            "transform": "mcar",
            "scorer": "hazard",
            "event": "death",
            "horizon_hours": 24.0,
            "n_at_risk": 50,
            "auroc": 0.8,
            "auroc_delta": -0.05,
            "auprc": 0.3,
            "auprc_delta": None,
            "ece": 0.02,
            "ece_delta": 0.01,
        }
    ]
    json_path = tmp_path / "out" / "degradation_table.json"
    md_path = tmp_path / "out" / "degradation_table.md"
    write_json(table, json_path)
    write_markdown(table, md_path)
    assert json_path.is_file()
    assert md_path.is_file()
    md = md_path.read_text()
    assert "mcar_0_1" in md
    assert "death" in md
    assert "-" in md  # the None auprc_delta renders as a placeholder, not "None"
    rendered_directly = render_markdown(table)
    assert rendered_directly == md


def test_render_markdown_notes_reduced_row_sets_when_unscoreable() -> None:
    table = [
        {
            "cell": "lag_4h",
            "transform": "lab_lag",
            "scorer": "hazard",
            "event": "vasopressor_start",
            "horizon_hours": 8.0,
            "n_at_risk": 50,
            "n_unscoreable": 129,
            "auroc": 0.8,
            "auroc_delta": -0.05,
            "auprc": 0.3,
            "auprc_delta": None,
            "ece": 0.02,
            "ece_delta": 0.01,
        },
        {
            "cell": "mcar_0_3",
            "transform": "mcar",
            "scorer": "hazard",
            "event": "vasopressor_start",
            "horizon_hours": 8.0,
            "n_at_risk": 50,
            "n_unscoreable": 0,
            "auroc": 0.8,
            "auroc_delta": -0.05,
            "auprc": 0.3,
            "auprc_delta": None,
            "ece": 0.02,
            "ece_delta": 0.01,
        },
    ]
    md = render_markdown(table)
    assert "129" in md
    assert "Reduced row sets" in md
    assert "lag_4h: 129 rows unscoreable" in md
    assert "mcar_0_3: 0 rows unscoreable" not in md


def test_render_markdown_no_note_when_no_cell_has_unscoreable_rows() -> None:
    table = [
        {
            "cell": "mcar_0_3",
            "transform": "mcar",
            "scorer": "hazard",
            "event": "vasopressor_start",
            "horizon_hours": 8.0,
            "n_at_risk": 50,
            "n_unscoreable": 0,
            "auroc": 0.8,
            "auroc_delta": -0.05,
            "auprc": 0.3,
            "auprc_delta": None,
            "ece": 0.02,
            "ece_delta": 0.01,
        }
    ]
    assert "Reduced row sets" not in render_markdown(table)
