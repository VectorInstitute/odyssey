"""Tests for the missingness stress protocol sweep's non-GPU-dependent pieces.

docs/missingness_protocol.md. run_sweep() itself needs a real trained
run/checkpoint (evaluate_alerts) and isn't exercised here -- these tests
cover what's testable without one: cell generation and the aggregation step,
both pure/file-based.

scripts/ isn't a Python package (no __init__.py, by design -- every other
script under it is invoked directly, not imported), so the module under
test is loaded from its file path rather than a normal import.
"""

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts" / "missingness_sweep.py"

_spec = importlib.util.spec_from_file_location("missingness_sweep", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
missingness_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(missingness_sweep)


T0 = datetime(2024, 1, 1)
_SCHEMA = {
    "subject_id": pl.Int64,
    "code": pl.Utf8,
    "time": pl.Datetime,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _shard(subject_ids: List[int]) -> pl.DataFrame:
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in subject_ids:
        hadm = 1000 + sid
        rows.append((sid, "ICU_ADMISSION//MICU", T0, None, hadm))
        rows.append((sid, "LAB//RESULT//50912//", T0 + timedelta(hours=2), 1.1, hadm))
        rows.append((sid, "HOSPITAL_DISCHARGE//", T0 + timedelta(hours=10), None, hadm))
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def test_generate_cells_writes_all_eight_by_default(tmp_path: Path) -> None:
    held_out = tmp_path / "held_out"
    held_out.mkdir()
    _shard([1, 2, 3]).write_parquet(held_out / "0.parquet")

    cell_dirs = missingness_sweep._generate_cells(
        held_out,
        tmp_path / "degraded",
        seed=0,
        cells=None,
        source="mimic_iv",
        overwrite=False,
    )
    assert len(cell_dirs) == 8
    for name, cell_dir in cell_dirs.items():
        assert (cell_dir / "metadata.json").is_file(), name
        assert (cell_dir / "0.parquet").is_file(), name


def test_generate_cells_subset_and_unknown_name(tmp_path: Path) -> None:
    held_out = tmp_path / "held_out"
    held_out.mkdir()
    _shard([1]).write_parquet(held_out / "0.parquet")

    cell_dirs = missingness_sweep._generate_cells(
        held_out,
        tmp_path / "degraded",
        seed=0,
        cells=["blackout_labs", "lag_4h"],
        source="mimic_iv",
        overwrite=False,
    )
    assert set(cell_dirs) == {"blackout_labs", "lag_4h"}

    with pytest.raises(ValueError, match="unknown cell name"):
        missingness_sweep._generate_cells(
            held_out,
            tmp_path / "degraded2",
            seed=0,
            cells=["not_a_real_cell"],
            source="mimic_iv",
            overwrite=False,
        )


def test_generate_cells_skips_already_generated_unless_overwrite(
    tmp_path: Path,
) -> None:
    held_out = tmp_path / "held_out"
    held_out.mkdir()
    _shard([1]).write_parquet(held_out / "0.parquet")
    degraded_root = tmp_path / "degraded"

    missingness_sweep._generate_cells(
        held_out,
        degraded_root,
        seed=0,
        cells=["blackout_labs"],
        source="mimic_iv",
        overwrite=False,
    )
    stamp = (degraded_root / "blackout_labs" / "metadata.json").stat().st_mtime

    missingness_sweep._generate_cells(
        held_out,
        degraded_root,
        seed=0,
        cells=["blackout_labs"],
        source="mimic_iv",
        overwrite=False,
    )
    assert (degraded_root / "blackout_labs" / "metadata.json").stat().st_mtime == stamp


def _fake_alerts_json(
    path: Path,
    *,
    cell: str,
    transform: Optional[str],
    auroc: float,
    n_unscoreable: Optional[int] = None,
) -> None:
    payload = {
        "cell": cell,
        "cell_metadata": {"transform": transform} if transform else None,
        "run_dir": "/fake/run",
        "held_out_shard_dir": "/fake/held_out",
        "n_unscoreable": n_unscoreable,
        "metrics": [
            {
                "event": "icu_admission",
                "horizon_hours": 8.0,
                "scorer": "hazard",
                "n_at_risk": 20,
                "n_positive": 5,
                "n_censored": 0,
                "auroc": auroc,
                "brier": 0.1,
                "calibration": [{"predicted": 0.5, "observed": 0.4, "n": 20}],
                "baseline_feature_set": None,
                "baseline_n_features": None,
                "baseline_params": None,
                "landmark_protocol_version": 3,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregate_builds_a_table_from_per_cell_json(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    clean_json = results_dir / "clean_alerts.json"
    cell_json = results_dir / "blackout_labs_alerts.json"
    _fake_alerts_json(clean_json, cell="clean", transform=None, auroc=0.9)
    _fake_alerts_json(
        cell_json, cell="blackout_labs", transform="family_blackout", auroc=0.7
    )

    json_path, md_path = missingness_sweep.aggregate(
        results_dir,
        {"clean": clean_json, "blackout_labs": cell_json},
        tmp_path,
    )
    assert json_path.is_file()
    assert md_path.is_file()
    table = json.loads(json_path.read_text())
    assert len(table) == 1
    row = table[0]
    assert row["cell"] == "blackout_labs"
    assert row["transform"] == "family_blackout"
    assert abs(row["auroc_delta"] - (-0.2)) < 1e-9
    assert row["n_unscoreable"] == 0  # the fake JSON above didn't set one


def test_aggregate_carries_n_unscoreable_into_the_table_and_markdown_note(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    clean_json = results_dir / "clean_alerts.json"
    cell_json = results_dir / "lag_4h_alerts.json"
    _fake_alerts_json(clean_json, cell="clean", transform=None, auroc=0.9)
    _fake_alerts_json(
        cell_json,
        cell="lag_4h",
        transform="lab_lag",
        auroc=0.8,
        n_unscoreable=129,
    )

    json_path, md_path = missingness_sweep.aggregate(
        results_dir,
        {"clean": clean_json, "lag_4h": cell_json},
        tmp_path,
    )
    table = json.loads(json_path.read_text())
    assert table[0]["n_unscoreable"] == 129
    md = md_path.read_text()
    assert "Reduced row sets" in md
    assert "lag_4h: 129 rows unscoreable" in md
