"""Tests for scripts/gemini/export_codes.py.

No real database, no real filesystem beyond ``tmp_path``.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("polars", reason=_SKIP_REASON)

import polars as pl  # noqa: E402


def _load_module(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "gemini" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_codes_parquet(root: Path, counts: dict[str, int]) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"code": list(counts.keys()), "count": list(counts.values())}
    ).write_parquet(root / "metadata" / "codes.parquet")


def test_suppressed_code_count_rounds_or_floors() -> None:
    mod = _load_module("export_codes")

    assert mod._suppressed_code_count(0) == "<1000"
    assert mod._suppressed_code_count(999) == "<1000"
    assert mod._suppressed_code_count(1000) == "1000"
    assert mod._suppressed_code_count(1499) == "1000"
    assert mod._suppressed_code_count(1500) == "2000"
    assert mod._suppressed_code_count(50_499) == "50000"


def test_export_codes_raises_when_no_codes_parquet(tmp_path: Path) -> None:
    mod = _load_module("export_codes")

    with pytest.raises(RuntimeError, match="not found"):
        mod.export_codes(output_dir=tmp_path)


def test_export_codes_writes_every_code_suppressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module("export_codes")
    inventory_path = tmp_path / "out" / "codes_inventory.json"
    monkeypatch.setattr(mod, "CODES_INVENTORY_PATH", inventory_path)

    dataset_dir = tmp_path / "dataset"
    _write_codes_parquet(
        dataset_dir,
        {
            "LAB//3020564//umol/l": 52_431,
            "MEDICATION//furosemide//Administered": 1_501,
            "DIAGNOSIS//ICD//10//I50": 3,
        },
    )

    summary = mod.export_codes(output_dir=dataset_dir)

    assert summary == {
        "n_codes_total": 3,
        "n_codes_written": 3,
        "n_codes_dropped_for_size": 0,
        "n_bytes": inventory_path.stat().st_size,
    }
    written = json.loads(inventory_path.read_text())
    # Every code present -- exhaustive vocabulary is the whole point -- with
    # counts suppressed, never the real number.
    assert written == {
        "LAB//3020564//umol/l": "52000",
        "MEDICATION//furosemide//Administered": "2000",
        "DIAGNOSIS//ICD//10//I50": "<1000",
    }


def test_export_codes_drops_low_count_entries_when_oversized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression guard for the real risk this step exists to handle.

    At real GEMINI scale (~13-50k codes), the full inventory could exceed
    run.sh's 900 KB commit cap. Below-threshold entries carry no real count
    information (they're all just "<1000") -- exercised here with a budget
    forced tiny enough to trigger without needing tens of thousands of rows.
    """
    mod = _load_module("export_codes")
    inventory_path = tmp_path / "out" / "codes_inventory.json"
    monkeypatch.setattr(mod, "CODES_INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(mod, "_COMMIT_SIZE_BUDGET", 10)

    dataset_dir = tmp_path / "dataset"
    _write_codes_parquet(
        dataset_dir,
        {
            "LAB//3020564//umol/l": 52_431,
            "DIAGNOSIS//ICD//10//I50": 3,
        },
    )

    summary = mod.export_codes(output_dir=dataset_dir)

    assert summary["n_codes_total"] == 2
    assert summary["n_codes_written"] == 1
    assert summary["n_codes_dropped_for_size"] == 1
    written = json.loads(inventory_path.read_text())
    assert written == {"LAB//3020564//umol/l": "52000"}
