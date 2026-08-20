"""Tests for scripts/gemini/extract_dry.py.

No real database is used -- ``odyssey.data.gemini.db.query`` is monkeypatched,
matching ``tests/scripts/gemini/test_explore_schema.py``.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)

import pandas as pd  # noqa: E402


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "gemini" / "extract_dry.py"
    spec = importlib.util.spec_from_file_location("extract_dry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_prints_pending_when_schema_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "SCHEMA_PATH", tmp_path / "schema.json")
    mod.main()
    assert "pending schema report" in capsys.readouterr().out


def test_build_report_skips_views_and_suppresses_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        if "COUNT(*) - COUNT(" in sql:
            return pd.DataFrame({"n_null": [2]})
        if "COUNT(*)" in sql:
            return pd.DataFrame({"n": [1234]})
        raise AssertionError(f"unexpected query: {sql}")

    monkeypatch.setattr(mod.db, "query", fake_query)

    schema = {
        "datacut": "some_cut",
        "objects": [
            {
                "kind": "table",
                "name": "admdad_subset",
                "row_count": "1000",
                "columns": [{"name": "genc_id", "type": "integer"}],
            },
            {"kind": "view", "name": "some_view", "row_count": "1000", "columns": []},
        ],
    }
    report = mod.build_report(schema)

    assert report["datacut"] == "some_cut"
    assert len(report["tables"]) == 1  # the view is skipped
    table = report["tables"][0]
    assert table["name"] == "admdad_subset"
    assert table["row_count"] == "1000"  # 1234 rounded to nearest 1000
    assert table["columns"] == [{"name": "genc_id", "n_null": "<6"}]  # 2 -> suppressed


def test_render_markdown_includes_tables_and_columns() -> None:
    mod = _load_module()
    report = {
        "datacut": "some_cut",
        "tables": [
            {
                "name": "admdad_subset",
                "row_count": "1000",
                "columns": [{"name": "genc_id", "n_null": "<6"}],
            }
        ],
    }
    text = mod.render_markdown(report)
    assert "some_cut" in text
    assert "admdad_subset" in text
    assert "genc_id" in text
    assert "<6" in text


def test_main_writes_report_when_schema_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_module()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "datacut": "some_cut",
                "objects": [
                    {
                        "kind": "table",
                        "name": "admdad_subset",
                        "row_count": "1000",
                        "columns": [{"name": "genc_id", "type": "integer"}],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(mod, "SCHEMA_PATH", schema_path)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        if "COUNT(*) - COUNT(" in sql:
            return pd.DataFrame({"n_null": [0]})
        return pd.DataFrame({"n": [10]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    mod.main()

    assert (tmp_path / "extract_dry.json").exists()
    assert (tmp_path / "extract_dry.md").exists()
