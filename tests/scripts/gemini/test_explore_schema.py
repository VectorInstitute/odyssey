"""Tests for scripts/gemini/explore_schema.py.

No real database is used -- ``odyssey.data.gemini.db.query`` is monkeypatched
to return canned :class:`pandas.DataFrame` objects, matching the fake-
connection approach in ``tests/odyssey/data/gemini/test_db.py``.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)

import pandas as pd  # noqa: E402


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3] / "scripts" / "gemini" / "explore_schema.py"
    )
    spec = importlib.util.spec_from_file_location("explore_schema", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("n", "expected"),
    # round() is round-half-to-even, so 600 (-> 0.6) and 12345 (-> 12.345)
    # are chosen to stay unambiguous rather than landing on a .5 boundary.
    [(0, "<6"), (1, "<6"), (5, "<6"), (6, "0"), (600, "1000"), (12345, "12000")],
)
def test_suppressed_row_count(n: int, expected: str) -> None:
    mod = _load_module()
    assert mod.suppressed_row_count(n) == expected


def test_render_markdown_includes_columns_and_row_count() -> None:
    mod = _load_module()
    report = {
        "datacut": "some_cut",
        "objects": [
            {
                "kind": "table",
                "name": "admdad_subset",
                "row_count": "<6",
                "columns": [{"name": "genc_id", "type": "integer"}],
            }
        ],
    }
    text = mod.render_markdown(report)
    assert "some_cut" in text
    assert "admdad_subset" in text
    assert "<6" in text
    assert "genc_id" in text
    assert "integer" in text


def test_build_schema_report_uses_suppressed_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod.config, "DATACUT", "some_cut")

    def fake_query(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        if "pg_tables" in sql:
            return pd.DataFrame({"kind": ["table"], "name": ["admdad_subset"]})
        if "pg_attribute" in sql:
            return pd.DataFrame({"column": ["genc_id"], "type": ["integer"]})
        if "COUNT(*)" in sql:
            return pd.DataFrame({"n": [3]})
        raise AssertionError(f"unexpected query: {sql}")

    monkeypatch.setattr(mod.db, "query", fake_query)

    report = mod.build_schema_report()

    assert report["datacut"] == "some_cut"
    assert len(report["objects"]) == 1
    obj = report["objects"][0]
    assert obj["kind"] == "table"
    assert obj["name"] == "admdad_subset"
    assert obj["row_count"] == "<6"  # 3 rows -> suppressed
    assert obj["columns"] == [{"name": "genc_id", "type": "integer"}]


def test_main_lists_available_schemata_when_datacut_unset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod.config, "DB_URL", "postgresql+psycopg2://u:p@h:5432/db")
    monkeypatch.setattr(mod.config, "DATACUT", None)
    monkeypatch.setattr(
        mod.db,
        "list_available_schemata",
        lambda: pd.DataFrame({"schema_name": ["cut_a", "cut_b"]}),
    )

    mod.main()

    out = capsys.readouterr().out
    assert "GEMINI_DATACUT is not set" in out
    assert "cut_a" in out
    assert "cut_b" in out
    assert "set gemini_datacut to one of these" in out.lower()
