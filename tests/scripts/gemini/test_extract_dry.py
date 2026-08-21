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
                "kind": "matview",
                "name": "admdad_subset",
                "row_count": "1000",
                "columns": [{"name": "genc_id", "type": "integer"}],
            },
            {"kind": "view", "name": "some_view", "row_count": "1000", "columns": []},
        ],
    }
    report = mod.build_report(schema)

    assert report["datacut"] == "some_cut"
    assert len(report["tables"]) == 1  # the view is skipped, matview is not
    table = report["tables"][0]
    assert table["name"] == "admdad_subset"
    assert table["row_count"] == "1000"  # 1234 rounded to nearest 1000
    assert table["columns"] == [{"name": "genc_id", "n_null": "<6"}]  # 2 -> suppressed


def test_build_report_skips_null_fraction_check_for_large_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        raise AssertionError(f"should not query a table over the size threshold: {sql}")

    monkeypatch.setattr(mod.db, "query", fake_query)

    schema = {
        "datacut": "some_cut",
        "objects": [
            {
                "kind": "matview",
                "name": "lab_subset",
                "row_count": str(mod.LARGE_TABLE_ROW_THRESHOLD),
                "columns": [{"name": "genc_id", "type": "integer"}],
            }
        ],
    }
    report = mod.build_report(schema)

    table = report["tables"][0]
    assert table["name"] == "lab_subset"
    assert table["row_count"] == str(mod.LARGE_TABLE_ROW_THRESHOLD)
    assert table["columns"] is None


def test_quote_ident_double_quotes_and_escapes_embedded_quotes() -> None:
    mod = _load_module()
    assert mod._quote_ident("Pop2021") == '"Pop2021"'
    assert mod._quote_ident('weird"name') == '"weird""name"'


def test_null_fraction_quotes_a_mixed_case_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real bug: Postgres lowercases unquoted identifiers, so an
    # unquoted mixed-case column like GEMINI's real `Pop2021`
    # (lookup_statcan_v2021) resolves to a different, usually nonexistent,
    # all-lowercase name and raises UndefinedColumn.
    mod = _load_module()
    captured_sql = {}

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        captured_sql["sql"] = sql
        return pd.DataFrame({"n_null": [0]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    mod.null_fraction("lookup_statcan_v2021", "Pop2021")

    assert '"Pop2021"' in captured_sql["sql"]
    assert '"lookup_statcan_v2021"' in captured_sql["sql"]
    # unquoted would silently pass through mypy/tests but fail against a
    # real mixed-case-sensitive database:
    assert "COUNT(Pop2021)" not in captured_sql["sql"]


def test_null_fraction_or_error_recovers_from_a_failing_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One pathological column must not kill the whole extract-dry run --
    # Amrit cannot iterate interactively on the GEMINI node.
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        raise RuntimeError('column "Pop2021" does not exist')

    monkeypatch.setattr(mod.db, "query", fake_query)

    result = mod._null_fraction_or_error("lookup_statcan_v2021", "Pop2021")
    assert result.startswith("error:")
    assert "Pop2021" in result


def test_build_report_records_a_column_error_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        if "COUNT(*) - COUNT(" in sql:
            raise RuntimeError("boom")
        return pd.DataFrame({"n": [10]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    schema = {
        "datacut": "some_cut",
        "objects": [
            {
                "kind": "matview",
                "name": "lookup_statcan_v2021",
                "row_count": "1000",
                "columns": [{"name": "Pop2021", "type": "double precision"}],
            }
        ],
    }
    report = mod.build_report(schema)

    columns = report["tables"][0]["columns"]
    assert columns[0]["name"] == "Pop2021"
    assert columns[0]["n_null"].startswith("error:")


def test_concept_frequencies_returns_code_desc_and_suppressed_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        assert "lab_subset" in sql
        assert "lookup_lab_concept" in sql
        return pd.DataFrame(
            {"code": ["3020891"], "concept_desc": ["Creatinine"], "n": [12345]}
        )

    monkeypatch.setattr(mod.db, "query", fake_query)

    rows = mod.concept_frequencies(
        "lab_subset", "test_type_mapped_omop", "lookup_lab_concept"
    )
    assert rows == [{"code": "3020891", "concept_desc": "Creatinine", "n": "12000"}]


def test_table_date_ranges_handles_null_years(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod, "DATE_COLUMNS_BY_TABLE", {"admdad_subset": ["admission_date_time"]}
    )

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        return pd.DataFrame({"min_year": [None], "max_year": [None]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    ranges = mod.table_date_ranges()
    assert ranges == {
        "admdad_subset": {"admission_date_time": {"min_year": None, "max_year": None}}
    }


def test_hospital_coverage_stringifies_row_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        assert "lookup_data_coverage" in sql
        return pd.DataFrame(
            {
                "data": ["lab_subset"],
                "min_date": ["2015-01-01"],
                "max_date": ["2023-12-31"],
                "hospital_num": [1],
                "additional_info": [None],
            }
        )

    monkeypatch.setattr(mod.db, "query", fake_query)

    rows = mod.hospital_coverage()
    assert rows == [
        {
            "data": "lab_subset",
            "min_date": "2015-01-01",
            "max_date": "2023-12-31",
            "hospital_num": "1",
            "additional_info": None,
        }
    ]


def test_encounters_per_year_suppresses_and_handles_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        return pd.DataFrame({"year": [2020, None], "n": [54321, 2]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    counts = mod.encounters_per_year()
    assert counts == {"2020": "54000", "unknown": "<6"}


def test_lookup_emptiness_uses_exists_not_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "SUSPECT_EMPTY_LOOKUPS", ["lookup_vitals_concept"])

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        assert "EXISTS" in sql
        return pd.DataFrame({"any_rows": [False]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    assert mod.lookup_emptiness() == {"lookup_vitals_concept": True}


def test_render_markdown_includes_tables_and_design_queries() -> None:
    mod = _load_module()
    report = {
        "datacut": "some_cut",
        "tables": [
            {
                "name": "admdad_subset",
                "row_count": "1000",
                "columns": [{"name": "genc_id", "n_null": "<6"}],
            },
            {"name": "lab_subset", "row_count": "659000000", "columns": None},
        ],
        "design_queries": {
            "lab_concept_frequencies": [
                {"code": "123", "concept_desc": "Creatinine", "n": "12000"}
            ],
            "vitals_concept_frequencies": [
                {"code": "456", "concept_desc": None, "n": "8000"}
            ],
            "table_date_ranges": {
                "admdad_subset": {
                    "admission_date_time": {"min_year": 2015, "max_year": 2023}
                }
            },
            "hospital_coverage": [{"data": "lab_subset", "hospital_num": "1"}],
            "encounters_per_year": {"2020": "54000"},
            "lookup_emptiness": {"lookup_vitals_concept": True},
        },
    }
    text = mod.render_markdown(report)
    assert "some_cut" in text
    assert "admdad_subset" in text
    assert "genc_id" in text
    assert "too large" in text.lower()
    assert "Creatinine" in text
    assert "2015" in text and "2023" in text
    assert "54000" in text
    assert "lookup_vitals_concept" in text


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
                        "kind": "matview",
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
    # Keep the design-query surface small and deterministic for this
    # end-to-end test -- each piece is already unit-tested above.
    monkeypatch.setattr(mod, "DATE_COLUMNS_BY_TABLE", {})
    monkeypatch.setattr(mod, "SUSPECT_EMPTY_LOOKUPS", [])

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        if "COUNT(*) - COUNT(" in sql:
            return pd.DataFrame({"n_null": [0]})
        if "lookup_data_coverage" in sql:
            return pd.DataFrame({"data": [], "hospital_num": []})
        if "GROUP BY year" in sql or "year" in sql.lower() and "COUNT" in sql:
            return pd.DataFrame({"year": [2020], "n": [10]})
        if "lookup_lab_concept" in sql or "lookup_vitals_concept" in sql:
            return pd.DataFrame({"code": [], "concept_desc": [], "n": []})
        return pd.DataFrame({"n": [10]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    mod.main()

    assert (tmp_path / "extract_dry.json").exists()
    assert (tmp_path / "extract_dry.md").exists()
    written = json.loads((tmp_path / "extract_dry.json").read_text())
    assert "design_queries" in written
