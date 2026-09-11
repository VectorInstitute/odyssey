"""load_code_descriptions: codes.parquet -> readable names, with safe fallbacks."""

from pathlib import Path

import polars as pl

from odyssey.data.code_metadata import load_code_descriptions


def test_reads_non_empty_descriptions(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "code": ["LAB//220045//bpm", "MEDICATION//x", "LAB//y"],
            "description": ["Heart Rate", None, ""],
        }
    ).write_parquet(tmp_path / "codes.parquet")
    assert load_code_descriptions(tmp_path) == {"LAB//220045//bpm": "Heart Rate"}


def test_accepts_a_string_path_and_ignores_extra_columns(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "code": ["DIAGNOSIS//ICD//10//I5021"],
            "description": ["Acute systolic heart failure"],
            "parent_codes": [["ICD10CM/I50.21"]],
        }
    ).write_parquet(tmp_path / "codes.parquet")
    assert load_code_descriptions(str(tmp_path)) == {
        "DIAGNOSIS//ICD//10//I5021": "Acute systolic heart failure"
    }


def test_missing_inputs_give_an_empty_mapping(tmp_path: Path) -> None:
    assert load_code_descriptions(None) == {}
    assert load_code_descriptions(tmp_path) == {}
    pl.DataFrame({"code": ["A"]}).write_parquet(tmp_path / "codes.parquet")
    assert load_code_descriptions(tmp_path) == {}
