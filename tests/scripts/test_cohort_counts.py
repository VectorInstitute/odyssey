"""Tests for the aggregate cohort description (scripts/cohort_counts.py).

A tiny MIMIC-shaped MEDS split: 24 subjects, one three-day admission
each, half of them readmitted ten days later, half of them dying, one
ICU admission. Every count the test reads back is either at or above
the suppression threshold by construction, or deliberately below it to
check the marker.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from odyssey.inference.alerts import READMISSION_HORIZONS_HOURS
from scripts import cohort_counts
from scripts.cohort_counts import (
    OUTPUT_KEYS,
    READMISSION_WINDOW_HOURS,
    SUPPRESSED,
    build_report,
    suppress,
)


T0 = datetime(2150, 3, 1)
BIRTH = datetime(2100, 1, 1)
N_SUBJECTS = 24
SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _rows(
    subject: int,
) -> list[tuple[int, datetime | None, str, float | None, int | None]]:
    start = T0 + timedelta(days=subject)
    hadm = 1000 + subject
    rows: list[tuple[int, datetime | None, str, float | None, int | None]] = [
        (subject, None, "GENDER//F" if subject % 2 == 0 else "GENDER//M", None, None),
        (subject, BIRTH, "MEDS_BIRTH", None, None),
        (subject, start, "HOSPITAL_ADMISSION//EW EMER.", None, hadm),
        (subject, start + timedelta(hours=2), "LAB//50912//mg/dL", 1.0, hadm),
        (subject, start + timedelta(days=3), "HOSPITAL_DISCHARGE//HOME", None, hadm),
    ]
    if subject == 5:
        rows.append(
            (subject, start + timedelta(hours=6), "ICU_ADMISSION//MICU", None, hadm)
        )
    if subject % 2 == 0:
        # A second admission ten days after the first one's discharge.
        again = start + timedelta(days=13)
        rows += [
            (subject, again, "HOSPITAL_ADMISSION//EW EMER.", None, hadm + 500),
            (
                subject,
                again + timedelta(days=1),
                "HOSPITAL_DISCHARGE//HOME",
                None,
                hadm + 500,
            ),
        ]
    if subject % 2 == 1:
        rows.append((subject, start + timedelta(days=20), "MEDS_DEATH", None, None))
    return rows


def _shard(subjects: range) -> pl.DataFrame:
    rows = [r for s in subjects for r in _rows(s)]
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "held_out"
    shard_dir.mkdir()
    _shard(range(0, 12)).write_parquet(shard_dir / "0.parquet")
    _shard(range(12, N_SUBJECTS)).write_parquet(shard_dir / "1.parquet")
    return shard_dir


def _keys_in(node: object) -> set[str]:
    """Every dict key anywhere in a JSON-like tree (the export validator's scan)."""
    keys: set[str] = set()
    if isinstance(node, dict):
        keys |= set(node)
        for value in node.values():
            keys |= _keys_in(value)
    elif isinstance(node, list):
        for value in node:
            keys |= _keys_in(value)
    return keys


def test_readmission_window_matches_the_alerts_horizon() -> None:
    assert READMISSION_HORIZONS_HOURS[-1] == READMISSION_WINDOW_HOURS


def test_suppress_marks_small_cells() -> None:
    assert suppress(10) == 10
    assert suppress(9) == SUPPRESSED == "<10"
    assert suppress(0) == "<10"


def test_report_counts_admissions_los_years_sex_age_and_events(split_dir: Path) -> None:
    report = build_report(
        {"held_out": split_dir},
        source="mimic_iv",
        task_set="v2",
        normalize_medications=False,
        with_events=True,
        hospital_parquet=None,
        max_shards=None,
    )
    assert set(report) == set(OUTPUT_KEYS)
    assert report["events_dropped"] == []
    split = report["splits"]["held_out"]
    assert split["n_shards_read"] == split["n_shards_total"] == 2
    assert split["n_subjects"] == N_SUBJECTS
    assert split["n_admissions"] == N_SUBJECTS + N_SUBJECTS // 2
    assert split["n_hospitals"] is None
    assert "not available" in split["hospitals_note"]
    assert split["admission_years"] == {"min": 2150, "max": 2150}
    # 24 three-day stays and 12 one-day stays: median 3, IQR 1 to 3.
    assert split["los_days"]["n"] == 36
    assert split["los_days"]["median"] == 3.0
    assert split["los_days"]["q1"] == 1.0
    assert split["los_days"]["q3"] == 3.0
    assert split["sex"]["n_subjects_with_sex"] == N_SUBJECTS
    assert split["sex"]["counts"] == {"F": 12, "M": 12}
    assert 50.0 <= split["age_years"]["median"] <= 50.2
    events = split["events"]
    assert set(events) == {
        "vasopressor_start",
        "icu_admission",
        "acute_kidney_injury",
        "death",
        "sepsis3",
        "readmission_30d",
    }
    assert events["death"]["n_subjects_positive"] == 12
    assert events["death"]["prevalence_per_subject"] == 0.5
    assert events["death"]["n_admissions_positive"] is None  # subject-scoped
    assert events["readmission_30d"]["n_subjects_positive"] == 12
    assert events["readmission_30d"]["n_admissions_positive"] == 12
    assert events["readmission_30d"]["prevalence_per_admission"] == round(12 / 36, 4)
    # One ICU admission: the count and its rate are both suppressed.
    assert events["icu_admission"]["n_subjects_positive"] == "<10"
    assert events["icu_admission"]["prevalence_per_subject"] is None
    assert events["acute_kidney_injury"]["n_subjects_positive"] == "<10"
    # The pooled block equals the single split here.
    assert report["all_splits"]["n_subjects"] == N_SUBJECTS
    assert not _keys_in(report) & {"subject_id", "subject_ids", "rows", "per_subject"}


def test_hospitals_come_from_the_metadata_table(
    split_dir: Path, tmp_path: Path
) -> None:
    table = tmp_path / "hadm_id_hospital.parquet"
    pl.DataFrame(
        {
            "hadm_id": [1000 + s for s in range(N_SUBJECTS)],
            "hospital_num": [101 if s < 12 else 202 for s in range(N_SUBJECTS)],
        }
    ).write_parquet(table)
    report = build_report(
        {"held_out": split_dir},
        source="mimic_iv",
        task_set="v1",
        normalize_medications=False,
        with_events=False,
        hospital_parquet=table,
        max_shards=1,
    )
    assert report["hospital_metadata"] == "hadm_id_hospital.parquet"
    assert report["max_shards"] == 1
    split = report["splits"]["held_out"]
    assert split["n_shards_read"] == 1 and split["n_shards_total"] == 2
    assert split["n_hospitals"] == 1  # shard 0 holds subjects 0-11: only hospital 101
    assert split["events"] is None
    assert report["events"] == []


def test_small_split_is_suppressed_end_to_end(tmp_path: Path) -> None:
    shard_dir = tmp_path / "tiny"
    shard_dir.mkdir()
    _shard(range(0, 4)).write_parquet(shard_dir / "0.parquet")
    report = build_report(
        {"tuning": shard_dir},
        source="mimic_iv",
        task_set="v1",
        normalize_medications=False,
        with_events=True,
        hospital_parquet=None,
        max_shards=None,
    )
    split = report["splits"]["tuning"]
    assert split["n_subjects"] == "<10"
    assert split["n_admissions"] == "<10"
    assert split["admission_years"] is None
    assert split["los_days"] is None
    assert split["age_years"] is None
    assert split["sex"]["counts"] == {"F": "<10", "M": "<10"}
    assert split["events"]["death"]["n_subjects_positive"] == "<10"
    assert split["events"]["death"]["prevalence_per_subject"] is None


def test_gemini_source_without_sex_or_birth_reports_not_available(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "gemini"
    shard_dir.mkdir()
    start = T0
    rows = []
    for s in range(12):
        hadm = 7000 + s
        rows += [
            (s, start, "ADMISSION", None, hadm),
            (s, start + timedelta(hours=1), "VITALS//3027018//", 80.0, hadm),
            (s, start + timedelta(days=2), "DISCHARGE", None, hadm),
        ]
    pl.DataFrame(rows, schema=SCHEMA, orient="row").write_parquet(
        shard_dir / "shard_0000.parquet"
    )
    report = build_report(
        {"train": shard_dir},
        source="gemini",
        task_set="v3",
        normalize_medications=False,
        with_events=True,
        hospital_parquet=None,
        max_shards=None,
    )
    assert report["events_dropped"] == ["sepsis3"]
    split = report["splits"]["train"]
    assert split["n_subjects"] == 12
    assert split["sex"] is None and "not available" in split["sex_note"]
    assert split["age_years"] is None and "not available" in split["age_note"]
    assert split["los_days"]["median"] == 2.0


def test_main_reads_the_run_config_for_source_and_splits(
    split_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "source": "mimic_iv",
                "task_set": "v1",
                "normalize_medications": False,
                "train_shard_dir": str(split_dir),
                "tuning_shard_dir": str(tmp_path / "missing"),
            }
        )
    )
    out = tmp_path / "cohort_counts.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cohort_counts",
            "--run-dir",
            str(run_dir),
            "--split",
            f"held_out={split_dir}",
            "--no-events",
            "--output-json",
            str(out),
        ],
    )
    cohort_counts.main()
    report = json.loads(out.read_text())
    assert report["source"] == "mimic_iv" and report["task_set"] == "v1"
    assert set(report["splits"]) == {"train", "held_out"}  # missing tuning dir skipped
    assert report["all_splits"]["n_subjects"] == 2 * N_SUBJECTS


def test_main_requires_a_source_without_a_run_dir(
    split_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cohort_counts",
            "--split",
            f"x={split_dir}",
            "--output-json",
            str(tmp_path / "o.json"),
        ],
    )
    with pytest.raises(SystemExit, match="--source"):
        cohort_counts.main()
