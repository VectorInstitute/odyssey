"""Tests for scripts/gemini/extract_meds.py.

No real database or filesystem beyond ``tmp_path`` is used --
``odyssey.data.gemini.db.query`` is monkeypatched, matching every other
``tests/scripts/gemini/test_*.py``.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)
pytest.importorskip("pyarrow", reason=_SKIP_REASON)

import pandas as pd  # noqa: E402


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3] / "scripts" / "gemini" / "extract_meds.py"
    )
    spec = importlib.util.spec_from_file_location("extract_meds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sequenced_fake_query(
    by_table: dict[str, list[pd.DataFrame]],
) -> Callable[[str, object], pd.DataFrame]:
    """Build a fake ``db.query`` that returns one fixture batch per call, per table.

    Matches the source table name in the SQL text (the only thing this
    module's queries reliably contain) and pops the next fixture batch for
    it; once a table's list is exhausted, returns an empty frame with the
    same columns as the last batch, so ``_paginate_rows`` terminates the
    same way a real "no more rows" response would.
    """
    remaining = {table: list(batches) for table, batches in by_table.items()}
    empty_cols: dict[str, list[str]] = {
        table: list(batches[0].columns) if batches else []
        for table, batches in by_table.items()
    }

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        for table, batches in remaining.items():
            if f'"{table}"' in sql:
                if batches:
                    return batches.pop(0)
                return pd.DataFrame(columns=empty_cols[table])
        raise AssertionError(f"no fixture registered for query: {sql}")

    return fake_query


# --- small helpers -----------------------------------------------------


def test_quote_ident_double_quotes_and_escapes() -> None:
    mod = _load_module()
    assert mod._quote_ident("Pop2021") == '"Pop2021"'
    assert mod._quote_ident('a"b') == '"a""b"'


def test_parse_gemini_datetime_handles_valid_missing_and_garbage() -> None:
    mod = _load_module()
    assert mod._parse_gemini_datetime("2020-01-15 08:30:00") == pd.Timestamp(
        "2020-01-15 08:30:00"
    )
    assert mod._parse_gemini_datetime(None) is None
    assert mod._parse_gemini_datetime(float("nan")) is None
    assert mod._parse_gemini_datetime("not a date at all") is None


def test_within_admission_guard_rejects_the_real_9022_outlier() -> None:
    # The actual outlier docs/gemini_extraction.md documents: a real
    # pharmacy_subset.med_start_date_time year of 9022 next to a real
    # admission in 2020.
    mod = _load_module()
    admission = pd.Timestamp("2020-03-01")
    within = pd.Timestamp("2020-03-05")
    outlier = pd.Timestamp("9022-01-01")

    assert mod._within_admission_guard(within, admission) is True
    assert mod._within_admission_guard(outlier, admission) is False
    assert mod._within_admission_guard(None, admission) is False
    assert mod._within_admission_guard(within, None) is False


def test_parse_numeric_handles_numeric_and_categorical_values() -> None:
    mod = _load_module()
    assert mod._parse_numeric("3.5") == 3.5
    assert mod._parse_numeric("POSITIVE") is None
    assert mod._parse_numeric(None) is None


# --- fetch_admission_index / fetch_lab_concept_lookup -------------------


def test_fetch_admission_index_paginates_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    batch1 = pd.DataFrame(
        {
            "genc_id": [1, 2],
            "patient_id_hashed": ["pA", "pB"],
            "admission_date_time": ["2020-01-01", "2020-02-01"],
            "row_num": [1, 2],
        }
    )
    batch2 = pd.DataFrame(
        {
            "genc_id": [3],
            "patient_id_hashed": ["pC"],
            "admission_date_time": ["2020-03-01"],
            "row_num": [3],
        }
    )
    monkeypatch.setattr(
        mod.db, "query", _sequenced_fake_query({"admdad_subset": [batch1, batch2]})
    )

    subject_by_genc, admission_by_genc = mod.fetch_admission_index()

    assert subject_by_genc == {1: "pA", 2: "pB", 3: "pC"}
    assert admission_by_genc[1] == pd.Timestamp("2020-01-01")
    assert admission_by_genc[3] == pd.Timestamp("2020-03-01")


def test_fetch_lab_concept_lookup_uses_distinct_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real bug from docs/gemini_extraction.md's open question 4:
    # lookup_lab_concept has a duplicate row per concept_id, one with a
    # real description, one NULL -- the query itself must filter/dedupe,
    # not the caller.
    mod = _load_module()
    captured_sql = {}

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        captured_sql["sql"] = sql
        return pd.DataFrame({"concept_id": ["3019550"], "concept_desc": ["Sodium"]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    lookup = mod.fetch_lab_concept_lookup()

    assert lookup == {3019550: "Sodium"}
    assert "DISTINCT ON" in captured_sql["sql"]
    assert "IS NOT NULL" in captured_sql["sql"]


# --- per-table extraction ------------------------------------------------


def test_extract_admissions_yields_admission_and_discharge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    batch = pd.DataFrame(
        {
            "genc_id": [1],
            "admission_date_time": ["2020-01-01"],
            "discharge_date_time": ["2020-01-05"],
            "row_num": [1],
        }
    )
    monkeypatch.setattr(
        mod.db, "query", _sequenced_fake_query({"admdad_subset": [batch]})
    )

    rows = pd.concat(
        list(mod.extract_admissions(subject_by_genc, admission_by_genc)),
        ignore_index=True,
    )

    assert set(rows["code"]) == {"ADMISSION", "DISCHARGE"}
    assert (rows["subject_id"] == "pA").all()
    assert (rows["hadm_id"] == 1).all()


def test_extract_labs_drops_rows_with_no_mapped_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    lab_concepts = {3020564: "Creatinine"}  # only this one is "real"
    batch = pd.DataFrame(
        {
            "genc_id": [1, 1],
            "test_type_mapped_omop": [3020564, 999999],  # second is unmapped
            "result_value": ["1.2", "5.0"],
            "result_unit": ["umol/L", "mg/dL"],
            "collection_date_time": ["2020-01-02 08:00:00", "2020-01-02 09:00:00"],
            "row_num": [1, 2],
        }
    )
    monkeypatch.setattr(mod.db, "query", _sequenced_fake_query({"lab_subset": [batch]}))

    rows = pd.concat(
        list(mod.extract_labs(subject_by_genc, lab_concepts)), ignore_index=True
    )

    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "LAB//3020564//umol/l"
    assert rows.iloc[0]["numeric_value"] == 1.2


def test_extract_labs_carries_the_literal_unit_and_normalizes_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real reason: GEMINI is multi-hospital and the same OMOP concept
    # can carry different units per site -- the unit must ride in the
    # code, and mixed units under one concept must produce different codes.
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    lab_concepts = {3020564: "Creatinine"}
    batch = pd.DataFrame(
        {
            "genc_id": [1, 2],
            "test_type_mapped_omop": [3020564, 3020564],
            "result_value": ["70.0", "1.1"],
            "result_unit": [" umol/L ", None],  # whitespace/case, then missing
            "collection_date_time": ["2020-01-02 08:00:00", "2020-01-02 09:00:00"],
            "row_num": [1, 2],
        }
    )
    monkeypatch.setattr(mod.db, "query", _sequenced_fake_query({"lab_subset": [batch]}))

    rows = pd.concat(
        list(mod.extract_labs(subject_by_genc, lab_concepts)), ignore_index=True
    )

    codes = sorted(rows["code"])
    assert codes == ["LAB//3020564//UNK", "LAB//3020564//umol/l"]


def test_extract_vitals_carries_the_literal_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    batch = pd.DataFrame(
        {
            "genc_id": [1],
            "measurement_mapped_omop": [3027018],
            "measurement_value": ["88"],
            "measurement_unit": ["bpm"],
            "measure_date_time": ["2020-01-02 08:00:00"],
            "row_num": [1],
        }
    )
    monkeypatch.setattr(
        mod.db, "query", _sequenced_fake_query({"vitals_subset": [batch]})
    )

    rows = pd.concat(list(mod.extract_vitals(subject_by_genc)), ignore_index=True)

    assert rows.iloc[0]["code"] == "VITALS//3027018//bpm"


def test_normalize_unit_strips_lowercases_and_falls_back_to_unk() -> None:
    mod = _load_module()
    assert mod._normalize_unit(" Umol/L ") == "umol/l"
    assert mod._normalize_unit(None) == "UNK"
    assert mod._normalize_unit(float("nan")) == "UNK"
    assert mod._normalize_unit("") == "UNK"
    assert mod._normalize_unit("   ") == "UNK"


def test_normalize_unit_maps_every_sentinel_string_case_insensitively() -> None:
    # Real strings from scripts/gemini/out/extract_dry.md's unit samples --
    # each is a real, high-frequency "no unit recorded" value, not a guess.
    mod = _load_module()
    for raw in ["None", "NULL", "null", "(null)", "nan", "NaN", "", "   "]:
        assert mod._normalize_unit(raw) == "UNK", raw


def test_normalize_unit_collapses_the_x10e9_family() -> None:
    # Every one of these is a real result_unit value observed for the
    # same underlying x10^9/L concept (WBC differentials, platelets).
    mod = _load_module()
    variants = [
        "X10 9/L",
        "X10  9/L",  # double space
        "X 10 9/L",
        "x 10^9/L",
        "X 10^9/L",
        "x10^9/L",
        "X10^9/L",
        "x10*9/L",
        "10*9/L",
        "10e9/L",
        "10E9/L",
        "x10e9/L",
        "x10E9/L",
        "E9/L",
    ]
    for raw in variants:
        assert mod._normalize_unit(raw) == "x10e9/l", raw


def test_normalize_unit_collapses_the_x10e6_family_separately_from_x10e9() -> None:
    mod = _load_module()
    for raw in ["x10E6/L", "X 10^6/L", "x 10^6/L", "x10 6/L"]:
        assert mod._normalize_unit(raw) == "x10e6/l", raw
    # A different magnitude -- must never collide with the x10^9/L token.
    assert mod._normalize_unit("x10^9/L") != mod._normalize_unit("x10E6/L")


def test_normalize_unit_collapses_the_x10e12_family_separately_from_others() -> None:
    # Erythrocyte counts -- same fragmentation pattern as x10^9/L, a third
    # distinct magnitude that must never collide with either other family.
    mod = _load_module()
    variants = [
        "x10^12/L",
        "X10^12/L",
        "x10 12/L",
        "X10 12/L",
        "x 10^12/L",
        "X 10^12/L",
        "x10*12/L",
        "10*12/L",
        "10e12/L",
        "x10e12/L",
        "x10E12/L",
        "E12/L",
        "x E12/L",
    ]
    for raw in variants:
        assert mod._normalize_unit(raw) == "x10e12/l", raw
    tokens = {
        mod._normalize_unit("x10^9/L"),
        mod._normalize_unit("x10E6/L"),
        mod._normalize_unit("x10^12/L"),
    }
    assert tokens == {"x10e9/l", "x10e6/l", "x10e12/l"}


def test_normalize_unit_fixes_the_real_mmhd_typo() -> None:
    # ~3.5M vitals rows carry this exact typo for systolic BP.
    mod = _load_module()
    assert mod._normalize_unit("mmHd") == "mmhg"
    assert mod._normalize_unit("mmHg") == "mmhg"


def test_normalize_unit_collapses_the_100wbc_family() -> None:
    mod = _load_module()
    variants = [
        "/100 LKC",
        "/100LKC",
        "/100 WBC",
        "/100(WBCs)",
        "/100WBC",
        "/100 WBC's",
        "/100 WBCs",
    ]
    for raw in variants:
        assert mod._normalize_unit(raw) == "/100wbc", raw


def test_normalize_unit_collapses_the_cv_family() -> None:
    mod = _load_module()
    assert mod._normalize_unit("%CV") == "%cv"
    assert mod._normalize_unit("CV") == "%cv"
    assert mod._normalize_unit("% cv") == "%cv"


def test_extract_pharmacy_applies_the_admission_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real bug this exists for: pharmacy_subset.med_start_date_time
    # has a real value of year 9022 (docs/gemini_extraction.md).
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    admission_by_genc = {
        1: pd.Timestamp("2020-01-01"),
        2: pd.Timestamp("2020-06-01"),
    }
    batch = pd.DataFrame(
        {
            "genc_id": [1, 2],
            "med_id_generic_name_raw": ["acetaminophen", "ibuprofen"],
            "med_start_date_time": ["2020-01-02 08:00:00", "9022-01-01 00:00:00"],
            "med_end_date_time": ["2020-01-03 08:00:00", "8186-01-01 00:00:00"],
            "row_num": [1, 2],
        }
    )
    monkeypatch.setattr(
        mod.db, "query", _sequenced_fake_query({"pharmacy_subset": [batch]})
    )

    rows = pd.concat(
        list(mod.extract_pharmacy(subject_by_genc, admission_by_genc)),
        ignore_index=True,
    )

    # Only patient pA's real-year events survive; pB's insane years are
    # dropped by the guard, not extracted as nonsense events.
    assert set(rows["subject_id"]) == {"pA"}
    assert set(rows["code"]) == {
        "MEDICATION//acetaminophen//started",
        "MEDICATION//acetaminophen//ended",
    }


def test_extract_radiology_applies_the_admission_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    admission_by_genc = {
        1: pd.Timestamp("2021-05-01"),
        2: pd.Timestamp("2021-05-01"),
    }
    batch = pd.DataFrame(
        {
            "genc_id": [1, 2],
            "modality_mapped": ["CT", "XR"],
            "body_part_mapped": ["Head", "Chest"],
            "performed_date_time": ["2021-05-02 10:00:00", "9999-12-31 00:00:00"],
            "row_num": [1, 2],
        }
    )
    monkeypatch.setattr(
        mod.db, "query", _sequenced_fake_query({"radiology_subset": [batch]})
    )

    rows = pd.concat(
        list(mod.extract_radiology(subject_by_genc, admission_by_genc)),
        ignore_index=True,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "IMAGING//CT//Head"


# --- preflight -------------------------------------------------------


def test_count_distinct_subjects_issues_a_count_distinct_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        assert "COUNT(DISTINCT" in sql
        assert '"admdad_subset"' in sql
        return pd.DataFrame({"n": [12345]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    assert mod.count_distinct_subjects() == 12345


def test_preflight_shard_capacity_returns_shard_count_when_within_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    # Pretend the process already has plenty of headroom, regardless of
    # the real limits on whatever machine runs this test.
    monkeypatch.setattr(mod.resource, "getrlimit", lambda _which: (10_000, 10_000))

    n_shards = mod.preflight_shard_capacity(2500, subjects_per_shard=1000)

    assert n_shards == 3  # ceil(2500 / 1000)


def test_preflight_shard_capacity_raises_with_the_exact_ulimit_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    # Both soft and hard limits are too low, and setrlimit can't fix a
    # hard-limited ceiling -- this must fail loudly, not silently proceed
    # into a run that will crash hours in on file descriptor exhaustion.
    monkeypatch.setattr(mod.resource, "getrlimit", lambda _which: (256, 256))
    monkeypatch.setattr(mod.resource, "setrlimit", lambda _which, _limits: None)

    with pytest.raises(RuntimeError, match=r"ulimit -n \d+"):
        mod.preflight_shard_capacity(1_000_000, subjects_per_shard=1000)


# --- sharding and the writer ---------------------------------------------


def test_assign_shards_is_deterministic_and_covers_all_subjects() -> None:
    mod = _load_module()
    subjects = [f"patient-{i}" for i in range(2500)]

    first = mod.assign_shards(subjects, subjects_per_shard=1000)
    second = mod.assign_shards(subjects, subjects_per_shard=1000)

    assert first == second  # deterministic, not Python's salted hash()
    assert set(first) == set(subjects)
    assert len(set(first.values())) == 3  # ceil(2500 / 1000)


def test_assign_shards_at_least_one_shard_for_a_small_universe() -> None:
    mod = _load_module()
    shards = mod.assign_shards(["only-one"], subjects_per_shard=1000)
    assert shards == {"only-one": 0}


def test_meds_shard_writer_streams_batches_to_parquet(tmp_path: Path) -> None:
    mod = _load_module()
    shard_by_subject = {"pA": 0, "pB": 1}
    writer = mod.MedsShardWriter(tmp_path, shard_by_subject)

    batch = pd.DataFrame(
        {
            "subject_id": ["pA", "pB"],
            "time": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
            "code": ["ADMISSION", "ADMISSION"],
            "numeric_value": [None, None],
            "hadm_id": [1, 2],
        }
    )
    writer.write_batch("admdad_subset", batch)
    counts = writer.close()

    assert counts == {0: 1, 1: 1}
    assert (tmp_path / "shard_0000.parquet").exists()
    assert (tmp_path / "shard_0001.parquet").exists()
    written = pd.read_parquet(tmp_path / "shard_0000.parquet")
    assert written["subject_id"].tolist() == ["pA"]
    assert writer.rows_written_per_table == {"admdad_subset": 2}


def test_meds_shard_writer_drops_and_counts_unshardable_rows(tmp_path: Path) -> None:
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    batch = pd.DataFrame(
        {
            "subject_id": ["pA", "unknown-subject"],
            "time": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "code": ["ADMISSION", "ADMISSION"],
            "numeric_value": [None, None],
            "hadm_id": [1, 2],
        }
    )

    writer.write_batch("admdad_subset", batch)
    counts = writer.close()

    assert counts == {0: 1}
    assert writer.rows_dropped_unshardable == 1
