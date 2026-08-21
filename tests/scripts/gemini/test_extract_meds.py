"""Tests for scripts/gemini/extract_meds.py.

No real database or filesystem beyond ``tmp_path`` is used --
``odyssey.data.gemini.db.query``/``stream_query``/``copy_to_sink`` are
monkeypatched, matching every other ``tests/scripts/gemini/test_*.py``.
"""

import importlib.util
import queue
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)
pytest.importorskip("pyarrow", reason=_SKIP_REASON)
pytest.importorskip("polars", reason=_SKIP_REASON)

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3] / "scripts" / "gemini" / "extract_meds.py"
    )
    spec = importlib.util.spec_from_file_location("extract_meds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_stream_table(
    by_table: dict[str, list[pl.DataFrame]],
) -> Callable[[str, list[str]], Iterator[pl.DataFrame]]:
    """Build a fake ``_stream_table`` that yields fixture chunks per table.

    Matches every ``extract_<table>``/index-building function's contract
    (real ``_stream_table`` is unordered and no longer takes
    ``resume_from`` -- see the module docstring's "Fetch strategy") without
    going through the ``COPY``/cursor machinery those functions are tested
    separately.
    """

    def fake(table: str, _select_cols: list[str]) -> Iterator[pl.DataFrame]:
        yield from by_table.get(table, [])

    return fake


# --- small helpers -----------------------------------------------------


def test_quote_ident_double_quotes_and_escapes() -> None:
    mod = _load_module()
    assert mod._quote_ident("Pop2021") == '"Pop2021"'
    assert mod._quote_ident('a"b') == '"a""b"'


def test_parse_datetime_series_handles_valid_missing_and_garbage() -> None:
    mod = _load_module()
    raw = pl.Series(["2020-01-15 08:30:00", None, "not a date at all"])
    parsed = mod._parse_datetime_series(raw)
    assert parsed.to_list() == [pd.Timestamp("2020-01-15 08:30:00"), None, None]


def test_within_admission_guard_mask_rejects_the_real_9022_outlier() -> None:
    # The actual outlier docs/gemini_extraction.md documents: a real
    # pharmacy_subset.med_start_date_time year of 9022 next to a real
    # admission in 2020.
    mod = _load_module()
    ts = pl.Series(
        [
            pd.Timestamp("2020-03-05"),
            pd.Timestamp("9022-01-01"),
            None,
            pd.Timestamp("2020-03-05"),
        ]
    )
    admission = pl.Series(
        [
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-03-01"),
            None,
        ]
    )
    mask = mod._within_admission_guard_mask(ts, admission)
    assert mask.to_list() == [True, False, False, False]


def test_normalize_unit_series_strips_lowercases_and_falls_back_to_unk() -> None:
    mod = _load_module()
    raw = pl.Series([" Umol/L ", None, "", "   "])
    result = mod._normalize_unit_series(raw)
    assert result.to_list() == ["umol/l", "UNK", "UNK", "UNK"]


def test_normalize_unit_series_maps_every_sentinel_string_case_insensitively() -> None:
    # Real strings from scripts/gemini/out/extract_dry.md's unit samples --
    # each is a real, high-frequency "no unit recorded" value, not a guess.
    mod = _load_module()
    raw = ["None", "NULL", "null", "(null)", "nan", "NaN", "", "   "]
    result = mod._normalize_unit_series(pl.Series(raw))
    assert result.to_list() == ["UNK"] * len(raw)


def test_normalize_unit_series_collapses_the_x10e9_family() -> None:
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
    result = mod._normalize_unit_series(pl.Series(variants))
    assert result.to_list() == ["x10e9/l"] * len(variants)


def test_normalize_unit_series_collapses_the_x10e6_family_separately_from_x10e9() -> (
    None
):
    mod = _load_module()
    variants = ["x10E6/L", "X 10^6/L", "x 10^6/L", "x10 6/L"]
    result = mod._normalize_unit_series(pl.Series(variants))
    assert result.to_list() == ["x10e6/l"] * len(variants)
    # A different magnitude -- must never collide with the x10^9/L token.
    both = mod._normalize_unit_series(pl.Series(["x10^9/L", "x10E6/L"]))
    assert both[0] != both[1]


def test_normalize_unit_series_collapses_the_x10e12_family_separately_from_others() -> (
    None
):
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
    result = mod._normalize_unit_series(pl.Series(variants))
    assert result.to_list() == ["x10e12/l"] * len(variants)
    tokens = set(
        mod._normalize_unit_series(pl.Series(["x10^9/L", "x10E6/L", "x10^12/L"]))
    )
    assert tokens == {"x10e9/l", "x10e6/l", "x10e12/l"}


def test_normalize_unit_series_fixes_the_real_mmhd_typo() -> None:
    # ~3.5M vitals rows carry this exact typo for systolic BP.
    mod = _load_module()
    result = mod._normalize_unit_series(pl.Series(["mmHd", "mmHg"]))
    assert result.to_list() == ["mmhg", "mmhg"]


def test_normalize_unit_series_collapses_the_100wbc_family() -> None:
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
    result = mod._normalize_unit_series(pl.Series(variants))
    assert result.to_list() == ["/100wbc"] * len(variants)


def test_normalize_unit_series_collapses_the_cv_family() -> None:
    mod = _load_module()
    result = mod._normalize_unit_series(pl.Series(["%CV", "CV", "% cv"]))
    assert result.to_list() == ["%cv", "%cv", "%cv"]


# --- _CopyChunkSink / _stream_table_copy / _stream_table_cursor / _stream_table ------


def test_copy_chunk_sink_chunks_and_flushes_the_remainder() -> None:
    mod = _load_module()
    out_queue: "queue.Queue[pl.DataFrame]" = queue.Queue()
    sink = mod._CopyChunkSink(
        chunk_rows=2, out_queue=out_queue, stop_requested=threading.Event()
    )
    csv_text = "a,b\n1,x\n2,y\n3,z\n4,\\N\n5,w\n"
    data = csv_text.encode()
    # Feed it in small, arbitrary byte boundaries, including mid-line --
    # this is exactly what psycopg2's real write() calls look like.
    for i in range(0, len(data), 5):
        sink.write(data[i : i + 5])
    sink.close()

    frames = []
    while not out_queue.empty():
        frames.append(out_queue.get())

    assert len(frames) == 3  # two full 2-row chunks + one 1-row remainder
    result = pl.concat(frames)
    assert result["a"].to_list() == [1, 2, 3, 4, 5]
    # NULL '\N' marker, not the CSV-default empty-field convention -- a
    # real empty string must never be conflated with a real NULL.
    assert result["b"][3] is None


def test_copy_chunk_sink_distinguishes_null_marker_from_empty_string() -> None:
    mod = _load_module()
    out_queue: "queue.Queue[pl.DataFrame]" = queue.Queue()
    sink = mod._CopyChunkSink(
        chunk_rows=10, out_queue=out_queue, stop_requested=threading.Event()
    )
    sink.write(b'a,b\n1,\\N\n2,""\n')
    sink.close()
    result = out_queue.get()
    assert result["b"].to_list() == [None, ""]


def test_copy_chunk_sink_write_raises_once_stop_is_requested() -> None:
    mod = _load_module()
    out_queue: "queue.Queue[pl.DataFrame]" = queue.Queue()
    stop_requested = threading.Event()
    sink = mod._CopyChunkSink(
        chunk_rows=10, out_queue=out_queue, stop_requested=stop_requested
    )
    stop_requested.set()
    with pytest.raises(mod._StreamAbandonedError):
        sink.write(b"a,b\n1,x\n")


def test_stream_table_copy_streams_all_rows_via_fake_copy_to_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def fake_copy_to_sink(sql: str, sink: Any) -> None:
        assert "lab_subset" in sql
        assert "ORDER BY" not in sql  # unordered -- see the module docstring
        csv = b"genc_id,x\n1,10\n2,20\n3,30\n"
        for i in range(0, len(csv), 7):
            sink.write(csv[i : i + 7])

    monkeypatch.setattr(mod.db, "copy_to_sink", fake_copy_to_sink)

    chunks = list(mod._stream_table_copy("lab_subset", ["genc_id", "x"], chunk_rows=2))
    total = pl.concat(chunks)
    assert sorted(total["genc_id"].to_list()) == [1, 2, 3]


def test_stream_table_copy_propagates_a_producer_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def failing_copy_to_sink(sql: str, sink: Any) -> None:
        raise RuntimeError("permission denied for COPY")

    monkeypatch.setattr(mod.db, "copy_to_sink", failing_copy_to_sink)

    with pytest.raises(RuntimeError, match="permission denied"):
        list(mod._stream_table_copy("lab_subset", ["genc_id"], chunk_rows=2))


def test_stream_table_cursor_has_no_order_by(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    captured = {}

    def fake_stream_query(
        sql: str, params: Any = None, chunksize: int = 100_000
    ) -> Any:
        captured["sql"] = sql
        yield pd.DataFrame({"genc_id": [1], "x": [10]})

    monkeypatch.setattr(mod.db, "stream_query", fake_stream_query)

    chunks = list(mod._stream_table_cursor("lab_subset", ["genc_id", "x"]))
    assert len(chunks) == 1
    assert "ORDER BY" not in captured["sql"]
    assert "lab_subset" in captured["sql"]


def test_stream_table_falls_back_to_cursor_when_copy_fails_before_yielding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def failing_copy_to_sink(sql: str, sink: Any) -> None:
        raise RuntimeError("no COPY perms")

    def fake_stream_query(
        sql: str, params: Any = None, chunksize: int = 100_000
    ) -> Any:
        yield pd.DataFrame({"genc_id": [9]})

    monkeypatch.setattr(mod.db, "copy_to_sink", failing_copy_to_sink)
    monkeypatch.setattr(mod.db, "stream_query", fake_stream_query)

    chunks = list(mod._stream_table("admdad_subset", ["genc_id"]))
    assert len(chunks) == 1
    assert chunks[0]["genc_id"].to_list() == [9]


def test_stream_table_copy_raises_after_yielding_a_real_chunk_on_a_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failure after the queue has already delivered a real chunk must
    # still surface once the generator is drained -- exercised directly on
    # _stream_table_copy (with a small chunk_rows so a real flush happens
    # before the failure), separately from _stream_table's fallback
    # boundary (see the next test).
    mod = _load_module()

    def flaky_copy_to_sink(sql: str, sink: Any) -> None:
        sink.write(b"genc_id\n1\n")  # flushes as its own chunk_rows=1 chunk
        raise RuntimeError("connection dropped mid-stream")

    monkeypatch.setattr(mod.db, "copy_to_sink", flaky_copy_to_sink)

    gen = mod._stream_table_copy("admdad_subset", ["genc_id"], chunk_rows=1)
    first = next(gen)
    assert first["genc_id"].to_list() == [1]
    with pytest.raises(RuntimeError, match="connection dropped"):
        next(gen)


def test_stream_table_copy_closing_early_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real incident this guards against: if a caller elsewhere abandons a
    # _stream_table_copy generator before it's exhausted (e.g. run_extraction
    # raises for an unrelated reason mid-table), the old code's unconditional
    # `producer.join()` in the generator's `finally` could hang forever --
    # the producer thread ends up blocked on a full out_queue.put() with no
    # consumer left to drain it. Write enough rows to guarantee the producer
    # blocks on a full queue (_STREAM_QUEUE_MAXSIZE=4, one row per chunk),
    # consume exactly one chunk, then close the generator early and assert
    # that completes promptly instead of hanging.
    mod = _load_module()

    def fake_copy_to_sink(sql: str, sink: Any) -> None:
        for i in range(1, 21):  # far more rows than the queue can hold
            sink.write(f"genc_id\n{i}\n".encode() if i == 1 else f"{i}\n".encode())

    monkeypatch.setattr(mod.db, "copy_to_sink", fake_copy_to_sink)

    gen = mod._stream_table_copy("admdad_subset", ["genc_id"], chunk_rows=1)
    first = next(gen)
    assert first["genc_id"].to_list() == [1]

    result: list[str] = []

    def _close_it() -> None:
        gen.close()
        result.append("closed")

    closer = threading.Thread(target=_close_it, daemon=True)
    closer.start()
    closer.join(timeout=10)
    assert result == ["closed"], (
        "gen.close() did not return -- producer/consumer deadlocked"
    )


def test_stream_table_does_not_fall_back_once_copy_has_yielded_real_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failure after the COPY path has already produced real chunks must
    # surface, not silently restart from the cursor path (which would
    # double-count everything already yielded) -- see the module
    # docstring's "Fetch strategy" section.
    mod = _load_module()

    def flaky_stream_table_copy(
        table: str, select_cols: list[str], *, chunk_rows: int = 1
    ) -> Iterator[pl.DataFrame]:
        yield pl.DataFrame({"genc_id": [1]})
        raise RuntimeError("connection dropped mid-stream")

    cursor_called = []

    def fake_stream_table_cursor(
        table: str, select_cols: list[str], *, chunk_rows: int = 1
    ) -> Iterator[pl.DataFrame]:
        cursor_called.append(True)
        yield pl.DataFrame({"genc_id": [9]})

    monkeypatch.setattr(mod, "_stream_table_copy", flaky_stream_table_copy)
    monkeypatch.setattr(mod, "_stream_table_cursor", fake_stream_table_cursor)

    with pytest.raises(RuntimeError, match="connection dropped"):
        list(mod._stream_table("admdad_subset", ["genc_id"]))
    assert cursor_called == []


# --- _filter_valid_genc_id -----------------------------------------------


def test_filter_valid_genc_id_drops_unparsable_rows_not_the_whole_batch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Real incident this guards against: a hard pl.col("genc_id").cast(pl.Int64)
    # raises and crashes the whole chunk over one malformed join key -- and,
    # combined with the (separately fixed) producer-thread deadlock, a crash
    # deep in a chunk could hang the whole extraction silently instead of
    # surfacing a traceback. genc_id must be handled the same way every
    # other messy field in this module already is: drop the bad row, keep
    # going, log loudly.
    mod = _load_module()
    chunk = pl.DataFrame({"genc_id": ["1", "not-a-genc-id", "3"], "x": [10, 20, 30]})

    with caplog.at_level("WARNING"):
        result = mod._filter_valid_genc_id(chunk, "admdad_subset")

    assert result["genc_id"].to_list() == [1, 3]
    assert result["x"].to_list() == [10, 30]
    assert any("not-a-genc-id" in r.message for r in caplog.records)
    assert any("admdad_subset" in r.message for r in caplog.records)


def test_filter_valid_genc_id_drops_null_genc_id_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A genuinely null genc_id (not a parse failure) is dropped the same
    # way -- it can't join against anything either -- but isn't itself
    # evidence of a data-format problem worth a warning.
    mod = _load_module()
    chunk = pl.DataFrame({"genc_id": ["1", None, "3"], "x": [10, 20, 30]})

    with caplog.at_level("WARNING"):
        result = mod._filter_valid_genc_id(chunk, "admdad_subset")

    assert result["genc_id"].to_list() == [1, 3]
    assert caplog.records == []


# --- fetch_admission_index / fetch_lab_concept_lookup -------------------


def test_fetch_admission_index_reads_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    chunk1 = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "patient_id_hashed": ["pA", "pB"],
            "admission_date_time": ["2020-01-01", "2020-02-01"],
        }
    )
    chunk2 = pl.DataFrame(
        {
            "genc_id": [3],
            "patient_id_hashed": ["pC"],
            "admission_date_time": ["2020-03-01"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk1, chunk2]})
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
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "admission_date_time": ["2020-01-01"],
            "discharge_date_time": ["2020-01-05"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_admissions(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert set(rows["code"]) == {"ADMISSION", "DISCHARGE"}
    assert (rows["subject_id"] == "pA").all()
    assert (rows["hadm_id"] == 1).all()
    assert rows["numeric_value"].isna().all()


def test_extract_icu_only_extracts_icu_flagged_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "scu_admit_date_time": ["2020-01-01", "2020-01-01"],
            "scu_discharge_date_time": ["2020-01-05", "2020-01-05"],
            "icu_flag": [True, False],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"ipscu_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_icu(subject_by_genc)], ignore_index=True
    )

    assert set(rows["code"]) == {"ICU_ADMISSION", "ICU_DISCHARGE"}
    assert (rows["subject_id"] == "pA").all()


def test_extract_labs_drops_rows_with_no_mapped_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    lab_concepts = {3020564: "Creatinine"}  # only this one is "real"
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 1],
            "test_type_mapped_omop": [3020564, 999999],  # second is unmapped
            "result_value": ["1.2", "5.0"],
            "result_unit": ["umol/L", "mg/dL"],
            "collection_date_time": ["2020-01-02 08:00:00", "2020-01-02 09:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"lab_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_labs(subject_by_genc, lab_concepts)],
        ignore_index=True,
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
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "test_type_mapped_omop": [3020564, 3020564],
            "result_value": ["70.0", "1.1"],
            "result_unit": [" umol/L ", None],  # whitespace/case, then missing
            "collection_date_time": ["2020-01-02 08:00:00", "2020-01-02 09:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"lab_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_labs(subject_by_genc, lab_concepts)],
        ignore_index=True,
    )

    codes = sorted(rows["code"])
    assert codes == ["LAB//3020564//UNK", "LAB//3020564//umol/l"]


def test_extract_vitals_carries_the_literal_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "measurement_mapped_omop": [3027018],
            "measurement_value": ["88"],
            "measurement_unit": ["bpm"],
            "measure_date_time": ["2020-01-02 08:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"vitals_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_vitals(subject_by_genc)], ignore_index=True
    )

    assert rows.iloc[0]["code"] == "VITALS//3027018//bpm"


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
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "med_id_generic_name_raw": ["acetaminophen", "ibuprofen"],
            "med_start_date_time": ["2020-01-02 08:00:00", "9022-01-01 00:00:00"],
            "med_end_date_time": ["2020-01-03 08:00:00", "8186-01-01 00:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"pharmacy_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_pharmacy(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    # Only patient pA's real-year events survive; pB's insane years are
    # dropped by the guard, not extracted as nonsense events.
    assert set(rows["subject_id"]) == {"pA"}
    assert set(rows["code"]) == {
        "MEDICATION//acetaminophen//started",
        "MEDICATION//acetaminophen//ended",
    }


def test_extract_diagnoses_uses_encounter_discharge_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admdad_chunk = pl.DataFrame({"genc_id": [1], "discharge_date_time": ["2020-01-05"]})
    diag_chunk = pl.DataFrame({"genc_id": [1, 1], "diagnosis_code": ["A01", None]})
    monkeypatch.setattr(
        mod,
        "_stream_table",
        _fake_stream_table(
            {"admdad_subset": [admdad_chunk], "ipdiagnosis_subset": [diag_chunk]}
        ),
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_diagnoses(subject_by_genc)], ignore_index=True
    )

    assert len(rows) == 1  # the null diagnosis_code row is dropped
    assert rows.iloc[0]["code"] == "DIAGNOSIS//A01"
    assert rows.iloc[0]["time"] == pd.Timestamp("2020-01-05")


def test_extract_procedures_namespaces_the_raw_cci_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "intervention_code": ["1.AB.10"],
            "intervention_episode_start_date_time": ["2020-01-02 08:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"ipintervention_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_procedures(subject_by_genc)], ignore_index=True
    )

    assert rows.iloc[0]["code"] == "PROCEDURE//1.AB.10"


def test_extract_radiology_applies_the_admission_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    admission_by_genc = {
        1: pd.Timestamp("2021-05-01"),
        2: pd.Timestamp("2021-05-01"),
    }
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "modality_mapped": ["CT", "XR"],
            "body_part_mapped": ["Head", "Chest"],
            "performed_date_time": ["2021-05-02 10:00:00", "9999-12-31 00:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"radiology_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_radiology(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "IMAGING//CT//Head"


def test_extract_radiology_falls_back_to_unknown_for_missing_modality_or_body_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2021-05-01")}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "modality_mapped": [None],
            "body_part_mapped": [None],
            "performed_date_time": ["2021-05-02 10:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"radiology_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_radiology(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert rows.iloc[0]["code"] == "IMAGING//UNKNOWN//UNKNOWN"


def test_extract_providers_skips_nulls_and_namespaces_by_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01"), 2: pd.Timestamp("2020-02-01")}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "mrp_cpso_hashed": ["hashA", None],
            "adm_phy_cpso_hashed": ["hashB", "hashC"],
            "dis_phy_cpso_hashed": [None, "hashD"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"physicians_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_providers(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    # 2 rows x 3 roles, minus the 2 null hashes -- nulls are dropped, not
    # extracted as empty/placeholder events.
    assert len(rows) == 4
    codes = sorted(rows["code"])
    assert codes == [
        "PROVIDER//ADMITTING//hashB",
        "PROVIDER//ADMITTING//hashC",
        "PROVIDER//DISCHARGING//hashD",
        "PROVIDER//MRP//hashA",
    ]
    # No event-level timestamp on physicians_subset -- attributed to the
    # encounter's admission time, same convention as extract_diagnoses's
    # discharge-time attribution.
    subject_a_rows = rows[rows["subject_id"] == "pA"]
    assert (subject_a_rows["time"] == pd.Timestamp("2020-01-01")).all()


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


# --- resume manifest / run_extraction -------------------------------------


def test_manifest_round_trips_and_is_absent_for_a_fresh_run(tmp_path: Path) -> None:
    mod = _load_module()
    assert mod._load_manifest(tmp_path) == {}
    mod._save_manifest(tmp_path, {"admdad_subset": "complete"})
    assert mod._load_manifest(tmp_path) == {"admdad_subset": "complete"}


def test_run_extraction_skips_tables_already_marked_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "count_distinct_subjects", lambda: 1)
    monkeypatch.setattr(
        mod.db,
        "query",
        lambda sql, params=None: pd.DataFrame({"concept_id": [], "concept_desc": []}),
    )

    admdad_chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "patient_id_hashed": ["pA"],
            "admission_date_time": ["2020-01-01"],
            "discharge_date_time": ["2020-01-05"],
        }
    )
    calls: dict[str, int] = {}

    def fake_stream_table(
        table: str, _select_cols: list[str]
    ) -> Iterator[pl.DataFrame]:
        calls[table] = calls.get(table, 0) + 1
        if table == "admdad_subset":
            yield admdad_chunk

    monkeypatch.setattr(mod, "_stream_table", fake_stream_table)

    # Every table except admdad_subset already "complete" -- only
    # admdad_subset's own extractor (not fetch_admission_index's separate,
    # always-rebuilt pass over the same table) should still run.
    manifest = dict.fromkeys(
        [
            "ipscu_subset",
            "lab_subset",
            "vitals_subset",
            "pharmacy_subset",
            "ipdiagnosis_subset",
            "ipintervention_subset",
            "radiology_subset",
        ],
        "complete",
    )
    mod._save_manifest(tmp_path, manifest)
    calls.clear()

    summary = mod.run_extraction(output_dir=tmp_path)

    assert summary["rows_per_table"].keys() == {"admdad_subset"}
    final_manifest = mod._load_manifest(tmp_path)
    assert final_manifest["admdad_subset"] == "complete"


def test_run_extraction_resumed_run_does_not_duplicate_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "count_distinct_subjects", lambda: 1)
    monkeypatch.setattr(
        mod.db,
        "query",
        lambda sql, params=None: pd.DataFrame(
            {"concept_id": [3020564], "concept_desc": ["Creatinine"]}
        ),
    )

    fixtures = {
        "admdad_subset": [
            pl.DataFrame(
                {
                    "genc_id": [1],
                    "patient_id_hashed": ["pA"],
                    "admission_date_time": ["2020-01-01"],
                    "discharge_date_time": ["2020-01-05"],
                }
            )
        ],
        "lab_subset": [
            pl.DataFrame(
                {
                    "genc_id": [1],
                    "test_type_mapped_omop": [3020564],
                    "result_value": ["1.2"],
                    "result_unit": ["umol/L"],
                    "collection_date_time": ["2020-01-02 08:00:00"],
                }
            )
        ],
    }
    monkeypatch.setattr(mod, "_stream_table", _fake_stream_table(fixtures))

    summary1 = mod.run_extraction(output_dir=tmp_path)
    written_after_first = pd.read_parquet(tmp_path / "shard_0000.parquet")

    summary2 = mod.run_extraction(output_dir=tmp_path)
    written_after_second = pd.read_parquet(tmp_path / "shard_0000.parquet")

    assert len(written_after_first) == len(written_after_second)
    assert summary1["n_subjects"] == summary2["n_subjects"]
