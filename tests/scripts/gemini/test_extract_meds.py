"""Tests for scripts/gemini/extract_meds.py.

No real database or filesystem beyond ``tmp_path`` is used --
``odyssey.data.gemini.db.query``/``stream_query``/``copy_to_sink`` are
monkeypatched, matching every other ``tests/scripts/gemini/test_*.py``.

**Every fixture chunk fed to an ``extract_<table>``/``fetch_*`` function
must be built via** :func:`_utf8_chunk`, **not a bare ``pl.DataFrame(...)``**
-- real ``_stream_table`` chunks are 100% ``pl.Utf8`` (see
``extract_meds.py``'s module docstring, afc50c4), and a native-Python-typed
fixture exercises a chunk shape that never occurs against the real
database. This is the second real dtype-assumption bug this project has
shipped past its own tests since afc50c4 (the first: ``test_type_mapped_omop``/
``measurement_mapped_omop``; the second: ``extract_death``'s
``discharge_disposition`` cross-check, which crashed the first real
GEMINI re-extract with an ``is_in()`` dtype mismatch that every existing
test's typed fixture papered over). See :func:`_utf8_chunk`'s own
docstring for the exact convention.
"""

import gc
import importlib.util
import queue
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _utf8_chunk(data: dict[str, list]) -> pl.DataFrame:
    """Build a fixture chunk with every column ``pl.Utf8``, like a real one.

    Real ``_stream_table`` chunks are 100% ``pl.Utf8`` (``pl.read_csv``'s
    ``infer_schema_length=0``, see the module docstring's afc50c4 entry) --
    every ``extract_<table>``/``fetch_*`` function does its own explicit
    lenient cast downstream, never relies on an inferred dtype. A fixture
    built from native Python ints/bools instead (plain ``pl.DataFrame(...)``,
    which polars infers a typed column from) exercises a chunk shape that
    never occurs against the real database. That gap is exactly what let a
    real ``is_in()`` dtype-mismatch crash (``discharge_disposition`` arrived
    ``Utf8``, compared against an ``Int64`` literal tuple) ship past every
    ``extract_death`` test and crash the first real re-extract.

    **Every new** ``extract_*``/``fetch_*`` **test must build its fixture
    chunks through this helper**, not a bare ``pl.DataFrame(...)``, unless
    it is deliberately testing a non-Utf8 edge case.

    Parameters
    ----------
    data : dict[str, list]
        Column name -> raw values, exactly as they'd render over the wire
        (e.g. ``7`` for an integer column -- stringified here -- or the
        literal ``"t"``/``"f"`` for a Postgres boolean; pass the exact
        string yourself for anything that doesn't stringify the same way
        Python's ``str()`` would). ``None`` stays ``None`` (SQL ``NULL``).

    Returns
    -------
    polars.DataFrame
        Every column ``pl.Utf8``, matching a real chunk.
    """
    return pl.DataFrame(
        {
            col: [None if v is None else str(v) for v in values]
            for col, values in data.items()
        },
        schema=dict.fromkeys(data, pl.Utf8),
    )


# --- small helpers -----------------------------------------------------


def test_quote_ident_double_quotes_and_escapes() -> None:
    mod = _load_module()
    assert mod._quote_ident("Pop2021") == '"Pop2021"'
    assert mod._quote_ident('a"b') == '"a""b"'


def test_select_expr_sql_wraps_text_columns_with_newline_strip_and_cap() -> None:
    mod = _load_module()
    sql = mod._select_expr_sql("result_value")
    assert sql == (
        "left(regexp_replace(\"result_value\", E'[\\n\\r]+', ' ', 'g'), 128) "
        'AS "result_value"'
    )


def test_select_expr_sql_leaves_confirmed_integer_boolean_columns_bare() -> None:
    # Real crash this guards against being reintroduced: genc_id is
    # integer, and every extract_<table> function's join logic (and every
    # test fixture) assumes it comes back unwrapped.
    mod = _load_module()
    for column in [
        "genc_id",
        "test_type_mapped_omop",
        "measurement_mapped_omop",
        "icu_flag",
    ]:
        assert mod._select_expr_sql(column) == mod._quote_ident(column)


def test_select_expr_sql_covers_every_column_selected_by_the_extractors() -> None:
    # Real incident this guards against: the module used to assume no
    # selected column could contain a newline "by construction" -- wrong,
    # since every one of these is Postgres text/character varying (see
    # scripts/gemini/out/schema.md), which can embed a literal newline
    # regardless of declared length. Every one of them must be wrapped
    # unless it's confirmed integer/boolean.
    mod = _load_module()
    free_text_columns = [
        "patient_id_hashed",
        "admission_date_time",
        "discharge_date_time",
        "scu_admit_date_time",
        "scu_discharge_date_time",
        "result_value",
        "result_unit",
        "collection_date_time",
        "measurement_value",
        "measurement_unit",
        "measure_date_time",
        "med_id_generic_name_raw",
        "med_start_date_time",
        "med_end_date_time",
        "diagnosis_code",
        "intervention_code",
        "intervention_episode_start_date_time",
        "modality_mapped",
        "body_part_mapped",
        "performed_date_time",
        "mrp_cpso_hashed",
        "adm_phy_cpso_hashed",
        "dis_phy_cpso_hashed",
        "registration_date_time",
        "triage_date_time",
        "left_er_date_time",
        "er_diagnosis_code",
        "consult_service_code",
        "consult_request_date_time",
        "institution_to_mns",
        "cmg",
        "hig_code",
    ]
    for column in free_text_columns:
        sql = mod._select_expr_sql(column)
        assert "regexp_replace" in sql
        assert f'AS "{column}"' in sql


def test_parse_datetime_series_handles_valid_missing_and_garbage() -> None:
    mod = _load_module()
    raw = pl.Series(["2020-01-15 08:30:00", None, "not a date at all"])
    parsed = mod._parse_datetime_series(raw)
    assert parsed.to_list() == [pd.Timestamp("2020-01-15 08:30:00"), None, None]


def test_parse_datetime_series_fast_path_matches_pandas_mixed_exactly() -> None:
    # The fast path (polars-native, tried first) must never disagree with
    # the pandas format="mixed" fallback it's meant to shortcut -- both ISO
    # separators, with and without fractional seconds, plus a residue mix
    # (a genuinely different format, garbage, null, empty) that must still
    # fall through to the slow path unchanged.
    mod = _load_module()
    raw = pl.Series(
        [
            "2020-01-15 08:30:00",
            "2020-01-15 08:30:00.123456",
            "2020-01-15T08:30:00",
            "2020-01-15T08:30:00.5",
            None,
            "",
            "not a date at all",
            "01/15/2020 5:30 PM",  # genuinely different format -> residue
        ]
    )
    expected = pd.to_datetime(raw.to_pandas(), errors="coerce", format="mixed")
    parsed = mod._parse_datetime_series(raw)
    assert parsed.to_list() == pl.Series(expected).cast(pl.Datetime("us")).to_list()


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
    out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
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
    # infer_schema_length=0 -- every column comes back Utf8, never
    # auto-inferred; typed parsing is each extract_<table>'s own job.
    assert result["a"].to_list() == ["1", "2", "3", "4", "5"]
    # NULL '\N' marker, not the CSV-default empty-field convention -- a
    # real empty string must never be conflated with a real NULL.
    assert result["b"][3] is None


def test_copy_chunk_sink_distinguishes_null_marker_from_empty_string() -> None:
    mod = _load_module()
    out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
    sink = mod._CopyChunkSink(
        chunk_rows=10, out_queue=out_queue, stop_requested=threading.Event()
    )
    sink.write(b'a,b\n1,\\N\n2,""\n')
    sink.close()
    result = out_queue.get()
    assert result["b"].to_list() == [None, ""]


def test_copy_chunk_sink_never_infers_a_dtype_even_within_one_chunk() -> None:
    """Real crash this guards against.

    pl.read_csv infers each column's dtype from a sample -- a column that's
    all-numeric for the first rows of a chunk locked Float64 for that whole
    chunk, then a genuinely free-text value later in the SAME chunk
    (lab_subset.result_value's real shape: a numeric reading region
    followed by a free-text region, e.g. "103@POST") raised
    polars.ComputeError mid-parse. infer_schema_length=0 means every
    column always comes back Utf8 regardless of what came before it in the
    chunk -- typed parsing is each extract_<table>'s own job downstream.
    """
    mod = _load_module()
    out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
    sink = mod._CopyChunkSink(
        chunk_rows=10, out_queue=out_queue, stop_requested=threading.Event()
    )
    # All-numeric first, then a free-text value -- all one chunk (chunk_rows
    # not reached until close()), exactly the real failure shape.
    csv_rows = "\n".join([f"{i},{i}.5" for i in range(1, 6)] + ["6,103@POST"])
    sink.write(f"genc_id,result_value\n{csv_rows}\n".encode())
    sink.close()

    result = out_queue.get()
    assert result["result_value"].to_list() == [
        "1.5",
        "2.5",
        "3.5",
        "4.5",
        "5.5",
        "103@POST",
    ]
    # The whole point: strict=False downstream must be able to parse the
    # numeric residue and null the free-text row, not crash.
    numeric = result["result_value"].cast(pl.Float64, strict=False)
    assert numeric.to_list() == [1.5, 2.5, 3.5, 4.5, 5.5, None]


def test_copy_chunk_sink_write_raises_once_stop_is_requested() -> None:
    mod = _load_module()
    out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
    stop_requested = threading.Event()
    sink = mod._CopyChunkSink(
        chunk_rows=10, out_queue=out_queue, stop_requested=stop_requested
    )
    stop_requested.set()
    with pytest.raises(mod._StreamAbandonedError):
        sink.write(b"a,b\n1,x\n")


def test_copy_chunk_sink_flushes_on_byte_cap_before_chunk_rows_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Byte-size insurance: a huge chunk_rows target (never reached in this
    # test) must not stop a flush from happening once the buffer itself
    # crosses the byte cap -- defense in depth against an unexpectedly
    # large region this module hasn't already bounded server-side.
    mod = _load_module()
    monkeypatch.setattr(mod, "_SINK_BUFFER_BYTE_CAP", 100)
    out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
    sink = mod._CopyChunkSink(
        chunk_rows=1_000_000, out_queue=out_queue, stop_requested=threading.Event()
    )
    sink.write(b"a,b\n")
    for i in range(20):
        sink.write(f"{i},{'x' * 20}\n".encode())

    assert not out_queue.empty()  # flushed well before chunk_rows
    frames = []
    while not out_queue.empty():
        frames.append(out_queue.get())
    result = pl.concat(frames)
    assert result["a"].to_list() == [str(i) for i in range(20)]


def test_copy_chunk_sink_handles_one_write_call_per_row() -> None:
    """Real libpq delivers ``COPY TO`` rows one at a time, not in blocks.

    ``PQgetCopyData`` returns exactly one row per call, so psycopg2 invokes
    ``sink.write()`` once per ~60-byte row -- not once per network-buffer's
    worth of bytes. Regression coverage for the quadratic ``_drain`` bug: a
    full-buffer ``self._buffer.count(b"\\n")`` on every such call made a
    500k-row chunk take hours; this pins correctness under that exact
    per-row delivery pattern, and the scaling-guard test below pins that it
    stays fast.
    """
    mod = _load_module()
    chunk_rows = 200
    out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
    sink = mod._CopyChunkSink(
        chunk_rows=chunk_rows, out_queue=out_queue, stop_requested=threading.Event()
    )
    sink.write(b"a,b\n")
    for i in range(chunk_rows):
        sink.write(f"{i},x\n".encode())
    sink.close()

    frames = []
    while not out_queue.empty():
        frames.append(out_queue.get())

    result = pl.concat(frames)
    assert result["a"].to_list() == [str(i) for i in range(chunk_rows)]


def test_copy_chunk_sink_drain_does_not_scale_quadratically_with_chunk_size() -> None:
    """Scaling guard for the ``_drain`` quadratic-rescan bug.

    Feeds one ``write()`` call per row (the real libpq delivery pattern) for
    two chunk sizes and asserts the larger one doesn't take disproportionately
    longer. A per-call full-buffer ``self._buffer.count(b"\\n")`` rescan is
    quadratic in chunk size and would blow this ratio far past the generous
    threshold below; the incremental-counter fix is linear.
    """
    mod = _load_module()

    def _feed(n_rows: int) -> float:
        out_queue: queue.Queue[pl.DataFrame] = queue.Queue()
        sink = mod._CopyChunkSink(
            chunk_rows=n_rows, out_queue=out_queue, stop_requested=threading.Event()
        )
        sink.write(b"a,b\n")
        start = time.perf_counter()
        for i in range(n_rows):
            sink.write(f"{i},x\n".encode())
        sink.close()
        return time.perf_counter() - start

    small_seconds = _feed(50_000)
    large_seconds = _feed(100_000)

    # Linear scaling: 2x the rows should cost roughly 2x the time. A
    # generous 3x ceiling absorbs CI noise while still failing hard on the
    # quadratic pattern (which would blow this ratio out to ~2x*(100k/50k),
    # i.e. another ~2x on top, for a combined ~4x+ at this row count).
    assert large_seconds < small_seconds * 3 + 0.05


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
    assert sorted(total["genc_id"].to_list()) == ["1", "2", "3"]


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
    assert first["genc_id"].to_list() == ["1"]
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
    assert first["genc_id"].to_list() == ["1"]

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


# --- _coerce_boolean_flag -------------------------------------------------


def test_coerce_boolean_flag_maps_postgres_copy_style_and_common_spellings() -> None:
    # Real crash this guards against: ipscu_subset.icu_flag arrived as
    # 't'/'f' strings (Postgres COPY's own boolean encoding), which
    # polars' .cast(pl.Boolean) rejects unconditionally -- there is no
    # supported Utf8 -> Boolean cast at all, strict or not.
    mod = _load_module()
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "flag": ["t", "f", "T", "F", "true", "FALSE", "1", "0", " Yes ", "no"],
        }
    )
    result = mod._coerce_boolean_flag(chunk, "flag", "some_table")
    assert result["flag"].to_list() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_coerce_boolean_flag_drops_and_logs_unresolved_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mod = _load_module()
    chunk = pl.DataFrame({"genc_id": [1, 2, 3], "flag": [None, "garbage", "t"]})

    with caplog.at_level("WARNING"):
        result = mod._coerce_boolean_flag(chunk, "flag", "ipscu_subset")

    assert result["flag"].to_list() == [None, None, True]
    assert any("2 rows had a flag value" in r.message for r in caplog.records)
    assert any("ipscu_subset" in r.message for r in caplog.records)


def test_coerce_boolean_flag_filter_excludes_null_alongside_false() -> None:
    # A caller gating a filter on the coerced column (extract_icu) gets
    # unresolved values excluded automatically, same as a real False --
    # DataFrame.filter already drops null predicates.
    mod = _load_module()
    chunk = pl.DataFrame({"genc_id": [1, 2, 3], "flag": ["t", "f", "unrecognized"]})
    result = mod._coerce_boolean_flag(chunk, "flag", "ipscu_subset")
    kept = result.filter(pl.col("flag"))
    assert kept["genc_id"].to_list() == [1]


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

    subject_by_genc, admission_by_genc, n_dropped_null_subject = (
        mod.fetch_admission_index()
    )

    assert subject_by_genc == {1: "pA", 2: "pB", 3: "pC"}
    assert admission_by_genc[1] == pd.Timestamp("2020-01-01")
    assert admission_by_genc[3] == pd.Timestamp("2020-03-01")
    assert n_dropped_null_subject == 0


def test_fetch_admission_index_drops_null_and_empty_patient_id_hashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real incident: a null patient_id_hashed crashed assign_shards.

    It reached assign_shards's sorted(set(...)), which crashed comparing
    None against str -- the encounter is unattributable to any subject and
    must never enter subject_by_genc in the first place.
    """
    mod = _load_module()
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2, 3],
            "patient_id_hashed": ["pA", None, ""],
            "admission_date_time": ["2020-01-01", "2020-02-01", "2020-03-01"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    subject_by_genc, _admission_by_genc, n_dropped_null_subject = (
        mod.fetch_admission_index()
    )

    assert subject_by_genc == {1: "pA"}
    assert None not in subject_by_genc.values()
    assert n_dropped_null_subject == 2

    # The real failure mode: assign_shards must never see the dropped Nones.
    shard_by_subject = mod.assign_shards(subject_by_genc.values())
    assert set(shard_by_subject) == {"pA"}


def test_fetch_mortality_index_reads_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    chunk1 = _utf8_chunk(
        {"genc_id": [1, 2], "in_hospital_mortality_derived": ["t", "f"]}
    )
    chunk2 = _utf8_chunk({"genc_id": [3], "in_hospital_mortality_derived": ["t"]})
    monkeypatch.setattr(
        mod,
        "_stream_table",
        _fake_stream_table({"derived_variables_subset": [chunk1, chunk2]}),
    )

    mortality_by_genc = mod.fetch_mortality_index()

    assert mortality_by_genc == {1: True, 2: False, 3: True}


def test_fetch_mortality_index_treats_null_flag_as_not_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    chunk = _utf8_chunk(
        {"genc_id": [1, 2], "in_hospital_mortality_derived": [None, "t"]}
    )
    monkeypatch.setattr(
        mod,
        "_stream_table",
        _fake_stream_table({"derived_variables_subset": [chunk]}),
    )

    mortality_by_genc = mod.fetch_mortality_index()

    assert mortality_by_genc == {1: False, 2: True}


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


def test_extract_death_emits_bare_meds_death_for_derived_true_admissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB", 3: "pC"}
    admission_by_genc = {
        1: pd.Timestamp("2020-01-01"),
        2: pd.Timestamp("2020-01-01"),
        3: pd.Timestamp("2020-01-01"),
    }
    # 1, 2 died per the derived flag; 3 did not.
    mortality_by_genc = {1: True, 2: True, 3: False}
    chunk = _utf8_chunk(
        {
            "genc_id": [1, 2, 3],
            "discharge_disposition": [7, 72, 1],
            "discharge_date_time": ["2020-01-05"] * 3,
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    rows = pd.concat(
        [
            b.frame
            for b in mod.extract_death(
                subject_by_genc, admission_by_genc, mortality_by_genc
            )
        ],
        ignore_index=True,
    )

    assert rows["code"].tolist() == ["MEDS_DEATH", "MEDS_DEATH"]
    assert set(rows["subject_id"]) == {"pA", "pB"}
    assert rows["numeric_value"].isna().all()


def test_extract_death_ignores_discharge_disposition_for_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One event stream only (derived-primary) -- disposition never gates it.

    Subject 1: derived flag true but disposition says "discharged home"
    (not a death code) -- still emitted, the derived flag alone decides.
    Subject 2: derived flag false but disposition says "died" -- not
    emitted. The disposition side only feeds the cross-check tally.
    """
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01"), 2: pd.Timestamp("2020-01-01")}
    mortality_by_genc = {1: True, 2: False}
    chunk = _utf8_chunk(
        {
            "genc_id": [1, 2],
            "discharge_disposition": [1, 7],
            "discharge_date_time": ["2020-01-05", "2020-01-05"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    rows = pd.concat(
        [
            b.frame
            for b in mod.extract_death(
                subject_by_genc, admission_by_genc, mortality_by_genc
            )
        ],
        ignore_index=True,
    )

    assert rows["subject_id"].tolist() == ["pA"]
    assert rows["code"].tolist() == ["MEDS_DEATH"]


def test_extract_death_treats_genc_absent_from_mortality_index_as_not_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = _utf8_chunk(
        {
            "genc_id": [1],
            "discharge_disposition": [7],
            "discharge_date_time": ["2020-01-05"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_death(subject_by_genc, admission_by_genc, {})],
        ignore_index=True,
    )

    assert rows.empty


def test_extract_death_drops_discharge_recorded_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    admission_by_genc = {
        1: pd.Timestamp("2020-01-10"),
        2: pd.Timestamp("2020-01-01"),
    }
    mortality_by_genc = {1: True, 2: True}
    chunk = _utf8_chunk(
        {
            "genc_id": [1, 2],
            "discharge_disposition": [7, 7],
            # Subject 1's "death" predates their own admission -- a data
            # artifact, not extracted. Subject 2 is a real, plausible death.
            "discharge_date_time": ["2020-01-05", "2020-01-05"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    rows = pd.concat(
        [
            b.frame
            for b in mod.extract_death(
                subject_by_genc, admission_by_genc, mortality_by_genc
            )
        ],
        ignore_index=True,
    )

    assert rows["subject_id"].tolist() == ["pB"]


def test_extract_death_logs_the_cross_check_tally(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB", 3: "pC"}
    admission_by_genc = {
        1: pd.Timestamp("2020-01-01"),
        2: pd.Timestamp("2020-01-01"),
        3: pd.Timestamp("2020-01-01"),
    }
    # 1: both agree (derived true, disposition death). 2: derived-only
    # (derived true, disposition not a death code). 3: disposition-only
    # (derived false, disposition says death).
    mortality_by_genc = {1: True, 2: True, 3: False}
    chunk = _utf8_chunk(
        {
            "genc_id": [1, 2, 3],
            "discharge_disposition": [7, 1, 7],
            "discharge_date_time": ["2020-01-05"] * 3,
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    with caplog.at_level("INFO"):
        list(mod.extract_death(subject_by_genc, admission_by_genc, mortality_by_genc))

    messages = [r.message for r in caplog.records]
    assert any("1 both agree" in m for m in messages)
    assert any("1 derived-only" in m for m in messages)
    assert any("1 disposition-only" in m for m in messages)


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


def test_extract_icu_handles_postgres_copy_style_boolean_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real crash: ipscu_subset.icu_flag arrives as 't'/'f' text, not the
    # native pl.Boolean this test's sibling above uses -- that shape used
    # to crash extract_icu outright (see _coerce_boolean_flag's docstring).
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB", 3: "pC"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2, 3],
            "scu_admit_date_time": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "scu_discharge_date_time": ["2020-01-05", "2020-01-05", "2020-01-05"],
            "icu_flag": ["t", "f", None],
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


def test_extract_labs_handles_a_garbage_test_type_mapped_omop_without_crashing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # test_type_mapped_omop used a hard (non-lenient) Int64 cast until the
    # all-Utf8 CSV read removed the implicit per-chunk numeric-inference
    # safety net that used to mask this -- a non-null, unparsable value
    # must be dropped (treated as unmapped) and logged, not crash the batch.
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    lab_concepts = {3020564: "Creatinine"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "test_type_mapped_omop": ["3020564", "not-a-concept-id"],
            "result_value": ["1.2", "5.0"],
            "result_unit": ["umol/L", "mg/dL"],
            "collection_date_time": ["2020-01-02 08:00:00", "2020-01-02 09:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"lab_subset": [chunk]})
    )

    with caplog.at_level("WARNING"):
        rows = pd.concat(
            [b.frame for b in mod.extract_labs(subject_by_genc, lab_concepts)],
            ignore_index=True,
        )

    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "LAB//3020564//umol/l"
    assert any("not-a-concept-id" in r.message for r in caplog.records)
    assert any("test_type_mapped_omop" in r.message for r in caplog.records)


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


def test_extract_vitals_handles_a_garbage_measurement_mapped_omop_without_crashing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pB"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "measurement_mapped_omop": ["3027018", "not-a-concept-id"],
            "measurement_value": ["88", "99"],
            "measurement_unit": ["bpm", "bpm"],
            "measure_date_time": ["2020-01-02 08:00:00", "2020-01-02 09:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"vitals_subset": [chunk]})
    )

    with caplog.at_level("WARNING"):
        rows = pd.concat(
            [b.frame for b in mod.extract_vitals(subject_by_genc)], ignore_index=True
        )

    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "VITALS//3027018//bpm"
    assert any("not-a-concept-id" in r.message for r in caplog.records)
    assert any("measurement_mapped_omop" in r.message for r in caplog.records)


def test_normalize_name_series_casefolds_and_collapses_whitespace_only() -> None:
    # Deliberately NOT _normalize_unit_series's canonicalization map --
    # no cross-name variant clustering yet, only casefold + whitespace.
    mod = _load_module()
    raw = pl.Series([" R  FIO2 ", "VSFiO2", None, "", "Pain Score"])
    result = mod._normalize_name_series(raw)
    assert result.to_list() == ["r fio2", "vsfio2", "UNK", "UNK", "pain score"]


def test_extract_vitals_unmapped_is_the_exact_complement_of_extract_vitals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real incident this rescues: extract_vitals drops every
    # measurement_mapped_omop-null row -- ~119M of ~412M rows, the whole
    # 71%-retention story. The two functions must partition the coded
    # rows exactly: no row in both outputs, no row in neither.
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pA", 3: "pA"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2, 3],
            "measurement_mapped_omop": ["3027018", None, None],
            "measurement_name": ["Heart Rate", "VSFiO2", None],
            "measurement_value": ["88", "40", "irrelevant"],
            "measurement_unit": ["bpm", "pct", "pct"],
            "measure_date_time": [
                "2020-01-02 08:00:00",
                "2020-01-02 09:00:00",
                "2020-01-02 10:00:00",
            ],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"vitals_subset": [chunk]})
    )

    mapped = pd.concat(
        [b.frame for b in mod.extract_vitals(subject_by_genc)], ignore_index=True
    )
    unmapped = pd.concat(
        [b.frame for b in mod.extract_vitals_unmapped(subject_by_genc)],
        ignore_index=True,
    )

    assert mapped["code"].tolist() == ["VITALS//3027018//bpm"]
    # row 2 (VSFiO2, numeric value) -> unit segment; row 3 (no name) dropped.
    assert unmapped["code"].tolist() == ["VITALS//vsfio2//pct"]
    assert unmapped["numeric_value"].tolist() == [40.0]
    assert set(mapped["code"]) & set(unmapped["code"]) == set()


def test_extract_vitals_unmapped_folds_a_non_numeric_value_into_the_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "measurement_mapped_omop": [None],
            "measurement_name": ["Pain Score"],
            "measurement_value": ["Unable to assess"],
            "measurement_unit": [None],
            "measure_date_time": ["2020-01-02 08:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"vitals_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_vitals_unmapped(subject_by_genc)],
        ignore_index=True,
    )

    assert rows.iloc[0]["code"] == "VITALS//pain score//unable to assess"
    assert pd.isna(rows.iloc[0]["numeric_value"])


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


# --- ER / transfer / billing families --------------------------------


def test_fetch_discharge_index_reads_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2],
            "discharge_date_time": ["2020-01-05", "2020-02-05"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"admdad_subset": [chunk]})
    )

    discharge_by_genc = mod.fetch_discharge_index()

    assert discharge_by_genc == {
        1: pd.Timestamp("2020-01-05"),
        2: pd.Timestamp("2020-02-05"),
    }


def test_extract_er_produces_registration_triage_and_out_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "registration_date_time": ["2020-01-01 08:00:00"],
            "triage_date_time": ["2020-01-01 08:10:00"],
            "left_er_date_time": ["2020-01-01 12:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"er_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_er(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert sorted(rows["code"]) == ["ED_OUT", "ED_REGISTRATION", "ED_TRIAGE"]
    assert (rows["subject_id"] == "pA").all()
    assert (rows["hadm_id"] == 1).all()


def test_extract_er_applies_the_admission_guard_per_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real-shape guard: a garbage-year registration time must not produce
    # an event, even though triage/leave times on the same row are fine --
    # each of the three events is guarded independently.
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "registration_date_time": ["9022-01-01 08:00:00"],
            "triage_date_time": ["2020-01-01 08:10:00"],
            "left_er_date_time": ["2020-01-01 12:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"er_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_er(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert sorted(rows["code"]) == ["ED_OUT", "ED_TRIAGE"]


def test_extract_er_diagnoses_namespaces_and_attributes_to_admission_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = pl.DataFrame(
        {"genc_id": [1, 1], "er_diagnosis_code": ["R51", None]},
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"erdiagnosis_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_er_diagnoses(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert len(rows) == 1  # null diagnosis code dropped
    assert rows.iloc[0]["code"] == "ED_DIAGNOSIS//R51"
    assert rows.iloc[0]["time"] == pd.Timestamp("2020-01-01")


def test_extract_er_procedures_reuses_the_procedure_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberate: same CCI coding system as ipintervention_subset, so this
    # reuses PROCEDURE// rather than a new ER-specific prefix.
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = pl.DataFrame(
        {
            "genc_id": [1],
            "intervention_code": ["1.ZZ.35"],
            "intervention_episode_start_date_time": ["2020-01-01 09:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"erintervention_subset": [chunk]})
    )

    rows = pd.concat(
        [
            b.frame
            for b in mod.extract_er_procedures(subject_by_genc, admission_by_genc)
        ],
        ignore_index=True,
    )

    assert rows.iloc[0]["code"] == "PROCEDURE//1.ZZ.35"


def test_extract_er_procedures_and_untimed_partition_the_coded_rows_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real incident this guards against: intervention_episode_start_date_time
    # is blank on ~96.7% of real rows, which combined with the admission
    # guard left only ~90k of ~2.94M coded interventions in the timed
    # pass's output. The untimed pass must catch exactly the rows the
    # timed pass drops -- no row in both outputs, no row in neither.
    mod = _load_module()
    subject_by_genc = {1: "pA", 2: "pA", 3: "pA", 4: "pA"}
    admission_by_genc = {
        1: pd.Timestamp("2020-01-01"),
        2: pd.Timestamp("2020-01-01"),
        3: pd.Timestamp("2020-01-01"),
        4: None,  # genc not in admdad_subset at all
    }
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 2, 3, 4],
            "intervention_code": ["A", "B", "C", "D"],
            "intervention_episode_start_date_time": [
                "2020-01-01 09:00:00",  # 1: real, in-window -- kept by the timed pass
                "",  # 2: blank -- rescued by the untimed pass
                "9022-01-01 09:00:00",  # 3: real but out-of-window -- rescued
                "2020-01-01 09:00:00",  # 4: real, in-window, but no admission time
            ],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"erintervention_subset": [chunk]})
    )

    timed = pd.concat(
        [
            b.frame
            for b in mod.extract_er_procedures(subject_by_genc, admission_by_genc)
        ],
        ignore_index=True,
    )
    untimed = pd.concat(
        [
            b.frame
            for b in mod.extract_er_procedures_untimed(
                subject_by_genc, admission_by_genc
            )
        ],
        ignore_index=True,
    )

    assert sorted(timed["code"]) == ["PROCEDURE//A"]
    assert sorted(untimed["code"]) == ["PROCEDURE//B", "PROCEDURE//C"]
    # 4 is dropped by both -- no admission time to fall back to.
    assert set(timed["code"]) & set(untimed["code"]) == set()
    assert "PROCEDURE//D" not in set(timed["code"]) | set(untimed["code"])
    untimed_c = untimed[untimed["code"] == "PROCEDURE//C"]
    assert (untimed_c["time"] == pd.Timestamp("2020-01-01")).all()


def test_extract_er_consults_namespaces_by_service_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = pl.DataFrame(
        {
            "genc_id": [1, 1],
            "consult_service_code": ["CARD", ""],
            "consult_request_date_time": ["2020-01-01 10:00:00", "2020-01-01 10:00:00"],
        }
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"erconsults_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_er_consults(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert len(rows) == 1  # blank service code dropped
    assert rows.iloc[0]["code"] == "ER_CONSULT//CARD"


def test_extract_transfers_attributes_to_admission_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    admission_by_genc = {1: pd.Timestamp("2020-01-01")}
    chunk = pl.DataFrame(
        {"genc_id": [1, 1], "institution_to_mns": ["H002", None]},
    )
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"lookup_transfer_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_transfers(subject_by_genc, admission_by_genc)],
        ignore_index=True,
    )

    assert len(rows) == 1  # null destination dropped
    assert rows.iloc[0]["code"] == "TRANSFER_TO//H002"
    assert rows.iloc[0]["time"] == pd.Timestamp("2020-01-01")


def test_extract_billing_cmg_attributes_to_discharge_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    discharge_by_genc = {1: pd.Timestamp("2020-01-05")}
    chunk = pl.DataFrame({"genc_id": [1, 1], "cmg": ["123", None]})
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"ipcmg_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_billing_cmg(subject_by_genc, discharge_by_genc)],
        ignore_index=True,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["code"] == "BILLING_CMG//123"
    assert rows.iloc[0]["time"] == pd.Timestamp("2020-01-05")


def test_extract_billing_hig_attributes_to_discharge_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    subject_by_genc = {1: "pA"}
    discharge_by_genc = {1: pd.Timestamp("2020-01-05")}
    chunk = pl.DataFrame({"genc_id": [1, 1], "hig_code": ["H01", ""]})
    monkeypatch.setattr(
        mod, "_stream_table", _fake_stream_table({"iphig_subset": [chunk]})
    )

    rows = pd.concat(
        [b.frame for b in mod.extract_billing_hig(subject_by_genc, discharge_by_genc)],
        ignore_index=True,
    )

    assert len(rows) == 1  # blank HIG code dropped
    assert rows.iloc[0]["code"] == "BILLING_HIG//H01"
    assert rows.iloc[0]["time"] == pd.Timestamp("2020-01-05")


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


def test_assign_shards_skips_none_defensively() -> None:
    """Defensive guard, independent of fetch_admission_index's own filtering.

    Real incident: a null patient_id_hashed reached here as a bare None and
    sorted(set(...)) crashed comparing None against str. Covered here in
    case a future index source reintroduces one.
    """
    mod = _load_module()
    shards = mod.assign_shards(["pA", None, "pB", None], subjects_per_shard=1000)
    assert shards == {"pA": 0, "pB": 0}


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


def test_meds_shard_writer_buffers_below_threshold_and_flushes_at_close(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    batch = pd.DataFrame(
        {
            "subject_id": ["pA"],
            "time": [pd.Timestamp("2020-01-01")],
            "code": ["ADMISSION"],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )

    writer.write_batch("admdad_subset", batch)
    # Below SHARD_FLUSH_ROW_THRESHOLD -- nothing on disk yet, still buffered.
    assert not (tmp_path / "shard_0000.parquet").exists()

    counts = writer.close()
    assert counts == {0: 1}
    assert (tmp_path / "shard_0000.parquet").exists()


def test_meds_shard_writer_flushes_once_a_shard_crosses_the_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "SHARD_FLUSH_ROW_THRESHOLD", 3)
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    batch = pd.DataFrame(
        {
            "subject_id": ["pA"] * 3,
            "time": [pd.Timestamp("2020-01-01")] * 3,
            "code": ["ADMISSION"] * 3,
            "numeric_value": [None] * 3,
            "hadm_id": [1, 2, 3],
        }
    )

    writer.write_batch("admdad_subset", batch)
    # At the threshold -- flushed mid-batch, buffer cleared before close()
    # is ever called (the file isn't readable yet -- ParquetWriter only
    # writes a valid footer at close() -- so check internal buffer state
    # rather than reading the file here).
    assert writer._buffer_row_counts.get(0, 0) == 0
    assert writer._shard_row_counts.get(0, 0) == 3

    counts = writer.close()
    assert counts == {0: 3}
    assert pd.read_parquet(tmp_path / "shard_0000.parquet").shape[0] == 3


def test_meds_shard_writer_flush_all_makes_a_below_threshold_batch_durable_without_close(
    tmp_path: Path,
) -> None:
    """Real incident: buffered rows lost when the process crashed before close().

    ipscu_subset's events were buffered under SHARD_FLUSH_ROW_THRESHOLD,
    the manifest was marked complete, and a *later* table's crash killed
    the process before MedsShardWriter.close() -- which back then was the
    only thing that ever flushed a buffered shard or wrote a valid
    Parquet footer -- ever ran. flush_all() must make a table's rows
    durable and readable on its own, with no dependence on close() ever
    being reached.
    """
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    batch = pd.DataFrame(
        {
            "subject_id": ["pA"],
            "time": [pd.Timestamp("2020-01-01")],
            "code": ["ICU_ADMISSION"],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )
    writer.write_batch("ipscu_subset", batch)
    assert not (tmp_path / "shard_0000.parquet").exists()  # still buffered

    writer.flush_all()
    # Simulate the process being killed right here, before any later
    # table's generator runs and before writer.close() is ever reached --
    # flush_all()'s output must already be valid, readable Parquet.
    assert (tmp_path / "shard_0000.parquet").exists()
    on_disk = pd.read_parquet(tmp_path / "shard_0000.parquet")
    assert on_disk["code"].tolist() == ["ICU_ADMISSION"]
    assert mod._logical_shard_row_counts(tmp_path) == {0: 1}


def test_meds_shard_writer_flush_all_lets_the_next_table_use_a_fresh_part_file(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    row = lambda code: pd.DataFrame(  # noqa: E731
        {
            "subject_id": ["pA"],
            "time": [pd.Timestamp("2020-01-01")],
            "code": [code],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )
    writer.write_batch("ipscu_subset", row("ICU_ADMISSION"))
    writer.flush_all()
    writer.write_batch("lab_subset", row("LAB//1//UNK"))
    writer.close()

    assert (tmp_path / "shard_0000.parquet").exists()
    assert (tmp_path / "shard_0000_part1.parquet").exists()
    assert mod._logical_shard_row_counts(tmp_path) == {0: 2}


def test_flush_all_leaves_no_retained_buffers_or_closed_writers(tmp_path: Path) -> None:
    """Regression guard for a suspected reference-retention leak.

    Real production incident this checks for: a lab_subset extraction was
    OOM-killed at 340M buffered rows in a 96 GB job shortly after the
    per-table flush_all() durability fix landed -- close in time enough
    that a retention leak in the new writer lifecycle (frames/writers
    never actually released) was suspected as the cause. Runs many
    batches through many flush_all() cycles (simulating many tables all
    writing to the same shards) and asserts: (1) every internal buffer
    structure is genuinely empty afterward, not just zeroed, and (2) a
    closed pq.ParquetWriter is actually garbage-collected once flush_all()
    drops the last reference to it, not kept alive by anything else.
    """
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {f"p{i}": i % 4 for i in range(20)})
    closed_writer_refs = []

    for table_i in range(10):
        batch = pd.DataFrame(
            {
                "subject_id": [f"p{i}" for i in range(20)],
                "time": [pd.Timestamp("2020-01-01")] * 20,
                "code": [f"CODE//{table_i}"] * 20,
                "numeric_value": [None] * 20,
                "hadm_id": list(range(20)),
            }
        )
        writer.write_batch(f"table_{table_i}", batch)
        # Every shard actually used this table must have an open writer to
        # track a weakref against, proving flush_all() really drops it.
        closed_writer_refs.extend(weakref.ref(w) for w in writer._writers.values())
        writer.flush_all()
        assert writer._buffers == {}
        assert all(v == 0 for v in writer._buffer_row_counts.values())
        assert writer._writers == {}

    gc.collect()
    assert all(ref() is None for ref in closed_writer_refs), (
        "a closed pq.ParquetWriter from an earlier table is still alive -- "
        "something outside MedsShardWriter's own cleared dicts is retaining it"
    )
    writer.close()
    assert sum(mod._logical_shard_row_counts(tmp_path).values()) == 20 * 10


def test_write_batch_flushes_the_fullest_shards_once_the_global_cap_is_exceeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The global cap must flush the fullest shards first, keeping the total bounded.

    Real incident: uniform hash-sharding brings every shard to its own
    per-shard SHARD_FLUSH_ROW_THRESHOLD in lockstep, producing a
    synchronized aggregate high-water of ~SHARD_FLUSH_ROW_THRESHOLD *
    n_shards resident rows -- enough to OOM-kill a 64 GB and a 96 GB job.
    """
    mod = _load_module()
    monkeypatch.setattr(mod, "SHARD_FLUSH_ROW_THRESHOLD", 10_000)
    monkeypatch.setattr(mod, "WRITER_MAX_BUFFERED_ROWS", 100)
    n_shards = 20
    shard_by_subject = {f"p{i}": i for i in range(n_shards)}
    writer = mod.MedsShardWriter(tmp_path, shard_by_subject)

    # Every shard gets a few rows per batch, well under its own
    # per-shard threshold (10_000) -- only the GLOBAL cap (100) can
    # possibly trigger a flush here.
    for _ in range(10):
        batch = pd.DataFrame(
            {
                "subject_id": [f"p{i}" for i in range(n_shards)],
                "time": [pd.Timestamp("2020-01-01")] * n_shards,
                "code": ["CODE"] * n_shards,
                "numeric_value": [None] * n_shards,
                "hadm_id": list(range(n_shards)),
            }
        )
        writer.write_batch("big_table", batch)
        assert writer.total_buffered_rows <= 100 + n_shards, (
            "global cap must keep total buffered rows bounded, not let "
            "every shard's own per-shard threshold be the only limit"
        )

    writer.close()
    assert sum(mod._logical_shard_row_counts(tmp_path).values()) == n_shards * 10


def test_run_extraction_survives_a_later_tables_crash_without_losing_an_earlier_ones_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins the exact attempt-5 scenario odyssey-db forensically diagnosed.

    ipscu_subset's rows were buffered below SHARD_FLUSH_ROW_THRESHOLD,
    manifest-marked complete, and then lab_subset's crash killed the
    process before the single end-of-run writer.close() ever ran --
    silently losing every ipscu_subset row while the manifest still
    claimed completion. Reproduced here: admdad_subset (small, buffered)
    completes and is marked complete, then lab_subset's generator raises
    -- admdad_subset's rows must already be durable on disk, and the
    manifest's claim about it must be true.
    """
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
            # 1 = discharged home, not a death code -- admdad_subset__death
            # reads this too (for its cross-check tally), needs it present.
            "discharge_disposition": [1],
        }
    )

    def fake_stream_table(
        table: str, _select_cols: list[str]
    ) -> Iterator[pl.DataFrame]:
        if table == "admdad_subset":
            yield admdad_chunk
        elif table == "lab_subset":
            raise RuntimeError("simulated crash: 103@POST")

    monkeypatch.setattr(mod, "_stream_table", fake_stream_table)

    with pytest.raises(RuntimeError, match="103@POST"):
        mod.run_extraction(output_dir=tmp_path)

    manifest = mod._load_manifest(tmp_path)
    assert manifest.get("admdad_subset") == "complete"
    # The claim must be true: admdad_subset's rows are actually durable on
    # disk, not sitting lost in an unflushed, never-closed writer buffer.
    on_disk = pd.concat(pd.read_parquet(p) for p in tmp_path.glob("shard_*.parquet"))
    assert set(on_disk["code"]) == {"ADMISSION", "DISCHARGE"}


# --- MedsShardWriter part-file resumability -------------------------------


def test_next_shard_write_path_uses_the_base_name_when_nothing_exists_yet(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    assert mod._next_shard_write_path(tmp_path, 7) == tmp_path / "shard_0007.parquet"


def test_next_shard_write_path_picks_the_next_free_part_when_base_exists(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    (tmp_path / "shard_0007.parquet").touch()
    assert (
        mod._next_shard_write_path(tmp_path, 7) == tmp_path / "shard_0007_part1.parquet"
    )
    (tmp_path / "shard_0007_part1.parquet").touch()
    assert (
        mod._next_shard_write_path(tmp_path, 7) == tmp_path / "shard_0007_part2.parquet"
    )


def test_logical_shard_row_counts_sums_base_and_part_files(tmp_path: Path) -> None:
    mod = _load_module()
    writer1 = mod.MedsShardWriter(tmp_path, {"pA": 0})
    writer1.write_batch(
        "admdad_subset",
        pd.DataFrame(
            {
                "subject_id": ["pA"],
                "time": [pd.Timestamp("2020-01-01")],
                "code": ["ADMISSION"],
                "numeric_value": [None],
                "hadm_id": [1],
            }
        ),
    )
    writer1.close()

    # A second writer instance touching the same shard must never reopen
    # the first writer's file -- it gets a _part1 file instead.
    writer2 = mod.MedsShardWriter(tmp_path, {"pA": 0})
    writer2.write_batch(
        "lab_subset",
        pd.DataFrame(
            {
                "subject_id": ["pA"],
                "time": [pd.Timestamp("2020-01-02")],
                "code": ["LAB//1//UNK"],
                "numeric_value": [1.0],
                "hadm_id": [1],
            }
        ),
    )
    writer2.close()

    assert (tmp_path / "shard_0000.parquet").exists()
    assert (tmp_path / "shard_0000_part1.parquet").exists()
    assert mod._logical_shard_row_counts(tmp_path) == {0: 2}


# --- Atomic tmp-then-rename part-file visibility --------------------------


def test_writer_writes_to_a_tmp_path_and_only_the_rename_makes_it_countable(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    batch = pd.DataFrame(
        {
            "subject_id": ["pA"],
            "time": [pd.Timestamp("2020-01-01")],
            "code": ["ADMISSION"],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )
    writer.write_batch("admdad_subset", batch)
    writer.flush_all()  # below threshold -- only flush_all() writes it

    # The moment flush_all() closes the writer, the file must exist under
    # its tmp name for zero time -- by the time we can observe it, only
    # the final, renamed name should be present.
    assert (tmp_path / "shard_0000.parquet").exists()
    assert not (tmp_path / "shard_0000.parquet.tmp").exists()
    assert mod._logical_shard_row_counts(tmp_path) == {0: 1}


def test_clean_leftover_tmp_files_deletes_and_logs_an_orphan_at_startup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fresh MedsShardWriter must never let a leftover .tmp file linger.

    Real incident: a killed process's truncated tmp part-file, left under
    a countable final name before this fix existed, was only discovered
    much later by a corrupt-file crash deep in a resumed run.
    """
    mod = _load_module()
    orphan = tmp_path / "shard_0000.parquet.tmp"
    orphan.write_bytes(b"not a valid parquet file, truncated mid-write")

    with caplog.at_level("WARNING"):
        mod.MedsShardWriter(tmp_path, {"pA": 0})

    assert not orphan.exists()
    assert any("leftover tmp part-file" in r.message for r in caplog.records)
    assert any(str(orphan) in r.message for r in caplog.records)


def test_clean_leftover_tmp_files_is_a_noop_when_nothing_is_there(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    mod.MedsShardWriter(tmp_path, {"pA": 0})  # must not raise


def test_logical_shard_row_counts_fails_loud_with_the_offending_path(
    tmp_path: Path,
) -> None:
    """A corrupt final-named file must never be silently skipped.

    Real incident: an OOM-killed process left a truncated, footer-less
    file under a *final*, countable name (before the atomic tmp-then-
    rename fix existed); pq.ParquetFile's ArrowInvalid carried no path,
    forcing a manual scan of 1,000+ files to find the one bad one. The
    error must name exactly which path is bad.
    """
    mod = _load_module()
    corrupt = tmp_path / "shard_0000.parquet"
    corrupt.write_bytes(b"not a valid parquet file, truncated mid-write")

    with pytest.raises(RuntimeError, match="corrupt or unreadable"):
        mod._logical_shard_row_counts(tmp_path)

    # The whole point: the offending path must be nameable from the error
    # alone, not require re-scanning every file to find it.
    try:
        mod._logical_shard_row_counts(tmp_path)
    except RuntimeError as exc:
        assert str(corrupt) in str(exc)


def test_discard_incomplete_writers_removes_only_tmp_files_and_clears_state(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0, "pB": 1})
    batch = pd.DataFrame(
        {
            "subject_id": ["pA", "pB"],
            "time": [pd.Timestamp("2020-01-01")] * 2,
            "code": ["ADMISSION"] * 2,
            "numeric_value": [None, None],
            "hadm_id": [1, 2],
        }
    )
    writer.write_batch("admdad_subset", batch)  # below threshold -- still buffered
    # Force both shards' writers open (as a real mid-table crash would
    # have, once SHARD_FLUSH_ROW_THRESHOLD had been crossed at least once).
    writer._writer_for(0)
    writer._writer_for(1)
    tmp_paths = [
        tmp_path / "shard_0000.parquet.tmp",
        tmp_path / "shard_0001.parquet.tmp",
    ]
    assert all(p.exists() for p in tmp_paths)

    writer.discard_incomplete_writers()

    assert not any(p.exists() for p in tmp_paths)
    assert writer._writers == {}
    assert writer._writer_paths == {}
    assert writer._buffers == {}
    assert writer.total_buffered_rows == 0
    assert mod._logical_shard_row_counts(tmp_path) == {}


def test_discard_incomplete_writers_never_touches_an_already_flushed_shard(
    tmp_path: Path,
) -> None:
    """A prior table's already-renamed, final-named file must survive."""
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})
    writer.write_batch(
        "admdad_subset",
        pd.DataFrame(
            {
                "subject_id": ["pA"],
                "time": [pd.Timestamp("2020-01-01")],
                "code": ["ADMISSION"],
                "numeric_value": [None],
                "hadm_id": [1],
            }
        ),
    )
    writer.flush_all()  # table 1 completes durably, renamed to its final name

    writer.write_batch(
        "lab_subset",
        pd.DataFrame(
            {
                "subject_id": ["pA"],
                "time": [pd.Timestamp("2020-01-02")],
                "code": ["LAB//1//UNK"],
                "numeric_value": [1.0],
                "hadm_id": [1],
            }
        ),
    )  # table 2's own tmp part-file is now open, not yet flushed

    writer.discard_incomplete_writers()

    assert (tmp_path / "shard_0000.parquet").exists()  # table 1 survives
    assert not (tmp_path / "shard_0000_part1.parquet.tmp").exists()  # table 2 discarded
    assert mod._logical_shard_row_counts(tmp_path) == {0: 1}


def test_extract_one_table_discards_incomplete_writers_on_a_mid_table_crash(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    writer = mod.MedsShardWriter(tmp_path, {"pA": 0})

    def crashing_generator():
        yield mod.ExtractedBatch(
            pd.DataFrame(
                {
                    "subject_id": ["pA"],
                    "time": [pd.Timestamp("2020-01-01")],
                    "code": ["ADMISSION"],
                    "numeric_value": [None],
                    "hadm_id": [1],
                }
            ),
            1,
        )
        raise RuntimeError("simulated crash: 103@POST")

    with pytest.raises(RuntimeError, match="103@POST"):
        mod._extract_one_table("admdad_subset", crashing_generator(), writer)

    assert writer._writers == {}
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.parquet")) == []


def test_run_extraction_resumed_run_never_truncates_a_completed_tables_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real near-miss this guards against.

    A resumed run's MedsShardWriter is a brand-new instance every process
    invocation; the old (pre-fix) _writer_for unconditionally reopened
    shard_{i:04d}.parquet, silently truncating whatever a *different*,
    already-completed table had written there in a prior run. Reproduces
    the exact shape of the relaunch this was caught before: run 1 writes
    admdad_subset only (table A), run 2 skips A (manifest already marks it
    complete) and writes lab_subset (table B) into the same, overlapping
    shard -- table A's rows must all survive.
    """
    mod = _load_module()
    monkeypatch.setattr(mod, "count_distinct_subjects", lambda: 1)
    monkeypatch.setattr(
        mod.db,
        "query",
        lambda sql, params=None: pd.DataFrame(
            {"concept_id": [3020564], "concept_desc": ["Creatinine"]}
        ),
    )

    admdad_fixture = {
        "admdad_subset": [
            pl.DataFrame(
                {
                    "genc_id": [1],
                    "patient_id_hashed": ["pA"],
                    "admission_date_time": ["2020-01-01"],
                    "discharge_date_time": ["2020-01-05"],
                    # 1 = discharged home, not a death code -- admdad_subset__death
                    # (a separate table_generators entry over this same table)
                    # needs the column present but shouldn't add rows here.
                    "discharge_disposition": [1],
                }
            )
        ]
    }
    monkeypatch.setattr(mod, "_stream_table", _fake_stream_table(admdad_fixture))
    mod.run_extraction(output_dir=tmp_path)

    admdad_rows_after_run1 = mod._logical_shard_row_counts(tmp_path)[0]
    assert admdad_rows_after_run1 == 2  # ADMISSION + DISCHARGE

    # run_extraction marks every table generator "complete" once drained --
    # including the other 8 tables, which produced zero rows this run
    # since the fake _stream_table has no fixture for them. Reset
    # lab_subset's entry to simulate it genuinely not having completed yet
    # (e.g. the process was killed before it ran) -- the real shape of a
    # partial-completion resume, which admdad_subset's own "complete" entry
    # must survive untouched.
    manifest = mod._load_manifest(tmp_path)
    assert manifest.get("admdad_subset") == "complete"
    manifest.pop("lab_subset", None)
    mod._save_manifest(tmp_path, manifest)

    # Run 2: admdad_subset is already "complete" in the manifest and will
    # be skipped; only lab_subset (same subject, same shard) actually runs.
    both_fixture = {
        "admdad_subset": admdad_fixture["admdad_subset"],
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
    monkeypatch.setattr(mod, "_stream_table", _fake_stream_table(both_fixture))
    mod.run_extraction(output_dir=tmp_path)

    logical_counts = mod._logical_shard_row_counts(tmp_path)
    assert logical_counts[0] == 3  # 2 admdad rows survive + 1 new lab row
    all_rows = pd.concat(
        pd.read_parquet(p) for p in tmp_path.glob("shard_0000*.parquet")
    )
    assert set(all_rows["code"]) == {"ADMISSION", "DISCHARGE", "LAB//3020564//umol/l"}


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
    # admdad_subset__death is marked complete too, deliberately: it reads
    # the same "admdad_subset" fixture table name (unlike the other
    # not-listed-here tables, which get no fixture chunk at all and so
    # produce zero rows harmlessly), and this fixture's chunk doesn't
    # carry discharge_disposition -- not what this test is about.
    manifest = dict.fromkeys(
        [
            "ipscu_subset",
            "lab_subset",
            "vitals_subset",
            "pharmacy_subset",
            "ipdiagnosis_subset",
            "ipintervention_subset",
            "radiology_subset",
            "admdad_subset__death",
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
                    # 1 = discharged home, not a death code -- see the same
                    # comment in test_run_extraction_resumed_run_never_
                    # truncates_a_completed_tables_shard.
                    "discharge_disposition": [1],
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
