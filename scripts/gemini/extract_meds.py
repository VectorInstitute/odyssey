#!/usr/bin/env python
"""GEMINI -> MEDS extraction: streaming Postgres -> sharded parquet.

See ``docs/gemini_extraction.md`` for the full source-table-by-source-table
design and its open questions; this module implements the resolved parts of
that mapping (admission-anchored timestamp guards, the deduplicated lab
concept lookup, subject-hash sharding) and leaves the OMOP -> LOINC bridge
and value-binning integration to ``odyssey/data/code_mapping.py`` -- MEDS
``code`` values here are GEMINI's own raw identifiers (OMOP concept ids,
ICD-10-CA codes, CCI codes, ...), namespaced but not yet translated to the
canonical LOINC-keyed vocabulary the rest of the pipeline uses. That
translation is a later, separate concept-labeling stage, same as MIMIC-IV
and eICU. Lab/vitals codes carry their literal, normalized unit
(:func:`_normalize_unit_series`) as a third ``//``-separated segment
(``LAB//<omop_id>//<unit>``/``VITALS//<omop_id>//<unit>``, the same shape
MIMIC-IV's own ``LAB//<itemid>//<unit>`` codes use) rather than assuming one
unit per concept: GEMINI is multi-hospital, and the same OMOP concept can
carry a different unit at a different site, so the unit has to be part of
the token identity for a later unit-aware clinical range to key on the
prefix correctly and for mixed-unit values to never share a quantile bin.

**Governance, same rule as everywhere else in this package (see
docs/gemini.md): MEDS parquet shards are patient-level data and never leave
GEMINI.** They are written to :data:`OUTPUT_DIR` on the enclave's own NFS
storage, never to a git-tracked path. Only a small, aggregate
``extraction_summary.json`` (rounded row/subject/shard counts, no
patient-level content) is meant to be committed -- ``scripts/gemini/run.sh
extract`` enforces this the same way it already does for every other step's
output (see ``run.sh``'s own path/size checks).

Design, one function per source table, streaming throughout:

- :func:`preflight_shard_capacity` -- one cheap ``COUNT(DISTINCT
  patient_id_hashed)`` (:func:`count_distinct_subjects`) sizes the shard
  count before anything else runs, then makes sure this process can
  actually open that many files at once (raises the soft ``NOFILE`` limit
  if needed, fails loudly with the exact ``ulimit -n`` to run otherwise) --
  see :class:`MedsShardWriter`'s one-open-file-per-shard design below.
- :func:`fetch_admission_index` -- one pass over ``admdad_subset`` mapping
  ``genc_id -> (patient_id_hashed, admission time)``, held in memory for the
  whole run (every event table carries only ``genc_id``, and the
  pharmacy/radiology timestamp guard needs the encounter's admission time
  as its anchor).
- One ``extract_<table>`` generator per source table, each a lazy iterator
  of :class:`ExtractedBatch` (a MEDS-shaped batch plus its source row
  count, the latter for progress logging only), reading its source table
  via :func:`_stream_table` and transforming each chunk with vectorized
  polars expressions -- never a Python ``for _, row in batch.iterrows()``
  loop, and never a whole table in memory at once, both of which matter at
  ``lab_subset``'s/``vitals_subset``'s real scale (hundreds of millions of
  rows). See "Fetch strategy" below.
- :func:`run_extraction` -- orchestrates all of the above, checkpointing
  each table's completion to a resume manifest (see "Resumability" below)
  and writing the suppressed summary at the end.

Fetch strategy (2026-08-21 rewrite, second pass): the first pass of this
rewrite kept row-ordered keyset pagination (``WHERE row_num > :cursor
ORDER BY row_num``) but read it through one continuous server-side cursor
instead of many small requeries. Amrit's real run of *that* immediately
showed the actual root cause: these matviews have no index on ``row_num``,
so ``ORDER BY row_num`` forces Postgres to fully scan and sort the table
before returning even the first row of any page -- confirmed by zero byte
growth over 5 minutes into ``lab_subset``. Ordering was never something
this extraction actually needed (:class:`MedsShardWriter` hashes every row
to a shard by subject, and any per-subject ordering downstream comes from
``build_patient_sequence``'s own sort) -- it was carried over from the
original per-page pattern without being questioned. This version drops
``ORDER BY``/``row_num``-keyset pagination entirely: :func:`_stream_table`
reads each source table exactly once, in whatever order Postgres returns
it, via one of two fetch mechanisms:

- :func:`_stream_table_copy` (preferred) -- ``COPY (SELECT ...) TO STDOUT
  WITH (FORMAT CSV, HEADER, NULL '\\N')`` through
  :func:`odyssey.data.gemini.db.copy_to_sink`, parsed incrementally by
  :class:`_CopyChunkSink` into :data:`CHUNK_ROWS`-row polars DataFrames on
  a background thread (see that class's docstring for why a thread is
  needed at all). The standard fastest bulk-export mechanism Postgres has;
  a full sequential scan with no sort node.
- :func:`_stream_table_cursor` (fallback) -- an un-ordered, server-side
  (named) cursor via :func:`odyssey.data.gemini.db.stream_query`, used only
  if the ``COPY`` attempt fails before yielding anything (e.g. ``COPY``
  isn't permitted for this connection/role).

Neither of these has been run against the real GEMINI database -- there is
no way to do that from outside the enclave -- but both are sequential
scans with no server-side sort, which is what the real failure actually
diagnosed. :func:`_normalize_unit_series`/:func:`_parse_datetime_series`/
the polars-vectorized ``extract_<table>`` bodies are unaffected by any of
this (they operate on whatever chunk arrives, regardless of source order)
and remain the other half of the original performance request: a local
synthetic benchmark (50,000 lab-shaped rows) measured the old
``.iterrows()`` + scalar ``pd.to_datetime``/``pd.to_numeric`` transform
path at ~7,424 rows/s versus ~889,349 rows/s vectorized (~120x), a real,
measured win now folded into every extractor below.

Resumability: **table granularity only.** A run's progress is checkpointed
to ``<output_dir>/extract_manifest.json`` (:func:`_load_manifest`,
:func:`_save_manifest`) as ``{table: "complete"}`` once a table's
extractor generator is fully drained and its output fully written,
skipped -- see :func:`run_extraction`. A killed run restarts any
not-yet-complete table from scratch (there is no more "resume from
row X" -- there is no more ordering to resume a position *in*). This is a
deliberate simplification from the first pass's row-level checkpointing,
matching the fetch strategy above: with no ``ORDER BY``/sort cost, each
table is now a single fast pass, so re-running one from zero after a kill
is a bounded, acceptable cost -- not the multi-hour re-scan a sorted
resume was originally meant to avoid. Index-building passes
(:func:`fetch_admission_index`, and the discharge-time index inside
:func:`extract_diagnoses`) are never checkpointed either -- they are held
only in memory for the run that builds them and are cheap to rebuild from
scratch on every restart.

Run on the GEMINI node (writes real patient data to ``OUTPUT_DIR`` -- not a
dry run):

    uv run python scripts/gemini/extract_meds.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import resource
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, Optional

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from odyssey.data.gemini import db


logger = logging.getLogger(__name__)

#: Where MEDS parquet shards are written -- enclave NFS only, never git.
#: Overridable via GEMINI_MEDS_OUTPUT_DIR; see docs/gemini_extraction.md's
#: Sharding and output section.
OUTPUT_DIR = Path(
    os.environ.get(
        "GEMINI_MEDS_OUTPUT_DIR",
        "/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1",
    )
)

SUMMARY_PATH = Path(__file__).resolve().parent / "out" / "extraction_summary.json"

#: Target subjects per shard (rounded, see docs/gemini_extraction.md).
SUBJECTS_PER_SHARD = 1000

#: Extra open file descriptors reserved for stdio, the DB connection, log
#: handles, etc., on top of one handle per shard -- see
#: :func:`preflight_shard_capacity`.
FD_HEADROOM = 64

#: Rows per chunk yielded by :func:`_stream_table_copy` -- see the module
#: docstring's "Fetch strategy" section.
CHUNK_ROWS = 500_000

#: Rows per chunk (psycopg2 cursor ``itersize``, via ``chunksize``) for the
#: :func:`_stream_table_cursor` fallback -- kept smaller than
#: :data:`CHUNK_ROWS` because a named cursor keeps its whole ``itersize``
#: buffer server-side per fetch, unlike ``COPY``'s pure streaming.
CURSOR_FALLBACK_CHUNK_ROWS = 100_000

#: How often (in seconds) :func:`run_extraction` logs cumulative
#: rows-done/rows-per-second/ETA for the table currently being extracted.
PROGRESS_LOG_INTERVAL_SECONDS = 30.0

#: The only columns anything downstream reads -- see
#: odyssey/training/data.py's ``_MEDS_EVENT_COLUMNS``, kept in sync
#: deliberately (not imported, to keep this script's only real dependency
#: on the rest of odyssey to :mod:`odyssey.data.gemini`, consistent with
#: every other GEMINI-facing script -- see docs/gemini.md).
MEDS_COLUMNS = ["subject_id", "time", "code", "numeric_value", "hadm_id"]

MEDS_ARROW_SCHEMA = pa.schema(
    [
        ("subject_id", pa.string()),
        ("time", pa.timestamp("us")),
        ("code", pa.string()),
        ("numeric_value", pa.float64()),
        ("hadm_id", pa.int64()),
    ]
)

#: The admission-anchored timestamp guard's window -- accept a parsed
#: pharmacy/radiology timestamp only if it falls within this of the owning
#: encounter's admission time. See docs/gemini_extraction.md's open
#: question 7: real outliers found span 1840-9999, data-entry artifacts,
#: not real events centuries away.
GUARD_WINDOW = pd.Timedelta(days=366)

#: Sentinel strings meaning "no unit recorded" -- checked after casefold,
#: so this set only needs the lowercase form. See
#: :func:`_normalize_unit_series`.
_UNIT_SENTINELS = {"", "none", "(null)", "null", "nan"}

#: Explicit unit-string canonicalization, applied (after casefold + internal
#: whitespace collapse) before falling back to the collapsed string as-is.
#: Keys are already casefold+whitespace-collapsed. Sourced directly from
#: the real value counts in ``scripts/gemini/out/extract_dry.md``'s "Lab
#: unit samples"/"Vitals unit samples" sections (fetched from
#: ``gemini/main`` 2026-08-21) -- GEMINI's raw unit strings are extremely
#: inconsistent (case, spacing, punctuation, real typos) for what's the
#: same actual unit, and left uncanonicalized would silently fragment one
#: quantile bin into a dozen near-empty ones. Add new variants here as
#: one-line entries when a fresh extract-dry unit-sample run finds more --
#: when this map exists at all, the intent is to cover every family the
#: report shows fragmenting, not just the ones first noticed.
UNIT_CANONICALIZATION_MAP: dict[str, str] = {
    # x10^9/L (WBC differentials, platelets, ...): a genuine single unit
    # fragmented across a dozen spacing/notation variants in the raw data.
    "x10 9/l": "x10e9/l",
    "x 10 9/l": "x10e9/l",
    "x 10^9/l": "x10e9/l",
    "x10^9/l": "x10e9/l",
    "x10*9/l": "x10e9/l",
    "10*9/l": "x10e9/l",
    "10e9/l": "x10e9/l",
    "e9/l": "x10e9/l",
    # x10^6/L: a different magnitude from x10^9/L -- never merged with it.
    "x 10^6/l": "x10e6/l",
    "x10 6/l": "x10e6/l",
    # x10^12/L (erythrocyte counts): same fragmentation pattern, a third
    # distinct magnitude -- never merged with x10^9/L or x10^6/L.
    "x10^12/l": "x10e12/l",
    "x10 12/l": "x10e12/l",
    "x 10^12/l": "x10e12/l",
    "x10*12/l": "x10e12/l",
    "10*12/l": "x10e12/l",
    "10e12/l": "x10e12/l",
    "e12/l": "x10e12/l",
    "x e12/l": "x10e12/l",
    # mmHg -- a real typo in the raw data ("mmHd", ~3.5M vitals rows).
    "mmhd": "mmhg",
    # /100 WBC (differential counts as a fraction of 100 leukocytes).
    "/100 lkc": "/100wbc",
    "/100lkc": "/100wbc",
    "/100 wbc": "/100wbc",
    "/100(wbcs)": "/100wbc",
    "/100 wbc's": "/100wbc",
    "/100 wbcs": "/100wbc",
    # %CV (coefficient of variation, e.g. RDW-CV).
    "cv": "%cv",
    "% cv": "%cv",
}


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded double quotes.

    Duplicated from ``extract_dry.py`` rather than imported -- see this
    module's own docstring on why GEMINI-facing scripts don't cross-import.

    Parameters
    ----------
    name : str
        A table or column name.

    Returns
    -------
    str
        ``name``, double-quoted and safe to interpolate directly into SQL.
    """
    return '"' + name.replace('"', '""') + '"'


def _suppressed(n: int) -> str:
    """Round ``n`` to the nearest 1000, or mask small counts.

    Mirrors ``explore_schema.suppressed_row_count``/``extract_dry._suppressed``
    exactly -- see either for why (small-cell suppression).

    Parameters
    ----------
    n : int
        A real count.

    Returns
    -------
    str
        ``"<6"`` under 6, otherwise rounded to the nearest 1000.
    """
    if n < 6:
        return "<6"
    return str(round(n / 1000) * 1000)


def _parse_datetime_series(raw: pl.Series) -> pl.Series:
    """Vectorized best-effort parse of one of GEMINI's text datetime columns.

    Real format for valid rows isn't confirmed yet (see
    docs/gemini_extraction.md's open question 7), so this keeps pandas'
    flexible, per-element ``format="mixed"`` inference (the same semantics
    the original scalar ``pd.to_datetime(raw, errors="coerce")`` used)
    rather than switching to polars' native ``str.to_datetime``, which
    infers one format from a sample of the column and applies it to every
    row -- a real risk against a column whose format consistency is an
    open question. This is still whole-column vectorized: one
    ``pandas.to_datetime`` call per chunk (a fast, C-level operation on the
    whole column), not a Python function call per row -- the round trip
    through pandas is the deliberate, safe choice here, not a leftover of
    a scalar implementation.

    Parameters
    ----------
    raw : polars.Series
        One text datetime column, one chunk at a time.

    Returns
    -------
    polars.Series
        ``Datetime("us")``, with unparsable/missing entries as ``null``.
    """
    parsed = pd.to_datetime(raw.to_pandas(), errors="coerce", format="mixed")
    return pl.Series(parsed).cast(pl.Datetime("us"))


def _normalize_unit_series(raw: pl.Series) -> pl.Series:
    """Vectorized normalization of a raw unit column for inclusion in a MEDS ``code``.

    Casefolded and whitespace-collapsed, sentinel strings (``"none"``,
    ``"null"``, ``"(null)"``, ``"nan"``, blank -- :data:`_UNIT_SENTINELS`)
    mapped to ``"UNK"``, then a handful of known real-data variant clusters
    (:data:`UNIT_CANONICALIZATION_MAP`) collapsed onto one canonical token
    each -- otherwise the collapsed string itself. GEMINI is multi-hospital
    and the same OMOP concept can carry different units per site (see
    docs/gemini_extraction.md's Units section), so the unit must ride in
    the code itself, the same ``LAB//<id>//<unit>`` shape MIMIC-IV already
    uses, rather than being dropped or assumed constant -- and GEMINI's raw
    unit strings for what's genuinely the *same* unit are themselves wildly
    inconsistent (case, spacing, punctuation, typos), so left as raw text
    they'd fragment one quantile bin into a dozen near-empty ones just as
    badly as a truly different unit would. Mixed units under one code must
    never share a bin; the *same* unit spelled five ways must never scatter
    across five.

    Parameters
    ----------
    raw : polars.Series
        One ``result_unit``/``measurement_unit`` column, one chunk at a
        time.

    Returns
    -------
    polars.Series
        The canonicalized unit, or ``"UNK"``.
    """
    collapsed = (
        raw.cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
    )
    is_sentinel = raw.is_null() | collapsed.is_in(list(_UNIT_SENTINELS)).fill_null(
        False
    )
    mapped = collapsed.replace(UNIT_CANONICALIZATION_MAP)
    return pl.select(
        pl.when(is_sentinel).then(pl.lit("UNK")).otherwise(mapped)
    ).to_series()


def _within_admission_guard_mask(ts: pl.Series, admission: pl.Series) -> pl.Series:
    """Vectorized admission-anchored timestamp guard (see the module docstring).

    Both missing/unparsable fails the guard (``False``) -- there is
    nothing to anchor to. See docs/gemini_extraction.md's open question 7
    for why this exists: pharmacy/radiology timestamps include real,
    physically impossible years, data-entry artifacts rather than genuine
    far-future/far-past events.

    Parameters
    ----------
    ts : polars.Series
        The event timestamp column (already parsed, ``Datetime``).
    admission : polars.Series
        The owning encounter's admission time column (already looked up
        via :func:`fetch_admission_index`'s ``admission_by_genc``,
        ``Datetime``).

    Returns
    -------
    polars.Series
        ``True`` where ``ts`` should be trusted.
    """
    return pl.select(
        ts.is_not_null()
        & admission.is_not_null()
        & ((ts - admission).abs() <= pl.duration(days=GUARD_WINDOW.days))
    ).to_series()


class _StreamAbandonedError(Exception):
    """Internal sentinel: the consumer stopped reading before ``COPY`` finished.

    Raised by :class:`_CopyChunkSink`'s ``write()`` once
    :func:`_stream_table_copy`'s cleanup has requested a stop, to abort
    psycopg2's ``copy_expert`` early rather than let it keep calling
    ``write()`` on a sink nobody is draining anymore -- see
    :func:`_stream_table_copy`'s ``finally`` block. Caught and swallowed on
    the producer thread; never meant to reach a caller.
    """


class _CopyChunkSink:
    """File-like sink for psycopg2's ``copy_expert``, chunking CSV into polars frames.

    ``copy_expert(sql, sink)`` is one synchronous call that blocks until
    the whole ``COPY`` finishes -- but *within* that call, psycopg2 invokes
    ``sink.write(bytes)`` repeatedly as data arrives off the wire, not once
    at the end (``COPY``'s wire protocol is itself chunked). This class is
    that ``sink``: it buffers bytes until it has a complete header line
    plus :data:`CHUNK_ROWS` complete data lines, parses exactly that slice
    with :func:`polars.read_csv`, and puts the resulting
    :class:`polars.DataFrame` onto ``out_queue`` -- so a consumer reading
    from ``out_queue`` on another thread sees bounded-size chunks as they
    become available, instead of the whole table materializing in memory
    at once (a real concern at ``lab_subset``'s ~659M-row scale). See
    :func:`_stream_table_copy` for the producer-thread/consumer-generator
    wiring this exists for.

    **Column constraint, not currently violated but not enforced either:**
    row boundaries are found by counting raw ``\\n`` bytes
    (:meth:`_drain`), not by CSV-aware parsing. A column value containing a
    literal newline inside a CSV-quoted field (Postgres ``COPY`` quotes a
    field containing the delimiter, quote character, or a newline) would
    make this miscount row boundaries and split a single logical row across
    two chunks, silently corrupting both. Every column every
    ``extract_<table>`` function currently selects (ids, codes, units,
    hashed identifiers, datetimes) is free of this risk by construction --
    genuine free-text fields (e.g. ``radiology_subset.imaging_result``) are
    deliberately never selected (see the module docstring's MEDS mapping
    table / ``docs/gemini_extraction.md``). If a future change ever selects
    a free-text column through :func:`_stream_table_copy`, verify first
    that it can't contain a literal newline, or add real CSV-aware
    chunking here instead of the newline-count shortcut.

    Parameters
    ----------
    chunk_rows : int
        Data rows to accumulate before parsing and enqueuing one chunk.
    out_queue : queue.Queue[polars.DataFrame]
        Bounded queue (see :func:`_stream_table_copy`'s ``maxsize``, which
        provides backpressure) that completed chunks are put onto.
    stop_requested : threading.Event
        Checked on every ``write()`` call; once set, ``write()`` raises
        :class:`_StreamAbandonedError` to abort the in-flight ``COPY`` instead
        of continuing to buffer/parse/enqueue for a consumer that has
        already given up (see :func:`_stream_table_copy`'s ``finally``
        block -- this is what makes early abandonment terminate instead of
        deadlocking on a producer blocked on a full, undrained queue).
    """

    def __init__(
        self,
        chunk_rows: int,
        out_queue: "queue.Queue[pl.DataFrame]",
        stop_requested: threading.Event,
    ) -> None:
        self._buffer = bytearray()
        self._chunk_rows = chunk_rows
        self._queue = out_queue
        self._header: Optional[bytes] = None
        self._stop_requested = stop_requested

    def write(self, data: bytes) -> int:
        """Accept one chunk of bytes from psycopg2 as the CSV output arrives."""
        if self._stop_requested.is_set():
            raise _StreamAbandonedError
        self._buffer.extend(data)
        self._drain()
        return len(data)

    def _drain(self) -> None:
        while True:
            if self._header is None:
                idx = self._buffer.find(b"\n")
                if idx == -1:
                    return
                self._header = bytes(self._buffer[: idx + 1])
                del self._buffer[: idx + 1]
                continue
            if self._buffer.count(b"\n") < self._chunk_rows:
                return
            pos = -1
            for _ in range(self._chunk_rows):
                pos = self._buffer.index(b"\n", pos + 1)
            self._parse_and_enqueue(bytes(self._buffer[: pos + 1]))
            del self._buffer[: pos + 1]

    def _parse_and_enqueue(self, body: bytes) -> None:
        assert self._header is not None
        frame = pl.read_csv(io.BytesIO(self._header + body), null_values=["\\N"])
        self._queue.put(frame)

    def close(self) -> None:
        """Flush any final, incomplete-sized chunk still buffered."""
        if self._header is not None and self._buffer:
            self._parse_and_enqueue(bytes(self._buffer))
            self._buffer.clear()


#: Sentinel signaling end-of-stream on :func:`_stream_table_copy`'s internal
#: queue -- a plain ``object()`` rather than ``None``, since ``None`` could
#: never legitimately be mistaken for a real item on this queue but a typed
#: sentinel documents the intent better than relying on that.
_STREAM_DONE = object()

#: Chunks in flight (produced but not yet consumed) before
#: :func:`_stream_table_copy`'s producer thread blocks -- bounds memory to
#: roughly this many chunks regardless of table size.
_STREAM_QUEUE_MAXSIZE = 4


def _stream_table_copy(
    table: str, select_cols: list[str], *, chunk_rows: int = CHUNK_ROWS
) -> Iterator[pl.DataFrame]:
    """Preferred fetch path: one full-table ``COPY ... TO STDOUT`` scan, unordered.

    Runs ``COPY`` on a background thread (psycopg2's ``copy_expert`` is a
    single blocking call with no generator interface of its own) writing
    into a :class:`_CopyChunkSink`, which puts each completed
    :data:`CHUNK_ROWS`-row chunk onto a bounded queue
    (:data:`_STREAM_QUEUE_MAXSIZE`); this generator, running on the calling
    thread, pulls chunks off that queue and yields them, giving true
    bounded-memory streaming (the producer blocks once the queue is full,
    rather than racing ahead and buffering the whole table) without this
    module needing its own OS pipe. Any exception on the producer thread
    (a real DB error, a malformed ``COPY`` response, ...) is captured and
    re-raised here, on the consumer side, once the producer thread has
    finished.

    No bound parameters, no ``WHERE``, no ``ORDER BY`` -- see the module
    docstring's "Fetch strategy" section for why: these matviews have no
    index on ``row_num``, so an ordered read forces a full sort before the
    first row comes back, which is what made the original keyset-paginated
    fetch unusable at real scale. Row order from an unordered ``COPY`` is
    whatever the table's physical storage order happens to be, which
    nothing downstream depends on.

    Parameters
    ----------
    table : str
        Table name, from this module's own hardcoded per-function calls
        (never external input).
    select_cols : list[str]
        Columns to select.
    chunk_rows : int
        Rows per yielded chunk.

    Yields
    ------
    polars.DataFrame
        One chunk at a time, in whatever order the server returns rows.
    """
    cols_sql = ", ".join(_quote_ident(c) for c in dict.fromkeys(select_cols))
    table_sql = _quote_ident(table)
    copy_sql = (
        f"COPY (SELECT {cols_sql} FROM {table_sql}) "
        f"TO STDOUT WITH (FORMAT CSV, HEADER, NULL '\\N')"
    )

    out_queue: "queue.Queue[Any]" = queue.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
    errors: list[BaseException] = []
    stop_requested = threading.Event()

    def _produce() -> None:
        sink = _CopyChunkSink(chunk_rows, out_queue, stop_requested)
        try:
            db.copy_to_sink(copy_sql, sink)
            sink.close()
        except _StreamAbandonedError:
            pass  # consumer gave up early -- not a real error, nothing to report
        except BaseException as exc:  # noqa: BLE001 -- must reach the consumer thread
            errors.append(exc)
        finally:
            out_queue.put(_STREAM_DONE)

    producer = threading.Thread(target=_produce, daemon=True)
    producer.start()
    try:
        while True:
            item = out_queue.get()
            if item is _STREAM_DONE:
                break
            yield item
    finally:
        # Not an unconditional producer.join(): if this generator is being
        # abandoned early (e.g. a caller elsewhere in run_extraction raises
        # mid-table and this generator is garbage-collected/closed without
        # being exhausted), the producer thread can be blocked on
        # out_queue.put() with a full queue and no one left to drain it --
        # join()ing unconditionally here would then hang forever. Setting
        # stop_requested makes the next write() abort the COPY
        # (_StreamAbandonedError), and draining the queue here unblocks any put()
        # already in flight so that abort can actually be observed. On the
        # normal exhaustion path (loop broke on _STREAM_DONE), the queue is
        # already empty and the producer already finished, so this is a
        # no-op there.
        stop_requested.set()
        while producer.is_alive():
            try:
                out_queue.get_nowait()
            except queue.Empty:
                producer.join(timeout=0.1)
    if errors:
        raise errors[0]


def _stream_table_cursor(
    table: str, select_cols: list[str], *, chunk_rows: int = CURSOR_FALLBACK_CHUNK_ROWS
) -> Iterator[pl.DataFrame]:
    """Fallback fetch path: one un-ordered server-side-cursor scan of ``table``.

    Used only when :func:`_stream_table_copy` fails before yielding
    anything (e.g. ``COPY`` isn't permitted for this connection/role) --
    see :func:`_stream_table`. No ``ORDER BY``, same reasoning as
    :func:`_stream_table_copy`.

    Parameters
    ----------
    table : str
        Table name, from this module's own hardcoded per-function calls
        (never external input).
    select_cols : list[str]
        Columns to select.
    chunk_rows : int
        Rows per yielded chunk (psycopg2 cursor ``itersize``, via
        :func:`odyssey.data.gemini.db.stream_query`'s ``chunksize``).

    Yields
    ------
    polars.DataFrame
        One chunk at a time, in whatever order the server returns rows.
    """
    cols_sql = ", ".join(_quote_ident(c) for c in dict.fromkeys(select_cols))
    table_sql = _quote_ident(table)
    sql = f"SELECT {cols_sql} FROM {table_sql}"
    for chunk in db.stream_query(sql, chunksize=chunk_rows):
        if chunk.empty:
            continue
        yield pl.from_pandas(chunk)


def _stream_table(table: str, select_cols: list[str]) -> Iterator[pl.DataFrame]:
    """Stream ``table`` in full, trying :func:`_stream_table_copy` first.

    Falls back to :func:`_stream_table_cursor` only if the ``COPY`` attempt
    fails *before yielding anything* -- once it has started yielding real
    chunks, a later failure is not caught here: silently restarting a large
    table from the cursor path mid-stream would double-count everything
    already written for it. A failure that deep should surface and let
    table-level resumability (see the module docstring's "Resumability"
    section) handle the restart on the next run instead.

    Parameters
    ----------
    table : str
        Table name, from this module's own hardcoded per-function calls
        (never external input).
    select_cols : list[str]
        Columns to select.

    Yields
    ------
    polars.DataFrame
        One chunk at a time, in whatever order the server returns rows.
    """
    copy_gen = _stream_table_copy(table, select_cols)
    try:
        first = next(copy_gen)
    except StopIteration:
        return
    except Exception:
        logger.warning(
            "[extract_meds] COPY fetch failed for %s before yielding any rows; "
            "falling back to a server-side cursor",
            table,
            exc_info=True,
        )
        yield from _stream_table_cursor(table, select_cols)
        return
    yield first
    yield from copy_gen


def fetch_admission_index() -> tuple[dict[int, str], dict[int, Optional[pd.Timestamp]]]:
    """One pass over ``admdad_subset``: ``genc_id -> (subject, admission time)``.

    Built once and held in memory for the whole extraction: every other
    table's rows carry only ``genc_id``, not ``patient_id_hashed`` (see
    docs/gemini_extraction.md's open question 6), and the pharmacy/
    radiology timestamp guard needs the encounter's admission time as its
    anchor. ~2.27M encounters in the schema-exploration cut -- two plain
    ``int``/``str``-keyed dicts, not a reason to re-query per row. Never
    checkpointed -- see the module docstring's "Resumability" section;
    cheap to rebuild in full on every restart.

    Returns
    -------
    tuple[dict[int, str], dict[int, pandas.Timestamp | None]]
        ``(subject_by_genc, admission_by_genc)``.
    """
    subject_by_genc: dict[int, str] = {}
    admission_by_genc: dict[int, Optional[pd.Timestamp]] = {}
    for chunk in _stream_table(
        "admdad_subset", ["genc_id", "patient_id_hashed", "admission_date_time"]
    ):
        frame = chunk.with_columns(
            pl.col("genc_id").cast(pl.Int64),
            _parse_datetime_series(chunk["admission_date_time"]).alias(
                "admission_date_time"
            ),
        )
        gencs = frame["genc_id"].to_list()
        subject_by_genc.update(
            zip(gencs, frame["patient_id_hashed"].cast(pl.Utf8).to_list(), strict=True)
        )
        admission_by_genc.update(
            zip(
                gencs,
                (
                    pd.Timestamp(ts) if ts is not None else None
                    for ts in frame["admission_date_time"].to_list()
                ),
                strict=True,
            )
        )
    return subject_by_genc, admission_by_genc


_LAB_CONCEPT_LOOKUP_SQL = """
SELECT DISTINCT ON ({concept_id}) {concept_id} AS concept_id, {concept_desc} AS concept_desc
FROM {lookup}
WHERE {concept_desc} IS NOT NULL
ORDER BY {concept_id}
"""


def fetch_lab_concept_lookup() -> dict[int, str]:
    """Deduplicated ``concept_id -> concept_desc`` from ``lookup_lab_concept``.

    ``lookup_lab_concept`` has more than one row for at least the
    high-frequency concept ids -- one with a real description, one with a
    ``NULL`` description, same row count either way when naively joined
    (see docs/gemini_extraction.md's open question 4). ``DISTINCT ON``
    picks exactly one (a non-null description, per the ``WHERE``) per id,
    so a caller joining against this dict's keys can't double-count. This
    table is small (one row per distinct lab concept, not per lab result)
    -- unlike every ``extract_<table>`` function's source table, so it
    stays on :func:`odyssey.data.gemini.db.query`'s plain single
    round-trip rather than :func:`_stream_table`.

    Used by :func:`extract_labs` only to drop lab rows whose
    ``test_type_mapped_omop`` has no real mapped concept at all (garbage
    codes); the OMOP -> LOINC translation itself is a later stage (see the
    module docstring), so ``concept_desc`` itself isn't attached to output
    rows -- this dict's keys are what matters here, its values are for a
    later stage's convenience.

    Returns
    -------
    dict[int, str]
        ``{concept_id: concept_desc}``, one entry per id with a real
        description.
    """
    result = db.query(
        _LAB_CONCEPT_LOOKUP_SQL.format(
            concept_id=_quote_ident("concept_id"),
            concept_desc=_quote_ident("concept_desc"),
            lookup=_quote_ident("lookup_lab_concept"),
        )
    )
    return dict(
        zip(result["concept_id"].astype(int), result["concept_desc"].astype(str))
    )


class ExtractedBatch(NamedTuple):
    """One yielded unit of work from an ``extract_<table>`` generator.

    ``source_rows`` is the number of rows read from the source chunk that
    produced ``frame`` -- not the number of MEDS rows in ``frame``, which
    can differ (filtered rows drop some; one source row can produce more
    than one MEDS event, e.g. admission/discharge). Used only for
    :func:`run_extraction`'s progress logging, never for resumability
    (table granularity only -- see the module docstring).
    """

    frame: pd.DataFrame
    source_rows: int


def _finalize_meds_batch(frame: pl.DataFrame) -> pd.DataFrame:
    """Select :data:`MEDS_COLUMNS`, drop rows missing subject/time/code, return pandas.

    A row missing any of subject, time, or code isn't a usable MEDS event
    at all -- dropped here rather than passed downstream as a row of
    nulls the writer would have to special-case. Converts to pandas at
    this one boundary since :class:`MedsShardWriter` (shard split, pyarrow
    write) stays pandas-based -- everything upstream of this call, inside
    each ``extract_<table>`` function, is vectorized polars.

    Parameters
    ----------
    frame : polars.DataFrame
        Must carry exactly :data:`MEDS_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        :data:`MEDS_COLUMNS`, one row per usable event.
    """
    finalized = frame.select(MEDS_COLUMNS).drop_nulls(
        subset=["subject_id", "time", "code"]
    )
    return finalized.to_pandas()


def extract_admissions(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Admission and discharge events from ``admdad_subset``.

    No timestamp guard applied -- ``admdad_subset``'s own timestamps land
    entirely in the plausible 2010-2024 range (confirmed in
    docs/gemini_extraction.md's date-range check), unlike pharmacy/radiology.

    Event shape: one row produces two MEDS rows, ``ADMISSION`` at
    ``admission_date_time`` and ``DISCHARGE`` at ``discharge_date_time``,
    both ``hadm_id = genc_id``, no ``numeric_value``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "admdad_subset", ["genc_id", "admission_date_time", "discharge_date_time"]
    ):
        frame = chunk.with_columns(pl.col("genc_id").cast(pl.Int64))
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admitted = pl.DataFrame(
            {
                "subject_id": subject,
                "time": genc.replace_strict(
                    admission_by_genc, default=None, return_dtype=pl.Datetime("us")
                ),
                "code": "ADMISSION",
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        discharged = pl.DataFrame(
            {
                "subject_id": subject,
                "time": _parse_datetime_series(frame["discharge_date_time"]),
                "code": "DISCHARGE",
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        meds = pl.concat([admitted, discharged])
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_icu(subject_by_genc: dict[int, str]) -> Iterator[ExtractedBatch]:
    """ICU admission/discharge events from ``ipscu_subset``.

    Only rows where ``icu_flag`` is true count as ICU specifically --
    ``scu_unit_number`` implies non-ICU special-care units exist too (see
    docs/gemini_extraction.md).

    Event shape: one ``icu_flag`` row produces two MEDS rows,
    ``ICU_ADMISSION`` at ``scu_admit_date_time`` and ``ICU_DISCHARGE`` at
    ``scu_discharge_date_time``, both ``hadm_id = genc_id``, no
    ``numeric_value``.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "ipscu_subset",
        ["genc_id", "scu_admit_date_time", "scu_discharge_date_time", "icu_flag"],
    ):
        frame = chunk.filter(pl.col("icu_flag").cast(pl.Boolean)).with_columns(
            pl.col("genc_id").cast(pl.Int64)
        )
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admitted = pl.DataFrame(
            {
                "subject_id": subject,
                "time": _parse_datetime_series(frame["scu_admit_date_time"]),
                "code": "ICU_ADMISSION",
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        discharged = pl.DataFrame(
            {
                "subject_id": subject,
                "time": _parse_datetime_series(frame["scu_discharge_date_time"]),
                "code": "ICU_DISCHARGE",
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        meds = pl.concat([admitted, discharged])
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_labs(
    subject_by_genc: dict[int, str], lab_concepts: dict[int, str]
) -> Iterator[ExtractedBatch]:
    """Lab result events from ``lab_subset``.

    ``code`` is ``LAB//<omop_id>//<unit>`` -- the raw OMOP concept id plus
    the normalized unit (:func:`_normalize_unit_series`), the same
    ``LAB//<itemid>//<unit>`` shape MIMIC-IV already uses. GEMINI is
    multi-hospital and the same concept can carry different units per
    site (see docs/gemini_extraction.md's Units section) -- the unit rides
    in the code itself so the per-unit clinical ranges can key on the
    prefix and mixed-unit values never share a quantile bin. Not yet
    translated to LOINC, see the module docstring. Rows whose
    ``test_type_mapped_omop`` has no real mapped concept at all (not in
    ``lab_concepts``, the deduplicated lookup from
    :func:`fetch_lab_concept_lookup`) are dropped, not extracted as
    garbage codes.

    Event shape: one row produces one MEDS row, ``time`` =
    ``collection_date_time``, ``numeric_value`` from ``result_value``
    (categorical/unparsable results still produce a row, just with a null
    ``numeric_value``), ``hadm_id = genc_id``. No timestamp guard --
    ``lab_subset`` timestamps aren't documented as having the
    pharmacy/radiology outlier problem.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.
    lab_concepts : dict
        From :func:`fetch_lab_concept_lookup`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "lab_subset",
        [
            "genc_id",
            "test_type_mapped_omop",
            "result_value",
            "result_unit",
            "collection_date_time",
        ],
    ):
        frame = chunk.with_columns(
            pl.col("genc_id").cast(pl.Int64),
            pl.col("test_type_mapped_omop").cast(pl.Int64),
        ).filter(pl.col("test_type_mapped_omop").is_in(list(lab_concepts)))
        genc = frame["genc_id"]
        concept = frame["test_type_mapped_omop"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": _parse_datetime_series(frame["collection_date_time"]),
                "code": "LAB//"
                + concept.cast(pl.Utf8)
                + "//"
                + _normalize_unit_series(frame["result_unit"]),
                "numeric_value": frame["result_value"].cast(pl.Float64, strict=False),
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_vitals(subject_by_genc: dict[int, str]) -> Iterator[ExtractedBatch]:
    """Vital-sign events from ``vitals_subset``.

    ``code`` is ``VITALS//<omop_id>//<unit>`` -- same reasoning as
    :func:`extract_labs`: GEMINI is multi-hospital, units can differ per
    site for the same concept, so the normalized unit
    (:func:`_normalize_unit_series`) rides in the code itself. No dedup
    step needed here -- ``lookup_vitals_concept`` (unlike
    ``lookup_lab_concept``) doesn't exhibit the duplicate-row issue at the
    scale checked so far (see docs/gemini_extraction.md's open question 3).

    Event shape: one row produces one MEDS row, ``time`` =
    ``measure_date_time``, ``numeric_value`` from ``measurement_value``,
    ``hadm_id = genc_id``. Rows with no mapped concept
    (``measurement_mapped_omop`` null) are dropped.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "vitals_subset",
        [
            "genc_id",
            "measurement_mapped_omop",
            "measurement_value",
            "measurement_unit",
            "measure_date_time",
        ],
    ):
        frame = chunk.with_columns(
            pl.col("genc_id").cast(pl.Int64),
            pl.col("measurement_mapped_omop").cast(pl.Int64),
        ).filter(pl.col("measurement_mapped_omop").is_not_null())
        genc = frame["genc_id"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": _parse_datetime_series(frame["measure_date_time"]),
                "code": "VITALS//"
                + frame["measurement_mapped_omop"].cast(pl.Utf8)
                + "//"
                + _normalize_unit_series(frame["measurement_unit"]),
                "numeric_value": frame["measurement_value"].cast(
                    pl.Float64, strict=False
                ),
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_pharmacy(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Medication start/end events from ``pharmacy_subset``.

    ``code`` uses ``med_id_generic_name_raw`` as the identity (the
    RxNorm/ingredient bridge is a later stage, see the module docstring
    and docs/gemini_extraction.md's open question 5). **The admission
    guard applies here** (:func:`_within_admission_guard_mask`): real
    timestamps in this table include physically impossible years
    (1930-9022, 1840-8186) -- a start/end time outside +-1y of the
    encounter's admission is dropped, not extracted as a nonsense event.

    Event shape: one row with a non-blank drug name produces up to two MEDS
    rows, ``MEDICATION//<name>//started`` at ``med_start_date_time`` and
    ``MEDICATION//<name>//ended`` at ``med_end_date_time`` -- each dropped
    independently if its own timestamp fails the guard (one side of a row
    can pass while the other fails).

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "pharmacy_subset",
        [
            "genc_id",
            "med_id_generic_name_raw",
            "med_start_date_time",
            "med_end_date_time",
        ],
    ):
        frame = chunk.with_columns(pl.col("genc_id").cast(pl.Int64)).filter(
            pl.col("med_id_generic_name_raw").is_not_null()
            & (pl.col("med_id_generic_name_raw").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admission = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        name = frame["med_id_generic_name_raw"].cast(pl.Utf8)
        started_time = _parse_datetime_series(frame["med_start_date_time"])
        ended_time = _parse_datetime_series(frame["med_end_date_time"])

        started = pl.DataFrame(
            {
                "subject_id": subject,
                "time": started_time,
                "code": "MEDICATION//" + name + "//started",
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(_within_admission_guard_mask(started_time, admission))
        ended = pl.DataFrame(
            {
                "subject_id": subject,
                "time": ended_time,
                "code": "MEDICATION//" + name + "//ended",
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(_within_admission_guard_mask(ended_time, admission))
        meds = pl.concat([started, ended])
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_diagnoses(subject_by_genc: dict[int, str]) -> Iterator[ExtractedBatch]:
    """Diagnosis events from ``ipdiagnosis_subset``.

    ``code`` is the raw ICD-10-CA code, namespaced (``DIAGNOSIS//<code>``)
    -- no event-level timestamp exists on this table (diagnoses are coded
    at the encounter level), so ``time`` is the encounter's discharge time
    (a diagnosis is a fact about the whole stay, attributed at its close,
    the same convention MIMIC-IV/eICU discharge diagnoses already use).
    The discharge-time index below is rebuilt in full on every call --
    never checkpointed, same as :func:`fetch_admission_index` (see the
    module docstring's "Resumability" section).

    Event shape: one row with a non-blank diagnosis code produces one MEDS
    row at the encounter's discharge time, ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    discharge_by_genc: dict[int, Optional[pd.Timestamp]] = {}
    for chunk in _stream_table("admdad_subset", ["genc_id", "discharge_date_time"]):
        frame = chunk.with_columns(
            pl.col("genc_id").cast(pl.Int64),
            _parse_datetime_series(chunk["discharge_date_time"]).alias(
                "discharge_date_time"
            ),
        )
        discharge_by_genc.update(
            zip(
                frame["genc_id"].to_list(),
                (
                    pd.Timestamp(ts) if ts is not None else None
                    for ts in frame["discharge_date_time"].to_list()
                ),
                strict=True,
            )
        )

    for chunk in _stream_table("ipdiagnosis_subset", ["genc_id", "diagnosis_code"]):
        frame = chunk.with_columns(pl.col("genc_id").cast(pl.Int64)).filter(
            pl.col("diagnosis_code").is_not_null()
            & (pl.col("diagnosis_code").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": genc.replace_strict(
                    discharge_by_genc, default=None, return_dtype=pl.Datetime("us")
                ),
                "code": "DIAGNOSIS//" + frame["diagnosis_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_procedures(subject_by_genc: dict[int, str]) -> Iterator[ExtractedBatch]:
    """Procedure events from ``ipintervention_subset``.

    ``code`` is the raw CCI code, namespaced (``PROCEDURE//<code>``), timed
    at ``intervention_episode_start_date_time``.

    Event shape: one row with a non-blank intervention code produces one
    MEDS row, ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "ipintervention_subset",
        ["genc_id", "intervention_code", "intervention_episode_start_date_time"],
    ):
        frame = chunk.with_columns(pl.col("genc_id").cast(pl.Int64)).filter(
            pl.col("intervention_code").is_not_null()
            & (pl.col("intervention_code").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": _parse_datetime_series(
                    frame["intervention_episode_start_date_time"]
                ),
                "code": "PROCEDURE//" + frame["intervention_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_radiology(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Imaging events from ``radiology_subset``.

    ``code`` combines modality and body part (``IMAGING//<modality>//
    <body_part>``), timed at ``performed_date_time``. **The admission
    guard applies here too** (:func:`_within_admission_guard_mask`) --
    ``performed_date_time`` includes years up to 9999 (see
    docs/gemini_extraction.md's open question 7), same fix as pharmacy.

    Event shape: one row produces one MEDS row if its ``performed_date_time``
    passes the admission guard; missing ``modality_mapped``/
    ``body_part_mapped`` fall back to ``"UNKNOWN"`` rather than dropping
    the row (unlike the guard, a missing modality/body part doesn't make
    the row unusable).

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "radiology_subset",
        ["genc_id", "modality_mapped", "body_part_mapped", "performed_date_time"],
    ):
        frame = chunk.with_columns(pl.col("genc_id").cast(pl.Int64))
        genc = frame["genc_id"]
        admission = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        ts = _parse_datetime_series(frame["performed_date_time"])
        modality = frame["modality_mapped"].cast(pl.Utf8).fill_null("UNKNOWN")
        body_part = frame["body_part_mapped"].cast(pl.Utf8).fill_null("UNKNOWN")
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": ts,
                "code": "IMAGING//" + modality + "//" + body_part,
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(_within_admission_guard_mask(ts, admission))
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_providers(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Extract provider (physician) events from ``physicians_subset``.

    Not consumed by any current stage -- extracted anyway to keep a
    tabled, not abandoned, physician-preference IV study option-preserving
    at near-zero cost (one more unordered ~2.27M-row table scan while this
    module is already being touched) rather than needing a whole separate
    extraction pass through GEMINI later if that study is ever picked back
    up. See docs/gemini_extraction.md's "Why provider ids are preserved".

    Event shape: one row can produce up to three MEDS rows, one per
    non-null hashed physician id -- ``PROVIDER//MRP//<hash>``
    (``mrp_cpso_hashed``, most responsible physician),
    ``PROVIDER//ADMITTING//<hash>`` (``adm_phy_cpso_hashed``), and
    ``PROVIDER//DISCHARGING//<hash>`` (``dis_phy_cpso_hashed``) -- all at
    the encounter's admission time (no event-level timestamp of its own,
    same convention as :func:`extract_diagnoses`'s discharge-time
    attribution), ``hadm_id = genc_id``. Ids are already hashed upstream
    (``character varying(64)``, GEMINI's own de-identification) -- passed
    through as-is, same as every other raw-identifier code in this module.
    Null hashed ids (real: 1-11% of rows per role, see
    docs/gemini_extraction.md's MEDS mapping table) are skipped, not
    extracted as empty/placeholder events.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "physicians_subset",
        ["genc_id", "mrp_cpso_hashed", "adm_phy_cpso_hashed", "dis_phy_cpso_hashed"],
    ):
        frame = chunk.with_columns(pl.col("genc_id").cast(pl.Int64))
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        time = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        roles = [
            ("MRP", "mrp_cpso_hashed"),
            ("ADMITTING", "adm_phy_cpso_hashed"),
            ("DISCHARGING", "dis_phy_cpso_hashed"),
        ]
        meds = pl.concat(
            [
                pl.DataFrame(
                    {
                        "subject_id": subject,
                        "time": time,
                        "code": f"PROVIDER//{role}//" + frame[column].cast(pl.Utf8),
                        "numeric_value": None,
                        "hadm_id": genc,
                    }
                )
                for role, column in roles
            ]
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


_SUBJECT_COUNT_SQL = "SELECT COUNT(DISTINCT {column}) AS n FROM {table}"


def count_distinct_subjects() -> int:
    """One ``COUNT(DISTINCT patient_id_hashed)`` on ``admdad_subset``.

    Cheap and targeted -- used by :func:`preflight_shard_capacity` to size
    the shard count before paying for the full admission index
    (:func:`fetch_admission_index`, which reads every column of every row).

    Returns
    -------
    int
        The real (unsuppressed) distinct subject count.
    """
    result = db.query(
        _SUBJECT_COUNT_SQL.format(
            column=_quote_ident("patient_id_hashed"),
            table=_quote_ident("admdad_subset"),
        )
    )
    return int(result["n"].iloc[0])


def preflight_shard_capacity(
    n_subjects: int, *, subjects_per_shard: int = SUBJECTS_PER_SHARD
) -> int:
    """Compute the shard count and make sure this process can open that many files.

    :class:`MedsShardWriter` keeps one Parquet writer (one open file
    handle) open per shard for the whole run -- a real constraint once
    ``n_subjects / subjects_per_shard`` climbs past a typical 1024 default
    soft ``NOFILE`` limit. Tries to raise the soft limit toward the hard
    limit first (``resource.setrlimit``, works without root as long as the
    hard limit itself is high enough); if that's not enough, fails loudly
    with the exact command to run before retrying, rather than dying deep
    into a multi-hour extraction on file descriptor #1025.

    Parameters
    ----------
    n_subjects : int
        From :func:`count_distinct_subjects`.
    subjects_per_shard : int
        Same default as :func:`assign_shards`.

    Returns
    -------
    int
        The shard count that will be used.

    Raises
    ------
    RuntimeError
        If even the hard limit doesn't allow enough headroom -- the
        message names the exact ``ulimit -n`` to run first.
    """
    n_shards = _shard_count(n_subjects, subjects_per_shard)
    needed = n_shards + FD_HEADROOM

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < needed:
        target = min(needed, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    if soft < needed:
        raise RuntimeError(
            f"Need at least {needed} open file descriptors for {n_shards} shards "
            f"(+{FD_HEADROOM} headroom), but the hard NOFILE limit ({hard}) won't "
            f"allow raising the soft limit ({soft}) that high. Run this in your "
            f"shell before retrying:\n\n"
            f"    ulimit -n {needed}\n"
        )
    logger.info(
        "[extract_meds] %d subjects -> %d shards, NOFILE soft limit %d (needed %d)",
        n_subjects,
        n_shards,
        soft,
        needed,
    )
    return n_shards


def _shard_count(n_subjects: int, subjects_per_shard: int = SUBJECTS_PER_SHARD) -> int:
    """``ceil(n_subjects / subjects_per_shard)``, at least 1.

    Shared by :func:`assign_shards` and :func:`preflight_shard_capacity` so
    the two never compute a different shard count for the same subject
    universe.
    """
    return max(1, -(-n_subjects // subjects_per_shard))


def assign_shards(
    subject_ids: Iterable[str], *, subjects_per_shard: int = SUBJECTS_PER_SHARD
) -> dict[str, int]:
    """Stable hash-based subject -> shard assignment.

    Deterministic (SHA-256 of the subject id, not Python's per-process-
    salted ``hash()``) -- re-running the extraction assigns the same
    subject to the same shard every time, which reproducible shards
    require.

    Parameters
    ----------
    subject_ids : Iterable[str]
        Every subject id the extraction will see (typically
        ``fetch_admission_index``'s ``subject_by_genc`` values).
    subjects_per_shard : int
        Target subjects per shard; shard count is
        ``ceil(n_subjects / subjects_per_shard)``, at least 1.

    Returns
    -------
    dict[str, int]
        ``{subject_id: shard_index}``.
    """
    unique_subjects = sorted(set(subject_ids))
    n_shards = _shard_count(len(unique_subjects), subjects_per_shard)
    return {
        subject_id: int(hashlib.sha256(subject_id.encode()).hexdigest(), 16) % n_shards
        for subject_id in unique_subjects
    }


class MedsShardWriter:
    """Streaming per-shard MEDS Parquet writer.

    Memory use is bounded by one incoming batch plus one open Parquet
    writer per shard -- nothing accumulates across the whole run, which
    matters at ``lab_subset``'s/``vitals_subset``'s real scale (hundreds of
    millions of rows). One open file handle per shard for the whole
    extraction is a real, deliberate tradeoff: with thousands of subjects
    per :data:`SUBJECTS_PER_SHARD`\\ =1000 shard, a large subject universe
    could need `ulimit -n` raised above a typical default (1024) -- flagged
    here rather than solved by a multi-pass design, which would need
    re-scanning every source table once per shard group at real scale, a
    far worse tradeoff.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to write ``shard_{i:04d}.parquet`` files into (created if
        missing). Never a git-tracked path -- see the module docstring.
    shard_by_subject : dict[str, int]
        From :func:`assign_shards`.
    """

    def __init__(self, output_dir: Path, shard_by_subject: dict[str, int]) -> None:
        self.output_dir = output_dir
        self.shard_by_subject = shard_by_subject
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._shard_row_counts: dict[int, int] = {}
        self.rows_written_per_table: dict[str, int] = {}
        self.rows_dropped_unshardable = 0

    def _writer_for(self, shard: int) -> pq.ParquetWriter:
        if shard not in self._writers:
            path = self.output_dir / f"shard_{shard:04d}.parquet"
            self._writers[shard] = pq.ParquetWriter(str(path), MEDS_ARROW_SCHEMA)
            self._shard_row_counts[shard] = 0
        return self._writers[shard]

    def write_batch(self, table: str, batch: pd.DataFrame) -> None:
        """Split one MEDS-shaped batch by shard and stream it to disk.

        Parameters
        ----------
        table : str
            Source table name, for the per-table row-count tally in the
            eventual summary -- not written to the Parquet itself.
        batch : pandas.DataFrame
            Must carry exactly :data:`MEDS_COLUMNS`.
        """
        if batch.empty:
            return
        shard_ids = batch["subject_id"].map(self.shard_by_subject)
        unmatched = shard_ids.isna()
        if unmatched.any():
            n_unmatched = int(unmatched.sum())
            self.rows_dropped_unshardable += n_unmatched
            logger.warning(
                "%d rows for %s had a subject_id with no shard assignment, dropping",
                n_unmatched,
                table,
            )
            batch = batch.loc[~unmatched]
            shard_ids = shard_ids.loc[~unmatched]
        if batch.empty:
            return
        self.rows_written_per_table[table] = self.rows_written_per_table.get(
            table, 0
        ) + len(batch)
        for shard, group in batch.groupby(shard_ids.astype(int)):
            arrow_table = pa.Table.from_pandas(
                group[MEDS_COLUMNS], schema=MEDS_ARROW_SCHEMA, preserve_index=False
            )
            self._writer_for(int(shard)).write_table(arrow_table)
            self._shard_row_counts[int(shard)] = self._shard_row_counts.get(
                int(shard), 0
            ) + len(group)

    def close(self) -> dict[int, int]:
        """Close every open shard writer and return per-shard row counts.

        Returns
        -------
        dict[int, int]
            ``{shard_index: row_count}``.
        """
        for writer in self._writers.values():
            writer.close()
        return dict(self._shard_row_counts)


def _manifest_path(output_dir: Path) -> Path:
    """Return the resume manifest's path (see the module docstring's Resumability)."""
    return output_dir / "extract_manifest.json"


def _load_manifest(output_dir: Path) -> dict[str, str]:
    """Load the resume manifest, or ``{}`` if this is a fresh run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Same directory :class:`MedsShardWriter` writes shards into.

    Returns
    -------
    dict[str, str]
        ``{table: "complete"}`` -- table-granularity only, see the module
        docstring's "Resumability" section.
    """
    path = _manifest_path(output_dir)
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()))


def _save_manifest(output_dir: Path, manifest: dict[str, str]) -> None:
    """Atomically write the resume manifest (write-tmp-then-rename).

    Called once a table's extractor is fully drained and its output fully
    written (see the module docstring's "Resumability" section) -- atomic
    replace means a crash mid-write never leaves a truncated/corrupt
    manifest behind for the next restart to choke on.

    Parameters
    ----------
    output_dir : pathlib.Path
        Same directory :class:`MedsShardWriter` writes shards into.
    manifest : dict[str, str]
        The manifest to persist.
    """
    path = _manifest_path(output_dir)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_path.replace(path)


def _log_table_progress(table: str) -> Any:
    """Return a callback that logs cumulative rows/rows-per-second for ``table``.

    Called by :func:`run_extraction` after every batch with that batch's
    ``source_rows`` -- logs at most once per
    :data:`PROGRESS_LOG_INTERVAL_SECONDS` (not every batch, which at
    :data:`CHUNK_ROWS`/:data:`CURSOR_FALLBACK_CHUNK_ROWS` granularity could
    still be many times a minute) so a long-running extraction's log stays
    readable rather than flooded.

    Parameters
    ----------
    table : str
        Table name, for the log line only.

    Returns
    -------
    Callable[[int], None]
        Call with each batch's ``source_rows``.
    """
    start = time.monotonic()
    last_logged = start
    state = {"rows": 0}

    def _report(source_rows: int) -> None:
        nonlocal last_logged
        state["rows"] += source_rows
        now = time.monotonic()
        if now - last_logged < PROGRESS_LOG_INTERVAL_SECONDS:
            return
        last_logged = now
        elapsed = now - start
        rate = state["rows"] / elapsed if elapsed > 0 else 0.0
        logger.info(
            "[extract_meds] %s: %d source rows read so far (%.0f rows/s)",
            table,
            state["rows"],
            rate,
        )

    return _report


def run_extraction(output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Run the full GEMINI -> MEDS extraction and write the suppressed summary.

    Resumable at table granularity: see the module docstring's
    "Resumability" section. A table already marked ``"complete"`` in
    ``<output_dir>/extract_manifest.json`` is skipped entirely; any other
    table is (re-)run from scratch.

    Parameters
    ----------
    output_dir : pathlib.Path, optional
        Overrides :data:`OUTPUT_DIR` (mainly for tests).

    Returns
    -------
    dict[str, Any]
        The summary that gets written to :data:`SUMMARY_PATH` --
        ``{"n_subjects", "n_shards", "rows_per_table", "shard_row_counts",
        "rows_dropped_unshardable"}``, all counts suppressed via
        :func:`_suppressed`.
    """
    target_dir = output_dir if output_dir is not None else OUTPUT_DIR
    manifest = _load_manifest(target_dir)

    logger.info("[extract_meds] counting distinct subjects for the preflight check...")
    n_subjects_estimate = count_distinct_subjects()
    preflight_shard_capacity(n_subjects_estimate)

    logger.info("[extract_meds] building admission index...")
    subject_by_genc, admission_by_genc = fetch_admission_index()
    logger.info("[extract_meds] %d encounters indexed", len(subject_by_genc))

    logger.info("[extract_meds] fetching deduplicated lab concept lookup...")
    lab_concepts = fetch_lab_concept_lookup()

    shard_by_subject = assign_shards(subject_by_genc.values())
    n_shards = len(set(shard_by_subject.values()))
    logger.info(
        "[extract_meds] %d subjects -> %d shards", len(shard_by_subject), n_shards
    )

    writer = MedsShardWriter(target_dir, shard_by_subject)
    table_generators: list[tuple[str, Iterator[ExtractedBatch]]] = [
        ("admdad_subset", extract_admissions(subject_by_genc, admission_by_genc)),
        ("ipscu_subset", extract_icu(subject_by_genc)),
        ("lab_subset", extract_labs(subject_by_genc, lab_concepts)),
        ("vitals_subset", extract_vitals(subject_by_genc)),
        ("pharmacy_subset", extract_pharmacy(subject_by_genc, admission_by_genc)),
        ("ipdiagnosis_subset", extract_diagnoses(subject_by_genc)),
        ("ipintervention_subset", extract_procedures(subject_by_genc)),
        ("radiology_subset", extract_radiology(subject_by_genc, admission_by_genc)),
        ("physicians_subset", extract_providers(subject_by_genc, admission_by_genc)),
    ]
    for table_name, generator in table_generators:
        if manifest.get(table_name) == "complete":
            logger.info("[extract_meds] %s already complete, skipping", table_name)
            continue
        logger.info("[extract_meds] extracting %s...", table_name)
        report_progress = _log_table_progress(table_name)
        for batch, source_rows in generator:
            writer.write_batch(table_name, batch)
            report_progress(source_rows)
        manifest[table_name] = "complete"
        _save_manifest(target_dir, manifest)

    shard_row_counts = writer.close()
    summary = {
        "n_subjects": _suppressed(len(shard_by_subject)),
        "n_shards": n_shards,
        "rows_per_table": {
            table: _suppressed(n) for table, n in writer.rows_written_per_table.items()
        },
        "shard_row_counts": {
            str(shard): _suppressed(n) for shard, n in shard_row_counts.items()
        },
        "rows_dropped_unshardable": _suppressed(writer.rows_dropped_unshardable),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("[extract_meds] wrote %s", SUMMARY_PATH)
    return summary


def main() -> None:
    """Run the extraction and print where the summary landed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    summary = run_extraction()
    print(f"Wrote {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
