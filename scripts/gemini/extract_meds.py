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
instead of many small requeries. A real run of *that* immediately
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

Real-run hardening (2026-08-21, second incident): a real run hung silently
for ~48 minutes on "building admission index...", root-caused to two
compounding issues, both fixed. First, :func:`fetch_admission_index` (and
the discharge-time index inside :func:`extract_diagnoses`) streamed
``admdad_subset`` with no progress output at all between the initial log
line and completion -- now wrapped in :func:`_log_table_progress`, same as
every primary table, so a long pass is never silent. Second, every
``genc_id`` column was cast with a hard ``pl.col("genc_id").cast(pl.Int64)``
that raises on any row with an unparsable value -- that alone would just
be a crash, but combined with the (separately fixed) producer-thread
deadlock in :func:`_stream_table_copy`, an exception raised mid-chunk could
leave the whole extraction hung with no traceback ever printed, rather than
failing loudly. :func:`_filter_valid_genc_id` replaces every such cast: a
row with an unparsable (not just missing) ``genc_id`` is dropped and
logged, not allowed to crash the batch -- the same "one bad row must not
kill the whole run" principle already applied to every other messy field in
this module.

Real-run hardening (2026-08-21, third incident): the relaunch under the
above two fixes surfaced three more real issues, all fixed together.
First, ``ipscu_subset.icu_flag`` arrived as ``t``/``f`` text (Postgres
``COPY``'s own boolean encoding) -- ``pl.col("icu_flag").cast(pl.Boolean)``
crashed :func:`extract_icu` outright, since polars has no supported
``Utf8 -> Boolean`` cast at all; :func:`_coerce_boolean_flag` replaces it
with a lenient membership map. Second, a genuine correctness bug in
resumability: :class:`MedsShardWriter` unconditionally reopened
``shard_{i:04d}.parquet`` on first touch each run, and ``pq.ParquetWriter``
silently truncates whatever file it opens -- since a resumed run
constructs a brand-new writer instance and may skip an already-complete
table's generator while re-running an incomplete one, any shard touched by
both a completed and a re-run table would have the completed table's rows
silently destroyed. :func:`_next_shard_write_path` now never reopens an
existing file, writing a ``_partN`` file instead -- **a logical shard is
the union of its base file and every ``_partN`` file** (see
:func:`_logical_shard_row_counts`, which any accounting must use instead
of assuming one file per shard). Third, throughput: a real run sustained
only ~2,000 rows/s through the transform+write path (vs. ~280k rows/s for
the pure ``COPY``+index-building pass over the same table), which would
have made ``lab_subset``'s ~659M rows take days. Local benchmarking found
the in-process parse/transform/write cost negligible against a local
disk, isolating the real cost to :class:`MedsShardWriter` making one small
``write_table()`` call per shard per chunk against GEMINI's NFS-mounted
output directory -- fixed by buffering each shard's rows in memory and
flushing only once :data:`SHARD_FLUSH_ROW_THRESHOLD` rows have
accumulated, collapsing the write-call count by roughly that factor.
:func:`_parse_datetime_series` also gained a vectorized ISO fast path
(measured negligible for this bottleneck specifically, but a genuine,
correctness-preserving speedup kept regardless), and
:func:`run_extraction` now logs a per-batch parse/transform/write timing
line so a real run's phase split is measured directly, not inferred.

Real-run hardening (2026-08-21, fourth incident): the relaunch under the
throughput fix above died in :meth:`_CopyChunkSink._parse_and_enqueue`
with a ``polars.ComputeError`` casting ``'103@POST'`` to ``f64`` in
``lab_subset.result_value``. Root cause: ``pl.read_csv`` infers each
column's dtype from a sample *of that chunk* -- early ``lab_subset``
chunks were all-numeric ``result_value`` values, locking ``Float64`` for
that chunk, until the stream reached a free-text result region within the
same chunk. Fixed at the actual source, not with a per-column patch:
``_parse_and_enqueue`` now reads every column as ``Utf8``
(``infer_schema_length=0``) -- all typed parsing already happens
downstream, in each ``extract_<table>``'s own explicit ``.cast()`` calls,
so this class was never supposed to be inferring types at all. That also
retired the implicit type safety net a numeric-looking chunk used to
provide for two other hard casts (``lab_subset.test_type_mapped_omop``,
``vitals_subset.measurement_mapped_omop``) -- both now lenient
(``strict=False``) with :func:`_warn_on_unparsable_int_cast` logging any
non-null value that fails to parse, the same principle as
:func:`_filter_valid_genc_id`. Confirming ``result_value`` is genuine free
text also retired :class:`_CopyChunkSink`'s former "no selected column can
contain a newline, by construction" assumption -- see that class's updated
docstring and :func:`_select_expr_sql` for the server-side fix (every
``text``/``character varying`` column is newline-stripped and length-capped
in the ``SELECT`` itself now, auditing all 9 tables by Postgres column
type rather than by inspection). :data:`_SINK_BUFFER_BYTE_CAP` adds a
byte-size flush trigger alongside the row-count one as further insurance
against any future large region this module hasn't already bounded.

Real-run hardening (2026-08-21, fifth incident): the newly-added ER
extraction landed only 90k of ~2.94M coded ``erintervention_subset`` rows.
Root cause, confirmed server-side: ``intervention_episode_start_date_time``
is blank (empty string, not SQL ``NULL``) on 96.7% of coded rows --
invisible to ``extract_dry.py``'s old null counting (a plain ``COUNT(*) -
COUNT(column)``, which counts a blank string as "present," not null; fixed
there too, see that module). :func:`extract_er_procedures_untimed` rescues
the complement -- every coded row :func:`extract_er_procedures`'s own
admission-guard predicate rejects (blank/unparsable timestamp, or outside
the guard window), attributed to admission time instead. The extractor
itself was never at fault here; this is a genuine data-shape discovery,
not a bug fix in the fetch/parsing layers those first four incidents
covered.

Real-run hardening (2026-08-21, sixth incident): ``vitals_subset``'s own
71% retention turned out to be a different, larger mechanism than the
fifth incident's -- measured blank timestamps there are only ~44k. Real
cause: :func:`extract_vitals` drops every row whose
``measurement_mapped_omop`` doesn't resolve to a concept id, ~119M of
~412M rows. :func:`extract_vitals_unmapped` rescues the exact complement
via ``measurement_name`` as a fallback identity (the same convention
eICU's own extraction already uses for an unmapped vital/lab) --
:func:`_normalize_name_series` handles the name/value text (casefold +
whitespace-collapse only, no cross-name canonicalization map yet, unlike
:func:`_normalize_unit_series`'s unit handling). Closes every retention
anomaly found in the real-run accounting so far.

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
import re
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

#: Byte-size insurance ceiling on :class:`_CopyChunkSink`'s buffer -- see
#: its ``_drain`` for how this bounds memory even if row *count* stays
#: below :data:`CHUNK_ROWS` but per-row byte size is unexpectedly large.
_SINK_BUFFER_BYTE_CAP = 256 * 1024 * 1024  # 256 MiB

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


#: Columns selected anywhere in this module whose Postgres column type is
#: ``integer``/``boolean`` (confirmed against ``scripts/gemini/out/schema.md``)
#: -- these can never contain an embedded newline by construction and are
#: selected as bare identifiers. Every other selected column across all 9
#: source tables is ``text``/``character varying`` -- including every
#: datetime column and every ``character varying(64)`` hashed-id column,
#: since a length bound doesn't exclude newline *content* -- and gets
#: :func:`_select_expr_sql`'s newline-stripping, length-capping wrap by
#: default. Keep this set in sync if a future extractor selects a new
#: genuinely numeric/boolean column; anything not listed here is wrapped,
#: which is harmless (a no-op) for a column that happens to be safe anyway.
_NEWLINE_SAFE_COLUMNS = frozenset(
    {
        "genc_id",
        "test_type_mapped_omop",
        "measurement_mapped_omop",
        "icu_flag",
        "in_hospital_mortality_derived",
        "discharge_disposition",
    }
)

#: Max characters kept per text/varchar column after
#: :func:`_select_expr_sql`'s newline-stripping -- generous for any
#: numeric+unit reading, medical code, hashed identifier, or datetime
#: string actually used downstream (see docs/gemini_extraction.md's
#: truncation note); none of those are remotely close to this length in
#: practice.
_SELECTED_COLUMN_MAX_CHARS = 128


def _select_expr_sql(column: str) -> str:
    """Build SQL for one selected column, closing the newline-in-CSV-field hazard.

    Real incident this closes: every ``text``/``character varying`` column
    GEMINI exposes can contain a literal newline regardless of its declared
    length bound, and :class:`_CopyChunkSink` finds row boundaries by
    counting raw ``\\n`` bytes, not by CSV-aware parsing -- a literal
    newline inside a quoted CSV field would silently corrupt row alignment.
    This module's earlier assumption that no selected column could contain
    one "by construction" was wrong -- ``lab_subset.result_value`` is a
    real, demonstrated counterexample (see the module docstring's third
    real-run-hardening entry). Every ``text``/``character varying`` column
    is rewritten server-side to strip embedded newlines/carriage returns
    and cap length -- both a correctness fix and a transfer-volume win,
    since a free-text region no longer needs to cross the wire in full.
    Columns confirmed ``integer``/``boolean``
    (:data:`_NEWLINE_SAFE_COLUMNS`) can never contain a newline by
    construction and are selected bare, unchanged from their real value.

    Parameters
    ----------
    column : str
        Column name, from this module's own hardcoded per-function calls
        (never external input).

    Returns
    -------
    str
        A ``SELECT``-list entry, aliased back to ``column``'s own quoted
        name so downstream code never has to know which columns were
        wrapped.
    """
    quoted = _quote_ident(column)
    if column in _NEWLINE_SAFE_COLUMNS:
        return quoted
    return (
        f"left(regexp_replace({quoted}, E'[\\n\\r]+', ' ', 'g'), "
        f"{_SELECTED_COLUMN_MAX_CHARS}) AS {quoted}"
    )


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


#: Candidate fast-path formats tried, in order, before falling back to
#: pandas' flexible parser (see :func:`_parse_datetime_series`). Both are
#: ISO field order (year-month-day), so a match can never be a
#: misinterpretation the flexible fallback would have read differently --
#: unlike e.g. ``%m/%d/%Y`` vs ``%d/%m/%Y``, there is no ambiguity to get
#: wrong here. ``%.f`` matches an optional fractional-seconds suffix.
_DATETIME_FAST_PATH_FORMATS = (
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
)

#: Cumulative wall-clock time spent inside :func:`_parse_datetime_series`,
#: reset once per batch by :func:`run_extraction`'s phase-timing
#: instrumentation so a "parse" number can be isolated from the rest of
#: each extractor's per-chunk transform work, which otherwise all happens
#: inside the same generator call. Module-level, not thread-local: only
#: :func:`_stream_table_copy`'s producer thread runs concurrently with the
#: consumer-side extractor code that calls this function, and that thread
#: never calls it.
_datetime_parse_seconds = 0.0


def _parse_datetime_series(raw: pl.Series) -> pl.Series:
    """Vectorized best-effort parse of one of GEMINI's text datetime columns.

    Tries :data:`_DATETIME_FAST_PATH_FORMATS` first via polars' native,
    vectorized ``str.to_datetime`` (fast, but only matches rows in exactly
    that shape); whatever's left null after every fast-path attempt --
    genuinely different formats, garbage, real nulls -- falls back to
    pandas' flexible, per-element ``format="mixed"`` inference (the same
    semantics the original all-rows-through-pandas implementation used),
    applied only to that residue. A real run measured this residue as a
    small minority of rows, so this keeps the flexible fallback's exact
    correctness (nothing here assumes GEMINI's format is fully uniform --
    see docs/gemini_extraction.md's open question 7) while avoiding paying
    its cost for the common case.

    Parameters
    ----------
    raw : polars.Series
        One text datetime column, one chunk at a time.

    Returns
    -------
    polars.Series
        ``Datetime("us")``, with unparsable/missing entries as ``null``.
    """
    global _datetime_parse_seconds  # noqa: PLW0603
    t0 = time.time()
    try:
        stripped = raw.cast(pl.Utf8).str.strip_chars()
        fast = stripped.to_frame("raw").select(
            pl.coalesce(
                [
                    pl.col("raw").str.to_datetime(
                        format=fmt, strict=False, time_unit="us"
                    )
                    for fmt in _DATETIME_FAST_PATH_FORMATS
                ]
            ).alias("parsed")
        )["parsed"]
        residue_mask = stripped.is_not_null() & fast.is_null()
        residue_idx = residue_mask.arg_true()
        if len(residue_idx) > 0:
            residue_values = pd.to_datetime(
                stripped.gather(residue_idx).to_pandas(),
                errors="coerce",
                format="mixed",
            )
            fast = fast.scatter(
                residue_idx, pl.Series(residue_values).cast(pl.Datetime("us"))
            )
        return fast
    finally:
        _datetime_parse_seconds += time.time() - t0


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


def _normalize_name_series(raw: pl.Series) -> pl.Series:
    """Casefold + whitespace-collapse a raw name/value column for a MEDS ``code``.

    Same textual cleanup :func:`_normalize_unit_series` uses (strip,
    casefold, collapse internal whitespace) but deliberately *without*
    that function's sentinel-to-``"UNK"`` word-list or its
    canonicalization map: there is no known-variant-cluster map for vitals
    *names* the way :data:`UNIT_CANONICALIZATION_MAP` exists for units, so
    two spellings of what's genuinely the same measurement will not
    collapse onto one code here -- that cross-name canonicalization is
    future ``odyssey/data/code_mapping.py`` work, not something to
    improvise in the extractor. Null or blank (after stripping) still
    becomes ``"UNK"``, same fallback as :func:`_normalize_unit_series`.

    Parameters
    ----------
    raw : polars.Series
        A raw name or (non-numeric) value column, one chunk at a time.

    Returns
    -------
    polars.Series
        Casefolded, whitespace-collapsed text, or ``"UNK"``.
    """
    collapsed = (
        raw.cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
    )
    return pl.select(
        pl.when(raw.is_null() | (collapsed == ""))
        .then(pl.lit("UNK"))
        .otherwise(collapsed)
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

    Row boundaries are tracked via an incremental newline counter
    (:attr:`_pending_newlines`, updated by ``O(len(data))`` per ``write()``
    call) rather than rescanning the whole buffer on every call --
    libpq's ``COPY TO`` wire protocol delivers ``PQgetCopyData`` results
    one row at a time, so ``copy_expert`` invokes ``write()`` once per
    ~60-byte row; a full-buffer ``self._buffer.count(b"\\n")`` on every
    such call is quadratic in chunk size and was slow enough in practice
    (~200 rows/s on a 500k-row first chunk) to look identical to a hang.

    **Column constraint, closed server-side, not by CSV-aware parsing
    here:** row boundaries are found by counting raw ``\\n`` bytes
    (:meth:`_drain`), not by CSV-aware parsing. A column value containing a
    literal newline inside a CSV-quoted field (Postgres ``COPY`` quotes a
    field containing the delimiter, quote character, or a newline) would
    make this miscount row boundaries and split a single logical row across
    two chunks, silently corrupting both. This module's earlier assumption
    that no selected column could contain one "by construction" was wrong
    -- ``lab_subset.result_value`` is free text and a real, demonstrated
    counterexample (see the module docstring's third real-run-hardening
    entry), so ``_stream_table_copy``/``_stream_table_cursor`` now wrap
    every ``text``/``character varying`` selected column server-side via
    :func:`_select_expr_sql` to strip embedded newlines/carriage returns
    before the value ever reaches this class -- the constraint is enforced
    at the source, not merely documented as currently-unviolated. If a
    future change selects a column through either fetch path, no action is
    needed here specifically: :func:`_select_expr_sql` wraps it by default
    unless it's added to :data:`_NEWLINE_SAFE_COLUMNS` (confirmed
    integer/boolean).

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
        self._pending_newlines = 0
        """Newlines currently sitting in ``self._buffer``, maintained
        incrementally so ``_drain``'s threshold check never rescans the
        whole buffer -- see the class docstring."""

    def write(self, data: bytes) -> int:
        """Accept one chunk of bytes from psycopg2 as the CSV output arrives."""
        if self._stop_requested.is_set():
            raise _StreamAbandonedError
        self._buffer.extend(data)
        self._pending_newlines += data.count(b"\n")
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
                self._pending_newlines -= 1
                continue
            rows_ready = self._pending_newlines >= self._chunk_rows
            # Byte-size insurance: even with _select_expr_sql's server-side
            # newline-stripping/length-capping, flush whatever complete
            # rows are already buffered once the buffer itself crosses
            # _SINK_BUFFER_BYTE_CAP, rather than waiting for a full
            # _chunk_rows -- defense in depth against an unexpectedly large
            # region this module didn't already bound, not the primary fix.
            bytes_ready = (
                not rows_ready
                and self._pending_newlines > 0
                and len(self._buffer) >= _SINK_BUFFER_BYTE_CAP
            )
            if not (rows_ready or bytes_ready):
                return
            n_rows = self._chunk_rows if rows_ready else self._pending_newlines
            pos = -1
            for _ in range(n_rows):
                pos = self._buffer.index(b"\n", pos + 1)
            self._parse_and_enqueue(bytes(self._buffer[: pos + 1]))
            del self._buffer[: pos + 1]
            self._pending_newlines -= n_rows

    def _parse_and_enqueue(self, body: bytes) -> None:
        assert self._header is not None
        # infer_schema_length=0: never infer a dtype from a sample of this
        # chunk -- every column comes back Utf8, full stop. Real incident:
        # inference is per-chunk, so a column that happens to be all-numeric
        # in early chunks (lab_subset.result_value, before the free-text
        # result region) locks in Float64 for that chunk, then a later
        # non-numeric value in the SAME chunk raises ComputeError. All typed
        # parsing already happens downstream, in each extract_<table>'s own
        # explicit .cast() calls -- this class's job is CSV chunking, not
        # type inference, and it should never have been doing any.
        frame = pl.read_csv(
            io.BytesIO(self._header + body),
            null_values=["\\N"],
            infer_schema_length=0,
        )
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
    cols_sql = ", ".join(_select_expr_sql(c) for c in dict.fromkeys(select_cols))
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
    cols_sql = ", ".join(_select_expr_sql(c) for c in dict.fromkeys(select_cols))
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


def _filter_valid_genc_id(chunk: pl.DataFrame, table: str) -> pl.DataFrame:
    """Cast ``genc_id`` to ``Int64`` leniently, dropping (and logging) unparsable rows.

    A hard ``.cast(pl.Int64)`` (this function's predecessor, used
    everywhere in this module until a real hang traced back to it) raises
    on any row whose ``genc_id`` isn't cleanly integer-parseable -- crashing
    the whole chunk, and worse, silently: a crash mid-chunk during
    :func:`_stream_table_copy`'s producer/consumer handoff could leave the
    producer thread blocked on a full, undrained queue, and the *original*
    exception's traceback never got printed until that deadlock was fixed
    separately (see the module docstring's "Fetch strategy"). Dropping an
    unparsable row instead is the same "one bad row must not kill the
    batch" principle already applied to every other messy field in this
    module (unmapped lab concepts, blank diagnosis codes, ...) --
    ``genc_id`` is the join key for everything downstream, so a row that
    fails to parse here can't usably join against anything either way.

    Parameters
    ----------
    chunk : polars.DataFrame
        Must carry a ``genc_id`` column.
    table : str
        Source table name, for the warning log only.

    Returns
    -------
    polars.DataFrame
        ``chunk`` with ``genc_id`` cast to ``Int64``, unparsable rows
        dropped.
    """
    original = chunk["genc_id"]
    cast = original.cast(pl.Int64, strict=False)
    unparsable_mask = original.is_not_null() & cast.is_null()
    if unparsable_mask.any():
        bad_values = original.filter(unparsable_mask).head(5).to_list()
        logger.warning(
            "%s: %d rows had a non-null genc_id that failed to parse as an "
            "integer, dropping (sample raw values: %s)",
            table,
            int(unparsable_mask.sum()),
            bad_values,
        )
    return chunk.with_columns(cast).filter(pl.col("genc_id").is_not_null())


def _warn_on_unparsable_int_cast(
    original: pl.Series, cast: pl.Series, column: str, table: str
) -> None:
    """Log loudly for non-null values a lenient ``Int64`` cast turned to ``null``.

    Doesn't do the cast or drop anything itself -- callers of this already
    treat a null concept-id column as "no mapped concept" via their own
    existing filter (``is_in``/``is_not_null``), so a garbage value and a
    genuine ``NULL`` end up handled identically either way. This only makes
    the garbage case *visible* instead of silently indistinguishable from a
    real null -- the same "log loudly, never let a bad value hide" principle
    as :func:`_filter_valid_genc_id`, applied here because ``test_type_mapped_omop``/
    ``measurement_mapped_omop`` used a hard (non-lenient) cast until the
    all-Utf8 CSV read (see the module docstring's third real-run-hardening
    entry) removed the implicit type safety net a numeric-looking chunk
    used to provide.

    Parameters
    ----------
    original : polars.Series
        The column before casting.
    cast : polars.Series
        The same column after a ``strict=False`` cast to ``Int64``.
    column, table : str
        For the warning log only.
    """
    unparsable_mask = original.is_not_null() & cast.is_null()
    if unparsable_mask.any():
        bad_values = original.filter(unparsable_mask).head(5).to_list()
        logger.warning(
            "%s: %d rows had a non-null %s that failed to parse as an "
            "integer, treating as unmapped (sample raw values: %s)",
            table,
            int(unparsable_mask.sum()),
            column,
            bad_values,
        )


#: Case-insensitive, stripped membership map for :func:`_coerce_boolean_flag`.
#: Postgres COPY renders a real ``boolean`` column as bare ``t``/``f``, not
#: ``true``/``false`` -- the other spellings here are defensive, for any
#: GEMINI flag column that turns out to encode booleans differently.
_TRUE_FLAG_VALUES = frozenset({"t", "true", "1", "y", "yes"})
_FALSE_FLAG_VALUES = frozenset({"f", "false", "0", "n", "no"})


def _coerce_boolean_flag(chunk: pl.DataFrame, column: str, table: str) -> pl.DataFrame:
    """Leniently coerce a string flag column to ``Boolean`` via a membership map.

    polars has no supported ``Utf8 -> Boolean`` cast at all -- ``.cast(pl.Boolean)``
    on *any* string content raises ``InvalidOperationError`` unconditionally,
    strict or not (confirmed directly; this isn't a strictness setting to
    flip). That was this function's predecessor, used for ``icu_flag`` until
    a real crash on GEMINI hit it -- ``ipscu_subset`` came back over the wire
    as ``t``/``f`` (Postgres ``COPY``'s own boolean encoding), a value the
    cast would reject even if it accepted strings at all.

    A value that doesn't map to either set -- including a genuine SQL
    ``NULL`` -- becomes ``null`` in the returned column and is counted and
    logged once per chunk; this is a semantic flag, not a join key, so
    ``null`` here means "can't confirm true," not "malformed data to
    salvage" -- callers that gate a filter on the column (e.g. ``icu_flag``)
    get the conservative behavior for free, since ``DataFrame.filter``
    already excludes ``null`` alongside ``False``.

    Parameters
    ----------
    chunk : polars.DataFrame
        Must carry ``column``.
    column : str
        Name of the string flag column to coerce in place.
    table : str
        Source table name, for the warning log only.

    Returns
    -------
    polars.DataFrame
        ``chunk`` with ``column`` cast to ``Boolean``.
    """
    normalized = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    coerced = chunk.select(
        pl.when(normalized.is_in(_TRUE_FLAG_VALUES))
        .then(True)
        .when(normalized.is_in(_FALSE_FLAG_VALUES))
        .then(False)
        .otherwise(None)
        .alias(column)
    )[column]
    unresolved_mask = coerced.is_null()
    if unresolved_mask.any():
        bad_values = chunk[column].filter(unresolved_mask).head(5).to_list()
        logger.warning(
            "%s: %d rows had a %s value that didn't resolve to true/false, "
            "treating as not-true (sample raw values: %s)",
            table,
            int(unresolved_mask.sum()),
            column,
            bad_values,
        )
    return chunk.with_columns(coerced)


def fetch_admission_index() -> tuple[
    dict[int, str], dict[int, Optional[pd.Timestamp]], int
]:
    """One pass over ``admdad_subset``: ``genc_id -> (subject, admission time)``.

    Built once and held in memory for the whole extraction: every other
    table's rows carry only ``genc_id``, not ``patient_id_hashed`` (see
    docs/gemini_extraction.md's open question 6), and the pharmacy/
    radiology timestamp guard needs the encounter's admission time as its
    anchor. ~2.27M encounters in the schema-exploration cut -- two plain
    ``int``/``str``-keyed dicts, not a reason to re-query per row. Never
    checkpointed -- see the module docstring's "Resumability" section;
    cheap to rebuild in full on every restart.

    A row with a null or empty ``patient_id_hashed`` is unattributable to
    any subject and is dropped here rather than added to
    ``subject_by_genc`` -- the real incident this guards against is
    :func:`assign_shards` crashing on ``sorted()`` comparing ``None`` against
    ``str`` once such a value reaches it. Every other table's genc_id ->
    subject lookup already treats a missing key as "no subject" and routes
    the row to :attr:`MedsShardWriter.rows_dropped_unshardable`, so simply
    never adding these gencs to the index is sufficient to make them flow
    through that same existing path everywhere downstream, including
    ``admdad_subset``'s own extraction.

    Returns
    -------
    tuple[dict[int, str], dict[int, pandas.Timestamp | None], int]
        ``(subject_by_genc, admission_by_genc, n_dropped_null_subject)``.
    """
    subject_by_genc: dict[int, str] = {}
    admission_by_genc: dict[int, Optional[pd.Timestamp]] = {}
    n_dropped_null_subject = 0
    report_progress = _log_table_progress("admdad_subset (admission index)")
    for chunk in _stream_table(
        "admdad_subset", ["genc_id", "patient_id_hashed", "admission_date_time"]
    ):
        frame = _filter_valid_genc_id(chunk, "admdad_subset")
        frame = frame.with_columns(
            _parse_datetime_series(frame["admission_date_time"]).alias(
                "admission_date_time"
            ),
            frame["patient_id_hashed"].cast(pl.Utf8).alias("patient_id_hashed"),
        )
        unattributable_mask = (pl.col("patient_id_hashed").is_null()) | (
            pl.col("patient_id_hashed") == ""
        )
        unattributable_count = frame.select(unattributable_mask.sum()).item()
        if unattributable_count:
            n_dropped_null_subject += int(unattributable_count)
            logger.warning(
                "admdad_subset: %d rows had a null/empty patient_id_hashed, "
                "dropping (encounter is unattributable to any subject)",
                unattributable_count,
            )
            frame = frame.filter(~unattributable_mask)
        gencs = frame["genc_id"].to_list()
        subject_by_genc.update(
            zip(gencs, frame["patient_id_hashed"].to_list(), strict=True)
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
        report_progress(chunk.height)
    return subject_by_genc, admission_by_genc, n_dropped_null_subject


def fetch_mortality_index() -> dict[int, bool]:
    """One pass over ``derived_variables_subset``: ``genc_id -> mortality flag``.

    Primary mortality signal (docs/gemini_extraction.md's open question 2,
    resolved): the derived boolean is essentially fully populated (fewer
    than 6 nulls out of 2,268,000 rows, per extract-dry's null-fraction
    check), so it's used directly rather than decoding ``admdad_subset``'s
    ``discharge_disposition`` ourselves -- see :func:`extract_death`.

    ``in_hospital_mortality_derived`` is a real ``boolean`` column, same as
    ``ipscu_subset.icu_flag`` -- COPY's CSV output renders it ``t``/``f``
    text, not a value polars infers as ``Boolean`` on its own, so this
    reuses :func:`_coerce_boolean_flag` exactly as :func:`extract_icu`
    does. A null or missing flag is treated as "not known to have died"
    (``False``), not dropped -- absence of a mortality signal isn't itself
    a data-quality problem worth logging here the way an unattributable
    genc_id is in :func:`fetch_admission_index`.

    Returns
    -------
    dict[int, bool]
        ``genc_id -> in_hospital_mortality_derived``.
    """
    mortality_by_genc: dict[int, bool] = {}
    report_progress = _log_table_progress("derived_variables_subset (mortality index)")
    for chunk in _stream_table(
        "derived_variables_subset", ["genc_id", "in_hospital_mortality_derived"]
    ):
        frame = _filter_valid_genc_id(
            _coerce_boolean_flag(
                chunk, "in_hospital_mortality_derived", "derived_variables_subset"
            ),
            "derived_variables_subset",
        )
        mortality_by_genc.update(
            zip(
                frame["genc_id"].to_list(),
                frame["in_hospital_mortality_derived"].fill_null(False).to_list(),
                strict=True,
            )
        )
        report_progress(chunk.height)
    return mortality_by_genc


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
        frame = _filter_valid_genc_id(chunk, "admdad_subset")
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


#: CIHI DAD's ``discharge_disposition`` codes hypothesized to mean death.
#: Both the coarse code (7) and its more granular expansion (72/73/74) are
#: treated as death together -- a proper superset covering either
#: convention, since it isn't yet confirmed which one this datacut
#: actually uses. No longer used to decide event emission (see
#: :func:`extract_death`'s docstring) -- kept only as the comparison side
#: of that function's derived-flag-vs-disposition cross-check tally.
#: Revisit once real observed values are confirmed (scripts/gemini/out/
#: codes_inventory.json, or a value-distribution query).
DEATH_DISPOSITION_CODES = (7, 72, 73, 74)


def extract_death(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
    mortality_by_genc: dict[int, bool],
) -> Iterator[ExtractedBatch]:
    """Death events from ``derived_variables_subset.in_hospital_mortality_derived``.

    Load-bearing: without this, GEMINI has no mortality signal at all --
    no death alerts, no mortality task. docs/gemini_extraction.md's open
    question 2 already resolved which signal to trust: the derived
    boolean (essentially fully populated -- fewer than 6 nulls out of
    2,268,000 rows) is the PRIMARY signal, not a ``discharge_disposition``
    decode. One event per admission where :func:`fetch_mortality_index`'s
    flag is true, emitted as a single bare ``MEDS_DEATH`` code (no
    ``//``-suffixed variant) -- matching MIMIC-IV MEDS's own convention
    exactly, so cross-source event definitions stay uniform and
    ``vocabulary.py``'s existing ``"MEDS_DEATH": DEMOGRAPHIC_TYPE`` mapping
    applies directly with no change there.

    Timed at ``admdad_subset.discharge_date_time`` -- the derived table has
    no timestamp of its own, and ``admdad_subset`` has no
    ``disposition_date_time`` column (that name belongs to ``er_subset``'s
    own, unrelated column); a discharge disposition -- and thus a death --
    is recorded at the same event as the discharge itself.

    Guarded against discharge-before-admission specifically -- a distinct,
    narrower check from the far-future/far-past artifact class
    :func:`extract_admissions` already ruled out for this table (its own
    docstring: timestamps land entirely in the plausible 2010-2024 range).
    Even within that plausible range, a row could still have its discharge
    time recorded before its admission time by data-entry error; such rows
    are dropped, not extracted as a physically impossible death.

    ``discharge_disposition`` is still read from ``admdad_subset`` here,
    but purely for a cross-check: whether :data:`DEATH_DISPOSITION_CODES`
    agrees with the derived flag, tallied across the whole generator's run
    and logged once at the end (both-agree / derived-only /
    disposition-only counts). It never gates or shapes the emitted event
    stream -- a disagreement is a data-quality finding to record in
    docs/gemini_extraction.md once a real run reports it, not a reason to
    trust one signal over the other here.

    Event shape: one qualifying row produces one MEDS row, ``MEDS_DEATH``
    at ``discharge_date_time``, ``hadm_id = genc_id``, no ``numeric_value``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.
    mortality_by_genc : dict
        From :func:`fetch_mortality_index`.

    Yields
    ------
    ExtractedBatch
    """
    n_both_agree = 0
    n_derived_only = 0
    n_disposition_only = 0
    for chunk in _stream_table(
        "admdad_subset",
        ["genc_id", "discharge_disposition", "discharge_date_time"],
    ):
        frame = _filter_valid_genc_id(chunk, "admdad_subset")
        genc = frame["genc_id"]
        derived_dead = genc.replace_strict(
            mortality_by_genc, default=False, return_dtype=pl.Boolean
        )
        # discharge_disposition arrives as raw Utf8 like every other chunk
        # column (see the module docstring's afc50c4 entry) -- DEATH_
        # DISPOSITION_CODES is a tuple of Python ints, so is_in() needs a
        # leniently-cast Int64 series first, not the raw string column.
        # Real incident: shipped without this and crashed the first real
        # re-extract with a dtype-mismatch error. Unparsable/null values
        # simply aren't in the death-code set -- this is the cross-check
        # side only, never gates emission, so no row is dropped over it.
        disposition_int = frame["discharge_disposition"].cast(pl.Int64, strict=False)
        _warn_on_unparsable_int_cast(
            frame["discharge_disposition"],
            disposition_int,
            "discharge_disposition",
            "admdad_subset",
        )
        disposition_dead = disposition_int.is_in(DEATH_DISPOSITION_CODES).fill_null(
            False
        )
        n_both_agree += int((derived_dead & disposition_dead).sum())
        n_derived_only += int((derived_dead & ~disposition_dead).sum())
        n_disposition_only += int((~derived_dead & disposition_dead).sum())

        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admission_time = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        death_time = _parse_datetime_series(frame["discharge_date_time"])
        died = pl.DataFrame(
            {
                "subject_id": subject,
                "time": death_time,
                "code": "MEDS_DEATH",
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(
            derived_dead
            & death_time.is_not_null()
            & admission_time.is_not_null()
            & (death_time >= admission_time)
        )
        yield ExtractedBatch(_finalize_meds_batch(died), chunk.height)

    logger.info(
        "[extract_death] derived-flag vs discharge_disposition cross-check: "
        "%d both agree, %d derived-only, %d disposition-only (candidate "
        "death codes %s) -- disagreement above a trivial rate is a "
        "data-quality finding worth recording in docs/gemini_extraction.md",
        n_both_agree,
        n_derived_only,
        n_disposition_only,
        DEATH_DISPOSITION_CODES,
    )


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
        frame = _filter_valid_genc_id(
            _coerce_boolean_flag(chunk, "icu_flag", "ipscu_subset").filter(
                pl.col("icu_flag")
            ),
            "ipscu_subset",
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
        frame = _filter_valid_genc_id(chunk, "lab_subset")
        concept_raw = frame["test_type_mapped_omop"]
        concept_cast = concept_raw.cast(pl.Int64, strict=False)
        _warn_on_unparsable_int_cast(
            concept_raw, concept_cast, "test_type_mapped_omop", "lab_subset"
        )
        frame = frame.with_columns(concept_cast).filter(
            pl.col("test_type_mapped_omop").is_in(list(lab_concepts))
        )
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
    (``measurement_mapped_omop`` null) are dropped -- rescued separately
    by :func:`extract_vitals_unmapped`, the exact complement, since this
    drop accounts for ~119M of ~412M rows (the entire 71%-retention
    story, confirmed real, not a blank-timestamp artifact -- see that
    function's docstring).

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
        frame = _filter_valid_genc_id(chunk, "vitals_subset")
        concept_raw = frame["measurement_mapped_omop"]
        concept_cast = concept_raw.cast(pl.Int64, strict=False)
        _warn_on_unparsable_int_cast(
            concept_raw, concept_cast, "measurement_mapped_omop", "vitals_subset"
        )
        frame = frame.with_columns(concept_cast).filter(
            pl.col("measurement_mapped_omop").is_not_null()
        )
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


def extract_vitals_unmapped(
    subject_by_genc: dict[int, str],
) -> Iterator[ExtractedBatch]:
    """Vital-sign events from ``vitals_subset`` with no OMOP concept mapping.

    Real incident this rescues: :func:`extract_vitals` drops every row
    whose ``measurement_mapped_omop`` doesn't resolve to a concept id --
    ~119M of ~412M rows, the entire 71%-retention story. Confirmed real
    and server-side, not a blank-timestamp artifact the way
    ``erintervention_subset``'s own gap was (measured blank timestamps
    here are only ~44k). Real unmapped names by frequency: FiO2 variants,
    oxygen delivery method/flow/therapy, pain score, ``NEURO.PIR``/``PIA``
    (pupillary response -- core signal for a subdural-hematoma cohort),
    ``VS.NWS.LOC1`` (level of consciousness), plus assorted qualifier
    fields (HR source, BP location/position).

    This is the exact complement of :func:`extract_vitals`'s own
    concept-id filter (``measurement_mapped_omop`` null after the same
    lenient cast -- not re-logged here, :func:`extract_vitals`'s own pass
    over this same table already does that once), rescued via
    ``measurement_name`` as a fallback identity -- the same convention
    eICU's own extraction already uses for an unmapped vital/lab (a
    device/measurement label rather than no signal at all). ``code`` is
    ``VITALS//<name>//<unit>`` when ``measurement_value`` parses as
    numeric (``numeric_value`` set, same as :func:`extract_vitals`'s own
    mapped rows), or ``VITALS//<name>//<value>`` when it doesn't
    (``numeric_value`` null, the categorical value folded into the code
    itself, the same style demographic codes elsewhere in this project
    already use). Name and non-numeric-value normalization is casefold +
    whitespace-collapse only (:func:`_normalize_name_series`) --
    deliberately not :func:`_normalize_unit_series`'s fuller
    sentinel/canonicalization treatment: there is no known-variant-cluster
    map for vitals *names* yet the way there is for units, so two
    spellings of the same measurement will not collapse onto one code
    here -- future ``odyssey/data/code_mapping.py`` work, not something to
    improvise in the extractor. No admission-window guard, matching
    :func:`extract_vitals`'s own convention (it doesn't apply one either).

    Own manifest key (``vitals_subset__unmapped``) in
    :func:`run_extraction`'s ``table_generators``, resumable
    independently of the mapped pass. Rows with no usable
    ``measurement_name`` (null/blank) are dropped -- with no concept id
    *and* no name, there is no identity left to build a code from.

    Event shape: one row with an unmapped concept and a non-blank name
    produces one MEDS row, ``hadm_id = genc_id``.

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
            "measurement_name",
            "measurement_value",
            "measurement_unit",
            "measure_date_time",
        ],
    ):
        frame = _filter_valid_genc_id(chunk, "vitals_subset")
        concept_cast = frame["measurement_mapped_omop"].cast(pl.Int64, strict=False)
        frame = frame.with_columns(concept_cast).filter(
            pl.col("measurement_mapped_omop").is_null()
            & pl.col("measurement_name").is_not_null()
            & (pl.col("measurement_name").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        name = _normalize_name_series(frame["measurement_name"])
        numeric_value = frame["measurement_value"].cast(pl.Float64, strict=False)
        unit = _normalize_unit_series(frame["measurement_unit"])
        value_text = _normalize_name_series(frame["measurement_value"])
        value_or_unit = pl.DataFrame(
            {"numeric_value": numeric_value, "unit": unit, "value_text": value_text}
        ).select(
            pl.when(pl.col("numeric_value").is_not_null())
            .then(pl.col("unit"))
            .otherwise(pl.col("value_text"))
            .alias("value_or_unit")
        )["value_or_unit"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": _parse_datetime_series(frame["measure_date_time"]),
                "code": "VITALS//" + name + "//" + value_or_unit,
                "numeric_value": numeric_value,
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
        frame = _filter_valid_genc_id(chunk, "pharmacy_subset").filter(
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
    report_progress = _log_table_progress("admdad_subset (discharge index)")
    for chunk in _stream_table("admdad_subset", ["genc_id", "discharge_date_time"]):
        frame = _filter_valid_genc_id(chunk, "admdad_subset")
        frame = frame.with_columns(
            _parse_datetime_series(frame["discharge_date_time"]).alias(
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
        report_progress(chunk.height)

    for chunk in _stream_table("ipdiagnosis_subset", ["genc_id", "diagnosis_code"]):
        frame = _filter_valid_genc_id(chunk, "ipdiagnosis_subset").filter(
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
        frame = _filter_valid_genc_id(chunk, "ipintervention_subset").filter(
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
        frame = _filter_valid_genc_id(chunk, "radiology_subset")
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
        frame = _filter_valid_genc_id(chunk, "physicians_subset")
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


def fetch_discharge_index() -> dict[int, Optional[pd.Timestamp]]:
    """One pass over ``admdad_subset``: ``genc_id -> discharge time``.

    A second, discharge-only index alongside :func:`fetch_admission_index`'s
    own admission-time one -- shared by :func:`extract_billing_cmg` and
    :func:`extract_billing_hig`, both of which anchor to discharge (a
    grouper/casemix code is finalized once the whole stay is coded, the
    same "coded at the encounter level, attributed at its close" reasoning
    :func:`extract_diagnoses` already uses for its own, separate discharge
    index -- kept separate rather than shared with that function to avoid
    touching its already-tested internals for an unrelated addition).
    Never checkpointed, rebuilt in full on every call, same as
    :func:`fetch_admission_index` -- see the module docstring's
    "Resumability" section.

    Returns
    -------
    dict[int, pandas.Timestamp | None]
        ``{genc_id: discharge_time}``.
    """
    discharge_by_genc: dict[int, Optional[pd.Timestamp]] = {}
    report_progress = _log_table_progress("admdad_subset (billing discharge index)")
    for chunk in _stream_table("admdad_subset", ["genc_id", "discharge_date_time"]):
        frame = _filter_valid_genc_id(chunk, "admdad_subset")
        frame = frame.with_columns(
            _parse_datetime_series(frame["discharge_date_time"]).alias(
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
        report_progress(chunk.height)
    return discharge_by_genc


def extract_er(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """ED registration/triage/leave events from ``er_subset``.

    Three real, independent timestamps on this table, each its own event
    -- ``ED_REGISTRATION`` (``registration_date_time``, the same prefix
    MIMIC-IV's own ED extraction uses -- already in
    ``odyssey/data/vocabulary.py``'s ``_PREFIX_TO_TYPE``), ``ED_TRIAGE``
    (``triage_date_time``, a new prefix), ``ED_OUT`` (``left_er_date_time``,
    also already in the vocabulary). ``disposition_date_time`` and the
    ambulance/physician-assessment timestamps are not extracted as events
    here -- registration/triage/leave are the three that mark real stage
    transitions in the ED visit; the rest describe the visit rather than
    bounding it. Same admission-window guard as every other real-timestamp
    table in this module (:func:`_within_admission_guard_mask`) -- not
    documented as having outlier years yet, but guarded defensively rather
    than assumed clean.

    A ``genc_id`` here may not appear in ``admdad_subset`` at all -- an ED
    visit that didn't result in an admission has no corresponding row
    there, so :func:`fetch_admission_index` never saw it, and these rows
    are dropped as unattributable (no ``subject_id``) via the same
    existing ``genc_id -> subject`` lookup-miss path every other table
    already uses -- not a new failure mode, just a real one that bites this
    table more than most.

    Event shape: one row can produce up to three MEDS rows, ``hadm_id =
    genc_id``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "er_subset",
        [
            "genc_id",
            "registration_date_time",
            "triage_date_time",
            "left_er_date_time",
        ],
    ):
        frame = _filter_valid_genc_id(chunk, "er_subset")
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admission = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        events = [
            ("ED_REGISTRATION", "registration_date_time"),
            ("ED_TRIAGE", "triage_date_time"),
            ("ED_OUT", "left_er_date_time"),
        ]
        batches = []
        for code, column in events:
            event_time = _parse_datetime_series(frame[column])
            batches.append(
                pl.DataFrame(
                    {
                        "subject_id": subject,
                        "time": event_time,
                        "code": code,
                        "numeric_value": None,
                        "hadm_id": genc,
                    }
                ).filter(_within_admission_guard_mask(event_time, admission))
            )
        meds = pl.concat(batches)
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_er_diagnoses(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """ED diagnosis events from ``erdiagnosis_subset``.

    ``code`` is the raw ER diagnosis code, namespaced (``ED_DIAGNOSIS//<code>``,
    a new prefix -- kept distinct from ``ipdiagnosis_subset``'s own
    ``DIAGNOSIS//`` since these are coded in a different clinical context,
    even though both fall under ``DIAGNOSIS_TYPE``). No event-level
    timestamp exists on this table -- ``time`` is the encounter's admission
    time, the same convention :func:`extract_providers` already uses for
    another no-timestamp table.

    Event shape: one row with a non-blank diagnosis code produces one MEDS
    row, ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table("erdiagnosis_subset", ["genc_id", "er_diagnosis_code"]):
        frame = _filter_valid_genc_id(chunk, "erdiagnosis_subset").filter(
            pl.col("er_diagnosis_code").is_not_null()
            & (pl.col("er_diagnosis_code").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": genc.replace_strict(
                    admission_by_genc, default=None, return_dtype=pl.Datetime("us")
                ),
                "code": "ED_DIAGNOSIS//" + frame["er_diagnosis_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_er_procedures(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """ED procedure events from ``erintervention_subset`` with a passing timestamp.

    ``code`` reuses ``ipintervention_subset``'s own ``PROCEDURE//<code>``
    prefix, not a new ER-specific one: ``intervention_code`` is CCI-coded
    in both tables (same column name, same coding system), so an ER
    intervention and an inpatient one are genuinely the same vocabulary,
    just from a different source table -- unlike diagnoses (see
    :func:`extract_er_diagnoses`), there's no source-context distinction
    worth losing by merging the code spaces. Own real timestamp
    (``intervention_episode_start_date_time``), same admission-window guard
    as every other real-timestamp table.

    **Only a real minority of coded rows have a usable timestamp here**:
    ``intervention_episode_start_date_time`` is blank (empty string, not
    SQL ``NULL`` -- invisible to ``extract-dry``'s old ``COUNT(column)``-based
    null counting, see :func:`extract_dry.null_fraction`'s fix) on ~96.7%
    of rows with a real code. This function keeps exactly the rows whose
    timestamp both parses and passes the admission-window guard;
    :func:`extract_er_procedures_untimed` is the complement -- every
    coded row this function's own guard rejects, attributed instead to
    admission time. Split into two passes, rather than one function
    picking a single attribution per row, so a row's real, trustworthy
    timestamp is never discarded in favor of the coarser admission-time
    fallback just because *some* rows need that fallback.

    Event shape: one row with a non-blank intervention code produces one
    MEDS row, ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "erintervention_subset",
        ["genc_id", "intervention_code", "intervention_episode_start_date_time"],
    ):
        frame = _filter_valid_genc_id(chunk, "erintervention_subset").filter(
            pl.col("intervention_code").is_not_null()
            & (pl.col("intervention_code").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admission = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        event_time = _parse_datetime_series(
            frame["intervention_episode_start_date_time"]
        )
        meds = pl.DataFrame(
            {
                "subject_id": subject,
                "time": event_time,
                "code": "PROCEDURE//" + frame["intervention_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(_within_admission_guard_mask(event_time, admission))
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_er_procedures_untimed(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """ED procedure events from ``erintervention_subset`` with no usable timestamp.

    Real incident this rescues: :func:`extract_er_procedures`'s admission
    guard, combined with ``intervention_episode_start_date_time`` being
    blank (not ``NULL``) on ~96.7% of rows, left only ~90k of ~2.94M
    coded ER interventions in that function's own output. This is the
    exact complement -- same code-validity filter, same admission-window
    guard predicate, inverted (``NOT`` guard-passed: blank/unparsable
    timestamp, or a timestamp outside the guard window) -- attributed
    instead to the encounter's admission time, the same no-usable-own-
    timestamp convention :func:`extract_er_diagnoses`/:func:`extract_transfers`
    already use. Must never re-emit a row :func:`extract_er_procedures`
    already kept: since both functions compute the identical
    ``_within_admission_guard_mask`` predicate over the identical
    code-filtered row set and simply keep opposite sides of it, every
    coded row lands in exactly one of the two outputs, never both, never
    neither. Own manifest key (``erintervention_subset__untimed``) in
    :func:`run_extraction`'s ``table_generators``, so this pass is
    resumable independently of the timed one.

    Event shape: one row with a non-blank intervention code AND a
    failing admission guard produces one MEDS row at the encounter's
    admission time, ``hadm_id = genc_id``. A row whose ``genc_id`` has no
    admission time either (not in ``admdad_subset`` at all) is dropped
    here too -- there is nothing left to attribute it to, same as every
    other admission-time-anchored table in this module.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "erintervention_subset",
        ["genc_id", "intervention_code", "intervention_episode_start_date_time"],
    ):
        frame = _filter_valid_genc_id(chunk, "erintervention_subset").filter(
            pl.col("intervention_code").is_not_null()
            & (pl.col("intervention_code").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admission = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        event_time = _parse_datetime_series(
            frame["intervention_episode_start_date_time"]
        )
        guard_passed = _within_admission_guard_mask(event_time, admission)
        meds = pl.DataFrame(
            {
                "subject_id": subject,
                "time": admission,
                "code": "PROCEDURE//" + frame["intervention_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(~guard_passed)
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_er_consults(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """ED consult-request events from ``erconsults_subset``.

    ``code`` is ``ER_CONSULT//<consult_service_code>`` (a new prefix,
    ``OTHER_TYPE`` -- a consult is a referral to a service, not itself a
    diagnosis or procedure, so it doesn't fit an existing type bucket any
    more precisely). Timed at ``consult_request_date_time`` (the
    consult-ordering clinician's action, the natural anchor -- matches
    this module's convention elsewhere of using the *initiating* timestamp
    when a row carries more than one candidate, e.g. pharmacy's
    ``med_start_date_time``); ``consult_arrival_date_time`` is not
    extracted as a second event, just the one meaningful "this was
    requested" fact. Same admission-window guard as every other
    real-timestamp table.

    Event shape: one row with a non-blank consult service code produces
    one MEDS row, ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "erconsults_subset",
        ["genc_id", "consult_service_code", "consult_request_date_time"],
    ):
        frame = _filter_valid_genc_id(chunk, "erconsults_subset").filter(
            pl.col("consult_service_code").is_not_null()
            & (pl.col("consult_service_code").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        subject = genc.replace_strict(
            subject_by_genc, default=None, return_dtype=pl.Utf8
        )
        admission = genc.replace_strict(
            admission_by_genc, default=None, return_dtype=pl.Datetime("us")
        )
        event_time = _parse_datetime_series(frame["consult_request_date_time"])
        meds = pl.DataFrame(
            {
                "subject_id": subject,
                "time": event_time,
                "code": "ER_CONSULT//" + frame["consult_service_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        ).filter(_within_admission_guard_mask(event_time, admission))
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_transfers(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Institution-transfer events from ``lookup_transfer_subset``.

    ``code`` is ``TRANSFER_TO//<institution_to_mns>`` -- despite the table
    name, these are real per-encounter unit-transfer rows, not a static
    lookup; ``TRANSFER_TO`` is already in
    ``odyssey/data/vocabulary.py``'s ``_PREFIX_TO_TYPE``, matching MIMIC's
    own convention. No event-level timestamp exists on this table (~1.25M
    rows for ~2.27M encounters -- most encounters have none, consistent
    with a transfer being a real but not-always-present sub-event) --
    ``time`` is the encounter's admission time, same convention as
    :func:`extract_er_diagnoses`.

    Event shape: one row with a non-blank destination institution produces
    one MEDS row, ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table(
        "lookup_transfer_subset", ["genc_id", "institution_to_mns"]
    ):
        frame = _filter_valid_genc_id(chunk, "lookup_transfer_subset").filter(
            pl.col("institution_to_mns").is_not_null()
            & (pl.col("institution_to_mns").cast(pl.Utf8).str.strip_chars() != "")
        )
        genc = frame["genc_id"]
        meds = pl.DataFrame(
            {
                "subject_id": genc.replace_strict(
                    subject_by_genc, default=None, return_dtype=pl.Utf8
                ),
                "time": genc.replace_strict(
                    admission_by_genc, default=None, return_dtype=pl.Datetime("us")
                ),
                "code": "TRANSFER_TO//" + frame["institution_to_mns"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_billing_cmg(
    subject_by_genc: dict[int, str],
    discharge_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Case Mix Group billing events from ``ipcmg_subset``.

    ``code`` is ``BILLING_CMG//<cmg>`` (a new prefix, ``BILLING_TYPE`` --
    the same type bucket MIMIC's own ``DRG`` prefix uses, but kept a
    distinct code identity from it: CMG is CIHI's Canadian casemix-group
    system, not the US DRG system, so collapsing them onto the literal
    ``DRG`` string would conflate two different grouper vocabularies under
    one code). No event-level timestamp -- a grouper code is finalized
    once the whole stay is coded, so ``time`` is the encounter's discharge
    time (:func:`fetch_discharge_index`), the same "attributed at its
    close" convention :func:`extract_diagnoses` already uses.

    Event shape: one row with a non-blank CMG code produces one MEDS row,
    ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.
    discharge_by_genc : dict
        From :func:`fetch_discharge_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table("ipcmg_subset", ["genc_id", "cmg"]):
        frame = _filter_valid_genc_id(chunk, "ipcmg_subset").filter(
            pl.col("cmg").is_not_null()
            & (pl.col("cmg").cast(pl.Utf8).str.strip_chars() != "")
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
                "code": "BILLING_CMG//" + frame["cmg"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
        )
        yield ExtractedBatch(_finalize_meds_batch(meds), chunk.height)


def extract_billing_hig(
    subject_by_genc: dict[int, str],
    discharge_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[ExtractedBatch]:
    """Health-based Inpatient Group billing events from ``iphig_subset``.

    ``code`` is ``BILLING_HIG//<hig_code>`` (a new prefix, ``BILLING_TYPE``
    -- same reasoning as :func:`extract_billing_cmg`: HIG is a distinct
    CIHI grouper system from both CMG and US DRG, kept as its own code
    identity rather than collapsed onto either). Same discharge-time
    anchor as :func:`extract_billing_cmg`, same reasoning.

    Event shape: one row with a non-blank HIG code produces one MEDS row,
    ``hadm_id = genc_id``.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.
    discharge_by_genc : dict
        From :func:`fetch_discharge_index`.

    Yields
    ------
    ExtractedBatch
    """
    for chunk in _stream_table("iphig_subset", ["genc_id", "hig_code"]):
        frame = _filter_valid_genc_id(chunk, "iphig_subset").filter(
            pl.col("hig_code").is_not_null()
            & (pl.col("hig_code").cast(pl.Utf8).str.strip_chars() != "")
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
                "code": "BILLING_HIG//" + frame["hig_code"].cast(pl.Utf8),
                "numeric_value": None,
                "hadm_id": genc,
            }
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
        ``fetch_admission_index``'s ``subject_by_genc`` values). A ``None``
        is silently skipped rather than raising -- defensive, since
        :func:`fetch_admission_index` itself already excludes unattributable
        (null/empty ``patient_id_hashed``) rows from ``subject_by_genc``, so
        this should never actually see one; it's a second guard against
        ``sorted()`` crashing on a ``None``-vs-``str`` comparison if a future
        index source reintroduces one.
    subjects_per_shard : int
        Target subjects per shard; shard count is
        ``ceil(n_subjects / subjects_per_shard)``, at least 1.

    Returns
    -------
    dict[str, int]
        ``{subject_id: shard_index}``.
    """
    unique_subjects = sorted({s for s in subject_ids if s is not None})
    n_shards = _shard_count(len(unique_subjects), subjects_per_shard)
    return {
        subject_id: int(hashlib.sha256(subject_id.encode()).hexdigest(), 16) % n_shards
        for subject_id in unique_subjects
    }


#: Rows buffered in memory per shard before :class:`MedsShardWriter` actually
#: calls ``write_table()``. The real fix for a measured real-run bottleneck:
#: at ``CHUNK_ROWS`` = 500,000 scattered across ~1,118 shards, the
#: un-buffered writer made one small ``write_table()`` call per shard per
#: chunk (a few hundred rows each); on GEMINI's NFS-mounted ``output_dir``
#: the per-call round-trip cost of that many small writes -- not CPU
#: parse/transform time, both measured negligible locally against a local
#: disk -- is what dominated wall-clock: ``admdad_subset``'s 2.27M rows
#: (~5 chunks * ~1,118 shards ~= 5,590 writes) took ~21 minutes, which
#: projects out to ~4 days for ``lab_subset``'s 659M rows (~1,318 chunks *
#: ~1,118 shards ~= 1.47M writes) at the same per-write cost. Buffering
#: collapses that call count by roughly (this threshold / average rows
#: landing in a shard per chunk), independent of table size -- for a table
#: whose total per-shard row count never reaches this threshold, every
#: batch stays buffered until :meth:`MedsShardWriter.close`, i.e. one write
#: per shard for the whole table instead of one per chunk.
SHARD_FLUSH_ROW_THRESHOLD = 250_000

#: Global cap on rows buffered across *all* shards at once, independent of
#: :data:`SHARD_FLUSH_ROW_THRESHOLD`'s per-shard bound. Overridable via
#: ``GEMINI_WRITER_MAX_BUFFERED_ROWS``.
#:
#: Real incident this closes (second OOM kill, new mechanism): subject-hash
#: sharding is uniform, so with ~1,118 shards and a table's rows arriving
#: roughly evenly across subjects, every shard accumulates rows at close to
#: the same rate -- all ~1,118 shards cross
#: :data:`SHARD_FLUSH_ROW_THRESHOLD` within the same processing window,
#: producing a synchronized high-water of up to
#: ``SHARD_FLUSH_ROW_THRESHOLD * n_shards`` (~279M rows for 1,118 shards)
#: resident at once, not the ``SHARD_FLUSH_ROW_THRESHOLD``-per-shard bound
#: :class:`MedsShardWriter`'s docstring used to claim. Real per-row cost in
#: a pandas object-dtype frame with string codes measured at ~250-400
#: bytes (not an earlier ~130-byte estimate), plus each flush's
#: ``pd.concat`` copy -- 318M then 340M buffered lab_subset rows OOM-killed
#: both a 64 GB and a 96 GB job at almost exactly this ~280M-row mark. This
#: cap tracks :attr:`MedsShardWriter.total_buffered_rows` across every
#: shard and force-flushes the fullest ones first once it's exceeded (see
#: :meth:`MedsShardWriter._flush_fullest_shards_until_under_cap`) --
#: bounding peak memory to roughly this cap's own row count, and
#: desynchronizing future waves since the flushed shards restart from 0
#: while the others keep accumulating. ``SHARD_FLUSH_ROW_THRESHOLD`` itself
#: is untouched -- this is an *additional*, aggregate bound, not a
#: replacement.
WRITER_MAX_BUFFERED_ROWS = int(
    os.environ.get("GEMINI_WRITER_MAX_BUFFERED_ROWS", "40000000")
)

_SHARD_FILE_RE = re.compile(r"^shard_(\d{4})(?:_part(\d+))?\.parquet$")


def _shard_glob(output_dir: Path, shard: int) -> list[Path]:
    """Every on-disk file belonging to one logical shard, base file first."""
    prefix = f"shard_{shard:04d}"
    matches = [
        p for p in output_dir.glob(f"{prefix}*.parquet") if _SHARD_FILE_RE.match(p.name)
    ]
    return sorted(
        matches,
        key=lambda p: (
            int(m.group(2) or 0) if (m := _SHARD_FILE_RE.match(p.name)) else 0
        ),
    )


def _next_shard_write_path(output_dir: Path, shard: int) -> Path:
    """Pick a path for a *new* ``pq.ParquetWriter`` on this shard.

    Guaranteed not to already exist -- ``pq.ParquetWriter`` silently
    truncates whatever file it opens, and a resumed run constructs a brand
    new :class:`MedsShardWriter` (with an empty ``self._writers``) every
    process invocation, so reopening the base filename unconditionally
    would destroy any rows a *different*, already-completed table wrote
    into this same shard in a prior run. The base ``shard_{i:04d}.parquet``
    name is used if nothing is there yet; otherwise the next free
    ``shard_{i:04d}_part{N}.parquet`` -- a logical shard is the union of
    its base file plus every ``_partN`` file, never just one of them (see
    the class docstring and :func:`_logical_shard_row_counts`).
    """
    base = output_dir / f"shard_{shard:04d}.parquet"
    if not base.exists():
        return base
    existing = _shard_glob(output_dir, shard)
    part_numbers = [
        int(m.group(2))
        for p in existing
        if (m := _SHARD_FILE_RE.match(p.name)) and m.group(2)
    ]
    return (
        output_dir / f"shard_{shard:04d}_part{max(part_numbers, default=0) + 1}.parquet"
    )


def _logical_shard_row_counts(output_dir: Path) -> dict[int, int]:
    """On-disk row count per logical shard, summed across all its part files.

    Reads only each file's Parquet footer metadata (``num_rows``), never
    the actual row data -- cheap even at real scale. The authoritative
    source for the extraction summary's ``shard_row_counts``, since a
    resumed run's :class:`MedsShardWriter` only tracks rows *it* wrote,
    not what a prior run already committed to a shard's base file.

    A file that fails to open as valid Parquet is **never** silently
    skipped -- real incident: a killed process (an OOM kill, before the
    ``.tmp``-then-``rename`` atomic-visibility fix existed) left a
    truncated, footer-less file under a *final*, countable name;
    ``pq.ParquetFile`` raised ``ArrowInvalid`` here with no path in the
    message, forcing a manual scan of 1,000+ files to find the one bad
    one. Same no-blanket-silence doctrine as the durability fix elsewhere
    in this module: a corrupt final-named file is real, unexpected
    damage (with the atomic-visibility fix in place, every final-named
    file is supposed to be a completed, valid write) and must fail loud
    with exactly which path is bad, not be papered over or guessed at.
    """
    counts: dict[int, int] = {}
    for path in output_dir.glob("shard_*.parquet"):
        match = _SHARD_FILE_RE.match(path.name)
        if match is None:
            continue
        shard = int(match.group(1))
        try:
            n_rows = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            raise RuntimeError(
                f"[extract_meds] corrupt or unreadable Parquet file at {path} -- "
                "refusing to silently skip it; this is real damage (e.g. a "
                "truncated file from a killed process), not something to "
                "paper over. Investigate and remove/replace this specific "
                "file before retrying."
            ) from exc
        counts[shard] = counts.get(shard, 0) + n_rows
    return counts


class MedsShardWriter:
    """Streaming per-shard MEDS Parquet writer.

    A **logical shard is the union of its ``shard_{i:04d}.parquet`` base
    file and every ``shard_{i:04d}_partN.parquet`` file** -- see
    :func:`_next_shard_write_path` for why a resumed run may add part
    files rather than writing into the base file directly, and
    :func:`_logical_shard_row_counts` for the aggregation any accounting
    surface must use instead of this class's own (this-run-only)
    :attr:`rows_dropped_unshardable`-style counters. Downstream readers of
    this output must glob and concatenate every file sharing a shard
    prefix, not assume one file per shard.

    Each shard's rows are buffered in memory
    (:data:`SHARD_FLUSH_ROW_THRESHOLD`) and only written once that many
    rows have accumulated for it (or at :meth:`close`) -- collapsing many
    small per-chunk ``write_table()`` calls into far fewer, larger ones;
    see :data:`SHARD_FLUSH_ROW_THRESHOLD`'s docstring for the real-run
    measurement this fixes. **Measured memory reality, corrected (second
    OOM incident)**: ``SHARD_FLUSH_ROW_THRESHOLD`` alone bounds each
    shard's *own* buffer, but with subject-hash sharding spreading rows
    uniformly across ~1,118 shards, every shard tends to cross that
    threshold within the same processing window -- the real aggregate
    high-water mark is closer to ``SHARD_FLUSH_ROW_THRESHOLD * n_shards``
    (a real, measured ~279M rows resident at once, not the per-shard
    figure this docstring used to claim), at ~250-400 bytes/row in a
    pandas object-dtype frame with string codes plus each flush's own
    ``pd.concat`` copy -- enough to OOM-kill both a 64 GB and a 96 GB job.
    :data:`WRITER_MAX_BUFFERED_ROWS` (:attr:`total_buffered_rows`,
    force-flushing the fullest shards first via
    :meth:`_flush_fullest_shards_until_under_cap`) is the actual memory
    bound now; see its own docstring for the full incident. Also needs one
    open Parquet writer per shard actually touched -- with thousands of
    subjects per :data:`SUBJECTS_PER_SHARD`\\ =1000 shard, a large subject
    universe could need `ulimit -n` raised above a typical default (1024),
    flagged here rather than solved by a multi-pass design, which would
    need re-scanning every source table once per shard group at real
    scale, a far worse tradeoff.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to write shard files into (created if missing). Never a
        git-tracked path -- see the module docstring.
    shard_by_subject : dict[str, int]
        From :func:`assign_shards`.
    """

    def __init__(self, output_dir: Path, shard_by_subject: dict[str, int]) -> None:
        self.output_dir = output_dir
        self.shard_by_subject = shard_by_subject
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._shard_row_counts: dict[int, int] = {}
        self._buffers: dict[int, list[pd.DataFrame]] = {}
        self._buffer_row_counts: dict[int, int] = {}
        self.rows_written_per_table: dict[str, int] = {}
        self.rows_dropped_unshardable = 0
        #: Rows currently buffered across *every* shard -- see
        #: :data:`WRITER_MAX_BUFFERED_ROWS`'s docstring for why this,
        #: not the per-shard :data:`SHARD_FLUSH_ROW_THRESHOLD`, is the
        #: real memory bound. Kept exactly in sync with
        #: ``sum(self._buffer_row_counts.values())`` by every mutation
        #: site (buffering in :meth:`write_batch`, clearing in
        #: :meth:`_flush_shard`) rather than recomputed, since it's read
        #: on every :meth:`write_batch` call.
        self.total_buffered_rows = 0
        #: ``shard -> (tmp_path, final_path)`` for every currently-open
        #: writer -- see :meth:`_writer_for`'s docstring for why a writer
        #: is opened against ``tmp_path`` (never counted, never globbed)
        #: and only renamed to ``final_path`` once :meth:`flush_all`
        #: closes it successfully.
        self._writer_paths: dict[int, tuple[Path, Path]] = {}
        self._clean_leftover_tmp_files()

    def _clean_leftover_tmp_files(self) -> None:
        """Delete (and log) any ``*.parquet.tmp`` file already on disk at startup.

        A ``.tmp`` file can only exist here because a *previous* process
        was killed mid-write, before :meth:`flush_all` ever renamed it to
        its final name (see :meth:`_writer_for`) -- this fresh instance's
        own ``self._writers``/``self._writer_paths`` start empty, so it
        never created any of these itself. Real incident this prevents: a
        killed process's truncated, footer-less part-file used to land
        under a *countable* final name, and :func:`_logical_shard_row_counts`
        would only discover it much later (loudly, per that function's own
        docstring, but only once something finally tried to read it) --
        cleaning up known-orphaned ``.tmp`` files here, at the start of
        every run, means a stale one is never even reachable by that path.
        Deliberately global (every ``.tmp`` file under ``output_dir``, not
        scoped to one table) since a ``.tmp`` filename carries no table
        identity to scope by, and any one found here is unconditionally
        safe to discard regardless of which table's crashed attempt made it
        -- see :func:`_next_shard_write_path`/:func:`_logical_shard_row_counts`,
        neither of which ever counts a ``.tmp``-suffixed path as real output.
        """
        for tmp_path in self.output_dir.glob("shard_*.parquet.tmp"):
            logger.warning(
                "[extract_meds] deleting leftover tmp part-file from a prior "
                "killed process (never renamed, so never counted as real "
                "output): %s",
                tmp_path,
            )
            tmp_path.unlink()

    def _writer_for(self, shard: int) -> pq.ParquetWriter:
        """Return (opening if needed) the open writer for ``shard``.

        Opens against ``<final_name>.parquet.tmp``, never the final
        ``<final_name>.parquet`` name directly -- real incident this
        prevents: a process killed mid-write previously left a truncated,
        footer-less file under a *final*, countable name, since
        ``pq.ParquetWriter`` only ever writes a valid footer at
        ``close()``. A ``.tmp``-suffixed path is never matched by
        :data:`_SHARD_FILE_RE`, :func:`_shard_glob`,
        :func:`_next_shard_write_path`, or
        :func:`_logical_shard_row_counts`'s glob (all of which require a
        literal ``.parquet`` at the end) -- an ``os.rename`` in
        :meth:`flush_all` promotes it to the final, countable name only
        once ``close()`` has actually written a valid footer, and NFS's
        same-directory rename is atomic enough that no reader can ever
        observe a partially-renamed file.
        """
        if shard not in self._writers:
            final_path = _next_shard_write_path(self.output_dir, shard)
            tmp_path = final_path.parent / f"{final_path.name}.tmp"
            self._writers[shard] = pq.ParquetWriter(str(tmp_path), MEDS_ARROW_SCHEMA)
            self._writer_paths[shard] = (tmp_path, final_path)
            self._shard_row_counts[shard] = 0
        return self._writers[shard]

    def _flush_shard(self, shard: int) -> None:
        frames = self._buffers.pop(shard, None)
        self.total_buffered_rows -= self._buffer_row_counts.get(shard, 0)
        self._buffer_row_counts[shard] = 0
        if not frames:
            return
        combined = (
            frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        )
        arrow_table = pa.Table.from_pandas(
            combined[MEDS_COLUMNS], schema=MEDS_ARROW_SCHEMA, preserve_index=False
        )
        self._writer_for(shard).write_table(arrow_table)
        self._shard_row_counts[shard] = self._shard_row_counts.get(shard, 0) + len(
            combined
        )

    def _flush_fullest_shards_until_under_cap(self) -> None:
        """Force-flush the most-buffered shards first until under the global cap.

        Real incident this fixes: uniform subject-hash sharding means every
        shard tends to fill at close to the same rate, so
        :data:`SHARD_FLUSH_ROW_THRESHOLD`'s per-shard bound alone lets all
        ~1,118 shards approach it in lockstep -- a synchronized aggregate
        high-water of up to ``SHARD_FLUSH_ROW_THRESHOLD * n_shards`` (a
        real, measured ~279M rows), which OOM-killed both a 64 GB and a
        96 GB extraction job at almost exactly that mark. Called from
        :meth:`write_batch` once :attr:`total_buffered_rows` exceeds
        :data:`WRITER_MAX_BUFFERED_ROWS`; flushing the fullest shards
        first (rather than, say, insertion order) both clears the most
        memory per flush and desynchronizes future waves, since a flushed
        shard restarts from 0 while less-full shards keep accumulating
        independently.
        """
        while self.total_buffered_rows > WRITER_MAX_BUFFERED_ROWS:
            fullest_shard = max(
                self._buffer_row_counts,
                key=lambda s: self._buffer_row_counts[s],
                default=None,
            )
            if fullest_shard is None or self._buffer_row_counts[fullest_shard] == 0:
                break  # nothing left buffered -- total_buffered_rows must already be 0
            self._flush_shard(fullest_shard)

    def write_batch(self, table: str, batch: pd.DataFrame) -> tuple[int, int]:
        """Split one MEDS-shaped batch by shard and buffer it for writing.

        Parameters
        ----------
        table : str
            Source table name, for the per-table row-count tally in the
            eventual summary -- not written to the Parquet itself.
        batch : pandas.DataFrame
            Must carry exactly :data:`MEDS_COLUMNS`.

        Returns
        -------
        tuple[int, int]
            ``(n_written, n_dropped_unshardable)`` for this call --
            :func:`run_extraction` sums these per table and asserts the
            total against what the table's generator actually emitted
            (see :data:`SHARD_FLUSH_ROW_THRESHOLD`'s neighboring incident
            note): a mismatch means a row silently vanished somewhere
            between the generator and durable disk, the exact failure
            class this accounting exists to make loud instead of silent.
        """
        if batch.empty:
            return 0, 0
        shard_ids = batch["subject_id"].map(self.shard_by_subject)
        unmatched = shard_ids.isna()
        n_dropped = 0
        if unmatched.any():
            n_dropped = int(unmatched.sum())
            self.rows_dropped_unshardable += n_dropped
            logger.warning(
                "%d rows for %s had a subject_id with no shard assignment, dropping",
                n_dropped,
                table,
            )
            batch = batch.loc[~unmatched]
            shard_ids = shard_ids.loc[~unmatched]
        if batch.empty:
            return 0, n_dropped
        n_written = len(batch)
        self.rows_written_per_table[table] = (
            self.rows_written_per_table.get(table, 0) + n_written
        )
        for raw_shard, group in batch.groupby(shard_ids.astype(int)):
            shard = int(raw_shard)
            self._buffers.setdefault(shard, []).append(group)
            n_group = len(group)
            self._buffer_row_counts[shard] = (
                self._buffer_row_counts.get(shard, 0) + n_group
            )
            self.total_buffered_rows += n_group
            if self._buffer_row_counts[shard] >= SHARD_FLUSH_ROW_THRESHOLD:
                self._flush_shard(shard)
        if self.total_buffered_rows > WRITER_MAX_BUFFERED_ROWS:
            self._flush_fullest_shards_until_under_cap()
        return n_written, n_dropped

    def flush_all(self) -> None:
        """Force-flush every buffered shard and close its writer, durably.

        A Parquet file's footer (schema, row-group index -- everything a
        reader needs) is only written on ``pq.ParquetWriter.close()``;
        ``write_table()`` alone leaves an unreadable, footer-less file if
        the process dies before ``close()`` runs. Real incident this
        closes: a resumed run's manifest is marked ``"complete"`` per
        table (see :func:`run_extraction`), but before this method
        existed, every writer's actual ``close()`` happened exactly once,
        at the very end of the whole run, after every table. One real run
        buffered ``ipscu_subset``'s ~1.08M events (under
        :data:`SHARD_FLUSH_ROW_THRESHOLD` per shard), marked it complete,
        then crashed on a later table (``lab_subset``) before that
        end-of-run ``close()`` ever ran -- silently losing every
        `ipscu_subset`` row while the manifest still claimed completion,
        so every subsequent resumed run skipped re-extracting it.

        Called by :func:`run_extraction` immediately after each table's
        generator is drained, *before* that table's manifest entry is
        marked complete -- durability must precede the claim of
        durability, always. Closing every open writer (not just flushing
        buffers into it) is what makes a shard's on-disk state actually
        valid, readable Parquet. The next table's writes to the same
        shard reopen a fresh ``_partN`` file via
        :func:`_next_shard_write_path`'s existing multi-part-shard design
        (already built for the resumed-run case) -- this costs one extra
        small part file per shard per table boundary, never a rewrite of
        anything already on disk.

        Each writer is closed against its ``.tmp`` path and only then
        ``os.rename``d to its real, countable final name (see
        :meth:`_writer_for`) -- the rename is the single moment a shard's
        new part file becomes visible to :func:`_shard_glob`,
        :func:`_next_shard_write_path`, and
        :func:`_logical_shard_row_counts`, and it only happens after
        ``close()`` has already written a valid footer, so nothing ever
        observes a countable-but-truncated file.
        """
        for shard in list(self._buffers):
            self._flush_shard(shard)
        for shard, writer in self._writers.items():
            writer.close()
            tmp_path, final_path = self._writer_paths[shard]
            os.rename(tmp_path, final_path)
        self._writers.clear()
        self._writer_paths.clear()
        self._shard_row_counts.clear()

    def close(self) -> dict[int, int]:
        """Flush every buffered shard, close every writer, return logical row counts.

        Returns
        -------
        dict[int, int]
            ``{shard_index: row_count}``, aggregated across every part file
            on disk (see :func:`_logical_shard_row_counts`) -- not just
            what this instance itself wrote, since a resumed run's rows for
            an already-completed table live in a file this instance never
            touched.
        """
        self.flush_all()
        return _logical_shard_row_counts(self.output_dir)

    def discard_incomplete_writers(self) -> None:
        """Safely abandon every still-open writer's ``.tmp`` file, in-memory state included.

        Meant to be called from an ``except`` clause wrapping the table
        currently mid-flight when it raises -- see :func:`_extract_one_table`.
        Unconditionally safe, unlike the blanket "close writers on any
        exception" the durability fix's own commit message explicitly
        rejected: back then, closing a partially-written writer would
        finalize a *countable*, final-named file that a retry (which
        redoes the whole table from scratch) would then double-count
        alongside its own fresh part files. Now that a writer only ever
        touches its ``.tmp`` name (see :meth:`_writer_for`) until
        :meth:`flush_all` renames it, discarding here is just deleting a
        file nothing has ever counted or globbed -- the retry recreates it
        from scratch under a fresh part number, losing nothing and
        double-counting nothing.

        Deliberately does *not* try to close each ``pq.ParquetWriter``
        cleanly first: the exception being handled may have left the
        writer or the underlying connection in an unknown state, and
        since the ``.tmp`` file is being deleted regardless, there is
        nothing to gain from risking a second exception out of ``close()``
        -- any secondary error there is swallowed, not re-raised, so the
        caller's original exception is what actually propagates.
        """
        for shard, writer in self._writers.items():
            try:
                writer.close()
            except Exception:  # noqa: BLE001 -- best-effort; the tmp file is discarded either way
                pass
            tmp_path, _final_path = self._writer_paths.get(shard, (None, None))
            if tmp_path is not None and tmp_path.exists():
                logger.warning(
                    "[extract_meds] discarding incomplete tmp part-file after "
                    "an exception mid-table (safe -- never counted as real "
                    "output): %s",
                    tmp_path,
                )
                tmp_path.unlink()
        self._writers.clear()
        self._writer_paths.clear()
        self._shard_row_counts.clear()
        for shard in list(self._buffers):
            self._buffers.pop(shard, None)
            self.total_buffered_rows -= self._buffer_row_counts.get(shard, 0)
            self._buffer_row_counts[shard] = 0


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


def _extract_one_table(
    table_name: str, generator: Iterator[ExtractedBatch], writer: MedsShardWriter
) -> None:
    """Drain one table's generator into ``writer``, durably, before returning.

    Split out of :func:`run_extraction` so that function's own statement
    count stays under ruff's PLR0915 threshold; this holds the per-table
    batch loop, timing log, and the durability guard below.

    Calls :meth:`MedsShardWriter.flush_all` once the generator is
    exhausted -- **before** :func:`run_extraction` marks this table
    complete in the manifest -- and asserts the generator's own emitted
    row count against what the writer durably accounts for
    (written + dropped-unshardable), raising loudly on any mismatch
    rather than letting a silently-lost row through. See
    :meth:`MedsShardWriter.flush_all`'s docstring for the real incident
    (``ipscu_subset``'s buffered-but-unflushed rows lost when a *later*
    table crashed before the single end-of-run ``close()`` ever ran) this
    ordering closes.

    If this table's own batch loop raises, :meth:`MedsShardWriter.discard_incomplete_writers`
    cleans up before the exception propagates -- safe now that a writer
    only ever touches a ``.tmp`` name until :meth:`MedsShardWriter.flush_all`
    renames it (see that method's own docstring for why the durability
    fix's original commit explicitly rejected doing this before the
    atomic-visibility fix existed).
    """
    logger.info("[extract_meds] extracting %s...", table_name)
    report_progress = _log_table_progress(table_name)
    gen_iter = iter(generator)
    global _datetime_parse_seconds  # noqa: PLW0603
    n_generated = 0
    n_written = 0
    n_dropped = 0
    try:
        while True:
            _datetime_parse_seconds = 0.0
            t_generate_start = time.time()
            try:
                batch, source_rows = next(gen_iter)
            except StopIteration:
                break
            t_generate = time.time() - t_generate_start
            t_parse = _datetime_parse_seconds
            t_transform = max(t_generate - t_parse, 0.0)
            t_write_start = time.time()
            n_generated += len(batch)
            batch_written, batch_dropped = writer.write_batch(table_name, batch)
            n_written += batch_written
            n_dropped += batch_dropped
            t_write = time.time() - t_write_start
            logger.info(
                "[extract_meds] %s: batch phase timing (%d rows) "
                "parse=%.1fms transform=%.1fms write=%.1fms total_buffered=%d",
                table_name,
                source_rows,
                t_parse * 1000,
                t_transform * 1000,
                t_write * 1000,
                writer.total_buffered_rows,
            )
            report_progress(source_rows)
    except BaseException:
        writer.discard_incomplete_writers()
        raise
    writer.flush_all()
    if n_generated != n_written + n_dropped:
        raise RuntimeError(
            f"[extract_meds] {table_name}: generator emitted {n_generated} rows "
            f"but the writer only accounts for {n_written} written + "
            f"{n_dropped} dropped-unshardable -- rows went missing between "
            "the generator and durable disk, refusing to mark this table "
            "complete"
        )


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
    subject_by_genc, admission_by_genc, n_dropped_null_subject = fetch_admission_index()
    logger.info("[extract_meds] %d encounters indexed", len(subject_by_genc))

    mortality_by_genc = fetch_mortality_index()
    logger.info(
        "[extract_meds] %d encounters with a known mortality flag",
        len(mortality_by_genc),
    )

    logger.info("[extract_meds] fetching deduplicated lab concept lookup...")
    lab_concepts = fetch_lab_concept_lookup()

    logger.info("[extract_meds] building billing discharge index...")
    discharge_by_genc = fetch_discharge_index()

    shard_by_subject = assign_shards(subject_by_genc.values())
    n_shards = len(set(shard_by_subject.values()))
    logger.info(
        "[extract_meds] %d subjects -> %d shards", len(shard_by_subject), n_shards
    )

    writer = MedsShardWriter(target_dir, shard_by_subject)
    writer.rows_dropped_unshardable += n_dropped_null_subject
    table_generators: list[tuple[str, Iterator[ExtractedBatch]]] = [
        ("admdad_subset", extract_admissions(subject_by_genc, admission_by_genc)),
        (
            "admdad_subset__death",
            extract_death(subject_by_genc, admission_by_genc, mortality_by_genc),
        ),
        ("ipscu_subset", extract_icu(subject_by_genc)),
        ("lab_subset", extract_labs(subject_by_genc, lab_concepts)),
        ("vitals_subset", extract_vitals(subject_by_genc)),
        ("vitals_subset__unmapped", extract_vitals_unmapped(subject_by_genc)),
        ("pharmacy_subset", extract_pharmacy(subject_by_genc, admission_by_genc)),
        ("ipdiagnosis_subset", extract_diagnoses(subject_by_genc)),
        ("ipintervention_subset", extract_procedures(subject_by_genc)),
        ("radiology_subset", extract_radiology(subject_by_genc, admission_by_genc)),
        ("physicians_subset", extract_providers(subject_by_genc, admission_by_genc)),
        ("er_subset", extract_er(subject_by_genc, admission_by_genc)),
        (
            "erdiagnosis_subset",
            extract_er_diagnoses(subject_by_genc, admission_by_genc),
        ),
        (
            "erintervention_subset",
            extract_er_procedures(subject_by_genc, admission_by_genc),
        ),
        (
            "erintervention_subset__untimed",
            extract_er_procedures_untimed(subject_by_genc, admission_by_genc),
        ),
        ("erconsults_subset", extract_er_consults(subject_by_genc, admission_by_genc)),
        (
            "lookup_transfer_subset",
            extract_transfers(subject_by_genc, admission_by_genc),
        ),
        ("ipcmg_subset", extract_billing_cmg(subject_by_genc, discharge_by_genc)),
        ("iphig_subset", extract_billing_hig(subject_by_genc, discharge_by_genc)),
    ]
    for table_name, generator in table_generators:
        if manifest.get(table_name) == "complete":
            logger.info("[extract_meds] %s already complete, skipping", table_name)
            continue
        _extract_one_table(table_name, generator, writer)
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
