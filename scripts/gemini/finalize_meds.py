#!/usr/bin/env python
"""GEMINI MEDS finalize: hash->int64 remap, split assignment, MEDS-conformant layout.

Post-extraction pass, run only once ``extract_meds.py``'s manifest shows
every table complete (checked here, not assumed). ``extract_meds.py``
writes a flat, fast-to-stream layout that ``odyssey/data/meds_validation.py``
does not accept as-is: ``subject_id`` is the raw hashed patient string (not
``Int64``), shards live directly under the output dir rather than under
``data/<split>/``, a shard may be split across ``shard_NNNN.parquet`` +
``shard_NNNN_partN.parquet`` files (the resumability fix's own artifact),
and there is no ``metadata/`` directory at all. This module rewrites that
output into the conformant layout in one pass, then calls
:func:`odyssey.data.meds_validation.validate_meds_dataset` itself and
refuses to declare success on anything but zero errors -- the conformance
check is this step's own definition of done, not a separate manual step.

Deliberately never imports from or modifies ``extract_meds.py`` (a
different script, not a shared library -- see that module's own docstring
on why GEMINI-facing scripts don't cross-import; small constants/helpers
needed here are duplicated, not imported) and never touches Amrit's
extraction run: everything here operates on already-written output plus
one small, independent, read-only query of its own (:func:`_fetch_hadm_id_hospital`).

Design
------
**Split**: seeded (:data:`FINALIZE_SPLIT_SEED`) subject-random three-way,
:data:`FINALIZE_SPLIT_FRACS` = 80/10/10 train/tuning/held_out -- this is
not an arbitrary choice "matching convention," it is ``MEDS_extract``'s own
default ``split_fracs_dict`` (the package this repo's MIMIC-IV pipeline
already depends on), so it is the one real shared default across sources,
not a guess. Baked into ``metadata/subject_splits.parquet``; the seed and
fractions are also recorded in ``metadata/dataset.json`` so the exact
assignment is reproducible from the recorded config alone. Two stronger,
GEMINI-specific evaluation protocols are deliberately left *derivable at
eval time* rather than baked into this split: hospital-held-out (via
``metadata/hadm_id_hospital.parquet`` -- ``hadm_id`` is already a standing
extension column on every MEDS row, so this is a plain join, not a new
key) and temporal validation (a scoring-time cutoff against the real
``Datetime`` values already on every event -- nothing new to persist).

**Subject id**: ``subject_id = int(sha256(patient_id_hashed).hexdigest()[:16], 16)
& 0x7FFFFFFFFFFFFFFF`` -- the same hash-then-mod-into-a-fixed-space pattern
:func:`extract_meds.assign_shards` already uses for shard hashing, applied
here to get a stable, deterministic, re-runnable ``Int64``. Deterministic
in the sense that matters: a given hashed string always maps to the same
int64 regardless of what other subjects exist in the run, unlike a
positional/sequential enumeration, which would shift under any change to
the subject universe. Collision-checked at generation time (2^63 space,
vanishingly unlikely at real scale) and raises loudly if one is ever found
-- the same "fail loud, never let a bad case hide" principle
``extract_meds.py`` uses throughout. Persisted as its own
``<output_dir>/subject_id_mapping.parquet`` -- deliberately *not* under
``metadata/``: unlike everything else this step writes, it is real linkage
back to the raw hashed patient id, so it gets the same server-side-only
governance as the rest of this pipeline's output, kept visibly separate
from the actual MEDS-consumed structure.

**Layout rewrite and its two-pass memory shape**: a full remap-and-reshard
is unavoidable here (subject_id's type changes, and the split/shard
scheme for the *output* is unrelated to the *input* shard a row happened
to land in), so this step is structured as two passes with two different,
deliberately bounded memory shapes -- not the whole dataset resident at
once either way:

1. :func:`_repartition_pass` lazily scans every input shard file (base
   plus every ``_partN`` file) as one multi-file ``pl.scan_parquet``,
   computes each row's remapped ``subject_id`` and destination
   ``(split, output_shard)`` as chained lazy expressions (all three
   derived from the same ``remapped_subject`` expression object in one
   ``with_columns`` call, mirroring the prior eager version's Series-level
   dependency order exactly -- see :func:`_repartition_pass`'s own
   docstring), and hands the whole pipeline to polars' native
   ``sink_parquet(pl.PartitionBy(key=["__split", "__out_shard"], ...))``.
   The engine owns its own internal buffering/flushing -- no hand-rolled
   per-destination buffer-and-flush loop. This replaced an earlier
   version of this pass that buffered rows per destination in Python
   (flushed at :data:`REPARTITION_FLUSH_ROW_THRESHOLD`-row intervals, the
   same fix ``extract_meds.MedsShardWriter`` uses) -- a real scratch
   benchmark (20M rows, 100 input files, 400 output destinations at real
   ``FINALIZE_SUBJECTS_PER_SHARD`` sizing, 2 runs each, peak RSS via
   ``RUSAGE_CHILDREN``) found the native sink both faster and lighter:
   0.52-0.72s wall / 1.35-1.42GB peak RSS for the native sink vs.
   4.62-4.77s wall / 1.86-1.88GB peak RSS for the buffered version, with
   an exhaustive row-set equality check (every one of 400 destinations,
   not a sample) confirming byte-identical output between the two. The
   buffered version's overhead scales with row count through
   Python-level ``group_by`` dispatch and an ``to_pandas()``/
   ``pa.Table.from_pandas()`` round trip per flush, neither of which
   vectorizes the way the native engine's internal scatter does, so the
   gap is expected to hold or widen at real (~279M-row worst-case) scale,
   not shrink -- not independently verified at that scale, since this
   pass is now the sub-minute part of the run either way. ``PartitionBy``
   is marked **unstable** by polars itself (may change without a
   semver-breaking bump); the exact surface this module depends on
   (``key=`` with multiple columns, ``include_key=False``,
   ``file_path_provider=``) is pinned down by its own dedicated test
   (see ``tests/scripts/gemini/test_finalize_meds.py``), so a future
   polars bump that changes this API fails the test suite, not a live
   GEMINI run.
2. :func:`_sort_and_finalize_shard` runs once per *output* shard, after
   every input shard has been fully repartitioned (a subject's complete
   row set for that shard destination is only known once every input
   shard has been scanned). This is the pass whose memory really is
   bounded to "roughly one shard's rows at a time": it reads one
   unsorted, already-repartitioned output shard fully into memory, sorts
   it by ``(subject_id, time)`` with nulls first (MEDS's convention:
   static-fact events with a null time sort first within their subject),
   and writes it to its final path.

**Part-file compaction** falls out of the above for free: every input
shard's base file and every ``_partN`` file are read together in pass 1
and never referenced again, so the multi-part layout is fully erased by
construction -- there is nothing left to separately compact.

**Crash semantics**: while this step is running, the *original*
flat-layout files under ``output_dir`` are never touched -- ``data/`` and
``metadata/`` are new subdirectories, and the old files are only deleted
at the very end, after :func:`~odyssey.data.meds_validation.validate_meds_dataset`
has already returned zero errors on the freshly-written output. A finalize
that dies partway through therefore always leaves the original extraction
output intact. It does **not** leave a resumable partial state, though: a
completion sentinel (``metadata/.finalize_complete``) is written only as
the very last step. A re-run that finds ``data/`` or ``metadata/`` present
without that sentinel treats it as a dead attempt and deletes both before
starting over from scratch -- fresh-start-over-partial, no resume
complexity, since the whole pass is single-digit-hours at worst at real
scale. A re-run that finds the sentinel already present refuses outright
(never silently redoes already-successful, expensive work); delete
``data/``/``metadata/`` by hand first if a genuine re-run is wanted.

**Preflight**: two checks before anything is touched. Disk:
:func:`_preflight_disk` requires free space on ``output_dir``'s filesystem
at least equal to the current flat-layout dataset size (the rewrite
temporarily roughly doubles on-disk footprint until the old files are
deleted at the end) -- fails loudly with the exact byte counts, not a
generic "disk full" surfaced mid-pass. File descriptors:
:func:`_preflight_nofile` is the same raise-the-soft-limit-or-fail-loudly
check ``extract_meds.preflight_shard_capacity`` uses, sized to
``n_output_shards_total`` (this pass's writers fan out across every
``(split, output_shard)`` destination over the course of pass 1, not one
shard at a time, so the fd count that matters is the total, computed
after split assignment, before pass 1 starts).

Run on the GEMINI node, after ``extract`` has fully completed:

    uv run python scripts/gemini/finalize_meds.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl

from odyssey.data.gemini import db
from odyssey.data.meds_validation import validate_meds_dataset


logger = logging.getLogger(__name__)

#: Same default as extract_meds.OUTPUT_DIR -- duplicated, not imported,
#: see the module docstring.
OUTPUT_DIR = Path(
    os.environ.get(
        "GEMINI_MEDS_OUTPUT_DIR",
        "/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1",
    )
)

#: Same convention as extract_meds.SUMMARY_PATH: a small, suppressed
#: (no patient-level detail) summary under the repo's own scripts/gemini/out/,
#: so run.sh's commit-and-push step can ride it back to git the same way
#: extraction_summary.json already does. Real incident this exists to fix:
#: a completed finalize run's tail showed the summary on stdout, then
#: "Nothing to commit" -- the summary was never written to a tracked path
#: at all, so it never made it back to git.
SUMMARY_PATH = Path(__file__).resolve().parent / "out" / "finalize_summary.json"

#: Matches extract_meds.SUBJECTS_PER_SHARD -- duplicated, not imported.
FINALIZE_SUBJECTS_PER_SHARD = 1000

#: Fixed literal, recorded in dataset.json -- the exact value doesn't
#: matter for correctness, only that it's fixed and reproducible.
FINALIZE_SPLIT_SEED = 0

#: MEDS_extract's own default split_fracs_dict -- see the module docstring.
#: Dict order matters: the last split absorbs whatever subject count
#: rounding leaves over, so every subject is assigned exactly once.
FINALIZE_SPLIT_FRACS = {"train": 0.8, "tuning": 0.1, "held_out": 0.1}

MEDS_COLUMNS = ["subject_id", "time", "code", "numeric_value", "hadm_id"]

#: Same shape as extract_meds.MEDS_ARROW_SCHEMA except subject_id is Int64
#: here, not the raw hashed string -- the whole point of this rewrite.
#: Expressed as polars dtypes (not a pyarrow schema) since _repartition_pass
#: now casts to these directly in the lazy pipeline before sinking -- see
#: that function's docstring.
MEDS_POLARS_DTYPES_FINAL: dict[str, pl.DataType] = {
    "subject_id": pl.Int64(),
    "time": pl.Datetime("us"),
    "code": pl.Utf8(),
    "numeric_value": pl.Float64(),
    "hadm_id": pl.Int64(),
}

#: See extract_meds.FD_HEADROOM -- duplicated, not imported.
FD_HEADROOM = 64

#: Matches extract_meds.py's own flat shard filename convention.
_SHARD_FILE_RE = re.compile(r"^shard_(\d{4})(?:_part(\d+))?\.parquet$")

#: Matches _repartition_pass's own sink_parquet(PartitionBy(...))
#: file_path_provider naming convention: "{split}__{out_idx:04d}_{part:03d}.parquet".
#: Parts beyond _000 only occur if the sink engine ever splits one
#: destination across multiple physical files -- not observed at the
#: scratch benchmark's 20M-row scale (400/400 destinations were 1 file
#: each), but _sort_and_finalize_shard's caller globs on this pattern and
#: concatenates every match per destination regardless, so it's correct
#: either way.
_PARTITION_FILE_RE = re.compile(r"^([a-z_]+)__(\d{4})_(\d{3})\.parquet$")

#: Non-negative-63-bit mask for the hash-based subject_id (see
#: _build_subject_id_mapping) -- clears the sign bit so the value is
#: always a valid, non-negative signed Int64.
_SUBJECT_ID_MASK = (1 << 63) - 1

#: Written as the very last step of a successful run -- see the module
#: docstring's "Crash semantics" section.
_FINALIZE_SENTINEL_RELATIVE = "metadata/.finalize_complete"

_HOSPITAL_LOOKUP_SQL = (
    "SELECT {genc_id} AS genc_id, {hospital_num} AS hospital_num FROM {table}"
)


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded double quotes.

    Duplicated from extract_meds.py rather than imported -- see the module
    docstring on why GEMINI-facing scripts don't cross-import.
    """
    return '"' + name.replace('"', '""') + '"'


def _suppressed(n: int) -> str:
    """Round ``n`` to the nearest 1000, or mask small counts.

    Mirrors ``extract_meds._suppressed``/``explore_schema.suppressed_row_count``/
    ``extract_dry._suppressed`` exactly (small-cell suppression) -- duplicated,
    not imported, see the module docstring.

    Returns
    -------
    str
        ``"<6"`` under 6, otherwise rounded to the nearest 1000.
    """
    if n < 6:
        return "<6"
    return str(round(n / 1000) * 1000)


def _shard_count(
    n_subjects: int, subjects_per_shard: int = FINALIZE_SUBJECTS_PER_SHARD
) -> int:
    """Return ``ceil(n_subjects / subjects_per_shard)``, at least 1.

    Duplicated from extract_meds.py, not imported -- see the module docstring.
    """
    return max(1, -(-n_subjects // subjects_per_shard))


def _input_shard_files(output_dir: Path) -> dict[int, list[Path]]:
    """Group the flat ``shard_NNNN[_partN].parquet`` files by logical shard index."""
    groups: dict[int, list[Path]] = {}
    for path in sorted(output_dir.glob("shard_*.parquet")):
        match = _SHARD_FILE_RE.match(path.name)
        if match is None:
            continue
        groups.setdefault(int(match.group(1)), []).append(path)
    return groups


def _dataset_size_bytes(shard_files: dict[int, list[Path]]) -> int:
    return sum(p.stat().st_size for files in shard_files.values() for p in files)


def _preflight_disk(root: Path, shard_files: dict[int, list[Path]]) -> None:
    """Fail loudly, before touching anything, if there isn't room for the rewrite.

    See the module docstring's "Preflight" section: the rewrite keeps the
    old flat-layout files in place until the very end, so peak disk usage
    is roughly the old layout's size plus the new one's (comparable size),
    not just the new layout alone.
    """
    needed = _dataset_size_bytes(shard_files)
    free = shutil.disk_usage(root).free
    if free < needed:
        raise RuntimeError(
            f"finalize needs at least {needed / 1e9:.1f} GB free on {root}'s "
            f"filesystem (the current flat-layout dataset's size -- the "
            f"rewrite keeps it in place alongside the new output until "
            f"validation passes) but only {free / 1e9:.1f} GB is free. Free "
            f"up space before retrying; nothing has been touched."
        )
    logger.info(
        "[finalize] disk preflight OK: %.1f GB needed, %.1f GB free",
        needed / 1e9,
        free / 1e9,
    )


def _preflight_nofile(n_output_shards_total: int) -> None:
    """Raise the soft NOFILE limit if needed, or fail loudly with the exact fix.

    Same pattern as extract_meds.preflight_shard_capacity, duplicated (not
    imported) -- see the module docstring. Sized to n_output_shards_total
    since the repartition pass's writers fan out across every (split,
    output_shard) destination over the pass, not one at a time.
    """
    needed = n_output_shards_total + FD_HEADROOM
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < needed:
        target = min(needed, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < needed:
        raise RuntimeError(
            f"Need at least {needed} open file descriptors for "
            f"{n_output_shards_total} output shards (+{FD_HEADROOM} headroom), "
            f"but the hard NOFILE limit ({hard}) won't allow raising the soft "
            f"limit ({soft}) that high. Run `ulimit -n {needed}` (or higher) "
            f"in your shell before retrying."
        )
    logger.info(
        "[finalize] NOFILE preflight OK: %d output shards, soft limit %d",
        n_output_shards_total,
        soft,
    )


def _wipe_partial_output(root: Path) -> None:
    """Enforce the module docstring's crash semantics before any new work starts.

    Refuses outright if a prior finalize already completed successfully
    (the sentinel is present) -- never silently redoes expensive,
    already-successful work. Otherwise, any data/ or metadata/ left behind
    by a prior, incomplete attempt is deleted (fresh-start-over-partial,
    no resume complexity).
    """
    sentinel = root / _FINALIZE_SENTINEL_RELATIVE
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    if sentinel.is_file():
        raise RuntimeError(
            f"{root} already has a completed finalize output ({sentinel} "
            "exists). Refusing to redo already-successful work -- delete "
            "data/ and metadata/ yourself first if a genuine re-run is "
            "really wanted."
        )
    if data_dir.exists() or metadata_dir.exists():
        logger.warning(
            "[finalize] partial output found with no completion sentinel -- "
            "wiping data/ and metadata/ from a prior, incomplete attempt "
            "before starting over."
        )
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(metadata_dir, ignore_errors=True)


def _collect_subject_universe(shard_files: dict[int, list[Path]]) -> list[str]:
    """Every distinct raw (hashed-string) subject_id across the whole dataset.

    A column-only scan (just ``subject_id``, not full rows) across every
    physical shard file, once each -- orders of magnitude cheaper than a
    full-row read, and unavoidable: the split/shard assignment computed
    from this can't start until the full subject universe is known.
    """
    subjects: set[str] = set()
    for files in shard_files.values():
        for path in files:
            subjects.update(
                pl.read_parquet(path, columns=["subject_id"])["subject_id"]
                .unique()
                .to_list()
            )
    return sorted(subjects)


def _build_subject_id_mapping(subjects: list[str]) -> dict[str, int]:
    """Build a deterministic hash-string -> Int64 map, collision-checked.

    See the module docstring's "Subject id" section for the full design.
    """
    mapping: dict[str, int] = {}
    seen: dict[int, str] = {}
    for subject in subjects:
        digest = hashlib.sha256(subject.encode()).hexdigest()
        candidate = int(digest[:16], 16) & _SUBJECT_ID_MASK
        prior = seen.get(candidate)
        if prior is not None and prior != subject:
            raise RuntimeError(
                f"subject_id hash collision: {subject!r} and {prior!r} both "
                f"map to {candidate} -- this is a real (if vanishingly "
                "unlikely) collision in the 63-bit hash space, not a bug in "
                "the collision check; the mapping scheme needs a wider hash "
                "or a collision-resolution strategy before this can proceed."
            )
        seen[candidate] = subject
        mapping[subject] = candidate
    return mapping


def _assign_splits(
    subject_ids: list[int],
    *,
    seed: int = FINALIZE_SPLIT_SEED,
    fracs: dict[str, float] = FINALIZE_SPLIT_FRACS,
) -> dict[int, str]:
    """Seeded subject-random three-way split. See the module docstring."""
    ordered = sorted(subject_ids)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    assignment: dict[int, str] = {}
    names = list(fracs)
    start = 0
    for i, name in enumerate(names):
        end = n if i == len(names) - 1 else start + round(fracs[name] * n)
        for subject_id in shuffled[start:end]:
            assignment[subject_id] = name
        start = end
    return assignment


def _assign_output_shards(
    split_by_subject: dict[int, str],
    *,
    subjects_per_shard: int = FINALIZE_SUBJECTS_PER_SHARD,
) -> tuple[dict[int, int], dict[str, int]]:
    """Compute a per-split, hash-based output shard index.

    Same pattern as extract_meds.assign_shards, applied per split.

    Returns
    -------
    tuple[dict[int, int], dict[str, int]]
        ``(output_shard_by_subject, n_shards_by_split)``.
    """
    subjects_by_split: dict[str, list[int]] = {}
    for subject_id, split in split_by_subject.items():
        subjects_by_split.setdefault(split, []).append(subject_id)
    n_shards_by_split: dict[str, int] = {}
    output_shard_by_subject: dict[int, int] = {}
    for split, subject_ids in subjects_by_split.items():
        n_shards = _shard_count(len(subject_ids), subjects_per_shard)
        n_shards_by_split[split] = n_shards
        for subject_id in subject_ids:
            output_shard_by_subject[subject_id] = (
                int(hashlib.sha256(str(subject_id).encode()).hexdigest(), 16) % n_shards
            )
    return output_shard_by_subject, n_shards_by_split


def _fetch_hadm_id_hospital() -> pd.DataFrame:
    """``genc_id -> hospital_num`` for every admission, straight from ``admdad_subset``.

    A small, independent DB read -- ~2.27M rows of two integer columns is
    tiny next to any real extraction table, so this goes through
    ``db.query``'s plain single round trip (same pattern as
    ``extract_meds.fetch_lab_concept_lookup``), not the COPY streaming
    machinery. Deliberately *not* sourced from ``extract_meds.py``'s own
    output, which doesn't capture ``hospital_num`` at all: this step must
    never touch or depend on the extractor itself, so Amrit's in-flight
    (or already-complete) extraction run is never put at risk.
    """
    return db.query(
        _HOSPITAL_LOOKUP_SQL.format(
            genc_id=_quote_ident("genc_id"),
            hospital_num=_quote_ident("hospital_num"),
            table=_quote_ident("admdad_subset"),
        )
    )


def _partition_file_path(args: pl.FileProviderArgs, *, tmp_dir: Path) -> str:
    """Deterministic ``{split}__{out_idx:04d}_{part:03d}.parquet`` naming.

    Matches :data:`_PARTITION_FILE_RE`. Keeping the naming under our own
    control (rather than polars' default) is what lets
    :func:`_repartition_pass`'s caller reconstruct ``{(split, out_shard):
    [paths]}`` from a plain directory listing after the sink completes,
    with no callback needed.
    """
    split_name = args.partition_keys["__split"][0]
    out_idx = args.partition_keys["__out_shard"][0]
    return str(
        tmp_dir / f"{split_name}__{out_idx:04d}_{args.index_in_partition:03d}.parquet"
    )


def _repartition_pass(
    shard_files: dict[int, list[Path]],
    subject_id_mapping: dict[str, int],
    output_shard_by_subject: dict[int, int],
    split_by_subject: dict[int, str],
    tmp_dir: Path,
) -> dict[tuple[str, int], list[Path]]:
    """Lazily scan every input shard once, sinking rows into per-destination files.

    Writes unsorted -- see the module docstring's "Layout rewrite and its
    two-pass memory shape" section for the full design, the native
    ``sink_parquet(pl.PartitionBy(...))`` this uses, and the benchmark it's
    based on. Returns ``{(split, output_shard_index): [paths]}`` for
    :func:`_sort_and_finalize_shard` -- a list, not one path, since the
    sink engine is free to split one destination across more than one
    physical file (not observed at benchmark scale, but not something to
    assume never happens at real scale either).

    The three lazy lookups (subject_id remap, split, output shard) are all
    built from the *same* ``remapped_subject`` expression object and
    applied in a single ``with_columns`` call, deliberately mirroring the
    prior eager implementation's dependency order: that version computed
    ``subject_int`` as one Series from the original (pre-remap)
    ``subject_id`` column, then derived ``split``/``out_shard`` from that
    *same* Series, then added all three as new columns in one call --
    never re-reading a column through its own just-written alias. The
    lazy version preserves that exactly, rather than e.g. re-referencing
    ``pl.col("subject_id")`` after it's been aliased to the remapped value
    (which would also work here, since a later ``with_columns`` call sees
    an earlier one's output, but wouldn't be the same translation being
    tested).
    """
    all_paths = [str(p) for files in shard_files.values() for p in files]
    lf = pl.scan_parquet(all_paths)

    remapped_subject = pl.col("subject_id").replace_strict(
        subject_id_mapping, default=None, return_dtype=pl.Int64
    )
    # __raw_subject_id keeps the pre-remap value around under its own name
    # purely so a null-remap failure can name the actual offending raw
    # subject_id in its error message (see below) -- by the end of this
    # with_columns call, "subject_id" itself has already been overwritten
    # with the remapped value, same as the prior eager version.
    lf = lf.with_columns(
        remapped_subject.alias("subject_id"),
        remapped_subject.replace_strict(
            split_by_subject, default=None, return_dtype=pl.Utf8
        ).alias("__split"),
        remapped_subject.replace_strict(
            output_shard_by_subject, default=None, return_dtype=pl.Int64
        ).alias("__out_shard"),
        pl.col("subject_id").alias("__raw_subject_id"),
    )

    # One combined check-and-count pass, not two: happy-path cost is a
    # single column-projected scan (same class of cost as
    # _collect_subject_universe's own scan), and doubles as the row total
    # for the completion heartbeat below. The (rare, error-path-only)
    # per-row examples for a loud failure message cost a second, smaller
    # scan -- only run if n_missing is actually nonzero.
    check = lf.select(
        pl.col("subject_id").is_null().sum().alias("n_missing"),
        pl.len().alias("n_rows"),
    ).collect()
    n_missing = int(check["n_missing"][0])
    total_rows = int(check["n_rows"][0])
    if n_missing:
        examples = (
            lf.filter(pl.col("subject_id").is_null())
            .select("__raw_subject_id")
            .unique()
            .head(3)
            .collect()["__raw_subject_id"]
            .to_list()
        )
        raise RuntimeError(
            f"{n_missing} rows have a subject_id not in the collected "
            f"universe (e.g. {examples}) -- the subject-collection pass "
            "and this pass must see the exact same input files"
        )

    lf = lf.select(
        pl.col("subject_id").cast(MEDS_POLARS_DTYPES_FINAL["subject_id"]),
        pl.col("time").cast(MEDS_POLARS_DTYPES_FINAL["time"]),
        pl.col("code").cast(MEDS_POLARS_DTYPES_FINAL["code"]),
        pl.col("numeric_value").cast(MEDS_POLARS_DTYPES_FINAL["numeric_value"]),
        pl.col("hadm_id").cast(MEDS_POLARS_DTYPES_FINAL["hadm_id"]),
        pl.col("__split"),
        pl.col("__out_shard"),
    )

    logger.info(
        "[finalize] repartition pass: scanning %d input files (%d rows) into "
        "destinations across %d splits...",
        len(all_paths),
        total_rows,
        len(set(split_by_subject.values())),
    )
    t0 = time.perf_counter()
    lf.sink_parquet(
        pl.PartitionBy(
            str(tmp_dir),
            key=["__split", "__out_shard"],
            include_key=False,
            file_path_provider=lambda args: _partition_file_path(args, tmp_dir=tmp_dir),
        ),
        mkdir=True,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "[finalize] repartition pass complete: %d rows across %d input files in %.1fs",
        total_rows,
        len(all_paths),
        elapsed,
    )

    paths: dict[tuple[str, int], list[Path]] = {}
    for path in tmp_dir.glob("*.parquet"):
        match = _PARTITION_FILE_RE.match(path.name)
        if match is None:
            continue
        key = (match.group(1), int(match.group(2)))
        paths.setdefault(key, []).append(path)
    return paths


def _sort_and_finalize_shard(
    tmp_paths: list[Path], final_path: Path
) -> tuple[int, dict[str, int]]:
    """Sort one repartitioned-but-unsorted output shard and write it to its final path.

    See the module docstring's "Layout rewrite" section: this is the pass
    whose peak memory really is bounded to roughly one output shard's row
    count. Also tallies this shard's per-code row counts for
    :func:`_write_codes_parquet`'s aggregate, since the data is already
    resident in memory here -- avoids a third full read of the same rows.

    ``tmp_paths`` is normally a single file (see
    :func:`_repartition_pass`'s docstring on why it's a list), read
    together the same way :func:`_input_shard_files`'s multi-part input
    shards are.

    Returns
    -------
    tuple[int, dict[str, int]]
        ``(row_count, code_counts)`` for this shard.
    """
    frame = (
        pl.concat([pl.read_parquet(p) for p in tmp_paths])
        if len(tmp_paths) > 1
        else pl.read_parquet(tmp_paths[0])
    )
    frame = frame.sort(["subject_id", "time"], nulls_last=False)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = final_path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp_final)
    os.replace(tmp_final, final_path)
    counts = frame.group_by("code").len()
    code_counts = dict(zip(counts["code"].to_list(), counts["len"].to_list()))
    return frame.height, code_counts


def _current_extractor_commit() -> Optional[str]:
    """Return the extractor's git commit hash for dataset.json's etl_version."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 -- purely informational, never worth failing over
        return None


def _write_dataset_json(root: Path, *, extractor_commit: Optional[str]) -> None:
    payload = {
        "dataset_name": "gemini_meds",
        "dataset_version": None,
        "etl_name": "extract_meds.py + finalize_meds.py",
        "etl_version": extractor_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": FINALIZE_SPLIT_SEED,
        "split_fracs": FINALIZE_SPLIT_FRACS,
    }
    (root / "metadata" / "dataset.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def _write_codes_parquet(root: Path, code_counts: dict[str, int]) -> None:
    frame = pl.DataFrame(
        {"code": list(code_counts.keys()), "count": list(code_counts.values())}
    ).sort("code")
    frame.write_parquet(root / "metadata" / "codes.parquet")


def _write_subject_splits(root: Path, split_by_subject: dict[int, str]) -> None:
    frame = pl.DataFrame(
        {
            "subject_id": list(split_by_subject.keys()),
            "split": list(split_by_subject.values()),
        }
    )
    frame.write_parquet(root / "metadata" / "subject_splits.parquet")


def _write_hadm_id_hospital(root: Path, hospital: pd.DataFrame) -> None:
    pl.from_pandas(hospital.rename(columns={"genc_id": "hadm_id"})).write_parquet(
        root / "metadata" / "hadm_id_hospital.parquet"
    )


def _write_all_metadata(
    root: Path,
    subject_id_mapping: dict[str, int],
    split_by_subject: dict[int, str],
    code_counts: dict[str, int],
) -> None:
    """Write subject_id_mapping.parquet plus every metadata/ file."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "patient_id_hashed": list(subject_id_mapping.keys()),
            "subject_id": list(subject_id_mapping.values()),
        }
    ).write_parquet(root / "subject_id_mapping.parquet")
    _write_dataset_json(root, extractor_commit=_current_extractor_commit())
    _write_codes_parquet(root, code_counts)
    _write_subject_splits(root, split_by_subject)
    _write_hadm_id_hospital(root, _fetch_hadm_id_hospital())


def run_finalize(output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Run the full finalize pass end to end and return a summary.

    Remaps subject ids, assigns splits, reshards, sorts, writes metadata,
    and validates the result -- see the module docstring for the design.

    Parameters
    ----------
    output_dir : pathlib.Path, optional
        Overrides :data:`OUTPUT_DIR` (mainly for tests). Must already hold
        a fully-complete ``extract_meds.py`` run (checked against
        ``extract_manifest.json``).

    Returns
    -------
    dict[str, Any]
        Summary: suppressed subject/split counts, shard count, warning
        count -- the same dict written to :data:`SUMMARY_PATH` (small
        enough, and suppressed enough via :func:`_suppressed`, to commit
        to git the way ``extract_meds.py``'s own ``extraction_summary.json``
        already does). Raises (does not return) if the resulting dataset
        fails conformance -- see the module docstring's "Crash semantics"
        section for what state that leaves on disk.
    """
    root = output_dir if output_dir is not None else OUTPUT_DIR
    # Checked before the manifest: a successful finalize deletes
    # extract_manifest.json as part of its own cleanup (see the end of
    # this function), so on a re-run after success the manifest is
    # legitimately gone -- checking the sentinel first gives a clear
    # "already finalized" error instead of a confusing "manifest not
    # found" that could read as extraction itself being broken.
    _wipe_partial_output(root)

    manifest_path = root / "extract_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{manifest_path} not found -- run `extract` to completion first"
        )
    manifest = json.loads(manifest_path.read_text())
    incomplete = [table for table, status in manifest.items() if status != "complete"]
    if not manifest or incomplete:
        raise RuntimeError(
            "extraction is not fully complete yet -- finalize only runs "
            f"post-extraction (incomplete: {incomplete or 'manifest is empty'})"
        )

    shard_files = _input_shard_files(root)
    if not shard_files:
        raise RuntimeError(f"no shard_*.parquet files found under {root}")

    _preflight_disk(root, shard_files)

    logger.info("[finalize] collecting the subject universe...")
    subjects = _collect_subject_universe(shard_files)
    logger.info("[finalize] %d distinct subjects", len(subjects))

    subject_id_mapping = _build_subject_id_mapping(subjects)
    split_by_subject = _assign_splits(list(subject_id_mapping.values()))
    output_shard_by_subject, n_shards_by_split = _assign_output_shards(split_by_subject)
    n_output_shards_total = sum(n_shards_by_split.values())
    _preflight_nofile(n_output_shards_total)

    data_dir = root / "data"
    tmp_dir = root / ".finalize_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[finalize] repartitioning %d input shards into %d output shards across %d splits...",
        len(shard_files),
        n_output_shards_total,
        len(n_shards_by_split),
    )
    tmp_paths = _repartition_pass(
        shard_files,
        subject_id_mapping,
        output_shard_by_subject,
        split_by_subject,
        tmp_dir,
    )

    logger.info("[finalize] sorting %d output shards...", len(tmp_paths))
    code_counts: dict[str, int] = {}
    for (split_name, out_idx), these_tmp_paths in tmp_paths.items():
        final_path = data_dir / split_name / f"shard_{out_idx:04d}.parquet"
        _, this_shard_codes = _sort_and_finalize_shard(these_tmp_paths, final_path)
        for code, count in this_shard_codes.items():
            code_counts[code] = code_counts.get(code, 0) + count
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("[finalize] writing metadata...")
    _write_all_metadata(root, subject_id_mapping, split_by_subject, code_counts)

    logger.info("[finalize] validating...")
    findings = validate_meds_dataset(root, deep=True)
    error_findings = [f for f in findings if f.severity == "error"]
    for finding in findings:
        (logger.error if finding.severity == "error" else logger.warning)(str(finding))
    if error_findings:
        raise RuntimeError(
            f"finalize produced a non-conformant dataset ({len(error_findings)} "
            "error(s), see log above) -- NOT deleting the old flat-layout "
            "output, NOT writing the completion sentinel. The next finalize "
            "run will wipe this partial data/+metadata/ output and start over."
        )

    sentinel = root / _FINALIZE_SENTINEL_RELATIVE
    sentinel.write_text(datetime.now(timezone.utc).isoformat() + "\n")

    logger.info("[finalize] deleting old flat-layout output...")
    for files in shard_files.values():
        for path in files:
            path.unlink()
    manifest_path.unlink()

    summary = {
        "n_subjects": _suppressed(len(subject_id_mapping)),
        "n_output_shards": n_output_shards_total,
        "splits": {
            name: _suppressed(sum(1 for s in split_by_subject.values() if s == name))
            for name in FINALIZE_SPLIT_FRACS
        },
        "warnings": len(findings) - len(error_findings),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("[finalize] wrote %s", SUMMARY_PATH)
    return summary


def main() -> None:
    """Run finalize and print where the summary landed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    summary = run_finalize()
    print(f"Wrote {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
