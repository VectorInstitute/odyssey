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
and eICU.

**Governance, same rule as everywhere else in this package (see
docs/gemini.md): MEDS parquet shards are patient-level data and never leave
GEMINI.** They are written to :data:`OUTPUT_DIR` on the enclave's own NFS
storage, never to a git-tracked path. Only a small, aggregate
``extraction_summary.json`` (rounded row/subject/shard counts, no
patient-level content) is meant to be committed -- ``scripts/gemini/run.sh
extract`` enforces this the same way it already does for every other step's
output (see ``run.sh``'s own path/size checks).

Design, one function per source table, streaming throughout:

- :func:`fetch_admission_index` -- one pass over ``admdad_subset`` mapping
  ``genc_id -> (patient_id_hashed, admission time)``, held in memory for the
  whole run (every event table carries only ``genc_id``, and the
  pharmacy/radiology timestamp guard needs the encounter's admission time
  as its anchor).
- One ``extract_<table>`` generator per source table, each a lazy,
  ``row_num``-keyset-paginated (:func:`_paginate_rows`) iterator of
  MEDS-shaped batches (:data:`MEDS_COLUMNS`) -- never loads a whole table
  into memory, which matters at ``lab_subset``'s/``vitals_subset``'s real
  scale (hundreds of millions of rows).
- :class:`MedsShardWriter` -- the shared writer: hashes each subject to a
  shard once (:func:`assign_shards`, deterministic, not Python's salted
  ``hash()``), then streams each incoming batch straight to that shard's
  open Parquet writer. Nothing accumulates across the whole run; memory use
  is bounded by one batch plus one open file handle per shard.
- :func:`run_extraction` -- orchestrates all of the above and writes the
  suppressed summary.

Run on the GEMINI node (writes real patient data to ``OUTPUT_DIR`` -- not a
dry run):

    uv run python scripts/gemini/extract_meds.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from odyssey.data.gemini import db


logger = logging.getLogger(__name__)

#: Where MEDS parquet shards are written -- enclave NFS only, never git.
#: Overridable via GEMINI_MEDS_OUTPUT_DIR; see docs/gemini_extraction.md's
#: Sharding and output section.
OUTPUT_DIR = Path(
    os.environ.get("GEMINI_MEDS_OUTPUT_DIR", str(Path.home() / "gemini_meds_v1"))
)

SUMMARY_PATH = Path(__file__).resolve().parent / "out" / "extraction_summary.json"

#: Target subjects per shard (rounded, see docs/gemini_extraction.md).
SUBJECTS_PER_SHARD = 1000

#: Rows per keyset-paginated batch read from Postgres.
BATCH_ROWS = 50_000

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


def _parse_gemini_datetime(raw: object) -> Optional[pd.Timestamp]:
    """Best-effort parse of one of GEMINI's text datetime columns.

    Real format for valid rows isn't confirmed yet (see
    docs/gemini_extraction.md's open question 7) -- uses pandas' flexible
    parser rather than a hardcoded format string. Returns ``None`` for
    anything missing or unparsable rather than raising: one malformed
    timestamp must not stop the whole extraction, the same principle
    ``extract_dry.py``'s per-column error recovery already established.

    Parameters
    ----------
    raw : object
        A single cell's raw value (``str``, ``None``, or ``NaN``).

    Returns
    -------
    pandas.Timestamp, optional
        The parsed timestamp, or ``None``.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _within_admission_guard(
    ts: Optional[pd.Timestamp], admission: Optional[pd.Timestamp]
) -> bool:
    """Whether ``ts`` is within :data:`GUARD_WINDOW` of ``admission``.

    Both missing/unparsable fails the guard (``False``) -- there is
    nothing to anchor to. See the module docstring and
    docs/gemini_extraction.md's open question 7 for why this exists:
    pharmacy/radiology timestamps include real, physically impossible
    years, data-entry artifacts rather than genuine far-future/far-past
    events.

    Parameters
    ----------
    ts : pandas.Timestamp, optional
        The timestamp being checked.
    admission : pandas.Timestamp, optional
        The owning encounter's admission time, from
        :func:`fetch_admission_index`.

    Returns
    -------
    bool
        ``True`` if ``ts`` should be trusted.
    """
    if ts is None or admission is None:
        return False
    return bool(abs(ts - admission) <= GUARD_WINDOW)


def _parse_numeric(raw: object) -> Optional[float]:
    """Parse a text lab/vitals value as a float, or ``None`` if it isn't one.

    Same numeric-parse-with-categorical-fallback shape already used for
    MIMIC-IV/eICU labs: a non-numeric result (e.g. a qualitative flag)
    still produces a MEDS row (the event and its ``code`` are real), just
    with a null ``numeric_value``.

    Parameters
    ----------
    raw : object
        A single cell's raw value.

    Returns
    -------
    float, optional
        The parsed value, or ``None``.
    """
    value = pd.to_numeric(raw, errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _paginate_rows(
    table: str, select_cols: list[str], *, batch_size: int = BATCH_ROWS
) -> Iterator[pd.DataFrame]:
    """Yield ``table`` in ``row_num``-ordered batches via keyset pagination.

    Every base table this module reads from carries its own ``row_num``
    column (confirmed for every matview in ``schema.json``) -- a stable,
    monotonic per-table cursor. Cheaper than ``OFFSET`` pagination at the
    hundreds-of-millions-of-rows scale some of these tables are at (``OFFSET``
    re-scans and discards every prior page; a ``WHERE row_num > :cursor``
    range filter does not).

    Parameters
    ----------
    table : str
        Table name, from this module's own hardcoded per-function calls
        (never external input).
    select_cols : list[str]
        Columns to select; ``row_num`` is always included even if not
        listed here.
    batch_size : int
        Rows per batch.

    Yields
    ------
    pandas.DataFrame
        One batch at a time, ordered by ``row_num`` ascending.
    """
    cursor = -1
    cols = list(dict.fromkeys([*select_cols, "row_num"]))
    cols_sql = ", ".join(_quote_ident(c) for c in cols)
    table_sql = _quote_ident(table)
    row_num_sql = _quote_ident("row_num")
    while True:
        sql = (
            f"SELECT {cols_sql} FROM {table_sql} "
            f"WHERE {row_num_sql} > {int(cursor)} "
            f"ORDER BY {row_num_sql} LIMIT {int(batch_size)}"
        )
        batch = db.query(sql)
        if batch.empty:
            return
        yield batch
        cursor = int(batch["row_num"].iloc[-1])


def fetch_admission_index() -> tuple[dict[int, str], dict[int, Optional[pd.Timestamp]]]:
    """One pass over ``admdad_subset``: ``genc_id -> (subject, admission time)``.

    Built once and held in memory for the whole extraction: every other
    table's rows carry only ``genc_id``, not ``patient_id_hashed`` (see
    docs/gemini_extraction.md's open question 6), and the pharmacy/
    radiology timestamp guard needs the encounter's admission time as its
    anchor. ~2.27M encounters in the schema-exploration cut -- two plain
    ``int``/``str``-keyed dicts, not a reason to re-query per row.

    Returns
    -------
    tuple[dict[int, str], dict[int, pandas.Timestamp | None]]
        ``(subject_by_genc, admission_by_genc)``.
    """
    subject_by_genc: dict[int, str] = {}
    admission_by_genc: dict[int, Optional[pd.Timestamp]] = {}
    for batch in _paginate_rows(
        "admdad_subset", ["genc_id", "patient_id_hashed", "admission_date_time"]
    ):
        for _, row in batch.iterrows():
            genc_id = int(row["genc_id"])
            subject_by_genc[genc_id] = str(row["patient_id_hashed"])
            admission_by_genc[genc_id] = _parse_gemini_datetime(
                row["admission_date_time"]
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
    so a caller joining against this dict's keys can't double-count.

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
    return {
        int(row["concept_id"]): str(row["concept_desc"]) for _, row in result.iterrows()
    }


def _meds_batch(
    subject_ids: list[Optional[str]],
    times: list[Optional[pd.Timestamp]],
    codes: list[Optional[str]],
    numeric_values: list[Optional[float]],
    hadm_ids: list[Optional[int]],
) -> pd.DataFrame:
    """Assemble one MEDS-shaped batch, dropping rows with no subject/time/code.

    A row missing any of subject, time, or code isn't a usable MEDS event
    at all -- dropped here rather than passed downstream as a row of
    nulls the writer would have to special-case.

    Parameters
    ----------
    subject_ids, times, codes, numeric_values, hadm_ids : list
        Parallel lists, one entry per candidate row.

    Returns
    -------
    pandas.DataFrame
        :data:`MEDS_COLUMNS`, one row per usable event.
    """
    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "time": times,
            "code": codes,
            "numeric_value": numeric_values,
            "hadm_id": hadm_ids,
        }
    )
    return frame.dropna(subset=["subject_id", "time", "code"]).reset_index(drop=True)


def extract_admissions(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[pd.DataFrame]:
    """Admission and discharge events from ``admdad_subset``.

    No timestamp guard applied -- ``admdad_subset``'s own timestamps land
    entirely in the plausible 2010-2024 range (confirmed in
    docs/gemini_extraction.md's date-range check), unlike pharmacy/radiology.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "admdad_subset", ["genc_id", "admission_date_time", "discharge_date_time"]
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            genc_id = int(row["genc_id"])
            subject = subject_by_genc.get(genc_id)
            admit = admission_by_genc.get(genc_id)
            discharge = _parse_gemini_datetime(row["discharge_date_time"])
            for ts, code in ((admit, "ADMISSION"), (discharge, "DISCHARGE")):
                subject_ids.append(subject)
                times.append(ts)
                codes.append(code)
                hadm_ids.append(genc_id)
        n = len(subject_ids)
        yield _meds_batch(subject_ids, times, codes, [None] * n, hadm_ids)


def extract_icu(
    subject_by_genc: dict[int, str],
) -> Iterator[pd.DataFrame]:
    """ICU admission/discharge events from ``ipscu_subset``.

    Only rows where ``icu_flag`` is true count as ICU specifically --
    ``scu_unit_number`` implies non-ICU special-care units exist too (see
    docs/gemini_extraction.md).

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "ipscu_subset",
        ["genc_id", "scu_admit_date_time", "scu_discharge_date_time", "icu_flag"],
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            if not bool(row["icu_flag"]):
                continue
            genc_id = int(row["genc_id"])
            subject = subject_by_genc.get(genc_id)
            admit = _parse_gemini_datetime(row["scu_admit_date_time"])
            discharge = _parse_gemini_datetime(row["scu_discharge_date_time"])
            for ts, code in ((admit, "ICU_ADMISSION"), (discharge, "ICU_DISCHARGE")):
                subject_ids.append(subject)
                times.append(ts)
                codes.append(code)
                hadm_ids.append(genc_id)
        n = len(subject_ids)
        yield _meds_batch(subject_ids, times, codes, [None] * n, hadm_ids)


def extract_labs(
    subject_by_genc: dict[int, str], lab_concepts: dict[int, str]
) -> Iterator[pd.DataFrame]:
    """Lab result events from ``lab_subset``.

    ``code`` is the raw OMOP concept id, namespaced (``LAB//<id>``) -- not
    yet translated to LOINC, see the module docstring. Rows whose
    ``test_type_mapped_omop`` has no real mapped concept at all (not in
    ``lab_concepts``, the deduplicated lookup from
    :func:`fetch_lab_concept_lookup`) are dropped, not extracted as
    garbage codes.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.
    lab_concepts : dict
        From :func:`fetch_lab_concept_lookup`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "lab_subset",
        ["genc_id", "test_type_mapped_omop", "result_value", "collection_date_time"],
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        numeric_values: list[Optional[float]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            concept_id = row["test_type_mapped_omop"]
            if pd.isna(concept_id) or int(concept_id) not in lab_concepts:
                continue
            genc_id = int(row["genc_id"])
            subject_ids.append(subject_by_genc.get(genc_id))
            times.append(_parse_gemini_datetime(row["collection_date_time"]))
            codes.append(f"LAB//{int(concept_id)}")
            numeric_values.append(_parse_numeric(row["result_value"]))
            hadm_ids.append(genc_id)
        yield _meds_batch(subject_ids, times, codes, numeric_values, hadm_ids)


def extract_vitals(subject_by_genc: dict[int, str]) -> Iterator[pd.DataFrame]:
    """Vital-sign events from ``vitals_subset``.

    ``code`` is the raw OMOP concept id, namespaced (``VITAL//<id>``); no
    dedup step needed here -- ``lookup_vitals_concept`` (unlike
    ``lookup_lab_concept``) doesn't exhibit the duplicate-row issue at the
    scale checked so far (see docs/gemini_extraction.md's open question 3).

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "vitals_subset",
        [
            "genc_id",
            "measurement_mapped_omop",
            "measurement_value",
            "measure_date_time",
        ],
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        numeric_values: list[Optional[float]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            concept_id = row["measurement_mapped_omop"]
            if pd.isna(concept_id):
                continue
            genc_id = int(row["genc_id"])
            subject_ids.append(subject_by_genc.get(genc_id))
            times.append(_parse_gemini_datetime(row["measure_date_time"]))
            codes.append(f"VITAL//{int(concept_id)}")
            numeric_values.append(_parse_numeric(row["measurement_value"]))
            hadm_ids.append(genc_id)
        yield _meds_batch(subject_ids, times, codes, numeric_values, hadm_ids)


def extract_pharmacy(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[pd.DataFrame]:
    """Medication start/end events from ``pharmacy_subset``.

    ``code`` uses ``med_id_generic_name_raw`` as the identity (the
    RxNorm/ingredient bridge is a later stage, see the module docstring
    and docs/gemini_extraction.md's open question 5). **The admission
    guard applies here**: real timestamps in this table include
    physically impossible years (1930-9022, 1840-8186) -- a start/end
    time outside +-1y of the encounter's admission is dropped, not
    extracted as a nonsense event.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "pharmacy_subset",
        [
            "genc_id",
            "med_id_generic_name_raw",
            "med_start_date_time",
            "med_end_date_time",
        ],
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            name = row["med_id_generic_name_raw"]
            if pd.isna(name) or not str(name).strip():
                continue
            genc_id = int(row["genc_id"])
            admit = admission_by_genc.get(genc_id)
            started = _parse_gemini_datetime(row["med_start_date_time"])
            ended = _parse_gemini_datetime(row["med_end_date_time"])
            for ts, suffix in ((started, "started"), (ended, "ended")):
                if not _within_admission_guard(ts, admit):
                    continue
                subject_ids.append(subject_by_genc.get(genc_id))
                times.append(ts)
                codes.append(f"MEDICATION//{name}//{suffix}")
                hadm_ids.append(genc_id)
        n = len(subject_ids)
        yield _meds_batch(subject_ids, times, codes, [None] * n, hadm_ids)


def extract_diagnoses(subject_by_genc: dict[int, str]) -> Iterator[pd.DataFrame]:
    """Diagnosis events from ``ipdiagnosis_subset``.

    ``code`` is the raw ICD-10-CA code, namespaced (``DIAGNOSIS//<code>``)
    -- no event-level timestamp exists on this table (diagnoses are coded
    at the encounter level), so ``time`` is the encounter's discharge time
    (a diagnosis is a fact about the whole stay, attributed at its close,
    the same convention MIMIC-IV/eICU discharge diagnoses already use).

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    discharge_by_genc: dict[int, Optional[pd.Timestamp]] = {}
    for batch in _paginate_rows("admdad_subset", ["genc_id", "discharge_date_time"]):
        for _, row in batch.iterrows():
            discharge_by_genc[int(row["genc_id"])] = _parse_gemini_datetime(
                row["discharge_date_time"]
            )
    for batch in _paginate_rows("ipdiagnosis_subset", ["genc_id", "diagnosis_code"]):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            code = row["diagnosis_code"]
            if pd.isna(code) or not str(code).strip():
                continue
            genc_id = int(row["genc_id"])
            subject_ids.append(subject_by_genc.get(genc_id))
            times.append(discharge_by_genc.get(genc_id))
            codes.append(f"DIAGNOSIS//{code}")
            hadm_ids.append(genc_id)
        n = len(subject_ids)
        yield _meds_batch(subject_ids, times, codes, [None] * n, hadm_ids)


def extract_procedures(subject_by_genc: dict[int, str]) -> Iterator[pd.DataFrame]:
    """Procedure events from ``ipintervention_subset``.

    ``code`` is the raw CCI code, namespaced (``PROCEDURE//<code>``), timed
    at ``intervention_episode_start_date_time``.

    Parameters
    ----------
    subject_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "ipintervention_subset",
        ["genc_id", "intervention_code", "intervention_episode_start_date_time"],
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            code = row["intervention_code"]
            if pd.isna(code) or not str(code).strip():
                continue
            genc_id = int(row["genc_id"])
            subject_ids.append(subject_by_genc.get(genc_id))
            times.append(
                _parse_gemini_datetime(row["intervention_episode_start_date_time"])
            )
            codes.append(f"PROCEDURE//{code}")
            hadm_ids.append(genc_id)
        n = len(subject_ids)
        yield _meds_batch(subject_ids, times, codes, [None] * n, hadm_ids)


def extract_radiology(
    subject_by_genc: dict[int, str],
    admission_by_genc: dict[int, Optional[pd.Timestamp]],
) -> Iterator[pd.DataFrame]:
    """Imaging events from ``radiology_subset``.

    ``code`` combines modality and body part (``IMAGING//<modality>//
    <body_part>``), timed at ``performed_date_time``. **The admission
    guard applies here too** -- ``performed_date_time`` includes years up
    to 9999 (see docs/gemini_extraction.md's open question 7), same fix as
    pharmacy.

    Parameters
    ----------
    subject_by_genc, admission_by_genc : dict
        From :func:`fetch_admission_index`.

    Yields
    ------
    pandas.DataFrame
        :data:`MEDS_COLUMNS` batches.
    """
    for batch in _paginate_rows(
        "radiology_subset",
        ["genc_id", "modality_mapped", "body_part_mapped", "performed_date_time"],
    ):
        subject_ids: list[Optional[str]] = []
        times: list[Optional[pd.Timestamp]] = []
        codes: list[Optional[str]] = []
        hadm_ids: list[Optional[int]] = []
        for _, row in batch.iterrows():
            genc_id = int(row["genc_id"])
            admit = admission_by_genc.get(genc_id)
            ts = _parse_gemini_datetime(row["performed_date_time"])
            if not _within_admission_guard(ts, admit):
                continue
            modality = (
                row["modality_mapped"]
                if not pd.isna(row["modality_mapped"])
                else "UNKNOWN"
            )
            body_part = (
                row["body_part_mapped"]
                if not pd.isna(row["body_part_mapped"])
                else "UNKNOWN"
            )
            subject_ids.append(subject_by_genc.get(genc_id))
            times.append(ts)
            codes.append(f"IMAGING//{modality}//{body_part}")
            hadm_ids.append(genc_id)
        n = len(subject_ids)
        yield _meds_batch(subject_ids, times, codes, [None] * n, hadm_ids)


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
    n_shards = max(1, -(-len(unique_subjects) // subjects_per_shard))
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


def run_extraction(output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Run the full GEMINI -> MEDS extraction and write the suppressed summary.

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
    table_generators: list[tuple[str, Iterator[pd.DataFrame]]] = [
        ("admdad_subset", extract_admissions(subject_by_genc, admission_by_genc)),
        ("ipscu_subset", extract_icu(subject_by_genc)),
        ("lab_subset", extract_labs(subject_by_genc, lab_concepts)),
        ("vitals_subset", extract_vitals(subject_by_genc)),
        ("pharmacy_subset", extract_pharmacy(subject_by_genc, admission_by_genc)),
        ("ipdiagnosis_subset", extract_diagnoses(subject_by_genc)),
        ("ipintervention_subset", extract_procedures(subject_by_genc)),
        ("radiology_subset", extract_radiology(subject_by_genc, admission_by_genc)),
    ]
    for table_name, generator in table_generators:
        logger.info("[extract_meds] extracting %s...", table_name)
        for batch in generator:
            writer.write_batch(table_name, batch)

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
