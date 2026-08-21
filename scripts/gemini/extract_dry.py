#!/usr/bin/env python
"""Pre-extraction sanity pass: row counts, null fractions, and design queries.

Driven mostly by ``scripts/gemini/out/schema.json`` (see ``explore_schema.py``)
rather than a hardcoded table list -- every base data object the schema step
found (``kind in ("table", "matview")``; GEMINI's real schema uses only
``matview`` for its base clinical tables, confirmed from the first real
schema report, ``schema.json``'s ``datacut: subdural_hematoma_v1_0_0`` run)
gets a row-count and per-column null-fraction check. A handful of
:func:`design_queries` are hardcoded against specific, verified column names
instead, because they answer specific extraction-design questions (which
concept lookup a code maps through, whether a datetime column exists, what
years a table actually covers) that no generic per-table loop can answer.

Prints ``pending schema report`` and exits cleanly if ``schema.json`` doesn't
exist yet, so this is safe to wire into ``run.sh`` before the real GEMINI
database has ever been reached.

Same governance discipline as ``explore_schema.py``: only suppressed/rounded
counts (:func:`_suppressed`) ever get written or printed, no patient-level
rows are read. The largest event-level matviews (``lab_subset`` at ~659M
rows, ``vitals_subset`` at ~412M, ``pharmacy_subset`` at ~84M,
``ipdiagnosis_subset`` at ~14M, ``radiology_subset`` at ~9M -- see
:data:`LARGE_TABLE_ROW_THRESHOLD`) skip the per-column null-fraction loop and
the fresh row-count re-verification: a full per-column ``COUNT(*) -
COUNT(column)`` scan across hundreds of millions of rows, repeated per
column, is a real cost on a single shared node and not what this step is
for. Their row counts are carried over from ``schema.json``'s own count
instead (already paid for once by the ``schema`` step).

Run on the GEMINI node, after ``schema``:

    uv run python scripts/gemini/extract_dry.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from odyssey.data.gemini import db


logger = logging.getLogger(__name__)


OUT_DIR = Path(__file__).resolve().parent / "out"
SCHEMA_PATH = OUT_DIR / "schema.json"

#: Matviews at or above this row count skip the per-column null-fraction
#: loop and the fresh row-count re-query; see the module docstring.
LARGE_TABLE_ROW_THRESHOLD = 5_000_000

_NULL_COUNT_SQL = "SELECT COUNT(*) - COUNT({column}) AS n_null FROM {table}"
_ROW_COUNT_SQL = "SELECT COUNT(*) AS n FROM {table}"

#: Which text datetime-ish column(s) to pull a year range from, per matview
#: that has one -- hand-curated from the real schema report (column names
#: vary per table, e.g. ``admission_date_time`` vs ``collection_date_time``,
#: and several tables have more than one candidate; every value here is a
#: column this project has actually seen in ``schema.json``, not a guess).
#: The exact text format (e.g. whether every row is ``YYYY-MM-DD...``) is
#: NOT yet confirmed -- the query extracts a leading 4-digit year via regex
#: and ignores rows that don't match, rather than assuming a fixed format.
DATE_COLUMNS_BY_TABLE: dict[str, list[str]] = {
    "admdad_subset": ["admission_date_time", "discharge_date_time"],
    "er_subset": ["triage_date_time", "disposition_date_time"],
    "ipscu_subset": ["scu_admit_date_time", "scu_discharge_date_time"],
    "lab_subset": ["collection_date_time"],
    "vitals_subset": ["measure_date_time"],
    "pharmacy_subset": ["med_start_date_time", "med_end_date_time"],
    "radiology_subset": ["ordered_date_time", "performed_date_time"],
}

#: Lookup tables schema.json's rounded row count showed as ``"0"`` --
#: ambiguous under nearest-1000 rounding (a real count anywhere from 0 to
#: ~500 also rounds to ``"0"``), so :func:`lookup_emptiness` checks each
#: with a real ``EXISTS`` query instead of trusting the rounded count.
SUSPECT_EMPTY_LOOKUPS = [
    "lookup_hospital",
    "lookup_pharmacy_route",
    "lookup_transfusion_concept",
    "lookup_vitals_concept",
]

_YEAR_RANGE_SQL = """
SELECT MIN(y) AS min_year, MAX(y) AS max_year FROM (
    SELECT NULLIF(SUBSTRING({column} FROM '^\\d{{4}}'), '')::int AS y
    FROM {table}
) sub
"""

_EXISTS_SQL = "SELECT EXISTS (SELECT 1 FROM {table}) AS any_rows"

_CONCEPT_FREQUENCY_SQL = """
SELECT e.{code_col}::text AS code, l.concept_desc, COUNT(*) AS n
FROM {table} e
LEFT JOIN {lookup} l ON l.concept_id = e.{code_col}::text
WHERE e.{code_col} IS NOT NULL
GROUP BY e.{code_col}, l.concept_desc
ORDER BY n DESC
LIMIT 200
"""

_YEAR_COUNT_SQL = """
SELECT NULLIF(SUBSTRING({column} FROM '^\\d{{4}}'), '')::int AS year, COUNT(*) AS n
FROM {table}
GROUP BY year
ORDER BY year
"""

#: Top ~30 lab concepts by frequency (this project's first real
#: extract-dry run against lab_subset.test_type_mapped_omop), plus every
#: lab concept already in odyssey/data/value_binning.py's
#: CANONICAL_CLINICAL_RANGES that wasn't already in that top 30 (lactate's
#: three concept ids ranked far lower by raw frequency but matter just as
#: much for the units question -- see docs/gemini_extraction.md's Units
#: section). Hand-curated from real report data, not a guess.
TOP_LAB_CONCEPTS = [
    3019550,  # Sodium
    3023103,  # Potassium
    3014576,  # Chloride
    3002385,  # Erythrocyte distribution width
    3000963,  # Hemoglobin
    3010813,  # Leukocytes
    3040151,  # Glucose (capillary)
    3009542,  # Hematocrit
    3007461,  # Platelets
    3020564,  # Creatinine -- existing canonical range assumes mg/dL, GEMINI is SI
    3024731,  # MCV
    3026361,  # Erythrocytes
    3045716,  # Anion gap
    3016293,  # Bicarbonate
    3019198,  # Lymphocytes
    3017732,  # Neutrophils
    3003338,  # MCHC
    3001604,  # Monocytes
    3006315,  # Basophils
    3035941,  # MCH
    3013115,  # Eosinophils
    3001123,  # Platelet mean volume
    3040168,  # Immature granulocytes
    40771922,  # eGFR
    3001490,  # Nucleated erythrocytes
    3013826,  # Glucose (serum/plasma)
    3024641,  # Urea nitrogen
    3032080,  # INR
    3006140,  # Bilirubin.total
    # Not in the top 30 by frequency, but already in
    # CANONICAL_CLINICAL_RANGES -- unit confirmation matters regardless of
    # frequency rank.
    3018405,  # Lactate (arterial)
    3008037,  # Lactate (venous)
    3020138,  # Lactate (ambiguous Moles/Mass join, see docs/gemini_extraction.md's open question 4)
]

#: All 13 vitals concepts observed in vitals_subset.measurement_mapped_omop
#: -- no "top N" cut needed, this is the entire vocabulary seen so far.
TOP_VITALS_CONCEPTS = [
    3013502,  # Oxygen saturation in Blood
    3027018,  # Heart rate
    3024171,  # Respiratory rate
    3020891,  # Body temperature
    36203185,  # Blood pressure panel with all children optional
    3004249,  # Systolic blood pressure
    3012888,  # Diastolic blood pressure
    3034263,  # Pain severity - Reported
    3014080,  # Oxygen gas flow Oxygen delivery system
    3005629,  # Inhaled oxygen flow rate
    3020716,  # Inhaled oxygen concentration
    3025315,  # Body weight
    4326744,  # Blood pressure
]

_UNIT_SAMPLE_SQL = """
SELECT {code_col} AS code, {unit_col} AS unit, COUNT(*) AS n
FROM {table}
WHERE {code_col} IN {codes}
GROUP BY {code_col}, {unit_col}
ORDER BY {code_col}, n DESC
"""


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded double quotes.

    Postgres lowercases unquoted identifiers, so an unquoted mixed-case
    column (real ones exist in GEMINI's schema, e.g. ``Pop2021``,
    ``households_dwellings_DA21`` on ``lookup_statcan_v2021``) resolves to
    a different, usually nonexistent, all-lowercase name and raises
    ``UndefinedColumn`` -- every table/column name this module interpolates
    into SQL (all sourced from ``schema.json``, i.e. the database's own
    catalog, never external input) goes through this first.

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


def _int_list_sql(values: list[int]) -> str:
    """Render a list of ints as a safe SQL literal list, e.g. ``"(1, 2, 3)"``.

    Only ever called with this module's own hardcoded concept-id constants
    (:data:`TOP_LAB_CONCEPTS`/:data:`TOP_VITALS_CONCEPTS`, never external
    input) -- the ``int()`` call is defensive, not a security boundary in
    itself.

    Parameters
    ----------
    values : list[int]
        Concept ids.

    Returns
    -------
    str
        ``"(v1, v2, ...)"``, safe to interpolate directly into SQL.
    """
    return "(" + ", ".join(str(int(v)) for v in values) + ")"


def _suppressed(n: int) -> str:
    """Round ``n`` to the nearest 1000, or mask small counts.

    Mirrors :func:`explore_schema.suppressed_row_count` exactly (kept as a
    separate copy rather than an import so this script has no dependency on
    ``explore_schema.py`` beyond the ``schema.json`` file it produces).

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


def null_fraction(table: str, column: str) -> str:
    """Suppressed count of null values in one column of one table.

    Parameters
    ----------
    table : str
        Table name, as found in ``schema.json`` (i.e. already validated
        against the database's own catalog by the ``schema`` step -- never
        called with external input).
    column : str
        Column name, same provenance as ``table``.

    Returns
    -------
    str
        A suppressed count (see :func:`_suppressed`).
    """
    result = db.query(
        _NULL_COUNT_SQL.format(column=_quote_ident(column), table=_quote_ident(table))
    )
    return _suppressed(int(result["n_null"].iloc[0]))


def fresh_row_count(table: str) -> str:
    """Suppressed, freshly-queried row count for one table.

    Re-queries rather than trusting ``schema.json``'s own count, since the
    schema step and this one can run on different days.

    Parameters
    ----------
    table : str
        Table name, same provenance note as :func:`null_fraction`.

    Returns
    -------
    str
        A suppressed count (see :func:`_suppressed`).
    """
    result = db.query(_ROW_COUNT_SQL.format(table=_quote_ident(table)))
    return _suppressed(int(result["n"].iloc[0]))


def _null_fraction_or_error(table: str, column: str) -> str:
    """Suppressed null count for one column, or an error marker if the query fails.

    One pathological column (a type Postgres can't ``COUNT`` cleanly, an
    unexpected encoding, ...) must not kill the whole run -- Amrit cannot
    iterate interactively on the GEMINI node (see ``docs/gemini.md``), so a
    single bad column becomes a note in the report instead of a crash.

    Parameters
    ----------
    table : str
        Table name.
    column : str
        Column name.

    Returns
    -------
    str
        A suppressed count (see :func:`_suppressed`), or ``"error: ..."``.
    """
    try:
        return null_fraction(table, column)
    except Exception as exc:
        logger.warning("null_fraction(%s, %s) failed: %s", table, column, exc)
        return f"error: {exc}"


def build_report(schema: dict[str, Any]) -> dict[str, Any]:
    """Run row-count and null-fraction sanity queries for every base object.

    Parameters
    ----------
    schema : dict[str, Any]
        A report from ``explore_schema.build_schema_report`` (or matching
        shape), loaded from ``schema.json``.

    Returns
    -------
    dict[str, Any]
        ``{"datacut": ..., "tables": [{"name", "row_count",
        "columns": [{"name", "n_null"}, ...] | None}, ...]}``, base data
        objects only (``kind in ("table", "matview")``), all counts
        suppressed. ``columns`` is ``None`` for objects at or above
        :data:`LARGE_TABLE_ROW_THRESHOLD` -- see the module docstring.
    """
    tables = []
    for obj in schema["objects"]:
        if obj["kind"] not in ("table", "matview"):
            continue
        name = obj["name"]
        # schema.json's own count is already suppressed/rounded; parse the
        # numeric part back out just to compare against the threshold (an
        # already-suppressed "<6" or "12000" both parse as ints fine here).
        schema_count = obj["row_count"].replace("<", "")
        if int(schema_count) >= LARGE_TABLE_ROW_THRESHOLD:
            tables.append(
                {"name": name, "row_count": obj["row_count"], "columns": None}
            )
            continue
        tables.append(
            {
                "name": name,
                "row_count": fresh_row_count(name),
                "columns": [
                    {
                        "name": col["name"],
                        "n_null": _null_fraction_or_error(name, col["name"]),
                    }
                    for col in obj["columns"]
                ],
            }
        )
    return {"datacut": schema["datacut"], "tables": tables}


def concept_frequencies(table: str, code_col: str, lookup: str) -> list[dict[str, Any]]:
    """Top ~200 distinct codes in one OMOP-mapped column, by rounded frequency.

    Parameters
    ----------
    table : str
        Event table name (``lab_subset`` or ``vitals_subset``).
    code_col : str
        The OMOP-mapped code column (``test_type_mapped_omop`` or
        ``measurement_mapped_omop``).
    lookup : str
        The concept lookup table to join against
        (``lookup_lab_concept``/``lookup_vitals_concept``), matched on its
        ``concept_id`` column. If the lookup is empty, ``concept_desc`` is
        ``None`` for every row -- not an error, see :func:`lookup_emptiness`.

    Returns
    -------
    list[dict[str, Any]]
        ``[{"code", "concept_desc", "n"}, ...]``, ``n`` suppressed.
    """
    result = db.query(
        _CONCEPT_FREQUENCY_SQL.format(
            table=_quote_ident(table),
            code_col=_quote_ident(code_col),
            lookup=_quote_ident(lookup),
        )
    )
    return [
        {
            "code": row["code"],
            "concept_desc": row["concept_desc"],
            "n": _suppressed(int(row["n"])),
        }
        for _, row in result.iterrows()
    ]


def table_date_ranges() -> dict[str, dict[str, Any]]:
    """Year-only min/max date range for every table in :data:`DATE_COLUMNS_BY_TABLE`.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{table: {column: {"min_year", "max_year"}}}``. Years are never
        patient-identifying on their own, so not suppressed the way row
        counts are.
    """
    ranges: dict[str, dict[str, Any]] = {}
    for table, columns in DATE_COLUMNS_BY_TABLE.items():
        ranges[table] = {}
        for column in columns:
            result = db.query(
                _YEAR_RANGE_SQL.format(
                    column=_quote_ident(column), table=_quote_ident(table)
                )
            )
            row = result.iloc[0]
            ranges[table][column] = {
                "min_year": (
                    None if pd.isna(row["min_year"]) else int(row["min_year"])
                ),
                "max_year": (
                    None if pd.isna(row["max_year"]) else int(row["max_year"])
                ),
            }
    return ranges


def hospital_coverage() -> list[dict[str, Any]]:
    """Dump ``lookup_data_coverage`` directly: per-hospital, per-data-type coverage.

    Returns
    -------
    list[dict[str, Any]]
        One dict per row (``data``, ``min_date``, ``max_date``,
        ``hospital_num``, ``additional_info``) -- already a small,
        aggregate lookup table (~1000 rounded rows in the schema report),
        not raw patient rows, so not suppressed further.
    """
    result = db.query(f"SELECT * FROM {_quote_ident('lookup_data_coverage')}")
    return [
        {k: (None if pd.isna(v) else str(v)) for k, v in row.items()}
        for _, row in result.iterrows()
    ]


def encounters_per_year() -> dict[str, str]:
    """Encounter count per admission year, suppressed and rounded to 1000.

    Returns
    -------
    dict[str, str]
        ``{year_or_"unknown": suppressed_count}``.
    """
    result = db.query(
        _YEAR_COUNT_SQL.format(
            column=_quote_ident("admission_date_time"),
            table=_quote_ident("admdad_subset"),
        )
    )
    out = {}
    for _, row in result.iterrows():
        key = "unknown" if pd.isna(row["year"]) else str(int(row["year"]))
        out[key] = _suppressed(int(row["n"]))
    return out


def lookup_emptiness() -> dict[str, bool]:
    """Whether each table in :data:`SUSPECT_EMPTY_LOOKUPS` genuinely has zero rows.

    schema.json's rounded row count showed ``"0"`` for these, which is
    ambiguous (nearest-1000 rounding maps anything from 0 to ~500 rows to
    the same ``"0"``) -- this uses ``EXISTS`` instead, which reveals only a
    boolean, never a count, so it stays governance-safe even for a table
    that turns out to have a handful of rows.

    Returns
    -------
    dict[str, bool]
        ``{table: is_genuinely_empty}``.
    """
    out = {}
    for table in SUSPECT_EMPTY_LOOKUPS:
        result = db.query(_EXISTS_SQL.format(table=_quote_ident(table)))
        out[table] = not bool(result["any_rows"].iloc[0])
    return out


def unit_samples(
    table: str, code_col: str, unit_col: str, codes: list[int]
) -> dict[str, list[dict[str, Any]]]:
    """Suppressed value counts of the raw unit string, per concept code.

    Answers docs/gemini_extraction.md's Units section's open gap: concept
    descriptions distinguish Mass/volume from Moles/volume but not the
    literal unit (``umol/L`` vs ``mmol/L``, etc.) -- this samples
    ``result_unit``/``measurement_unit`` directly for exactly the concepts
    that matter (every LOINC already in
    ``odyssey/data/value_binning.py``'s canonical clinical ranges, plus
    every observed vitals concept).

    Parameters
    ----------
    table : str
        ``lab_subset`` or ``vitals_subset``.
    code_col : str
        The OMOP-mapped code column.
    unit_col : str
        The raw unit column (``result_unit``/``measurement_unit``).
    codes : list[int]
        Concept ids to check -- this module's own hardcoded
        :data:`TOP_LAB_CONCEPTS`/:data:`TOP_VITALS_CONCEPTS`.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        ``{str(code): [{"unit", "n"}, ...]}``, ``n`` suppressed.
    """
    result = db.query(
        _UNIT_SAMPLE_SQL.format(
            code_col=_quote_ident(code_col),
            unit_col=_quote_ident(unit_col),
            table=_quote_ident(table),
            codes=_int_list_sql(codes),
        )
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for _, row in result.iterrows():
        key = str(int(row["code"]))
        out.setdefault(key, []).append(
            {
                "unit": None if pd.isna(row["unit"]) else str(row["unit"]),
                "n": _suppressed(int(row["n"])),
            }
        )
    return out


def _safe_query(name: str, fn: Any) -> Any:
    """Run one design-critical query, or record why it failed.

    One query failing (a column that doesn't exist the way expected, an
    unexpected type, ...) must not cost the whole `extract-dry` round trip
    -- Amrit cannot iterate interactively on the GEMINI node. Mirrors
    :func:`_null_fraction_or_error`'s per-column recovery, generalized to
    every design query.

    Parameters
    ----------
    name : str
        Query name, for the warning log only.
    fn : Callable[[], Any]
        The query to run, with no arguments (a ``functools.partial`` or
        lambda).

    Returns
    -------
    Any
        The query's result, or ``{"error": "..."}`` if it raised.
    """
    try:
        return fn()
    except Exception as exc:
        logger.warning("design query %s failed: %s", name, exc)
        return {"error": str(exc)}


def design_queries() -> dict[str, Any]:
    """Run every design-critical query needed before writing the MEDS extraction spec.

    See ``docs/gemini_extraction.md`` for what each answer decides. Each
    query is independently recovered from a failure (:func:`_safe_query`)
    so one bad query doesn't cost the whole report.

    Returns
    -------
    dict[str, Any]
        ``{"lab_concept_frequencies", "vitals_concept_frequencies",
        "table_date_ranges", "hospital_coverage", "encounters_per_year",
        "lookup_emptiness", "lab_unit_samples", "vitals_unit_samples"}``.
    """
    return {
        "lab_concept_frequencies": _safe_query(
            "lab_concept_frequencies",
            lambda: concept_frequencies(
                "lab_subset", "test_type_mapped_omop", "lookup_lab_concept"
            ),
        ),
        "vitals_concept_frequencies": _safe_query(
            "vitals_concept_frequencies",
            lambda: concept_frequencies(
                "vitals_subset", "measurement_mapped_omop", "lookup_vitals_concept"
            ),
        ),
        "table_date_ranges": _safe_query("table_date_ranges", table_date_ranges),
        "hospital_coverage": _safe_query("hospital_coverage", hospital_coverage),
        "encounters_per_year": _safe_query("encounters_per_year", encounters_per_year),
        "lookup_emptiness": _safe_query("lookup_emptiness", lookup_emptiness),
        "lab_unit_samples": _safe_query(
            "lab_unit_samples",
            lambda: unit_samples(
                "lab_subset", "test_type_mapped_omop", "result_unit", TOP_LAB_CONCEPTS
            ),
        ),
        "vitals_unit_samples": _safe_query(
            "vitals_unit_samples",
            lambda: unit_samples(
                "vitals_subset",
                "measurement_mapped_omop",
                "measurement_unit",
                TOP_VITALS_CONCEPTS,
            ),
        ),
    }


def _render_tables_section(tables: list[dict[str, Any]]) -> list[str]:
    """Render the per-object row-count/null-fraction section.

    Parameters
    ----------
    tables : list[dict[str, Any]]
        ``report["tables"]`` from :func:`build_report`.

    Returns
    -------
    list[str]
        Markdown lines.
    """
    lines = ["## Per-object row counts and null fractions", ""]
    for table in tables:
        lines.append(f"### `{table['name']}` (rows: {table['row_count']})")
        lines.append("")
        if table["columns"] is None:
            lines.append(
                "*Skipped per-column null-fraction check -- too large "
                f"(>= {LARGE_TABLE_ROW_THRESHOLD:,} rows).*"
            )
        else:
            lines.append("| column | null count |")
            lines.append("| --- | --- |")
            for col in table["columns"]:
                lines.append(f"| {col['name']} | {col['n_null']} |")
        lines.append("")
    return lines


def _render_section_or_error(heading: str, value: Any, render_fn: Any) -> list[str]:
    """Render one design-query result, or a short error note if it failed.

    :func:`_safe_query` substitutes ``{"error": "..."}`` for a query that
    raised -- rendered here as a visible note rather than crashing the
    whole markdown render on a shape ``render_fn`` doesn't expect.

    Parameters
    ----------
    heading : str
        Markdown ``###`` heading text.
    value : Any
        The design query's result (or an error dict).
    render_fn : Callable[[Any], list[str]]
        Renders the successful-result shape into markdown lines (heading
        and trailing blank line not included).

    Returns
    -------
    list[str]
        Markdown lines, heading included.
    """
    lines = [f"### {heading}", ""]
    if isinstance(value, dict) and "error" in value:
        lines.append(f"*Query failed: {value['error']}*")
    else:
        lines += render_fn(value)
    lines.append("")
    return lines


def _render_concept_frequency_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render one code/concept_desc/n Markdown table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows from :func:`concept_frequencies`.

    Returns
    -------
    list[str]
        Markdown lines.
    """
    lines = ["| code | concept_desc | n |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['code']} | {row['concept_desc']} | {row['n']} |")
    return lines


def _render_unit_samples_table(samples: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Render one code/unit/n Markdown table from :func:`unit_samples`.

    Parameters
    ----------
    samples : dict[str, list[dict[str, Any]]]
        ``{code: [{"unit", "n"}, ...]}`` from :func:`unit_samples`.

    Returns
    -------
    list[str]
        Markdown lines.
    """
    lines = ["| code | unit | n |", "| --- | --- | --- |"]
    for code, rows in samples.items():
        for row in rows:
            lines.append(f"| {code} | {row['unit']} | {row['n']} |")
    return lines


def _render_design_queries_section(dq: dict[str, Any]) -> list[str]:
    """Render the design-critical-queries section.

    Parameters
    ----------
    dq : dict[str, Any]
        ``report["design_queries"]`` from :func:`design_queries`.

    Returns
    -------
    list[str]
        Markdown lines.
    """
    lines = ["## Design-critical queries", ""]
    lines += _render_section_or_error(
        "Lab concept frequencies (`lab_subset.test_type_mapped_omop`)",
        dq["lab_concept_frequencies"],
        _render_concept_frequency_table,
    )
    lines += _render_section_or_error(
        "Vitals concept frequencies (`vitals_subset.measurement_mapped_omop`)",
        dq["vitals_concept_frequencies"],
        _render_concept_frequency_table,
    )
    lines += _render_section_or_error(
        "Lab unit samples, top ~30 concepts (`lab_subset.result_unit`)",
        dq.get("lab_unit_samples", {}),
        _render_unit_samples_table,
    )
    lines += _render_section_or_error(
        "Vitals unit samples, all 13 concepts (`vitals_subset.measurement_unit`)",
        dq.get("vitals_unit_samples", {}),
        _render_unit_samples_table,
    )
    lines += _render_section_or_error(
        "Table date ranges (year only)",
        dq["table_date_ranges"],
        _render_date_ranges_table,
    )
    lines += _render_section_or_error(
        "Per-hospital data coverage (`lookup_data_coverage`)",
        dq["hospital_coverage"],
        _render_hospital_coverage_table,
    )
    lines += _render_section_or_error(
        "Encounters per year (`admdad_subset`)",
        dq["encounters_per_year"],
        _render_encounters_per_year_table,
    )
    lines += _render_section_or_error(
        "Lookup tables confirmed genuinely empty (real EXISTS check)",
        dq["lookup_emptiness"],
        _render_lookup_emptiness_table,
    )
    return lines


def _render_date_ranges_table(ranges: dict[str, dict[str, Any]]) -> list[str]:
    """Render :func:`table_date_ranges`'s result as a Markdown table."""
    lines = ["| table | column | min year | max year |", "| --- | --- | --- | --- |"]
    for table, columns in ranges.items():
        for column, rng in columns.items():
            lines.append(
                f"| {table} | {column} | {rng['min_year']} | {rng['max_year']} |"
            )
    return lines


def _render_hospital_coverage_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render :func:`hospital_coverage`'s result as a Markdown table."""
    lines = [
        "| data | min_date | max_date | hospital_num | additional_info |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('data')} | {row.get('min_date')} | {row.get('max_date')} "
            f"| {row.get('hospital_num')} | {row.get('additional_info')} |"
        )
    return lines


def _render_encounters_per_year_table(counts: dict[str, str]) -> list[str]:
    """Render :func:`encounters_per_year`'s result as a Markdown table."""
    lines = ["| year | count |", "| --- | --- |"]
    for year, n in counts.items():
        lines.append(f"| {year} | {n} |")
    return lines


def _render_lookup_emptiness_table(results: dict[str, bool]) -> list[str]:
    """Render :func:`lookup_emptiness`'s result as a Markdown table."""
    lines = ["| table | genuinely empty |", "| --- | --- |"]
    for table, is_empty in results.items():
        lines.append(f"| {table} | {is_empty} |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render an extract-dry report as human-readable Markdown.

    Parameters
    ----------
    report : dict[str, Any]
        A report from :func:`build_report`, with ``design_queries`` added
        under that key (or matching shape).

    Returns
    -------
    str
        Markdown text.
    """
    lines = [f"# GEMINI extract-dry report: `{report['datacut']}`", ""]
    lines += _render_tables_section(report["tables"])
    dq = report.get("design_queries")
    if dq:
        lines += _render_design_queries_section(dq)
    return "\n".join(lines)


def main() -> None:
    """Run the extract-dry sanity pass, or report that schema.json is missing."""
    if not SCHEMA_PATH.exists():
        print(f"pending schema report ({SCHEMA_PATH} does not exist yet)")
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    report = build_report(schema)
    report["design_queries"] = design_queries()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "extract_dry.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "extract_dry.md").write_text(render_markdown(report))
    print(f"Wrote {OUT_DIR / 'extract_dry.json'} and {OUT_DIR / 'extract_dry.md'}")


if __name__ == "__main__":
    main()
