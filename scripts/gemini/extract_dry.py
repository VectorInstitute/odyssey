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
from pathlib import Path
from typing import Any

import pandas as pd

from odyssey.data.gemini import db


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
    result = db.query(_NULL_COUNT_SQL.format(column=column, table=table))
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
    result = db.query(_ROW_COUNT_SQL.format(table=table))
    return _suppressed(int(result["n"].iloc[0]))


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
                    {"name": col["name"], "n_null": null_fraction(name, col["name"])}
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
        _CONCEPT_FREQUENCY_SQL.format(table=table, code_col=code_col, lookup=lookup)
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
            result = db.query(_YEAR_RANGE_SQL.format(column=column, table=table))
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
    result = db.query("SELECT * FROM lookup_data_coverage")
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
        _YEAR_COUNT_SQL.format(column="admission_date_time", table="admdad_subset")
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
        result = db.query(_EXISTS_SQL.format(table=table))
        out[table] = not bool(result["any_rows"].iloc[0])
    return out


def design_queries() -> dict[str, Any]:
    """Run every design-critical query needed before writing the MEDS extraction spec.

    See ``docs/gemini_extraction.md`` for what each answer decides.

    Returns
    -------
    dict[str, Any]
        ``{"lab_concept_frequencies", "vitals_concept_frequencies",
        "table_date_ranges", "hospital_coverage", "encounters_per_year",
        "lookup_emptiness"}``.
    """
    return {
        "lab_concept_frequencies": concept_frequencies(
            "lab_subset", "test_type_mapped_omop", "lookup_lab_concept"
        ),
        "vitals_concept_frequencies": concept_frequencies(
            "vitals_subset", "measurement_mapped_omop", "lookup_vitals_concept"
        ),
        "table_date_ranges": table_date_ranges(),
        "hospital_coverage": hospital_coverage(),
        "encounters_per_year": encounters_per_year(),
        "lookup_emptiness": lookup_emptiness(),
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


def _render_concept_frequency_table(
    heading: str, rows: list[dict[str, Any]]
) -> list[str]:
    """Render one code/concept_desc/n Markdown table under ``heading``.

    Parameters
    ----------
    heading : str
        Markdown ``###`` heading text.
    rows : list[dict[str, Any]]
        Rows from :func:`concept_frequencies`.

    Returns
    -------
    list[str]
        Markdown lines.
    """
    lines = [f"### {heading}", "", "| code | concept_desc | n |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['code']} | {row['concept_desc']} | {row['n']} |")
    lines.append("")
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
    lines += _render_concept_frequency_table(
        "Lab concept frequencies (`lab_subset.test_type_mapped_omop`)",
        dq["lab_concept_frequencies"],
    )
    lines += _render_concept_frequency_table(
        "Vitals concept frequencies (`vitals_subset.measurement_mapped_omop`)",
        dq["vitals_concept_frequencies"],
    )

    lines += [
        "### Table date ranges (year only)",
        "",
        "| table | column | min year | max year |",
        "| --- | --- | --- | --- |",
    ]
    for table, columns in dq["table_date_ranges"].items():
        for column, rng in columns.items():
            lines.append(
                f"| {table} | {column} | {rng['min_year']} | {rng['max_year']} |"
            )
    lines.append("")

    lines += [
        "### Per-hospital data coverage (`lookup_data_coverage`)",
        "",
        "| data | min_date | max_date | hospital_num | additional_info |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in dq["hospital_coverage"]:
        lines.append(
            f"| {row.get('data')} | {row.get('min_date')} | {row.get('max_date')} "
            f"| {row.get('hospital_num')} | {row.get('additional_info')} |"
        )
    lines.append("")

    lines += [
        "### Encounters per year (`admdad_subset`)",
        "",
        "| year | count |",
        "| --- | --- |",
    ]
    for year, n in dq["encounters_per_year"].items():
        lines.append(f"| {year} | {n} |")
    lines.append("")

    lines += [
        "### Lookup tables confirmed genuinely empty (real EXISTS check)",
        "",
        "| table | genuinely empty |",
        "| --- | --- |",
    ]
    for table, is_empty in dq["lookup_emptiness"].items():
        lines.append(f"| {table} | {is_empty} |")
    lines.append("")
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
