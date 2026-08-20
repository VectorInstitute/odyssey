#!/usr/bin/env python
"""Pre-extraction sanity pass: row counts and per-column null fractions.

Driven entirely by ``scripts/gemini/out/schema.json`` (see
``explore_schema.py``) rather than any hardcoded table list, since this
project has no verified knowledge yet of which GEMINI tables the eventual
extraction spec will use -- every base table (``kind == "table"``) the
schema step actually found is checked. Prints ``pending schema report`` and
exits cleanly if that file doesn't exist yet, so this is safe to wire into
``run.sh`` before the real GEMINI database has ever been reached.

Same governance discipline as ``explore_schema.py``: only suppressed counts
(:func:`~explore_schema.suppressed_row_count`) ever get written or printed,
no patient-level rows are read, and views/materialized views are skipped
(derived, not base data the extraction would read from directly).

Run on the GEMINI node, after ``schema``:

    uv run python scripts/gemini/extract_dry.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odyssey.data.gemini import db


OUT_DIR = Path(__file__).resolve().parent / "out"
SCHEMA_PATH = OUT_DIR / "schema.json"

_NULL_COUNT_SQL = "SELECT COUNT(*) - COUNT({column}) AS n_null FROM {table}"
_ROW_COUNT_SQL = "SELECT COUNT(*) AS n FROM {table}"


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
    """Run row-count and null-fraction sanity queries for every base table.

    Parameters
    ----------
    schema : dict[str, Any]
        A report from ``explore_schema.build_schema_report`` (or matching
        shape), loaded from ``schema.json``.

    Returns
    -------
    dict[str, Any]
        ``{"datacut": ..., "tables": [{"name", "row_count",
        "columns": [{"name", "n_null"}, ...]}, ...]}``, tables only
        (``kind == "table"``), all counts suppressed.
    """
    tables = []
    for obj in schema["objects"]:
        if obj["kind"] != "table":
            continue
        name = obj["name"]
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


def render_markdown(report: dict[str, Any]) -> str:
    """Render an extract-dry report as human-readable Markdown.

    Parameters
    ----------
    report : dict[str, Any]
        A report from :func:`build_report` (or matching shape).

    Returns
    -------
    str
        Markdown text.
    """
    lines = [f"# GEMINI extract-dry report: `{report['datacut']}`", ""]
    for table in report["tables"]:
        lines.append(f"## `{table['name']}` (rows: {table['row_count']})")
        lines.append("")
        lines.append("| column | null count |")
        lines.append("| --- | --- |")
        for col in table["columns"]:
            lines.append(f"| {col['name']} | {col['n_null']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the extract-dry sanity pass, or report that schema.json is missing."""
    if not SCHEMA_PATH.exists():
        print(f"pending schema report ({SCHEMA_PATH} does not exist yet)")
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    report = build_report(schema)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "extract_dry.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "extract_dry.md").write_text(render_markdown(report))
    print(f"Wrote {OUT_DIR / 'extract_dry.json'} and {OUT_DIR / 'extract_dry.md'}")


if __name__ == "__main__":
    main()
