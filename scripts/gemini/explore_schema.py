#!/usr/bin/env python
"""Report the shape of the GEMINI data cut: objects, columns, row counts.

Schema-metadata only -- no patient-level rows are read. Row counts are
suppressed the same way ``gemini-variation-study`` suppresses every printed
table (see :func:`suppressed_row_count`): a count under 6 could itself
identify a patient and is reported as ``"<6"``; everything else is rounded
to the nearest 1000, since individually identifying precision is never
needed for a schema-discovery report. Writes
``scripts/gemini/out/schema.json`` and ``schema.md``, both small enough to
push through the 1 MiB-per-push cap on the ``gemini`` remote (see
``docs/gemini.md``).

Run on the GEMINI node, where the database is reachable and
``.env``/environment credentials are set (see ``odyssey/data/gemini/config.py``):

    uv run python scripts/gemini/explore_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from odyssey.data.gemini import config, db


OUT_DIR = Path(__file__).resolve().parent / "out"

_OBJECTS_SQL = """
SELECT 'table'   AS kind, tablename   AS name FROM pg_tables   WHERE schemaname = :s
UNION ALL
SELECT 'view'    AS kind, viewname    AS name FROM pg_views    WHERE schemaname = :s
UNION ALL
SELECT 'matview' AS kind, matviewname AS name FROM pg_matviews WHERE schemaname = :s
ORDER BY kind, name
"""

_COLUMNS_SQL = """
SELECT a.attname AS column,
       pg_catalog.format_type(a.atttypid, a.atttypmod) AS type
FROM pg_attribute a
JOIN pg_class     c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = :s AND c.relname = :t
  AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attnum
"""


def suppressed_row_count(n: int) -> str:
    """Round ``n`` to the nearest 1000, or mask small counts.

    Parameters
    ----------
    n : int
        A real row count.

    Returns
    -------
    str
        ``"<6"`` if ``n`` is under 6 (small-cell suppression: a small count
        could itself identify a patient, same threshold as
        ``gemini_variation.suppress``); otherwise ``n`` rounded to the
        nearest 1000, as a string.
    """
    if n < 6:
        return "<6"
    return str(round(n / 1000) * 1000)


def list_objects() -> pd.DataFrame:
    """List tables, views, and materialized views in the configured data cut.

    Returns
    -------
    pandas.DataFrame
        Columns ``kind`` and ``name``.
    """
    return db.query(_OBJECTS_SQL, {"s": config.DATACUT})


def list_columns(object_name: str) -> pd.DataFrame:
    """List columns and types for one object in the configured data cut.

    Parameters
    ----------
    object_name : str
        Table, view, or materialized view name.

    Returns
    -------
    pandas.DataFrame
        Columns ``column`` and ``type``.
    """
    return db.query(_COLUMNS_SQL, {"s": config.DATACUT, "t": object_name})


def row_count(object_name: str) -> int:
    """Return the real row count for one object.

    Parameters
    ----------
    object_name : str
        Table, view, or materialized view name, as returned by
        :func:`list_objects`. SQL identifiers cannot be bound parameters, so
        this is interpolated directly; it must only ever be called with a
        name that came from the database's own catalog, never external
        input.

    Returns
    -------
    int
        The real (unsuppressed) row count.
    """
    result = db.query(f"SELECT COUNT(*) AS n FROM {object_name}")
    return int(result["n"].iloc[0])


def build_schema_report() -> dict[str, Any]:
    """Query the configured data cut and assemble a suppressed schema report.

    Returns
    -------
    dict[str, Any]
        ``{"datacut": ..., "objects": [{"kind", "name", "row_count",
        "columns": [{"name", "type"}, ...]}, ...]}``. ``row_count`` is
        already suppressed via :func:`suppressed_row_count`.
    """
    objects = []
    for _, obj in list_objects().iterrows():
        kind = str(obj["kind"])
        name = str(obj["name"])
        columns = list_columns(name)
        objects.append(
            {
                "kind": kind,
                "name": name,
                "row_count": suppressed_row_count(row_count(name)),
                "columns": [
                    {"name": str(row["column"]), "type": str(row["type"])}
                    for _, row in columns.iterrows()
                ],
            }
        )
    return {"datacut": config.DATACUT, "objects": objects}


def render_markdown(report: dict[str, Any]) -> str:
    """Render a schema report as a human-readable Markdown document.

    Parameters
    ----------
    report : dict[str, Any]
        A report from :func:`build_schema_report` (or matching shape).

    Returns
    -------
    str
        Markdown text.
    """
    lines = [f"# GEMINI schema report: `{report['datacut']}`", ""]
    for obj in report["objects"]:
        lines.append(f"## {obj['kind']}: `{obj['name']}` (rows: {obj['row_count']})")
        lines.append("")
        lines.append("| column | type |")
        lines.append("| --- | --- |")
        for col in obj["columns"]:
            lines.append(f"| {col['name']} | {col['type']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Query the configured data cut and write schema.json and schema.md.

    If the database connection is configured but no data cut
    (``GEMINI_DATACUT``) has been chosen yet, lists the schemata actually
    visible in the database instead of raising the generic credentials
    error, so there's something to set ``GEMINI_DATACUT`` to.
    """
    if config.DB_URL is not None and config.DATACUT is None:
        names = db.list_available_schemata()["schema_name"].tolist()
        print("GEMINI_DATACUT is not set. Available schemata:")
        for name in names:
            print(f"  {name}")
        print("Set GEMINI_DATACUT to one of these.")
        return
    report = build_schema_report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "schema.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "schema.md").write_text(render_markdown(report))
    print(f"Wrote {OUT_DIR / 'schema.json'} and {OUT_DIR / 'schema.md'}")


if __name__ == "__main__":
    main()
