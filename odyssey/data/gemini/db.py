"""Thin database access layer for the GEMINI data cut.

Everything that touches the network lives here, mirroring the pattern
already validated in production by the ``gemini-variation-study`` repo.
Extraction/exploration code built on top of this should operate on the
:class:`pandas.DataFrame` objects :func:`query` returns, not on a live
connection, so it stays testable without a database -- see
``tests/odyssey/data/gemini/test_db.py``, which exercises this module
against a fake connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from . import config


class _CopySink(Protocol):
    """The minimal write side of a file-like object :func:`copy_to_sink` needs.

    Deliberately narrower than ``BinaryIO`` -- ``copy_to_sink`` (and
    psycopg2's ``copy_expert``, which it wraps) only ever calls
    ``write()``, so a caller-supplied sink only has to implement that,
    not the rest of ``BinaryIO``'s read/seek surface (see
    ``scripts/gemini/extract_meds.py``'s ``_CopyChunkSink``, which parses
    incrementally rather than accumulating bytes -- it isn't a real
    file-like object beyond ``write()``).
    """

    def write(self, data: bytes) -> int: ...


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine for the configured data cut.

    Returns
    -------
    Engine
        A cached SQLAlchemy engine.

    Raises
    ------
    RuntimeError
        If database configuration is incomplete (missing credentials, host,
        or database name -- see :data:`odyssey.data.gemini.config.DB_URL`).
    """
    if config.DB_URL is None:
        raise RuntimeError(config.credentials_help())
    return create_engine(config.DB_URL)


def query(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run ``sql`` against the configured GEMINI data cut.

    The schema search path is set per-connection so unqualified table names
    resolve to the active data cut
    (:data:`odyssey.data.gemini.config.DATACUT`).

    Parameters
    ----------
    sql : str
        A SQL statement. Use bound parameters (``:name``) rather than string
        interpolation for any value that could vary.
    params : dict[str, Any], optional
        Bound-parameter values.

    Returns
    -------
    pandas.DataFrame
        The query result.

    Raises
    ------
    RuntimeError
        If database configuration is incomplete, including an unset data
        cut (:data:`odyssey.data.gemini.config.DATACUT`).
    """
    if config.DATACUT is None:
        raise RuntimeError(config.credentials_help())
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(f"SET search_path TO {config.DATACUT};"))
        return pd.read_sql(text(sql), conn, params=params)


def stream_query(
    sql: str, params: dict[str, Any] | None = None, *, chunksize: int = 500_000
) -> Iterator[pd.DataFrame]:
    """Stream ``sql`` from the configured GEMINI data cut via a server-side cursor.

    Unlike :func:`query`, which materializes the whole result client-side in
    one round trip, this keeps the query open on the server (a named/
    server-side cursor, via SQLAlchemy's ``stream_results`` execution
    option -- the standard way to get psycopg2's server-side-cursor
    behavior through SQLAlchemy) and yields it back in ``chunksize``-row
    pandas DataFrames. Built for exactly the fetch pattern
    ``scripts/gemini/extract_meds.py`` needs at real GEMINI scale
    (``lab_subset``/``vitals_subset`` are hundreds of millions of rows):
    one continuous query streamed in large chunks, instead of
    :func:`query`'s usage pattern of re-executing a fresh
    ``WHERE ... LIMIT ...`` query per page, which pays a full round trip
    (and, without a covering index on the ordering column, a rescan cost)
    on every single page.

    Parameters
    ----------
    sql : str
        A SQL statement. Use bound parameters (``:name``) rather than
        string interpolation for any value that could vary.
    params : dict[str, Any], optional
        Bound-parameter values.
    chunksize : int
        Rows per yielded DataFrame.

    Yields
    ------
    pandas.DataFrame
        Successive chunks of the result, in query order.

    Raises
    ------
    RuntimeError
        If database configuration is incomplete, including an unset data
        cut (:data:`odyssey.data.gemini.config.DATACUT`).
    """
    if config.DATACUT is None:
        raise RuntimeError(config.credentials_help())
    engine = get_engine()
    with engine.connect().execution_options(stream_results=True) as conn:
        conn.execute(text(f"SET search_path TO {config.DATACUT};"))
        yield from pd.read_sql(text(sql), conn, params=params, chunksize=chunksize)


def copy_to_sink(sql: str, sink: _CopySink) -> None:
    """Run a ``COPY ... TO STDOUT`` query, writing incrementally to ``sink``.

    Uses the DBAPI connection directly (psycopg2's ``copy_expert``, which
    ``COPY`` needs -- SQLAlchemy's ``Connection.execute`` cannot run
    ``COPY ... TO STDOUT``, since it isn't a normal result-returning
    statement) rather than SQLAlchemy's query interface, unlike every other
    function in this module. Unlike buffering the whole result and handing
    it back, this calls ``sink.write()`` exactly as psycopg2 calls it --
    repeatedly, as bytes arrive off the wire, not once at the end -- so a
    ``sink`` that itself parses and drains incrementally (see
    ``scripts/gemini/extract_meds.py``'s ``_CopyChunkSink``) keeps memory
    bounded to one chunk regardless of how large the ``COPY`` output is.
    That matters here specifically because ``COPY`` (unlike a named cursor)
    has no server-side pause/resume of its own -- the *only* way to bound
    memory on a ``COPY`` of ``lab_subset``'s real ~659M rows is to drain
    ``sink`` incrementally while the copy is still running, which is why
    this takes a caller-supplied sink rather than returning bytes.

    Parameters
    ----------
    sql : str
        A complete, literal ``COPY (...) TO STDOUT WITH (...)`` statement.
        ``COPY`` does not support bound parameters -- the caller is
        responsible for building it safely (only from this module's own
        hardcoded identifiers, never external input; see ``_quote_ident``
        in ``extract_meds.py``).
    sink : _CopySink
        A file-like object; ``write(bytes) -> int`` is called repeatedly
        with successive chunks of the server's CSV output.

    Raises
    ------
    RuntimeError
        If database configuration is incomplete, including an unset data
        cut (:data:`odyssey.data.gemini.config.DATACUT`).
    """
    if config.DATACUT is None:
        raise RuntimeError(config.credentials_help())
    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        try:
            cursor.execute(f"SET search_path TO {config.DATACUT};")
            cursor.copy_expert(sql, sink)
        finally:
            cursor.close()
    finally:
        raw_conn.close()


def list_available_schemata() -> pd.DataFrame:
    """List schema names visible in the configured database, no data cut needed.

    Bypasses :data:`odyssey.data.gemini.config.DATACUT` entirely (unlike
    :func:`query`, which requires it) -- meant for exactly one situation:
    the database connection is configured (:data:`config.DB_URL` is set)
    but no data cut has been chosen yet, so callers don't know what to set
    ``GEMINI_DATACUT`` to. See ``scripts/gemini/explore_schema.py``.

    Returns
    -------
    pandas.DataFrame
        One column, ``schema_name``, excluding Postgres' own internal
        schemas (``pg_%``, ``information_schema``).

    Raises
    ------
    RuntimeError
        If the database connection itself is not configured
        (:data:`odyssey.data.gemini.config.DB_URL` is unset).
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT LIKE 'pg_%' "
                "AND schema_name != 'information_schema' "
                "ORDER BY schema_name"
            ),
            conn,
        )
