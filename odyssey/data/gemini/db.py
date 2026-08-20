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

from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from . import config


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


def query(sql: str, params: Optional[dict[str, Any]] = None) -> pd.DataFrame:
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
