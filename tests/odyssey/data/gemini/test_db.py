"""Tests for the GEMINI database access layer.

No real database is used or required -- :func:`odyssey.data.gemini.db.query`
is exercised against a hand-rolled fake connection, and
:func:`odyssey.data.gemini.db.get_engine`'s credential/data-cut checks are
exercised by monkeypatching :mod:`odyssey.data.gemini.config` directly.
"""

from typing import Any, Optional

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)

import pandas as pd  # noqa: E402

from odyssey.data.gemini import config, db  # noqa: E402


class _FakeConnection:
    """Records every statement passed to ``execute``."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, statement: Any) -> None:
        self.executed.append(str(statement))

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _FakeEngine:
    """Fake ``Engine``; ``connect()`` always returns the same connection."""

    def __init__(self) -> None:
        self.connection = _FakeConnection()

    def connect(self) -> _FakeConnection:
        return self.connection


def test_get_engine_raises_clearly_when_url_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.get_engine.cache_clear()
    monkeypatch.setattr(config, "DB_URL", None)
    with pytest.raises(RuntimeError, match="GEMINI database configuration"):
        db.get_engine()
    db.get_engine.cache_clear()


def test_query_raises_clearly_when_datacut_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATACUT", None)
    with pytest.raises(RuntimeError, match="GEMINI database configuration"):
        db.query("SELECT 1")


def test_query_sets_search_path_and_returns_read_sql_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATACUT", "some_datacut")
    fake_engine = _FakeEngine()
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)

    expected = pd.DataFrame({"a": [1, 2]})
    captured: dict[str, Any] = {}

    def fake_read_sql(
        sql: Any, con: Any, params: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        captured["sql"] = str(sql)
        captured["con"] = con
        captured["params"] = params
        return expected

    monkeypatch.setattr(db.pd, "read_sql", fake_read_sql)

    result = db.query("SELECT * FROM some_table", params={"x": 1})

    assert result is expected
    assert captured["con"] is fake_engine.connection
    assert captured["params"] == {"x": 1}
    assert "SELECT * FROM some_table" in captured["sql"]
    assert any(
        "SET search_path TO some_datacut" in stmt
        for stmt in fake_engine.connection.executed
    )
