"""Tests for the GEMINI database access layer.

No real database is used or required -- :func:`odyssey.data.gemini.db.query`
is exercised against a hand-rolled fake connection, and
:func:`odyssey.data.gemini.db.get_engine`'s credential/data-cut checks are
exercised by monkeypatching :mod:`odyssey.data.gemini.config` directly.
"""

import io
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


class _FakeStreamingConnection(_FakeConnection):
    """Like :class:`_FakeConnection`, but records ``execution_options`` calls.

    Mirrors what :func:`~odyssey.data.gemini.db.stream_query` actually calls:
    ``engine.connect().execution_options(stream_results=True)``, which on a
    real SQLAlchemy connection returns a new options-bound connection but
    here just records the call and returns ``self`` so the same fake can
    assert both the statements executed and the streaming option requested.
    """

    def __init__(self) -> None:
        super().__init__()
        self.execution_option_calls: list[dict[str, Any]] = []

    def execution_options(self, **kwargs: Any) -> "_FakeStreamingConnection":
        self.execution_option_calls.append(kwargs)
        return self


class _FakeStreamingEngine:
    """Fake ``Engine`` whose connection records ``execution_options`` calls."""

    def __init__(self) -> None:
        self.connection = _FakeStreamingConnection()

    def connect(self) -> _FakeStreamingConnection:
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


def test_stream_query_raises_clearly_when_datacut_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATACUT", None)
    with pytest.raises(RuntimeError, match="GEMINI database configuration"):
        next(db.stream_query("SELECT 1"))


def test_stream_query_sets_search_path_and_requests_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATACUT", "some_datacut")
    fake_engine = _FakeStreamingEngine()
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)

    chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4]})]
    captured: dict[str, Any] = {}

    def fake_read_sql(
        sql: Any,
        con: Any,
        params: Optional[dict[str, Any]] = None,
        chunksize: Optional[int] = None,
    ) -> Any:
        captured["sql"] = str(sql)
        captured["con"] = con
        captured["params"] = params
        captured["chunksize"] = chunksize
        return iter(chunks)

    monkeypatch.setattr(db.pd, "read_sql", fake_read_sql)

    result = list(
        db.stream_query("SELECT * FROM lab_subset", params={"x": 1}, chunksize=123_456)
    )

    assert result == chunks
    assert captured["con"] is fake_engine.connection
    assert captured["params"] == {"x": 1}
    assert captured["chunksize"] == 123_456
    assert "SELECT * FROM lab_subset" in captured["sql"]
    assert fake_engine.connection.execution_option_calls == [{"stream_results": True}]
    assert any(
        "SET search_path TO some_datacut" in stmt
        for stmt in fake_engine.connection.executed
    )


def test_stream_query_is_lazy_until_iterated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # stream_query is a generator function: calling it must not touch the
    # database (execute SET search_path, call read_sql) until the caller
    # actually starts iterating -- otherwise a caller who builds several
    # stream_query() generators up front before consuming any of them would
    # unexpectedly open several server-side cursors at once.
    monkeypatch.setattr(config, "DATACUT", "some_datacut")
    fake_engine = _FakeStreamingEngine()
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)

    read_sql_calls: list[Any] = []

    def fake_read_sql(*args: Any, **kwargs: Any) -> Any:
        read_sql_calls.append((args, kwargs))
        return iter([pd.DataFrame({"a": [1]})])

    monkeypatch.setattr(db.pd, "read_sql", fake_read_sql)

    generator = db.stream_query("SELECT * FROM lab_subset")

    assert fake_engine.connection.executed == []
    assert read_sql_calls == []

    next(generator)

    assert read_sql_calls != []
    assert any(
        "SET search_path TO some_datacut" in stmt
        for stmt in fake_engine.connection.executed
    )


class _FakeCursor:
    """Fake DBAPI cursor recording ``execute``/``copy_expert`` calls."""

    def __init__(self, csv_bytes: bytes) -> None:
        self.csv_bytes = csv_bytes
        self.executed: list[str] = []
        self.copy_expert_calls: list[str] = []
        self.closed = False

    def execute(self, statement: str) -> None:
        self.executed.append(statement)

    def copy_expert(self, sql: str, file: Any) -> None:
        self.copy_expert_calls.append(sql)
        file.write(self.csv_bytes)

    def close(self) -> None:
        self.closed = True


class _FakeRawConnection:
    """Fake ``raw_connection()`` result: one cursor, records ``close()``."""

    def __init__(self, csv_bytes: bytes) -> None:
        self.cursor_obj = _FakeCursor(csv_bytes)
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


class _FakeRawEngine:
    """Fake ``Engine`` whose ``raw_connection()`` returns a fake DBAPI connection."""

    def __init__(self, csv_bytes: bytes) -> None:
        self.raw_conn = _FakeRawConnection(csv_bytes)

    def raw_connection(self) -> _FakeRawConnection:
        return self.raw_conn


def test_copy_to_sink_raises_clearly_when_datacut_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATACUT", None)
    with pytest.raises(RuntimeError, match="GEMINI database configuration"):
        db.copy_to_sink("COPY (SELECT 1) TO STDOUT WITH (FORMAT CSV)", io.BytesIO())


def test_copy_to_sink_sets_search_path_and_writes_to_sink_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DATACUT", "some_datacut")
    fake_engine = _FakeRawEngine(b"a,b\n1,2\n")
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)

    sink = io.BytesIO()
    db.copy_to_sink("COPY (SELECT * FROM lab_subset) TO STDOUT WITH (FORMAT CSV)", sink)

    assert sink.getvalue() == b"a,b\n1,2\n"
    cursor = fake_engine.raw_conn.cursor_obj
    assert any("SET search_path TO some_datacut" in stmt for stmt in cursor.executed)
    assert cursor.copy_expert_calls == [
        "COPY (SELECT * FROM lab_subset) TO STDOUT WITH (FORMAT CSV)"
    ]
    assert cursor.closed is True
    assert fake_engine.raw_conn.closed is True


def test_list_available_schemata_does_not_require_datacut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No DATACUT set at all -- must not raise, unlike query().
    monkeypatch.setattr(config, "DATACUT", None)
    fake_engine = _FakeEngine()
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)

    expected = pd.DataFrame({"schema_name": ["cut_a", "cut_b"]})

    def fake_read_sql(sql: Any, _con: Any) -> pd.DataFrame:
        assert "information_schema.schemata" in str(sql)
        return expected

    monkeypatch.setattr(db.pd, "read_sql", fake_read_sql)

    result = db.list_available_schemata()

    assert result is expected
    # No search_path statement -- this path never needs a chosen data cut.
    assert fake_engine.connection.executed == []
