r"""Tests for GEMINI connection-config handling.

Regression guard: environment-sourced credentials must be whitespace-stripped
so a value exported with a trailing newline does not reach the database
driver (which otherwise fails auth with a confusing ``user "name\n"``
error) -- see :func:`odyssey.data.gemini.config._env`.
"""

import os
from pathlib import Path

from odyssey.data.gemini import config
from odyssey.data.gemini.config import _env, _env_optional, _load_dotenv


def test_env_strips_trailing_newline(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_DB_USER", "someuser\n")
    assert _env("GEMINI_DB_USER", "fallback") == "someuser"


def test_env_strips_surrounding_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_DB_PASS", "  secret  \n")
    assert _env("GEMINI_DB_PASS", "fallback") == "secret"


def test_env_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_DB_HOST", raising=False)
    assert _env("GEMINI_DB_HOST", "db.gemini-hpc.ca") == "db.gemini-hpc.ca"


def test_env_optional_none_when_unset_or_blank(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_DATACUT", raising=False)
    assert _env_optional("GEMINI_DATACUT") is None
    monkeypatch.setenv("GEMINI_DATACUT", "   \n")
    assert _env_optional("GEMINI_DATACUT") is None


def test_env_optional_strips_value(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_DATACUT", "  some_cut \n")
    assert _env_optional("GEMINI_DATACUT") == "some_cut"


def test_load_dotenv_sets_missing_but_respects_existing(
    monkeypatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        'GEMINI_DB_USER="fromfile"\n'
        "GEMINI_DB_PASS = spaced-value \n"
        "GEMINI_DB_NAME=already_set\n"
    )
    monkeypatch.delenv("GEMINI_DB_USER", raising=False)
    monkeypatch.delenv("GEMINI_DB_PASS", raising=False)
    monkeypatch.setenv("GEMINI_DB_NAME", "real_env_wins")

    _load_dotenv(env_file)

    assert os.environ["GEMINI_DB_USER"] == "fromfile"
    assert os.environ["GEMINI_DB_PASS"] == "spaced-value"
    # A pre-existing real env var is not overridden by the .env file.
    assert os.environ["GEMINI_DB_NAME"] == "real_env_wins"


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    _load_dotenv(tmp_path / "does-not-exist.env")  # must not raise


def test_load_dotenv_accepts_shell_export_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_EXPORTED_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('export SOME_EXPORTED_KEY="v1"\n')
    _load_dotenv(env_file)
    assert os.environ.get("SOME_EXPORTED_KEY") == "v1"
    monkeypatch.delenv("SOME_EXPORTED_KEY", raising=False)


def test_credentials_help_names_each_missing_var_when_url_unset(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "DB_URL", None)
    monkeypatch.setattr(config, "DB_USER", None)
    monkeypatch.setattr(config, "DB_PASS", "set")
    monkeypatch.setattr(config, "DB_NAME", None)
    msg = config.credentials_help()
    assert "GEMINI_DB_USER" in msg
    assert "GEMINI_DB_NAME" in msg
    assert "GEMINI_DB_PASS" not in msg  # was set, shouldn't be named as missing


def test_credentials_help_names_datacut_when_url_complete(monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_URL", "postgresql+psycopg2://u:p@h:5432/db")
    monkeypatch.setattr(config, "DATACUT", None)
    msg = config.credentials_help()
    assert "GEMINI_DATACUT" in msg
