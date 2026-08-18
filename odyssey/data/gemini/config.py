"""GEMINI database connection configuration.

Credentials are never hard-coded. ``GEMINI_DB_USER`` and ``GEMINI_DB_PASS``
must be supplied via the real environment or a git-ignored ``.env`` file at
the repository root (see ``.env.example`` and ``docs/gemini.md``). Non-secret
connection parameters (host, port) have defaults the environment can
override; ``GEMINI_DB_NAME``/``GEMINI_DATACUT`` do not, since which database
and data cut odyssey actually uses on GEMINI is not decided yet (see the
"GEMINI to MEDS" section of ``docs/gemini.md``).

Missing configuration is reported at connection time
(:func:`odyssey.data.gemini.db.get_engine`), not at import, so code that only
imports this module -- including the offline test suite -- keeps working
without any secrets or a chosen data cut.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


#: Repository root (four levels up from this file:
#: odyssey/data/gemini/config.py).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from ``path`` into the environment if unset.

    Parameters
    ----------
    path : Path
        Path to a ``.env``-style file. Blank lines and ``#``-prefixed
        comments are skipped, and surrounding quotes are stripped. Real
        environment variables always win over values loaded here.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str) -> str:
    r"""Read an environment variable, stripping surrounding whitespace.

    Stripping matters: a credential exported with a trailing newline (e.g.
    via ``export GEMINI_DB_USER=$(cat file)``) would otherwise be sent
    verbatim and fail authentication with a confusing ``user "name\n"``
    error.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Value returned when the variable is unset.

    Returns
    -------
    str
        The stripped value, or ``default``.
    """
    value = os.environ.get(name)
    return value.strip() if value is not None else default


def _env_optional(name: str) -> Optional[str]:
    """Return a stripped environment variable, or ``None`` if unset/blank.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    str, optional
        The stripped value, or ``None`` if unset or blank after stripping.
    """
    value = os.environ.get(name)
    value = value.strip() if value is not None else ""
    return value or None


# ── Database ────────────────────────────────────────────────────────────────
# Non-secret parameters: real infrastructure defaults, environment may
# override.
DB_HOST = _env("GEMINI_DB_HOST", "db.gemini-hpc.ca")
DB_PORT = int(_env("GEMINI_DB_PORT", "5432"))

# Project-specific, no default: which database and data cut odyssey queries
# is not decided yet, so both must be set explicitly for a connection to be
# attempted at all.
DB_NAME = _env_optional("GEMINI_DB_NAME")

#: Schema (data cut) that every query runs against via ``SET search_path``.
DATACUT = _env_optional("GEMINI_DATACUT")

# Secrets: no defaults. Must come from the environment or .env.
DB_USER = _env_optional("GEMINI_DB_USER")
DB_PASS = _env_optional("GEMINI_DB_PASS")

#: Full SQLAlchemy URL. Prefer an explicit ``GEMINI_DB_URL``; otherwise
#: assemble it from the parts, but only when user, password, and database
#: name are all present. ``None`` means configuration is incomplete and
#: connecting will raise a clear error.
_explicit_url = _env_optional("GEMINI_DB_URL")
if _explicit_url:
    DB_URL: Optional[str] = _explicit_url
elif DB_USER and DB_PASS and DB_NAME:
    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_URL = None

#: Human-readable hint shown when configuration is incomplete.
CREDENTIALS_HELP = (
    "GEMINI database configuration is incomplete. Create a .env file at the "
    "repository root (see docs/gemini.md) with GEMINI_DB_USER, GEMINI_DB_PASS, "
    "GEMINI_DB_NAME, and GEMINI_DATACUT, or export them / GEMINI_DB_URL in the "
    "environment."
)
