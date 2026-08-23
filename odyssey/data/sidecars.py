"""Label-only sidecar tables that ride next to a MEDS extraction.

A sidecar is a small parquet under ``<meds root>/sidecars/<name>.parquet``
(sibling of ``data/`` and ``metadata/``) carrying information an outcome
definition needs but the extraction does not tokenize -- today the
MIMIC-IV microbiology specimen times the Sepsis-3 label anchors
"suspected infection" on (``scripts/build_mimic_sidecars.py``). Sidecars
are read by the label pipeline only (:mod:`odyssey.data.concepts`,
:mod:`odyssey.data.alert_events`); tokenization and every baseline's
featurization never see them, so no model family is handed an input the
others lack.

Sidecars are dataset-level context rather than a per-call argument: the
label functions are reached from a dozen entry points (training, the
alerts harness, inference, interventions, case studies, baseline prep),
and a concept's definition must not depend on which of them asked. Each
entry point therefore *activates* the sidecars for the shard directory it
is about to read (:func:`activate_sidecars`), and rules that need one ask
:func:`active_sidecar`. Tests use :func:`sidecar_context`.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Union

import polars as pl


logger = logging.getLogger(__name__)

SIDECAR_DIRNAME = "sidecars"
# Sidecar name: one row per culture specimen (subject_id, hadm_id nullable,
# time, spec_type_desc, positive_culture).
MICROBIOLOGY = "microbiology"
# Sidecar name: antibacterial prescription orders (subject_id, hadm_id, time,
# stoptime, drug, route) -- mimic-code's suspicion-of-infection anchor.
ANTIBIOTIC_ORDERS = "antibiotic_orders"

_ACTIVE: Dict[str, pl.DataFrame] = {}
_ACTIVE_ROOT: Optional[Path] = None


def sidecar_root_for(shard_dir: Union[str, Path]) -> Path:
    """``<root>/sidecars`` for a split directory like ``<root>/data/train``.

    Also accepts the ``<root>/data`` directory or ``<root>`` itself (the
    first ancestor, including the path, whose ``sidecars/`` exists wins;
    otherwise the conventional two-levels-up location is returned).
    """
    p = Path(shard_dir)
    for candidate in (p, p.parent, p.parent.parent):
        if (candidate / SIDECAR_DIRNAME).is_dir():
            return candidate / SIDECAR_DIRNAME
    return p.parent.parent / SIDECAR_DIRNAME


def discover_sidecars(shard_dir: Union[str, Path]) -> Dict[str, pl.DataFrame]:
    """Load every ``*.parquet`` under the sidecar root for ``shard_dir``."""
    root = sidecar_root_for(shard_dir)
    if not root.is_dir():
        return {}
    out: Dict[str, pl.DataFrame] = {}
    for path in sorted(root.glob("*.parquet")):
        out[path.stem] = pl.read_parquet(path)
        logger.info(
            "[sidecars] %s: %d rows from %s", path.stem, out[path.stem].height, path
        )
    return out


def activate_sidecars(shard_dir: Union[str, Path, None]) -> List[str]:
    """Make the sidecars next to ``shard_dir`` the active set; return their names.

    ``None`` clears the active set. Activating the same root twice is a
    no-op (the tables are cached).
    """
    global _ACTIVE, _ACTIVE_ROOT  # noqa: PLW0603
    if shard_dir is None:
        _ACTIVE, _ACTIVE_ROOT = {}, None
        return []
    root = sidecar_root_for(shard_dir)
    if _ACTIVE_ROOT is not None and root == _ACTIVE_ROOT:
        return sorted(_ACTIVE)
    _ACTIVE = discover_sidecars(shard_dir)
    _ACTIVE_ROOT = root
    return sorted(_ACTIVE)


def active_sidecar(name: str) -> Optional[pl.DataFrame]:
    """Return the active sidecar ``name`` or ``None`` if not activated/present."""
    return _ACTIVE.get(name)


def active_sidecar_names() -> List[str]:
    """Names of the currently active sidecars."""
    return sorted(_ACTIVE)


@contextmanager
def sidecar_context(tables: Mapping[str, pl.DataFrame]) -> Iterator[None]:
    """Temporarily make ``tables`` the active sidecars (tests, one-off scripts)."""
    global _ACTIVE, _ACTIVE_ROOT  # noqa: PLW0603
    saved, saved_root = _ACTIVE, _ACTIVE_ROOT
    _ACTIVE, _ACTIVE_ROOT = dict(tables), None
    try:
        yield
    finally:
        _ACTIVE, _ACTIVE_ROOT = saved, saved_root


__all__ = [
    "ANTIBIOTIC_ORDERS",
    "MICROBIOLOGY",
    "SIDECAR_DIRNAME",
    "activate_sidecars",
    "active_sidecar",
    "active_sidecar_names",
    "discover_sidecars",
    "sidecar_context",
    "sidecar_root_for",
]
