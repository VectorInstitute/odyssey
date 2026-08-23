"""Missingness stress protocol: seeded, deterministic degradation of MEDS shards.

See ``docs/missingness_protocol.md`` for the full design (this module builds
exactly the "degraded-shard generator + staleness filter" that doc's
sequencing section calls lead-owned, tested infrastructure). Three axes, all
implemented as MEDS-row-level transforms on the raw shard -- Principle 2's
"applied once, at the MEDS row level, before any family-specific feature
construction or tokenization" holds uniformly across all three, so no family
(baseline or the flagship sequence model) needs axis-specific glue code:

- **A. MCAR event dropout** (:func:`apply_mcar`): drop each non-anchor row
  independently with probability ``p``. Anchor rows (admission/discharge/
  demographic statics, :func:`odyssey.data.vocabulary.is_anchor`) are never
  dropped -- a record with no visit envelope is a different task, not a
  degraded one.
- **B. Family blackout** (:func:`apply_family_blackout`): remove every row
  of one family (labs, vitals/charting, medications;
  :func:`odyssey.data.vocabulary.row_family`).
- **C. Lab availability lag** (:func:`apply_lab_lag`): shift every
  lab-family event's ``time`` forward by ``lag_hours`` -- the result
  "returns from the lab" later, and every family's existing "strictly
  before index time" visibility rule (already how baseline feature
  lookups and token ordering both work) does the rest, no separate filter
  needed. Two hard constraints, both enforced here: a subject's origin
  (first timed non-birth event) must never move, so a lab row sitting
  exactly at that time is exempted from the shift; and static/anchor rows
  are never shifted either. Verified post-transform, not just designed in:
  :func:`apply_lab_lag` asserts every subject's minimum non-birth time is
  unchanged before returning.

Every degraded shard directory carries a ``metadata.json`` (written by
:func:`generate_cell`) recording the transform, seed, params, and a sha256
of every source shard file -- so a scored result always carries exactly
what produced it (Principle 4).

CLI: ``python -m odyssey.data.degrade --held-out-shard-dir ... --output-root
... --cells all|<names> --seed 0`` writes one shard directory per cell
under ``--output-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from odyssey.data.sequences import BIRTH_CODE
from odyssey.data.vocabulary import (
    LAB_FAMILY,
    ROW_FAMILIES,
    is_anchor,
    row_family,
)
from odyssey.training.data import shard_sort_key


logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"

#: The protocol's fixed grid (docs/missingness_protocol.md: "8 cells + clean
#: baseline"). Not overridable via the CLI -- a different grid is a design
#: change to the protocol itself, not a run-time knob.
MCAR_PROBABILITIES: Tuple[float, ...] = (0.1, 0.3, 0.5)
LAG_HOURS_GRID: Tuple[float, ...] = (4.0, 8.0)

KNOWN_SOURCES: Tuple[str, ...] = ("mimic_iv", "eicu", "gemini")


# ---------------------------------------------------------------------------
# Cell spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One degradation cell's identity: transform + seed + params.

    Written verbatim (plus source shard hashes) to ``metadata.json`` next
    to the degraded shard directory, so a scored result always carries
    exactly what produced it (docs/missingness_protocol.md, Principle 4).
    """

    name: str
    transform: str  # "mcar" | "family_blackout" | "lab_lag"
    seed: int
    params: Dict[str, object] = field(default_factory=dict)


def all_cells(seed: int) -> Dict[str, Cell]:
    """Build the protocol's fixed 8-cell grid, seeded."""
    cells: Dict[str, Cell] = {}
    for p in MCAR_PROBABILITIES:
        name = f"mcar_{p:g}".replace(".", "_")
        cells[name] = Cell(name=name, transform="mcar", seed=seed, params={"p": p})
    for family in ROW_FAMILIES:
        name = f"blackout_{family}"
        cells[name] = Cell(
            name=name, transform="family_blackout", seed=seed, params={"family": family}
        )
    for lag in LAG_HOURS_GRID:
        name = f"lag_{lag:g}h".replace(".", "_")
        cells[name] = Cell(
            name=name, transform="lab_lag", seed=seed, params={"lag_hours": lag}
        )
    return cells


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def _shard_seed(base_seed: int, path: Path) -> Tuple[int, int]:
    """Per-shard seed: (base seed, shard's own numeric index).

    Not processing order -- shards are logically independent (a MEDS
    invariant: subjects never span shards), so the result must not depend
    on what order they happen to be generated in, only on the shard's own
    identity. ``numpy``'s ``default_rng`` accepts a sequence of ints as a
    single seed, combining them deterministically.
    """
    return (base_seed, shard_sort_key(path))


def apply_mcar(
    events: pl.DataFrame, *, p: float, seed: Union[int, Tuple[int, int]]
) -> pl.DataFrame:
    """Drop each non-anchor row independently with probability ``p`` (axis A)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    codes = events["code"].to_list()
    anchor = np.fromiter((is_anchor(c) for c in codes), dtype=bool, count=len(codes))
    rng = np.random.default_rng(seed)
    draws = rng.random(len(codes))
    keep = anchor | (draws >= p)
    return events.filter(pl.Series(keep))


def apply_family_blackout(
    events: pl.DataFrame, *, family: str, source: str
) -> pl.DataFrame:
    """Remove every row of ``family`` (axis B)."""
    if family not in ROW_FAMILIES:
        raise ValueError(f"unknown family {family!r}, expected one of {ROW_FAMILIES}")
    codes = events["code"].to_list()
    keep = np.fromiter(
        (row_family(c, source=source) != family for c in codes),
        dtype=bool,
        count=len(codes),
    )
    return events.filter(pl.Series(keep))


def apply_lab_lag(
    events: pl.DataFrame, *, lag_hours: float, source: str
) -> pl.DataFrame:
    """Shift every lab-family event's ``time`` forward by ``lag_hours`` (axis C).

    Not by editing the shard's row set (axis A/B's mechanism) -- every row
    survives, only lab-family timestamps move, so a lab "returns from the
    lab" ``lag_hours`` later than it really was drawn. Two exemptions, both
    required so a subject's time origin (first timed non-birth event,
    :func:`odyssey.data.alert_events.origin_hours`) never moves: a
    lab-family row sitting exactly at that subject's minimum non-birth
    time, and any anchor/static row (never lab-family in practice, kept as
    a belt-and-suspenders check). Verified, not just designed in: asserts
    every subject's minimum non-birth time is unchanged before returning.
    """
    if lag_hours < 0:
        raise ValueError(f"lag_hours must be >= 0, got {lag_hours}")
    codes = events["code"].to_list()
    is_lab = np.fromiter(
        (row_family(c, source=source) == LAB_FAMILY for c in codes),
        dtype=bool,
        count=len(codes),
    )
    is_protected = np.fromiter(
        (is_anchor(c) for c in codes), dtype=bool, count=len(codes)
    )

    non_birth = pl.col("code") != BIRTH_CODE
    origin_before = (
        events.filter(non_birth)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
    )

    is_origin_row = np.zeros(len(codes), dtype=bool)
    if origin_before.height:
        origin_map = dict(
            zip(
                origin_before["subject_id"].to_list(),
                origin_before["_origin"].to_list(),
            )
        )
        subject_ids = events["subject_id"].to_list()
        times = events["time"].to_list()
        is_origin_row = np.fromiter(
            (times[i] == origin_map.get(subject_ids[i]) for i in range(len(codes))),
            dtype=bool,
            count=len(codes),
        )

    shift = is_lab & ~is_protected & ~is_origin_row
    shifted = events.with_columns(
        pl.when(pl.Series(shift))
        .then(pl.col("time") + pl.duration(hours=lag_hours))
        .otherwise(pl.col("time"))
        .alias("time")
    )

    origin_after = (
        shifted.filter(non_birth)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
        .sort("subject_id")
    )
    if not origin_before.sort("subject_id").equals(origin_after):
        raise RuntimeError(
            "apply_lab_lag moved at least one subject's time origin -- this "
            "must never happen (a shifted lab row's exemption logic has a "
            "bug); refusing to return a corrupted degraded shard."
        )
    return shifted


# ---------------------------------------------------------------------------
# Cell generation
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet_atomic(frame: pl.DataFrame, dest: Path) -> None:
    """Write ``frame`` atomically.

    Via a ``.tmp`` sibling then rename -- a crash never leaves a truncated
    file at ``dest`` (same convention as extract_meds.py's
    ``MedsShardWriter``).
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    frame.write_parquet(tmp)
    os.replace(tmp, dest)


def generate_cell(
    cell: Cell, shard_paths: Sequence[Path], output_dir: Path, *, source: str
) -> None:
    """Materialize one degraded shard directory plus its ``metadata.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes: Dict[str, str] = {}
    for path in shard_paths:
        source_hashes[path.name] = _sha256_file(path)
        dest = output_dir / path.name
        events = pl.read_parquet(path)
        if cell.transform == "mcar":
            p = cell.params["p"]
            assert isinstance(p, float)
            degraded = apply_mcar(events, p=p, seed=_shard_seed(cell.seed, path))
        elif cell.transform == "family_blackout":
            family = cell.params["family"]
            assert isinstance(family, str)
            degraded = apply_family_blackout(events, family=family, source=source)
        elif cell.transform == "lab_lag":
            lag_hours = cell.params["lag_hours"]
            assert isinstance(lag_hours, float)
            degraded = apply_lab_lag(events, lag_hours=lag_hours, source=source)
        else:
            raise ValueError(f"unknown transform {cell.transform!r}")
        _write_parquet_atomic(degraded, dest)
    metadata = {
        "cell": cell.name,
        "transform": cell.transform,
        "seed": cell.seed,
        "params": cell.params,
        "source": source,
        "source_shard_hashes": source_hashes,
    }
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    logger.info(
        "[degrade] cell %s (%s): %d shards -> %s",
        cell.name,
        cell.transform,
        len(shard_paths),
        output_dir,
    )


def load_cell_metadata(cell_dir: Path) -> Dict[str, object]:
    """Read back a degraded cell directory's ``metadata.json``."""
    path = cell_dir / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{cell_dir} has no {METADATA_FILENAME} -- not a degrade.py output directory"
        )
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missingness-stress-protocol degraded MEDS shards "
            "(docs/missingness_protocol.md)."
        )
    )
    parser.add_argument("--held-out-shard-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["all"],
        help="'all', or one or more cell names (see all_cells()).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        default="mimic_iv",
        choices=KNOWN_SOURCES,
        help="MEDS source institution.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: generate one degraded shard directory per selected cell."""
    args = _parse_args(argv)

    shard_paths = sorted(args.held_out_shard_dir.glob("*.parquet"), key=shard_sort_key)
    if not shard_paths:
        raise FileNotFoundError(
            f"no .parquet shards found in {args.held_out_shard_dir}"
        )

    cells = all_cells(args.seed)
    if args.cells != ["all"]:
        unknown = sorted(set(args.cells) - set(cells))
        if unknown:
            raise ValueError(
                f"unknown cell name(s) {unknown}, expected 'all' or one of {sorted(cells)}"
            )
        cells = {name: cells[name] for name in args.cells}

    for name, cell in cells.items():
        generate_cell(cell, shard_paths, args.output_root / name, source=args.source)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
