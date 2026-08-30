"""Missingness stress protocol: seeded, deterministic degradation of MEDS shards.

See ``docs/missingness_protocol.md`` for the full design (this module builds
exactly the "degraded-shard generator + staleness filter" that doc's
sequencing section calls lead-owned, tested infrastructure). Three axes, all
implemented as MEDS-row-level transforms on the raw shard -- Principle 2's
"applied once, at the MEDS row level, before any family-specific feature
construction or tokenization" holds uniformly across all three, so no family
(baseline or the flagship sequence model) needs axis-specific glue code:

- **A. MCAR event dropout** (:func:`apply_mcar`): drop each non-anchor,
  non-origin row independently with probability ``p``. Anchor rows
  (admission/discharge/demographic statics,
  :func:`odyssey.data.vocabulary.is_anchor`) are never dropped -- a record
  with no visit envelope is a different task, not a degraded one.
- **B. Family blackout** (:func:`apply_family_blackout`): remove every
  non-origin row of one family (labs, vitals/charting, medications;
  :func:`odyssey.data.vocabulary.row_family`).
- **C. Lab availability lag** (:func:`apply_lab_lag`): shift every
  lab-family event's ``time`` forward by ``lag_hours`` -- the result
  "returns from the lab" later, and every family's existing "strictly
  before index time" visibility rule (already how baseline feature
  lookups and token ordering both work) does the rest, no separate filter
  needed. Non-origin only, same as A/B.

Every axis protects each subject's time origin (first timed non-birth
event, :func:`_origin_row_mask`) BY CONSTRUCTION, not just by chance:
anchor rows are typically the origin but not always (a subject's very
first charted event can legitimately be a lab), so all three transforms
explicitly exempt the origin row regardless of what else they touch. A
cell must never fail downstream at scoring time
(:func:`odyssey.inference.baseline_prep._verify_matching_origins`) because
of a rare unlucky draw -- that check is now a belt-and-suspenders guard,
not the only line of defense. Verified here too, not just designed in:
every transform asserts (:func:`_assert_origin_preserved`) every subject's
origin is unchanged before returning.

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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
MCAR_PROBABILITIES: tuple[float, ...] = (0.1, 0.3, 0.5)
LAG_HOURS_GRID: tuple[float, ...] = (4.0, 8.0)

KNOWN_SOURCES: tuple[str, ...] = ("mimic_iv", "eicu", "gemini")


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
    params: dict[str, object] = field(default_factory=dict)


def all_cells(seed: int) -> dict[str, Cell]:
    """Build the protocol's fixed 8-cell grid, seeded."""
    cells: dict[str, Cell] = {}
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


def _shard_seed(base_seed: int, path: Path) -> tuple[int, int]:
    """Per-shard seed: (base seed, shard's own numeric index).

    Not processing order -- shards are logically independent (a MEDS
    invariant: subjects never span shards), so the result must not depend
    on what order they happen to be generated in, only on the shard's own
    identity. ``numpy``'s ``default_rng`` accepts a sequence of ints as a
    single seed, combining them deterministically.
    """
    return (base_seed, shard_sort_key(path))


def _origin_row_mask(events: pl.DataFrame) -> np.ndarray:
    """Per-row: True where the row IS that subject's own time origin.

    "Origin" = each subject's first timed non-birth event
    (:func:`odyssey.data.alert_events.origin_hours`'s own definition) --
    the row every downstream ``_hours`` computation (landmark buckets,
    feature lookups) is relative to. Shared by every axis A/B/C transform
    so "never touch the origin row" means the same thing everywhere.
    """
    n = events.height
    non_birth = pl.col("code") != BIRTH_CODE
    origins = (
        events.filter(non_birth)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
    )
    if not origins.height:
        return np.zeros(n, dtype=bool)
    origin_map = dict(
        zip(origins["subject_id"].to_list(), origins["_origin"].to_list())
    )
    subject_ids = events["subject_id"].to_list()
    times = events["time"].to_list()
    return np.fromiter(
        (times[i] == origin_map.get(subject_ids[i]) for i in range(n)),
        dtype=bool,
        count=n,
    )


def _assert_origin_preserved(
    before: pl.DataFrame, after: pl.DataFrame, *, caller: str
) -> None:
    """Fail loud if any subject's time origin moved between ``before``/``after``.

    Belt-and-suspenders, not the only line of defense: each transform
    already protects the origin row by construction (never dropped in
    :func:`apply_mcar`/:func:`apply_family_blackout`, exempted from the
    shift in :func:`apply_lab_lag`) -- this catches a bug in that
    protection immediately, at generation time, rather than downstream in
    :func:`odyssey.inference.baseline_prep._verify_matching_origins` at
    scoring time.
    """
    non_birth = pl.col("code") != BIRTH_CODE
    origin_before = (
        before.filter(non_birth)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
        .sort("subject_id")
    )
    origin_after = (
        after.filter(non_birth)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
        .sort("subject_id")
    )
    if not origin_before.equals(origin_after):
        raise RuntimeError(
            f"{caller} moved at least one subject's time origin -- this must "
            "never happen (the origin-row protection has a bug); refusing to "
            "return a corrupted degraded shard."
        )


def apply_mcar(
    events: pl.DataFrame, *, p: float, seed: int | tuple[int, int]
) -> pl.DataFrame:
    """Drop each non-anchor, non-origin row independently with probability ``p``.

    Axis A.

    The origin row (:func:`_origin_row_mask`) is protected the same way
    anchor rows are: a record whose very first event vanished isn't a
    degraded record, it's a different time base entirely. Anchor rows are
    typically the origin in practice, but not always (a subject's very
    first charted event can legitimately be a lab) -- protecting both by
    construction means a rare unlucky draw can never produce a broken cell.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    codes = events["code"].to_list()
    anchor = np.fromiter((is_anchor(c) for c in codes), dtype=bool, count=len(codes))
    is_origin = _origin_row_mask(events)
    rng = np.random.default_rng(seed)
    draws = rng.random(len(codes))
    keep = anchor | is_origin | (draws >= p)
    degraded = events.filter(pl.Series(keep))
    _assert_origin_preserved(events, degraded, caller="apply_mcar")
    return degraded


def apply_family_blackout(
    events: pl.DataFrame, *, family: str, source: str
) -> pl.DataFrame:
    """Remove every row of ``family`` (axis B), except the origin row.

    Same protection as :func:`apply_mcar`: if the blacked-out family
    happens to be the family of a subject's very first charted event, that
    one row survives anyway -- the origin must never move, regardless of
    which axis is doing the removing.
    """
    if family not in ROW_FAMILIES:
        raise ValueError(f"unknown family {family!r}, expected one of {ROW_FAMILIES}")
    codes = events["code"].to_list()
    is_origin = _origin_row_mask(events)
    keep = (
        np.fromiter(
            (row_family(c, source=source) != family for c in codes),
            dtype=bool,
            count=len(codes),
        )
        | is_origin
    )
    degraded = events.filter(pl.Series(keep))
    _assert_origin_preserved(events, degraded, caller="apply_family_blackout")
    return degraded


def apply_lab_lag(
    events: pl.DataFrame, *, lag_hours: float, source: str
) -> pl.DataFrame:
    """Shift every lab-family event's ``time`` forward by ``lag_hours`` (axis C).

    Not by editing the shard's row set (axis A/B's mechanism) -- every row
    survives, only lab-family timestamps move, so a lab "returns from the
    lab" ``lag_hours`` later than it really was drawn. Two exemptions, both
    required so a subject's time origin never moves: a lab-family row
    sitting exactly at that subject's origin (:func:`_origin_row_mask`),
    and any anchor/static row (never lab-family in practice, kept as a
    belt-and-suspenders check).
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
    is_origin = _origin_row_mask(events)

    shift = is_lab & ~is_protected & ~is_origin
    shifted = events.with_columns(
        pl.when(pl.Series(shift))
        .then(pl.col("time") + pl.duration(hours=lag_hours))
        .otherwise(pl.col("time"))
        .alias("time")
    )
    _assert_origin_preserved(events, shifted, caller="apply_lab_lag")
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
    source_hashes: dict[str, str] = {}
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


def load_cell_metadata(cell_dir: Path) -> dict[str, object]:
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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


def main(argv: Sequence[str] | None = None) -> None:
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
