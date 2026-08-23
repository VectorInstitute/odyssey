"""Build the MIMIC-IV-Note text sidecar (Track A item 7, text-modality probe).

Clinical text enters the program as a timeline modality through the same
label-only/feature-only sidecar mechanism as the microbiology table
(:mod:`odyssey.data.sidecars`): a parquet next to the MEDS ``data/``
directory, never tokenized into the event stream. This script writes the
raw notes table; :mod:`odyssey.text.embed_notes` turns it into pooled
embeddings; the probe then hands those embeddings to the tuned GBM as extra
alert features (cheap headroom test) before any fusion work is considered.

Radiology reports carry in-visit ``charttime`` (the alert-time signal);
discharge summaries are stamped at discharge (only the *next* visit or a
readmission task can see them). Both are kept, typed by ``note_type``.

Usage::

    uv run python scripts/build_mimic_note_sidecar.py \
        --note-root /Volumes/clinical-data/physionet.org/files/mimic-iv-note/2.2/note \
        --meds-root /Volumes/clinical-data/meds/mimiciv_3.1_v1 \
        --shard-dir <meds-root>/data/train --max-shards 30 \
        --shard-dir <meds-root>/data/held_out --max-shards 4

restricts the sidecar to the subjects of the given shard selections (the
probe scope: the subset runs' 30 train shards + the 4 held-out shards the
v3 dumps cover); omit ``--shard-dir`` for every subject. Output:
``<meds-root>/sidecars/notes.parquet`` with ``note_id, subject_id, hadm_id,
note_type, charttime, text``. Patient text stays on the local volume / VM;
the parquet is never committed.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Set

import polars as pl

from odyssey.training.shard_stream import shard_paths


logger = logging.getLogger("build_mimic_note_sidecar")


def subjects_of(shard_dir: Path, max_shards: Optional[int]) -> Set[int]:
    """Distinct subject ids across the first ``max_shards`` of ``shard_dir``."""
    ids: Set[int] = set()
    for path in shard_paths(shard_dir, max_shards=max_shards):
        ids.update(
            pl.read_parquet(path, columns=["subject_id"])["subject_id"].to_list()
        )
    return ids


def build_notes(note_root: Path, subjects: Optional[Set[int]]) -> pl.DataFrame:
    """Read radiology + discharge notes, optionally restricted to ``subjects``."""
    frames: List[pl.DataFrame] = []
    for name in ("radiology", "discharge"):
        path = note_root / f"{name}.csv.gz"
        lf = pl.scan_csv(
            path,
            schema_overrides={
                "note_id": pl.Utf8,
                "subject_id": pl.Int64,
                "hadm_id": pl.Int64,
                "note_type": pl.Utf8,
                "charttime": pl.Utf8,
                "text": pl.Utf8,
            },
        ).select("note_id", "subject_id", "hadm_id", "note_type", "charttime", "text")
        if subjects is not None:
            lf = lf.filter(pl.col("subject_id").is_in(sorted(subjects)))
        frame = lf.collect().with_columns(
            pl.col("charttime").str.strptime(
                pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False
            )
        )
        logger.info("[notes] %s: %d notes", name, frame.height)
        frames.append(frame)
    return pl.concat(frames).sort(["subject_id", "charttime"])


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--note-root", type=Path, required=True)
    parser.add_argument("--meds-root", type=Path, required=True)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        default=[],
        help="restrict to subjects of this split (repeatable; pairs with --max-shards)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        action="append",
        default=[],
        help="per --shard-dir: how many (sorted) shards to take; omit = all",
    )
    parser.add_argument("--output-name", default="notes")
    args = parser.parse_args()

    subjects: Optional[Set[int]] = None
    if args.shard_dir:
        subjects = set()
        for i, shard_dir in enumerate(args.shard_dir):
            cap = args.max_shards[i] if i < len(args.max_shards) else None
            ids = subjects_of(shard_dir, cap)
            logger.info("[notes] %s (max %s): %d subjects", shard_dir, cap, len(ids))
            subjects |= ids
    notes = build_notes(args.note_root, subjects)
    out_dir = args.meds_root / "sidecars"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.output_name}.parquet"
    notes.write_parquet(out)
    logger.info(
        "[notes] wrote %d notes for %d subjects (%s) -> %s",
        notes.height,
        notes["subject_id"].n_unique(),
        ", ".join(f"{k}={v}" for k, v in notes["note_type"].value_counts().iter_rows()),
        out,
    )


if __name__ == "__main__":
    main()
