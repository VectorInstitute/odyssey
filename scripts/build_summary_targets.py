"""Precompute the self-supervised window-summary targets for a data root.

Writes ``<out>/train/<shard>.parquet`` and ``<out>/tuning/<shard>.parquet``
(one raw target frame per shard, see
:func:`odyssey.training.summary_targets.compute_summary_targets`) and
``<out>/stats.json`` (standardization fitted on the train frames). Point a
training run at ``<out>`` with ``summary_targets_dir``.

Shards are independent, so they are processed in parallel. The prepare
step (code normalization, history recap) is the training run's own, taken
from a run config or the defaults, so the target rows' time stamps match
the tokens the model streams.

    uv run python scripts/build_summary_targets.py \\
        --data-root ~/data/mimiciv_3.1_v1/data \\
        --out ~/data/mimiciv_3.1_v1/summary_targets \\
        --config ~/runs/full_run_v10/config.json --workers 12
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from odyssey.training.data import load_meds_shard
from odyssey.training.shard_stream import make_preparer, shard_paths
from odyssey.training.summary_targets import (
    DEFAULT_LANDMARK_HOURS,
    compute_summary_targets,
    fit_summary_stats,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_summary_targets")


def _one(args: tuple[str, str, str, bool, bool, float]) -> str:
    path, out, source, normalize, recap, landmark_hours = args
    out_path = Path(out) / (Path(path).stem + ".parquet")
    if out_path.exists():
        return f"skip {out_path.name}"
    prepare = make_preparer(
        source=source, normalize_medications=normalize, history_recap=recap
    )
    events = prepare(load_meds_shard(Path(path)))
    frame = compute_summary_targets(
        events, source=source, landmark_hours=landmark_hours
    )
    frame.write_parquet(out_path)
    return f"{out_path.name}: {frame.height} rows"


def main() -> None:
    """Build targets for the train and tuning splits and fit the stats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--config", default=None, help="a run's config.json for source/prepare flags"
    )
    parser.add_argument("--max-train-shards", type=int, default=None)
    parser.add_argument("--max-tuning-shards", type=int, default=None)
    parser.add_argument("--landmark-hours", type=float, default=DEFAULT_LANDMARK_HOURS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    source = str(cfg.get("source", "mimic_iv"))
    normalize = bool(cfg.get("normalize_medications", True))
    recap = bool(cfg.get("history_recap", False))
    out = Path(args.out)
    for split, cap in (
        ("train", args.max_train_shards),
        ("tuning", args.max_tuning_shards),
    ):
        (out / split).mkdir(parents=True, exist_ok=True)
        paths = shard_paths(Path(args.data_root) / split, cap)
        logger.info("%s: %d shards -> %s", split, len(paths), out / split)
        jobs = [
            (str(p), str(out / split), source, normalize, recap, args.landmark_hours)
            for p in paths
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for msg in pool.map(_one, jobs):
                logger.info("  %s", msg)
    logger.info("fitting standardization on the train frames")
    frames = (pl.read_parquet(p) for p in sorted((out / "train").glob("*.parquet")))
    stats = fit_summary_stats(frames)
    stats.save(out / "stats.json")
    logger.info("wrote %s (%d targets)", out / "stats.json", len(stats.names))


if __name__ == "__main__":
    main()
