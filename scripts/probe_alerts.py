"""CLI over odyssey.inference.probe_baseline's EHRSHOT-style probe benchmark.

Thin driver, same shape as scripts/tabicl_strong_compare.py over
odyssey.inference.tabicl_baseline: all the logic lives in the library
module (odyssey/inference/probe_baseline.py, embedding_probe.py) so it is
importable and testable; this script is just argument parsing and a
formatted report.

    uv run python scripts/probe_alerts.py \
        --run-dir ~/runs/subset_run_v8_taskset_v3 \
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --max-train-shards 5 --max-held-out-shards 4
"""

from __future__ import annotations

import argparse
import logging

from odyssey.inference.probe_baseline import run_probe_benchmark


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_alerts")


def _fmt_ci(auroc: float | None, ci: object) -> str:
    if auroc is None:
        return "n/a"
    if ci is None:
        return f"{auroc:.3f} [no CI]"
    low = getattr(ci, "ci_low", None)
    high = getattr(ci, "ci_high", None)
    if low is None or high is None:
        return f"{auroc:.3f} [no CI]"
    return f"{auroc:.3f} [{low:.3f}, {high:.3f}]"


def main() -> None:
    """Run the probe benchmark and print an EHRSHOT-style report table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--max-train-shards", type=int, default=5)
    parser.add_argument("--max-held-out-shards", type=int, default=4)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--num-lanes", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_probe_benchmark(
        args.run_dir,
        args.train_shard_dir,
        args.held_out_shard_dir,
        max_train_shards=args.max_train_shards,
        max_held_out_shards=args.max_held_out_shards,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    print(
        "\ntask,horizon_h,n_at_risk,n_positive,"
        "probe_pre_auroc,probe_post_auroc[ci],gbm_auroc[ci]"
    )
    for cell in result.cells:
        h = f"{cell.horizon_hours:g}" if cell.horizon_hours is not None else "snapshot"
        pre = (
            f"{cell.probe_pre_auroc:.3f}" if cell.probe_pre_auroc is not None else "n/a"
        )
        post = _fmt_ci(cell.probe_post_auroc, cell.probe_post_ci)
        gbm = _fmt_ci(cell.gbm_auroc, cell.gbm_ci)
        print(f"{cell.task},{h},{cell.n_at_risk},{cell.n_positive},{pre},{post},{gbm}")


if __name__ == "__main__":
    main()
