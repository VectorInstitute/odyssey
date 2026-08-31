"""Paired subject-clustered CIs for intervention-mode accuracy deltas.

Input: the ``*_per_subject.json`` written by ``python -m
odyssey.inference.interventions --dump-per-subject`` (``{mode:
{subject_id: [top1_hits, n_predictions]}}``). For each requested mode
pair (default: truth-flip, truth-none, flip-none), draws subjects with
replacement ONCE per resample and computes the pooled-accuracy difference
on that identical subject multiset, so the interval is on the PAIRED
delta -- the within-run sign claims of the paper (truth beats/loses to
flip) carry these intervals. Finite-sample variance only, as always.

Usage::

    uv run python scripts/intervention_cis.py \
        --per-subject ~/runs/<run>/interventions_band15_per_subject.json \
        --output-json ~/runs/<run>/intervention_cis.json
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("intervention_cis")

DEFAULT_PAIRS = (
    ("truth", "flip"),
    ("truth", "none"),
    ("flip", "none"),
    ("flip_gated", "none"),
    ("random", "none"),
    ("zero_known", "none"),
    ("zero_unknown", "none"),
    ("truth_calibrated", "flip_calibrated"),
    ("truth_calibrated", "none"),
    ("flip_calibrated", "none"),
)


def paired_accuracy_delta(
    counts_a: dict[int, list[int]],
    counts_b: dict[int, list[int]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired subject-clustered bootstrap of pooled accuracy(a) - accuracy(b).

    Subjects are the shared keys (both modes score the same stream, so a
    mismatch is a data error and raises rather than silently
    intersecting). One subject draw per resample; each drawn subject
    contributes its whole (hits, n) pair to BOTH arms, so the delta is
    paired at the cluster level.
    """
    if set(counts_a) != set(counts_b):
        only_a = len(set(counts_a) - set(counts_b))
        only_b = len(set(counts_b) - set(counts_a))
        raise ValueError(
            f"per-subject keys differ between modes ({only_a} only in the "
            f"first, {only_b} only in the second) -- both modes score the "
            "same stream, so this is a data error, not something to intersect"
        )
    sids = sorted(counts_a)
    hits_a = np.array([counts_a[s][0] for s in sids], dtype=np.float64)
    n_a = np.array([counts_a[s][1] for s in sids], dtype=np.float64)
    hits_b = np.array([counts_b[s][0] for s in sids], dtype=np.float64)
    n_b = np.array([counts_b[s][1] for s in sids], dtype=np.float64)

    point = float(hits_a.sum() / n_a.sum() - hits_b.sum() / n_b.sum())
    rng = np.random.default_rng(seed)
    n_subjects = len(sids)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        drawn = rng.integers(0, n_subjects, size=n_subjects)
        deltas[i] = (
            hits_a[drawn].sum() / n_a[drawn].sum()
            - hits_b[drawn].sum() / n_b[drawn].sum()
        )
    ci_low = float(np.percentile(deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "separated": ci_low > 0.0 or ci_high < 0.0,
        "n_subjects": n_subjects,
        "n_predictions": int(n_a.sum()),
        "n_boot": n_boot,
    }


def main() -> None:
    """Compute paired CIs for the standard mode pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-subject", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.per_subject) as f:
        data: dict[str, dict[str, list[int]]] = json.load(f)
    modes = {m: {int(k): v for k, v in d.items()} for m, d in data.items()}

    out: dict[str, Any] = {"n_boot": args.n_boot, "seed": args.seed, "pairs": {}}
    for a, b in DEFAULT_PAIRS:
        if a not in modes or b not in modes:
            continue
        res = paired_accuracy_delta(
            modes[a], modes[b], n_boot=args.n_boot, seed=args.seed
        )
        out["pairs"][f"{a}_minus_{b}"] = res
        logger.info(
            "%s - %s: %+0.4f [%+0.4f, %+0.4f] pt %s",
            a,
            b,
            res["point"] * 100,
            res["ci_low"] * 100,
            res["ci_high"] * 100,
            "SEPARATED" if res["separated"] else "within noise",
        )
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=1)
    logger.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
