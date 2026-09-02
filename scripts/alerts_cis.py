"""Per-cell CIs and paired scorer deltas from an alerts row dump.

The alerts pipeline reports point estimates; this post-processor adds the
uncertainty the paper's statistics policy requires, computed from the
per-index-row dump (``--dump-rows`` output of ``python -m
odyssey.inference.alerts``): for every (event, horizon, scorer) cell, a
subject-clustered bootstrap CI on AUROC and AUPRC, and for every scorer
pair on the identical rows, a PAIRED subject-clustered bootstrap of the
AUROC and AUPRC differences (:mod:`odyssey.inference.uncertainty`; never
CI overlap). Bold-in-table significance = the paired delta CI excludes 0.

Intervals here carry FINITE-SAMPLE variance only (one fitted model, one
held-out draw); cross-run/refit variance needs seed replicates and is out
of scope for this script -- see the uncertainty module docstring.

Usage::

    uv run python scripts/alerts_cis.py \
        --dump ~/runs/<run>/alerts_rows_v4.parquet \
        [--dump ~/runs/<run>/alerts_rows_readmission_v4.parquet] \
        --output-json ~/runs/<run>/alerts_cis_v4.json \
        [--scorers hazard gbm] [--n-boot 1000] [--seed 0]

Scorer columns are the dump's ``{scorer}@{h}h`` columns; the default pair
list compares every requested scorer against the first one. The dump names
the tuned GBM column ``gbm`` while alerts.json's records call the same
scorer ``baseline_gbm``; both spellings are accepted here and mapped to the
dump's column.

Runtime notes from the full-held-out runs: this script imports sklearn (for
AUPRC), so on the GEMINI node it needs the GPU venv, not the lightweight
one. A 1000-draw subject bootstrap over ~30M index rows takes 15-20 h
(per-scorer CIs, then the paired pass); ``--max-subjects`` draws a
subject-level subsample before bootstrapping when that is not affordable,
and the subsample size is recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from odyssey.inference.uncertainty import (
    bootstrap_auroc,
    bootstrap_auroc_delta,
    bootstrap_metric,
    bootstrap_metric_delta,
)


# alerts.json calls the tuned GBM "baseline_gbm"; the row dump's columns
# call it "gbm". Accept both on the command line.
SCORER_ALIASES = {"baseline_gbm": "gbm"}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alerts_cis")


def _ci_dict(b: Any) -> dict[str, Any] | None:
    if b is None:
        return None
    return {
        "point": b.point_estimate,
        "ci_low": b.ci_low,
        "ci_high": b.ci_high,
        "n_boot_used": b.n_boot_used,
        "n_boot_skipped": b.n_boot_skipped,
    }


def _delta_dict(d: Any) -> dict[str, Any] | None:
    out = _ci_dict(d)
    if out is not None:
        excludes = d.excludes_zero()
        out["separated"] = excludes
    return out


def horizons_in(frame: pl.DataFrame, scorer: str) -> list[float]:
    """Horizons for which ``{scorer}@{h}h`` and ``y@{h}h`` columns exist."""
    out = []
    for col in frame.columns:
        m = re.fullmatch(rf"{re.escape(scorer)}@([0-9.]+)h", col)
        if m and f"y@{m.group(1)}h" in frame.columns:
            out.append(float(m.group(1)))
    return sorted(out)


def score_cell(
    frame: pl.DataFrame,
    scorers: list[str],
    horizon: float,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any] | None:
    """CIs + paired deltas for one (event-filtered frame, horizon).

    Rows are the intersection: a valid label AND a non-null score for
    EVERY requested scorer, so per-scorer intervals and paired deltas all
    describe the same sample (the unpaired-rows trap probe_ci_check.py
    once had).
    """
    h = f"{horizon:g}h"
    cols = [f"{s}@{h}" for s in scorers]
    sub = frame.filter(
        pl.col(f"y@{h}").is_not_null()
        & pl.all_horizontal([pl.col(c).is_not_null() for c in cols])
    )
    if sub.height == 0:
        return None
    y = sub[f"y@{h}"].to_numpy().astype(np.float64)
    subj = sub["subject_id"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "n_positive": int(y.sum()), "unscoreable": True}

    result: dict[str, Any] = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "scorers": {},
        "paired_deltas": {},
    }
    preds = {s: sub[f"{s}@{h}"].to_numpy().astype(np.float64) for s in scorers}
    for s in scorers:
        result["scorers"][s] = {
            "auroc": _ci_dict(
                bootstrap_auroc(y, preds[s], subj, n_boot=n_boot, seed=seed)
            ),
            "auprc": _ci_dict(
                bootstrap_metric(
                    y, preds[s], subj, average_precision_score, n_boot=n_boot, seed=seed
                )
            ),
        }
    ref = scorers[0]
    for s in scorers[1:]:
        result["paired_deltas"][f"{ref}_minus_{s}"] = {
            "auroc": _delta_dict(
                bootstrap_auroc_delta(
                    y, preds[ref], preds[s], subj, n_boot=n_boot, seed=seed
                )
            ),
            "auprc": _delta_dict(
                bootstrap_metric_delta(
                    y,
                    preds[ref],
                    preds[s],
                    subj,
                    average_precision_score,
                    n_boot=n_boot,
                    seed=seed,
                )
            ),
        }
    return result


def subsample_subjects(
    frame: pl.DataFrame, *, max_subjects: int | None, seed: int
) -> pl.DataFrame:
    """Keep every row of a seeded random subset of subjects.

    Subject-level (not row-level) so the clustered bootstrap still sees
    whole subjects; a no-op when ``max_subjects`` is None or not smaller
    than the subject count.
    """
    if max_subjects is None:
        return frame
    subjects = frame["subject_id"].unique().sort().to_numpy()
    if len(subjects) <= max_subjects:
        return frame
    keep = np.random.default_rng(seed).choice(
        subjects, size=max_subjects, replace=False
    )
    logger.info("subsampling %d of %d subjects", max_subjects, len(subjects))
    return frame.filter(pl.col("subject_id").is_in(keep.tolist()))


def main() -> None:
    """Compute per-cell CIs and paired deltas from alerts row dumps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scorers", nargs="+", default=["hazard", "gbm"])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="subject-level subsample (seeded) before bootstrapping; default all",
    )
    args = parser.parse_args()
    args.scorers = [SCORER_ALIASES.get(s, s) for s in args.scorers]

    out: dict[str, Any] = {
        "scorers": args.scorers,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "max_subjects": args.max_subjects,
        "variance_scope": "finite-sample only (single fitted model); refit "
        "variance requires seed replicates",
        "cells": {},
    }
    for path in args.dump:
        frame = subsample_subjects(
            pl.read_parquet(path), max_subjects=args.max_subjects, seed=args.seed
        )
        events = (
            frame["event"].unique().to_list() if "event" in frame.columns else [None]
        )
        for event in sorted(e for e in events if e is not None) or [None]:
            ev_frame = (
                frame.filter(pl.col("event") == event) if event is not None else frame
            )
            for horizon in horizons_in(ev_frame, args.scorers[0]):
                cell = score_cell(
                    ev_frame, args.scorers, horizon, n_boot=args.n_boot, seed=args.seed
                )
                if cell is None:
                    continue
                key = f"{event or 'all'}@{horizon:g}h"
                out["cells"][key] = cell
                sc = cell.get("scorers", {})
                logger.info(
                    "%-28s n=%d (+%d)  %s",
                    key,
                    cell["n"],
                    cell["n_positive"],
                    "  ".join(
                        f"{s}: auroc={v['auroc']['point']:.3f} "
                        f"[{v['auroc']['ci_low']:.3f},{v['auroc']['ci_high']:.3f}]"
                        for s, v in sc.items()
                        if v["auroc"] and v["auroc"]["ci_low"] is not None
                    ),
                )
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=1)
    logger.info("wrote %s (%d cells)", args.output_json, len(out["cells"]))


if __name__ == "__main__":
    main()
