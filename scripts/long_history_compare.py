"""Does unbounded history help? Two backbones on the same rows, split by truncation.

The hybrid backbone streams a record of any length at constant memory per
step; the transformer's packed-context evaluation keeps each subject's most
recent ``max_context`` tokens, so a long record is scored only inside that
tail window, with the window's start as the model's first visible token.
Both models' alert dumps (``alerts_rows.parquet``) are joined on (subject,
visit, time, event) and every subject is tagged from the dumps themselves:
``truncated`` if the transformer's first scored row comes later than the
hybrid's (the transformer never saw the record's start), else ``whole``.
The hazard heads' AUROC is compared per (event, horizon) on the two strata
with a paired subject-clustered bootstrap of the difference.

Read the truncated stratum with care: every one of its rows lies within
``max_context`` tokens of the record's end, and a transformer trained with
the same end-anchored truncation can read time-to-end off its position in
the window. Only the whole stratum is a clean backbone comparison.

Usage::

    uv run python scripts/long_history_compare.py \\
        --dump-a ~/runs/full_run_DEC_v12/alerts_rows.parquet --label-a hybrid \\
        --dump-b ~/runs/full_run_DEC_v12_tfm/alerts_rows.parquet \\
        --label-b transformer \\
        --output-json ~/runs/full_run_DEC_v12_tfm/long_history_compare.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from odyssey.inference.uncertainty import bootstrap_auroc_delta


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("long_history_compare")

KEY = ["subject_id", "visit_id", "time_hours", "event"]


def tag_truncation(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Per subject, ``truncated`` = dump b's first scored row is later than dump a's."""
    first_a = a.group_by("subject_id").agg(pl.col("time_hours").min().alias("_a0"))
    first_b = b.group_by("subject_id").agg(pl.col("time_hours").min().alias("_b0"))
    return first_a.join(first_b, on="subject_id", how="inner").select(
        "subject_id", (pl.col("_b0") > pl.col("_a0")).alias("truncated")
    )


def join_dumps(
    a: pl.DataFrame, b: pl.DataFrame, label_a: str, label_b: str
) -> pl.DataFrame:
    """Inner-join two dumps on the row key, keeping y@h and each model's hazard@h."""
    horizons = sorted({c.split("@")[1] for c in a.columns if c.startswith("hazard@")})
    keep = KEY + [f"y@{h}" for h in horizons] + [f"hazard@{h}" for h in horizons]
    ra = a.select(keep).rename({f"hazard@{h}": f"{label_a}@{h}" for h in horizons})
    rb = b.select(KEY + [f"hazard@{h}" for h in horizons]).rename(
        {f"hazard@{h}": f"{label_b}@{h}" for h in horizons}
    )
    return ra.join(rb, on=KEY, how="inner")


def compare(
    joined: pl.DataFrame,
    *,
    label_a: str,
    label_b: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Per (event, horizon, stratum): both AUROCs and the paired delta a - b."""
    horizons = sorted(
        {c.split("@")[1] for c in joined.columns if c.startswith(f"{label_a}@")}
    )
    cells: list[dict[str, Any]] = []
    for event in sorted(joined["event"].unique().to_list()):
        for h in horizons:
            for stratum, cond in (
                ("whole", ~pl.col("truncated")),
                ("truncated", pl.col("truncated")),
            ):
                sub = joined.filter(
                    (pl.col("event") == event) & cond & pl.col(f"y@{h}").is_not_null()
                )
                y = sub[f"y@{h}"].to_numpy().astype(int)
                if len(y) < 50 or y.min() == y.max():
                    continue
                pa = sub[f"{label_a}@{h}"].to_numpy().astype(float)
                pb = sub[f"{label_b}@{h}"].to_numpy().astype(float)
                sids = sub["subject_id"].to_numpy().astype(int)
                delta = bootstrap_auroc_delta(y, pa, pb, sids, n_boot=n_boot, seed=seed)
                cells.append(
                    {
                        "event": event,
                        "horizon": h,
                        "stratum": stratum,
                        "n_rows": int(len(y)),
                        "n_subjects": int(len(np.unique(sids))),
                        "n_positive": int(y.sum()),
                        f"auroc_{label_a}": float(roc_auc_score(y, pa)),
                        f"auroc_{label_b}": float(roc_auc_score(y, pb)),
                        "delta_a_minus_b": None
                        if delta is None
                        else dataclasses.asdict(delta),
                    }
                )
    return cells


def main() -> None:
    """Join the two dumps, tag truncation, and write the stratified comparison."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dump-a", required=True, type=Path)
    parser.add_argument("--label-a", default="hybrid")
    parser.add_argument("--dump-b", required=True, type=Path)
    parser.add_argument("--label-b", default="transformer")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    a = pl.read_parquet(args.dump_a)
    b = pl.read_parquet(args.dump_b)
    joined = join_dumps(a, b, args.label_a, args.label_b)
    logger.info(
        "dump a %d rows, dump b %d rows, joined %d", a.height, b.height, joined.height
    )
    tags = tag_truncation(a, b)
    joined = joined.join(tags, on="subject_id", how="inner")
    n_trunc = int(tags["truncated"].sum())
    rows_trunc = int(joined["truncated"].sum())
    logger.info(
        "truncated subjects: %d of %d (%d of %d joined rows)",
        n_trunc,
        tags.height,
        rows_trunc,
        joined.height,
    )
    cells = compare(
        joined,
        label_a=args.label_a,
        label_b=args.label_b,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    for c in cells:
        d = c["delta_a_minus_b"] or {}
        logger.info(
            "%s@%s %s: %s %.3f vs %s %.3f, delta %s (n %d, pos %d)",
            c["event"],
            c["horizon"],
            c["stratum"],
            args.label_a,
            c[f"auroc_{args.label_a}"],
            args.label_b,
            c[f"auroc_{args.label_b}"],
            {k: round(v, 4) for k, v in d.items() if isinstance(v, float)},
            c["n_rows"],
            c["n_positive"],
        )
    out = {
        "dump_a": str(args.dump_a),
        "label_a": args.label_a,
        "dump_b": str(args.dump_b),
        "label_b": args.label_b,
        "n_subjects": int(tags.height),
        "n_truncated_subjects": n_trunc,
        "n_joined_rows": int(joined.height),
        "n_truncated_rows": rows_trunc,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "cells": cells,
    }
    args.output_json.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (%d cells)", args.output_json, len(cells))


if __name__ == "__main__":
    main()
