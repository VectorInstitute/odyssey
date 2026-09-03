"""Does unbounded history help? Two backbones on the same rows, split by history length.

The hybrid backbone streams a record of any length at constant memory per
step; a transformer sees at most ``--window`` tokens (its packed-context
evaluation keeps the most recent ones). Both models' alert dumps
(``alerts_rows.parquet``) are joined on (subject, visit, time, event), each
row is tagged with how many events the patient had accumulated by that
landmark, and the hazard heads' AUROC is compared per (event, horizon) on
two strata: rows within the window (both backbones saw the whole history)
and rows beyond it (only the hybrid did). A paired subject-clustered
bootstrap of the AUROC difference is reported per cell and stratum. If the
streaming property matters, the difference should appear in the long
stratum and not in the short one.

History length is the number of timed events with time <= the landmark,
counted from the held-out MEDS shards in the same hours-since-origin
frame the dumps use; one token per event, so it is the token position.

Usage::

    uv run python scripts/long_history_compare.py \\
        --dump-a ~/runs/full_run_DEC_v12/alerts_rows.parquet --label-a hybrid \\
        --dump-b ~/runs/full_run_DEC_v12_tfm/alerts_rows.parquet \\
        --label-b transformer \\
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out --window 2048 \\
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

from odyssey.data.alert_events import hours_since_origin, origin_hours
from odyssey.inference.uncertainty import bootstrap_auroc_delta
from odyssey.training.data import load_meds_shards


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("long_history_compare")

KEY = ["subject_id", "visit_id", "time_hours", "event"]


def history_lengths(events: pl.DataFrame) -> dict[int, np.ndarray]:
    """Per subject, the sorted hours-since-origin of every timed event.

    ``events`` is a raw MEDS frame (``subject_id``, ``time``, ``code``);
    the origin is each subject's first timed non-birth event, as in the
    alerts protocol, so a dump's ``time_hours`` is on the same axis.
    """
    origins = origin_hours(events)
    timed = events.filter(pl.col("time").is_not_null())
    hours = hours_since_origin(timed, "time", origins).select("subject_id", "time")
    out: dict[int, np.ndarray] = {}
    for sid, group in hours.group_by("subject_id"):
        key = int(sid[0]) if isinstance(sid, tuple) else int(sid)
        out[key] = np.sort(group["time"].to_numpy().astype(float))
    return out


def tag_history(frame: pl.DataFrame, lengths: dict[int, np.ndarray]) -> pl.DataFrame:
    """Add ``history_len``: events accumulated by each row's landmark time."""
    subjects = frame["subject_id"].to_numpy().astype(int)
    times = frame["time_hours"].to_numpy().astype(float)
    counts = np.zeros(len(frame), dtype=np.int64)
    for sid in np.unique(subjects):
        mask = subjects == sid
        t = lengths.get(int(sid))
        counts[mask] = 0 if t is None else np.searchsorted(t, times[mask], side="right")
    return frame.with_columns(pl.Series("history_len", counts))


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
    window: int,
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
                ("within_window", pl.col("history_len") <= window),
                ("beyond_window", pl.col("history_len") > window),
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
    """Join the two dumps, tag history length, and write the stratified comparison."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dump-a", required=True, type=Path)
    parser.add_argument("--label-a", default="hybrid")
    parser.add_argument("--dump-b", required=True, type=Path)
    parser.add_argument("--label-b", default="transformer")
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--window", type=int, default=2048)
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
    events = load_meds_shards(args.held_out_shard_dir)
    lengths = history_lengths(events)
    del events
    joined = tag_history(joined, lengths)
    beyond = int((joined["history_len"] > args.window).sum())
    logger.info(
        "rows beyond the %d-token window: %d of %d", args.window, beyond, joined.height
    )
    cells = compare(
        joined,
        label_a=args.label_a,
        label_b=args.label_b,
        window=args.window,
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
        "window": args.window,
        "n_joined_rows": int(joined.height),
        "n_beyond_window": beyond,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "cells": cells,
    }
    args.output_json.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (%d cells)", args.output_json, len(cells))


if __name__ == "__main__":
    main()
