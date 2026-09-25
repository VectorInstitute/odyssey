"""Paired hazard-head AUROC deltas between two runs on identical landmark rows.

The WP2 question (docs/ml4h2026_rebuttal_plan.md): what does the concept
bottleneck cost? The answer is a PAIRED difference between two runs of
the same backbone, one with the bottleneck (``full_run_v10``) and one
without (``full_run_baseline_v10``, ``model_kind=baseline``), scored by
the same alert chain on the same held-out shards. Each run's chain
writes a per-index-row dump (``alerts_rows.parquet``); this script
inner-joins the two dumps on the row key, checks that the joined row
set is the row set of BOTH dumps, and hands the aligned columns to
:func:`odyssey.inference.uncertainty.bootstrap_auroc_delta` (subject-
clustered, paired). The refusal on a row-set mismatch is deliberate:
numbers from different row sets in one table is the coverage-mismatch
bug this project has hit three times, so both counts and the unmatched
count are printed and the script stops.

Row key: ``(event, subject_id, visit_id, time_hours)``, the columns
:func:`odyssey.inference.alerts.index_row_table` writes. Score columns
are ``{scorer}@{h}h`` (``hazard`` for the model's heads, ``gbm`` for the
per-run GBM refit); the label column is ``y@{h}h`` with null meaning
censored before the horizon.

The GBM is refit inside each run's chain, so the two dumps carry two
different GBMs on the same rows. The ``gbm`` block reports both AUROCs
and their difference WITHOUT a bootstrap: that difference is refit
variance, and it is reported so the reader can put the hazard-head
delta next to it.

Usage::

    uv run python scripts/compare_runs_paired.py \
        --dump-a ~/runs/full_run_v10/alerts_rows.parquet \
        --dump-b ~/runs/full_run_baseline_v10/alerts_rows.parquet \
        --label-a bottleneck --label-b baseline \
        [--inference-a ~/runs/full_run_v10/inference_results.json] \
        [--inference-b ~/runs/full_run_baseline_v10/inference_results.json] \
        [--scorer hazard] [--events aki_stage_3 ...] [--horizons 8 24 72] \
        [--n-boot 1000] [--seed 0] [--max-subjects N] \
        --output-json ~/runs/full_run_baseline_v10/paired_vs_v10.json

Intervals carry finite-sample variance only (one fitted model per arm,
one held-out draw), exactly as in ``scripts/alerts_cis.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from odyssey.inference.uncertainty import bootstrap_auroc_delta
from scripts.alerts_cis import SCORER_ALIASES, horizons_in, subsample_subjects


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_runs_paired")

#: Columns that identify one landmark row across two dumps of the same
#: held-out shards (``event`` is added per event when joining).
ROW_KEY: tuple[str, ...] = ("subject_id", "visit_id", "time_hours")

#: Largest tolerated fraction of either dump's rows (per event) that the
#: inner join may leave unmatched before the script refuses.
MAX_UNMATCHED_FRACTION = 0.001

#: ``task_metrics`` keys of ``inference_results.json`` reported side by side.
INFERENCE_KEYS: tuple[str, ...] = (
    "set_top1_accuracy",
    "top1_accuracy",
    "top5_accuracy",
    "cross_entropy",
    "n_predictions",
    "n_set_predictions",
)


class RowSetMismatchError(RuntimeError):
    """The two dumps do not describe the same landmark rows."""


def load_dump(path: str | Path) -> pl.DataFrame:
    """Read a row dump and normalise the key columns for an exact join."""
    frame = pl.read_parquet(path)
    missing = [c for c in (*ROW_KEY, "event") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: dump is missing key columns {missing}")
    version = (
        frame["landmark_protocol_version"][0]
        if "landmark_protocol_version" in frame.columns and frame.height
        else None
    )
    logger.info(
        "%s: %d rows / %d subjects, landmark protocol %s",
        path,
        frame.height,
        frame["subject_id"].n_unique(),
        version,
    )
    return frame.with_columns(
        pl.col("subject_id").cast(pl.Int64),
        pl.col("visit_id").cast(pl.Int64),
        pl.col("time_hours").cast(pl.Float64),
        pl.col("event").cast(pl.Utf8),
    )


def _protocol_version(frame: pl.DataFrame) -> Any:
    if "landmark_protocol_version" not in frame.columns or frame.height == 0:
        return None
    return frame["landmark_protocol_version"][0]


def join_event(
    a: pl.DataFrame, b: pl.DataFrame, event: str, *, label_a: str, label_b: str
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Inner-join one event's rows of both dumps on :data:`ROW_KEY`.

    Refuses (:class:`RowSetMismatchError`) when a key repeats inside either
    dump, or when the join leaves more than :data:`MAX_UNMATCHED_FRACTION`
    of either dump's rows unmatched. Columns of ``b`` get a ``_b``
    suffix; ``a`` keeps its names.
    """
    ev_a = a.filter(pl.col("event") == event)
    ev_b = b.filter(pl.col("event") == event)
    keys = list(ROW_KEY)
    for name, frame in ((label_a, ev_a), (label_b, ev_b)):
        n_dup = frame.height - frame.select(keys).n_unique()
        if n_dup:
            raise RowSetMismatchError(
                f"{event}: {name} dump has {n_dup} duplicate row keys "
                f"{keys}; the join would multiply rows"
            )
    non_key_b = [c for c in ev_b.columns if c not in (*keys, "event")]
    joined = ev_a.join(
        ev_b.select([*keys, *non_key_b]).rename({c: f"{c}_b" for c in non_key_b}),
        on=keys,
        how="inner",
    )
    counts = {
        "n_rows_a": ev_a.height,
        "n_rows_b": ev_b.height,
        "n_rows_joined": joined.height,
        "n_unmatched_a": ev_a.height - joined.height,
        "n_unmatched_b": ev_b.height - joined.height,
        "n_subjects_joined": int(joined["subject_id"].n_unique()),
    }
    logger.info(
        "%s: %s n=%d, %s n=%d, joined n=%d (unmatched %d / %d), n_subjects=%d",
        event,
        label_a,
        counts["n_rows_a"],
        label_b,
        counts["n_rows_b"],
        counts["n_rows_joined"],
        counts["n_unmatched_a"],
        counts["n_unmatched_b"],
        counts["n_subjects_joined"],
    )
    for name, n_rows, n_unmatched in (
        (label_a, ev_a.height, counts["n_unmatched_a"]),
        (label_b, ev_b.height, counts["n_unmatched_b"]),
    ):
        if n_rows == 0 or n_unmatched / n_rows > MAX_UNMATCHED_FRACTION:
            raise RowSetMismatchError(
                f"{event}: row sets differ. {label_a} has {counts['n_rows_a']} "
                f"rows, {label_b} has {counts['n_rows_b']} rows, the join keeps "
                f"{counts['n_rows_joined']}; {n_unmatched} of {name}'s {n_rows} "
                f"rows are unmatched (limit {MAX_UNMATCHED_FRACTION:.1%}). The "
                "two dumps must be scored on identical held-out shards under "
                "the same landmark protocol."
            )
    return joined, counts


def _point_auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def score_pair(
    joined: pl.DataFrame,
    scorer: str,
    horizon: float,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any] | None:
    """Paired AUROC delta (b minus a) for one joined event frame and horizon.

    Rows are those with a label at the horizon in dump A, the same label
    in dump B, and a non-null score in both arms; a label disagreement is
    refused because the two chains ran the same label rule on the same
    events. Returns ``None`` when no row qualifies.
    """
    h = f"{horizon:g}h"
    y_col, s_col = f"y@{h}", f"{scorer}@{h}"
    needed = [y_col, s_col, f"{y_col}_b", f"{s_col}_b"]
    missing = [c for c in needed if c not in joined.columns]
    if missing:
        logger.warning("horizon %s: missing columns %s, skipped", h, missing)
        return None
    both_labelled = joined.filter(
        pl.col(y_col).is_not_null() & pl.col(f"{y_col}_b").is_not_null()
    )
    n_disagree = int((both_labelled[y_col] != both_labelled[f"{y_col}_b"]).sum())
    if n_disagree:
        raise RowSetMismatchError(
            f"horizon {h}: {n_disagree} rows carry different labels in the two "
            "dumps; the chains did not run the same label rule"
        )
    one_sided = int((joined[y_col].is_null() != joined[f"{y_col}_b"].is_null()).sum())
    sub = both_labelled.filter(
        pl.col(s_col).is_not_null() & pl.col(f"{s_col}_b").is_not_null()
    )
    if sub.height == 0:
        return None
    y = sub[y_col].to_numpy().astype(np.float64)
    p_a = sub[s_col].to_numpy().astype(np.float64)
    p_b = sub[f"{s_col}_b"].to_numpy().astype(np.float64)
    subj = sub["subject_id"].to_numpy()
    result: dict[str, Any] = {
        "n_at_risk": int(len(y)),
        "n_positive": int(y.sum()),
        "n_subjects": int(len(np.unique(subj))),
        "n_label_one_sided": one_sided,
        "auroc_a": _point_auroc(y, p_a),
        "auroc_b": _point_auroc(y, p_b),
        "delta_b_minus_a": None,
    }
    delta = bootstrap_auroc_delta(y, p_b, p_a, subj, n_boot=n_boot, seed=seed)
    if delta is None:
        result["unscoreable"] = True
        return result
    result["delta_b_minus_a"] = {
        "point": delta.point_estimate,
        "ci_low": delta.ci_low,
        "ci_high": delta.ci_high,
        "separated": delta.excludes_zero(),
        "n_boot_used": delta.n_boot_used,
        "n_boot_skipped": delta.n_boot_skipped,
    }
    return result


def gbm_pair(joined: pl.DataFrame, horizon: float) -> dict[str, Any] | None:
    """Both dumps' GBM AUROCs on the joined rows, and their difference.

    No bootstrap: the two GBMs are separate refits, so the difference is
    refit variance, reported for scale only.
    """
    h = f"{horizon:g}h"
    y_col, g_col = f"y@{h}", f"gbm@{h}"
    if g_col not in joined.columns or f"{g_col}_b" not in joined.columns:
        return None
    sub = joined.filter(
        pl.col(y_col).is_not_null()
        & pl.col(g_col).is_not_null()
        & pl.col(f"{g_col}_b").is_not_null()
    )
    if sub.height == 0:
        return None
    y = sub[y_col].to_numpy().astype(np.float64)
    auroc_a = _point_auroc(y, sub[g_col].to_numpy().astype(np.float64))
    auroc_b = _point_auroc(y, sub[f"{g_col}_b"].to_numpy().astype(np.float64))
    return {
        "n_at_risk": int(len(y)),
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "delta_b_minus_a": (
            None if auroc_a is None or auroc_b is None else auroc_b - auroc_a
        ),
    }


def read_inference(path: str | Path | None) -> dict[str, Any] | None:
    """Read the ``task_metrics`` keys of an ``inference_results.json``."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("%s: no inference_results.json, skipped", p)
        return None
    with open(p) as f:
        task = json.load(f).get("task_metrics", {})
    return {k: task.get(k) for k in INFERENCE_KEYS}


def inference_block(
    inf_a: dict[str, Any] | None, inf_b: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Side-by-side next-event metrics with b minus a on each numeric key."""
    if inf_a is None and inf_b is None:
        return None
    out: dict[str, Any] = {"a": inf_a, "b": inf_b, "delta_b_minus_a": {}}
    for k in INFERENCE_KEYS:
        va = (inf_a or {}).get(k)
        vb = (inf_b or {}).get(k)
        out["delta_b_minus_a"][k] = (
            None
            if va is None or vb is None or k.startswith("n_")
            else float(vb) - float(va)
        )
    return out


def summarise(cells: dict[str, Any]) -> dict[str, list[str]]:
    """Cells where b beats a, a beats b, or the paired interval covers 0."""
    out: dict[str, list[str]] = {"b_beats_a": [], "a_beats_b": [], "ties": []}
    for key, cell in cells.items():
        delta = cell.get("delta_b_minus_a")
        if delta is None or delta["separated"] is None:
            continue
        if not delta["separated"]:
            out["ties"].append(key)
        elif delta["point"] > 0:
            out["b_beats_a"].append(key)
        else:
            out["a_beats_b"].append(key)
    return out


def _fmt(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def markdown_table(
    cells: dict[str, Any], gbm: dict[str, Any], *, label_a: str, label_b: str
) -> str:
    """One row per (event, horizon): both AUROCs, the paired delta, the GBM refits."""
    lines = [
        f"| event | h | n | n+ | subjects | {label_a} | {label_b} | "
        f"{label_b} minus {label_a} [95% CI] | sep | gbm {label_a} | gbm {label_b} |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, cell in cells.items():
        event, _, horizon = key.rpartition("@")
        delta = cell.get("delta_b_minus_a")
        g = gbm.get(key) or {}
        delta_txt = (
            "n/a"
            if delta is None
            else f"{delta['point']:+.3f} [{_fmt(delta['ci_low'])}, "
            f"{_fmt(delta['ci_high'])}]"
        )
        sep = "" if delta is None else ("yes" if delta["separated"] else "no")
        lines.append(
            f"| {event} | {horizon} | {cell['n_at_risk']} | {cell['n_positive']} | "
            f"{cell['n_subjects']} | {_fmt(cell['auroc_a'])} | {_fmt(cell['auroc_b'])} "
            f"| {delta_txt} | {sep} | {_fmt(g.get('auroc_a'))} | "
            f"{_fmt(g.get('auroc_b'))} |"
        )
    return "\n".join(lines)


def compare(
    a: pl.DataFrame,
    b: pl.DataFrame,
    *,
    label_a: str,
    label_b: str,
    scorer: str,
    events: list[str] | None,
    horizons: list[float] | None,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Join the two dumps per event and score every (event, horizon) cell."""
    events_a = set(a["event"].unique().to_list())
    events_b = set(b["event"].unique().to_list())
    wanted = sorted(events) if events else sorted(events_a & events_b)
    absent = [e for e in wanted if e not in events_a or e not in events_b]
    if absent:
        raise RowSetMismatchError(
            f"events {absent} are not in both dumps ({label_a}: {sorted(events_a)}, "
            f"{label_b}: {sorted(events_b)})"
        )
    if not events and events_a != events_b:
        logger.warning(
            "event sets differ, scoring the intersection only: %s only in %s, "
            "%s only in %s",
            sorted(events_a - events_b),
            label_a,
            sorted(events_b - events_a),
            label_b,
        )
    out: dict[str, Any] = {"events": {}, "cells": {}, "gbm": {}}
    for event in wanted:
        joined, counts = join_event(a, b, event, label_a=label_a, label_b=label_b)
        out["events"][event] = counts
        hs = horizons if horizons else horizons_in(joined, scorer)
        for horizon in hs:
            cell = score_pair(joined, scorer, horizon, n_boot=n_boot, seed=seed)
            if cell is None:
                continue
            key = f"{event}@{horizon:g}h"
            out["cells"][key] = cell
            g = gbm_pair(joined, horizon)
            if g is not None:
                out["gbm"][key] = g
            delta = cell.get("delta_b_minus_a")
            logger.info(
                "%-28s n=%d (+%d, %d subjects)  %s=%s  %s=%s  delta=%s",
                key,
                cell["n_at_risk"],
                cell["n_positive"],
                cell["n_subjects"],
                label_a,
                _fmt(cell["auroc_a"]),
                label_b,
                _fmt(cell["auroc_b"]),
                "n/a"
                if delta is None
                else f"{delta['point']:+.4f} [{_fmt(delta['ci_low'], 4)}, "
                f"{_fmt(delta['ci_high'], 4)}]",
            )
    out["summary"] = summarise(out["cells"])
    return out


def main() -> None:
    """Paired AUROC deltas between two runs' alert row dumps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-a", required=True)
    parser.add_argument("--dump-b", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--inference-a", default=None)
    parser.add_argument("--inference-b", default=None)
    parser.add_argument("--scorer", default="hazard")
    parser.add_argument("--events", nargs="+", default=None)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=None,
        help="default: every horizon with score and label columns",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="seeded subject-level subsample (same subjects in both dumps)",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    scorer = SCORER_ALIASES.get(args.scorer, args.scorer)

    a = load_dump(args.dump_a)
    b = load_dump(args.dump_b)
    version_a, version_b = _protocol_version(a), _protocol_version(b)
    if version_a != version_b:
        logger.warning(
            "landmark protocol versions differ (%s: %s, %s: %s); the join "
            "check below decides whether the row sets still agree",
            args.label_a,
            version_a,
            args.label_b,
            version_b,
        )
    n_before = {"a": a.height, "b": b.height}
    a = subsample_subjects(a, max_subjects=args.max_subjects, seed=args.seed)
    if args.max_subjects is not None:
        # the same subjects in both arms, whatever their order in dump B
        b = b.filter(pl.col("subject_id").is_in(a["subject_id"].unique().to_list()))

    try:
        result = compare(
            a,
            b,
            label_a=args.label_a,
            label_b=args.label_b,
            scorer=scorer,
            events=args.events,
            horizons=args.horizons,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    except RowSetMismatchError as exc:
        logger.error("REFUSED: %s", exc)
        sys.exit(2)

    out: dict[str, Any] = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "dump_a": str(args.dump_a),
        "dump_b": str(args.dump_b),
        "scorer": scorer,
        "row_key": ["event", *ROW_KEY],
        "n_boot": args.n_boot,
        "seed": args.seed,
        "max_subjects": args.max_subjects,
        "landmark_protocol_version": {"a": version_a, "b": version_b},
        "n_rows_in_dumps": n_before,
        "n_rows_scored": {"a": a.height, "b": b.height},
        "variance_scope": "finite-sample only (one fitted model per arm); "
        "the gbm block shows refit variance without a bootstrap",
        **result,
        "inference": inference_block(
            read_inference(args.inference_a), read_inference(args.inference_b)
        ),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=1)
    print(
        markdown_table(
            out["cells"], out["gbm"], label_a=args.label_a, label_b=args.label_b
        )
    )
    if out["inference"] is not None:
        print()
        print(f"| metric | {args.label_a} | {args.label_b} | b minus a |")
        print("|---|---|---|---|")
        for k in INFERENCE_KEYS:
            va = (out["inference"]["a"] or {}).get(k)
            vb = (out["inference"]["b"] or {}).get(k)
            d = out["inference"]["delta_b_minus_a"][k]
            print(
                f"| {k} | {va if va is None or k.startswith('n_') else _fmt(va, 4)} | "
                f"{vb if vb is None or k.startswith('n_') else _fmt(vb, 4)} | "
                f"{'' if d is None else f'{d:+.4f}'} |"
            )
    s = out["summary"]
    logger.info(
        "wrote %s (%d cells): %s beats %s in %d, %s beats %s in %d, ties %d",
        args.output_json,
        len(out["cells"]),
        args.label_b,
        args.label_a,
        len(s["b_beats_a"]),
        args.label_a,
        args.label_b,
        len(s["a_beats_b"]),
        len(s["ties"]),
    )


if __name__ == "__main__":
    main()
