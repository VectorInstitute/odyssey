"""Why the tuned GBM beats the hazard head: a feature-group ablation.

Partitions the strong GBM panel (odyssey.inference.baseline_features, 609
features) into five groups by what the feature *is*, refits the paper's
GBM with each group dropped (its unique contribution) and with each group
kept alone (its total contribution), and scores every refit on the exact
held-out rows the paper's alerts table used, against the hazard head's
own AUROC on those rows. Hyperparameters are held at the paper's tuned
values per cell (alerts.json ``baseline_params``), the fit rows are the
same seed-0 subsample of the same baseline shards, so the ``full`` refit
reproduces the paper's GBM and every other number differs from it by the
feature subset alone.

Groups (name pattern -> size on the 609-feature strong panel):

* ``static`` (3): age, sex, hours since origin.
* ``recency`` (64): hours since the last value of each signal and drug
  class, hours into the visit, ICU status and hours since ICU admission.
* ``latest_value`` (48): the last value of each panel signal.
* ``summary_stats`` (384): per-signal window aggregates and trend
  (mean/min/max over 24 h, min/max over 6 h, delta from the previous
  value and from the visit's first, ratio to the visit minimum).
* ``counts_occurrence`` (110): per-signal 24 h counts, drug-class counts
  over 6 h/24 h and ever-in-visit, code-family counts, prior visits and
  events this visit.

Reports, per (event, horizon): the hazard head's AUROC, the full GBM's,
and per group the drop-one and keep-alone AUROCs plus the share of the
GBM-minus-hazard gap each explains. CPU only; the dump gives the rows,
outcomes and hazard scores, so no model forward pass is needed.

Usage::

    uv run python scripts/gbm_feature_ablation.py \\
        --run-dir ~/runs/<run> \\
        --dump ~/runs/<run>/alerts_rows.parquet \\
        --alerts-json ~/runs/<run>/alerts.json \\
        --held-out-shard-dir ~/data/<db>/data/held_out \\
        --baseline-shard-dir ~/data/<db>/data/train --max-baseline-shards 30 \\
        --output-json ~/runs/<run>/gbm_feature_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from odyssey.data.alert_events import alert_events_for
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.inference.alerts import (
    IndexRow,
    _fit_baseline_grid,
    _load_prepared_raw,
    features_for_events,
    load_index_row_table,
    stream_baseline_matrix,
)
from odyssey.inference.baseline_features import CONTEXT_FEATURES, feature_names
from odyssey.training.shard_stream import make_preparer, shard_paths
from odyssey.training.train import TrainingConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gbm_feature_ablation")

GROUP_ORDER: tuple[str, ...] = (
    "static",
    "recency",
    "latest_value",
    "summary_stats",
    "counts_occurrence",
)
_STATIC = {"age_years", "sex_female", "hours_since_origin"}
_RECENCY_CONTEXT = {"hours_into_visit", "in_icu", "hours_since_icu_admission"}
_COUNT_CONTEXT = {"n_prior_visits", "n_events_visit"}
_COUNT_SUFFIXES = (".n_6h", ".n_24h", ".n_visit", ".ever_visit")


def feature_groups(names: Sequence[str] | None = None) -> dict[str, list[int]]:
    """Partition the strong panel's columns into the five ablation groups.

    Every column lands in exactly one group; a name that fits none raises,
    so a new feature cannot be silently left out of the ablation.
    """
    names = list(feature_names() if names is None else names)
    groups: dict[str, list[int]] = {g: [] for g in GROUP_ORDER}
    for i, n in enumerate(names):
        if n in _STATIC:
            groups["static"].append(i)
        elif n in _RECENCY_CONTEXT or n.endswith(".hours_since_last"):
            groups["recency"].append(i)
        elif n in _COUNT_CONTEXT or n.endswith(_COUNT_SUFFIXES):
            groups["counts_occurrence"].append(i)
        elif n.endswith(".last"):
            groups["latest_value"].append(i)
        elif n in CONTEXT_FEATURES:
            raise ValueError(f"context feature {n!r} is not assigned to a group")
        else:
            groups["summary_stats"].append(i)
    assigned = sum(len(v) for v in groups.values())
    if assigned != len(names):
        raise ValueError(f"{assigned} of {len(names)} features assigned")
    return groups


def variants(groups: dict[str, list[int]], n_features: int) -> dict[str, np.ndarray]:
    """Column index arrays for ``full``, ``drop:<g>`` and ``keep:<g>``."""
    everything = np.arange(n_features)
    out: dict[str, np.ndarray] = {"full": everything}
    for g, cols in groups.items():
        drop = np.setdiff1d(everything, np.asarray(cols, dtype=int))
        out[f"drop:{g}"] = drop
        out[f"keep:{g}"] = np.asarray(sorted(cols), dtype=int)
    return out


def tuned_params(
    alerts_json: Path,
) -> dict[tuple[str, float], tuple[dict[str, float], int]]:
    """Read the paper's per-cell GBM hyperparameters from alerts.json."""
    out: dict[tuple[str, float], tuple[dict[str, float], int]] = {}
    for rec in json.loads(alerts_json.read_text()):
        if rec.get("scorer") != "baseline_gbm" or not rec.get("baseline_params"):
            continue
        params = dict(rec["baseline_params"])
        n_rounds = int(params.pop("n_rounds"))
        out[(rec["event"], float(rec["horizon_hours"]))] = (params, n_rounds)
    return out


def held_out_rows(
    dump: pl.DataFrame, events: Sequence[str]
) -> dict[str, list[IndexRow]]:
    """Per event, the dump's index rows in dump order."""
    rows: dict[str, list[IndexRow]] = {}
    for name in events:
        sub = dump.filter(pl.col("event") == name)
        rows[name] = [
            IndexRow(int(s), int(v), float(t))
            for s, v, t in zip(sub["subject_id"], sub["visit_id"], sub["time_hours"])
        ]
    return rows


def _cell_rows(
    dump: pl.DataFrame, event: str, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, y, hazard)`` for one cell: rows with an outcome at ``h``."""
    sub = dump.filter(pl.col("event") == event)
    y_col, hz_col = f"y@{h:g}h", f"hazard@{h:g}h"
    y = sub[y_col].to_numpy()
    mask = ~np.isnan(y.astype(float))
    hazard = sub[hz_col].to_numpy().astype(float)
    return mask, y.astype(float)[mask].astype(int), hazard[mask]


def gap_share(full: float, hazard: float, value: float, *, drop: bool) -> float | None:
    """Share of the GBM-over-hazard gap a group explains (None if no gap)."""
    gap = full - hazard
    if gap <= 1e-9:
        return None
    return (full - value) / gap if drop else (value - hazard) / gap


def run_ablation(
    x_train: np.ndarray,
    train_rows: Sequence[IndexRow],
    event_times: dict[str, Any],
    x_held: dict[str, np.ndarray],
    dump: pl.DataFrame,
    *,
    events: Sequence[str],
    horizons: Sequence[float],
    params: dict[tuple[str, float], tuple[dict[str, float], int]],
    groups: dict[str, list[int]],
    seed: int,
) -> list[dict[str, Any]]:
    """Fit every variant per cell and score it on the dump's rows."""
    cols = variants(groups, x_train.shape[1])
    results: list[dict[str, Any]] = []
    for event in events:
        if event not in event_times or event not in x_held:
            logger.warning("skipping %s: no train times or held-out features", event)
            continue
        for h in horizons:
            if (event, h) not in params:
                logger.warning(
                    "skipping %s@%gh: no tuned params in alerts.json", event, h
                )
                continue
            mask, y, hazard = _cell_rows(dump, event, h)
            if len(y) < 50 or len(set(y.tolist())) < 2:
                continue
            hazard_auroc = float(roc_auc_score(y, hazard))
            record: dict[str, Any] = {
                "event": event,
                "horizon_hours": h,
                "n_rows": int(len(y)),
                "n_positive": int(y.sum()),
                "hazard_auroc": hazard_auroc,
                "params": params[(event, h)][0],
                "n_rounds": params[(event, h)][1],
                "variants": {},
            }
            fixed = {h: params[(event, h)]}
            for name, idx in cols.items():
                fitted = _fit_baseline_grid(
                    x_train[:, idx],
                    train_rows,
                    event_times[event],
                    horizons=[h],
                    feature_set="strong",
                    seed=seed,
                    tune=False,
                    event_name=f"{event}[{name}]",
                    fixed=fixed,
                )
                if h not in fitted:
                    continue
                p = fitted[h].predict_proba(x_held[event][mask][:, idx])
                record["variants"][name] = {
                    "n_features": int(len(idx)),
                    "auroc": float(roc_auc_score(y, p)),
                }
            full = record["variants"].get("full", {}).get("auroc")
            record["full_auroc"] = full
            if full is not None:
                for g in groups:
                    for kind, drop in (("drop", True), ("keep", False)):
                        v = record["variants"].get(f"{kind}:{g}")
                        if v is not None:
                            v["gap_share"] = gap_share(
                                full, hazard_auroc, v["auroc"], drop=drop
                            )
            logger.info(
                "%s@%gh hazard %.3f full GBM %s: %s",
                event,
                h,
                hazard_auroc,
                f"{full:.3f}" if full is not None else "--",
                {
                    k: round(v["auroc"], 3)
                    for k, v in record["variants"].items()
                    if k != "full"
                },
            )
            results.append(record)
    return results


def main() -> None:
    """Build both feature matrices once, then refit and score every variant."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dump", required=True, help="alerts_rows.parquet of the paper's alerts pass"
    )
    parser.add_argument("--alerts-json", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--baseline-shard-dir", required=True)
    parser.add_argument(
        "--max-shards", type=int, default=None, help="held-out shards (None = all)"
    )
    parser.add_argument("--max-baseline-shards", type=int, default=30)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--horizons", type=float, nargs="+", default=[8.0, 24.0, 72.0])
    parser.add_argument("--events", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_config = json.loads((run_dir / "config.json").read_text())
    known = set(TrainingConfig.__dataclass_fields__)
    config = TrainingConfig(**{k: v for k, v in raw_config.items() if k in known})
    binner = QuantileBinner.load(run_dir / "quantile_binner.json")
    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    alerts = [a for a in alert_events_for(task_set, source=source) if not a.next_visit]
    dump = load_index_row_table(args.dump)
    events = (
        list(args.events) if args.events else sorted(dump["event"].unique().to_list())
    )
    alerts = [a for a in alerts if a.name in events]
    params = tuned_params(Path(args.alerts_json))
    groups = feature_groups()
    logger.info("groups: %s", {g: len(c) for g, c in groups.items()})

    logger.info(
        "building the baseline matrix on %s (%s shards)",
        args.baseline_shard_dir,
        args.max_baseline_shards,
    )
    activate_sidecars(args.baseline_shard_dir)
    prepare = make_preparer(
        normalize_medications=getattr(config, "normalize_medications", False),
        history_recap=getattr(config, "history_recap", False),
        source=source,
    )
    x_train, train_rows, event_times = stream_baseline_matrix(
        shard_paths(args.baseline_shard_dir, max_shards=args.max_baseline_shards),
        prepare,
        binner,
        alerts=alerts,
        source=source,
        landmark_hours=args.landmark_hours,
        feature_set="strong",
        task_set=task_set,
    )
    logger.info("train matrix %s", x_train.shape)

    logger.info(
        "building held-out features for the dump's rows from %s",
        args.held_out_shard_dir,
    )
    activate_sidecars(args.held_out_shard_dir)
    held_raw = _load_prepared_raw(
        args.held_out_shard_dir, args.max_shards, config, source
    )
    held_binned = add_value_tokens(held_raw, binner, source=source)
    del held_raw
    rows = held_out_rows(dump, events)
    x_held = features_for_events(held_binned, rows, source=source, feature_set="strong")
    del held_binned
    logger.info("held-out features: %s", {e: v.shape for e, v in x_held.items()})

    results = run_ablation(
        x_train,
        train_rows,
        event_times,
        x_held,
        dump,
        events=events,
        horizons=args.horizons,
        params=params,
        groups=groups,
        seed=args.seed,
    )
    out = {
        "run_dir": str(run_dir),
        "dump": str(args.dump),
        "source": source,
        "max_baseline_shards": args.max_baseline_shards,
        "max_held_out_shards": args.max_shards,
        "n_train_rows": int(len(train_rows)),
        "groups": {g: len(c) for g, c in groups.items()},
        "cells": results,
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (%d cells)", args.output_json, len(results))


if __name__ == "__main__":
    main()
