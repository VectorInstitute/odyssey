"""Full-capability TabICL(strong) vs GBM(strong) on MIMIC, with bootstrap CIs.

Reuses the existing GBM(strong) scores already dumped in alerts_rows_v3.parquet
(fit on 30 train shards, scored on 4 held-out shards, protocol v3) instead of
refitting the GBM. Fits TabICL at its real default capability (n_estimators=8,
TABICL_MAX_ROWS=50,000, both fit_tabicl_baselines/tabicl_baseline.py defaults)
on the strong 609-feature panel, offload_mode="cpu" (this host has 165GB free
RAM, no need for disk offload). One model fit+scored+dropped at a time, same
discipline as scripts/rescore_extra_baselines.py.
"""

import argparse
import dataclasses
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from odyssey.data.alert_events import alert_events_for
from odyssey.data.value_binning import QuantileBinner
from odyssey.inference.baseline_prep import prepare_baseline_data
from odyssey.inference.tabicl_baseline import fit_tabicl_baselines
from odyssey.inference.uncertainty import bootstrap_auroc
from odyssey.training.shard_stream import make_preparer, shard_paths
from odyssey.training.train import TrainingConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tabicl_full_compare")

HORIZONS = (8.0, 24.0, 72.0)


def main() -> None:  # noqa: PLR0915
    """Fit TabICL(strong, full capability) per core event, score vs. the GBM dump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-shard-dir", required=True, type=Path)
    parser.add_argument("--held-out-shard-dir", required=True, type=Path)
    parser.add_argument("--existing-dump", required=True, type=Path)
    parser.add_argument("--max-train-shards", type=int, default=8)
    parser.add_argument("--max-held-out-shards", type=int, default=4)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument(
        "--only-event", default=None, help="restrict to one event, for validation"
    )
    parser.add_argument("--offload-mode", default="cpu")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    raw_config = json.loads((args.run_dir / "config.json").read_text())
    known_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    dropped = sorted(set(raw_config) - known_fields)
    if dropped:
        logger.info(
            "config.json has fields no longer on TrainingConfig, dropping: %s", dropped
        )
    config = TrainingConfig(
        **{k: v for k, v in raw_config.items() if k in known_fields}
    )
    binner = QuantileBinner.load(args.run_dir / "quantile_binner.json")
    source = getattr(config, "source", "mimic_iv")
    prepare = make_preparer(
        normalize_medications=getattr(config, "normalize_medications", False),
        history_recap=getattr(config, "history_recap", False),
        source=source,
    )
    # The run's OWN task-set/source event list, not the v1 module default.
    # R2 trains task_set v3, which adds sepsis3 on MIMIC (resolved away on
    # eICU by source), and scoring TabICL on only the v1 four would leave the
    # comparator table with a sepsis3 row the tabular baseline never saw.
    # A v1 run still resolves to exactly the four core events, so this is a
    # superset, not a behaviour change for anything already run.
    # next_visit events (readmission_30d) are deliberately excluded: they are
    # discharge-anchored at 168/720h under index_mode=visit_end, and this
    # script builds a landmark grid at HORIZONS. They need their own pass.
    task_set = getattr(config, "task_set", "v1")
    alerts = [a for a in alert_events_for(task_set, source=source) if not a.next_visit]
    logger.info(
        "task_set=%s source=%s -> scoring %d event(s): %s",
        task_set,
        source,
        len(alerts),
        ", ".join(a.name for a in alerts),
    )
    if args.only_event:
        alerts = [a for a in alerts if a.name == args.only_event]

    t0 = time.time()
    logger.info("preparing %d train shard(s)", args.max_train_shards)
    train = prepare_baseline_data(
        shard_paths(args.train_shard_dir, max_shards=args.max_train_shards),
        prepare,
        binner,
        alerts=alerts,
        feature_sets=("strong",),
        source=source,
        landmark_hours=args.landmark_hours,
        task_set=task_set,
    )
    logger.info("train prep done in %.0fs", time.time() - t0)
    for name, rows in train.rows.items():
        logger.info("  train candidate rows %s: %d", name, len(rows))

    t0 = time.time()
    logger.info("preparing %d held-out shard(s)", args.max_held_out_shards)
    held = prepare_baseline_data(
        shard_paths(args.held_out_shard_dir, max_shards=args.max_held_out_shards),
        prepare,
        binner,
        alerts=alerts,
        feature_sets=("strong",),
        source=source,
        landmark_hours=args.landmark_hours,
        task_set=task_set,
    )
    logger.info("held-out prep done in %.0fs", time.time() - t0)

    existing = pl.read_parquet(args.existing_dump)

    results = {}
    for alert in alerts:
        event = alert.name
        rows = train.rows.get(event, [])
        if not rows:
            continue
        t0 = time.time()
        models = fit_tabicl_baselines(
            pl.DataFrame(),
            {event: rows},
            {event: train.times[event]},
            horizons=HORIZONS,
            source=source,
            feature_set="strong",
            features={event: train.features["strong"][event]},
            offload_mode=args.offload_mode,
        )
        fit_s = time.time() - t0
        logger.info("fit %s: %d models in %.0fs", event, len(models), fit_s)

        held_rows = held.rows.get(event, [])
        held_feats = held.features["strong"][event]
        existing_ev = existing.filter(pl.col("event") == event)

        for h in HORIZONS:
            model = models.pop((event, h), None)
            if model is None:
                continue
            t0 = time.time()
            proba = model.predict_proba(held_feats)
            predict_s = time.time() - t0
            del model
            gc.collect()

            new_cols = pl.DataFrame(
                {
                    "subject_id": [float(r.subject_id) for r in held_rows],
                    "visit_id": [float(r.visit_id) for r in held_rows],
                    "time_hours": [r.time_hours for r in held_rows],
                    f"tabicl@{h:g}h": [float(v) for v in proba],
                }
            )
            # time_hours reaches the two sides through independent float
            # code paths (the dump via the tokenizer, new_cols via polars
            # arithmetic), so round to the same 6-decimal tolerance
            # alerts._landmark_key_set uses before joining -- an exact
            # float join silently drops every row that differs in the
            # last bits. The height checks below make any remaining
            # mismatch (or a duplicate dump key fanning out) loud instead
            # of quietly shifting n and the CIs.
            key_cols = ["subject_id", "visit_id", "time_hours"]
            rounded = pl.col("time_hours").round(6)
            new_keyed = new_cols.with_columns(rounded)
            existing_keyed = existing_ev.with_columns(rounded)
            for side_name, frame in (("dump", existing_keyed), ("new", new_keyed)):
                n_dup = frame.height - frame.unique(subset=key_cols).height
                if n_dup:
                    raise RuntimeError(
                        f"{event}@{h:g}h: {n_dup} duplicate "
                        f"(subject, visit, time) keys on the {side_name} side -- "
                        "an inner join would fan out; refusing to score"
                    )
            joined = existing_keyed.join(new_keyed, on=key_cols, how="inner")
            if joined.height != new_keyed.height:
                raise RuntimeError(
                    f"{event}@{h:g}h: joined {joined.height} of "
                    f"{new_keyed.height} freshly-scored rows against the dump "
                    f"({existing_keyed.height} dump rows) -- the row sets "
                    "disagree (different shards/landmark_hours/protocol?); "
                    "refusing to score a silently-reduced overlap"
                )
            y = joined[f"y@{h:g}h"].to_numpy()
            gbm_p = joined[f"gbm@{h:g}h"].to_numpy()
            tabicl_p = joined[f"tabicl@{h:g}h"].to_numpy()
            sid = joined["subject_id"].to_numpy().astype(int)
            mask = ~np.isnan(y)
            y, gbm_p, tabicl_p, sid = y[mask], gbm_p[mask], tabicl_p[mask], sid[mask]

            gbm_ci = bootstrap_auroc(y, gbm_p, sid)
            tabicl_ci = bootstrap_auroc(y, tabicl_p, sid)

            cell = {
                "n": int(mask.sum()),
                "fit_s": fit_s,
                "predict_s": predict_s,
                "gbm": None if gbm_ci is None else vars(gbm_ci),
                "tabicl": None if tabicl_ci is None else vars(tabicl_ci),
            }
            results[f"{event}@{h:g}h"] = cell
            logger.info(
                "%s@%gh n=%d gbm=%.4f tabicl=%.4f (predict %.0fs)",
                event,
                h,
                cell["n"],
                gbm_ci.point_estimate if gbm_ci else float("nan"),
                tabicl_ci.point_estimate if tabicl_ci else float("nan"),
                predict_s,
            )
            args.output_json.write_text(json.dumps(results, indent=2))
        del models
        gc.collect()

    args.output_json.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
