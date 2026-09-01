"""Hazard vs GBM(strong) vs TabICL(strong) on MIMIC, on one matched row set.

Reuses the hazard and GBM(strong) scores already dumped in the run's
alerts_rows parquet instead of refitting either. Fits TabICL at its real
default capability (n_estimators=8, TABICL_MAX_ROWS=50,000, both
fit_tabicl_baselines/tabicl_baseline.py defaults) on the strong 609-feature
panel, offload_mode="cpu" (the ultra host has 165GB free RAM, no need for
disk offload). One model fit+scored+dropped at a time, same discipline as
scripts/rescore_extra_baselines.py.

ROW COVERAGE IS THE THING TO WATCH. TabICL costs about half an hour of
prediction per cell, so this scores ``--max-held-out-shards`` (4) of the 37
held-out shards, while the eval chain's alerts.json covers all 37. Those two
numbers are therefore NOT interchangeable, and a comparator table that takes
its hazard column from alerts.json and its TabICL column from here is
comparing different samples. That is why this script scores hazard itself,
off the same join: every column it reports describes one row set.

``--dump-rows`` writes that row set in the column layout
``scripts/alerts_cis.py`` consumes, which turns the three columns into
PAIRED subject-clustered deltas. Point estimates with separate intervals
cannot settle "does TabICL beat the hazard heads"; the paired delta can.

Usage::

    uv run python scripts/tabicl_strong_compare.py \\
        --run-dir ~/runs/full_run_v10 \\
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \\
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \\
        --existing-dump ~/runs/full_run_v10/alerts_rows.parquet \\
        --output-json ~/runs/full_run_v10/tabicl_strong_v4.json \\
        --dump-rows ~/runs/full_run_v10/tabicl_matched_rows.parquet

Add ``--skip-tabicl`` to get the hazard/GBM half in minutes: it exercises
the same prep, join and row-set checks without the fits, so it doubles as a
cheap smoke test before committing the host to a multi-hour run.
"""

import argparse
import dataclasses
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from odyssey.data.alert_events import alert_events_for
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import QuantileBinner
from odyssey.inference.baseline_prep import prepare_baseline_data
from odyssey.inference.tabicl_baseline import fit_tabicl_baselines
from odyssey.inference.uncertainty import bootstrap_auroc
from odyssey.training.shard_stream import make_preparer, shard_paths
from odyssey.training.train import TrainingConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tabicl_full_compare")

HORIZONS = (8.0, 24.0, 72.0)
# Scored in this order, and the paper's own model comes first: alerts_cis.py
# pairs every later scorer against the first, so hazard-vs-GBM and
# hazard-vs-TabICL are the deltas that come out of the dump.
SCORERS = ("hazard", "gbm", "tabicl")
# Key, label and score columns; the ~600 f_* feature columns and the ctx.*
# diagnostics are not part of a scoring dump and would multiply its size.
_DUMP_KEYS = frozenset({"subject_id", "visit_id", "time_hours", "event"})


def _dumped(column: str) -> bool:
    """Whether ``column`` belongs in the row dump alerts_cis.py consumes."""
    prefix = column.split("@", maxsplit=1)[0]
    return column in _DUMP_KEYS or (
        "@" in column and (prefix == "y" or prefix in SCORERS)
    )


def landmark_keys(rows: list[Any]) -> pl.DataFrame:
    """Build the (subject, visit, time) key frame for a set of landmark rows."""
    return pl.DataFrame(
        {
            "subject_id": [float(r.subject_id) for r in rows],
            "visit_id": [float(r.visit_id) for r in rows],
            "time_hours": [r.time_hours for r in rows],
        }
    )


def tabicl_columns(
    event: str,
    train_rows: list[Any],
    train: Any,
    held_features: Any,
    keys: pl.DataFrame,
    *,
    source: str,
    offload_mode: str,
) -> tuple[pl.DataFrame, float, dict[float, float]]:
    """Fit TabICL for one event and score every horizon onto ``keys``.

    Each model is dropped as soon as it has been scored (the same
    discipline as scripts/rescore_extra_baselines.py): at the strong
    panel's width these do not comfortably co-exist in RAM.
    """
    t0 = time.time()
    models = fit_tabicl_baselines(
        pl.DataFrame(),
        {event: train_rows},
        {event: train.times[event]},
        horizons=HORIZONS,
        source=source,
        feature_set="strong",
        features={event: train.features["strong"][event]},
        offload_mode=offload_mode,
    )
    fit_s = time.time() - t0
    logger.info("fit %s: %d models in %.0fs", event, len(models), fit_s)

    predict_s: dict[float, float] = {}
    for horizon in HORIZONS:
        model = models.pop((event, horizon), None)
        if model is None:
            continue
        t0 = time.time()
        proba = model.predict_proba(held_features)
        predict_s[horizon] = time.time() - t0
        del model
        gc.collect()
        # Scoring now happens after ALL horizons are predicted, so without
        # this the log goes silent for the better part of two hours and a
        # healthy run is indistinguishable from a hung one.
        logger.info(
            "%s@%gh predicted %d rows in %.0fs",
            event,
            horizon,
            len(proba),
            predict_s[horizon],
        )
        keys = keys.with_columns(
            pl.Series(f"tabicl@{horizon:g}h", [float(v) for v in proba])
        )
    del models
    gc.collect()
    return keys, fit_s, predict_s


def join_matched_rows(
    dump: pl.DataFrame, scored: pl.DataFrame, event: str
) -> pl.DataFrame:
    """Inner-join freshly scored rows onto the dump, on the landmark key.

    ``time_hours`` reaches the two sides through independent float code
    paths (the dump via the tokenizer, ``scored`` via polars arithmetic),
    so both are rounded to the 6 decimals ``alerts._landmark_key_set``
    uses; an exact float join silently drops every row differing in the
    last bits. A partial overlap is refused rather than scored, because
    the whole point of the join is that one row set backs every column.
    """
    key_cols = ["subject_id", "visit_id", "time_hours"]
    rounded = pl.col("time_hours").round(6)
    dump, scored = dump.with_columns(rounded), scored.with_columns(rounded)
    for side, frame in (("dump", dump), ("new", scored)):
        n_dup = frame.height - frame.unique(subset=key_cols).height
        if n_dup:
            raise RuntimeError(
                f"{event}: {n_dup} duplicate (subject, visit, time) keys on "
                f"the {side} side -- an inner join would fan out; refusing "
                "to score"
            )
    joined = dump.join(scored, on=key_cols, how="inner")
    if joined.height != scored.height:
        raise RuntimeError(
            f"{event}: joined {joined.height} of {scored.height} held-out "
            f"rows against the dump ({dump.height} dump rows) -- the row "
            "sets disagree (different shards / landmark_hours / protocol?); "
            "refusing to score a silently reduced overlap"
        )
    return joined


def score_horizon(joined: pl.DataFrame, horizon: float) -> dict[str, Any] | None:
    """Bootstrap AUROC for every scorer present at one horizon.

    All scorers share a single row mask: a row one scorer cannot score
    leaves every scorer, or the columns of a single table row describe
    different samples and no comparison across them is paired. Returns
    ``None`` when the horizon is absent from the dump or has no positives
    left, which is a skip rather than a failure.
    """
    hcol = f"{horizon:g}h"
    if f"y@{hcol}" not in joined.columns:
        return None
    y = joined[f"y@{hcol}"].to_numpy().astype(np.float64)
    sid = joined["subject_id"].to_numpy().astype(int)
    preds = {
        name: joined[f"{name}@{hcol}"].to_numpy().astype(np.float64)
        for name in SCORERS
        if f"{name}@{hcol}" in joined.columns
    }
    mask = ~np.isnan(y)
    for pred in preds.values():
        mask &= ~np.isnan(pred)
    y, sid = y[mask], sid[mask]
    if len(np.unique(y)) < 2:
        return None
    cell: dict[str, Any] = {"n": int(mask.sum()), "n_positive": int(y.sum())}
    for name, pred in preds.items():
        ci = bootstrap_auroc(y, pred[mask], sid)
        cell[name] = None if ci is None else vars(ci)
    return cell


def main() -> None:  # noqa: PLR0915  (argparse + pipeline wiring, not logic)
    """Fit TabICL per core event, then score every scorer on the matched rows."""
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
    parser.add_argument(
        "--dump-rows",
        type=Path,
        default=None,
        help=(
            "write the joined per-row scores here, in the column layout "
            "scripts/alerts_cis.py consumes, so the three scorers get paired "
            "subject-clustered deltas rather than three separate intervals"
        ),
    )
    parser.add_argument(
        "--skip-tabicl",
        action="store_true",
        help=(
            "score hazard and GBM on the rows TabICL would have been scored "
            "on, without fitting it. Minutes instead of hours, and the GBM it "
            "reproduces is a direct check that a previous full run's row set "
            "is the one being matched"
        ),
    )
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
    # Same call, same place, as every other entry point (alerts.py,
    # run_inference.py, concept_attribution.py). Without it the sepsis3
    # concept has no microbiology sidecar and is unobserved EVERYWHERE, so
    # its rows silently collapse to nothing and the event is skipped -- a
    # 6h run that quietly drops the task it was extended to cover. The
    # sidecar root resolves from the data root, so activating once off the
    # held-out dir also covers the train shards.
    names = activate_sidecars(args.held_out_shard_dir)
    logger.info("active sidecars: %s", ", ".join(names) if names else "NONE")
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

    # The training panel exists only to fit TabICL, and it is the larger of
    # the two preps (8 shards against 4, ~600 float columns each). Skipping
    # it under --skip-tabicl is most of what makes that mode cheap enough
    # to run beside another job.
    train = None
    if not args.skip_tabicl:
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

    results: dict[str, Any] = {}
    dumps: list[pl.DataFrame] = []
    for alert in alerts:
        event = alert.name
        held_rows = held.rows.get(event, [])
        rows = [] if train is None else train.rows.get(event, [])
        if not held_rows or (train is not None and not rows):
            continue

        held_feats = held.features["strong"][event]
        existing_ev = existing.filter(pl.col("event") == event)
        # Predict every horizon first, then join ONCE. The join is what
        # defines the row set the scorers share, so joining per horizon
        # would make "hazard, GBM and TabICL saw the same rows" three
        # separate assertions instead of one.
        new_cols = landmark_keys(held_rows)
        fit_s: float = 0.0
        predict_s: dict[float, float] = {}
        if train is not None:
            new_cols, fit_s, predict_s = tabicl_columns(
                event,
                rows,
                train,
                held_feats,
                new_cols,
                source=source,
                offload_mode=args.offload_mode,
            )

        joined = join_matched_rows(existing_ev, new_cols, event)
        for h in HORIZONS:
            cell = score_horizon(joined, h)
            if cell is None:
                logger.warning("%s@%gh has no scoreable rows, skipped", event, h)
                continue
            cell["fit_s"] = fit_s
            cell["predict_s"] = predict_s.get(h)
            results[f"{event}@{h:g}h"] = cell
            logger.info(
                "%s@%gh n=%d %s",
                event,
                h,
                cell["n"],
                " ".join(
                    f"{name}={cell[name]['point_estimate']:.4f}"
                    for name in SCORERS
                    if cell.get(name)
                ),
            )
            args.output_json.write_text(json.dumps(results, indent=2))

        if args.dump_rows is not None:
            dumps.append(joined.select([c for c in joined.columns if _dumped(c)]))

    args.output_json.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.output_json)
    if args.dump_rows is not None and dumps:
        # diagonal: an event whose TabICL fit was skipped contributes no
        # tabicl@ columns, and concat must not refuse the frame over it.
        rows_out = pl.concat(dumps, how="diagonal")
        args.dump_rows.parent.mkdir(parents=True, exist_ok=True)
        rows_out.write_parquet(args.dump_rows)
        logger.info(
            "wrote %s (%d rows); feed it to scripts/alerts_cis.py with "
            "--scorers %s for the paired deltas",
            args.dump_rows,
            rows_out.height,
            " ".join(SCORERS),
        )


if __name__ == "__main__":
    main()
