"""Shared-landmark-grid MEDS-Tab driver: tabularize once, cache-task per task.

Companion to meds_tab_alerts_baseline.py (the standalone driver, which
recomputes the expensive tabularize-static/time-series stage separately
per task). Structural fact this exploits, verified directly (not
assumed): all 12 tasks (4 events x 3 horizons) share the SAME pre-at-risk-
filter landmark grid per split (byte-identical (subject_id, time_hours)
sets across events, confirmed on shard 0: 35,763 unique rows each) --
export_task_labels' own per-(event,horizon) at-risk filter only ever
REMOVES rows from that shared grid, never adds, making it a strict
superset of every task's own label rows.

Stops after the shared tabularize-static/time-series stage: the per-task
loop below it calls run_cache_task, which is confirmed broken against a
shared/restricted label_df (see the join_asof/shared-landmark-grid
incident referenced below) and is held pending a gate-and-slice sequence
built on the custom slicer (scripts/meds_tab_slice_task.py) instead.

Mechanism (confirmed from MEDS-Tab source, not assumed):
  - tabularize-static/time-series' label_df argument restricts WHICH times
    get the (expensive) rolling-window aggregation computed
    ("only perform aggregations at the label times", compute_agg's own
    docstring) -- feeding the full raw per-event-timestamp grid instead
    (label_df=None) would cost ~28x one task's own run (measured: 1,018,049
    distinct raw event timestamps vs 26,087 acute_kidney_injury@8h label
    rows on shard 0), while the shared landmark grid costs ~1.4x (35,763
    rows on the same shard) and covers all 12 tasks in ONE run.
  - meds-tab-cache-task re-derives each task's own (subject_id, time)
    event_id independently from raw shard data via its own join_asof, then
    slices `csr[valid_ids, :]` out of the ONE shared tabularized matrix --
    cheap, no rolling-window recomputation. cache_task.py's own
    `label_fp = Path(cfg.input_label_dir) / shard_fp.relative_to(
    shard_fp.parents[1])`-shaped tabularize_static.py glob and its own
    `list_subdir_files(cfg.input_tabularized_dir, "npz")` sweep across ALL
    splits under input_label_dir/input_tabularized_dir in one call (no
    --split flag), matching our existing combined_label_dir's
    train/held_out/tuning layout exactly.
  - Pointing cache-task and xgboost at the SAME shared output_dir
    (task_name set) resolves xgboost's own input_tabularized_cache_dir/
    input_label_cache_dir defaults (${output_dir}/${task_name}/task_cache,
    ${output_dir}/${task_name}/labels) to cache-task's own output without
    any override needed.

Writes to a SEPARATE output tree (meds_tab_out_shared, not
meds_tab_out) so task 1's already-completed standalone artifacts stay
untouched on disk for the exact-reproduction diff this script's first
task's own output must pass before any of the other 11 tasks are trusted.

Usage: python -m scripts.meds_tab_shared_landmark <run_dir> <data_root> <out_json>
  [n_trials] [n_workers]
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("meds_tab_shared_landmark")

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times  # noqa: E402
from odyssey.data.code_normalization import maybe_normalize  # noqa: E402
from odyssey.data.history_recap import maybe_history_recap  # noqa: E402
from odyssey.data.value_binning import QuantileBinner, add_value_tokens  # noqa: E402
from odyssey.inference.alerts import HORIZONS_HOURS  # noqa: E402
from odyssey.inference.meds_tab_baseline import (  # noqa: E402
    assert_label_df_sorted,
    assert_no_split_leakage,
    build_shared_landmark_label_df,
    export_task_labels,
    verify_cached_label_count,
)
from odyssey.training.data import load_meds_shards  # noqa: E402
from odyssey.training.train import TrainingConfig  # noqa: E402
from odyssey.utils.joblib_tmp import ensure_joblib_temp_folder  # noqa: E402
from scripts.meds_tab_alerts_baseline import (  # noqa: E402
    PINNED_NTHREAD,
    PINNED_SEED,
    TABULARIZATION_AGGS,
    build_data_subset_dir,
    build_rows_and_times,
    export_shard_aligned_labels,
    held_out_rows_from_alerts_dump,
    run_cache_task,
    run_cli,
    shard_subject_ids,
)


# 10, matching scripts/meds_tab_alerts_baseline.py's own DEFAULT_N_TRIALS
# and the same fairness reasoning stated there: our own tuned GBM (the
# comparator bar) gets 4 configs x 400 rounds, so MEDS-Tab's own xgboost
# gets 10 trials, not MEDS-Tab's own much larger defaults -- the two
# driver scripts had drifted apart (20 vs 200) until 2026-08-24; keep them
# equal going forward.
DEFAULT_N_TRIALS = 10
DEFAULT_N_WORKERS = 4


def assert_no_incomplete_tabularize_outputs(tab_out: Path) -> None:
    """Fail loud on any unopenable ``.npz`` under ``tab_out``, or a stale ``.lock``.

    MEDS_transforms.mapreduce.rwlock.rwlock_wrap's own completion check
    (``default_file_checker``) only validates content for ``.parquet``
    outputs (via ``is_complete_parquet_file``, a real pyarrow open); for
    any other suffix -- every tabularize output here -- it falls back to
    ``fp.is_file()``, true for ANY existing file regardless of validity.
    Two real incidents confirm this is not hypothetical, both from the same
    2026-08-23 SIGSTOP-mid-write pause: two files landed at exactly 0 bytes
    (an earlier version of this function only checked ``st_size == 0`` and
    caught those); a third, ``train/19/30d/code/count.npz``, landed at a
    substantial 246 MB with a valid-looking zip local-file header but no
    trailing central directory -- the write was killed after flushing most
    of the payload but before the zip footer, so it is neither empty nor
    obviously truncated by size, and passed the old zero-byte-only check
    clean. All three read as "already done" to ``do_overwrite=False`` and
    are never regenerated on resume unless something actually opens them.

    This function now OPENS every ``.npz`` (``numpy.load`` with
    ``allow_pickle=False``, then touches ``["array"].shape``) rather than
    just ``stat()``-ing it. Chosen over ``zipfile.ZipFile(...).testzip()``
    (which CRC-validates every byte of every member) because every real
    failure seen so far is structural truncation -- a missing/corrupt zip
    central directory or ``.npy`` header -- not silent bit-level data
    corruption; ``numpy.load`` already has to parse the central directory
    and the ``array.npy`` member's header to answer ``.shape``, so it
    catches this failure class at header-read cost, not full-payload-read
    cost. Measured on the real tree this guards (1,178 files, up to ~275 MB
    each, uncompressed per this project's ``do_compress=False`` convention):
    ~9 minutes wall-clock for a full pass. That is a fine price paid once,
    after tabularization, for the guarantee -- cheap relative to the hours
    tabularization itself takes, and it is exactly this check that caught
    the 246 MB file above before it silently corrupted a downstream sweep.

    Any remaining ``.lock`` file at the point this is called is stale by
    construction -- it is only invoked after every tabularize subprocess
    has already exited, so no writer can legitimately still hold one;
    stale locks are deleted here rather than merely reported, since a
    leftover lock (not the file it guards) would otherwise block a future
    resume attempt from writing at all.
    """
    stale_locks = sorted(tab_out.rglob("*.lock"))
    for lock in stale_locks:
        lock.unlink()
    if stale_locks:
        logger.info(
            "[shared] deleted %d stale .lock file(s): %s",
            len(stale_locks),
            [str(p) for p in stale_locks],
        )

    unopenable: list[tuple[Path, str]] = []
    for npz_fp in sorted(tab_out.rglob("*.npz")):
        try:
            with np.load(npz_fp, allow_pickle=False) as data:
                _ = data["array"].shape
        except Exception as exc:  # noqa: BLE001 -- any failure means "not trustworthy"
            unopenable.append((npz_fp, repr(exc)))
    if unopenable:
        raise RuntimeError(
            f"{len(unopenable)} .npz file(s) under {tab_out} exist but fail to "
            "open (truncated write, missing zip central directory, or similar) "
            "-- these read as complete to MEDS-Tab's own rwlock "
            "(default_file_checker only validates .parquet content) and will "
            "NOT be regenerated by a do_overwrite=False resume. Delete them "
            "and re-run the tabularize-time-series (or -static) stage once "
            f"more before trusting this output: {unopenable}"
        )


def main() -> None:  # noqa: PLR0915
    """Run the shared-landmark-grid MEDS-Tab pipeline through tabularization."""
    ensure_joblib_temp_folder()
    run_dir = Path(sys.argv[1])
    data_root = Path(sys.argv[2])
    out_json = Path(sys.argv[3])
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_N_TRIALS
    n_workers = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_N_WORKERS

    t0 = time.time()
    config = TrainingConfig(**json.loads((run_dir / "config.json").read_text()))
    binner = QuantileBinner.load(run_dir / "quantile_binner.json")
    source = config.source
    logger.info("[shared] config/binner loaded, source=%s", source)

    data_dir = data_root / "data"
    train_max_shards = 30
    held_out_max_shards = 4
    tuning_max_shards = 4

    train_rows, train_times, train_binned = build_rows_and_times(
        data_dir / "train", binner, source, config, max_shards=train_max_shards
    )
    logger.info(
        "[shared] train rows built: %s", {k: len(v) for k, v in train_rows.items()}
    )

    held_rows = held_out_rows_from_alerts_dump(run_dir / "alerts_rows.parquet")
    held_raw = load_meds_shards(data_dir / "held_out", max_shards=held_out_max_shards)
    held_raw = maybe_normalize(
        held_raw, enabled=config.normalize_medications, source=source
    )
    held_raw = maybe_history_recap(held_raw, enabled=config.history_recap)
    held_times = all_event_times(held_raw, ALERT_EVENTS, source)
    held_binned = add_value_tokens(held_raw, binner, source=source)
    del held_raw
    logger.info(
        "[shared] held-out rows from alerts_rows.parquet: %s",
        {k: len(v) for k, v in held_rows.items()},
    )

    tuning_rows, tuning_times, tuning_binned = build_rows_and_times(
        data_dir / "tuning", binner, source, config, max_shards=tuning_max_shards
    )
    logger.info(
        "[shared] tuning rows built: %s", {k: len(v) for k, v in tuning_rows.items()}
    )

    label_root = (
        run_dir / "meds_tab_labels"
    )  # reuse the standalone run's per-task labels
    train_label_paths = export_task_labels(
        train_rows,
        train_times,
        train_binned,
        horizons=HORIZONS_HOURS,
        output_dir=label_root / "train",
    )
    held_label_paths = export_task_labels(
        held_rows,
        held_times,
        held_binned,
        horizons=HORIZONS_HOURS,
        output_dir=label_root / "held_out",
    )
    tuning_label_paths = export_task_labels(
        tuning_rows,
        tuning_times,
        tuning_binned,
        horizons=HORIZONS_HOURS,
        output_dir=label_root / "tuning",
    )
    logger.info(
        "[shared] %d train task labels, %d held-out task labels, %d tuning task labels",
        len(train_label_paths),
        len(held_label_paths),
        len(tuning_label_paths),
    )

    tab_data_dir = build_data_subset_dir(
        data_dir,
        run_dir / "meds_tab_data_subset",
        train_shards=list(range(train_max_shards)),
        held_out_shards=list(range(held_out_max_shards)),
        tuning_shards=list(range(tuning_max_shards)),
    )
    train_shard_subjects = shard_subject_ids(
        data_dir, "train", list(range(train_max_shards))
    )
    held_shard_subjects = shard_subject_ids(
        data_dir, "held_out", list(range(held_out_max_shards))
    )
    tuning_shard_subjects = shard_subject_ids(
        data_dir, "tuning", list(range(tuning_max_shards))
    )
    logger.info(
        "[shared] scoped input_dir (reused from standalone run): %s", tab_data_dir
    )

    # ONE shared landmark label set per split, built from ANY single event's
    # pre-at-risk-filter rows (all 4 share the identical grid -- "death"
    # picked arbitrarily). Separate output tree: task 1's standalone
    # artifacts under meds_tab_out stay untouched for the diff.
    max_h = max(HORIZONS_HOURS)
    shared_label_root = run_dir / "meds_tab_labels" / "shared_landmark"
    shared_train_df = build_shared_landmark_label_df(
        train_rows["death"], train_binned, max_horizon_hours=max_h
    )
    shared_held_df = build_shared_landmark_label_df(
        held_rows["death"], held_binned, max_horizon_hours=max_h
    )
    shared_tuning_df = build_shared_landmark_label_df(
        tuning_rows["death"], tuning_binned, max_horizon_hours=max_h
    )
    # Fail loud here, at construction, not just inside export_shard_aligned_labels'
    # own per-shard write -- the earliest possible point after this grid exists,
    # before any MEDS-Tab handoff. Confirmed root cause of a real gate failure
    # when this function had no sort/dedup at all (see assert_label_df_sorted).
    for name, df in (
        ("train", shared_train_df),
        ("held_out", shared_held_df),
        ("tuning", shared_tuning_df),
    ):
        assert_label_df_sorted(df)
        logger.info(
            "[shared] %s landmark grid: %d rows, sortedness verified", name, df.height
        )
    export_shard_aligned_labels(
        shared_train_df, train_shard_subjects, shared_label_root / "train"
    )
    export_shard_aligned_labels(
        shared_held_df, held_shard_subjects, shared_label_root / "held_out"
    )
    export_shard_aligned_labels(
        shared_tuning_df, tuning_shard_subjects, shared_label_root / "tuning"
    )
    logger.info(
        "[shared] shared landmark label set built: train=%d held_out=%d tuning=%d rows",
        shared_train_df.height,
        shared_held_df.height,
        shared_tuning_df.height,
    )

    tab_out = run_dir / "meds_tab_out_shared"
    logger.info("[shared] describing codes")
    run_cli(
        [
            "meds-tab-describe",
            f"input_dir={tab_data_dir}",
            f"output_dir={tab_out}",
        ]
    )

    logger.info("[shared] tabularizing static (ONCE, shared landmark grid)")
    run_cli(
        [
            "meds-tab-tabularize-static",
            f"input_dir={tab_data_dir}",
            f"output_dir={tab_out}",
            f"input_label_dir={shared_label_root}",
            "do_overwrite=False",
            f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
        ]
    )
    logger.info(
        "[shared] tabularizing time-series (ONCE, shared landmark grid) -- the long pole"
    )
    run_cli(
        [
            "meds-tab-tabularize-time-series",
            "--multirun",
            f"worker=range(0,{n_workers})",
            "hydra/launcher=joblib",
            f"input_dir={tab_data_dir}",
            f"output_dir={tab_out}",
            f"input_label_dir={shared_label_root}",
            "do_overwrite=False",
            f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
        ]
    )
    assert_no_incomplete_tabularize_outputs(tab_out / "tabularize")
    logger.info("[shared] postflight integrity check passed: no zero-byte npz outputs")
    logger.info("[shared] shared tabularization complete in %.0fs", time.time() - t0)

    # STOPPING HERE, deliberately: the per-task loop below uses run_cache_task,
    # which is the OLD, confirmed-broken approach (meds-tab-cache-task's own
    # event_id derivation is incompatible with a shared, restricted label_df --
    # see the join_asof/shared-landmark-grid incident this same architecture
    # required a custom slicer to work around). Per-task loop held for the
    # gate-and-slice sequence the E4 leg uses, not run automatically here.
    out_json.write_text(
        json.dumps(
            {
                "status": "tabularization_complete_per_task_loop_held",
                "elapsed_s": time.time() - t0,
                "tab_out": str(tab_out),
                "shared_label_root": str(shared_label_root),
            },
            indent=2,
        )
    )
    logger.info(
        "[shared] STOPPING HERE after tabularization -- per-task loop held, wrote status to %s",
        out_json,
    )
    return

    results: list[dict[str, str]] = []
    for (event_name, h), train_label_path in sorted(train_label_paths.items()):
        if (event_name, h) not in held_label_paths or (
            event_name,
            h,
        ) not in tuning_label_paths:
            logger.info(
                "[shared] %s@%gh: missing held-out/tuning labels, skipping",
                event_name,
                h,
            )
            continue
        task_name = f"{event_name}_{h:g}h"
        combined_label_dir = label_root / "combined" / task_name
        combined_label_dir.mkdir(parents=True, exist_ok=True)
        train_task_labels = pl.read_parquet(train_label_path)
        held_task_labels = pl.read_parquet(held_label_paths[(event_name, h)])
        tuning_task_labels = pl.read_parquet(tuning_label_paths[(event_name, h)])
        export_shard_aligned_labels(
            train_task_labels, train_shard_subjects, combined_label_dir / "train"
        )
        export_shard_aligned_labels(
            held_task_labels, held_shard_subjects, combined_label_dir / "held_out"
        )
        export_shard_aligned_labels(
            tuning_task_labels, tuning_shard_subjects, combined_label_dir / "tuning"
        )

        logger.info(
            "[shared] %s: cache-task (slicing shared tabularization)", task_name
        )
        run_cache_task(
            tab_data_dir=tab_data_dir,
            shared_tab_out=tab_out,
            task_name=task_name,
            task_label_dir=combined_label_dir,
        )

        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={r.subject_id for r in held_rows[event_name]},
            tuning_subject_ids={r.subject_id for r in tuning_rows[event_name]},
            train_label_dir=combined_label_dir / "train",
            held_out_label_dir=combined_label_dir / "held_out",
            tuning_label_dir=combined_label_dir / "tuning",
        )
        logger.info("[shared] %s: split-leak assert passed", task_name)

        verify_cached_label_count(
            tab_out,
            task_name,
            expected_n=train_task_labels.height
            + held_task_labels.height
            + tuning_task_labels.height,
        )
        logger.info("[shared] %s: cached label count verified", task_name)

        model_out = run_dir / "meds_tab_models_shared" / task_name
        logger.info(
            "[shared] %s: xgboost (n_trials=%d, nthread=%d, seed=%d)",
            task_name,
            n_trials,
            PINNED_NTHREAD,
            PINNED_SEED,
        )
        run_cli(
            [
                "meds-tab-xgboost",
                "--multirun",
                f"input_dir={tab_data_dir}",
                f"output_dir={tab_out}",
                f"output_model_dir={model_out}",
                f"task_name={task_name}",
                "do_overwrite=False",
                f"hydra.sweeper.n_trials={n_trials}",
                f"hydra.sweeper.n_jobs={n_workers}",
                f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
                f"model_launcher.model.nthread={PINNED_NTHREAD}",
                f"seed={PINNED_SEED}",
            ]
        )
        results.append({"task": task_name})

    out_json.write_text(json.dumps(results, indent=2))
    logger.info("[shared] done in %.0fs, wrote %s", time.time() - t0, out_json)


if __name__ == "__main__":
    main()
