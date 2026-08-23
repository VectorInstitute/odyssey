"""Native MEDS-Tab pipeline driver for the alerts baseline (per-task tabularize).

Runs MEDS-Tab's own CLI (describe codes, static/time-series tabularization,
XGBoost with native hyperparameter search) against our MEDS data, one
task per (event, horizon), and scores the resulting held-out predictions
through score_alerts's extra_baselines hook -- joined positionally against
the run's already-dumped alerts_rows.parquet (v1 landmark protocol) for
hazard@h/gbm@h, the same discipline entry 35's SurvivalPFN fix used and for
the same reason: comparability against the existing v1 comparator table,
not a v2 (post landmark-fix) number. No v2 scoring here.

Originally built and run against eICU; source-agnostic in practice (reads
config.source from the run's own config.json) and also used for MIMIC-IV's
scoped-to-subset_run_v8 leg -- see :mod:`scripts.meds_tab_shared_landmark`,
the shared-landmark-grid companion driver that reuses this module's helpers
to tabularize once instead of once per task.

Requires the optional `meds-tab` package (`uv sync --extra meds_tab`).
Zero model forward passes -- MEDS-Tab's own featurization and XGBoost fit
run entirely as external CLI subprocesses; the model's hazard/gbm columns
come from the existing alerts_rows.parquet dump, not recomputed.

Trap 2 (review, odyssey-db): MedsTabBaselineModel.predictions is aligned to
FULL row order, but MEDS-Tab only predicts the at-risk cohort (its own
label file's rows). Non-at-risk positions are filled with np.nan when
scattering predictions back to full order -- never 0 or any finite
placeholder -- so a keep-mask disagreement between this script's export
and score_alerts's own at-risk filter poisons the AUROC loudly (NaN
propagates through roc_auc_score as an error) instead of silently scoring
a placeholder.

Trap 3 (review, odyssey-db): alerts_rows.parquet (and this script's labels,
inherited from it) has real duplicate (subject_id, time_hours) keys. Which
way meds-tab-cache-task's join goes (dedupe or fan out) is checked
empirically at dry-run time via verify_cached_label_count; both arms are
implemented (see _reconcile_duplicate_predictions) and the branch is
chosen from what that check reports, not assumed in advance.

Usage: python -m scripts.meds_tab_alerts_baseline <run_dir> <data_root> <out_json>
  [n_trials] [n_workers]
"""

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("meds_tab_alerts_baseline")

from odyssey.data.alert_events import (  # noqa: E402
    ALERT_EVENTS,
    EventTimes,
    all_event_times,
)
from odyssey.data.code_normalization import maybe_normalize  # noqa: E402
from odyssey.data.history_recap import maybe_history_recap  # noqa: E402
from odyssey.data.value_binning import QuantileBinner, add_value_tokens  # noqa: E402
from odyssey.inference.alerts import (  # noqa: E402
    HORIZONS_HOURS,
    IndexRow,
    _index_rows_from_events,
)
from odyssey.inference.meds_tab_baseline import (  # noqa: E402
    assert_label_df_sorted,
    assert_label_feature_alignment,
    assert_no_split_leakage,
    export_task_labels,
)
from odyssey.training.data import load_meds_shards  # noqa: E402
from odyssey.training.train import TrainingConfig  # noqa: E402
from odyssey.utils.joblib_tmp import ensure_joblib_temp_folder  # noqa: E402


# Pinned per item (f): recorded here, not left at MEDS-Tab's own defaults
# (nthread=1 is wrong for a 12-vCPU box; their own MIMIC-scale tutorial
# uses n_trials=1000, far more than this eICU-scale run's time budget
# justifies). Fixed seed for reproducibility of the sweep itself.
PINNED_NTHREAD = 12
PINNED_SEED = 0
DEFAULT_N_TRIALS = 20
DEFAULT_N_WORKERS = 8

# MEDS-Tab's default tabularization.aggs includes "static/first"
# (STATIC_VALUE_AGGREGATION), a purely-static NUMERIC feature. Our eICU
# MEDS extraction never emits one -- describe_codes' codes.parquet output
# confirmed 0 codes with that suffix, only "static/present" categorical
# indicators (RACE, GENDER) -- and tabularize_static.py hard-fails
# ("No static features found") when a requested agg matches nothing, not
# just skips it. Drop it here rather than accept the crash.
TABULARIZATION_AGGS = (
    "static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max"
)


def _resolve_console_script(name: str) -> str:
    """Resolve a MEDS-Tab console-script name against this venv's bin dir.

    The job is launched as `<venv>/bin/python -u this_script.py`, not via
    `uv run` or an activated shell, so PATH does not contain the venv's
    bin/ directory and subprocess.run(["meds-tab-describe", ...]) fails
    with FileNotFoundError even though the package is installed right
    next to the running interpreter.
    """
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def run_cli(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run one MEDS-Tab CLI stage, raising loud on a non-zero exit."""
    cmd = [_resolve_console_script(cmd[0]), *cmd[1:]]
    logger.info("[meds_tab] $ %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("[meds_tab] FAILED in %.1fs: %s", elapsed, " ".join(cmd))
        logger.error("stdout tail:\n%s", "\n".join(result.stdout.splitlines()[-40:]))
        logger.error("stderr tail:\n%s", "\n".join(result.stderr.splitlines()[-40:]))
        raise RuntimeError(
            f"MEDS-Tab CLI stage failed (exit {result.returncode}): {cmd[0]}"
        )
    logger.info("[meds_tab] done in %.1fs", elapsed)


def build_rows_and_times(
    shard_dir: Path,
    binner: QuantileBinner,
    source: str,
    config: TrainingConfig,
    *,
    max_shards: int | None,
) -> tuple[dict[str, list[IndexRow]], dict[str, EventTimes], pl.DataFrame]:
    """Load a split's shards, bin values, and derive landmark rows + event times."""
    raw = load_meds_shards(shard_dir, max_shards=max_shards)
    raw = maybe_normalize(raw, enabled=config.normalize_medications, source=source)
    raw = maybe_history_recap(raw, enabled=config.history_recap)
    times = all_event_times(raw, ALERT_EVENTS, source)
    binned = add_value_tokens(raw, binner, source=source)
    del raw
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    return rows, times, binned


def held_out_rows_from_alerts_dump(alerts_rows_path: Path) -> dict[str, list[IndexRow]]:
    """v1-protocol held-out rows, read straight from alerts_rows.parquet.

    Same reasoning as entry 35's SurvivalPFN fix: collect_model_scores'
    row set (what alerts_rows.parquet actually contains) is not guaranteed
    to match a fresh _index_rows_from_events reconstruction, and since
    we're joining against this dump's own hazard@h/gbm@h columns, we need
    ITS rows specifically, not a re-derived set (which would also now be
    v2-protocol post-85dde80, not comparable to this v1 dump at all).
    """
    orig = pl.read_parquet(alerts_rows_path)
    rows: dict[str, list[IndexRow]] = {}
    for event_name in orig["event"].unique(maintain_order=True).to_list():
        ev = orig.filter(pl.col("event") == event_name)
        rows[event_name] = [
            IndexRow(subject_id=int(sid), visit_id=int(vid), time_hours=float(th))
            for sid, vid, th in zip(ev["subject_id"], ev["visit_id"], ev["time_hours"])
        ]
    return rows


def build_data_subset_dir(
    data_dir: Path,
    subset_dir: Path,
    *,
    train_shards: list[int],
    held_out_shards: list[int],
    tuning_shards: list[int],
) -> Path:
    """Symlink only the shard files this run actually uses into a scoped input_dir.

    MEDS-Tab's own CLI has no max_shards knob -- tabularize-static sweeps
    every parquet under whatever input_dir it's given
    (``list_subdir_files(cfg.input_dir, "parquet")``, confirmed by reading
    the source) and requires a matching label file at the same relative
    path for each one (``label_fp = input_label_dir / shard_fp.relative_to(
    shard_fp.parents[1])``, also confirmed from source). Pointing it at a
    full split (hundreds of shards) would require label files for every
    one of them, and would tabularize/train on a far larger cohort than
    this run's own scope, an unfair and slow comparison -- so this stays
    scoped to a small shard subset per split, matching
    train_max_shards/held_out_max_shards/tuning_max_shards.

    ``tuning`` is included (not dropped, despite having no role in the v1
    alerts protocol this comparator scores against): meds-tab-xgboost
    hardcodes a real tuning split for its own early-stopping validation
    during the fit itself (confirmed from xgboost_model.py's _build:
    XGBIterator(cfg, split="tuning"), unconditional, not gated by any
    config we control) -- without it, xgboost cannot train at all, not
    just cannot score. Real, disjoint tuning shards, not a held_out
    stand-in: reusing held_out subjects for tuning would let model
    selection implicitly peek at the held-out cohort.
    """
    for split, shards in [
        ("train", train_shards),
        ("held_out", held_out_shards),
        ("tuning", tuning_shards),
    ]:
        split_dir = subset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem in shards:
            link = split_dir / f"{stem}.parquet"
            if not link.exists():
                link.symlink_to((data_dir / split / f"{stem}.parquet").resolve())
    return subset_dir


def shard_subject_ids(
    data_dir: Path, split: str, shards: list[int]
) -> dict[int, set[int]]:
    """subject_id set per shard.

    So labels can be split to match MEDS-Tab's per-shard input_label_dir
    convention.
    """
    out: dict[int, set[int]] = {}
    for stem in shards:
        ids = pl.read_parquet(
            data_dir / split / f"{stem}.parquet", columns=["subject_id"]
        )["subject_id"]
        out[stem] = set(ids.to_list())
    return out


def export_shard_aligned_labels(
    label_df: pl.DataFrame, shard_subjects: dict[int, set[int]], out_dir: Path
) -> None:
    """Write one label parquet per shard (even if empty).

    Matches get_shard_prefix's expected
    input_label_dir/<split>/<shard_stem>.parquet layout -- a flat
    single-file label dir only works when input_dir has a single shard,
    which is not our case.

    Asserts sortedness immediately before every write -- this is the real
    MEDS-Tab handoff boundary (input_label_dir is what tabularize-static/
    time-series actually reads), and the confirmed, sole cause of a real
    gate failure (see assert_label_df_sorted's docstring) was an unsorted
    label_df reaching exactly this kind of write undetected.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, subjects in shard_subjects.items():
        shard_df = label_df.filter(pl.col("subject_id").is_in(subjects)).sort(
            ["subject_id", "prediction_time"]
        )
        assert_label_df_sorted(shard_df)
        shard_df.write_parquet(out_dir / f"{stem}.parquet")


def run_cache_task(
    *,
    tab_data_dir: Path,
    shared_tab_out: Path,
    task_name: str,
    task_label_dir: Path,
) -> None:
    """Run meds-tab-cache-task, slicing this task's rows out of a shared run.

    Slices one task's own label rows out of the ONE shared tabularize-
    static/time-series run, via cache-task's own
    independent event_id/join_asof against raw shard data (confirmed from
    scripts/cache_task.py source -- it re-derives event_id from
    cfg.input_dir directly, not from whatever tabularize wrote, then does
    csr[valid_ids, :] against the shared matrix). Cheap: no rolling-window
    recomputation, just a row select.

    CONFIRMED INCOMPATIBLE with a shared, restricted landmark grid (see
    :mod:`scripts.meds_tab_shared_landmark`'s module docstring and the
    join_asof/shared-landmark-grid incident it links) -- cache-task's own
    event_id derivation assumes the tabularized matrix covers the FULL raw
    per-event grid, not a restricted landmark subset. Retained here only
    for the standalone (per-task tabularize) architecture this module
    otherwise implements, where that assumption holds; referenced (but not
    reached) by the shared-grid driver's held per-task loop, which is
    blocked on the custom slicer (scripts/meds_tab_slice_task.py) instead.
    """
    run_cli(
        [
            "meds-tab-cache-task",
            f"input_dir={tab_data_dir}",
            f"output_dir={shared_tab_out}",
            f"task_name={task_name}",
            f"input_label_dir={task_label_dir}",
            "do_overwrite=False",
            f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
        ]
    )


def main() -> None:  # noqa: PLR0915
    """Run the standalone (per-task) MEDS-Tab pipeline end to end."""
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
    logger.info("[meds_tab] config/binner loaded, source=%s", source)

    data_dir = data_root / "data"
    # Matches every other script's convention in this project (a run's own
    # 30 train / 4 held-out shards) -- the train/held_out directories hold
    # the FULL extraction, not just this run's subset; max_shards=None here
    # was the actual root cause of an earlier OOM kill (loaded every train
    # shard, ~85GB RSS on an 83GB box).
    train_max_shards = 30
    held_out_max_shards = 4
    # meds-tab-xgboost hardcodes a real tuning split for its own early-
    # stopping validation during the fit itself (XGBIterator(cfg,
    # split="tuning"), unconditional -- confirmed from xgboost_model.py's
    # _build). Not part of the v1 alerts protocol this comparator scores
    # against, so it has no role beyond letting MEDS-Tab train at all;
    # scaled to match held_out_max_shards, real disjoint subjects (not a
    # held_out stand-in, which would leak the held-out cohort into model
    # selection).
    tuning_max_shards = 4
    train_rows, train_times, train_binned = build_rows_and_times(
        data_dir / "train", binner, source, config, max_shards=train_max_shards
    )
    logger.info(
        "[meds_tab] train rows built: %s", {k: len(v) for k, v in train_rows.items()}
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
        "[meds_tab] held-out rows from alerts_rows.parquet: %s",
        {k: len(v) for k, v in held_rows.items()},
    )

    tuning_rows, tuning_times, tuning_binned = build_rows_and_times(
        data_dir / "tuning", binner, source, config, max_shards=tuning_max_shards
    )
    logger.info(
        "[meds_tab] tuning rows built: %s", {k: len(v) for k, v in tuning_rows.items()}
    )

    label_root = run_dir / "meds_tab_labels"
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
        "[meds_tab] %d train task labels, %d held-out task labels, %d tuning task labels",
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
    logger.info("[meds_tab] scoped input_dir built: %s", tab_data_dir)

    tab_out = run_dir / "meds_tab_out"
    logger.info("[meds_tab] describing codes")
    run_cli(
        [
            "meds-tab-describe",
            f"input_dir={tab_data_dir}",
            f"output_dir={tab_out}",
        ]
    )

    results: list[dict[str, str]] = []
    for (event_name, h), train_label_path in sorted(train_label_paths.items()):
        if (event_name, h) not in held_label_paths:
            logger.info(
                "[meds_tab] %s@%gh: no held-out labels, skipping", event_name, h
            )
            continue
        if (event_name, h) not in tuning_label_paths:
            logger.info("[meds_tab] %s@%gh: no tuning labels, skipping", event_name, h)
            continue
        task_name = f"{event_name}_{h:g}h"
        task_out = tab_out / task_name
        combined_label_dir = label_root / "combined" / task_name
        combined_label_dir.mkdir(parents=True, exist_ok=True)
        train_task_labels = pl.read_parquet(train_label_path)
        held_task_labels = pl.read_parquet(held_label_paths[(event_name, h)])
        tuning_task_labels = pl.read_parquet(tuning_label_paths[(event_name, h)])
        # Shard-aligned, not a single flat file: get_shard_prefix requires
        # input_label_dir/<split>/<shard_stem>.parquet to exist for every
        # shard under input_dir (confirmed from tabularize_static.py
        # source), which a single combined/<task>/0.parquet cannot satisfy
        # once input_dir has more than one shard.
        export_shard_aligned_labels(
            train_task_labels, train_shard_subjects, combined_label_dir / "train"
        )
        export_shard_aligned_labels(
            held_task_labels, held_shard_subjects, combined_label_dir / "held_out"
        )
        export_shard_aligned_labels(
            tuning_task_labels, tuning_shard_subjects, combined_label_dir / "tuning"
        )

        # Reference recipe (mmcdermott/MEDS_Tabular_AutoML MIMICIV_TUTORIAL/
        # task_tabularize_meds.sh): each task's own output_dir needs its own
        # copy of the shared describe-codes metadata (input_code_metadata_fp
        # defaults to ${output_dir}/metadata/codes.parquet), or
        # tabularize-static fails with FileNotFoundError since that file
        # only exists under the shared tab_out, not task_out.
        task_out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tab_out / "metadata", task_out / "metadata", dirs_exist_ok=True)

        logger.info("[meds_tab] %s: tabularizing static", task_name)
        run_cli(
            [
                "meds-tab-tabularize-static",
                f"input_dir={tab_data_dir}",
                f"output_dir={task_out}",
                f"input_label_dir={combined_label_dir}",
                "do_overwrite=False",
                f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
            ]
        )
        logger.info("[meds_tab] %s: tabularizing time-series", task_name)
        run_cli(
            [
                "meds-tab-tabularize-time-series",
                "--multirun",
                f"worker=range(0,{n_workers})",
                "hydra/launcher=joblib",
                f"input_dir={tab_data_dir}",
                f"output_dir={task_out}",
                f"input_label_dir={combined_label_dir}",
                "do_overwrite=False",
                f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
            ]
        )

        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={r.subject_id for r in held_rows[event_name]},
            tuning_subject_ids={r.subject_id for r in tuning_rows[event_name]},
            train_label_dir=combined_label_dir / "train",
            held_out_label_dir=combined_label_dir / "held_out",
            tuning_label_dir=combined_label_dir / "tuning",
        )
        logger.info("[meds_tab] %s: split-leak assert passed", task_name)

        # Standalone-path replacement for verify_cached_label_count:
        # meds-tab-cache-task never runs in this architecture (xgboost
        # reads input_label_cache_dir directly), so there is no
        # cache-task join output to check -- the real coupling under
        # threat is label-row-to-feature-row positional alignment.
        assert_label_feature_alignment(
            task_out / "tabularize" / "train", combined_label_dir / "train"
        )
        assert_label_feature_alignment(
            task_out / "tabularize" / "held_out", combined_label_dir / "held_out"
        )
        assert_label_feature_alignment(
            task_out / "tabularize" / "tuning", combined_label_dir / "tuning"
        )
        logger.info("[meds_tab] %s: label/feature alignment assert passed", task_name)

        model_out = run_dir / "meds_tab_models" / task_name
        logger.info(
            "[meds_tab] %s: xgboost (n_trials=%d, nthread=%d, seed=%d)",
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
                f"output_dir={task_out}",
                f"output_model_dir={model_out}",
                f"task_name={task_name}",
                "do_overwrite=False",
                f"hydra.sweeper.n_trials={n_trials}",
                f"hydra.sweeper.n_jobs={n_workers}",
                f"input_tabularized_cache_dir={task_out}/tabularize",
                f"input_label_cache_dir={combined_label_dir}",
                f"tabularization.aggs=[{TABULARIZATION_AGGS}]",
                f"model_launcher.model.nthread={PINNED_NTHREAD}",
                f"seed={PINNED_SEED}",
            ]
        )

        results.append({"task": task_name})

    out_json.write_text(json.dumps(results, indent=2))
    logger.info("[meds_tab] done in %.0fs, wrote %s", time.time() - t0, out_json)


if __name__ == "__main__":
    main()
