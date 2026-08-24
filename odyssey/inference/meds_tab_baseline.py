"""MEDS-Tab: a native-pipeline comparator, not a reimplementation.

Alongside the in-process baseline families (:mod:`odyssey.inference.tabicl_baseline`,
:mod:`odyssey.inference.ebm_baseline`,
:mod:`odyssey.inference.survivalpfn_baseline`), this module wraps
`MEDS-Tab <https://github.com/mmcdermott/MEDS_Tabular_AutoML>`_ (McDermott et al.),
the field-standard tabularization-plus-XGBoost protocol for generic MEDS
datasets. Unlike the others, this is not a fit-in-process/predict-in-process
model: MEDS-Tab is a five-stage CLI pipeline (describe codes, static and
time-series tabularization, task-specific caching, XGBoost training with
native hyperparameter search) that runs against our MEDS data on disk and
writes predictions back to disk. The value of using it at all is running its
*own* featurization and tuning protocol, not reimplementing it -- so this
module's job is the glue on both ends: exporting our landmark rows as
MEDS-Tab's own task-label format, and reading its predictions back in a
shape :func:`~odyssey.inference.alerts.score_alerts` can score identically
to every other baseline.

Optional dependency, on PyPI (``uv sync --extra meds_tab``, no direct-
reference/hatchling wrinkle the way :mod:`survivalpfn_baseline` needed).
Nothing in this module requires ``meds-tab`` to be installed except the
CLI invocation itself (a separate driver script, not this module -- the
label export and prediction-wrapper logic here are plain MEDS/polars code).

Schema note (checked directly against a real run, not assumed): MEDS-Tab's
own file-discovery (``list_subdir_files``) recursively sweeps every parquet
file under whatever ``input_dir`` is given, with no ``data/`` scoping built
in -- pointing it at a MEDS root that also has a populated ``metadata/``
directory (ours does, e.g. ``subject_splits.parquet``, ``codes.parquet``)
sweeps those in as if they were event shards and crashes. ``input_dir``
must point at the ``data/`` subdirectory specifically, not the dataset
root.

Split note (checked against the installed package's own config, not
assumed): ``meds-tab-xgboost``'s default ``prediction_splits: [held_out,
tuning]`` and the fact that ``meds-tab-describe``'s cache output preserves
our ``train/tuning/held_out`` subdirectory structure exactly are strong
evidence split membership is read from that directory layout natively, not
re-derived from ``subject_splits.parquet`` at tabularization/fit time --
verified explicitly for a real run by
:func:`assert_no_split_leakage`, not just assumed
from the config read.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from odyssey.data.alert_events import EventTimes, origin_hours
from odyssey.data.sequences import BIRTH_CODE
from odyssey.inference.alerts import IndexRow, outcome_at_horizon


logger = logging.getLogger(__name__)


def _prediction_times(
    rows: Sequence[IndexRow], events_binned: pl.DataFrame, *, max_horizon_hours: float
) -> list[datetime]:
    """Absolute ``prediction_time`` per row, checked against the subject's event span.

    Our landmark rows carry ``time_hours``, hours since each subject's own
    sequence origin (:func:`odyssey.data.alert_events.origin_hours`);
    MEDS-Tab's task-label schema wants an absolute timestamp. Correctness
    here rests on construction, not a round trip: ``pred_time = origin +
    timedelta(hours=time_hours)`` uses the exact same
    ``origin_hours(events_binned)`` call every landmark row's ``time_hours``
    was itself built from (a pure, deterministic groupby-min over the same
    input frame), so both sides necessarily agree on which origin is being
    used. Re-deriving ``time_hours`` from ``pred_time`` by subtracting that
    *same* origin back out is not a useful check on its own: a wrong or
    shifted origin round-trips perfectly (the error cancels both ways), so
    it cannot catch what it would need to catch.

    What actually bites: asserting ``pred_time`` falls within the subject's
    own observed event span (at or after their first timed event, at or
    before their last timed event plus ``max_horizon_hours``) -- a wrong
    origin lands outside that window immediately, independent of the
    origin computation itself.
    """
    origins = origin_hours(events_binned)
    origin_map = dict(
        zip(origins["subject_id"].to_list(), origins["_origin"].to_list())
    )
    spans = (
        events_binned.filter(
            pl.col("time").is_not_null() & (pl.col("code") != BIRTH_CODE)
        )
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_first"), pl.col("time").max().alias("_last"))
    )
    first_map = dict(zip(spans["subject_id"].to_list(), spans["_first"].to_list()))
    last_map = dict(zip(spans["subject_id"].to_list(), spans["_last"].to_list()))
    times: list[datetime] = []
    for r in rows:
        origin = origin_map.get(r.subject_id)
        if origin is None:
            raise ValueError(
                f"no sequence origin found for subject_id={r.subject_id} "
                "(no timed non-birth event in events_binned)"
            )
        pred_time = origin + timedelta(hours=r.time_hours)
        first = first_map[r.subject_id]
        upper = last_map[r.subject_id] + timedelta(hours=max_horizon_hours)
        if pred_time < first or pred_time > upper:
            raise AssertionError(
                f"prediction_time outside the subject's own event span for "
                f"subject_id={r.subject_id}: pred_time={pred_time}, expected "
                f"within [{first}, {upper}] (time_hours={r.time_hours}, "
                f"max_horizon_hours={max_horizon_hours})"
            )
        times.append(pred_time)
    return times


def _sort_and_or_aggregate_labels(
    label_df: pl.DataFrame, *, log_context: str
) -> pl.DataFrame:
    """Canonicalize a MEDS-Tab label_df before it is written or handed to MEDS-Tab.

    Two invariants enforced here, both confirmed root causes of real bugs
    tonight, not hypothetical:

    1. Deterministic order. Row order traces back (through whichever
       caller-side ``IndexRow`` construction) to
       :func:`~odyssey.inference.alerts._index_rows_from_events`'s
       ``group_by`` with no ``maintain_order=True`` -- polars'
       multi-threaded hash-partition group_by does not guarantee row
       order across runs on identical input (confirmed: two exports of
       the same input produced byte-different files, content-identical
       only once both were sorted by this key).
    2. OR-aggregated collisions. Rows colliding on
       ``(subject_id, prediction_time)`` (eICU: concurrent/transferred
       ICU unit stays within one hospitalization, confirmed via a
       sampled overlap check, not distinct episodes) collapse to one row,
       true if any of them is -- not an arbitrary pick among them, which
       is what silently happened before this existed.

    Both matter beyond just this function's own output looking right:
    MEDS-Tab's own ``get_rolling_window_indicies`` does
    ``label_df.join_asof(event_df, by="subject_id", on="time")``, and
    polars' ``join_asof`` silently produces wrong window boundaries (not
    an error) when its input isn't sorted by ``(by, on)`` -- confirmed by
    reading ``MEDS_tabular_automl.generate_summarized_reps`` source
    directly. An unsorted ``label_df`` handed to a MEDS-Tab tabularize
    call doesn't just look wrong here, it corrupts every downstream
    aggregated feature value for that shard.
    """
    n_supplied = label_df.height
    label_df = (
        label_df.group_by(["subject_id", "prediction_time"])
        .agg(pl.col("boolean_value").any())
        .sort(["subject_id", "prediction_time"])
    )
    n_collapsed = n_supplied - label_df.height
    if n_collapsed:
        logger.info(
            "[meds_tab] %s: %d rows OR-collapsed across colliding "
            "(subject_id, prediction_time) keys (%d -> %d rows)",
            log_context,
            n_collapsed,
            n_supplied,
            label_df.height,
        )
    return label_df


def assert_label_df_sorted(
    label_df: pl.DataFrame,
    *,
    subject_col: str = "subject_id",
    time_col: str = "prediction_time",
) -> None:
    """Fail loud before any MEDS-Tab handoff if ``label_df`` isn't sorted.

    Enforces, in code, the precondition :func:`_sort_and_or_aggregate_labels`'s
    docstring explains: MEDS-Tab's ``get_rolling_window_indicies`` silently
    miscomputes rolling-window aggregations (not an error, not a crash --
    wrong feature values that still shape-check) when its ``label_df``
    input isn't sorted by ``(subject_col, time_col)``, because it feeds a
    polars ``join_asof`` that assumes sortedness without validating it.
    Confirmed as the exact, sole cause of a real gate failure: a shared
    landmark grid built without this sort produced tabularized feature
    rows matching a known-correct reference on row keys but not on values
    (60% mismatch on a random sample), while the same grid re-sorted and
    re-tabularized matched exactly. This assert is the guard that failure
    mode can never again reach a MEDS-Tab CLI call silently.
    """
    subject_ids = label_df[subject_col].to_list()
    times = label_df[time_col].to_list()
    keys = list(zip(subject_ids, times))
    if keys != sorted(keys):
        raise AssertionError(
            f"label_df is not sorted by ({subject_col}, {time_col}) -- "
            "handing this to MEDS-Tab's tabularize stage would silently "
            "corrupt every rolling-window feature value via join_asof's "
            "unvalidated sortedness precondition (confirmed root cause, "
            "not a hypothetical). Sort before writing, don't bypass this."
        )


def build_shared_landmark_label_df(
    any_event_rows: Sequence[IndexRow],
    events_binned: pl.DataFrame,
    *,
    max_horizon_hours: float,
) -> pl.DataFrame:
    """Build the PRE-at-risk-filter landmark grid as a MEDS-Tab ``label_df``.

    Verified directly (not assumed) that this row set is byte-identical
    across every alert event for a given split -- landmarks are a fixed
    every-N-hour grid per subject's stay, independent of which outcome is
    being predicted; :func:`export_task_labels`' own per-(event,horizon)
    at-risk filter only ever REMOVES rows from this shared grid, never
    adds. That makes it a strict superset of every task's own label rows,
    safe to use as the ONE shared ``label_df`` for
    tabularize-static/time-series: MEDS-Tab's own ``compute_agg`` only
    restricts WHICH times get the (expensive) rolling-window aggregation
    computed, not what a computed value at any given time is -- feeding
    it the full raw per-event-timestamp grid instead (``label_df=None``)
    costs far more (measured: ~28x one task's own run on a real shard),
    while this landmark grid costs ~1.4x and covers every task at once.
    ``boolean_value`` is a placeholder -- unused by
    tabularize-static/time-series, which only reads
    ``subject_id``/``prediction_time`` from ``label_df``; the real
    outcome label is applied later, per task, by
    :func:`export_task_labels` (or, in the standalone-slicer
    architecture, by slicing this grid's own tabularized output).

    Sorted and OR-aggregated the same way :func:`export_task_labels` is
    (see :func:`_sort_and_or_aggregate_labels`) -- this function did NOT
    carry that fix originally, and its unsorted output was the confirmed,
    sole root cause of a real gate failure (see
    :func:`assert_label_df_sorted`'s docstring). Every caller building a
    MEDS-Tab-bound label_df goes through the same canonicalization now,
    not a second, independently-maintained copy of it.
    """
    pred_times = _prediction_times(
        any_event_rows, events_binned, max_horizon_hours=max_horizon_hours
    )
    label_df = pl.DataFrame(
        {
            "subject_id": [r.subject_id for r in any_event_rows],
            "prediction_time": pred_times,
            "boolean_value": [False] * len(any_event_rows),
        }
    )
    return _sort_and_or_aggregate_labels(label_df, log_context="shared_landmark_grid")


def export_task_labels(
    rows: dict[str, list[IndexRow]],
    times: dict[str, EventTimes],
    events_binned: pl.DataFrame,
    *,
    horizons: Sequence[float],
    output_dir: Path,
) -> dict[tuple[str, float], Path]:
    """Write one MEDS-Tab task-label parquet per (event, horizon).

    Each file is always named ``0.parquet`` regardless of shard count --
    that name is NOT a per-raw-shard split, it covers the WHOLE split
    passed in. See the write site below for why this matters and what a
    per-shard consumer must do about it.

    Columns ``(subject_id, prediction_time, boolean_value)`` -- MEDS-Tab's
    own task-label schema (``meds-tab-cache-task``'s
    ``label_column: boolean_value`` default, confirmed against the
    installed package's config, not the README alone). ``boolean_value``
    is :func:`~odyssey.inference.alerts.outcome_at_horizon`'s outcome;
    rows where it is ``None`` (censored, or the event already happened) are
    excluded -- the identical at-risk filtering
    :func:`~odyssey.inference.alerts.score_alerts` applies to every other
    baseline, so the label file's cohort matches what every other scorer's
    AUROC is computed over.

    Verifies its own write immediately (rows supplied == rows on disk) --
    the first half of item (d)'s "zero silently dropped or duplicated"
    requirement; the second half (meds-tab-cache-task's own joined count)
    is :func:`verify_cached_label_count`, checked after the CLI stage runs,
    not here.

    Sorted and OR-aggregated via :func:`_sort_and_or_aggregate_labels`
    (see its docstring for the confirmed root causes both fix) immediately
    before writing -- every caller of this module gets the same
    canonicalization at the same, single choke point.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, float], Path] = {}
    for event_name, event_rows in rows.items():
        if not event_rows:
            continue
        pred_times = _prediction_times(
            event_rows, events_binned, max_horizon_hours=max(horizons)
        )
        for h in horizons:
            outcomes = [outcome_at_horizon(r, times[event_name], h) for r in event_rows]
            keep = [i for i, o in enumerate(outcomes) if o is not None]
            if not keep:
                logger.info(
                    "[meds_tab] %s@%gh: no at-risk rows, label file not written",
                    event_name,
                    h,
                )
                continue
            label_df = pl.DataFrame(
                {
                    "subject_id": [event_rows[i].subject_id for i in keep],
                    "prediction_time": [pred_times[i] for i in keep],
                    "boolean_value": [bool(outcomes[i]) for i in keep],
                }
            )
            label_df = _sort_and_or_aggregate_labels(
                label_df, log_context=f"{event_name}@{h:g}h"
            )
            task_dir = output_dir / f"{event_name}_{h:g}h"
            task_dir.mkdir(parents=True, exist_ok=True)
            # "0.parquet" is a fixed, literal filename, NOT a per-raw-shard
            # split -- this single file holds every row of the WHOLE split
            # passed in (every raw shard `rows`/`events_binned` covers), not
            # just shard 0's own subjects. A real gate-check confirmed this
            # the hard way (2026-08-24): treating this file as shard-0-scoped
            # pulled in 136,832 rows from all 4 held_out shards instead of
            # shard 0's own 33,035, and fed a single-shard MEDS-Tab
            # tabularization a label file whose subjects didn't match its
            # input, breaking get_sparse_static_rep's row-count alignment
            # with an opaque IndexError. Do not change this filename --
            # readers already depend on it -- but any per-shard consumer
            # MUST filter this file to that shard's own subject_ids first
            # (subject_id column, is_in shard's own subject set) before
            # treating any subset of these rows as "shard N's rows". For a
            # label file that IS already split per raw shard, use
            # export_shard_aligned_labels instead.
            out_path = task_dir / "0.parquet"
            label_df.write_parquet(out_path)
            written = pl.read_parquet(out_path)
            if written.height != label_df.height:
                raise AssertionError(
                    f"{event_name}@{h}h: wrote {written.height} label rows to "
                    f"{out_path}, expected {label_df.height} -- silent drop or "
                    "duplication in the write itself"
                )
            paths[(event_name, h)] = out_path
            logger.info(
                "[meds_tab] %s@%gh: %d label rows -> %s",
                event_name,
                h,
                label_df.height,
                out_path,
            )
    return paths


def verify_cached_label_count(cache_dir: Path, task_name: str, expected_n: int) -> None:
    """Assert meds-tab-cache-task's own joined label output has the expected row count.

    The second half of item (d): our own :func:`export_task_labels` already
    confirms the label file we wrote has the row count we intended: this
    confirms MEDS-Tab's own join against our label file (its
    ``meds-tab-cache-task`` stage, matching labels to tabularized feature
    rows) neither silently dropped rows (e.g. a ``prediction_time`` that
    fell outside every feature window) nor duplicated them (e.g. a
    many-to-one match against the tabularized frame).
    """
    label_files = sorted((cache_dir / task_name / "labels").glob("**/*.parquet"))
    if not label_files:
        raise AssertionError(
            f"{task_name}: no cached label files found under {cache_dir / task_name / 'labels'}"
        )
    cached = pl.read_parquet(label_files)
    if cached.height != expected_n:
        raise AssertionError(
            f"{task_name}: meds-tab-cache-task joined {cached.height} label rows, "
            f"we supplied {expected_n} -- silent drop or duplication in the cache-task join"
        )


def assert_label_feature_alignment(
    tabularize_dir: Path,
    label_dir: Path,
    *,
    representative_window: str = "1d",
    representative_agg: str = "code/count",
) -> None:
    """Standalone-path integrity check.

    Label rows must align 1:1 with tabularized feature rows, per shard.
    Only relevant when ``meds-tab-cache-task`` never runs -- our direct
    tabularize + xgboost architecture bypasses it entirely (confirmed from
    ``MEDS_tabular_automl.xgboost_model``'s own label-loading source: it
    reads ``input_label_cache_dir/<split>/<shard>.parquet`` directly, with
    feature rows and label rows joined purely by shared positional order,
    not a separate cache-task join to verify). The coupling actually under
    threat here is label-row-to-feature-row alignment: if a label file and
    its tabularized ``.npz`` disagree on row count, xgboost would silently
    train/score against misaligned rows.

    Deliberately avoids importing the ``meds-tab`` package (matching this
    module's existing no-optional-dependency design -- the CLI stages run
    outside it, in a separate driver script): reads only the ``shape`` key
    directly out of the ``.npz`` file via ``numpy.load``, the same key
    ``MEDS_tabular_automl.utils.load_matrix`` reads internally, confirmed
    against a real finished run (shard 0, acute_kidney_injury_8h): label
    parquet height 26,087 == npz ``shape[0]`` 26,087, for both a
    time-series aggregation file and the static/present file -- MEDS-Tab
    writes every (window, agg) file for a shard in the same row order as
    whatever label_df it was given for that shard.
    """
    label_files = sorted(label_dir.glob("*.parquet"))
    if not label_files:
        raise AssertionError(f"no label files found under {label_dir}")
    mismatches: list[str] = []
    for label_fp in label_files:
        shard = label_fp.stem
        npz_fp = (
            tabularize_dir / shard / representative_window / representative_agg
        ).with_suffix(".npz")
        if not npz_fp.exists():
            raise AssertionError(
                f"expected tabularized feature file not found: {npz_fp}"
            )
        n_label = pl.read_parquet(label_fp).height
        n_feature = int(np.load(npz_fp)["shape"][0])
        if n_label != n_feature:
            mismatches.append(
                f"shard {shard}: label rows={n_label}, feature rows={n_feature}"
            )
    if mismatches:
        raise AssertionError(
            "label/feature row-count mismatch (silent drop or reorder in "
            "MEDS-Tab's own tabularization): " + "; ".join(mismatches)
        )


def _subject_ids_in_parquet_dir(d: Path, *, column: str = "subject_id") -> set[int]:
    files = sorted(d.glob("**/*.parquet"))
    if not files:
        raise AssertionError(f"no parquet files found under {d}")
    ids: set[int] = set()
    for fp in files:
        ids |= set(pl.read_parquet(fp, columns=[column])[column].unique().to_list())
    return ids


def assert_no_split_leakage(
    tab_data_dir: Path,
    held_out_subject_ids: set[int],
    tuning_subject_ids: set[int],
    *,
    train_label_dir: Path,
    held_out_label_dir: Path | None = None,
    tuning_label_dir: Path | None = None,
    subject_id_column: str = "subject_id",
) -> None:
    """(e): eval-only subjects must be unreachable at fit time.

    Held-out and tuning subjects must not leak into training, and must not
    leak into each other either. Checked at both levels that could leak them.

    Originally (as ``assert_no_held_out_subject_in_training_features``)
    this globbed ``**/*.parquet`` under MEDS-Tab's own ``tabularize/``
    output dir -- but that output is exclusively ``.npz`` (scipy sparse
    matrices) on every real run, with no ``subject_id`` column to check at
    all. Confirmed on a real 11.3h tabularize-time-series run: 0 .parquet
    files, 930 .npz files under ``tabularize/train``. That glob could never
    match anything MEDS-Tab actually produces, so the check always raised
    "no files found" regardless of whether a real leak existed -- it never
    once actually verified anything.

    ``tuning`` exists because ``meds-tab-xgboost`` hardcodes a real tuning
    split for its own early-stopping validation during the fit itself
    (confirmed from ``xgboost_model.py``'s ``_build``: ``XGBIterator(cfg,
    split="tuning")``, unconditional, not gated by ``prediction_splits`` --
    that config only controls the separate final-prediction stage). A
    subject used for early-stopping/model-selection is a leak vector just
    like a training subject, so it gets the same scrutiny as held_out here,
    plus one more: held_out and tuning must not overlap each other either,
    or model selection would implicitly peek at the held-out cohort.

    Checks (all over real parquet with a real ``subject_id`` column, never
    MEDS-Tab's own ``.npz`` tabularized output):

    1. Raw input level: every subject in ``tab_data_dir/train/*.parquet``
       (our own scoped shard partitioning, the thing actually under our
       control -- MEDS-Tab's tabularization is a deterministic function of
       whichever shards we feed it, so it cannot introduce leakage beyond
       what's already in our own input partitioning) must be absent from
       both ``held_out_subject_ids`` and ``tuning_subject_ids``.
    2. Label level: every subject in ``train_label_dir``'s task-label files
       -- what ``meds-tab-xgboost`` actually trains against, via
       ``meds-tab-cache-task``'s own join -- must also be absent from both.
       This is the vector level (1) alone cannot see: a bug in MEDS-Tab's
       own label/cache-task join mixing splits.
    3. Raw held_out vs raw tuning: the two eval-only splits' raw subjects
       must be disjoint from each other.

    If ``held_out_label_dir``/``tuning_label_dir`` are given, also checks
    each one's mirror-image leak: every subject in that split's label files
    must be a member of its own raw split (not a stray subject from
    elsewhere that leaked in).
    """
    eval_subject_ids = held_out_subject_ids | tuning_subject_ids

    raw_train_subjects = _subject_ids_in_parquet_dir(
        tab_data_dir / "train", column=subject_id_column
    )
    leaked_raw = raw_train_subjects & eval_subject_ids
    if leaked_raw:
        sample = sorted(leaked_raw)[:5]
        raise AssertionError(
            f"{len(leaked_raw)} eval-only (held-out/tuning) subject_id(s) present "
            f"in raw training shards under {tab_data_dir / 'train'} (e.g. {sample}) "
            "-- shard partitioning leaked eval-only subjects into train"
        )

    label_train_subjects = _subject_ids_in_parquet_dir(
        train_label_dir, column=subject_id_column
    )
    leaked_label = label_train_subjects & eval_subject_ids
    if leaked_label:
        sample = sorted(leaked_label)[:5]
        raise AssertionError(
            f"{len(leaked_label)} eval-only (held-out/tuning) subject_id(s) "
            f"present in train-split task labels under {train_label_dir} "
            f"(e.g. {sample}) -- split enforcement failed in the label/"
            "cache-task join, eval-only subjects were reachable at fit time"
        )

    raw_held_out_subjects = _subject_ids_in_parquet_dir(
        tab_data_dir / "held_out", column=subject_id_column
    )
    raw_tuning_subjects = _subject_ids_in_parquet_dir(
        tab_data_dir / "tuning", column=subject_id_column
    )
    overlap = raw_held_out_subjects & raw_tuning_subjects
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(
            f"{len(overlap)} subject_id(s) present in both the raw held_out and "
            f"tuning splits (e.g. {sample}) -- tuning (used for model selection) "
            "and held_out (used for final scoring) must be disjoint"
        )

    for label_dir, raw_subjects, name in (
        (held_out_label_dir, raw_held_out_subjects, "held-out"),
        (tuning_label_dir, raw_tuning_subjects, "tuning"),
    ):
        if label_dir is None:
            continue
        label_subjects = _subject_ids_in_parquet_dir(
            label_dir, column=subject_id_column
        )
        stray = label_subjects - raw_subjects
        if stray:
            sample = sorted(stray)[:5]
            raise AssertionError(
                f"{len(stray)} subject_id(s) in {name} task labels under "
                f"{label_dir} are not in the raw {name} split (e.g. {sample}) "
                f"-- {name} label file contains subjects outside its own cohort"
            )


@dataclass
class MedsTabBaselineModel:
    """A "predictions from file" baseline: MEDS-Tab's own pre-computed predictions.

    Duck-typed to satisfy :class:`odyssey.inference.alerts._ScoredBaseline`
    (``predict_proba``, ``feature_set``, ``n_features``, ``params``), but
    unlike every other baseline family here, ``predict_proba`` runs no
    model at all -- MEDS-Tab already produced the predictions natively, via
    its own tabularization and XGBoost fit, entirely outside this process.
    The ``x`` argument :func:`~odyssey.inference.alerts.score_alerts`
    passes is therefore not real features: it is expected to be a
    ``(n, 1)`` array of row indices (``np.arange(len(rows)).reshape(-1, 1)``,
    supplied as this (event, horizon)'s entry in the ``features_by_event``
    dict passed alongside this model), which
    ``score_alerts``'s own ``[keep]`` row-subsetting -- identical for every
    baseline family -- carries through unchanged. ``predict_proba`` reads
    those indices back to look up the matching pre-computed prediction,
    rather than computing anything itself.
    """

    predictions: np.ndarray
    """``(n_rows,)`` predictions aligned to the full row order for this
    (event, horizon), i.e. the same order the index-matrix passed as this
    model's ``features_by_event`` entry was built in."""

    feature_set: str = "meds_tab"
    n_features: int = 1
    params: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Look up pre-computed predictions by row index (``x[:, 0]``)."""
        idx = x[:, 0].astype(int)
        result: np.ndarray = self.predictions[idx]
        return result


def index_matrix_by_event(rows: dict[str, list[IndexRow]]) -> dict[str, np.ndarray]:
    """``{event: arange(len(rows[event])).reshape(-1, 1)}`` for ``extra_baselines``.

    See :class:`MedsTabBaselineModel`'s docstring for why this stands in
    for a real feature matrix.
    """
    return {
        name: np.arange(len(event_rows)).reshape(-1, 1)
        for name, event_rows in rows.items()
    }


__all__ = [
    "MedsTabBaselineModel",
    "assert_label_df_sorted",
    "assert_no_split_leakage",
    "build_shared_landmark_label_df",
    "export_task_labels",
    "index_matrix_by_event",
    "verify_cached_label_count",
    "assert_label_feature_alignment",
]
