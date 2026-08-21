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
:func:`assert_no_held_out_subject_in_training_features`, not just assumed
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


def export_task_labels(
    rows: dict[str, list[IndexRow]],
    times: dict[str, EventTimes],
    events_binned: pl.DataFrame,
    *,
    horizons: Sequence[float],
    output_dir: Path,
) -> dict[tuple[str, float], Path]:
    """Write one MEDS-Tab task-label parquet per (event, horizon).

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
            task_dir = output_dir / f"{event_name}_{h:g}h"
            task_dir.mkdir(parents=True, exist_ok=True)
            out_path = task_dir / "0.parquet"
            label_df.write_parquet(out_path)
            written = pl.read_parquet(out_path)
            if written.height != len(keep):
                raise AssertionError(
                    f"{event_name}@{h}h: wrote {written.height} label rows to "
                    f"{out_path}, supplied {len(keep)} -- silent drop or "
                    "duplication in the write itself"
                )
            paths[(event_name, h)] = out_path
            logger.info(
                "[meds_tab] %s@%gh: %d label rows -> %s",
                event_name,
                h,
                len(keep),
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


def assert_no_held_out_subject_in_training_features(
    tabularize_dir: Path,
    held_out_subject_ids: set[int],
    *,
    subject_id_column: str = "subject_id",
) -> None:
    """(e): held-out subjects must be unreachable at fit time.

    MEDS-Tab's split handling is native (its own directory layout, not a
    subject_splits.parquet re-derivation at this stage -- see the module
    docstring), which should make this true by construction; checked
    explicitly here rather than assumed, over every tabularized training
    feature file MEDS-Tab actually produced.
    """
    train_files = sorted((tabularize_dir / "train").glob("**/*.parquet"))
    if not train_files:
        raise AssertionError(
            f"no tabularized training feature files found under {tabularize_dir / 'train'}"
        )
    leaked: set[int] = set()
    for fp in train_files:
        df = pl.read_parquet(fp, columns=[subject_id_column])
        leaked |= set(df[subject_id_column].unique().to_list()) & held_out_subject_ids
        if leaked:
            break
    if leaked:
        sample = sorted(leaked)[:5]
        raise AssertionError(
            f"{len(leaked)} held-out subject_id(s) present in tabularized training "
            f"features under {tabularize_dir / 'train'} (e.g. {sample}) -- split "
            "enforcement failed, held-out subjects were reachable at fit time"
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
    "assert_no_held_out_subject_in_training_features",
    "export_task_labels",
    "index_matrix_by_event",
    "verify_cached_label_count",
]
