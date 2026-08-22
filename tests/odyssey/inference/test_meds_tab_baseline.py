"""Tests for the MEDS-Tab native-pipeline comparator glue.

Everything here is plain MEDS/polars code with no optional dependency: this
module never imports ``meds-tab`` itself (the CLI stages run outside it, in
a separate driver script), so these tests do not need it installed --
unlike :mod:`test_tabicl_baseline`/:mod:`test_ebm_baseline`/
:mod:`test_survivalpfn_baseline`, there is no deferred-import error-path
test here, because there is nothing to defer.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from odyssey.data.alert_events import EventTimes
from odyssey.inference.alerts import IndexRow
from odyssey.inference.meds_tab_baseline import (
    MedsTabBaselineModel,
    _prediction_times,
    assert_label_feature_alignment,
    assert_no_split_leakage,
    export_task_labels,
    index_matrix_by_event,
    verify_cached_label_count,
)


T0 = datetime(2024, 1, 1, 6, 0, 0)


def _events_binned(subject_ids: list[int]) -> pl.DataFrame:
    """One timed non-birth event per subject, at T0 -- a clean, known origin.

    Single-event-per-subject fixture: first == last == origin == T0. Tests
    that need a subject with a real event span (first < last) build their
    own frame instead of using this one.
    """
    return pl.DataFrame(
        {
            "subject_id": subject_ids,
            "code": ["LAB//x//normal"] * len(subject_ids),
            "time": [T0] * len(subject_ids),
        }
    )


# ---------------------------------------------------------------------------
# _prediction_times: reconstruction + the event-span tripwire
# ---------------------------------------------------------------------------


def test_prediction_times_reconstructs_the_absolute_timestamp() -> None:
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=5.5)]
    times = _prediction_times(rows, _events_binned([1]), max_horizon_hours=8.0)
    assert times == [T0 + timedelta(hours=5.5)]


def test_prediction_times_accepts_rows_within_the_events_span_plus_horizon() -> None:
    # events span T0 to T0+50h; a landmark at 40h with an 8h horizon reaches
    # 48h, still inside [T0, T0+50h+8h] -- accepted.
    events = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "code": ["LAB//x//normal", "LAB//x//normal"],
            "time": [T0, T0 + timedelta(hours=50.0)],
        }
    )
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=h) for h in (0.0, 4.0, 40.0)]
    times = _prediction_times(rows, events, max_horizon_hours=8.0)
    for r, t in zip(rows, times):
        assert t == T0 + timedelta(hours=r.time_hours)


def test_prediction_times_rejects_a_row_past_the_last_event_plus_horizon() -> None:
    events = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "code": ["LAB//x//normal", "LAB//x//normal"],
            "time": [T0, T0 + timedelta(hours=50.0)],
        }
    )
    # time_hours=100 puts pred_time at T0+100h, past last(T0+50h) + 8h horizon.
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=100.0)]
    with pytest.raises(AssertionError, match="outside the subject's own event span"):
        _prediction_times(rows, events, max_horizon_hours=8.0)


def test_prediction_times_catches_a_wrong_origin_via_the_span_check() -> None:
    # Simulates the exact bug class the old round-trip check could not
    # catch: a row built against a DIFFERENT (wrong) origin than the one
    # this call recomputes from events_binned. Subject 1's real events
    # span T0..T0+10h; a row claiming time_hours=3.0 relative to a wrong
    # origin 500h before T0 would, if silently trusted, place prediction_time
    # far outside the subject's real span -- the row-order/index bookkeeping
    # bug this guards against.  Here we construct that directly: a row whose
    # intended prediction_time (computed by the caller against a bad origin)
    # falls outside the span, which is exactly what large/negative
    # time_hours values arriving from a mismatched origin would produce.
    events = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "code": ["LAB//x//normal", "LAB//x//normal"],
            "time": [T0, T0 + timedelta(hours=10.0)],
        }
    )
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=-500.0)]
    with pytest.raises(AssertionError, match="outside the subject's own event span"):
        _prediction_times(rows, events, max_horizon_hours=8.0)


def test_prediction_times_raises_for_a_subject_with_no_origin() -> None:
    rows = [IndexRow(subject_id=99, visit_id=1, time_hours=1.0)]
    with pytest.raises(ValueError, match="no sequence origin"):
        _prediction_times(
            rows, _events_binned([1]), max_horizon_hours=8.0
        )  # subject 1, not 99


# ---------------------------------------------------------------------------
# export_task_labels: at-risk filtering, write verification
# ---------------------------------------------------------------------------


def test_export_task_labels_writes_boolean_value_and_prediction_time(
    tmp_path: Path,
) -> None:
    rows = {
        "vasopressor_start": [
            IndexRow(subject_id=1, visit_id=10, time_hours=5.0),
            IndexRow(subject_id=2, visit_id=20, time_hours=3.0),
        ]
    }
    times = {
        "vasopressor_start": EventTimes(
            onset={(1, 10): 8.0},  # subject 1: observed within any horizon >= 3h
            censor={(1, 10): 30.0, (2, 20): 30.0},
            subject_scoped=False,
        )
    }
    events = _events_binned([1, 2])

    paths = export_task_labels(
        rows, times, events, horizons=(8.0,), output_dir=tmp_path
    )
    assert (("vasopressor_start", 8.0)) in paths
    df = pl.read_parquet(paths[("vasopressor_start", 8.0)])
    assert set(df.columns) == {"subject_id", "prediction_time", "boolean_value"}
    assert df.height == 2
    row1 = df.filter(pl.col("subject_id") == 1)
    assert row1["boolean_value"][0] is True
    assert row1["prediction_time"][0] == T0 + timedelta(hours=5.0)
    row2 = df.filter(pl.col("subject_id") == 2)
    assert row2["boolean_value"][0] is False


def test_export_task_labels_excludes_censored_and_not_at_risk_rows(
    tmp_path: Path,
) -> None:
    rows = {
        "death": [
            IndexRow(subject_id=1, visit_id=10, time_hours=5.0),  # censored before h
            IndexRow(subject_id=2, visit_id=20, time_hours=3.0),  # already happened
            IndexRow(subject_id=3, visit_id=30, time_hours=1.0),  # kept, observed
        ]
    }
    times = {
        "death": EventTimes(
            onset={(2, 20): 2.0, (3, 30): 6.0},
            censor={(1, 10): 6.0},  # follow-up ends before 5.0 + 8.0
            subject_scoped=False,
        )
    }
    events = _events_binned([1, 2, 3])

    paths = export_task_labels(
        rows, times, events, horizons=(8.0,), output_dir=tmp_path
    )
    df = pl.read_parquet(paths[("death", 8.0)])
    assert df.height == 1
    assert df["subject_id"][0] == 3
    assert df["boolean_value"][0] is True


def test_export_task_labels_skips_a_horizon_with_no_at_risk_rows(
    tmp_path: Path,
) -> None:
    rows = {"death": [IndexRow(subject_id=1, visit_id=10, time_hours=5.0)]}
    times = {"death": EventTimes(onset={}, censor={(1, 10): 5.0}, subject_scoped=False)}
    events = _events_binned([1])

    paths = export_task_labels(
        rows, times, events, horizons=(8.0,), output_dir=tmp_path
    )
    assert paths == {}


# ---------------------------------------------------------------------------
# verify_cached_label_count: the second half of (d)
# ---------------------------------------------------------------------------


def test_verify_cached_label_count_passes_when_counts_match(tmp_path: Path) -> None:
    labels_dir = tmp_path / "my_task" / "labels"
    labels_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(labels_dir / "0.parquet")
    verify_cached_label_count(tmp_path, "my_task", expected_n=3)  # no raise


def test_verify_cached_label_count_raises_on_a_mismatch(tmp_path: Path) -> None:
    labels_dir = tmp_path / "my_task" / "labels"
    labels_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(labels_dir / "0.parquet")
    with pytest.raises(AssertionError, match="joined 2 label rows, we supplied 3"):
        verify_cached_label_count(tmp_path, "my_task", expected_n=3)


def test_verify_cached_label_count_raises_when_nothing_was_cached(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError, match="no cached label files found"):
        verify_cached_label_count(tmp_path, "my_task", expected_n=1)


# ---------------------------------------------------------------------------
# assert_label_feature_alignment: standalone-path replacement for
# verify_cached_label_count when meds-tab-cache-task never runs
# ---------------------------------------------------------------------------


def _write_fake_npz(fp: Path, n_rows: int, n_cols: int = 3) -> None:
    """Real MEDS-Tab npz shape.

    An "array" key (row-major nonzero triplets,
    contents irrelevant here) and a "shape" key -- confirmed directly
    against a real finished tabularize-time-series run's own .npz files
    (np.load(fp).keys() == ['array', 'shape']), not assumed from the
    scipy.sparse.save_npz format (which MEDS-Tab does NOT use -- its own
    load_matrix reads these two keys itself, not scipy's own loader).
    """
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fp, array=np.zeros((0, 3)), shape=np.array([n_rows, n_cols]))


def test_alignment_assert_passes_when_rows_match(tmp_path: Path) -> None:
    tab_dir = tmp_path / "tabularize"
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(label_dir / "0.parquet")
    _write_fake_npz(tab_dir / "0" / "1d" / "code" / "count.npz", n_rows=3)
    assert_label_feature_alignment(tab_dir, label_dir)  # no raise


def test_alignment_assert_raises_on_row_count_mismatch(tmp_path: Path) -> None:
    tab_dir = tmp_path / "tabularize"
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(label_dir / "0.parquet")
    _write_fake_npz(tab_dir / "0" / "1d" / "code" / "count.npz", n_rows=5)
    with pytest.raises(AssertionError, match="label rows=3, feature rows=5"):
        assert_label_feature_alignment(tab_dir, label_dir)


def test_alignment_assert_raises_when_no_label_files_exist(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no label files found"):
        assert_label_feature_alignment(tmp_path / "tabularize", tmp_path / "labels")


def test_alignment_assert_raises_when_npz_missing_for_a_shard(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(
        AssertionError, match="expected tabularized feature file not found"
    ):
        assert_label_feature_alignment(tmp_path / "tabularize", label_dir)


# ---------------------------------------------------------------------------
# assert_no_split_leakage: item (e)
# ---------------------------------------------------------------------------


def _make_raw_split_dirs(
    root: Path,
    *,
    train_subjects: list[int],
    held_out_subjects: list[int],
    tuning_subjects: list[int] | None = None,
) -> Path:
    (root / "train").mkdir(parents=True)
    pl.DataFrame({"subject_id": train_subjects}).write_parquet(
        root / "train" / "0.parquet"
    )
    (root / "held_out").mkdir(parents=True)
    pl.DataFrame({"subject_id": held_out_subjects}).write_parquet(
        root / "held_out" / "0.parquet"
    )
    (root / "tuning").mkdir(parents=True)
    pl.DataFrame({"subject_id": tuning_subjects or []}).write_parquet(
        root / "tuning" / "0.parquet"
    )
    return root


def test_split_assert_passes_when_no_leak_at_either_level(tmp_path: Path) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2, 3],
        held_out_subjects=[4, 5],
        tuning_subjects=[6, 7],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(label_dir / "0.parquet")
    assert_no_split_leakage(
        tab_data_dir,
        held_out_subject_ids={4, 5},
        tuning_subject_ids={6, 7},
        train_label_dir=label_dir,
    )  # no raise


def test_split_assert_raises_when_raw_shard_leaks_held_out(tmp_path: Path) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2, 99],
        held_out_subjects=[99, 5],
        tuning_subjects=[6],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(AssertionError, match="raw training shards"):
        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={99, 5},
            tuning_subject_ids={6},
            train_label_dir=label_dir,
        )


def test_split_assert_raises_when_raw_shard_leaks_tuning(tmp_path: Path) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2, 6],
        held_out_subjects=[5],
        tuning_subjects=[6, 7],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(AssertionError, match="raw training shards"):
        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={5},
            tuning_subject_ids={6, 7},
            train_label_dir=label_dir,
        )


def test_split_assert_raises_when_label_leaks(tmp_path: Path) -> None:
    # Raw shards are clean -- this proves the label-level check is a real,
    # independent second gate, not just a rephrasing of the raw check.
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2, 3],
        held_out_subjects=[99],
        tuning_subjects=[6],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 99]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(AssertionError, match="train-split task labels"):
        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={99},
            tuning_subject_ids={6},
            train_label_dir=label_dir,
        )


def test_split_assert_raises_when_held_out_and_tuning_overlap(tmp_path: Path) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2],
        held_out_subjects=[4, 5],
        tuning_subjects=[5, 6],  # 5 leaked into both eval-only splits
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(AssertionError, match="both the raw held_out and tuning splits"):
        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={4, 5},
            tuning_subject_ids={5, 6},
            train_label_dir=label_dir,
        )


def test_split_assert_raises_when_no_raw_training_files_exist(tmp_path: Path) -> None:
    (tmp_path / "data" / "held_out").mkdir(parents=True)
    (tmp_path / "data" / "tuning").mkdir(parents=True)
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(AssertionError, match="no parquet files found"):
        assert_no_split_leakage(
            tmp_path / "data",
            held_out_subject_ids={1},
            tuning_subject_ids=set(),
            train_label_dir=label_dir,
        )


def test_split_assert_eval_label_dirs_pass_when_clean(tmp_path: Path) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2],
        held_out_subjects=[4, 5],
        tuning_subjects=[6, 7],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(label_dir / "0.parquet")
    held_out_label_dir = tmp_path / "labels" / "held_out"
    held_out_label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [4, 5]}).write_parquet(held_out_label_dir / "0.parquet")
    tuning_label_dir = tmp_path / "labels" / "tuning"
    tuning_label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [6, 7]}).write_parquet(tuning_label_dir / "0.parquet")
    assert_no_split_leakage(
        tab_data_dir,
        held_out_subject_ids={4, 5},
        tuning_subject_ids={6, 7},
        train_label_dir=label_dir,
        held_out_label_dir=held_out_label_dir,
        tuning_label_dir=tuning_label_dir,
    )  # no raise


def test_split_assert_raises_on_stray_subject_in_held_out_labels(
    tmp_path: Path,
) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2],
        held_out_subjects=[4, 5],
        tuning_subjects=[6],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(label_dir / "0.parquet")
    held_out_label_dir = tmp_path / "labels" / "held_out"
    held_out_label_dir.mkdir(parents=True)
    # 42 is not in the raw held-out split at all -- a stray subject.
    pl.DataFrame({"subject_id": [4, 5, 42]}).write_parquet(
        held_out_label_dir / "0.parquet"
    )
    with pytest.raises(AssertionError, match="not in the raw held-out split"):
        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={4, 5},
            tuning_subject_ids={6},
            train_label_dir=label_dir,
            held_out_label_dir=held_out_label_dir,
        )


def test_split_assert_raises_on_stray_subject_in_tuning_labels(
    tmp_path: Path,
) -> None:
    tab_data_dir = _make_raw_split_dirs(
        tmp_path / "data",
        train_subjects=[1, 2],
        held_out_subjects=[4],
        tuning_subjects=[6, 7],
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2]}).write_parquet(label_dir / "0.parquet")
    tuning_label_dir = tmp_path / "labels" / "tuning"
    tuning_label_dir.mkdir(parents=True)
    # 42 is not in the raw tuning split at all -- a stray subject.
    pl.DataFrame({"subject_id": [6, 7, 42]}).write_parquet(
        tuning_label_dir / "0.parquet"
    )
    with pytest.raises(AssertionError, match="not in the raw tuning split"):
        assert_no_split_leakage(
            tab_data_dir,
            held_out_subject_ids={4},
            tuning_subject_ids={6, 7},
            train_label_dir=label_dir,
            tuning_label_dir=tuning_label_dir,
        )


def test_split_assert_regression_memorial_npz_only_tabularize_dir(
    tmp_path: Path,
) -> None:
    """The original bug.

    assert_no_held_out_subject_in_training_features
    globbed **/*.parquet under MEDS-Tab's own tabularize/ output, which is
    exclusively .npz on every real run (confirmed on a real 11.3h
    tabularize-time-series run: 0 .parquet, 930 .npz under
    tabularize/train) -- so that glob could never match anything, and the
    check always failed with "no files found" regardless of real leakage.
    If the same mistake is repeated -- pointing tab_data_dir at an
    npz-bearing tabularize output dir instead of the raw scoped shard
    input dir -- this must still fail loudly, not silently accept.
    """
    old_wrong_dir = tmp_path / "meds_tab_out" / "some_task" / "tabularize"
    npz_shard_dir = old_wrong_dir / "train" / "0" / "1d" / "code"
    npz_shard_dir.mkdir(parents=True)
    (npz_shard_dir / "count.npz").write_bytes(
        b"\x00not a real npz, just marking the file exists"
    )
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1]}).write_parquet(label_dir / "0.parquet")
    with pytest.raises(AssertionError, match="no parquet files found"):
        assert_no_split_leakage(
            old_wrong_dir,
            held_out_subject_ids={1},
            tuning_subject_ids=set(),
            train_label_dir=label_dir,
        )


# ---------------------------------------------------------------------------
# MedsTabBaselineModel / index_matrix_by_event: the predict_proba bridge
# ---------------------------------------------------------------------------


def test_index_matrix_by_event_shapes() -> None:
    rows = {"death": [IndexRow(1, 1, 0.0), IndexRow(2, 2, 1.0), IndexRow(3, 3, 2.0)]}
    m = index_matrix_by_event(rows)
    assert m["death"].shape == (3, 1)
    assert m["death"].flatten().tolist() == [0, 1, 2]


def test_predict_proba_looks_up_predictions_by_row_index() -> None:
    model = MedsTabBaselineModel(predictions=np.array([0.1, 0.9, 0.3, 0.7]))
    x = np.array([[3], [0], [1]])  # a "[keep]"-subsetted index matrix
    p = model.predict_proba(x)
    assert np.allclose(p, [0.7, 0.1, 0.9])


def test_meds_tab_baseline_model_defaults() -> None:
    model = MedsTabBaselineModel(predictions=np.zeros(3))
    assert model.feature_set == "meds_tab"
    assert model.n_features == 1
    assert model.params == {}
