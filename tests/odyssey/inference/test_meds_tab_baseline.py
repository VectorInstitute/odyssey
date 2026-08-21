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
    assert_no_held_out_subject_in_training_features,
    export_task_labels,
    index_matrix_by_event,
    verify_cached_label_count,
)


T0 = datetime(2024, 1, 1, 6, 0, 0)


def _events_binned(subject_ids: list[int]) -> pl.DataFrame:
    """One timed non-birth event per subject, at T0 -- a clean, known origin."""
    return pl.DataFrame(
        {
            "subject_id": subject_ids,
            "code": ["LAB//x//normal"] * len(subject_ids),
            "time": [T0] * len(subject_ids),
        }
    )


# ---------------------------------------------------------------------------
# _prediction_times: the (d) round-trip
# ---------------------------------------------------------------------------


def test_prediction_times_reconstructs_the_absolute_timestamp() -> None:
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=5.5)]
    times = _prediction_times(rows, _events_binned([1]))
    assert times == [T0 + timedelta(hours=5.5)]


def test_prediction_times_round_trips_exactly_for_many_rows() -> None:
    rows = [
        IndexRow(subject_id=1, visit_id=10, time_hours=h)
        for h in (0.0, 4.0, 8.5, 100.25, 999.999)
    ]
    events = _events_binned([1])
    times = _prediction_times(rows, events)
    for r, t in zip(rows, times):
        back = (t - T0).total_seconds() / 3600.0
        assert back == pytest.approx(r.time_hours, abs=1e-9)


def test_prediction_times_raises_for_a_subject_with_no_origin() -> None:
    rows = [IndexRow(subject_id=99, visit_id=1, time_hours=1.0)]
    with pytest.raises(ValueError, match="no sequence origin"):
        _prediction_times(rows, _events_binned([1]))  # subject 1, not 99


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
# assert_no_held_out_subject_in_training_features: item (e)
# ---------------------------------------------------------------------------


def test_split_assert_passes_when_no_held_out_subject_leaks(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(train_dir / "0.parquet")
    assert_no_held_out_subject_in_training_features(
        tmp_path, held_out_subject_ids={4, 5}
    )


def test_split_assert_raises_when_a_held_out_subject_leaks(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(train_dir / "0.parquet")
    with pytest.raises(AssertionError, match="held-out subject_id.* present"):
        assert_no_held_out_subject_in_training_features(
            tmp_path, held_out_subject_ids={2, 99}
        )


def test_split_assert_raises_when_no_training_files_exist(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no tabularized training feature files"):
        assert_no_held_out_subject_in_training_features(
            tmp_path, held_out_subject_ids={1}
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
