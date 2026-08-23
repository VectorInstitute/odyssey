"""Tests for the optional TabICLv2 baseline.

Everything here runs without the real ``tabicl`` package installed:
:class:`TabICLBaselineModel` is duck-typed (any object with
``predict_proba``/``classes_`` works, tested with a fake stand-in), and
the fitting/row-capping logic in :func:`fit_tabicl_baselines` is tested
by monkeypatching :func:`odyssey.inference.tabicl_baseline._load_tabicl_classifier`
to return a fake classifier class rather than importing the real one --
this project's own logic (row selection, capping, skip conditions) is
what these tests are for, not TabICL's own fitting behavior. The one
test of the real optional-dependency error path
(:func:`test_load_tabicl_classifier_raises_a_clear_error_when_not_installed`)
runs precisely because ``tabicl`` is *not* installed in this environment.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import pytest

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference import tabicl_baseline as tabicl_module
from odyssey.inference.alerts import _index_rows_from_events
from odyssey.inference.tabicl_baseline import (
    TABICL_MAX_ROWS,
    TABICL_MIN_ROWS,
    TabICLBaselineModel,
    _load_tabicl_classifier,
    fit_tabicl_baselines,
)


T0 = datetime(2024, 1, 1)


def _events(n_subjects: int) -> pl.DataFrame:
    """Build the same planted-signal shape as test_alerts.py's fixture, smaller."""
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in range(1, n_subjects + 1):
        hadm = 1000 + sid
        for h in range(24):
            hr = 130.0 if sid % 2 == 0 and h >= 12 else 80.0
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), hr, hadm))
        if sid % 2 == 0:
            rows.append(
                (
                    sid,
                    "MEDICATION//norepinephrine//Administered",
                    T0 + timedelta(hours=14),
                    None,
                    hadm,
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


# ---------------------------------------------------------------------------
# TabICLBaselineModel: duck-typed, no real tabicl needed
# ---------------------------------------------------------------------------


class _FakeClf:
    """Stands in for a fitted tabicl.TabICLClassifier's public surface."""

    def __init__(self, classes: np.ndarray, proba: np.ndarray) -> None:
        self.classes_ = classes
        self._proba = proba

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._proba


def test_predict_proba_selects_the_positive_class_column() -> None:
    # classes_ in the sklearn-standard sorted order [0, 1]: column 1 is positive.
    clf = _FakeClf(classes=np.array([0, 1]), proba=np.array([[0.9, 0.1], [0.2, 0.8]]))
    model = TabICLBaselineModel(clf, feature_set="strong", n_features=3)
    p = model.predict_proba(np.zeros((2, 3)))
    assert np.allclose(p, [0.1, 0.8])


def test_predict_proba_handles_a_non_standard_class_column_order() -> None:
    # if classes_ were ever [1, 0] (reversed), the positive column moves too --
    # predict_proba must look it up, not assume column 1 is always positive.
    clf = _FakeClf(classes=np.array([1, 0]), proba=np.array([[0.1, 0.9], [0.8, 0.2]]))
    model = TabICLBaselineModel(clf, feature_set="strong", n_features=3)
    p = model.predict_proba(np.zeros((2, 3)))
    assert np.allclose(p, [0.1, 0.8])


class _BatchRecordingClf:
    """Records the size of every predict_proba call it receives.

    Returns a running-offset-based value (not reset per call) so
    concatenation order across batches is actually distinguishable, not
    just total row count.
    """

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.call_sizes: List[int] = []
        self._seen = 0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.call_sizes.append(x.shape[0])
        col1 = self._seen + np.arange(x.shape[0], dtype=float)
        self._seen += x.shape[0]
        return np.stack([1.0 - col1, col1], axis=1)


def test_predict_proba_batches_large_query_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real OOM.

    A real MIMIC rescore attempted a ~216GB allocation calling
    predict_proba unbatched on 552,000 query rows. Query-side batching
    must split large calls into chunks no larger than
    _PREDICT_BATCH_SIZE, and reassemble results in the original row
    order.
    """
    monkeypatch.setattr(tabicl_module, "_PREDICT_BATCH_SIZE", 10)
    clf = _BatchRecordingClf()
    model = TabICLBaselineModel(clf, feature_set="strong", n_features=3)

    n = 25
    p = model.predict_proba(np.zeros((n, 3)))

    assert clf.call_sizes == [10, 10, 5]  # 3 chunks, last one smaller
    assert np.allclose(p, np.arange(n, dtype=float))  # order preserved


def test_predict_proba_empty_input_returns_empty_without_calling_clf() -> None:
    clf = _BatchRecordingClf()
    model = TabICLBaselineModel(clf, feature_set="strong", n_features=3)
    p = model.predict_proba(np.zeros((0, 3)))
    assert p.shape == (0,)
    assert clf.call_sizes == []


class _RowDroppingClf:
    """Silently drops a row -- simulates the failure the length assert catches."""

    classes_ = np.array([0, 1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = max(x.shape[0] - 1, 0)
        return np.zeros((n, 2))


def test_predict_proba_raises_if_a_batch_silently_drops_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tabicl_module, "_PREDICT_BATCH_SIZE", 10)
    model = TabICLBaselineModel(_RowDroppingClf(), feature_set="strong", n_features=3)
    with pytest.raises(AssertionError, match="silently dropped"):
        model.predict_proba(np.zeros((5, 3)))


def test_tabicl_baseline_model_defaults_and_params() -> None:
    model = TabICLBaselineModel(_FakeClf(np.array([0, 1]), np.zeros((0, 2))))
    assert model.feature_set == "strong"
    assert model.n_features == 0
    assert model.params == {}


# ---------------------------------------------------------------------------
# _load_tabicl_classifier: the real optional-dependency error path
# ---------------------------------------------------------------------------


def test_load_tabicl_classifier_raises_a_clear_error_when_not_installed() -> None:
    with pytest.raises(ImportError, match="uv sync --extra tabicl"):
        _load_tabicl_classifier()


# ---------------------------------------------------------------------------
# fit_tabicl_baselines: row selection/capping, monkeypatched classifier
# (this project's own logic, independent of TabICL's own fit behavior)
# ---------------------------------------------------------------------------


class _RecordingFakeClassifier:
    """Records what it was fit on; predict_proba returns a fixed 50/50 split."""

    instances: List["_RecordingFakeClassifier"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.x_fit: Optional[np.ndarray] = None
        self.y_fit: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])
        _RecordingFakeClassifier.instances.append(self)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RecordingFakeClassifier":
        self.x_fit = x
        self.y_fit = y
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        return np.tile([0.5, 0.5], (n, 1))


@pytest.fixture(autouse=True)
def _fake_tabicl(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingFakeClassifier.instances.clear()
    monkeypatch.setattr(
        tabicl_module, "_load_tabicl_classifier", lambda: _RecordingFakeClassifier
    )


def test_fit_tabicl_baselines_fits_one_model_per_event_and_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a unit-test-sized fixture has far fewer than TABICL_MIN_ROWS=300 at-risk
    # rows; lower the floor rather than grow the fixture to real scale.
    monkeypatch.setattr(tabicl_module, "TABICL_MIN_ROWS", 10)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_tabicl_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", n_estimators=2
    )
    assert ("vasopressor_start", 8.0) in models
    model = models[("vasopressor_start", 8.0)]
    assert isinstance(model, TabICLBaselineModel)
    assert model.feature_set == "basic"
    assert model.params["n_estimators"] == 2.0
    p = model.predict_proba(np.zeros((5, model.n_features)))
    assert p.shape == (5,)


def test_fit_tabicl_baselines_passes_n_estimators_and_seed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tabicl_module, "TABICL_MIN_ROWS", 10)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    fit_tabicl_baselines(
        binned,
        rows,
        times,
        horizons=(8.0,),
        feature_set="basic",
        n_estimators=5,
        seed=7,
    )
    assert _RecordingFakeClassifier.instances
    fit_kwargs = _RecordingFakeClassifier.instances[0].kwargs
    assert fit_kwargs["n_estimators"] == 5
    assert fit_kwargs["random_state"] == 7


def test_fit_one_tabicl_caps_context_rows_at_tabicl_max_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same capping pattern as GBM_FIT_MAX_ROWS, applied to the ICL context."""
    monkeypatch.setattr(tabicl_module, "TABICL_MAX_ROWS", 10)
    monkeypatch.setattr(tabicl_module, "TABICL_MIN_ROWS", 2)
    events = _events(40)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    assert len(rows["vasopressor_start"]) > 10  # more rows than the lowered cap

    fit_tabicl_baselines(binned, rows, times, horizons=(8.0,), feature_set="basic")
    fit = _RecordingFakeClassifier.instances[0]
    assert fit.x_fit is not None
    assert fit.x_fit.shape[0] == 10


def test_fit_one_tabicl_skips_a_horizon_below_min_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tabicl_module, "TABICL_MIN_ROWS", 10_000)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_tabicl_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    assert models == {}
    assert not _RecordingFakeClassifier.instances


def test_module_constants_are_ordered_sensibly() -> None:
    assert TABICL_MIN_ROWS < TABICL_MAX_ROWS


# ---------------------------------------------------------------------------
# all-NaN column handling (entry 28's repro, entry 29's fix spec)
# ---------------------------------------------------------------------------


def test_fit_one_tabicl_neutralizes_an_all_nan_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit succeeds with a synthetic all-NaN column, and it stays present.

    tabicl's own predict_proba drops all-NaN columns internally before
    building its feature_mask, desyncing the mask from the caller's
    column count. Substituting 0.0 for these columns at both fit and
    predict time means tabicl's internal preprocessing never sees a
    real all-NaN column, so the drop-and-desync never happens; column
    counts stay aligned end to end.
    """
    monkeypatch.setattr(tabicl_module, "TABICL_MIN_ROWS", 10)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    real_features_for_events = tabicl_module.features_for_events

    def _with_all_nan_column(*args: object, **kwargs: object) -> dict:
        feats = real_features_for_events(*args, **kwargs)  # type: ignore[arg-type]
        out = {}
        for name, arr in feats.items():
            padded = np.full((arr.shape[0], arr.shape[1] + 1), np.nan, dtype=arr.dtype)
            padded[:, :-1] = arr
            out[name] = padded
        return out

    monkeypatch.setattr(tabicl_module, "features_for_events", _with_all_nan_column)

    models = fit_tabicl_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    model = models[("vasopressor_start", 8.0)]

    # the fixture's own real features may include other all-NaN or
    # partial-NaN columns (a small synthetic dataset naturally leaves some
    # columns sparsely or never populated); this test only asserts the
    # synthetic, deliberately all-NaN column is neutralized, not that no
    # NaN survives anywhere (partial-NaN columns are still left for
    # tabicl's own internal imputer, by design, not this fix's job).
    assert model.all_nan_cols is not None
    assert model.all_nan_cols[-1]

    fit = _RecordingFakeClassifier.instances[0]
    assert fit.x_fit is not None
    assert not np.isnan(fit.x_fit[:, -1]).any()
    assert fit.x_fit.shape[1] == model.n_features

    p = model.predict_proba(np.full((5, model.n_features), np.nan))
    assert p.shape == (5,)
    assert not np.isnan(p).any()
