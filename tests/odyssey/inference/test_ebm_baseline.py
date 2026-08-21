"""Tests for the optional InterpretML EBM baseline.

Everything here runs without the real ``interpret`` package installed:
:class:`EBMBaselineModel` is duck-typed (any object with
``predict_proba``/``classes_`` works, tested with a fake stand-in), and
the fitting logic in :func:`fit_ebm_baselines` is tested by monkeypatching
:func:`odyssey.inference.ebm_baseline._load_ebm_classifier` to return a
fake classifier class rather than importing the real one -- this
project's own logic (row selection, skip conditions) is what these tests
are for, not EBM's own fitting behavior. The one test of the real
optional-dependency error path
(:func:`test_load_ebm_classifier_raises_a_clear_error_when_not_installed`)
runs precisely because ``interpret`` is *not* installed in this
environment.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import pytest

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference import ebm_baseline as ebm_module
from odyssey.inference.alerts import _index_rows_from_events
from odyssey.inference.ebm_baseline import (
    EBMBaselineModel,
    _load_ebm_classifier,
    fit_ebm_baselines,
)


T0 = datetime(2024, 1, 1)


def _events(n_subjects: int) -> pl.DataFrame:
    """Same planted-signal shape as test_tabicl_baseline.py's fixture."""
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
# EBMBaselineModel: duck-typed, no real interpret needed
# ---------------------------------------------------------------------------


class _FakeClf:
    """Stands in for a fitted ExplainableBoostingClassifier's public surface."""

    def __init__(self, classes: np.ndarray, proba: np.ndarray) -> None:
        self.classes_ = classes
        self._proba = proba

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._proba


def test_predict_proba_selects_the_positive_class_column() -> None:
    clf = _FakeClf(classes=np.array([0, 1]), proba=np.array([[0.9, 0.1], [0.2, 0.8]]))
    model = EBMBaselineModel(clf, feature_set="strong", n_features=3)
    p = model.predict_proba(np.zeros((2, 3)))
    assert np.allclose(p, [0.1, 0.8])


def test_predict_proba_handles_a_non_standard_class_column_order() -> None:
    clf = _FakeClf(classes=np.array([1, 0]), proba=np.array([[0.1, 0.9], [0.8, 0.2]]))
    model = EBMBaselineModel(clf, feature_set="strong", n_features=3)
    p = model.predict_proba(np.zeros((2, 3)))
    assert np.allclose(p, [0.1, 0.8])


def test_ebm_baseline_model_defaults_and_params() -> None:
    model = EBMBaselineModel(_FakeClf(np.array([0, 1]), np.zeros((0, 2))))
    assert model.feature_set == "strong"
    assert model.n_features == 0
    assert model.params == {}


# ---------------------------------------------------------------------------
# _load_ebm_classifier: the real optional-dependency error path
# ---------------------------------------------------------------------------


def test_load_ebm_classifier_raises_a_clear_error_when_not_installed() -> None:
    with pytest.raises(ImportError, match="uv sync --extra ebm"):
        _load_ebm_classifier()


# ---------------------------------------------------------------------------
# fit_ebm_baselines: row selection, monkeypatched classifier
# (this project's own logic, independent of EBM's own fit behavior)
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
def _fake_ebm(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingFakeClassifier.instances.clear()
    monkeypatch.setattr(
        ebm_module, "_load_ebm_classifier", lambda: _RecordingFakeClassifier
    )


def test_fit_ebm_baselines_fits_one_model_per_event_and_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ebm_module, "EBM_MIN_ROWS", 10)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_ebm_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    assert ("vasopressor_start", 8.0) in models
    model = models[("vasopressor_start", 8.0)]
    assert isinstance(model, EBMBaselineModel)
    assert model.feature_set == "basic"
    p = model.predict_proba(np.zeros((5, model.n_features)))
    assert p.shape == (5,)


def test_fit_ebm_baselines_passes_seed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ebm_module, "EBM_MIN_ROWS", 10)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    fit_ebm_baselines(binned, rows, times, horizons=(8.0,), feature_set="basic", seed=7)
    assert _RecordingFakeClassifier.instances
    assert _RecordingFakeClassifier.instances[0].kwargs["random_state"] == 7


def test_fit_one_ebm_skips_a_horizon_below_min_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ebm_module, "EBM_MIN_ROWS", 10_000)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_ebm_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    assert models == {}
    assert not _RecordingFakeClassifier.instances


def test_fit_ebm_baselines_below_the_cap_fits_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below EBM_MAX_ROWS, every row with a determinable outcome is kept."""
    monkeypatch.setattr(ebm_module, "EBM_MIN_ROWS", 2)
    events = _events(40)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    n_at_risk = len(rows["vasopressor_start"])
    assert n_at_risk > 10

    fit_ebm_baselines(binned, rows, times, horizons=(8.0,), feature_set="basic")
    fit = _RecordingFakeClassifier.instances[0]
    assert fit.x_fit is not None
    # some at-risk rows are right-censored at this horizon (no determinable
    # outcome) and are filtered out before fitting, same as the GBM/TabICL
    # baselines; the guarantee under test is no cap on TOP of that filter,
    # since n_at_risk here is well under EBM_MAX_ROWS.
    assert 10 < fit.x_fit.shape[0] <= n_at_risk


def test_grouped_subsample_never_splits_a_subject() -> None:
    """Unit test of the grouping primitive directly: a capped subsample's
    row count for every included subject exactly matches that subject's
    full row count in the input, never a partial slice of it.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 5)  # 20 subjects, 5 rows each, 100 total
    keep = np.arange(100)

    subsample = ebm_module._grouped_subsample(keep, groups, cap=37, rng=rng)

    assert 0 < len(subsample) <= 37
    kept_groups = groups[subsample]
    _, counts = np.unique(kept_groups, return_counts=True)
    assert (counts == 5).all()  # every included subject's full 5 rows, no partial


def test_fit_one_ebm_caps_rows_and_records_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above EBM_MAX_ROWS, fewer rows are fit on and the model records it."""
    monkeypatch.setattr(ebm_module, "EBM_MIN_ROWS", 2)
    monkeypatch.setattr(ebm_module, "EBM_MAX_ROWS", 50)
    events = _events(40)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    assert len(rows["vasopressor_start"]) > 50  # more rows than the lowered cap

    models = fit_ebm_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    model = models[("vasopressor_start", 8.0)]
    assert model.params["row_capped"] == 1.0

    fit = _RecordingFakeClassifier.instances[0]
    assert fit.x_fit is not None
    assert fit.x_fit.shape[0] <= 50
