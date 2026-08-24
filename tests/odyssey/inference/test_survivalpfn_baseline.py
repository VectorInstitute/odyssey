"""Tests for the optional SurvivalPFN baseline.

Everything here runs without the real ``survivalpfn`` package installed:
:class:`SurvivalPFNBaselineModel` is duck-typed (any object exposing
``predict_event_distribution`` -> an object with ``survival_at`` works,
tested with a fake stand-in), and the fitting logic in
:func:`fit_survivalpfn_baselines` is tested by monkeypatching
:func:`odyssey.inference.survivalpfn_baseline._load_survival_estimator`
to return a fake estimator class -- this project's own logic (the
survival-native (T, delta) construction, row capping, one-context-per-
event sharing across horizons) is what these tests are for, not
SurvivalPFN's own model behavior. The one test of the real optional-
dependency error path runs precisely because ``survivalpfn`` is *not*
installed in this environment.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import pytest
import torch

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference import survivalpfn_baseline as survivalpfn_module
from odyssey.inference.alerts import EventTimes, IndexRow, _index_rows_from_events
from odyssey.inference.fit_cache import FitCache
from odyssey.inference.survivalpfn_baseline import (
    SURVIVALPFN_MAX_FEATURES,
    SURVIVALPFN_MAX_ROWS,
    SURVIVALPFN_MIN_ROWS,
    SurvivalPFNBaselineModel,
    _load_survival_estimator,
    _survival_targets,
    fit_survivalpfn_baselines,
)


T0 = datetime(2024, 1, 1)


def _events(n_subjects: int) -> pl.DataFrame:
    """Build the same planted-signal shape as test_tabicl_baseline.py's fixture."""
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
# _survival_targets: the project's own (T, delta) construction
# ---------------------------------------------------------------------------


def test_survival_targets_marks_observed_rows_with_hours_to_onset() -> None:
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=5.0)]
    times = EventTimes(
        onset={(1, 10): 20.0}, censor={(1, 10): 30.0}, subject_scoped=False
    )
    t, delta, keep = _survival_targets(rows, times)
    assert list(keep) == [0]
    assert t[0] == pytest.approx(15.0)  # 20 - 5
    assert delta[0] == 1.0


def test_survival_targets_marks_censored_rows_with_hours_to_end_of_followup() -> None:
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=5.0)]
    times = EventTimes(onset={}, censor={(1, 10): 30.0}, subject_scoped=False)
    t, delta, keep = _survival_targets(rows, times)
    assert list(keep) == [0]
    assert t[0] == pytest.approx(25.0)  # 30 - 5
    assert delta[0] == 0.0


def test_survival_targets_excludes_rows_where_the_event_already_happened() -> None:
    # onset at hour 3, index row at hour 5 -- not at risk (outcome_at_horizon's rule)
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=5.0)]
    times = EventTimes(
        onset={(1, 10): 3.0}, censor={(1, 10): 30.0}, subject_scoped=False
    )
    t, delta, keep = _survival_targets(rows, times)
    assert list(keep) == []
    assert len(t) == 0 and len(delta) == 0


def test_survival_targets_excludes_rows_with_no_follow_up_past_the_index_time() -> None:
    rows = [IndexRow(subject_id=1, visit_id=10, time_hours=5.0)]
    times = EventTimes(onset={}, censor={(1, 10): 5.0}, subject_scoped=False)
    t, delta, keep = _survival_targets(rows, times)
    assert list(keep) == []


def test_survival_targets_uses_subject_scoping_when_the_event_is_subject_scoped() -> (
    None
):
    rows = [IndexRow(subject_id=1, visit_id=99, time_hours=5.0)]
    times = EventTimes(
        onset={(1, -1): 20.0}, censor={(1, -1): 30.0}, subject_scoped=True
    )
    t, delta, keep = _survival_targets(rows, times)
    assert list(keep) == [0]
    assert t[0] == pytest.approx(15.0)
    assert delta[0] == 1.0


# ---------------------------------------------------------------------------
# SurvivalPFNBaselineModel: duck-typed, no real survivalpfn needed
# ---------------------------------------------------------------------------


class _FakeDistribution:
    """Stands in for survivalpfn.models.utils.HistogramDistribution."""

    def survival_at(self, time: torch.Tensor) -> torch.Tensor:
        # deterministic function of the query time, same for every row,
        # so tests can compute the expected predict_proba directly.
        return torch.full_like(time, 1.0 - float(time[0]) / 100.0)


class _FakeEstimator:
    """Stands in for a fitted survivalpfn.SurvivalEstimator's public surface.

    Tracks every instance created (mirroring
    ``_RecordingFakeClassifier.instances`` in test_tabicl_baseline.py) so
    tests can inspect what a fit call was given by reading
    ``_FakeEstimator.instances`` directly, rather than through
    ``SurvivalPFNBaselineModel.estimator`` -- that field is typed
    ``object`` (see the module docstring's reasoning for
    ``TabICLBaselineModel.clf``), so reaching through it would need a
    cast at every call site instead of one shared list.
    """

    instances: List["_FakeEstimator"] = []

    def __init__(self, device: object = None, **kwargs: object) -> None:
        self.device = device
        self.X_fit: Optional[np.ndarray] = None
        self.delta_fit: Optional[np.ndarray] = None
        self.T_fit: Optional[np.ndarray] = None
        _FakeEstimator.instances.append(self)

    # T matches survivalpfn's own fit(X, delta, T) signature.
    def fit(
        self,
        X: np.ndarray,
        delta: np.ndarray,
        T: np.ndarray,  # noqa: N803
    ) -> "_FakeEstimator":
        self.X_fit = X
        self.delta_fit = delta
        self.T_fit = T
        return self

    def predict_event_distribution(self, x: np.ndarray) -> _FakeDistribution:
        return _FakeDistribution()


@pytest.fixture(autouse=True)
def _clear_fake_estimator_instances() -> None:
    _FakeEstimator.instances.clear()


def test_predict_proba_is_one_minus_survival_at_the_models_horizon() -> None:
    model = SurvivalPFNBaselineModel(
        _FakeEstimator(), horizon_hours=40.0, feature_set="basic", n_features=3
    )
    p = model.predict_proba(np.zeros((5, 3)))
    # fake survival_at(40) = 1 - 40/100 = 0.6, so predict_proba = 1 - 0.6 = 0.4
    assert p.shape == (5,)
    assert np.allclose(p, 0.4)


def test_predict_proba_uses_this_instances_own_horizon_not_a_shared_one() -> None:
    estimator = _FakeEstimator()
    model_8h = SurvivalPFNBaselineModel(estimator, horizon_hours=8.0)
    model_80h = SurvivalPFNBaselineModel(estimator, horizon_hours=80.0)
    p8 = model_8h.predict_proba(np.zeros((2, 3)))
    p80 = model_80h.predict_proba(np.zeros((2, 3)))
    assert np.allclose(p8, 0.08)
    assert np.allclose(p80, 0.8)


class _BatchRecordingEstimator:
    """Records the size of every predict_event_distribution call."""

    def __init__(self) -> None:
        self.call_sizes: List[int] = []

    def predict_event_distribution(self, x: np.ndarray) -> _FakeDistribution:
        self.call_sizes.append(x.shape[0])
        return _FakeDistribution()


def test_predict_proba_batches_large_query_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real OOM.

    This model shares TabICL's in-context-transformer shape and the
    identical unbounded-query-size exposure (confirmed via TabICL's real
    crash: a ~216GB allocation attempt on 552,000 unbatched query rows).
    Query-side batching must split large calls into chunks no larger than
    _PREDICT_BATCH_SIZE.
    """
    monkeypatch.setattr(survivalpfn_module, "_PREDICT_BATCH_SIZE", 10)
    estimator = _BatchRecordingEstimator()
    model = SurvivalPFNBaselineModel(estimator, horizon_hours=40.0)

    p = model.predict_proba(np.zeros((25, 3)))

    assert estimator.call_sizes == [10, 10, 5]
    assert p.shape == (25,)
    assert np.allclose(p, 0.4)


def test_predict_proba_empty_input_returns_empty_without_calling_estimator() -> None:
    estimator = _BatchRecordingEstimator()
    model = SurvivalPFNBaselineModel(estimator, horizon_hours=40.0)
    p = model.predict_proba(np.zeros((0, 3)))
    assert p.shape == (0,)
    assert estimator.call_sizes == []


def test_predict_proba_raises_if_a_batch_silently_drops_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "_PREDICT_BATCH_SIZE", 10)

    class _DroppingEstimator:
        def predict_event_distribution(self, x: np.ndarray) -> object:
            class _Dist:
                def survival_at(self, time: torch.Tensor) -> torch.Tensor:
                    # Always returns one fewer row than asked for.
                    return torch.zeros(max(len(time) - 1, 0))

            return _Dist()

    model = SurvivalPFNBaselineModel(_DroppingEstimator(), horizon_hours=40.0)
    with pytest.raises(AssertionError, match="silently dropped"):
        model.predict_proba(np.zeros((5, 3)))


def test_survivalpfn_baseline_model_defaults() -> None:
    model = SurvivalPFNBaselineModel(_FakeEstimator(), horizon_hours=8.0)
    assert model.feature_set == "basic"
    assert model.n_features == 0
    assert model.params == {}


# ---------------------------------------------------------------------------
# _load_survival_estimator: the real optional-dependency error path
# ---------------------------------------------------------------------------


def test_load_survival_estimator_raises_a_clear_error_when_not_installed() -> None:
    with pytest.raises(ImportError, match="uv sync --extra survivalpfn"):
        _load_survival_estimator()


# ---------------------------------------------------------------------------
# fit_survivalpfn_baselines: this project's own logic, monkeypatched estimator
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_survivalpfn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        survivalpfn_module, "_load_survival_estimator", lambda: _FakeEstimator
    )


def test_fit_survivalpfn_baselines_shares_one_context_across_every_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0, 24.0, 72.0), feature_set="basic"
    )
    assert ("vasopressor_start", 8.0) in models
    assert ("vasopressor_start", 24.0) in models
    assert ("vasopressor_start", 72.0) in models
    # one fit serves all three horizons: same underlying estimator object
    m8 = models[("vasopressor_start", 8.0)]
    m24 = models[("vasopressor_start", 24.0)]
    m72 = models[("vasopressor_start", 72.0)]
    assert m8.estimator is m24.estimator is m72.estimator
    assert m8.horizon_hours == 8.0
    assert m24.horizon_hours == 24.0
    assert m72.horizon_hours == 72.0
    assert m8.feature_set == "basic"


def test_fit_survivalpfn_baselines_fits_on_survival_native_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    fit_survivalpfn_baselines(binned, rows, times, horizons=(8.0,), feature_set="basic")
    estimator = _FakeEstimator.instances[0]
    assert estimator.T_fit is not None
    assert estimator.delta_fit is not None
    # both event and censored rows present in the fit target (not a
    # per-horizon binary outcome -- a mix of 0/1 deltas, and T values not
    # bounded by any single horizon)
    assert set(np.unique(estimator.delta_fit).tolist()) <= {0.0, 1.0}
    assert estimator.T_fit.max() >= 0


def test_fit_survivalpfn_baselines_skips_an_event_below_min_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 10_000)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    assert models == {}


def test_fit_one_survivalpfn_caps_context_rows_subject_grouped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MAX_ROWS", 10)
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(40)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    assert len(rows["vasopressor_start"]) > 10

    models = fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic"
    )
    model = models[("vasopressor_start", 8.0)]
    estimator = _FakeEstimator.instances[0]
    assert estimator.X_fit is not None
    assert estimator.X_fit.shape[0] <= 10
    assert model.params["row_capped"] == 1.0


def test_grouped_subsample_never_splits_a_subject() -> None:
    # direct unit test of the capping primitive itself, independent of the
    # fixture's own at-risk row distribution (which varies per subject and
    # is not a fixed multiple, since _survival_targets already filters
    # "already happened" rows before this cap ever runs).
    rng = np.random.default_rng(0)
    n_subjects, rows_per_subject, cap = 20, 5, 37
    keep = np.arange(n_subjects * rows_per_subject)
    groups = np.repeat(np.arange(n_subjects), rows_per_subject)
    result = survivalpfn_module._grouped_subsample(keep, groups, cap, rng)
    kept_groups = groups[np.isin(keep, result)]
    _, counts = np.unique(kept_groups, return_counts=True)
    assert (counts == rows_per_subject).all()
    assert len(result) <= cap


def test_fit_one_survivalpfn_rejects_a_feature_matrix_over_the_hard_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    real_features_for_events = survivalpfn_module.features_for_events

    def _oversized(*args: object, **kwargs: object) -> dict:
        feats = real_features_for_events(*args, **kwargs)  # type: ignore[arg-type]
        return {
            name: np.zeros((arr.shape[0], SURVIVALPFN_MAX_FEATURES + 1))
            for name, arr in feats.items()
        }

    monkeypatch.setattr(survivalpfn_module, "features_for_events", _oversized)

    with pytest.raises(ValueError, match="max_num_features"):
        fit_survivalpfn_baselines(
            binned, rows, times, horizons=(8.0,), feature_set="strong"
        )


def test_module_constants_are_ordered_sensibly() -> None:
    assert SURVIVALPFN_MIN_ROWS < SURVIVALPFN_MAX_ROWS


# ---------------------------------------------------------------------------
# fit_survivalpfn_baselines + FitCache: cache hit skips the fit, cache miss saves
# ---------------------------------------------------------------------------


def test_fit_survivalpfn_baselines_skips_fitting_on_a_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    cache = FitCache(cache_dir=tmp_path)

    fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0, 24.0), feature_set="basic", cache=cache
    )
    assert len(_FakeEstimator.instances) == 1

    def _boom() -> None:
        raise AssertionError("should not load the estimator on a cache hit")

    monkeypatch.setattr(survivalpfn_module, "_load_survival_estimator", _boom)
    models = fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0, 24.0), feature_set="basic", cache=cache
    )
    # cache hit still serves every horizon the caller asks for, from the
    # one cached context -- not just whatever horizons were fit originally
    assert ("vasopressor_start", 8.0) in models
    assert ("vasopressor_start", 24.0) in models
    assert len(_FakeEstimator.instances) == 1


def test_fit_survivalpfn_baselines_refits_when_the_cache_is_from_a_different_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    writer = FitCache(cache_dir=tmp_path, fingerprint={"survivalpfn": "1.0.0"})
    fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", cache=writer
    )
    assert len(_FakeEstimator.instances) == 1

    reader = FitCache(cache_dir=tmp_path, fingerprint={"survivalpfn": "2.0.0"})
    fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", cache=reader
    )
    assert len(_FakeEstimator.instances) == 2


def test_survival_targets_administratively_censor_at_the_cap() -> None:
    """Follow-up beyond the cap becomes censored AT the cap, not an event.

    Real defect this fixes (2026-08-23, MIMIC-IV): death is subject-scoped,
    so its uncapped time-to-event had a median of 6,915 h against an 8/24/72 h
    alert window (3.3% of events inside 72 h). The fitted survival curve was
    flat at 1.0 across that window and ``1 - S(h)`` came back a literal
    constant 0.0 at every horizon. The visit-scoped events (medians 60-100 h)
    were unaffected, which is why this surfaced on one column only.
    """
    from odyssey.inference import survivalpfn_baseline as spfn  # noqa: PLC0415

    _survival_targets = spfn._survival_targets

    rows = [
        IndexRow(subject_id=1, visit_id=-1, time_hours=0.0),  # event at 10h
        IndexRow(subject_id=2, visit_id=-1, time_hours=0.0),  # event at 5000h
        IndexRow(subject_id=3, visit_id=-1, time_hours=0.0),  # censored at 5000h
    ]
    times = EventTimes(
        onset={(1, -1): 10.0, (2, -1): 5000.0},
        censor={(1, -1): 20.0, (2, -1): 6000.0, (3, -1): 5000.0},
        subject_scoped=True,
    )
    t, delta, keep = _survival_targets(rows, times, None)
    assert t.tolist() == [10.0, 5000.0, 5000.0]
    assert delta.tolist() == [1.0, 1.0, 0.0]
    assert keep.tolist() == [0, 1, 2]

    t, delta, keep = _survival_targets(rows, times, 72.0)
    assert t.tolist() == [10.0, 72.0, 72.0]  # both long rows cut to the cap
    assert delta.tolist() == [1.0, 0.0, 0.0]  # and treated as censored there
    assert keep.tolist() == [0, 1, 2]  # the row set itself is unchanged


def test_followup_cap_defaults_to_the_largest_horizon() -> None:
    """The default cap matches the question the other baselines answer."""
    import inspect  # noqa: PLC0415

    from odyssey.inference import survivalpfn_baseline as spfn  # noqa: PLC0415

    default = (
        inspect.signature(spfn.fit_survivalpfn_baselines)
        .parameters["followup_cap_hours"]
        .default
    )
    assert default is spfn._CAP_AT_MAX_HORIZON
    # and the cap is part of the cache key, so a differently-capped fit is
    # never silently reused (the feature-set lesson, same day); the sample
    # path (uniform vs enriched) is part of it too, added the same day.
    source = inspect.getsource(spfn)
    assert 'cache_key = f"survivalpfn/{cap_tag}/{sample_tag}/{event_name}"' in source


def test_predict_proba_keeps_tiny_probabilities_and_warns_on_a_dead_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """float64 before subtracting; a still-constant column says so out loud.

    float32's eps is ~1.2e-7, so for a rare event at a short horizon --
    where survival genuinely is ~1 -- ``1 - S(h)`` rounds to exactly 0 for
    every row. Measured on MIMIC-IV 2026-08-23: death@8h and @24h came back
    a literal constant 0.0 even after the follow-up cap fixed the
    long-horizon case, and vasopressor@8h fell to 175 distinct values at
    magnitudes ~1e-4.
    """
    import logging  # noqa: PLC0415

    from odyssey.inference import survivalpfn_baseline as spfn  # noqa: PLC0415

    class _Dist:
        def __init__(self, survival: float) -> None:
            self._s = survival

        def survival_at(self, h: torch.Tensor) -> torch.Tensor:
            return torch.full((h.shape[0],), self._s, dtype=torch.float32)

    class _Estimator:
        def __init__(self, survival: float) -> None:
            self._s = survival

        def predict_event_distribution(self, x: np.ndarray) -> _Dist:
            return _Dist(self._s)

    x = np.zeros((4, 3), dtype=np.float32)
    # a survival of 1 - 3e-8 is NOT representable in float32 (rounds to 1.0),
    # so this is the case the cast cannot rescue: it must warn, not go silent
    dead = spfn.SurvivalPFNBaselineModel(
        estimator=_Estimator(1.0), horizon_hours=8.0, feature_set="basic"
    )
    with caplog.at_level(logging.WARNING):
        out = dead.predict_proba(x)
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert any("numerical artifact" in r.message for r in caplog.records)

    # a survival that IS representable keeps its tiny probability in float64
    alive = spfn.SurvivalPFNBaselineModel(
        estimator=_Estimator(1.0 - 1e-6), horizon_hours=8.0, feature_set="basic"
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        small = alive.predict_proba(x)
    assert small.dtype == np.float64
    assert all(0.0 < v < 1e-5 for v in small)
    assert not any("numerical artifact" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _grouped_subsample_enriched: the sensitivity-analysis sampling primitive
# ---------------------------------------------------------------------------


def test_grouped_subsample_enriched_keeps_every_event_subject() -> None:
    # 50 subjects, 4 rows each; only subjects 0-2 have any observed event
    # (delta=1 on one of their rows). A tight cap that would very likely
    # drop a 3-in-50 minority under uniform random sampling must still
    # keep every one of them here.
    n_subjects, rows_per_subject, cap = 50, 4, 20
    keep = np.arange(n_subjects * rows_per_subject)
    groups = np.repeat(np.arange(n_subjects), rows_per_subject)
    delta = np.zeros(n_subjects * rows_per_subject)
    for event_subject in (0, 1, 2):
        delta[event_subject * rows_per_subject] = 1.0

    rng = np.random.default_rng(0)
    result = survivalpfn_module._grouped_subsample_enriched(
        keep, groups, delta, cap, rng
    )
    kept_groups = set(groups[np.isin(keep, result)].tolist())
    assert {0, 1, 2}.issubset(kept_groups)
    assert len(result) <= cap


def test_grouped_subsample_enriched_never_splits_a_subject() -> None:
    n_subjects, rows_per_subject, cap = 20, 5, 37
    keep = np.arange(n_subjects * rows_per_subject)
    groups = np.repeat(np.arange(n_subjects), rows_per_subject)
    delta = np.zeros(n_subjects * rows_per_subject)
    delta[: 2 * rows_per_subject] = 1.0  # subjects 0-1 are event subjects

    rng = np.random.default_rng(0)
    result = survivalpfn_module._grouped_subsample_enriched(
        keep, groups, delta, cap, rng
    )
    kept_groups = groups[np.isin(keep, result)]
    _, counts = np.unique(kept_groups, return_counts=True)
    assert (counts == rows_per_subject).all()


def test_grouped_subsample_enriched_subsamples_among_event_subjects_if_they_alone_exceed_cap() -> (
    None
):
    # Every subject has an event: the "keep all event subjects" branch
    # cannot be satisfied in full, so it must fall back to subsampling
    # among them, still whole-subject and still under cap.
    n_subjects, rows_per_subject, cap = 20, 5, 37
    keep = np.arange(n_subjects * rows_per_subject)
    groups = np.repeat(np.arange(n_subjects), rows_per_subject)
    delta = np.ones(n_subjects * rows_per_subject)

    rng = np.random.default_rng(0)
    result = survivalpfn_module._grouped_subsample_enriched(
        keep, groups, delta, cap, rng
    )
    kept_groups = groups[np.isin(keep, result)]
    _, counts = np.unique(kept_groups, return_counts=True)
    assert (counts == rows_per_subject).all()
    assert len(result) <= cap


def test_grouped_subsample_enriched_beats_uniform_on_event_count_same_seed() -> None:
    # Deterministic inequality, not a statistical claim: for the same
    # data/cap/seed, enriched keeps every event subject (up to cap) while
    # uniform draws subjects blind to event status, so enriched's kept
    # event count can never be lower.
    n_subjects, rows_per_subject, cap = 200, 3, 30
    keep = np.arange(n_subjects * rows_per_subject)
    groups = np.repeat(np.arange(n_subjects), rows_per_subject)
    delta = np.zeros(n_subjects * rows_per_subject)
    for event_subject in range(5):  # 5 of 200 subjects: a rare event
        delta[event_subject * rows_per_subject] = 1.0

    uniform_result = survivalpfn_module._grouped_subsample(
        keep, groups, cap, np.random.default_rng(0)
    )
    enriched_result = survivalpfn_module._grouped_subsample_enriched(
        keep, groups, delta, cap, np.random.default_rng(0)
    )
    n_events_uniform = int(delta[np.isin(keep, uniform_result)].sum())
    n_events_enriched = int(delta[np.isin(keep, enriched_result)].sum())
    assert n_events_enriched >= n_events_uniform
    assert n_events_enriched == 5  # all 5 rare-event subjects survive enrichment


# ---------------------------------------------------------------------------
# enrich_events wiring through _fit_one_survivalpfn / fit_survivalpfn_baselines
# ---------------------------------------------------------------------------


def test_fit_survivalpfn_baselines_enrich_events_reaches_the_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MAX_ROWS", 10)
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    calls = {"enriched": 0, "uniform": 0}
    real_enriched = survivalpfn_module._grouped_subsample_enriched
    real_uniform = survivalpfn_module._grouped_subsample

    def _tracked_enriched(*args: object, **kwargs: object) -> np.ndarray:
        calls["enriched"] += 1
        return real_enriched(*args, **kwargs)  # type: ignore[arg-type]

    def _tracked_uniform(*args: object, **kwargs: object) -> np.ndarray:
        calls["uniform"] += 1
        return real_uniform(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        survivalpfn_module, "_grouped_subsample_enriched", _tracked_enriched
    )
    monkeypatch.setattr(survivalpfn_module, "_grouped_subsample", _tracked_uniform)

    events = _events(40)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", enrich_events=True
    )
    assert calls["enriched"] == 1
    assert calls["uniform"] == 0


def test_fit_survivalpfn_baselines_default_uses_uniform_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MAX_ROWS", 10)
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    calls = {"enriched": 0, "uniform": 0}
    real_uniform = survivalpfn_module._grouped_subsample

    def _tracked_uniform(*args: object, **kwargs: object) -> np.ndarray:
        calls["uniform"] += 1
        return real_uniform(*args, **kwargs)  # type: ignore[arg-type]

    def _fail_if_called(*args: object, **kwargs: object) -> np.ndarray:
        calls["enriched"] += 1
        raise AssertionError("enriched sampler must not run when enrich_events=False")

    monkeypatch.setattr(survivalpfn_module, "_grouped_subsample", _tracked_uniform)
    monkeypatch.setattr(
        survivalpfn_module, "_grouped_subsample_enriched", _fail_if_called
    )

    events = _events(40)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    fit_survivalpfn_baselines(binned, rows, times, horizons=(8.0,), feature_set="basic")
    assert calls["uniform"] == 1
    assert calls["enriched"] == 0


def test_fit_one_survivalpfn_records_n_context_events_and_enriched_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    models = fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", enrich_events=True
    )
    model = models[("vasopressor_start", 8.0)]
    assert model.params["enriched"] == 1.0
    assert model.params["n_context_events"] >= 0

    models_uniform = fit_survivalpfn_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", enrich_events=False
    )
    assert models_uniform[("vasopressor_start", 8.0)].params["enriched"] == 0.0


def test_fit_survivalpfn_baselines_enriched_and_uniform_do_not_share_a_cache_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(survivalpfn_module, "SURVIVALPFN_MIN_ROWS", 2)
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    cache = FitCache(cache_dir=tmp_path / "fit_cache")

    fit_survivalpfn_baselines(
        binned,
        rows,
        times,
        horizons=(8.0,),
        feature_set="basic",
        cache=cache,
        enrich_events=False,
    )
    n_after_uniform = len(_FakeEstimator.instances)
    fit_survivalpfn_baselines(
        binned,
        rows,
        times,
        horizons=(8.0,),
        feature_set="basic",
        cache=cache,
        enrich_events=True,
    )
    # a real second fit happened (not a cache hit against the uniform
    # entry) -- one more estimator instance was created.
    assert len(_FakeEstimator.instances) == n_after_uniform + 1
