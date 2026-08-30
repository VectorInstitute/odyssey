"""Tests for odyssey.inference.probe_baseline.

Covers the pieces added for the EHRSHOT-style probe benchmark (Amrit's
2026-08-28 redirect: representation quality under a frozen probe, not more
trained alert heads): ProbeBaselineModel/fit_binary_probe/fit_probe_baselines
(mirrors the built-in GBM's fit shape and degenerate-label guards),
probe_features_by_event (the extra_baselines features shape), long_los_task
(the static-label snapshot path, deliberately NOT expressed via EventTimes),
fit_gbm_for_binary_label, and one integration test proving probe baselines
plug into score_alerts's existing extra_baselines hook cleanly, mirroring
tests/odyssey/inference/test_alerts.py's own
test_score_alerts_extra_baselines_scores_alongside_the_gbm.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch
from sklearn.metrics import roc_auc_score

from odyssey.data.alert_events import PROBE_EVENTS, EventTimes, visit_envelope
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.alerts import (
    IndexRow,
    _index_rows_from_events,
    _positive_class_proba,
    features_for_events,
    fit_baselines,
    score_alerts,
)
from odyssey.inference.embedding_probe import collect_embeddings
from odyssey.inference.probe_baseline import (
    MIN_FIT_ROWS,
    fit_binary_probe,
    fit_gbm_for_binary_label,
    fit_probe_baselines,
    long_los_task,
    probe_features_by_event,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


T0 = datetime(2024, 1, 1)
_EventRow = tuple[int, str, datetime, float | None, int]
_KEY = tuple[int, int, float]


def _separable_xy(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs in 4D, an easy binary classification."""
    rng = np.random.default_rng(seed)
    half = n // 2
    x = np.concatenate(
        [
            rng.normal(-3.0, 1.0, size=(half, 4)),
            rng.normal(3.0, 1.0, size=(n - half, 4)),
        ]
    )
    y = np.concatenate([np.zeros(half), np.ones(n - half)]).astype(int)
    return x, y


# --------------------------------------------------------------------------
# ProbeBaselineModel / fit_binary_probe
# --------------------------------------------------------------------------


def test_fit_binary_probe_discriminates_a_separable_signal() -> None:
    x_train, y_train = _separable_xy(200, seed=0)
    x_test, y_test = _separable_xy(100, seed=1)

    model = fit_binary_probe(x_train, y_train, feature_set="post_bottleneck")
    proba = model.predict_proba(x_test)

    assert proba.shape == (100,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert roc_auc_score(y_test, proba) > 0.95


def test_probe_baseline_model_satisfies_the_scored_baseline_interface() -> None:
    """feature_set/n_features/params/predict_proba -- what score_alerts reads."""
    x_train, y_train = _separable_xy(60, seed=0)
    model = fit_binary_probe(x_train, y_train, feature_set="pre_bottleneck")

    assert model.feature_set == "pre_bottleneck"
    assert model.n_features == 4
    assert isinstance(model.params, dict)
    assert model.predict_proba(x_train).shape == (60,)


# --------------------------------------------------------------------------
# fit_probe_baselines
# --------------------------------------------------------------------------


def test_fit_probe_baselines_skips_cells_with_too_few_at_risk_rows() -> None:
    """Below MIN_FIT_ROWS at-risk rows -- mirrors _fit_baseline_grid's own guard."""
    rows = {"anemia": [IndexRow(i, i, 4.0) for i in range(MIN_FIT_ROWS - 1)]}
    times = {
        "anemia": EventTimes(
            onset={},
            censor={(i, i): 1000.0 for i in range(MIN_FIT_ROWS - 1)},
            subject_scoped=False,
        )
    }
    embeddings = {"anemia": np.zeros((MIN_FIT_ROWS - 1, 4))}

    models = fit_probe_baselines(rows, embeddings, times, horizons=(8.0,))

    assert models == {}


def test_fit_probe_baselines_skips_cells_with_single_class_labels() -> None:
    """Enough rows, but every outcome is the same class -- AUROC would be undefined."""
    keys: list[_KEY] = [(1, 1001, float(h)) for h in range(60)]
    rows = {"anemia": [IndexRow(s, v, t) for s, v, t in keys]}
    # never onsets, follow-up always reaches past the horizon -> every row "0"
    times = {
        "anemia": EventTimes(onset={}, censor={(1, 1001): 1000.0}, subject_scoped=False)
    }
    embeddings = {"anemia": np.random.default_rng(0).normal(size=(60, 4))}

    models = fit_probe_baselines(rows, embeddings, times, horizons=(8.0,))

    assert models == {}


def test_fit_probe_baselines_fits_when_enough_rows_of_both_classes() -> None:
    keys: list[_KEY] = []
    onset: dict[tuple[int, int], float] = {}
    censor: dict[tuple[int, int], float] = {}
    for sid in range(1, 31):
        key = (sid, 1000 + sid)
        censor[key] = 100.0
        if sid % 2 == 0:  # half the cohort onsets at 14h -> both classes present
            onset[key] = 14.0
        for h in range(4):
            keys.append((sid, 1000 + sid, float(h * 6)))
    rows = {"anemia": [IndexRow(s, v, t) for s, v, t in keys]}
    times = {"anemia": EventTimes(onset=onset, censor=censor, subject_scoped=False)}
    rng = np.random.default_rng(0)
    embeddings = {"anemia": rng.normal(size=(len(keys), 4))}

    models = fit_probe_baselines(rows, embeddings, times, horizons=(8.0, 24.0))

    assert set(models) <= {("anemia", 8.0), ("anemia", 24.0)}
    assert len(models) > 0
    for model in models.values():
        assert model.feature_set == "post_bottleneck"  # the function's default


def _small_cohort(n_subjects: int) -> pl.DataFrame:
    """Hourly lab readings for a small cohort -- a generic model-driven fixture.

    Not shaped to trigger any particular PROBE_EVENTS concept (the real
    concept-labeling pipeline is exercised, but on real data whether
    "anemia" et al. actually onset for this synthetic cohort is incidental
    -- the integration test below only checks that fitted probes, if any,
    plug into score_alerts cleanly, not that a specific task is fittable).
    """
    rows: list[_EventRow] = []
    for sid in range(1, n_subjects + 1):
        hadm = 1000 + sid
        for h in range(24):
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), 80.0, hadm))
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


# --------------------------------------------------------------------------
# probe_features_by_event
# --------------------------------------------------------------------------


def test_probe_features_by_event_reuses_one_array_across_every_task() -> None:
    keys: list[_KEY] = [(1, 1001, 4.0), (1, 1001, 8.0)]
    embeddings = np.arange(8).reshape(2, 4).astype(np.float32)

    features = probe_features_by_event(PROBE_EVENTS, keys, embeddings)

    assert set(features) == {a.name for a in PROBE_EVENTS}
    for arr in features.values():
        assert arr is embeddings  # same object, not a copy, by design


def test_probe_features_by_event_raises_on_length_mismatch() -> None:
    keys: list[_KEY] = [(1, 1001, 4.0)]
    embeddings = np.zeros((2, 4))  # one too many rows

    with pytest.raises(ValueError, match="length mismatch"):
        probe_features_by_event(PROBE_EVENTS, keys, embeddings)


# --------------------------------------------------------------------------
# long_los_task
# --------------------------------------------------------------------------


def test_long_los_task_labels_by_total_visit_span_not_the_snapshot_time() -> None:
    """A visit lasting > 168h is positive even scored from a 24h snapshot."""
    envelope = {
        (1, 100): (0.0, 200.0),  # long stay: 200h > 168h threshold
        (2, 200): (0.0, 48.0),  # short stay: well under threshold
    }
    keys: list[_KEY] = [(1, 100, 24.0), (2, 200, 24.0)]
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

    filtered_keys, filtered_x, y = long_los_task(keys, embeddings, envelope)

    assert filtered_keys == keys
    assert list(y) == [1.0, 0.0]
    np.testing.assert_array_equal(filtered_x, embeddings)


def test_long_los_task_keeps_only_the_earliest_row_inside_the_snapshot_band() -> None:
    """Two candidate landmark rows for one visit inside the band -> earliest wins."""
    envelope = {(1, 100): (0.0, 200.0)}
    keys: list[_KEY] = [
        (1, 100, 26.0),
        (1, 100, 22.0),
        (1, 100, 40.0),
    ]  # 40h outside band
    embeddings = np.array([[1.0], [2.0], [3.0]])

    filtered_keys, filtered_x, y = long_los_task(keys, embeddings, envelope)

    assert filtered_keys == [(1, 100, 22.0)]
    np.testing.assert_array_equal(filtered_x, [[2.0]])
    assert list(y) == [1.0]


def test_long_los_task_drops_rows_with_no_envelope_entry() -> None:
    """A key whose visit never appears in ``envelope`` is silently excluded."""
    envelope: dict[tuple[int, int], tuple[float, float]] = {}
    keys: list[_KEY] = [(1, 100, 24.0)]
    embeddings = np.array([[1.0]])

    filtered_keys, filtered_x, y = long_los_task(keys, embeddings, envelope)

    assert filtered_keys == []
    assert filtered_x.shape == (0, 1)
    assert y.shape == (0,)


def test_long_los_task_drops_rows_outside_the_snapshot_band() -> None:
    envelope = {(1, 100): (0.0, 200.0)}
    keys: list[_KEY] = [(1, 100, 5.0), (1, 100, 100.0)]  # both outside (20, 28)
    embeddings = np.array([[1.0], [2.0]])

    filtered_keys, _filtered_x, y = long_los_task(keys, embeddings, envelope)

    assert filtered_keys == []
    assert y.shape == (0,)


def test_visit_envelope_feeds_long_los_task_end_to_end() -> None:
    """visit_envelope's real output plugs into long_los_task without adaptation."""
    rows: list[_EventRow] = [
        (1, "LAB//220045//bpm", T0, 80.0, 100),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=24), 80.0, 100),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=200), 80.0, 100),  # long stay
    ]
    events = pl.DataFrame(
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
    envelope = visit_envelope(events)
    keys: list[_KEY] = [(1, 100, 24.0)]
    embeddings = np.array([[1.0]])

    filtered_keys, _x, y = long_los_task(keys, embeddings, envelope)

    assert filtered_keys == [(1, 100, 24.0)]
    assert list(y) == [1.0]


# --------------------------------------------------------------------------
# fit_gbm_for_binary_label
# --------------------------------------------------------------------------


def test_fit_gbm_for_binary_label_discriminates_a_separable_signal() -> None:
    x_train, y_train = _separable_xy(200, seed=0)
    x_test, y_test = _separable_xy(100, seed=1)

    clf = fit_gbm_for_binary_label(x_train, y_train)
    proba = _positive_class_proba(clf, clf.predict_proba(x_test))

    assert roc_auc_score(y_test, proba) > 0.9


# --------------------------------------------------------------------------
# Integration: probe baselines through score_alerts's extra_baselines hook
# --------------------------------------------------------------------------


def _vocab(events_binned: pl.DataFrame) -> Vocabulary:
    return Vocabulary.build(events_binned["code"].to_list(), min_count=1)


def _model(vocab_size: int, num_concepts: int) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=num_concepts,
        embedding_dim=4,
        padding_idx=0,
    )


def test_probe_baselines_plug_into_score_alerts_extra_baselines_hook() -> None:
    """A probe fit on real streamed embeddings scores alongside the built-in GBM.

    Mirrors test_alerts.py's test_score_alerts_extra_baselines_scores_
    alongside_the_gbm, but with a REAL fitted ProbeBaselineModel (from
    embeddings a real, if tiny, model produced) instead of a fake constant
    predictor -- the end-to-end path this module exists for.

    ``times`` is built directly (not via the real anemia/hyperkalemia/...
    concept-labeling pipeline): ``_small_cohort`` has no lab values that
    would ever cross any of those clinical thresholds, so every real label
    would be a single-class "never triggers" and score_alerts would skip
    every cell (correctly -- see the degenerate-label tests above), scoring
    nothing and defeating the point of this test. What's under test here is
    the extra_baselines PLUMBING, not concept-triggering itself, so a
    synthetic onset (half the cohort "triggers" at 14h) is deliberate and
    sufficient.
    """
    events = _small_cohort(n_subjects=20)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    model = _model(len(vocab), num_concepts=len(PROBE_EVENTS))

    keys, _pre, post, _, _, _ = collect_embeddings(
        model,
        binned,
        vocab,
        landmark_alerts=PROBE_EVENTS,
        visit_end_alerts=[],
        visit_start={
            (int(s), int(v)): 0.0
            for s, v in zip(events["subject_id"], events["hadm_id"])
        },
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
    )
    rows = {a.name: [IndexRow(s, v, t) for s, v, t in keys] for a in PROBE_EVENTS}
    onset = {(sid, 1000 + sid): 14.0 for sid in range(1, 21) if sid % 2 == 0}
    censor = {(sid, 1000 + sid): 23.0 for sid in range(1, 21)}
    times = {
        a.name: EventTimes(onset=dict(onset), censor=dict(censor), subject_scoped=False)
        for a in PROBE_EVENTS
    }

    train_rows = _index_rows_from_events(binned, PROBE_EVENTS, landmark_hours=4.0)
    gbm_models = fit_baselines(
        binned, train_rows, times, horizons=(8.0,), feature_set="basic", tune=False
    )
    gbm_features = features_for_events(binned, rows, feature_set="basic")

    probe_models = fit_probe_baselines(
        rows, probe_features_by_event(PROBE_EVENTS, keys, post), times, horizons=(8.0,)
    )
    probe_features = probe_features_by_event(PROBE_EVENTS, keys, post)

    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        baselines=gbm_models,
        baseline_features_by_event=gbm_features,
        extra_baselines={"probe_post": (probe_models, probe_features)},
    )

    by_scorer = {r.scorer for r in results}
    # both the built-in GBM and the new probe family show up, neither
    # disturbing the other -- the whole point of the extra_baselines hook.
    assert "baseline_gbm" in by_scorer
    assert "probe_post" in by_scorer
    probe_rows = [r for r in results if r.scorer == "probe_post"]
    assert probe_rows
    assert all(r.brier is not None and r.calibration for r in probe_rows)
