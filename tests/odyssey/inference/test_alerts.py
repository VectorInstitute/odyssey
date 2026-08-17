"""Tests for the alert (time-to-event) evaluation harness."""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import pytest
import torch

from odyssey.data.alert_events import (
    ALERT_EVENTS,
    AlertEvent,
    EventTimes,
    all_event_times,
)
from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.alerts import (
    GBM_MIN_OBSERVED,
    IndexRow,
    _index_rows_from_events,
    _tune_gbm,
    _visit_starts,
    baseline_features,
    collect_model_scores,
    features_for_events,
    fit_baselines,
    index_row_table,
    outcome_at_horizon,
    score_alerts,
    sparse_columns,
)
from odyssey.inference.baseline_features import feature_names
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2024, 1, 1)


def _events(n_subjects: int = 24) -> pl.DataFrame:
    """Build hourly heart-rate readings with a planted deterioration signal.

    Every other subject spikes at hour 12 and starts norepinephrine at
    hour 14; every fourth also gets an ICU admission at hour 6.
    """
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in range(1, n_subjects + 1):
        hadm = 1000 + sid
        for h in range(24):
            hr = 80.0
            if sid % 2 == 0 and h >= 12:
                hr = 130.0
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
        if sid % 4 == 0:
            rows.append(
                (sid, "ICU_ADMISSION//MICU", T0 + timedelta(hours=6), None, hadm)
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


def test_event_times_and_outcomes() -> None:
    events = _events(8)
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")
    vaso = times["vasopressor_start"]
    # subject 2 starts norepinephrine at hour 14; subject 1 never
    assert vaso.onset[(2, 1002)] == 14.0
    assert (1, 1001) not in vaso.onset
    icu = times["icu_admission"]
    assert icu.onset[(4, 1004)] == 6.0
    # outcomes for subject 2 at t=10: within 8h -> yes; at t=2, 8h -> no
    row = IndexRow(2, 1002, 10.0)
    assert outcome_at_horizon(row, vaso, 8.0) == 1
    assert outcome_at_horizon(IndexRow(2, 1002, 2.0), vaso, 8.0) == 0
    # after onset: not at risk
    assert outcome_at_horizon(IndexRow(2, 1002, 15.0), vaso, 8.0) is None
    # subject 1 at t=20 with 8h horizon: follow-up ends at 23h -> censored
    assert outcome_at_horizon(IndexRow(1, 1001, 20.0), vaso, 8.0) is None
    assert outcome_at_horizon(IndexRow(1, 1001, 10.0), vaso, 8.0) == 0


def test_harness_end_to_end_with_planted_signal() -> None:
    events = _events(24)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _model(len(vocab), len(concepts))
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")
    rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
    )
    # landmarks every 4h over 24h -> ~6 per visit
    assert len(rows["vasopressor_start"]) >= 24 * 5
    assert all(
        "concept" in r.scores and "next_mass" in r.scores
        for r in rows["vasopressor_start"]
    )
    assert all("concept" not in r.scores for r in rows["icu_admission"])

    # baseline: fit on the same synthetic data (a separate split in real use)
    train_rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, train_rows, times, horizons=(8.0,), feature_set="basic", tune=False
    )
    feats = {name: baseline_features(binned, rs) for name, rs in rows.items() if rs}
    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
    )
    by = {(r.event, r.scorer): r for r in results}
    # the planted signal (HR HIGH from hour 12) separates the landmark-12
    # positives; the landmark-8 positives (onset at 14, within 8h) still
    # look normal, so the ceiling here is well short of 1 but far above chance
    gbm = by[("vasopressor_start", "baseline_gbm")]
    assert gbm.auroc is not None and gbm.auroc > 0.75
    assert gbm.brier is not None and gbm.calibration
    # untrained model scores exist and are valid AUROCs
    assert 0.0 <= by[("vasopressor_start", "concept")].auroc <= 1.0
    assert 0.0 <= by[("vasopressor_start", "next_mass")].auroc <= 1.0
    # censoring is counted, never silently dropped
    assert gbm.n_censored > 0


def test_baseline_features_shape_and_content() -> None:
    events = _events(4)
    binned = add_value_tokens(events)
    rows = [IndexRow(2, 1002, 13.0), IndexRow(1, 1001, 1.0)]
    x = baseline_features(binned, rows)
    assert x.shape[0] == 2
    # hours since visit start
    assert x[0, 0] == 13.0 and x[1, 0] == 1.0
    # subject 2 at hour 13 has HR HIGH as latest bin (ordinal 1)
    prefixes_start = 2
    assert 1.0 in x[0, prefixes_start:]
    assert not np.isnan(x[0, -6:]).any()  # counts present


def test_subject_scoped_event_ignores_visit() -> None:
    times = EventTimes(
        onset={(7, -1): 30.0}, censor={(7, -1): 40.0}, subject_scoped=True
    )
    assert outcome_at_horizon(IndexRow(7, 999, 25.0), times, 8.0) == 1
    assert outcome_at_horizon(IndexRow(7, 999, 10.0), times, 8.0) == 0
    assert AlertEvent(
        "death", code_prefix="MEDS_DEATH", subject_scoped=True
    ).subject_scoped


def test_hazard_scorer_reports_probability_metrics() -> None:
    events = _events(16)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=len(concepts),
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=[a.name for a in ALERT_EVENTS],
    )
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")
    rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
        horizons=(8.0, 24.0),
    )
    assert all(
        "hazard@8h" in r.scores and "hazard@24h" in r.scores for r in rows["death"]
    )
    results = score_alerts(rows, times, horizons=(8.0, 24.0))
    hazard = [r for r in results if r.scorer == "hazard"]
    assert hazard, "hazard scorer should be reported"
    for r in hazard:
        assert r.brier is not None and r.calibration
        assert 0.0 <= r.auroc <= 1.0
    # each hazard row scores its own horizon only: no 8h probability at 24h
    assert {(r.event, r.horizon_hours) for r in hazard} <= {
        (e, h) for e in rows for h in (8.0, 24.0)
    }


def test_icu_admission_prefix_excludes_admission_measurements() -> None:
    icu = next(a for a in ALERT_EVENTS if a.name == "icu_admission")
    assert icu.code_prefix is not None
    assert "ICU_ADMISSION//MICU".startswith(icu.code_prefix)
    assert "ICU_ADMISSION////admit".startswith(icu.code_prefix)
    assert not "ICU_ADMISSION_WEIGHT".startswith(icu.code_prefix)
    assert not "ICU_ADMISSION_HEIGHT".startswith(icu.code_prefix)


def test_strong_baseline_fits_tunes_and_records_metadata() -> None:
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="strong", tune=True
    )
    model = baselines[("vasopressor_start", 8.0)]
    assert model.feature_set == "strong"
    assert model.n_features == len(feature_names())
    assert model.params["n_rounds"] >= 1
    feats = features_for_events(binned, rows, feature_set="strong")
    assert feats["vasopressor_start"].shape[1] == len(feature_names())
    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
    )
    gbm = next(
        r
        for r in results
        if r.scorer == "baseline_gbm" and r.event == "vasopressor_start"
    )
    assert gbm.auroc is not None and gbm.auroc > 0.75
    assert gbm.baseline_feature_set == "strong"
    assert gbm.baseline_n_features == len(feature_names())
    assert gbm.baseline_params and "learning_rate" in gbm.baseline_params


def test_unknown_feature_set_is_rejected() -> None:
    events = _events(n_subjects=4)
    binned = add_value_tokens(events, None, source="mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    with pytest.raises(ValueError):
        features_for_events(binned, rows, feature_set="bogus")


def test_index_row_table_has_scores_outcomes_and_gbm_columns() -> None:
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    for r in rows["vasopressor_start"]:
        r.scores["next_mass"] = 0.5
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="strong", tune=False
    )
    feats = features_for_events(binned, rows, feature_set="strong")
    names = feature_names()
    table = index_row_table(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
        context_columns={
            k: v[:, [names.index("hours_into_visit")]] for k, v in feats.items()
        },
        context_names=["hours_into_visit"],
    )
    assert {"event", "subject_id", "visit_id", "time_hours", "y@8h"} <= set(
        table.columns
    )
    vaso = table.filter(pl.col("event") == "vasopressor_start")
    assert vaso["next_mass"].drop_nulls().len() == vaso.height
    assert (
        "gbm@8h" in table.columns and vaso["gbm@8h"].drop_nulls().len() == vaso.height
    )
    assert "ctx.hours_into_visit" in table.columns
    assert set(vaso["y@8h"].drop_nulls().unique().to_list()) <= {0.0, 1.0}
    assert vaso["y@8h"].null_count() > 0  # censored / not-at-risk rows are null


def test_tuning_survives_a_column_missing_only_inside_the_training_fold() -> None:
    rng = np.random.default_rng(0)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    # column 3 is observed only for subjects that land in the validation fold
    # (replicating _tune_gbm's seeded group shuffle), so the training fold
    # sees an all-NaN column that the full fit set does not
    shuffled = np.unique(groups)
    np.random.default_rng(0).shuffle(shuffled)
    val_groups = shuffled[: max(1, int(round(0.1 * len(shuffled))))]
    x[:, 3] = np.nan
    x[np.isin(groups, val_groups), 3] = 1.0
    params, n_rounds = _tune_gbm(x, y, groups, seed=0)
    assert n_rounds >= 1 and "learning_rate" in params


def test_sparse_columns_are_filled_at_fit_and_predict() -> None:
    x = np.full((1000, 3), np.nan, dtype=np.float32)
    x[:, 0] = 1.0
    x[:GBM_MIN_OBSERVED, 1] = 2.0  # exactly the minimum: kept
    x[: GBM_MIN_OBSERVED - 1, 2] = 3.0  # one short: filled
    assert sparse_columns(x).tolist() == [False, False, True]
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="strong", tune=False
    )
    model = baselines[("vasopressor_start", 8.0)]
    # the synthetic record never has most of the panel: those columns are filled
    assert model.fill_columns.sum() > 0
    feats = features_for_events(binned, rows, feature_set="strong")
    p = model.predict_proba(feats["vasopressor_start"])
    assert np.isfinite(p).all()
