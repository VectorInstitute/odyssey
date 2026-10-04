"""Occlusion aimed at an event's risk, and the progress hook of the shared core."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from odyssey.data.concepts import concept_display_name
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.concept_edit_attribution import (
    event_occlusion_attribution,
    occlude_codes,
)
from odyssey.inference.counterfactual import score_record_at
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2024, 1, 1)
SBP = "LAB//220179//mmHg"
CREAT = "LAB//RESULT//50912//mg/dL"
CONCEPTS = ["hypotension", "tachycardia"]


def _events() -> pl.DataFrame:
    rows: list[tuple[int, str, datetime, float | None, int]] = [
        (1, "HOSPITAL_ADMISSION//EMERGENCY", T0, None, 101)
    ]
    for h in range(1, 30):
        rows.append((1, SBP, T0 + timedelta(hours=h), 120.0, 101))
        if h % 6 == 0:
            rows.append((1, CREAT, T0 + timedelta(hours=h), 1.0, 101))
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def _setup() -> tuple[ConceptBottleneckSequenceModel, Vocabulary, pl.DataFrame]:
    events = _events()
    vocab = Vocabulary.build(add_value_tokens(events)["code"].to_list(), min_count=1)
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=len(CONCEPTS),
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=["vasopressor_start", "death"],
    )
    return model, vocab, events


def test_event_target_baseline_is_the_scored_risk_and_progress_reports() -> None:
    model, vocab, events = _setup()
    index = T0 + timedelta(hours=20)
    progress: list[tuple[int, int]] = []
    results = event_occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        event="death",
        horizon_hours=24.0,
        concept_names=CONCEPTS,
        chunk_size=16,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    expected = score_record_at(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_names=CONCEPTS,
        chunk_size=16,
    ).event_risk["death"]["24h"]
    assert {r.code for r in results} == {SBP, CREAT, "HOSPITAL_ADMISSION//EMERGENCY"}
    assert all(r.baseline == pytest.approx(expected) for r in results)
    assert progress[-1] == (3, 3) and len(progress) == 3
    assert results == sorted(results, key=lambda r: abs(r.delta), reverse=True)


def test_unknown_event_is_refused_before_any_scoring() -> None:
    model, vocab, events = _setup()
    with pytest.raises(ValueError, match="not a hazard head"):
        event_occlusion_attribution(
            model,
            vocab,
            None,
            events,
            index_time=T0 + timedelta(hours=20),
            event="readmission_30d",
            horizon_hours=24.0,
            concept_names=CONCEPTS,
        )


def test_generic_core_accepts_any_readout_target() -> None:
    model, vocab, events = _setup()
    results = occlude_codes(
        model,
        vocab,
        None,
        events,
        index_time=T0 + timedelta(hours=20),
        value_of=lambda r: r.concept_probs["tachycardia"],
        concept_names=CONCEPTS,
        candidate_codes=[SBP, "NOT//PRESENT"],
        chunk_size=16,
    )
    assert [r.code for r in results] == [SBP]


def test_empty_candidate_pool_scores_only_the_baseline() -> None:
    model, vocab, events = _setup()
    progress: list[tuple[int, int]] = []
    results = occlude_codes(
        model,
        vocab,
        None,
        events,
        index_time=T0 + timedelta(hours=20),
        value_of=lambda r: r.event_risk["death"]["24h"],
        concept_names=CONCEPTS,
        candidate_codes=[],
        chunk_size=16,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert results == [] and progress == []


def test_progress_counts_candidates_that_had_nothing_to_remove() -> None:
    model, vocab, events = _setup()
    progress: list[tuple[int, int]] = []
    results = occlude_codes(
        model,
        vocab,
        None,
        events,
        index_time=T0 + timedelta(hours=20),
        value_of=lambda r: r.event_risk["death"]["24h"],
        concept_names=CONCEPTS,
        candidate_codes=["NOT//PRESENT", SBP],
        chunk_size=16,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(1, 2), (2, 2)]
    assert [r.code for r in results] == [SBP]


def test_lookback_window_limits_event_candidates() -> None:
    """Creatinine is drawn at hours 6, 12, 18; (18.5, 20] holds only SBP at 19, 20."""
    model, vocab, events = _setup()
    results = event_occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=T0 + timedelta(hours=20),
        event="vasopressor_start",
        horizon_hours=8.0,
        concept_names=CONCEPTS,
        lookback_hours=1.5,
        chunk_size=16,
    )
    assert {r.code for r in results} == {SBP}
    assert results[0].n_rows == 2


def test_model_without_hazard_heads_is_refused() -> None:
    events = _events()
    vocab = Vocabulary.build(add_value_tokens(events)["code"].to_list(), min_count=1)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=len(vocab), hidden_size=8, padding_idx=0),
        vocab_size=len(vocab),
        num_concepts=len(CONCEPTS),
        embedding_dim=4,
        padding_idx=0,
    )
    with pytest.raises(ValueError, match="not a hazard head"):
        event_occlusion_attribution(
            model,
            vocab,
            None,
            events,
            index_time=T0 + timedelta(hours=20),
            event="death",
            horizon_hours=24.0,
            concept_names=CONCEPTS,
        )


def test_concept_display_names_read_as_prose() -> None:
    assert concept_display_name("hypokalemia") == "severe hypokalemia"
    assert concept_display_name("sepsis3") == "sepsis3"
    assert concept_display_name("anemia") == "severe anemia"
    assert concept_display_name("shock") == "sustained hypotension (MAP)"
    assert concept_display_name("acute_kidney_injury") == "acute kidney injury"
