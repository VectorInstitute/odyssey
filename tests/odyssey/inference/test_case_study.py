"""CPU-testable pieces of the case-study module.

extract_patient_case is exercised here with a TinyGRUBackbone; the
real-EHRHybridBackbone/CUDA end-to-end path is in test_case_study_gpu.py.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest
import torch

import odyssey.inference.case_study as case_study_module
from odyssey.data.sequences import PatientSequence
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.case_study import (
    _parse_args,
    build_case_studies,
    extract_patient_case,
    select_diverse_cases,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.train import TrainingConfig


T0 = datetime(2024, 1, 1, 0, 0)

_EventRow = Tuple[int, str, datetime, Optional[float], Optional[int]]


def _events(rows: List[_EventRow]) -> pl.DataFrame:
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


def _patient_events(subject_id: int, n_events: int) -> List[_EventRow]:
    return [
        (subject_id, f"DIAGNOSIS//{i}", T0 + timedelta(hours=i), None, None)
        for i in range(n_events)
    ]


def test_select_diverse_cases_respects_min_events() -> None:
    events = _events(_patient_events(1, 5) + _patient_events(2, 20))
    selected = select_diverse_cases(events, {}, n_cases=15, min_events=10)
    assert selected == [2]


def test_select_diverse_cases_returns_at_most_n_cases() -> None:
    rows: List[_EventRow] = []
    for sid in range(30):
        rows += _patient_events(sid, 15)
    events = _events(rows)
    selected = select_diverse_cases(events, {}, n_cases=15, min_events=10)
    assert len(selected) == 15
    assert len(set(selected)) == 15  # no duplicates


def test_select_diverse_cases_spans_both_short_and_long_stays() -> None:
    rows: List[_EventRow] = []
    for sid in range(10):
        rows += _patient_events(sid, 10)  # short
    for sid in range(10, 20):
        rows += _patient_events(sid, 500)  # long
    events = _events(rows)

    selected = select_diverse_cases(events, {}, n_cases=10, min_events=5)

    assert any(sid < 10 for sid in selected)
    assert any(sid >= 10 for sid in selected)


def test_select_diverse_cases_spans_concept_triggered_and_not() -> None:
    rows: List[_EventRow] = []
    for sid in range(20):
        rows += _patient_events(sid, 30)
    events = _events(rows)
    concept_labels = {
        sid: torch.tensor([1.0, 1.0, 1.0])
        if sid % 2 == 0
        else torch.tensor([0.0, 0.0, 0.0])
        for sid in range(20)
    }

    selected = select_diverse_cases(events, concept_labels, n_cases=10, min_events=5)

    assert any(sid % 2 == 0 for sid in selected)
    assert any(sid % 2 == 1 for sid in selected)


def test_select_diverse_cases_is_deterministic_given_a_seed() -> None:
    rows: List[_EventRow] = []
    for sid in range(20):
        rows += _patient_events(sid, 30)
    events = _events(rows)

    a = select_diverse_cases(events, {}, n_cases=10, min_events=5, seed=7)
    b = select_diverse_cases(events, {}, n_cases=10, min_events=5, seed=7)
    assert a == b


def test_select_diverse_cases_empty_when_nobody_meets_min_events() -> None:
    events = _events(_patient_events(1, 3))
    assert select_diverse_cases(events, {}, n_cases=15, min_events=10) == []


# ---------------------------------------------------------------------------
# build_case_studies: backbone="transformer" gate
# ---------------------------------------------------------------------------


def test_build_case_studies_rejects_transformer_backbone_before_touching_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = ConceptBottleneckSequenceModel(
        TinyGRUBackbone(vocab_size=10, hidden_size=4),
        vocab_size=10,
        num_concepts=2,
        embedding_dim=4,
    )
    fake_config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
    )
    monkeypatch.setattr(
        case_study_module,
        "load_run",
        lambda *a, **k: (fake_model, object(), object(), fake_config),
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not read shards before the backbone gate fires")

    monkeypatch.setattr(case_study_module, "load_meds_shards", _boom)

    with pytest.raises(NotImplementedError, match="backbone='transformer'"):
        build_case_studies("/runs/x", "/data/held_out")


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_checkpoint_to_best_and_15_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            "/out/cases.json",
        ],
    )
    args = _parse_args()
    assert args.checkpoint_path == Path("/runs/x/checkpoint_best.pt")
    assert args.n_cases == 15
    assert args.max_shards is None


# ---------------------------------------------------------------------------
# extract_patient_case: streaming trace extraction (CPU, TinyGRUBackbone)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 30
NUM_CONCEPTS = 3


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({f"LAB//{i}//": i for i in range(2, VOCAB_SIZE)})
    return Vocabulary(tokens)


def _model() -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB_SIZE, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        padding_idx=0,
    )


def _sequence(n: int) -> PatientSequence:
    return PatientSequence(
        subject_id=7,
        concept_ids=[2 + (i % (VOCAB_SIZE - 2)) for i in range(n)],
        type_ids=[1] * n,
        time_stamps=[float(i) for i in range(n)],
        ages=[50.0] * n,
        visit_orders=[0] * n,
        visit_segments=[0] * n,
    )


def test_streaming_trace_covers_every_position_exactly_once() -> None:
    n = 25
    trace = extract_patient_case(
        _model(), _sequence(n), _vocab(), ["a", "b", "c"], device="cpu", chunk_size=8
    )
    assert len(trace.input_codes) == n
    assert len(trace.concept_probs) == n
    assert len(trace.predicted_top_k) == n
    # Only the final position lacks a next-token target.
    assert trace.true_next_code[-1] is None
    assert trace.predicted_top_k[-1] == []
    assert all(code is not None for code in trace.true_next_code[:-1])
    assert all(rank is not None for rank in trace.true_next_rank[:-1])


def test_trace_works_with_recency_features_and_event_heads() -> None:
    """Event heads read the augmented feature vector, not the raw bottleneck.

    Under ``recency_features=True`` the hazard heads take base+RECENCY_DIM
    inputs; calling them on the raw bottleneck output crashes with a shape
    mismatch (the v9-MIMIC case-study failure). The trace must route head
    inputs through the same feature assembly every other inference
    consumer uses.
    """
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB_SIZE, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=[24.0, 72.0],
        event_names=["death", "icu_admission"],
        recency_features=True,
    )
    n = 12
    trace = extract_patient_case(
        model, _sequence(n), _vocab(), ["a", "b", "c"], device="cpu", chunk_size=5
    )
    assert trace.event_risk_names == ["death", "icu_admission"]
    assert len(trace.event_risk_24h) == n
    assert all(len(risks) == 2 for risks in trace.event_risk_24h)


def test_chunked_trace_equals_unchunked_for_an_exact_recurrence() -> None:
    """Chunk stitching is seamless whatever the chunk size.

    For a GRU the carried state is the exact sufficient statistic
    (unlike windowed attention), so the streamed trace must be
    numerically identical however the sequence is chunked.
    """
    model = _model()
    seq = _sequence(23)
    vocab = _vocab()
    names = ["a", "b", "c"]
    small = extract_patient_case(model, seq, vocab, names, device="cpu", chunk_size=5)
    big = extract_patient_case(model, seq, vocab, names, device="cpu", chunk_size=64)
    assert small.true_next_rank == big.true_next_rank
    for probs_small, probs_big in zip(small.concept_probs, big.concept_probs):
        assert probs_small == pytest.approx(probs_big, abs=1e-6)
    for top_small, top_big in zip(small.predicted_top_k, big.predicted_top_k):
        assert [c for c, _ in top_small] == [c for c, _ in top_big]
