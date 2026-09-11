"""The shared single-lane streaming loop and its multi-horizon risk readout."""

import pytest
import torch

from odyssey.data.sequences import PatientSequence
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.case_study import extract_patient_case
from odyssey.inference.patient_stream import risk_within, stream_patient
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import (
    DEFAULT_TIME_BIN_EDGES_HOURS,
    probability_within,
)


VOCAB_SIZE = 30
EVENTS = ["vasopressor_start", "death"]


def _model() -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB_SIZE, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB_SIZE,
        num_concepts=3,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=EVENTS,
    )


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({f"LAB//{i}//": i for i in range(2, VOCAB_SIZE)})
    return Vocabulary(tokens)


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


def test_spans_tile_the_sequence_exactly_once() -> None:
    spans = list(stream_patient(_model(), _sequence(23), device="cpu", chunk_size=8))
    starts = [s.start for s in spans]
    assert starts == [0, *(s.start + s.n_real for s in spans[:-1])]
    assert sum(s.n_real for s in spans) == 23
    # only the final position lacks a next-token target
    flags = torch.cat([s.has_target for s in spans]).tolist()
    assert flags == [True] * 22 + [False]


def test_stop_after_ends_once_the_position_is_covered() -> None:
    spans = list(
        stream_patient(
            _model(), _sequence(40), device="cpu", chunk_size=8, stop_after=10
        )
    )
    last = spans[-1]
    assert last.start <= 10 < last.start + last.n_real


def test_empty_sequence_yields_no_spans() -> None:
    assert (
        list(stream_patient(_model(), _sequence(0), device="cpu", chunk_size=8)) == []
    )


def test_single_token_sequence_is_one_span_without_a_target() -> None:
    (span,) = stream_patient(_model(), _sequence(1), device="cpu", chunk_size=8)
    assert (span.start, span.n_real) == (0, 1)
    assert span.has_target.tolist() == [False]


@pytest.mark.parametrize("stop_after", [0, 7, 8, 22, 500])
def test_stop_after_never_drops_the_requested_position(stop_after: int) -> None:
    n = 23
    spans = list(
        stream_patient(
            _model(), _sequence(n), device="cpu", chunk_size=8, stop_after=stop_after
        )
    )
    covered = spans[-1].start + spans[-1].n_real
    assert covered > min(stop_after, n - 1)
    # stops at the first span that covers it, not later
    assert len(spans) == 1 or spans[-2].start + spans[-2].n_real <= stop_after


def test_prefix_outputs_do_not_depend_on_stop_after() -> None:
    """Stopping early must not change what was already computed (causality)."""
    model, seq = _model().eval(), _sequence(30)  # eval: no dropout noise
    full = list(stream_patient(model, seq, device="cpu", chunk_size=8))
    early = list(stream_patient(model, seq, device="cpu", chunk_size=8, stop_after=9))
    for a, b in zip(early, full):
        assert torch.equal(a.fwd.features[0, : a.n_real], b.fwd.features[0, : b.n_real])


def test_risk_within_with_no_horizons_is_an_empty_axis() -> None:
    hazards = torch.zeros(4, 2, len(DEFAULT_TIME_BIN_EDGES_HOURS) + 2)
    assert risk_within(hazards, DEFAULT_TIME_BIN_EDGES_HOURS, ()).shape == (4, 2, 0)


def test_case_trace_without_hazard_heads_has_no_horizon_risk() -> None:
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=VOCAB_SIZE, hidden_size=8, padding_idx=0),
        vocab_size=VOCAB_SIZE,
        num_concepts=3,
        embedding_dim=4,
        padding_idx=0,
    )
    trace = extract_patient_case(
        model, _sequence(10), _vocab(), ["a", "b", "c"], device="cpu", chunk_size=4
    )
    assert trace.event_risk_by_horizon == {} and trace.event_risk_24h == []


def test_risk_within_matches_probability_within_per_horizon() -> None:
    torch.manual_seed(1)
    hazards = torch.randn(5, 2, len(DEFAULT_TIME_BIN_EDGES_HOURS) + 2)
    stacked = risk_within(hazards, DEFAULT_TIME_BIN_EDGES_HOURS, (8.0, 24.0, 72.0))
    assert stacked.shape == (5, 2, 3)
    for j, h in enumerate((8.0, 24.0, 72.0)):
        assert torch.equal(
            stacked[..., j],
            probability_within(hazards, DEFAULT_TIME_BIN_EDGES_HOURS, h),
        )


def test_case_trace_reports_every_horizon_and_24h_matches_legacy_field() -> None:
    trace = extract_patient_case(
        _model(), _sequence(23), _vocab(), ["a", "b", "c"], device="cpu", chunk_size=8
    )
    assert set(trace.event_risk_by_horizon) == {"8h", "24h", "72h"}
    assert trace.event_risk_names == EVENTS
    assert trace.event_risk_by_horizon["24h"] == trace.event_risk_24h
    for key in ("8h", "24h", "72h"):
        assert len(trace.event_risk_by_horizon[key]) == 23
    # cumulative risk cannot fall as the horizon lengthens
    for p8, p72 in zip(
        trace.event_risk_by_horizon["8h"], trace.event_risk_by_horizon["72h"]
    ):
        assert all(a <= b + 1e-7 for a, b in zip(p8, p72))


def test_multi_horizon_risk_is_chunk_size_invariant_for_a_recurrence() -> None:
    model, seq, vocab = _model(), _sequence(23), _vocab()
    small = extract_patient_case(
        model, seq, vocab, ["a", "b", "c"], device="cpu", chunk_size=5
    )
    big = extract_patient_case(
        model, seq, vocab, ["a", "b", "c"], device="cpu", chunk_size=64
    )
    for key in ("8h", "24h", "72h"):
        for a, b in zip(
            small.event_risk_by_horizon[key], big.event_risk_by_horizon[key]
        ):
            assert a == pytest.approx(b, abs=1e-6)
