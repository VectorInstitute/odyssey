"""Tests for the summary head, its masked loss, and its wiring into the models."""

import torch

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import PackedLaneSampler
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
    ForecastObjective,
)
from odyssey.models.summary_head import SummaryHead, masked_huber_loss
from odyssey.training.summary_targets import SummaryTargets


VOCAB = 40
HIDDEN = 8
K = 5


def _seq(sid: int, n: int) -> PatientSequence:
    return PatientSequence(
        subject_id=sid,
        concept_ids=[1 + (i % (VOCAB - 1)) for i in range(n)],
        type_ids=[1] * n,
        time_stamps=[float(i) for i in range(n)],
        ages=[40.0] * n,
        visit_orders=[0] * n,
        visit_segments=[0] * n,
    )


def _backbone() -> TinyGRUBackbone:
    return TinyGRUBackbone(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=1, padding_idx=0
    )


def test_masked_huber_loss_averages_over_the_mask_only() -> None:
    pred = torch.zeros(1, 3, 2)
    target = torch.tensor([[[1.0, 100.0], [0.5, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[[True, False], [True, False], [False, False]]])
    loss = masked_huber_loss(pred, target, mask)
    # huber(1.0) = 0.5, huber(0.5) = 0.125 -> mean 0.3125; the 100 is masked
    assert loss.item() == 0.3125


def test_masked_huber_loss_is_zero_with_a_graph_when_mask_is_empty() -> None:
    pred = torch.zeros(1, 2, 2, requires_grad=True)
    loss = masked_huber_loss(
        pred, torch.ones(1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.bool)
    )
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None


def test_summary_head_shapes_linear_and_mlp() -> None:
    x = torch.randn(2, 3, HIDDEN)
    assert SummaryHead(HIDDEN, K)(x).shape == (2, 3, K)
    mlp = SummaryHead(HIDDEN, K, hidden_size=6)
    assert mlp(x).shape == (2, 3, K)
    assert mlp.hidden_size == 6


def test_models_without_the_head_report_a_zero_summary_loss() -> None:
    model = ConceptBottleneckSequenceModel(
        backbone=_backbone(),
        vocab_size=VOCAB,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
    )
    assert model.summary_head is None
    chunk = PackedLaneSampler(
        iter([_seq(1, 6)]), num_lanes=1, chunk_size=6
    ).next_chunk()
    total, components, _ = model.compute_streaming_loss(chunk, {1: torch.zeros(2)})
    assert components["summary_loss"].item() == 0.0
    assert torch.isfinite(total)


def test_summary_loss_trains_the_state_and_is_weighted() -> None:
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=_backbone(),
        vocab_size=VOCAB,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        summary_targets=K,
    )
    assert model.summary_head is not None
    model.eval()  # dropout would otherwise make the two passes differ
    chunk = PackedLaneSampler(
        iter([_seq(1, 6)]), num_lanes=1, chunk_size=6
    ).next_chunk()
    values = torch.randn(1, 6, K)
    mask = torch.zeros(1, 6, K, dtype=torch.bool)
    mask[0, 3] = True
    targets = SummaryTargets(values=values, mask=mask)
    zero_w = ForecastObjective(summary_weight=0.0)
    one_w = ForecastObjective(summary_weight=1.0)
    total0, comp0, _ = model.compute_streaming_loss(
        chunk, {1: torch.zeros(2)}, objective=zero_w, summary_targets=targets
    )
    total1, comp1, _ = model.compute_streaming_loss(
        chunk, {1: torch.zeros(2)}, objective=one_w, summary_targets=targets
    )
    assert comp0["summary_loss"].item() > 0.0
    assert total1.item() > total0.item()
    assert abs((total1 - total0).item() - comp1["summary_loss"].item()) < 1e-5
    total1.backward()
    assert model.backbone.embeddings.embeddings.word_embeddings.weight.grad is not None


def test_baseline_model_accepts_summary_targets_too() -> None:
    model = BaselineSequenceModel(
        backbone=_backbone(), vocab_size=VOCAB, padding_idx=0, summary_targets=K
    )
    chunk = PackedLaneSampler(
        iter([_seq(1, 4)]), num_lanes=1, chunk_size=4
    ).next_chunk()
    targets = SummaryTargets(
        values=torch.zeros(1, 4, K), mask=torch.ones(1, 4, K, dtype=torch.bool)
    )
    _, components, _ = model.compute_streaming_loss(
        chunk, objective=ForecastObjective(summary_weight=1.0), summary_targets=targets
    )
    assert "summary_loss" in components
