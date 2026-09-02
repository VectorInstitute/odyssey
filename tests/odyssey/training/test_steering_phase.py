"""Steering phases fire on schedule, pick attributed positions, add two losses."""

import torch
from torch import nn

from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.steering_phase import SteeringSchedule, choose_injection


HIDDEN, K, VOCAB = 8, 3, 11


def _model() -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=VOCAB, hidden_size=HIDDEN, padding_idx=0),
        vocab_size=VOCAB,
        num_concepts=K,
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind="decomposed",
        event_names=["death"],
    )
    # The tiny GRU has no block list; give the hook something to attach to
    # so the plumbing can be exercised (the hook itself is tested elsewhere).
    model.backbone.layers = nn.ModuleList([nn.Identity()])  # type: ignore[attr-defined]
    return model


def _chunk() -> StreamingChunk:
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.tensor([[3, 5, 7, 9]]),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(1, 4, dtype=torch.long),
                time_stamps=torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
                ages=torch.full((1, 4), 40.0),
                visit_orders=torch.zeros(1, 4, dtype=torch.long),
                visit_segments=torch.zeros(1, 4, dtype=torch.long),
            ),
        ),
        targets=torch.tensor([[4, 6, 8, 10]]),
        reset_mask=torch.tensor([[True, False, False, False]]),
        real_mask=torch.tensor([[True, True, True, True]]),
        subject_ids=torch.tensor([[1, 1, 1, 1]]),
        patient_end=torch.tensor([[False, False, False, True]]),
        visit_ids=torch.tensor([[-1, -1, -1, -1]]),
        visit_end=torch.tensor([[False, False, False, False]]),
    )


def test_schedule_covers_consecutive_phases_after_warmup() -> None:
    s = SteeringSchedule(warmup_steps=100, phases=4, phase_steps=90)
    assert s.enabled and s.end_step == 460
    assert not s.is_steering_step(99)
    assert s.is_steering_step(100) and s.is_steering_step(459)
    assert not s.is_steering_step(460)
    assert not SteeringSchedule(100, 0, 90).enabled
    assert not SteeringSchedule(100, 0, 90).is_steering_step(150)


def test_choose_injection_uses_running_labels_and_skips_unlabeled_chunks() -> None:
    chunk = _chunk()
    labels = {1: torch.tensor([1.0, 0.0, 1.0])}
    masks = {1: torch.ones(K)}
    # concept 0 triggers at hour 1, concept 2 never (inf)
    first = {1: torch.tensor([1.0, float("inf"), float("inf")])}
    got = choose_injection(
        chunk,
        labels,
        masks,
        first,
        supervision="stay",
        num_concepts=K,
        generator=torch.Generator().manual_seed(0),
    )
    assert got is not None and got.concept_index == 0
    assert got.positions.tolist() == [[False, True, True, True]]
    assert (
        choose_injection(chunk, {}, {}, {}, supervision="stay", num_concepts=K) is None
    )


def test_steering_loss_adds_respond_and_express_at_injected_positions() -> None:
    model = _model()
    model.train()
    chunk = _chunk()
    injected = torch.tensor([[False, True, True, False]])
    direction = torch.nn.functional.normalize(torch.randn(HIDDEN), dim=0)
    total, comp, _ = model.compute_steering_loss(
        chunk,
        state=None,
        concept_index=1,
        injected=injected,
        direction=direction,
        gamma=1.0,
        layer_index=0,
        lifted_ids=torch.tensor([4, 6]),
    )
    assert torch.isfinite(total)
    assert comp["n_injected"].item() == 2
    assert comp["respond_loss"].item() > 0 and comp["express_loss"].item() > 0
    total.backward()
    assert model.bottleneck.known_proj.weight.grad is not None

    # No lifted tokens: express contributes nothing rather than -inf.
    _, comp0, _ = model.compute_steering_loss(
        chunk,
        state=None,
        concept_index=1,
        injected=injected,
        direction=direction,
        gamma=1.0,
        layer_index=0,
        lifted_ids=torch.tensor([], dtype=torch.long),
    )
    assert comp0["express_loss"].item() == 0.0
    # Nothing injected: both terms are zero and the loss is the forecast alone.
    _, comp_none, _ = model.compute_steering_loss(
        chunk,
        state=None,
        concept_index=1,
        injected=torch.zeros(1, 4, dtype=torch.bool),
        direction=direction,
        gamma=1.0,
        layer_index=0,
        lifted_ids=torch.tensor([4]),
    )
    assert (
        comp_none["respond_loss"].item() == 0.0 and comp_none["n_injected"].item() == 0
    )
