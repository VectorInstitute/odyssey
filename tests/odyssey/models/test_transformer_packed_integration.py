"""End-to-end integration: TransformerBackbone + PackedContextSampler + the model stack.

The unit tests in ``test_transformer.py`` and ``test_packed_context.py``
cover the backbone and the sampler each in isolation. This file is the one
place that proves the third scope requirement they don't individually
touch: the existing heads and losses (concept bottleneck, forecast
objective, time/event hazard heads) work completely unchanged on top of
this backbone -- no special-casing anywhere downstream of ``backbone()``.
"""

import torch

from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.sequences import PatientSequence
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


def _seq(subject_id: int, n: int) -> PatientSequence:
    return PatientSequence(
        subject_id=subject_id,
        concept_ids=[(subject_id * 10 + i) % 40 + 1 for i in range(n)],
        type_ids=[1] * n,
        time_stamps=[float(i) for i in range(n)],
        ages=[40.0] * n,
        visit_orders=[0] * n,
        visit_segments=[0] * n,
    )


def test_compute_streaming_loss_runs_on_a_packed_transformer_chunk() -> None:
    """The full stack (packing -> backbone -> bottleneck -> loss) must just work."""
    backbone = TransformerBackbone(
        vocab_size=41, hidden_size=16, num_hidden_layers=2, num_heads=4
    )
    model = ConceptBottleneckSequenceModel(
        backbone, vocab_size=41, num_concepts=3, embedding_dim=8
    )
    sampler = PackedContextSampler(
        iter([_seq(1, 5), _seq(2, 4), _seq(3, 6)]), batch_size=2, max_context=12
    )
    chunk = sampler.next_chunk()
    assert chunk is not None

    concept_labels = {1: torch.zeros(3), 2: torch.ones(3), 3: torch.zeros(3)}
    loss, components, new_state = model.compute_streaming_loss(
        chunk,
        concept_labels=concept_labels,
        loss_weights=ConceptBottleneckLossWeights(),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "task_loss" in components
    assert new_state is not None


def test_gradients_flow_back_through_a_packed_transformer_chunk() -> None:
    """Not just a forward pass -- the packed path must be trainable."""
    backbone = TransformerBackbone(
        vocab_size=41, hidden_size=16, num_hidden_layers=2, num_heads=4
    )
    model = ConceptBottleneckSequenceModel(
        backbone, vocab_size=41, num_concepts=3, embedding_dim=8
    )
    sampler = PackedContextSampler(
        iter([_seq(1, 5), _seq(2, 4)]), batch_size=1, max_context=12
    )
    chunk = sampler.next_chunk()
    assert chunk is not None

    concept_labels = {1: torch.zeros(3), 2: torch.ones(3)}
    loss, _, _ = model.compute_streaming_loss(
        chunk,
        concept_labels=concept_labels,
        loss_weights=ConceptBottleneckLossWeights(),
    )
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grad_norms
    assert any(g > 0 for g in grad_norms)
