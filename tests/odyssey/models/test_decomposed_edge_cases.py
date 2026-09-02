"""Guard rails and degenerate inputs of the decomposed bottleneck."""

import pytest
import torch

from odyssey.models.concept_bottleneck import (
    ConceptBottleneck,
    DecomposedConceptBottleneck,
    independence_loss,
    reconstruction_loss,
)


def test_constructors_reject_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="num_concepts"):
        DecomposedConceptBottleneck(hidden_size=8, num_concepts=0)
    with pytest.raises(ValueError, match="unknown_ratio"):
        DecomposedConceptBottleneck(hidden_size=8, num_concepts=2, unknown_ratio=0)
    with pytest.raises(ValueError, match="unknown_dim"):
        ConceptBottleneck(hidden_size=8, num_concepts=2, embedding_dim=4, unknown_dim=0)


def test_reconstruction_loss_is_zero_when_no_position_is_labeled() -> None:
    hidden = torch.randn(2, 3, 8)
    unknown = torch.randn(2, 3, 8, requires_grad=True)
    known = torch.randn(2, 8)
    labels = torch.zeros(2, 3, 2)
    mask = torch.zeros(2, 3, 2, dtype=torch.bool)
    loss = reconstruction_loss(unknown, hidden, known, labels, concept_mask=mask)
    assert loss.item() == 0.0 and loss.shape == ()


def test_independence_loss_needs_two_positions() -> None:
    known = torch.randn(1, 1, 8)
    unknown = torch.randn(1, 1, 8)
    assert independence_loss(known, unknown).item() == 0.0
    # with a mask that keeps one position of several the same guard applies
    known = torch.randn(1, 4, 8)
    unknown = torch.randn(1, 4, 8)
    keep = torch.tensor([[True, False, False, False]])
    assert independence_loss(known, unknown, keep).item() == 0.0
    assert independence_loss(known, unknown).item() >= 0.0


def test_decomposed_bottleneck_declines_the_orthogonality_fold_in() -> None:
    bottleneck = DecomposedConceptBottleneck(hidden_size=8, num_concepts=2)
    assert bottleneck.unaccounted_orthogonality().item() == 0.0
