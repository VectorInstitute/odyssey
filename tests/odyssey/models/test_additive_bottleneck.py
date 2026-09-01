"""The additive bottleneck must be steerable by construction.

:class:`ConceptBottleneck` mixes two pole embeddings that are themselves
functions of the hidden state, so an override re-weights vectors that
already encode the patient and the lever comes out inert. The additive
variant adds a fixed direction per concept to an untouched backbone
stream, which should make an override a known displacement.

"Should" is the thing worth testing. These assert the two properties the
design is FOR: an override moves the output by exactly the amount the
design promises, independent of the input, and the backbone representation
survives so interpretability is not paid for in capacity.
"""

import pytest
import torch

from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.concept_bottleneck import (
    AdditiveConceptBottleneck,
    BottleneckIntervention,
    ConceptBottleneck,
)
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


HIDDEN, K = 8, 3


def _bottleneck() -> AdditiveConceptBottleneck:
    """Build a deterministic additive bottleneck."""
    torch.manual_seed(0)
    bn = AdditiveConceptBottleneck(HIDDEN, K, concept_dropout=0.0)
    bn.eval()
    return bn


def _override(bn, h, value, concept=0):
    probs = torch.full((*h.shape[:-1], K), float(value))
    mask = torch.zeros((*h.shape[:-1], K), dtype=torch.bool)
    mask[..., concept] = True
    return bn(h, intervention=BottleneckIntervention(probs=probs, probs_mask=mask))


def test_override_displaces_output_by_exactly_delta_times_direction() -> None:
    """The defining property: moving p_i by delta moves the output by delta*v_i.

    This is what the mixture bottleneck cannot do, because there the two
    vectors being mixed are functions of h. Here the displacement is a
    known constant, which is what makes the concept a usable lever.
    """
    bn = _bottleneck()
    h = torch.randn(5, HIDDEN)
    base = bn(h)
    for value in (0.0, 0.25, 1.0):
        out = _override(bn, h, value, concept=0)
        delta = value - base.concept_probs[..., 0]
        expected = base.bottleneck + delta.unsqueeze(-1) * bn.concept_directions[0]
        assert torch.allclose(out.bottleneck, expected, atol=1e-5), value


def test_displacement_is_independent_of_the_hidden_state() -> None:
    """Two different patients, same forced change, same movement.

    The mixture design fails exactly here: its displacement depends on h,
    so "in shock" means something different for every patient.
    """
    bn = _bottleneck()
    h1, h2 = torch.randn(1, HIDDEN), torch.randn(1, HIDDEN)
    move1 = _override(bn, h1, 1.0).bottleneck - _override(bn, h1, 0.0).bottleneck
    move2 = _override(bn, h2, 1.0).bottleneck - _override(bn, h2, 0.0).bottleneck
    assert torch.allclose(move1, move2, atol=1e-5)
    # and it is precisely the concept's own direction
    assert torch.allclose(move1[0], bn.concept_directions[0], atol=1e-5)


def test_backbone_representation_passes_through_untouched() -> None:
    """Capacity is not taken from the model: zeroing concepts leaves h."""
    bn = _bottleneck()
    h = torch.randn(4, HIDDEN)
    out = bn(h, intervention=BottleneckIntervention(zero_known=True))
    assert torch.allclose(out.bottleneck, h, atol=1e-6)
    assert bn.output_dim == HIDDEN


def test_zero_unknown_leaves_the_concept_offset_alone() -> None:
    """The completeness probe stays meaningful and becomes symmetric.

    zero_known gives h alone; zero_unknown gives the concept offset alone,
    which is directly "what do the named concepts carry by themselves".
    """
    bn = _bottleneck()
    h = torch.randn(4, HIDDEN)
    out = bn(h, intervention=BottleneckIntervention(zero_unknown=True))
    expected = bn(h).concept_probs @ bn.concept_directions
    assert torch.allclose(out.bottleneck, expected, atol=1e-5)


def test_readouts_report_the_models_own_belief_not_the_override() -> None:
    """An intervention must never contaminate the reported concept probs."""
    bn = _bottleneck()
    h = torch.randn(3, HIDDEN)
    base = bn(h)
    forced = _override(bn, h, 1.0, concept=1)
    assert torch.allclose(base.concept_probs, forced.concept_probs, atol=1e-6)
    assert torch.allclose(base.observability_probs, forced.observability_probs)


def test_direction_orthogonality_is_width_agnostic_and_nonvacuous() -> None:
    """Unlike the mixture penalty, this cannot silently return zero."""
    bn = _bottleneck()
    assert bn.direction_orthogonality().item() > 0.0
    with torch.no_grad():
        bn.concept_directions.copy_(torch.eye(K, HIDDEN))
    assert bn.direction_orthogonality().item() < 1e-6


def test_sequence_model_can_be_built_with_either_bottleneck() -> None:
    """The kind is selectable and the heads size themselves off its output."""
    for kind in ("mixture", "additive"):
        model = ConceptBottleneckSequenceModel(
            backbone=TinyGRUBackbone(vocab_size=11, hidden_size=HIDDEN),
            vocab_size=11,
            num_concepts=K,
            embedding_dim=4,
            padding_idx=0,
            bottleneck_kind=kind,
        )
        if kind == "additive":
            assert isinstance(model.bottleneck, AdditiveConceptBottleneck)
            assert model.lm_head.in_features == HIDDEN
        else:
            assert isinstance(model.bottleneck, ConceptBottleneck)
            assert model.lm_head.in_features == K * 4 + 4


def test_unknown_bottleneck_kind_is_rejected() -> None:
    """An unrecognised kind fails at construction, not at first forward."""
    with pytest.raises(ValueError, match="bottleneck_kind"):
        ConceptBottleneckSequenceModel(
            backbone=TinyGRUBackbone(vocab_size=11, hidden_size=HIDDEN),
            vocab_size=11,
            num_concepts=K,
            embedding_dim=4,
            padding_idx=0,
            bottleneck_kind="nope",
        )
