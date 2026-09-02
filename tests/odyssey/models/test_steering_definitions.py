"""Edge cases of the steering definitions and the layer-injection hook."""

import pytest
import torch
from torch import nn

from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.injection import middle_layer, stream_injection
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.steering import steering_direction, steering_gamma


def _model(kind: str) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=9, hidden_size=8, padding_idx=0),
        vocab_size=9,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind=kind,
    )


class _TupleBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:  # noqa: D102
        return x, "cache"


class _Backbone(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TupleBlock() for _ in range(n)])


def test_injection_pushes_the_hidden_state_of_a_tuple_returning_block() -> None:
    backbone = _Backbone(3)
    x = torch.zeros(2, 4)
    vec = torch.ones(4)
    with stream_injection(backbone, 2, vec):
        hidden, rest = backbone.layers[2](x)
    assert torch.equal(hidden, torch.ones(2, 4))
    assert rest == "cache"
    hidden, _ = backbone.layers[2](x)
    assert torch.equal(hidden, x)


def test_middle_layer_is_the_middle_block_and_needs_blocks() -> None:
    assert middle_layer(_Backbone(8)) == 4
    assert middle_layer(_Backbone(1)) == 0
    with pytest.raises(TypeError, match="block-structured"):
        middle_layer(nn.Linear(2, 2))


def test_steering_direction_needs_the_decomposed_bottleneck() -> None:
    with pytest.raises(NotImplementedError, match="decomposed"):
        steering_direction(_model("mixture"), 0)


def test_gamma_rejects_a_non_positive_tau_and_a_direction_that_raises_nothing() -> None:
    model = _model("decomposed")
    direction = steering_direction(model, 0)
    with pytest.raises(ValueError, match="positive"):
        steering_gamma(model, direction, tau=0.0)
    with torch.no_grad():
        model.lm_head.weight.zero_()  # no logit can rise along any direction
    with pytest.raises(ValueError, match="raises no logit"):
        steering_gamma(model, direction, tau=1.0)
