"""Per-event hazard heads: linear vs MLP readout."""

import torch

from odyssey.models.time_to_event import EventHazardHeads


def test_event_hazard_heads_mlp_readout_shapes_and_keys() -> None:
    linear = EventHazardHeads(8, ["a", "b"], (1.0, 8.0))
    mlp = EventHazardHeads(8, ["a", "b"], (1.0, 8.0), hidden_size=16)
    x = torch.randn(3, 5, 8)
    assert linear(x).shape == (3, 5, 2, 4)
    assert mlp(x).shape == (3, 5, 2, 4)
    assert "proj.weight" in linear.state_dict()
    assert {"proj.0.weight", "proj.2.weight"} <= set(mlp.state_dict())
    assert mlp.hidden_size == 16 and linear.hidden_size == 0
