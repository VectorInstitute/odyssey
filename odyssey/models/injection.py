"""Layer injection: add a direction to a block-structured backbone's hidden state.

Steerling's Eq. 18, ``h^(l) <- h^(l) + gamma e_c`` for every layer
``l >= L_inj``, as a context manager of forward hooks. Kept free of model
imports so both the sequence model (steering-phase training) and the
steering tooling can use it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def stream_injection(
    backbone: torch.nn.Module, layer_index: int, vector: torch.Tensor
) -> Iterator[None]:
    """Add ``vector`` to the hidden state at every block from ``layer_index`` on.

    Registered as forward hooks on ``backbone.layers[layer_index:]``, so
    the signal accumulates toward the bottleneck and, in a recurrent
    backbone, is carried forward by the state. ``vector`` may be one
    direction ``(d,)`` or anything that broadcasts against a block output
    ``(lanes, T, d)``, so a training phase can push only the positions
    attributed to a concept. The hooks are removed on exit even if the
    forward raises.
    """
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError(
            f"{type(backbone).__name__} exposes no `layers` to inject into; the "
            "stream site needs a block-structured backbone"
        )
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"layer_index {layer_index} outside 0..{len(layers) - 1}")

    def push(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden, *rest = output
            return (hidden + vector.to(hidden), *rest)
        return output + vector.to(output)

    handles = [layer.register_forward_hook(push) for layer in layers[layer_index:]]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def middle_layer(backbone: torch.nn.Module) -> int:
    """Our default ``L_inj``: the middle block (the paper does not state its value)."""
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("stream injection needs a block-structured backbone")
    return len(layers) // 2
