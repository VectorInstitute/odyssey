"""Steerling's steering operation on the decomposed bottleneck.

Everything here is a definition from Madsen et al. (2026), Section 6.2,
stated on our model: the unit steering direction, the per-concept
calibration of the injection strength, the concept's alignment with the
output head, the ReLU-gated suppression mask, and the layer injection
that adds the direction to the backbone's hidden state at every block
from ``L_inj`` on. Inference (:mod:`odyssey.inference.steering`) and
steering training (:mod:`odyssey.training.steering_phase`) both build on
these, which is why they live with the model rather than with either.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from odyssey.models.concept_bottleneck import DecomposedConceptBottleneck
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


def _known_embedding(
    model: ConceptBottleneckSequenceModel, concept_index: int
) -> torch.Tensor:
    bottleneck = model.bottleneck
    if not isinstance(bottleneck, DecomposedConceptBottleneck):
        raise NotImplementedError(
            "steering needs the decomposed bottleneck, where a unit of a concept "
            "is the parameter K_c; the mixture's displacement is a function of "
            "the hidden state and would have to be estimated per position"
        )
    embedding: torch.Tensor = bottleneck.known_embeddings[concept_index].detach()
    return embedding


def steering_direction(
    model: ConceptBottleneckSequenceModel, concept_index: int
) -> torch.Tensor:
    """``e_c = K_c / ||K_c||_2``, Steerling's steering direction (their 6.2.1)."""
    embedding = _known_embedding(model, concept_index)
    unit: torch.Tensor = embedding / embedding.norm().clamp_min(1e-12)
    return unit


def steering_gamma(
    model: ConceptBottleneckSequenceModel, direction: torch.Tensor, *, tau: float
) -> float:
    """``gamma = tau / peak(e_c)`` with ``peak(e_c) = max_y e_c . W_y`` (Eq. 19).

    The maximum is signed, not absolute: the paper calibrates on the
    largest logit *increase* the direction can produce.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    weight = model.lm_head.weight.detach().to(direction)
    peak = float((weight @ direction).max().item())
    if peak <= 0:
        raise ValueError("the direction raises no logit; calibration is undefined")
    return tau / peak


def concept_alignment(
    model: ConceptBottleneckSequenceModel, direction: torch.Tensor
) -> torch.Tensor:
    """``a_c = W e_c``: the concept's contribution to every logit (Eq. 20)."""
    weight = model.lm_head.weight.detach().to(direction)
    alignment: torch.Tensor = weight @ direction
    return alignment


def suppress_logits(
    logits: torch.Tensor, alignment: torch.Tensor, strength: float
) -> torch.Tensor:
    """``l_v -> l_v - s . ReLU(a_c[v])``, the ReLU-gated mask (Eq. 21).

    Plain subtraction would promote tokens anti-aligned with the concept;
    the gate leaves them untouched.
    """
    return logits - strength * torch.relu(alignment).to(logits)


@contextmanager
def stream_injection(
    backbone: torch.nn.Module, layer_index: int, vector: torch.Tensor
) -> Iterator[None]:
    """Add ``vector`` to the hidden state at every block from ``layer_index`` on.

    Steerling's Eq. 18: ``h^(l) <- h^(l) + gamma e_c`` for ``l >= L_inj``, so
    the signal accumulates toward the bottleneck. Registered as forward
    hooks on those blocks; every position of every later chunk is pushed
    and the recurrent state carries the push forward. ``vector`` may be a
    single direction ``(d,)`` or anything that broadcasts against the
    block output ``(lanes, T, d)``, so a training phase can push only the
    positions attributed to the concept. The hooks are removed on exit
    even if the pass raises.
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


__all__ = [
    "concept_alignment",
    "steering_direction",
    "steering_gamma",
    "stream_injection",
    "suppress_logits",
]
