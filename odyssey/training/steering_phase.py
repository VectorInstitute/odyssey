"""Steering-phase training: teach the dials to respond (Steerling Section 10.2.4).

On a model trained only with the decomposition, an injected concept
direction is a perturbation the network never saw, and Steerling found
that a third of their concepts never activated under it. Their remedy is
a few short phases, front-loaded after a warmup, in which the direction is
injected on purpose at the positions attributed to the concept and two
losses ask the model to respond and to express it, with the forecasting
loss kept on so capability is not traded away.

What is theirs: the schedule shape (warmup, then consecutive phases, then
normal training), injection at attributed positions at every layer from
``L_inj`` on, the respond and express losses with weights 1 and 1, and
``gamma = 1`` for the injection (their Table 36). What is ours: positions
attributed to a concept are those where its running label holds; one
target concept per chunk is drawn uniformly among the concepts present;
and ``L_inj`` defaults to the middle block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from odyssey.data.streaming import StreamingChunk
from odyssey.models.sequence_model import ConceptLabelDict, ConceptSupervision
from odyssey.training.running_labels import position_running_labels


@dataclass(frozen=True)
class SteeringSchedule:
    """``phases`` consecutive phases of ``phase_steps`` after ``warmup_steps``."""

    warmup_steps: int
    phases: int
    phase_steps: int

    @property
    def enabled(self) -> bool:
        """Whether any steering step exists at all."""
        return self.phases > 0 and self.phase_steps > 0

    @property
    def end_step(self) -> int:
        """First step after the last phase."""
        return self.warmup_steps + self.phases * self.phase_steps

    def is_steering_step(self, step: int) -> bool:
        """Whether ``step`` falls inside a steering phase."""
        return self.enabled and self.warmup_steps <= step < self.end_step


@dataclass(frozen=True)
class Injection:
    """The target concept for one chunk and where its direction is pushed."""

    concept_index: int
    positions: torch.Tensor
    """``(lanes, T)`` bool: real positions where the concept has triggered."""


def choose_injection(
    chunk: StreamingChunk,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
    generator: torch.Generator | None = None,
    min_positions: int = 1,
) -> Injection | None:
    """Pick one concept present in the chunk and the positions attributed to it.

    Attribution follows the running labels: a position belongs to concept
    ``c`` once ``c`` has triggered in the visit and the concept is observed.
    Returns ``None`` when no concept has at least ``min_positions`` such
    positions, in which case the caller runs an ordinary step.
    """
    labels, observed = position_running_labels(
        chunk,
        concept_labels,
        concept_mask,
        concept_first_times,
        supervision=supervision,
        num_concepts=num_concepts,
    )
    active = (labels > 0) & (observed > 0) & chunk.real_mask.unsqueeze(-1)
    counts = active.sum(dim=(0, 1))
    candidates = torch.nonzero(counts >= min_positions).flatten()
    if candidates.numel() == 0:
        return None
    pick = int(torch.randint(candidates.numel(), (1,), generator=generator).item())
    concept = int(candidates[pick].item())
    return Injection(concept_index=concept, positions=active[..., concept])


__all__ = ["Injection", "SteeringSchedule", "choose_injection"]
