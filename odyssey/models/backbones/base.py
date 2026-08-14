"""Common interface for sequence backbones."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch


class SequenceBackbone(nn.Module, ABC):
    """Interface shared by every backbone the concept bottleneck sits on.

    Concrete backbones (the real EHR-Mamba3, or lightweight stand-ins for
    CPU dev/CI) all consume the same clinical inputs and produce the same
    hidden-state shape, so everything downstream of the backbone --
    embeddings fusion, the concept bottleneck, loss computation -- is
    written once and is backbone-agnostic.

    Stateful across chunks: a training loop using
    :class:`~odyssey.data.streaming.PackedLaneSampler` calls ``forward``
    once per chunk per lane, carrying each lane's ``state`` forward to the
    next chunk (state is opaque and backbone-specific; a lane starting
    fresh passes ``state=None``). ``reset_mask`` marks positions where the
    incoming state (whether from ``state`` at position 0, or from the
    previous position within this chunk) must be zeroed instead of carried
    -- a real patient boundary from packing, or a synthetic
    missing-history reset. See ``research_journal/02_sequence_scoping_methodology.html``
    Section 05 for the design this implements.
    """

    hidden_size: int

    @abstractmethod
    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: Optional[object] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Return ``(hidden_states, new_state)``.

        ``hidden_states`` has shape ``(batch, seq_len, hidden_size)``.
        ``reset_mask`` is ``(batch, seq_len)`` bool, ``True`` where state
        must be zeroed before that position; pass ``None`` for no resets.
        A caller doing truncated backpropagation through time is
        responsible for detaching ``new_state`` before reusing it as the
        next chunk's ``state``, if the backbone doesn't already return a
        non-differentiable state.
        """
        raise NotImplementedError
