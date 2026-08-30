"""Common interface for sequence backbones."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.embeddings import CachedEHREmbeddings


@dataclass
class TimeAwareState:
    """A backbone-specific recurrent state, plus each lane's last timestamp.

    The timestamp is needed because
    :class:`~odyssey.models.embeddings.TimeEmbeddingLayer` computes
    time-since-previous-event as a delta; without knowing the previous
    chunk's last real timestamp, the delta at position 0 of a
    continuation chunk gets zeroed as if it were a fresh sequence start --
    confirmed on real hardware to produce materially different hidden
    states (up to ~30% relative) on its own, independent of whether the
    recurrent state itself carries correctly. Every stateful backbone in
    this package uses this same wrapper so the fix lives in one place.
    """

    recurrent: object
    prev_time_stamps: torch.Tensor


def resolve_prev_time_stamps(
    state: TimeAwareState | None,
    batch: ClinicalSequenceBatch,
    reset_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the ``prev_value`` to pass to the embeddings layer.

    ``None`` if there's no carried state at all (a lane's very first
    chunk). Otherwise, the carried timestamp, with reset lanes (position 0
    of ``reset_mask``, if given) overridden to this chunk's own first
    timestamp -- which makes the computed first-delta come out to exactly
    ``0``, the same convention a genuinely fresh sequence gets.
    """
    if state is None:
        return None
    prev = state.prev_time_stamps
    if reset_mask is not None:
        reset_rows = reset_mask[:, 0]
        if reset_rows.any():
            prev = prev.clone()
            prev[reset_rows] = batch.aux.time_stamps[reset_rows, 0]
    return prev


class SequenceBackbone(nn.Module, ABC):
    """Interface shared by every backbone the concept bottleneck sits on.

    Concrete backbones (the real EHRHybridBackbone, or lightweight stand-ins for
    CPU dev/CI) all consume the same clinical inputs and produce the same
    hidden-state shape, so everything downstream of the backbone --
    embeddings fusion, the concept bottleneck, loss computation -- is
    written once and is backbone-agnostic.

    Stateful across chunks: a training loop using
    :class:`~odyssey.data.streaming.PackedLaneSampler` calls ``forward``
    once per chunk per lane, carrying each lane's ``state`` forward to the
    next chunk (state is a :class:`TimeAwareState`; a lane starting fresh
    passes ``state=None``). ``reset_mask`` marks positions where the
    incoming state (whether from ``state`` at position 0, or from the
    previous position within this chunk) must be zeroed instead of carried
    -- a real patient boundary from packing, or a synthetic
    missing-history reset. See ``research_journal/02_sequence_scoping_methodology.html``
    Section 05 for the design this implements.
    """

    hidden_size: int
    embeddings: CachedEHREmbeddings
    """Every concrete backbone (hybrid, transformer, the tiny-GRU CPU
    stand-in) assigns this in ``__init__`` -- declared here, not just
    left to duck-typing, so callers that reach into the embedding table
    directly (e.g. ``_SequenceModelBase._streaming_value_loss`` looking
    up a target token's own embedding for
    :mod:`odyssey.models.value_head`'s conditioning) type-check for real
    instead of resolving through ``nn.Module.__getattr__``'s untyped
    fallback."""

    @abstractmethod
    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
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
