"""Lightweight CPU-only backbone for local development and CI.

Not intended to produce real training results. `mamba-ssm` (the real
hybrid backbone, see ``hybrid.py``) requires a CUDA/`nvcc` build and
cannot be installed on a Mac dev machine or GitHub Actions' CPU runners, so
this stand-in exists to exercise the embeddings -> backbone -> concept
bottleneck -> loss wiring, and the streaming/chunked training pipeline in
:mod:`odyssey.data.streaming`, end to end without a GPU. It implements the
exact same :class:`~odyssey.models.backbones.base.SequenceBackbone`
interface, so swapping in the real backbone elsewhere is a one-line change
-- but validating the real backbone's behavior still requires a CUDA host;
this class only proves the surrounding pipeline is wired correctly.
"""

from typing import cast

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import (
    SequenceBackbone,
    TimeAwareState,
    resolve_prev_time_stamps,
)
from odyssey.models.embeddings import CachedEHREmbeddings


class TinyGRUBackbone(SequenceBackbone):
    """A tiny causal GRU backbone with exact per-position state resets.

    ``nn.GRU`` cannot reset its hidden state mid-sequence, so this steps a
    ``nn.GRUCell`` per layer one token at a time, zeroing the incoming
    state at any position ``reset_mask`` marks -- the same behavior a
    packed, multi-patient chunk from
    :class:`~odyssey.data.streaming.PackedLaneSampler` needs. Fine for a
    CPU test backbone; not how a performance-sensitive implementation
    would do this.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        padding_idx: int = 0,
        **embedding_kwargs: object,
    ) -> None:
        """Initialize the tiny GRU backbone."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        ``state.recurrent``, if given, must be a ``Tuple[torch.Tensor,
        ...]`` of length ``num_layers`` (as returned by a previous call).
        """
        prev_time_stamps = resolve_prev_time_stamps(state, batch, reset_mask)
        self.embeddings.set_aux_inputs(batch.aux, prev_time_stamps=prev_time_stamps)
        embeds = self.embeddings(batch.concept_ids)
        batch_size, seq_len, _ = embeds.shape

        if state is None:
            hidden = [
                embeds.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            hidden = list(cast(tuple[torch.Tensor, ...], state.recurrent))

        if reset_mask is None:
            reset_mask = embeds.new_zeros(batch_size, seq_len, dtype=torch.bool)

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            reset_t = reset_mask[:, t].unsqueeze(-1)
            layer_input = embeds[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_prev = torch.where(
                    reset_t, torch.zeros_like(hidden[layer_idx]), hidden[layer_idx]
                )
                hidden[layer_idx] = cell(layer_input, h_prev)
                layer_input = hidden[layer_idx]
            outputs.append(hidden[-1])

        hidden_states = torch.stack(outputs, dim=1)
        new_state = TimeAwareState(
            recurrent=tuple(hidden), prev_time_stamps=batch.aux.time_stamps[:, -1]
        )
        return hidden_states, new_state
