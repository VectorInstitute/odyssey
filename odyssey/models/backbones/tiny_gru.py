"""Lightweight CPU-only backbone for local development and CI.

Not intended to produce real training results. `mamba-ssm` (the real
EHR-Mamba3 backbone, see ``mamba3.py``) requires a CUDA/`nvcc` build and
cannot be installed on a Mac dev machine or GitHub Actions' CPU runners, so
this stand-in exists to exercise the embeddings -> backbone -> concept
bottleneck -> loss wiring end-to-end without a GPU. It implements the exact
same :class:`~odyssey.models.backbones.base.SequenceBackbone` interface, so
swapping in the real backbone elsewhere is a one-line change.
"""

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone
from odyssey.models.embeddings import CachedEHREmbeddings


class TinyGRUBackbone(SequenceBackbone):
    """A tiny causal GRU backbone."""

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
        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True
        )

    def forward(self, batch: ClinicalSequenceBatch) -> torch.Tensor:
        """Return hidden states of shape ``(batch, seq_len, hidden_size)``."""
        self.embeddings.set_aux_inputs(batch.aux)
        embeds = self.embeddings(batch.concept_ids)
        hidden_states, _ = self.gru(embeds)
        return hidden_states  # type: ignore[no-any-return]
