"""Common interface for sequence backbones."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch


class SequenceBackbone(nn.Module, ABC):
    """Interface shared by every backbone the concept bottleneck sits on.

    Concrete backbones (the real EHR-Mamba3, or lightweight stand-ins for
    CPU dev/CI) all consume the same clinical inputs and produce the same
    hidden-state shape, so everything downstream of the backbone —
    embeddings fusion, the concept bottleneck, loss computation — is
    written once and is backbone-agnostic.
    """

    hidden_size: int

    @abstractmethod
    def forward(self, batch: ClinicalSequenceBatch) -> torch.Tensor:
        """Return hidden states of shape ``(batch, seq_len, hidden_size)``."""
        raise NotImplementedError
