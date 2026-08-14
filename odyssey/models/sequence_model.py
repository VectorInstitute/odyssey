"""Concept-bottleneck sequence model: backbone -> bottleneck -> LM head.

Backbone-agnostic: works with any
:class:`~odyssey.models.backbones.base.SequenceBackbone`, so the same model
class runs against the lightweight CPU stand-in backbone in tests/CI and
the real EHR-Mamba3 backbone on a CUDA host.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone
from odyssey.models.concept_bottleneck import (
    ConceptBottleneck,
    ConceptBottleneckLossWeights,
    ConceptBottleneckOutput,
    combined_loss,
)


def _pool_last_non_padding(
    values: torch.Tensor, concept_ids: torch.Tensor, padding_idx: int
) -> torch.Tensor:
    """Select each sequence's last non-padding position from ``values``.

    ``values`` is ``(batch, seq_len, ...)``; the result is ``(batch, ...)``.
    Used to align per-token bottleneck activations with the subject-level
    concept labels produced by :mod:`odyssey.data.concepts` (e.g. "was this
    patient ever tachycardic"), which describe the whole stay, not a
    single timestep.
    """
    pad_mask = concept_ids == padding_idx
    last_idx = pad_mask.int().argmax(dim=-1) - 1
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(values.shape[0], device=values.device)
    return values[batch_idx, last_idx]


class ConceptBottleneckSequenceModel(nn.Module):
    """Next-token prediction, through a concept bottleneck, over event sequences."""

    def __init__(
        self,
        backbone: SequenceBackbone,
        vocab_size: int,
        num_concepts: int,
        embedding_dim: int,
        *,
        padding_idx: int = 0,
        concept_dropout: float = 0.1,
    ) -> None:
        """Initialize the concept-bottleneck sequence model."""
        super().__init__()
        self.backbone = backbone
        self.padding_idx = padding_idx
        self.bottleneck = ConceptBottleneck(
            hidden_size=backbone.hidden_size,
            num_concepts=num_concepts,
            embedding_dim=embedding_dim,
            concept_dropout=concept_dropout,
        )
        self.lm_head = nn.Linear((num_concepts + 1) * embedding_dim, vocab_size)

    def forward(
        self, batch: ClinicalSequenceBatch
    ) -> Tuple[torch.Tensor, ConceptBottleneckOutput]:
        """Return (next-token logits, bottleneck output)."""
        hidden_states = self.backbone(batch)
        bottleneck_out = self.bottleneck(hidden_states)
        logits = self.lm_head(bottleneck_out.bottleneck)
        return logits, bottleneck_out

    def compute_loss(
        self,
        batch: ClinicalSequenceBatch,
        concept_labels: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[ConceptBottleneckLossWeights] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute next-token + concept + orthogonality loss.

        ``concept_labels``/``concept_mask`` are ``(batch, num_concepts)`` —
        subject-level, supervising the bottleneck at each sequence's last
        non-padding position (see :func:`_pool_last_non_padding`).
        """
        logits, bottleneck_out = self(batch)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = (labels if labels is not None else batch.concept_ids)[
            :, 1:
        ].contiguous()
        next_token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=self.padding_idx,
        )

        pooled_concept_logits = _pool_last_non_padding(
            bottleneck_out.concept_logits, batch.concept_ids, self.padding_idx
        )
        pooled_concept_embeddings = _pool_last_non_padding(
            bottleneck_out.concept_embeddings, batch.concept_ids, self.padding_idx
        )
        pooled_unknown_embedding = _pool_last_non_padding(
            bottleneck_out.unknown_embedding, batch.concept_ids, self.padding_idx
        )

        return combined_loss(
            next_token_loss,
            pooled_concept_logits,
            concept_labels,
            pooled_concept_embeddings,
            pooled_unknown_embedding,
            concept_mask=concept_mask,
            weights=loss_weights,
        )
