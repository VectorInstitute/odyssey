"""Concept bottleneck layer for interpretable sequence models.

Implements the concept-embedding bottleneck from Ismail, Adebayo, Bravo, Ra
& Cho, "Concept Bottleneck Generative Models" (ICLR 2024,
https://github.com/prescient-design/CBGM), which adapts the Concept
Embedding Model layer of Espinosa Zarlenga et al. (NeurIPS 2022,
https://github.com/mateoespinosa/cem) to generative/sequence settings.

Each concept — including one extra, unsupervised "unknown" concept — is
represented by a pair of learned embeddings (active/inactive), mixed by a
predicted activation probability into that concept's final representation.
The known concepts' probabilities are supervised against clinical labels;
the unknown concept's embedding is regularized to be orthogonal to the
known concepts', so it can't just silently re-encode them. Backbone
agnostic: it only consumes hidden states of shape ``(..., hidden_size)``, so
it attaches equally to the EHR-Mamba3 backbone or any lighter stand-in used
for local (non-CUDA) development.
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ConceptBottleneckOutput(NamedTuple):
    """Outputs of a :class:`ConceptBottleneck` forward pass."""

    concept_logits: torch.Tensor
    """(..., num_concepts) known-concept activation logits, pre-sigmoid."""

    concept_probs: torch.Tensor
    """(..., num_concepts) sigmoid(concept_logits); what's shown to a clinician."""

    concept_embeddings: torch.Tensor
    """(..., num_concepts, embedding_dim) known concepts' mixed embeddings."""

    unknown_embedding: torch.Tensor
    """(..., embedding_dim) the extra, unsupervised concept's mixed embedding."""

    bottleneck: torch.Tensor
    """(..., (num_concepts + 1) * embedding_dim): concat of all mixed embeddings."""


class ConceptBottleneck(nn.Module):
    """Splits a hidden representation into known + unknown concept embeddings.

    For each of ``num_concepts`` known concepts, plus one extra unsupervised
    "unknown" concept, a context network maps the hidden state to a pair of
    embeddings ``(w+, w-)``; a probability network predicts that concept's
    activation probability ``c`` from ``[w+, w-]``; and the concept's final
    representation is the mixture ``c * w+ + (1 - c) * w-``. All
    ``num_concepts + 1`` mixed embeddings are concatenated into the
    bottleneck output. This mirrors the reference CEM/CBGM implementations
    exactly, just batched: one ``Linear`` producing every slot's ``(w+,
    w-)`` pair is mathematically equivalent to independent per-concept
    context networks, since each slot's output only ever depends on its own
    weight rows.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the incoming backbone hidden state.
    num_concepts : int
        Number of supervised, clinically-grounded concepts.
    embedding_dim : int
        Dimensionality of each concept's (and the unknown concept's)
        embedding.
    concept_dropout : float
        Dropout applied to the hidden state before the context projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_concepts: int,
        embedding_dim: int,
        *,
        concept_dropout: float = 0.1,
    ) -> None:
        """Initialize the concept bottleneck layer."""
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.num_slots = num_concepts + 1  # known concepts + 1 unknown concept

        self.dropout = nn.Dropout(concept_dropout)
        self.context_proj = nn.Linear(hidden_size, self.num_slots * 2 * embedding_dim)
        self.context_act = nn.LeakyReLU()

        # Per-slot probability network Psi_i([w+, w-]) -> logit. Implemented
        # as one (num_slots, 2*embedding_dim) weight so every slot's logit
        # only depends on that slot's own embeddings, not the others'.
        self.prob_weight = nn.Parameter(torch.empty(self.num_slots, 2 * embedding_dim))
        self.prob_bias = nn.Parameter(torch.zeros(self.num_slots))
        nn.init.xavier_uniform_(self.prob_weight)

    def forward(self, hidden_states: torch.Tensor) -> ConceptBottleneckOutput:
        """Project hidden states into known + unknown concept embeddings."""
        batch_shape = hidden_states.shape[:-1]
        x = self.dropout(hidden_states)

        context = self.context_act(self.context_proj(x))
        context = context.view(*batch_shape, self.num_slots, 2, self.embedding_dim)
        w_pos = context[..., 0, :]
        w_neg = context[..., 1, :]

        joint = torch.cat([w_pos, w_neg], dim=-1)  # (..., num_slots, 2*embedding_dim)
        logits = (
            torch.einsum("...sd,sd->...s", joint, self.prob_weight) + self.prob_bias
        )
        probs = torch.sigmoid(logits)

        mixed = probs.unsqueeze(-1) * w_pos + (1 - probs.unsqueeze(-1)) * w_neg

        concept_logits = logits[..., : self.num_concepts]
        concept_probs = probs[..., : self.num_concepts]
        concept_embeddings = mixed[..., : self.num_concepts, :]
        unknown_embedding = mixed[..., self.num_concepts, :]
        bottleneck = mixed.reshape(*batch_shape, self.num_slots * self.embedding_dim)

        return ConceptBottleneckOutput(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            concept_embeddings=concept_embeddings,
            unknown_embedding=unknown_embedding,
            bottleneck=bottleneck,
        )


def concept_loss(
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Supervised BCE loss over known concepts.

    ``concept_labels`` may be partially observed — e.g. a weak/rule-derived
    label is uncomputable because the underlying lab was never drawn for
    that patient. ``concept_mask`` (same shape, 1 = observed) excludes
    unobserved entries from the loss rather than penalizing them.
    """
    per_element = F.binary_cross_entropy_with_logits(
        concept_logits, concept_labels.float(), reduction="none"
    )
    if concept_mask is None:
        return per_element.mean()
    mask = concept_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_element * mask).sum() / denom


def orthogonality_loss(
    concept_embeddings: torch.Tensor, unknown_embedding: torch.Tensor
) -> torch.Tensor:
    """Penalize the unknown concept re-encoding the known concepts.

    Without this term the unknown concept is free to reconstruct the known
    concepts redundantly, in an uninterpretable embedding — the model would
    satisfy the concept loss without those concepts' embeddings being
    load-bearing for the task, defeating the point of the bottleneck.
    Mean absolute cosine similarity between each known concept's embedding
    and the unknown concept's embedding (Eq. 5 of the CBGM paper).
    """
    cos_sim = F.cosine_similarity(
        concept_embeddings, unknown_embedding.unsqueeze(-2), dim=-1
    )
    return cos_sim.abs().mean()


@dataclass
class ConceptBottleneckLossWeights:
    """Relative weights for the concept-bottleneck auxiliary losses."""

    concept: float = 1.0
    orthogonality: float = 0.1


def combined_loss(
    task_loss: torch.Tensor,
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_embeddings: torch.Tensor,
    unknown_embedding: torch.Tensor,
    *,
    concept_mask: Optional[torch.Tensor] = None,
    weights: Optional[ConceptBottleneckLossWeights] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine task, concept, and orthogonality losses.

    Returns the total loss plus a dict of the (detached) components for
    logging.
    """
    weights = weights or ConceptBottleneckLossWeights()
    c_loss = concept_loss(concept_logits, concept_labels, concept_mask)
    o_loss = orthogonality_loss(concept_embeddings, unknown_embedding)
    total = task_loss + weights.concept * c_loss + weights.orthogonality * o_loss
    components = {
        "task_loss": task_loss.detach(),
        "concept_loss": c_loss.detach(),
        "orthogonality_loss": o_loss.detach(),
    }
    return total, components
