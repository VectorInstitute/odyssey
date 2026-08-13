"""Concept bottleneck layer for interpretable sequence models.

Implements the loss recipe from Ismail, Adebayo, Bravo, Ra & Cho, "Concept
Bottleneck Generative Models" (ICLR 2024): a bottleneck that splits a
backbone hidden state into (a) known-concept dimensions, supervised against
clinically-grounded concept labels, and (b) free "unknown"/residual
dimensions that absorb whatever else the downstream task needs. Backbone
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
    """(..., num_concepts) known-concept activations, pre-sigmoid."""

    concept_probs: torch.Tensor
    """(..., num_concepts) sigmoid(concept_logits); what's shown to a clinician."""

    residual: torch.Tensor
    """(..., residual_dim) free "unknown concept" capacity."""

    bottleneck: torch.Tensor
    """(..., num_concepts + residual_dim) concat(concept_probs, residual)."""


class ConceptBottleneck(nn.Module):
    """Splits a hidden representation into known-concept + residual capacity.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the incoming backbone hidden state.
    num_concepts : int
        Number of supervised, clinically-grounded concepts.
    residual_dim : int
        Dimensionality of the free "unknown concept" capacity.
    concept_dropout : float
        Dropout applied to the hidden state before the concept/residual
        projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_concepts: int,
        residual_dim: int,
        concept_dropout: float = 0.1,
    ) -> None:
        """Initialize the concept bottleneck layer."""
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if residual_dim <= 0:
            raise ValueError("residual_dim must be positive")

        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.residual_dim = residual_dim

        self.dropout = nn.Dropout(concept_dropout)
        self.concept_proj = nn.Linear(hidden_size, num_concepts)
        self.residual_proj = nn.Linear(hidden_size, residual_dim)

    def forward(self, hidden_states: torch.Tensor) -> ConceptBottleneckOutput:
        """Project hidden states into known-concept and residual capacity."""
        x = self.dropout(hidden_states)
        concept_logits = self.concept_proj(x)
        residual = self.residual_proj(x)
        concept_probs = torch.sigmoid(concept_logits)
        bottleneck = torch.cat([concept_probs, residual], dim=-1)
        return ConceptBottleneckOutput(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            residual=residual,
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
    concept_probs: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Penalize correlation between known-concept and residual activations.

    Without this term the residual is free to re-encode the known concepts
    redundantly, in an uninterpretable subspace — the model would satisfy
    the concept loss without those activations being load-bearing for the
    task, defeating the point of the bottleneck. This is the squared
    Frobenius norm of the cross-covariance between mean-centered concept
    probabilities and residual features, normalized by ``n - 1`` so it's
    comparable across batch sizes.
    """
    c = concept_probs.reshape(-1, concept_probs.shape[-1])
    r = residual.reshape(-1, residual.shape[-1])
    n = c.shape[0]
    if n < 2:
        return c.new_zeros(())
    c = c - c.mean(dim=0, keepdim=True)
    r = r - r.mean(dim=0, keepdim=True)
    cross_cov = (c.T @ r) / (n - 1)
    return (cross_cov**2).sum()


@dataclass
class ConceptBottleneckLossWeights:
    """Relative weights for the concept-bottleneck auxiliary losses."""

    concept: float = 1.0
    orthogonality: float = 0.1


def combined_loss(
    task_loss: torch.Tensor,
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_probs: torch.Tensor,
    residual: torch.Tensor,
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
    o_loss = orthogonality_loss(concept_probs, residual)
    total = task_loss + weights.concept * c_loss + weights.orthogonality * o_loss
    components = {
        "task_loss": task_loss.detach(),
        "concept_loss": c_loss.detach(),
        "orthogonality_loss": o_loss.detach(),
    }
    return total, components
