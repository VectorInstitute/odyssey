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
it attaches equally to the real hybrid backbone or any lighter stand-in used
for local (non-CUDA) development.

A second, independent head predicts each known concept's *observability*
-- whether it would be measured at all -- supervised against the real
``{name}_observed`` mask from :mod:`odyssey.data.concepts`. This exists
because EHR missingness is informative, not incidental (a lab not being
drawn reflects clinical suspicion, not an annotation gap the way a missing
CUB-200 attribute label would be), and because feeding the observed mask
directly into the bottleneck as an input would leak the exact signal that
decides where ``concept_loss`` applies no gradient, letting the concept
probability at unobserved positions go unconstrained by anything but
``task_loss`` -- undermining the one property (a clinician can trust
``concept_probs``) the bottleneck exists for. A separately-supervised head
avoids that: its own output is a real, checkable prediction ("would this
concept have been tested"), not a free variable. It is deliberately NOT
wired back into the concept probability computation in this version --
see ``research_journal/05_missingness.html`` for the full reasoning.
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

    observability_logits: torch.Tensor
    """(..., num_concepts) predicted "would this concept be observed", pre-sigmoid."""

    observability_probs: torch.Tensor
    """(..., num_concepts) sigmoid(observability_logits)."""


@dataclass(frozen=True)
class BottleneckIntervention:
    """A do()-style edit applied inside the bottleneck's mixing step.

    The CEM/CBGM intervention mechanism: replace a concept's *predicted*
    activation probability with a chosen value before the embedding
    mixture ``c * w+ + (1 - c) * w-`` is formed, and/or zero out whole
    slots of the mixed-embedding concatenation. Only the mixture (and
    therefore everything downstream of the bottleneck, i.e. the task
    logits) is affected: ``concept_logits``/``concept_probs`` and the
    observability head still report the model's own, un-intervened
    predictions, so an intervention never contaminates the readouts used
    to evaluate the concept heads themselves.

    This is the machinery behind the completeness/reliance evaluation
    (:mod:`odyssey.inference.interventions`): feeding ground-truth
    concept values should *help* next-event prediction if the concepts
    causally steer it, flipped values should hurt, and zeroing the known
    vs. unknown slots apportions how much of the task signal flows
    through each channel.
    """

    probs: Optional[torch.Tensor] = None
    """(..., num_concepts) replacement mixing probabilities for the
    known concepts (the unknown slot always keeps its own). Broadcasts
    against the hidden-state batch shape."""

    probs_mask: Optional[torch.Tensor] = None
    """(..., num_concepts) bool: where True, ``probs`` replaces the
    model's own probability; elsewhere the model's own value is kept.
    None (with ``probs`` given) means replace everywhere."""

    zero_known: bool = False
    """Zero every known concept's mixed embedding (completeness probe:
    how much task signal survives on the unknown channel alone)."""

    zero_unknown: bool = False
    """Zero the unknown concept's mixed embedding (how much task signal
    flows outside the supervised concepts)."""


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

        # Independent of the concept-value pathway above: predicts whether
        # each known concept would be observed at all, from the same
        # (dropout-applied) hidden state. See the module docstring for why
        # this is a separate, supervised head rather than an input feature.
        self.observability_proj = nn.Linear(hidden_size, num_concepts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        intervention: Optional[BottleneckIntervention] = None,
    ) -> ConceptBottleneckOutput:
        """Project hidden states into known + unknown concept embeddings.

        ``intervention`` edits the mixing step only (see
        :class:`BottleneckIntervention`): the returned
        ``concept_logits``/``concept_probs``/observability outputs are
        always the model's own predictions.
        """
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

        mix_probs = probs
        if intervention is not None and intervention.probs is not None:
            own = probs[..., : self.num_concepts]
            override = intervention.probs.to(own.dtype).expand_as(own)
            if intervention.probs_mask is not None:
                override = torch.where(
                    intervention.probs_mask.expand_as(own), override, own
                )
            mix_probs = torch.cat([override, probs[..., self.num_concepts :]], dim=-1)

        mixed = mix_probs.unsqueeze(-1) * w_pos + (1 - mix_probs.unsqueeze(-1)) * w_neg
        if intervention is not None and (
            intervention.zero_known or intervention.zero_unknown
        ):
            slot_keep = torch.ones(
                self.num_slots, dtype=mixed.dtype, device=mixed.device
            )
            if intervention.zero_known:
                slot_keep[: self.num_concepts] = 0.0
            if intervention.zero_unknown:
                slot_keep[self.num_concepts] = 0.0
            mixed = mixed * slot_keep.unsqueeze(-1)

        concept_logits = logits[..., : self.num_concepts]
        concept_probs = probs[..., : self.num_concepts]
        concept_embeddings = mixed[..., : self.num_concepts, :]
        unknown_embedding = mixed[..., self.num_concepts, :]
        bottleneck = mixed.reshape(*batch_shape, self.num_slots * self.embedding_dim)

        observability_logits: torch.Tensor = self.observability_proj(x)
        observability_probs = torch.sigmoid(observability_logits)

        return ConceptBottleneckOutput(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            concept_embeddings=concept_embeddings,
            unknown_embedding=unknown_embedding,
            observability_logits=observability_logits,
            observability_probs=observability_probs,
            bottleneck=bottleneck,
        )


def concept_loss(
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Supervised BCE loss over known concepts.

    ``concept_labels`` may be partially observed — e.g. a weak/rule-derived
    label is uncomputable because the underlying lab was never drawn for
    that patient. ``concept_mask`` (same shape, 1 = observed) excludes
    unobserved entries from the loss rather than penalizing them.
    ``pos_weight`` is a per-concept ``(num_concepts,)`` positive-class
    weight (standard ``n_negative / n_positive``): without it, a 4%-
    prevalence concept like AKI stage 2 contributes almost no positive
    gradient next to a 90%-prevalence one, and the head can sit near the
    base rate.
    """
    per_element = F.binary_cross_entropy_with_logits(
        concept_logits,
        concept_labels.float(),
        reduction="none",
        pos_weight=pos_weight,
    )
    if concept_mask is None:
        return per_element.mean()
    mask = concept_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_element * mask).sum() / denom


def observability_loss(
    observability_logits: torch.Tensor, observed_mask: torch.Tensor
) -> torch.Tensor:
    """Supervised BCE loss: predict whether each concept would be observed.

    Unlike ``concept_labels`` (which can be genuinely unknown),
    ``observed_mask`` is never itself missing -- whether a lab was drawn
    is always a known fact about the encounter -- so this loss needs no
    masking of its own; every element has a real target. This is what
    grounds the model's response to concept missingness in real
    supervision, rather than the concept probability at unobserved
    positions being a free variable shaped only by ``task_loss`` (see the
    module docstring).
    """
    return F.binary_cross_entropy_with_logits(
        observability_logits, observed_mask.float()
    )


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
    observability: float = 0.1

    concept_pos_weight: Optional[torch.Tensor] = None
    """Optional per-concept ``(num_concepts,)`` positive-class weight for
    :func:`concept_loss` (see its docstring); ``None`` keeps plain BCE."""


def combined_loss(
    task_loss: torch.Tensor,
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_embeddings: torch.Tensor,
    unknown_embedding: torch.Tensor,
    *,
    observability_logits: torch.Tensor,
    concept_mask: Optional[torch.Tensor] = None,
    weights: Optional[ConceptBottleneckLossWeights] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine task, concept, orthogonality, and observability losses.

    ``concept_mask`` serves double duty when given: it excludes
    unobserved entries from ``concept_loss`` (unchanged from before), and
    is also the ground-truth target for ``observability_loss`` -- the
    same "was this concept observed" fact drives both, since it's a
    always-known property of the encounter, not something that can
    itself be missing. If ``concept_mask`` is not given, there is no
    ground truth to check an observability prediction against, so
    ``observability_loss`` is a zero tensor (this step contributes no
    gradient to that head) rather than being computed against a
    fabricated all-observed target.

    Returns the total loss plus a dict of the (detached) components for
    logging.
    """
    weights = weights or ConceptBottleneckLossWeights()
    c_loss = concept_loss(
        concept_logits,
        concept_labels,
        concept_mask,
        pos_weight=weights.concept_pos_weight,
    )
    o_loss = orthogonality_loss(concept_embeddings, unknown_embedding)
    obs_loss = (
        observability_loss(observability_logits, concept_mask)
        if concept_mask is not None
        else task_loss.new_zeros(())
    )
    total = (
        task_loss
        + weights.concept * c_loss
        + weights.orthogonality * o_loss
        + weights.observability * obs_loss
    )
    components = {
        "task_loss": task_loss.detach(),
        "concept_loss": c_loss.detach(),
        "orthogonality_loss": o_loss.detach(),
        "observability_loss": obs_loss.detach(),
    }
    return total, components
