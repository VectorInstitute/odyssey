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

import math
from dataclasses import dataclass
from typing import NamedTuple

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
    """(..., num_concepts * embedding_dim + unknown_dim): all mixed embeddings."""

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

    probs: torch.Tensor | None = None
    """(..., num_concepts) replacement mixing probabilities for the
    known concepts (the unknown slot always keeps its own). Broadcasts
    against the hidden-state batch shape."""

    probs_mask: torch.Tensor | None = None
    """(..., num_concepts) bool: where True, ``probs`` replaces the
    model's own probability; elsewhere the model's own value is kept.
    None (with ``probs`` given) means replace everywhere."""

    uncertain_band: float | None = None
    """When set, ``probs`` only replaces the model's own probability where
    that probability lies within ``uncertain_band`` of 0.5. Feeding a
    hard 0/1 value displaces the model's own ``p`` by ``1 - p`` toward
    one pole and by ``p`` toward the other. For a single position
    "truth" (injects the true label ``L``) and "flip" (injects ``1 -
    L``) displace by complementary amounts -- ``|L - p|`` and
    ``|1 - L - p|``, which sum to exactly 1 -- so they are equal only
    where ``p`` is exactly 0.5. Restricting to the uncertain band
    narrows the population toward that point, which shrinks but does
    NOT remove the mean-displacement asymmetry between "truth" and
    "flip"; it is a bias-narrowing device, not an equalizer. Report the
    residual asymmetry alongside any truth-vs-flip comparison rather
    than assuming it away (found in code+paper audit, 2026-08-31)."""

    zero_known: bool = False
    """Zero every known concept's mixed embedding (completeness probe:
    how much task signal survives on the unknown channel alone)."""

    zero_unknown: bool = False
    """Zero the unknown concept's mixed embedding (how much task signal
    flows outside the supervised concepts)."""

    zero_residual: bool = False
    """Zero the unexplained residual ``eps``, for a bottleneck that has
    one as a channel distinct from the unknown concepts
    (:class:`DecomposedConceptBottleneck`). The mixture bottleneck folds
    "unknown concept" and "unexplained remainder" into one slot, so there
    ``zero_unknown`` covers both and this is ignored."""


@dataclass(frozen=True)
class TeacherForcing:
    """Steerling's concept teacher forcing, for one training step.

    Routing ``h_bar`` through predicted activations lets the language
    modeling loss push those activations to encode more than the labeled
    concepts, which is concept leakage. Koh et al. (2020) avoid it by
    feeding ground truth forward always; Steerling does the same on a
    schedule, replacing ``k_hat`` with ``k_hat_gt`` with probability
    ``alpha_known`` and ``u_hat`` with ``u_hat_gt = h - k_hat_gt`` with
    probability ``alpha_unknown``, annealing both so the model comes to
    rely on its own heads as they become accurate.

    The draw is per forward pass, matching "with probability alpha(s) at
    training step s". Labels here are subject-level and broadcast across
    positions, since a concept is a property of the visit rather than of
    one token.

    NOTE on the schedules: Steerling's prose says ``alpha_unknown``
    anneals FROM 1, while its Table 26 lists ``0.0 -> 0.5``. The two
    disagree; :func:`annealed_alpha` takes explicit endpoints so the
    choice is made in the config and recorded with the run rather than
    guessed here.
    """

    concept_labels: torch.Tensor
    """(batch, num_concepts) ground-truth activations."""

    alpha_known: float = 0.0
    """Probability of substituting ``k_hat_gt`` this step."""

    alpha_unknown: float = 0.0
    """Probability of substituting ``u_hat_gt`` this step."""


def annealed_alpha(
    step: int,
    anneal_steps: int,
    *,
    start: float,
    end: float,
    cosine: bool = False,
) -> float:
    """Anneal from ``start`` to ``end`` over ``anneal_steps``, then hold.

    Steerling ramps both teacher-forcing probabilities to their steady
    state by step ``0.10 * max_steps`` and holds them there, cosine for
    the known head and linear for the unknown one. Our training loop is
    epoch-based over a stream and never knows ``max_steps``, so the ramp
    length is given in absolute steps instead; at our run lengths (about
    45k steps) their tenth corresponds to roughly 4,500.
    """
    if anneal_steps <= 0:
        return end
    progress = min(1.0, step / anneal_steps)
    if cosine:
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * progress


def intervention_apply_mask(
    intervention: BottleneckIntervention, own_probs: torch.Tensor
) -> torch.Tensor | None:
    """Where an intervention's ``probs`` actually replace the model's own.

    Combines ``probs_mask`` with the ``uncertain_band`` restriction;
    ``None`` means "everywhere". Exposed so an evaluation harness can
    account for exactly the entries the model replaced.
    """
    apply: torch.Tensor | None = None
    if intervention.probs_mask is not None:
        apply = intervention.probs_mask.expand_as(own_probs)
    if intervention.uncertain_band is not None:
        band = (own_probs - 0.5).abs() < intervention.uncertain_band
        apply = band if apply is None else (apply & band)
    return apply


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
    global_pairs : bool
        Leakage control (see the module docstring). ``False`` (default, the
        CEM design): each known concept's ``(w+, w-)`` pair is produced from
        the hidden state, so both vectors encode the context and overriding
        ``c`` only re-weights two context-carrying vectors. ``True``: each
        known concept's pair is a learned, input-independent parameter, so
        the concept slot carries exactly one number, ``c``, and an
        intervention on ``c`` fully determines that slot; the concept
        probability is then predicted directly from the hidden state. The
        unknown slot keeps its context-dependent pair either way (it is the
        residual channel by design).
    unknown_dim : Optional[int]
        Width of the unknown (residual) slot; defaults to ``embedding_dim``.
        Smaller widths cap how much task signal can bypass the concepts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_concepts: int,
        embedding_dim: int,
        *,
        concept_dropout: float = 0.1,
        global_pairs: bool = False,
        unknown_dim: int | None = None,
    ) -> None:
        """Initialize the concept bottleneck layer."""
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        unknown_dim = embedding_dim if unknown_dim is None else int(unknown_dim)
        if unknown_dim <= 0:
            raise ValueError("unknown_dim must be positive")

        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.unknown_dim = unknown_dim
        self.global_pairs = bool(global_pairs)
        self.num_slots = num_concepts + 1  # known concepts + 1 unknown concept
        self.output_dim = num_concepts * embedding_dim + unknown_dim

        self.dropout = nn.Dropout(concept_dropout)
        self.context_act = nn.LeakyReLU()
        # Parameter layout is kept byte-compatible with earlier checkpoints
        # for the default configuration (context pairs, unknown_dim ==
        # embedding_dim): one context_proj over all slots (unknown last) and
        # one (num_slots, 2*embedding_dim) probability weight.
        if self.global_pairs:
            # Known concepts: input-independent (w+, w-) per concept and a
            # direct probability head; the unknown slot keeps a context pair.
            self.pair_embeddings = nn.Parameter(
                torch.empty(num_concepts, 2, embedding_dim)
            )
            nn.init.xavier_uniform_(self.pair_embeddings)
            self.concept_prob_proj = nn.Linear(hidden_size, num_concepts)
            self.context_proj = nn.Linear(hidden_size, 2 * unknown_dim)
            self.unknown_prob_weight = nn.Parameter(torch.empty(1, 2 * unknown_dim))
            self.unknown_prob_bias = nn.Parameter(torch.zeros(1))
            nn.init.xavier_uniform_(self.unknown_prob_weight)
        else:
            # One Linear producing every slot's (w+, w-) pair: equivalent to
            # independent per-concept context networks since each slot's
            # output only depends on its own weight rows.
            self.context_proj = nn.Linear(
                hidden_size, num_concepts * 2 * embedding_dim + 2 * unknown_dim
            )
            if unknown_dim == embedding_dim:
                # Per-slot probability network Psi_i([w+, w-]) -> logit; one
                # weight row per slot so no slot's logit sees another's pair.
                self.prob_weight = nn.Parameter(
                    torch.empty(self.num_slots, 2 * embedding_dim)
                )
                self.prob_bias = nn.Parameter(torch.zeros(self.num_slots))
                nn.init.xavier_uniform_(self.prob_weight)
            else:
                # The unknown slot has its own width, so its own weight.
                self.prob_weight = nn.Parameter(
                    torch.empty(num_concepts, 2 * embedding_dim)
                )
                self.prob_bias = nn.Parameter(torch.zeros(num_concepts))
                nn.init.xavier_uniform_(self.prob_weight)
                self.unknown_prob_weight = nn.Parameter(torch.empty(1, 2 * unknown_dim))
                self.unknown_prob_bias = nn.Parameter(torch.zeros(1))
                nn.init.xavier_uniform_(self.unknown_prob_weight)

        # Independent of the concept-value pathway above: predicts whether
        # each known concept would be observed at all, from the same
        # (dropout-applied) hidden state. See the module docstring for why
        # this is a separate, supervised head rather than an input feature.
        self.observability_proj = nn.Linear(hidden_size, num_concepts)

    def concept_pair_directions(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-concept intervention direction ``w+ - w-``, shape (..., k, d).

        Raising concept ``i``'s mixing probability by ``delta`` moves that
        slot's mixed embedding by exactly ``delta * (w+ - w-)``, so through
        the linear LM head this direction determines which token logits a
        concept override moves (the Known Concept Alignment analysis,
        :mod:`odyssey.inference.concept_attribution`). With ``global_pairs``
        the direction is an input-independent parameter (exact); otherwise
        it is recomputed from the hidden state via the same context
        projection the forward pass uses. Call in eval mode: in train mode
        the dropout draw here would differ from the forward pass's.
        """
        k, d = self.num_concepts, self.embedding_dim
        batch_shape = hidden_states.shape[:-1]
        if self.global_pairs:
            diff = self.pair_embeddings[:, 0, :] - self.pair_embeddings[:, 1, :]
            return diff.expand(*batch_shape, k, d)
        x = self.dropout(hidden_states)
        context = self.context_act(self.context_proj(x))
        known_ctx = context[..., : k * 2 * d].view(*batch_shape, k, 2, d)
        directions: torch.Tensor = known_ctx[..., 0, :] - known_ctx[..., 1, :]
        return directions

    needs_calibration_directions = True
    """Output calibration needs the ``(w+ - w-)`` directions measured over
    data: the poles are functions of the hidden state unless
    ``global_pairs``, so the displacement per unit of probability is not a
    parameter and has to be estimated."""

    def unit_displacements(self, directions: torch.Tensor | None) -> torch.Tensor:
        """(k, output_dim) added to the bottleneck output per unit of ``p_i``.

        Concept ``i`` owns one embedding block of the concatenation, so a
        unit of its probability moves only that block, by
        ``(w+ - w-)_i``. Scattering the directions into their blocks lets
        output calibration project them through the LM head without
        knowing how this bottleneck is laid out.
        """
        if directions is None:
            raise ValueError(
                "the mixture bottleneck's per-unit displacement is data "
                "dependent; pass mean_concept_directions(...) output"
            )
        k, d = self.num_concepts, self.embedding_dim
        out = directions.new_zeros((k, self.output_dim))
        for i in range(k):
            out[i, i * d : (i + 1) * d] = directions[i]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        intervention: BottleneckIntervention | None = None,
    ) -> ConceptBottleneckOutput:
        """Project hidden states into known + unknown concept embeddings.

        ``intervention`` edits the mixing step only (see
        :class:`BottleneckIntervention`): the returned
        ``concept_logits``/``concept_probs``/observability outputs are
        always the model's own predictions.
        """
        batch_shape = hidden_states.shape[:-1]
        x = self.dropout(hidden_states)
        k, d, u = self.num_concepts, self.embedding_dim, self.unknown_dim

        # Known concepts: (w+, w-) pairs and activation logits.
        if self.global_pairs:
            pairs = self.pair_embeddings.expand(*batch_shape, k, 2, d)
            k_pos, k_neg = pairs[..., 0, :], pairs[..., 1, :]
            concept_logits = self.concept_prob_proj(x)
            unknown_ctx = self.context_act(self.context_proj(x)).view(
                *batch_shape, 2, u
            )
        else:
            context = self.context_act(self.context_proj(x))
            known_ctx = context[..., : k * 2 * d].view(*batch_shape, k, 2, d)
            unknown_ctx = context[..., k * 2 * d :].view(*batch_shape, 2, u)
            k_pos, k_neg = known_ctx[..., 0, :], known_ctx[..., 1, :]
            joint = torch.cat([k_pos, k_neg], dim=-1)  # (..., k, 2d)
            concept_logits = (
                torch.einsum("...sd,sd->...s", joint, self.prob_weight[:k])
                + self.prob_bias[:k]
            )
        concept_probs = torch.sigmoid(concept_logits)

        # Unknown (residual) slot: always a context-dependent pair.
        u_pos, u_neg = unknown_ctx[..., 0, :], unknown_ctx[..., 1, :]
        if hasattr(self, "unknown_prob_weight"):
            u_weight, u_bias = self.unknown_prob_weight[0], self.unknown_prob_bias[0]
        else:  # shared (num_slots, 2d) weight: the unknown slot is the last row
            u_weight, u_bias = self.prob_weight[k], self.prob_bias[k]
        unknown_logit = torch.cat([u_pos, u_neg], dim=-1) @ u_weight + u_bias
        unknown_prob = torch.sigmoid(unknown_logit)

        mix_probs = concept_probs
        if intervention is not None and intervention.probs is not None:
            override = intervention.probs.to(concept_probs.dtype).expand_as(
                concept_probs
            )
            apply = intervention_apply_mask(intervention, concept_probs)
            if apply is not None:
                override = torch.where(apply, override, concept_probs)
            mix_probs = override

        concept_embeddings = (
            mix_probs.unsqueeze(-1) * k_pos + (1 - mix_probs.unsqueeze(-1)) * k_neg
        )
        unknown_embedding = (
            unknown_prob.unsqueeze(-1) * u_pos
            + (1 - unknown_prob.unsqueeze(-1)) * u_neg
        )
        if intervention is not None and intervention.zero_known:
            concept_embeddings = torch.zeros_like(concept_embeddings)
        if intervention is not None and intervention.zero_unknown:
            unknown_embedding = torch.zeros_like(unknown_embedding)

        bottleneck = torch.cat(
            [concept_embeddings.reshape(*batch_shape, k * d), unknown_embedding],
            dim=-1,
        )

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
    concept_mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
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
    and the unknown concept's embedding (Eq. 5 of the CBGM paper). When the
    unknown slot has a different width (``unknown_dim``), the similarity is
    undefined and the term is zero: the width cap is then the leakage
    control instead of the orthogonality penalty.
    """
    if concept_embeddings.shape[-1] != unknown_embedding.shape[-1]:
        return unknown_embedding.new_zeros(())
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
    task: float = 1.0
    """Weight of the forecasting (task) loss. 0 excludes it from the
    backward pass entirely, so the backbone/bottleneck are shaped only
    by concept/orthogonality/observability supervision -- the
    "independent training" regime (Koh et al. 2020's classical CBM
    training scheme, applied to CEM-style embeddings): concept
    representations that carry no gradient signal from what helps the
    forecast, only from being a correct, well-separated concept."""

    reconstruction: float = 1.0
    """Weight of :func:`reconstruction_loss` (Steerling's lambda_rec, 1.0).
    Only :class:`DecomposedConceptBottleneck` produces this term."""

    independence: float = 1.0
    """Weight of :func:`independence_loss` (Steerling's lambda_indep, 1.0).
    Only :class:`DecomposedConceptBottleneck` produces this term."""

    concept_pos_weight: torch.Tensor | None = None
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
    concept_mask: torch.Tensor | None = None,
    weights: ConceptBottleneckLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        weights.task * task_loss
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


class DecomposedBottleneckOutput(NamedTuple):
    """Outputs of a :class:`DecomposedConceptBottleneck` forward pass."""

    concept_logits: torch.Tensor
    """(..., num_concepts) known-concept activation logits, pre-sigmoid."""

    concept_probs: torch.Tensor
    """(..., num_concepts) sigmoid(concept_logits); ``k`` in Equation (6)."""

    concept_embeddings: torch.Tensor
    """(..., num_concepts) each concept's scalar weight on its embedding.

    The per-concept contribution is ``k_i * K_i``, of shape
    ``(..., n, d)``; materializing it costs about a gigabyte at our
    training geometry, so the coefficient is reported and the caller
    multiplies by :attr:`DecomposedConceptBottleneck.known_embeddings`
    when it wants the vectors.
    """

    unknown_embedding: torch.Tensor
    """(..., hidden_size) ``u_hat``, the unknown concepts' weighted sum."""

    bottleneck: torch.Tensor
    """(..., hidden_size) ``h_bar = k_hat + u_hat + eps``; the only thing
    the heads see."""

    observability_logits: torch.Tensor
    """(..., num_concepts) predicted "would this concept be observed"."""

    observability_probs: torch.Tensor
    """(..., num_concepts) sigmoid(observability_logits)."""

    known_part: torch.Tensor
    """(..., hidden_size) ``k_hat``, the known concepts' weighted sum."""

    unknown_probs: torch.Tensor
    """(..., num_unknown) ``u`` in Equation (6)."""

    residual: torch.Tensor
    """(..., hidden_size) ``eps = h - k_hat - u_hat``, computed from the
    model's OWN parts and held fixed under an intervention."""

    hidden: torch.Tensor
    """(..., hidden_size) the backbone state ``h``, kept because the
    reconstruction target is defined against it."""

    unknown_embedding_detached: torch.Tensor
    """``u_hat`` recomputed from a detached head input, for Equations
    (12) and (14) ONLY.

    Steerling states of the two auxiliary losses that "gradients are
    detached so that only the unknown head is updated". Using
    :attr:`unknown_embedding` for them instead backpropagates the
    reconstruction and independence terms through the unknown head and
    into the backbone, which is not what the paper specifies. This
    carries the same values, computed from the same dropout mask, but
    with the path to the backbone cut: gradients still reach the unknown
    projection and the unknown embeddings, which are the unknown head.

    The LM loss still flows through :attr:`unknown_embedding` into the
    backbone, as it should; only the auxiliary terms are confined."""


class DecomposedConceptBottleneck(nn.Module):
    """Decompose the hidden state into known + unknown concepts + residual.

    This follows the concept module of \\citet{madsen2026steerling}
    exactly. For each token the backbone produces ``h``, and the module
    splits it three ways (their Equation 5)::

        h_bar = k_hat + u_hat + eps,    eps = h - k_hat - u_hat

    with ``k_hat = k @ K`` over the ``n`` supervised concepts and
    ``u_hat = u @ U`` over ``m >> n`` unsupervised ones, where
    ``k = sigmoid(f(h))`` and ``u = sigmoid(g(h))`` (their Equation 6/7).
    Only ``h_bar`` is passed downstream, so with a linear head every logit
    decomposes as ``k_hat @ W + u_hat @ W + eps @ W``.

    THE DECOMPOSITION IS AN ALGEBRAIC IDENTITY: ``h_bar == h`` exactly.
    That is worth stating plainly, because it is what our earlier additive
    attempt got wrong. Writing ``out = h + sum_i p_i v_i`` ADDS a concept
    term to an untouched backbone stream; nothing then relates the
    ``v_i`` to ``h``, the task loss is minimized through ``h`` alone, and
    the concept term decays into decoration. Measured on the full-scale
    eICU run: deleting all 26 concepts cost 1.6% of accuracy. Steerling
    names this failure mode, residual domination, and predicts it exactly
    for a model trained without the losses below -- "algebraically exact
    but practically vacuous".

    So the identity is not where the pressure comes from. Three things
    supply it, and all three must be present:

    * ``residual_dropout`` on ``eps`` during training, so the model cannot
      route prediction through the unexplained channel and expect it to
      survive;
    * :func:`reconstruction_loss`, which pins ``u_hat`` to
      ``h - k_hat_gt`` and so forces the decomposition to be real rather
      than nominal;
    * :func:`independence_loss`, which stops the unknown head re-encoding
      what the concepts already say.

    Every arm this project ran before 2026-09-01 lacked all three, and in
    the two arms that tried to control the residual by narrowing it, the
    only mechanism present (``orthogonality_loss``) is identically zero
    whenever the unknown slot's width differs from ``embedding_dim``. The
    residual has therefore never actually been constrained here.

    Steerability, and why the residual is frozen under an override
    ---------------------------------------------------------------
    ``eps`` is computed from the model's own ``k_hat`` and ``u_hat`` and
    then held fixed. This matters: if ``eps`` were recomputed after an
    override it would absorb the edit exactly, since
    ``k_hat' + u_hat + (h - k_hat' - u_hat) == h`` for any ``k_hat'``, and
    the intervention would be a no-op by construction. Holding it fixed
    makes an override move the output by exactly ``(k' - k) @ K``, a known
    displacement independent of the patient.

    Parameters
    ----------
    hidden_size : int
        Width of the backbone hidden state ``h``.
    num_concepts : int
        Number of supervised concepts, ``n``.
    unknown_ratio : int
        ``m = unknown_ratio * n`` unsupervised concepts. Steerling uses 3.
    unknown_rank : int | None
        Factorize ``U = A @ B`` with this rank, as Steerling does to stop
        a 101k-concept embedding matrix dominating the parameter count.
        At our ``n`` the full matrix is a few tens of thousands of
        parameters, so ``None`` (no factorization) is the sane default.
    concept_dropout : float
        Dropout on ``h`` before the heads (Steerling's ``p_cfg``, 0.1).
    residual_dropout : float
        Dropout on ``eps`` during training (their ``p_eps``). They use 0.1
        in pretraining and raise it to 0.3 when tightening the model,
        "applying more pressure on eps to vanish".
    """

    def __init__(
        self,
        hidden_size: int,
        num_concepts: int,
        *,
        unknown_ratio: int = 3,
        unknown_rank: int | None = None,
        concept_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        """Initialize the decomposition."""
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if unknown_ratio <= 0:
            raise ValueError("unknown_ratio must be positive")
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_unknown = unknown_ratio * num_concepts
        # The heads see the backbone's own width: this decomposes h, it
        # does not replace it with a narrower concatenation.
        self.output_dim = hidden_size

        self.dropout = nn.Dropout(concept_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.known_proj = nn.Linear(hidden_size, num_concepts)
        self.unknown_proj = nn.Linear(hidden_size, self.num_unknown)
        self.observability_proj = nn.Linear(hidden_size, num_concepts)

        self.known_embeddings = nn.Parameter(torch.empty(num_concepts, hidden_size))
        nn.init.xavier_uniform_(self.known_embeddings)
        self.unknown_rank = unknown_rank
        if unknown_rank is None:
            self.unknown_embeddings_full = nn.Parameter(
                torch.empty(self.num_unknown, hidden_size)
            )
            nn.init.xavier_uniform_(self.unknown_embeddings_full)
        else:
            self.unknown_factor_a = nn.Parameter(
                torch.empty(self.num_unknown, unknown_rank)
            )
            self.unknown_factor_b = nn.Parameter(torch.empty(unknown_rank, hidden_size))
            nn.init.xavier_uniform_(self.unknown_factor_a)
            nn.init.xavier_uniform_(self.unknown_factor_b)

    def unknown_embeddings(self) -> torch.Tensor:
        """(num_unknown, hidden_size) unknown-concept embeddings ``U``."""
        if self.unknown_rank is None:
            return self.unknown_embeddings_full
        return self.unknown_factor_a @ self.unknown_factor_b

    def known_contribution(self, probs: torch.Tensor) -> torch.Tensor:
        """``k_hat`` for given activations: Equation (7)'s known half."""
        return probs @ self.known_embeddings

    needs_calibration_directions = False
    """Output calibration needs no data pass here: a unit of ``k_i`` adds
    exactly ``K_i``, which is a parameter."""

    def unit_displacements(
        self, directions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(k, hidden_size) added to ``h_bar`` per unit of ``k_i``: ``K_i``.

        ``directions`` is accepted and ignored so both bottlenecks present
        one interface; it is meaningless here because the displacement is
        a parameter rather than an estimate.
        """
        return self.known_embeddings.detach()

    def unaccounted_orthogonality(self) -> torch.Tensor:
        """No parameter-space penalty: redundancy is handled by a loss.

        The mixture bottleneck hides its redundancy in the parameters, so
        ``fold_in_bottleneck_orthogonality`` exists to reach it. Here the
        known/unknown redundancy is measured on activations by
        :func:`independence_loss`, which is where Steerling puts it, so
        there is nothing left for the fold-in to add.
        """
        return self.known_embeddings.new_zeros(())

    def auxiliary_losses(
        self,
        output: DecomposedBottleneckOutput,
        concept_labels: torch.Tensor,
        concept_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the two losses that keep the decomposition honest.

        Steerling's Equations (12) and (14). Returned unweighted and keyed
        by the name they carry in the training logs, so
        :func:`fold_in_bottleneck_losses` can apply the weights without
        knowing which bottleneck produced them.
        """
        return {
            "reconstruction_loss": reconstruction_loss(
                output.unknown_embedding_detached,
                output.hidden,
                self.known_embeddings,
                concept_labels,
                concept_mask,
            ),
            "independence_loss": independence_loss(
                output.known_part,
                output.unknown_embedding_detached,
                None
                if concept_mask is None
                else _broadcast_over_positions(concept_mask, output.hidden)
                .any(dim=-1)
                .expand(output.hidden.shape[:-1]),
            ),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        intervention: BottleneckIntervention | None = None,
        *,
        teacher: TeacherForcing | None = None,
    ) -> DecomposedBottleneckOutput:
        """Split ``h`` into known concepts, unknown concepts and a residual.

        ``teacher`` substitutes the ground-truth ``k_hat_gt``/``u_hat_gt``
        when forming ``h_bar`` with the given probabilities, which is
        Steerling's teacher forcing. Substitution only happens in training
        mode; the annealing schedule lives in the training loop.
        """
        x = self.dropout(hidden_states)
        concept_logits = self.known_proj(x)
        unknown_logits = self.unknown_proj(x)
        observability_logits = self.observability_proj(x)
        concept_probs = torch.sigmoid(concept_logits)
        unknown_probs = torch.sigmoid(unknown_logits)

        own_known = self.known_contribution(concept_probs)
        unknown_embeddings = self.unknown_embeddings()
        unknown_part = unknown_probs @ unknown_embeddings
        # Same x, same dropout mask, but detached: the auxiliary losses must
        # update the unknown head without pushing on the backbone.
        unknown_part_detached = (
            torch.sigmoid(self.unknown_proj(x.detach())) @ unknown_embeddings
        )
        # eps from the model's OWN parts, before any edit. Recomputing it
        # after an override would cancel the override exactly.
        residual = hidden_states - own_known - unknown_part

        mix_probs = concept_probs
        if intervention is not None and intervention.probs is not None:
            override = intervention.probs.to(concept_probs.dtype).expand_as(
                concept_probs
            )
            apply = intervention_apply_mask(intervention, concept_probs)
            if apply is not None:
                override = torch.where(apply, override, concept_probs)
            mix_probs = override
        known_part = (
            own_known
            if mix_probs is concept_probs
            else self.known_contribution(mix_probs)
        )

        used_known, used_unknown = known_part, unknown_part
        if teacher is not None and self.training:
            labels = teacher.concept_labels.to(hidden_states.dtype)
            while labels.dim() < hidden_states.dim():
                labels = labels.unsqueeze(-2)
            known_gt = self.known_contribution(labels.expand(*concept_probs.shape))
            if torch.rand(()).item() < teacher.alpha_known:
                used_known = known_gt
            if torch.rand(()).item() < teacher.alpha_unknown:
                used_unknown = hidden_states - known_gt
        used_residual = self.residual_dropout(residual)
        if intervention is not None:
            if intervention.zero_known:
                used_known = torch.zeros_like(used_known)
            if intervention.zero_unknown:
                used_unknown = torch.zeros_like(used_unknown)
            if intervention.zero_residual:
                used_residual = torch.zeros_like(used_residual)

        return DecomposedBottleneckOutput(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            concept_embeddings=mix_probs,
            unknown_embedding=unknown_part,
            unknown_embedding_detached=unknown_part_detached,
            bottleneck=used_known + used_unknown + used_residual,
            observability_logits=observability_logits,
            observability_probs=torch.sigmoid(observability_logits),
            known_part=known_part,
            unknown_probs=unknown_probs,
            residual=residual,
            hidden=hidden_states,
        )


def _broadcast_over_positions(
    labels: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Give subject-level labels a position axis to match ``reference``.

    Concept labels here are one row per subject (or per visit), while the
    hidden state has a position axis. A concept is a property of the
    visit, so the label applies at every position within it; inserting
    singleton axes lets it broadcast rather than silently misaligning
    sequence length against concept count.
    """
    while labels.dim() < reference.dim():
        labels = labels.unsqueeze(-2)
    return labels


def reconstruction_loss(
    unknown_part: torch.Tensor,
    hidden: torch.Tensor,
    known_embeddings: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Steerling Equation (12): pin ``u_hat`` to ``h - k_hat_gt``.

    The unknown head is trained to carry exactly the part of the hidden
    state the true concepts do not explain, so the three-way split is a
    real decomposition rather than a bookkeeping identity. Without this
    the unknown channel is free to absorb the whole prediction, which is
    the residual-domination failure mode.

    The target is detached: it is a target, and Steerling notes that
    gradients here should reach the unknown head rather than the backbone
    or the known side.

    ``concept_mask`` marks positions where the concept labels are
    observed. Steerling averages over masked diffusion positions; the
    analogue here is to average over positions with usable labels, since
    an unobserved label would otherwise define a target from a
    ``k_hat_gt`` built out of zeros.
    """
    labels = _broadcast_over_positions(
        concept_labels.to(known_embeddings.dtype), hidden
    )
    if concept_mask is not None:
        labels = labels * _broadcast_over_positions(
            concept_mask.to(labels.dtype), hidden
        )
    target = (hidden - labels @ known_embeddings).detach()
    per_position = (unknown_part - target).pow(2).sum(-1)
    if concept_mask is None:
        return per_position.mean()
    usable = (
        _broadcast_over_positions(concept_mask, hidden)
        .any(dim=-1)
        .expand_as(per_position)
        .to(per_position.dtype)
    )
    total = usable.sum()
    if not bool(total > 0):
        return per_position.new_zeros(())
    return (per_position * usable).sum() / total


def independence_loss(
    known_part: torch.Tensor,
    unknown_part: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Steerling Equations (13)-(14): decorrelate known from unknown.

    A normalized cross-covariance penalty in the spirit of HSIC with a
    linear kernel. Centre both representations over the batch, then
    penalize ``||Psi^T Phi||_F^2 / (d^2 (N - 1))``. Reconstruction alone
    does not stop the unknown head re-encoding what the concepts already
    say; this does.

    Gradients flow only through the unknown side: Steerling treats the
    known representation as a fixed input here.
    """
    known = known_part.reshape(-1, known_part.shape[-1])
    unknown = unknown_part.reshape(-1, unknown_part.shape[-1])
    if valid_mask is not None:
        keep = valid_mask.reshape(-1).to(torch.bool)
        known, unknown = known[keep], unknown[keep]
    n_positions, width = unknown.shape
    if n_positions < 2:
        return unknown.new_zeros(())
    phi = (known - known.mean(dim=0, keepdim=True)).detach()
    psi = unknown - unknown.mean(dim=0, keepdim=True)
    cross = psi.transpose(0, 1) @ phi
    return cross.pow(2).sum() / (width**2 * (n_positions - 1))


def fold_in_bottleneck_losses(
    bottleneck: nn.Module,
    output: object,
    concept_labels: torch.Tensor,
    total: torch.Tensor,
    components: dict[str, torch.Tensor],
    *,
    concept_mask: torch.Tensor | None = None,
    weights: ConceptBottleneckLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add any auxiliary losses only the bottleneck itself can compute.

    ``combined_loss`` derives its terms from the shapes the mixture
    bottleneck produces, which suits that design and nothing else. Rather
    than branch on the bottleneck's type at every call site -- which would
    need editing for each new variant -- a bottleneck may expose
    ``auxiliary_losses()`` and this folds the result in under the matching
    weight. A bottleneck without one is left alone.
    """
    report = getattr(bottleneck, "auxiliary_losses", None)
    if report is None:
        return total, components
    weights = weights or ConceptBottleneckLossWeights()
    scale = {
        "reconstruction_loss": weights.reconstruction,
        "independence_loss": weights.independence,
    }
    extra = report(output, concept_labels, concept_mask)
    for name, value in extra.items():
        total = total + scale.get(name, 1.0) * value
        components[name] = value.detach()
    return total, components
