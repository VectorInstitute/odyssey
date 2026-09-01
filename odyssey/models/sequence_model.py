"""Sequence models built on a shared backbone: with and without a concept bottleneck.

:class:`ConceptBottleneckSequenceModel` is backbone -> bottleneck -> LM head,
the interpretable model this project is actually about.
:class:`BaselineSequenceModel` is backbone -> LM head directly, with no
bottleneck, kept alongside it so the two can be trained and compared: does
the bottleneck cost anything in raw forecasting accuracy, and does
attribution actually differ between them. Both are backbone-agnostic, so
the same classes run against the lightweight CPU stand-in backbone in
tests/CI and the real EHRHybridBackbone on a CUDA host.

Both models support two training regimes:

- Single-sequence-per-row (``compute_loss``): one full patient sequence per
  batch row, as :func:`odyssey.data.sequences.collate_patient_sequences`
  produces. Concept supervision pools at each row's own last non-padding
  position.
- Packed, chunked, multi-lane streaming (``compute_streaming_loss``): a
  :class:`~odyssey.data.streaming.StreamingChunk` from
  :class:`~odyssey.data.streaming.PackedLaneSampler`, where a lane can
  contain fragments of several different patients. Concept supervision
  pools only at each patient's true last event
  (``chunk.patient_end``), never merely their last position within one
  chunk -- a patient whose history spans multiple chunks must not be
  supervised early, since events later in their stay could still change
  the true label. See ``research_journal/02_sequence_scoping_methodology.html``
  Section 05.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone, TimeAwareState
from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    ConceptBottleneck,
    ConceptBottleneckLossWeights,
    ConceptBottleneckOutput,
    DecomposedConceptBottleneck,
    TeacherForcing,
    combined_loss,
    fold_in_bottleneck_losses,
)
from odyssey.models.time_to_event import (
    EventHazardHeads,
    TimeToEventHead,
    event_hazard_nll,
    gap_survival_valid_mask,
    hazard_nll,
)
from odyssey.models.value_head import (
    DEFAULT_QUANTILE_LEVELS,
    ValueQuantileHead,
    value_quantile_loss,
    value_target_valid_mask,
)


if TYPE_CHECKING:  # the targets live in training; avoid a runtime import cycle
    from odyssey.training.event_targets import EventHazardTargets


@dataclass
class ForecastObjective:
    """How the next-event forecasting loss is formed over a streaming chunk.

    The record is a sequence of *bundles* -- events sharing one
    timestamp (a lab panel, a medication order set, the diagnoses coded
    at discharge) with no meaningful order inside a bundle. Plain
    next-token cross-entropy grades the model on that arbitrary
    within-bundle order and, because labs are ~85% of positions, is
    dominated by them. This objective fixes both:

    - ``bundle_invariant``: at each position the likelihood credited is
      the total probability of every *not-yet-emitted* member of the
      target's bundle (``-log sum_{v in remaining} p(v)``), so the model
      is asked "which events are coming at this instant", not "which one
      did the ETL happen to write first". Reduces exactly to cross-entropy
      for singleton bundles and lower-bounds it otherwise. Matches the
      set-based evaluation in :mod:`odyssey.inference.run_inference`,
      including its same-family restriction whenever ``token_types`` is
      set (always, from :func:`odyssey.training.train.build_objective`).
    - ``family_weights`` / ``token_types``: per-position weights by the
      target's code family (``token_types`` maps token id -> family id,
      ``family_weights`` maps family id -> weight), normalized inside the
      loss so its scale stays a weighted mean. Lets medications,
      procedures, diagnoses and billing carry real gradient.
    - ``time_weight``: weight of the time-to-next-event hazard loss
      (:mod:`odyssey.models.time_to_event`), if the model has a time head.
    - ``event_hazard_weight``: weight of the per-event hazard loss (time
      to vasopressor start, ICU admission, ...), if the model has event
      heads; the targets come per chunk from
      :mod:`odyssey.training.event_targets`.
    - ``value_head_weight``: weight of the masked pinball loss for the
      next event's magnitude (:mod:`odyssey.models.value_head`), if the
      model has a value head. Additive alongside ``time_weight`` and
      ``event_hazard_weight``; the bin-token representation of value is
      unchanged either way (see the value head module docstring).

    The default instance reproduces the original objective exactly.
    """

    bundle_invariant: bool = False
    family_weights: torch.Tensor | None = None
    token_types: torch.Tensor | None = None
    time_weight: float = 0.0
    event_hazard_weight: float = 0.0
    value_head_weight: float = 0.0


class ForwardWithFeatures(NamedTuple):
    """Uniform forward result for both sequence model variants.

    ``features`` is what the time-to-event and per-event hazard heads
    read: the bottleneck output for the concept-bottleneck model, the
    backbone hidden state for the baseline. ``bottleneck`` is None for
    the baseline.
    """

    logits: torch.Tensor
    features: torch.Tensor
    bottleneck: ConceptBottleneckOutput | None
    state: TimeAwareState


def _bundle_log_likelihood(
    logp: torch.Tensor,
    targets: torch.Tensor,
    times: torch.Tensor,
    subject_ids: torch.Tensor,
    real: torch.Tensor,
    *,
    token_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """``log sum_{v in remaining bundle members} p_i(v)`` per position, ``(L, T)``.

    Bundles are recovered from the chunk's input timestamps exactly as
    the set-based evaluation does (the target at position ``j`` is the
    input at ``j+1``; sequences are time-sorted per subject, so a bundle
    is a contiguous run). "Remaining" means members at positions ``j >= i``
    of the same bundle: earlier members have already been emitted as
    inputs, so a good model has no business predicting them again.
    Duplicate tokens within the remaining members count once (the first
    remaining occurrence), so the credited mass never exceeds one. The
    final lane position has no in-chunk target time and is scored as a
    singleton (plain cross-entropy). Positions that are not ``real`` get 0.

    With ``token_types`` (token id -> code family), membership is further
    restricted to the *target's own family*, exactly like the set-based
    metric. Without it, a discharge bundle (diagnoses + the discharge
    event + DRG codes + procedures at one instant) would let the model
    earn full credit at a diagnosis position by predicting the
    always-present discharge or DRG token, and the subset run that
    trained that way learned to do exactly that deep inside diagnosis
    bundles (diagnosis set top-1 fell 31% -> 23% while every family-pure
    bundle improved). Callers with a family lookup should pass it.
    """
    lanes, chunk = targets.shape
    device = targets.device
    safe_targets = targets.clamp_min(0)

    tgt_t = times[:, 1:]
    tgt_s = subject_ids[:, 1:]
    new_block = torch.ones_like(tgt_s, dtype=torch.bool)
    new_block[:, 1:] = (tgt_t[:, 1:] != tgt_t[:, :-1]) | (tgt_s[:, 1:] != tgt_s[:, :-1])
    block_id = torch.zeros(lanes, chunk, dtype=torch.long, device=device)
    block_id[:, : chunk - 1] = new_block.long().cumsum(dim=1)
    block_id[:, chunk - 1] = (
        chunk + 1
    )  # unique: no in-chunk block for the last position

    # G[l, i, j] = log p_i(target_j)
    gathered = logp.gather(2, safe_targets.unsqueeze(1).expand(lanes, chunk, chunk))
    same_block = block_id.unsqueeze(2) == block_id.unsqueeze(1)  # (L, i, j)
    pos = torch.arange(chunk, device=device)
    j_ge_i = pos.unsqueeze(0) >= pos.unsqueeze(1)  # (i, j)
    member = same_block & j_ge_i.unsqueeze(0) & real.unsqueeze(1)
    if token_types is not None:
        families = token_types[safe_targets]  # (L, T)
        same_family = families.unsqueeze(2) == families.unsqueeze(1)  # (L, i, j)
        same_block = same_block & same_family
        member = member & same_family
    # Duplicate suppression: drop j if some j' in [i, j) of the same bundle
    # carries the same token. E[l, j', j] flags j' < j with equal tokens;
    # a reverse cumsum over j' answers "any j' >= i" for every i.
    equal = safe_targets.unsqueeze(2) == safe_targets.unsqueeze(1)  # (L, j', j)
    j_lt = pos.unsqueeze(1) < pos.unsqueeze(0)  # (j', j)
    earlier_twin = equal & j_lt.unsqueeze(0) & same_block & real.unsqueeze(2)
    dup = earlier_twin.flip(1).cumsum(1).flip(1) > 0  # (L, i, j)
    member = member & ~dup

    neg_inf = torch.finfo(gathered.dtype).min
    ll = torch.logsumexp(gathered.masked_fill(~member, neg_inf), dim=-1)
    return torch.where(real, ll, torch.zeros_like(ll))


def _pool_last_non_padding(
    values: torch.Tensor, concept_ids: torch.Tensor, padding_idx: int
) -> torch.Tensor:
    """Select each sequence's last non-padding position from ``values``.

    ``values`` is ``(batch, seq_len, ...)``; the result is ``(batch, ...)``.
    Used to align per-token bottleneck activations with the subject-level
    concept labels produced by :mod:`odyssey.data.concepts` (e.g. "was this
    patient ever tachycardic"), which describe the whole stay, not a
    single timestep. Only valid for a batch where each row is one whole
    patient sequence -- see :func:`_pool_patient_ends` for packed chunks.

    Counts non-padding tokens rather than searching for the first padding
    token, since :func:`~odyssey.data.sequences.collate_patient_sequences`
    right-pads to the batch's longest sequence: that longest row (or every
    row, in an already-uniform-length batch) has no padding token at all,
    so a "find the first pad" search would wrongly fall back to position 0
    for it instead of its true last position.
    """
    real_counts = (concept_ids != padding_idx).sum(dim=-1)
    last_idx = (real_counts - 1).clamp(min=0)
    batch_idx = torch.arange(values.shape[0], device=values.device)
    return values[batch_idx, last_idx]


def _pool_patient_ends(values: torch.Tensor, patient_end: torch.Tensor) -> torch.Tensor:
    """Select every position marked ``patient_end`` from ``values``.

    ``values`` is ``(lanes, chunk_size, ...)``; the result is
    ``(n_ends, ...)``, in row-major (lane, position) order, matching
    ``patient_end.nonzero()``. Zero or more patients can end within one
    chunk, including zero -- callers must handle an empty result.
    """
    lane_idx, pos_idx = patient_end.nonzero(as_tuple=True)
    return values[lane_idx, pos_idx]


def _gather_by_subject(
    subject_ids: torch.Tensor, labels: dict[int, torch.Tensor]
) -> torch.Tensor:
    """Stack ``labels[subject_id]`` for each id in ``subject_ids``, in order."""
    try:
        return torch.stack([labels[sid.item()] for sid in subject_ids])
    except KeyError as exc:
        raise KeyError(
            f"no concept labels provided for subject_id {exc.args[0]}, but a "
            "patient_end position in this chunk belongs to them"
        ) from exc


def _gather_by_visit(
    subject_ids: torch.Tensor,
    visit_ids: torch.Tensor,
    labels: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Stack ``labels[(subject_id, visit_id)]`` for each position, in order."""
    try:
        return torch.stack(
            [
                labels[(sid.item(), vid.item())]
                for sid, vid in zip(subject_ids, visit_ids)
            ]
        )
    except KeyError as exc:
        raise KeyError(
            f"no visit-scoped concept labels for (subject_id, visit_id) "
            f"{exc.args[0]}, but a visit_end position in this chunk belongs "
            "to that visit"
        ) from exc


ConceptSupervision = Literal["stay", "visit"]
SequenceModel = Union["BaselineSequenceModel", "ConceptBottleneckSequenceModel"]
ConceptLabelDict = dict[int, torch.Tensor] | dict[tuple[int, int], torch.Tensor]


class _SequenceModelBase(nn.Module):
    """Shared backbone/padding plumbing for the two sequence model variants.

    :class:`BaselineSequenceModel` and :class:`ConceptBottleneckSequenceModel`
    differ only in what sits between the backbone's hidden states and the
    LM head (nothing, vs. a concept bottleneck); the backbone/padding
    bookkeeping and the two next-token loss shapes (whole-sequence shifted,
    vs. pre-shifted streaming) are identical between them, so they live
    here once.
    """

    def __init__(self, backbone: SequenceBackbone, *, padding_idx: int) -> None:
        """Store the shared backbone and padding token id."""
        super().__init__()
        self.backbone = backbone
        self.padding_idx = padding_idx

    def _whole_sequence_next_token_loss(
        self,
        logits: torch.Tensor,
        batch: ClinicalSequenceBatch,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Cross-entropy loss for one full sequence per row, shifted by one.

        Raises rather than silently returning NaN when a sequence has
        fewer than 2 tokens: there is then nothing left after the shift to
        supervise, and ``F.cross_entropy`` averaging over zero elements
        returns NaN instead of failing.
        """
        seq_len = logits.shape[1]
        if seq_len < 2:
            raise ValueError(
                "next-token loss needs at least 2 tokens per sequence, got "
                f"seq_len={seq_len}"
            )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = (labels if labels is not None else batch.concept_ids)[
            :, 1:
        ].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=self.padding_idx,
        )

    def _streaming_next_token_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss for one packed chunk; ``targets`` are pre-shifted.

        A chunk can have zero real targets across every lane (e.g. right
        after :class:`~odyssey.data.streaming.PackedLaneSampler` truncates
        every lane at a reset boundary in the same call, or at the very
        end of an epoch) -- ``F.cross_entropy``'s default mean reduction
        divides by the count of non-ignored elements, which is 0/0 = NaN
        with no guard. Returned as ``logits.sum() * 0.0`` rather than a
        detached zero constant so it stays a real (zero-valued,
        zero-gradient) node in the graph: the training loop's
        ``total.backward()`` must not crash even if this is the only
        connected loss term in the chunk (every other term already
        degrades to a genuine disconnected zero -- see
        :func:`~odyssey.models.concept_bottleneck.combined_loss` --  when
        it has nothing to supervise).
        """
        flat_targets = targets.reshape(-1)
        if not bool((flat_targets != self.padding_idx).any()):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            flat_targets,
            ignore_index=self.padding_idx,
        )

    def _streaming_task_loss(
        self,
        logits: torch.Tensor,
        chunk: StreamingChunk,
        objective: ForecastObjective,
    ) -> torch.Tensor:
        """Next-event loss under ``objective`` (see :class:`ForecastObjective`).

        With the default objective this equals
        :meth:`_streaming_next_token_loss` (up to the family weighting
        being trivially uniform).
        """
        targets = chunk.targets
        real = chunk.real_mask & (targets != self.padding_idx)
        if not bool(real.any()):
            return logits.sum() * 0.0
        if not objective.bundle_invariant and objective.family_weights is None:
            return self._streaming_next_token_loss(logits, targets)

        logp = F.log_softmax(logits, dim=-1)
        if objective.bundle_invariant:
            per_position = -_bundle_log_likelihood(
                logp,
                targets,
                chunk.batch.aux.time_stamps,
                chunk.subject_ids,
                real,
                token_types=objective.token_types,
            )
        else:
            per_position = -logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(
                -1
            )
        weights = real.to(per_position.dtype)
        if objective.family_weights is not None:
            if objective.token_types is None:
                raise ValueError(
                    "family_weights needs token_types (token id -> family)"
                )
            families = objective.token_types[targets.clamp_min(0)]
            weights = weights * objective.family_weights[families].to(weights.dtype)
        return (per_position * weights).sum() / weights.sum()

    def _streaming_event_loss(
        self,
        event_heads: EventHazardHeads | None,
        features: torch.Tensor,
        event_targets: Optional["EventHazardTargets"],
    ) -> torch.Tensor:
        """Censored hazard NLL over the per-event heads (zero-graph if absent)."""
        if event_heads is None or event_targets is None:
            return features.sum() * 0.0
        hazard_logits = event_heads(features)
        return event_hazard_nll(
            hazard_logits,
            event_targets.gap_hours,
            event_targets.observed,
            event_targets.at_risk,
            event_heads.edges,
        )

    def _streaming_time_loss(
        self,
        time_head: TimeToEventHead | None,
        features: torch.Tensor,
        chunk: StreamingChunk,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Hazard NLL for time-to-next-event, and the hazard logits (or None)."""
        if time_head is None:
            return features.sum() * 0.0, None
        hazard_logits = time_head(features)
        gap, valid = gap_survival_valid_mask(
            chunk.batch.aux.time_stamps, chunk.real_mask
        )
        return hazard_nll(hazard_logits, gap, valid, time_head.edges), hazard_logits

    def _streaming_value_loss(
        self,
        value_head: ValueQuantileHead | None,
        features: torch.Tensor,
        chunk: StreamingChunk,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Masked pinball loss for the value-quantile head, and its quantiles (or None).

        Conditions on the TARGET token's own embedding, looked up from
        the model's own input embedding table (no second table) --
        ``self.backbone.embeddings`` is a
        :class:`~odyssey.models.embeddings.CachedEHREmbeddings` on every
        backbone this project has (hybrid, transformer, the CPU tiny-GRU
        stand-in), wrapping a
        :class:`~odyssey.models.embeddings.ClinicalEventEmbeddings` whose
        ``word_embeddings`` is exactly the table ``chunk.targets`` was
        produced against.
        """
        if value_head is None:
            return features.sum() * 0.0, None
        if chunk.batch.aux.values is None:
            return features.sum() * 0.0, None
        target_value, valid = value_target_valid_mask(
            chunk.batch.aux.values, chunk.real_mask
        )
        target_embedding = self.backbone.embeddings.embeddings.word_embeddings(
            chunk.targets
        )
        quantiles = value_head(features, target_embedding)
        loss = value_quantile_loss(
            quantiles, target_value, valid, value_head.quantile_levels
        )
        return loss, quantiles


class BaselineSequenceModel(_SequenceModelBase):
    """Next-token prediction directly from the backbone: no concept bottleneck."""

    def __init__(
        self,
        backbone: SequenceBackbone,
        vocab_size: int,
        *,
        padding_idx: int = 0,
        time_bin_edges: Sequence[float] | None = None,
        event_names: Sequence[str] | None = None,
        event_head_hidden: int = 0,
        value_head: bool = False,
        value_head_hidden: int = 0,
        source: str = "mimic_iv",
    ) -> None:
        """Initialize the baseline sequence model.

        ``time_bin_edges`` adds a time-to-next-event hazard head over the
        backbone hidden state (see :mod:`odyssey.models.time_to_event`);
        ``event_names`` adds per-event hazard heads (same bins);
        ``value_head`` adds the next-event value-quantile head (see
        :mod:`odyssey.models.value_head`), reading the same head features
        plus the target token's own embedding from ``backbone.embeddings``.
        """
        super().__init__(backbone, padding_idx=padding_idx)
        self.lm_head = nn.Linear(backbone.hidden_size, vocab_size)
        head_in = backbone.hidden_size
        self.time_head: TimeToEventHead | None = (
            TimeToEventHead(head_in, time_bin_edges)
            if time_bin_edges is not None
            else None
        )
        self.event_heads: EventHazardHeads | None = (
            EventHazardHeads(
                head_in,
                event_names,
                time_bin_edges if time_bin_edges is not None else (),
                hidden_size=event_head_hidden,
            )
            if event_names
            else None
        )
        self.value_head: ValueQuantileHead | None = (
            ValueQuantileHead(
                head_in,
                backbone.hidden_size,
                DEFAULT_QUANTILE_LEVELS,
                hidden=value_head_hidden,
            )
            if value_head
            else None
        )

    def forward_features(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TimeAwareState]:
        """Return ``(next-token logits, hidden states, new backbone state)``."""
        hidden_states, new_state = self.backbone(
            batch, state=state, reset_mask=reset_mask
        )
        logits: torch.Tensor = self.lm_head(hidden_states)
        return logits, hidden_states, new_state

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
        """Return ``(next-token logits, new backbone state)``."""
        logits, _, new_state = self.forward_features(
            batch, state=state, reset_mask=reset_mask
        )
        return logits, new_state

    def forward_with_features(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> ForwardWithFeatures:
        """Uniform forward (see :class:`ForwardWithFeatures`)."""
        logits, hidden, new_state = self.forward_features(
            batch, state=state, reset_mask=reset_mask
        )
        return ForwardWithFeatures(logits, hidden, None, new_state)

    def compute_loss(
        self, batch: ClinicalSequenceBatch, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute next-token loss over one full sequence per row."""
        logits, _ = self(batch)
        loss = self._whole_sequence_next_token_loss(logits, batch, labels)
        return loss, {"task_loss": loss.detach()}

    def compute_streaming_loss(
        self,
        chunk: StreamingChunk,
        state: TimeAwareState | None = None,
        *,
        objective: ForecastObjective | None = None,
        event_targets: Optional["EventHazardTargets"] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], TimeAwareState]:
        """Compute the forecasting loss over one packed, chunked training step."""
        objective = objective or ForecastObjective()
        logits, hidden, new_state = self.forward_features(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        task_loss = self._streaming_task_loss(logits, chunk, objective)
        time_loss, _ = self._streaming_time_loss(self.time_head, hidden, chunk)
        event_loss = self._streaming_event_loss(self.event_heads, hidden, event_targets)
        value_loss, _ = self._streaming_value_loss(self.value_head, hidden, chunk)
        total = (
            task_loss
            + objective.time_weight * time_loss
            + objective.event_hazard_weight * event_loss
            + objective.value_head_weight * value_loss
        )
        return (
            total,
            {
                "task_loss": task_loss.detach(),
                "time_loss": time_loss.detach(),
                "event_loss": event_loss.detach(),
                "value_loss": value_loss.detach(),
            },
            new_state,
        )


class ConceptBottleneckSequenceModel(_SequenceModelBase):
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
        time_bin_edges: Sequence[float] | None = None,
        event_names: Sequence[str] | None = None,
        event_head_hidden: int = 0,
        concept_global_pairs: bool = False,
        unknown_dim: int | None = None,
        bottleneck_kind: str = "mixture",
        unknown_ratio: int = 3,
        unknown_rank: int | None = None,
        residual_dropout: float = 0.1,
        value_head: bool = False,
        value_head_hidden: int = 0,
        source: str = "mimic_iv",
    ) -> None:
        """Initialize the concept-bottleneck sequence model.

        ``time_bin_edges`` adds a time-to-next-event hazard head that reads
        the bottleneck output (so timing forecasts flow through the
        concepts too); ``event_names`` adds per-event hazard heads on the
        same bins and features; ``value_head`` adds the next-event
        value-quantile head, also reading the bottleneck output (plus the
        target token embedding); see :mod:`odyssey.models.time_to_event`
        and :mod:`odyssey.models.value_head`.
        """
        super().__init__(backbone, padding_idx=padding_idx)
        if bottleneck_kind not in ("mixture", "decomposed"):
            raise ValueError(
                f"bottleneck_kind must be 'mixture' or 'decomposed', got "
                f"{bottleneck_kind!r}"
            )
        self.bottleneck_kind = bottleneck_kind
        self.bottleneck: ConceptBottleneck | DecomposedConceptBottleneck
        if bottleneck_kind == "decomposed":
            # embedding_dim/global_pairs/unknown_dim have no meaning here:
            # this decomposes the backbone state into known concepts,
            # unknown concepts and a residual, all at the backbone's width.
            self.bottleneck = DecomposedConceptBottleneck(
                hidden_size=backbone.hidden_size,
                num_concepts=num_concepts,
                unknown_ratio=unknown_ratio,
                unknown_rank=unknown_rank,
                concept_dropout=concept_dropout,
                residual_dropout=residual_dropout,
            )
        else:
            self.bottleneck = ConceptBottleneck(
                hidden_size=backbone.hidden_size,
                num_concepts=num_concepts,
                embedding_dim=embedding_dim,
                concept_dropout=concept_dropout,
                global_pairs=concept_global_pairs,
                unknown_dim=unknown_dim,
            )
        bottleneck_dim = self.bottleneck.output_dim
        self.lm_head = nn.Linear(bottleneck_dim, vocab_size)
        head_in = bottleneck_dim
        self.time_head: TimeToEventHead | None = (
            TimeToEventHead(head_in, time_bin_edges)
            if time_bin_edges is not None
            else None
        )
        self.event_heads: EventHazardHeads | None = (
            EventHazardHeads(
                head_in,
                event_names,
                time_bin_edges if time_bin_edges is not None else (),
                hidden_size=event_head_hidden,
            )
            if event_names
            else None
        )
        self.value_head: ValueQuantileHead | None = (
            ValueQuantileHead(
                head_in,
                backbone.hidden_size,
                DEFAULT_QUANTILE_LEVELS,
                hidden=value_head_hidden,
            )
            if value_head
            else None
        )

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
        intervention: BottleneckIntervention | None = None,
        teacher: TeacherForcing | None = None,
    ) -> tuple[torch.Tensor, ConceptBottleneckOutput, TimeAwareState]:
        """Return ``(next-token logits, bottleneck output, new backbone state)``.

        ``intervention`` performs a do()-style edit inside the bottleneck
        (see :class:`~odyssey.models.concept_bottleneck.BottleneckIntervention`):
        the task logits flow from the intervened mixture, while the
        concept/observability readouts stay the model's own.
        """
        hidden_states, new_state = self.backbone(
            batch, state=state, reset_mask=reset_mask
        )
        kwargs = {} if teacher is None else {"teacher": teacher}
        bottleneck_out = self.bottleneck(
            hidden_states, intervention=intervention, **kwargs
        )
        logits = self.lm_head(bottleneck_out.bottleneck)
        return logits, bottleneck_out, new_state

    def forward_with_features(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> ForwardWithFeatures:
        """Uniform forward (see :class:`ForwardWithFeatures`)."""
        logits, out, new_state = self(batch, state=state, reset_mask=reset_mask)
        return ForwardWithFeatures(logits, out.bottleneck, out, new_state)

    def compute_loss(
        self,
        batch: ClinicalSequenceBatch,
        concept_labels: torch.Tensor,
        concept_mask: torch.Tensor | None = None,
        loss_weights: ConceptBottleneckLossWeights | None = None,
        labels: torch.Tensor | None = None,
        *,
        teacher: TeacherForcing | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute next-token + concept + orthogonality loss.

        ``concept_labels``/``concept_mask`` are ``(batch, num_concepts)`` --
        subject-level, supervising the bottleneck at each sequence's last
        non-padding position (see :func:`_pool_last_non_padding`). Assumes
        one whole patient sequence per row; use :meth:`compute_streaming_loss`
        for packed, chunked batches.
        """
        logits, bottleneck_out, _ = self(batch, teacher=teacher)
        next_token_loss = self._whole_sequence_next_token_loss(logits, batch, labels)

        pooled_concept_logits = _pool_last_non_padding(
            bottleneck_out.concept_logits, batch.concept_ids, self.padding_idx
        )
        pooled_concept_embeddings = _pool_last_non_padding(
            bottleneck_out.concept_embeddings, batch.concept_ids, self.padding_idx
        )
        pooled_unknown_embedding = _pool_last_non_padding(
            bottleneck_out.unknown_embedding, batch.concept_ids, self.padding_idx
        )
        pooled_observability_logits = _pool_last_non_padding(
            bottleneck_out.observability_logits, batch.concept_ids, self.padding_idx
        )

        total, components = combined_loss(
            next_token_loss,
            pooled_concept_logits,
            concept_labels,
            pooled_concept_embeddings,
            pooled_unknown_embedding,
            observability_logits=pooled_observability_logits,
            concept_mask=concept_mask,
            weights=loss_weights,
        )
        # Per-position tensors, not the pooled ones: Steerling averages
        # the reconstruction and independence terms over token positions,
        # and pooling to one vector per subject would destroy the
        # cross-position covariance the independence loss measures.
        return fold_in_bottleneck_losses(
            self.bottleneck,
            bottleneck_out,
            concept_labels,
            total,
            components,
            concept_mask=concept_mask,
            weights=loss_weights,
        )

    def _streaming_position_labels(
        self,
        chunk: StreamingChunk,
        concept_labels: ConceptLabelDict,
        supervision: ConceptSupervision,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-position concept labels and a mask of where they were found.

        Labels arrive keyed by subject (or subject+visit) and pooled to
        supervision points, but teacher forcing and the reconstruction and
        independence losses all act per position, so each position needs
        its owning key's row. Positions with no label are returned as
        zeros AND marked False, so a caller can skip them rather than
        teach the head that an unlabeled position means "no concepts
        present".

        Returns ``None`` when nothing in the chunk is labeled.
        """
        # .tolist() ONCE, not element-wise iteration: chunk.subject_ids
        # lives on the GPU, and indexing a CUDA tensor per element forces a
        # device sync each time. At 64 lanes x 512 positions that was 32,768
        # syncs per training step and ran the whole run at 44% of the
        # mixture arm's throughput.
        subjects = chunk.subject_ids.reshape(-1).tolist()
        # cast, not dict(): the key type is a union here and rebuilding
        # the mapping just to satisfy it would copy it every chunk.
        lookup = cast("dict[object, torch.Tensor]", concept_labels)
        if supervision == "stay":
            keys: list[object] = subjects
        else:
            keys = list(
                zip(subjects, chunk.visit_ids.reshape(-1).tolist(), strict=True)
            )

        # One row per DISTINCT key, then gather. A chunk holds at most a few
        # dozen patients, so this stacks tens of rows rather than 32,768.
        order: dict[object, int] = {}
        index: list[int] = []
        for key in keys:
            slot = order.get(key)
            if slot is None:
                slot = len(order)
                order[key] = slot
            index.append(slot)
        distinct = [lookup.get(k) for k in order]
        present = next((r for r in distinct if r is not None), None)
        if present is None:
            return None

        device = chunk.batch.concept_ids.device
        zeros = torch.zeros_like(present)
        table = torch.stack([r if r is not None else zeros for r in distinct]).to(
            device
        )
        found = torch.tensor([r is not None for r in distinct], device=device)
        gather = torch.tensor(index, device=device)
        shape = (*chunk.subject_ids.shape, present.shape[-1])
        labels = table[gather].reshape(shape)
        mask = (
            found[gather].reshape(chunk.subject_ids.shape).unsqueeze(-1).expand(shape)
        )
        return labels, mask

    def compute_streaming_loss(
        self,
        chunk: StreamingChunk,
        concept_labels: ConceptLabelDict,
        concept_mask: ConceptLabelDict | None = None,
        *,
        state: TimeAwareState | None = None,
        loss_weights: ConceptBottleneckLossWeights | None = None,
        supervision: ConceptSupervision = "stay",
        intervention: BottleneckIntervention | None = None,
        objective: ForecastObjective | None = None,
        event_targets: Optional["EventHazardTargets"] = None,
        teacher_alpha_known: float = 0.0,
        teacher_alpha_unknown: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], TimeAwareState]:
        """Compute next-token + concept + orthogonality + observability loss.

        Two supervision modes, selecting both where the bottleneck is
        pooled and how ``concept_labels``/``concept_mask`` are keyed:

        - ``"stay"``: whole-stay labels keyed ``subject_id ->
          (num_concepts,)``, pooled once at each patient's true last event
          (``chunk.patient_end``) -- never merely their last position
          within a chunk (see the module docstring).
        - ``"visit"``: visit-scoped labels keyed ``(subject_id, visit_id)
          -> (num_concepts,)`` (see
          :func:`odyssey.data.concepts.label_concepts_by_visit`), pooled
          at every ``chunk.visit_end`` position -- each real visit's last
          event. Many grounded supervision points per subject instead of
          one, and a label whose memory demands match what a compressed
          recurrent state actually retains.

        If no supervision position falls within this chunk, the concept,
        orthogonality, and observability loss terms are zero and the
        total loss is the next-token loss alone.

        ``intervention`` is the intervention-aware training hook (CEM's
        RandInt, built per chunk by
        :func:`odyssey.training.running_labels.randint_intervention`):
        the task logits are computed from the intervened mixture, while
        the concept and observability losses supervise the model's own
        readouts exactly as without it. That is what teaches the task
        head that the concept probability is a trustworthy input, so
        that at deployment overriding a concept actually moves the
        forecast.
        """
        objective = objective or ForecastObjective()
        # Teacher forcing needs the labels laid out per position, which is
        # what the pooling step below builds; do that first when it is on.
        position_labels = self._streaming_position_labels(
            chunk, concept_labels, supervision
        )
        teacher = None
        if (
            position_labels is not None
            and max(teacher_alpha_known, teacher_alpha_unknown) > 0.0
        ):
            teacher = TeacherForcing(
                concept_labels=position_labels[0],
                alpha_known=teacher_alpha_known,
                alpha_unknown=teacher_alpha_unknown,
                # Unlabeled positions keep the model's own parts rather
                # than being forced to "no concepts" and u_hat_gt = h.
                concept_mask=position_labels[1],
            )
        logits, bottleneck_out, new_state = self(
            chunk.batch,
            state=state,
            reset_mask=chunk.reset_mask,
            intervention=intervention,
            teacher=teacher,
        )
        next_token_loss = self._streaming_task_loss(logits, chunk, objective)
        head_feats = bottleneck_out.bottleneck
        time_loss, _ = self._streaming_time_loss(self.time_head, head_feats, chunk)
        event_loss = self._streaming_event_loss(
            self.event_heads, head_feats, event_targets
        )
        value_loss, _ = self._streaming_value_loss(self.value_head, head_feats, chunk)
        forecast_loss = (
            next_token_loss
            + objective.time_weight * time_loss
            + objective.event_hazard_weight * event_loss
            + objective.value_head_weight * value_loss
        )

        pool_mask = chunk.patient_end if supervision == "stay" else chunk.visit_end
        if not pool_mask.any():
            zero = next_token_loss.new_zeros(())
            components = {
                "task_loss": next_token_loss.detach(),
                "time_loss": time_loss.detach(),
                "event_loss": event_loss.detach(),
                "value_loss": value_loss.detach(),
                "concept_loss": zero,
                "orthogonality_loss": zero,
                "observability_loss": zero,
            }
            return forecast_loss, components, new_state

        end_subject_ids = _pool_patient_ends(chunk.subject_ids, pool_mask)
        pooled_concept_logits = _pool_patient_ends(
            bottleneck_out.concept_logits, pool_mask
        )
        pooled_concept_embeddings = _pool_patient_ends(
            bottleneck_out.concept_embeddings, pool_mask
        )
        pooled_unknown_embedding = _pool_patient_ends(
            bottleneck_out.unknown_embedding, pool_mask
        )
        pooled_observability_logits = _pool_patient_ends(
            bottleneck_out.observability_logits, pool_mask
        )

        if supervision == "visit":
            end_visit_ids = _pool_patient_ends(chunk.visit_ids, pool_mask)
            labels_batch = _gather_by_visit(
                end_subject_ids,
                end_visit_ids,
                concept_labels,  # type: ignore[arg-type]
            )
            mask_batch = (
                _gather_by_visit(end_subject_ids, end_visit_ids, concept_mask)  # type: ignore[arg-type]
                if concept_mask is not None
                else None
            )
        else:
            labels_batch = _gather_by_subject(end_subject_ids, concept_labels)  # type: ignore[arg-type]
            mask_batch = (
                _gather_by_subject(end_subject_ids, concept_mask)  # type: ignore[arg-type]
                if concept_mask is not None
                else None
            )

        total, components = combined_loss(
            forecast_loss,
            pooled_concept_logits,
            labels_batch,
            pooled_concept_embeddings,
            pooled_unknown_embedding,
            observability_logits=pooled_observability_logits,
            concept_mask=mask_batch,
            weights=loss_weights,
        )
        # Per-position labels, not the pooled ones: the reconstruction
        # target is defined against each position's own hidden state, and
        # the independence loss measures covariance across positions.
        if position_labels is not None:
            total, components = fold_in_bottleneck_losses(
                self.bottleneck,
                bottleneck_out,
                position_labels[0],
                total,
                components,
                concept_mask=position_labels[1],
                weights=loss_weights,
            )
        # combined_loss reports the forecast term it was handed as
        # "task_loss"; split time back out so logs show both.
        components["task_loss"] = next_token_loss.detach()
        components["time_loss"] = time_loss.detach()
        components["event_loss"] = event_loss.detach()
        components["value_loss"] = value_loss.detach()
        return total, components, new_state
