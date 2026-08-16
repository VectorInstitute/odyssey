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

from typing import Dict, Literal, Optional, Tuple, Union

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
    subject_ids: torch.Tensor, labels: Dict[int, torch.Tensor]
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
    labels: Dict[Tuple[int, int], torch.Tensor],
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
ConceptLabelDict = Union[Dict[int, torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]


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
        labels: Optional[torch.Tensor],
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


class BaselineSequenceModel(_SequenceModelBase):
    """Next-token prediction directly from the backbone: no concept bottleneck."""

    def __init__(
        self,
        backbone: SequenceBackbone,
        vocab_size: int,
        *,
        padding_idx: int = 0,
    ) -> None:
        """Initialize the baseline sequence model."""
        super().__init__(backbone, padding_idx=padding_idx)
        self.lm_head = nn.Linear(backbone.hidden_size, vocab_size)

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: Optional[TimeAwareState] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TimeAwareState]:
        """Return ``(next-token logits, new backbone state)``."""
        hidden_states, new_state = self.backbone(
            batch, state=state, reset_mask=reset_mask
        )
        logits: torch.Tensor = self.lm_head(hidden_states)
        return logits, new_state

    def compute_loss(
        self, batch: ClinicalSequenceBatch, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute next-token loss over one full sequence per row."""
        logits, _ = self(batch)
        loss = self._whole_sequence_next_token_loss(logits, batch, labels)
        return loss, {"task_loss": loss.detach()}

    def compute_streaming_loss(
        self,
        chunk: StreamingChunk,
        state: Optional[TimeAwareState] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], TimeAwareState]:
        """Compute next-token loss over one packed, chunked training step."""
        logits, new_state = self(chunk.batch, state=state, reset_mask=chunk.reset_mask)
        loss = self._streaming_next_token_loss(logits, chunk.targets)
        return loss, {"task_loss": loss.detach()}, new_state


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
    ) -> None:
        """Initialize the concept-bottleneck sequence model."""
        super().__init__(backbone, padding_idx=padding_idx)
        self.bottleneck = ConceptBottleneck(
            hidden_size=backbone.hidden_size,
            num_concepts=num_concepts,
            embedding_dim=embedding_dim,
            concept_dropout=concept_dropout,
        )
        self.lm_head = nn.Linear((num_concepts + 1) * embedding_dim, vocab_size)

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: Optional[TimeAwareState] = None,
        reset_mask: Optional[torch.Tensor] = None,
        intervention: Optional[BottleneckIntervention] = None,
    ) -> Tuple[torch.Tensor, ConceptBottleneckOutput, TimeAwareState]:
        """Return ``(next-token logits, bottleneck output, new backbone state)``.

        ``intervention`` performs a do()-style edit inside the bottleneck
        (see :class:`~odyssey.models.concept_bottleneck.BottleneckIntervention`):
        the task logits flow from the intervened mixture, while the
        concept/observability readouts stay the model's own.
        """
        hidden_states, new_state = self.backbone(
            batch, state=state, reset_mask=reset_mask
        )
        bottleneck_out = self.bottleneck(hidden_states, intervention=intervention)
        logits = self.lm_head(bottleneck_out.bottleneck)
        return logits, bottleneck_out, new_state

    def compute_loss(
        self,
        batch: ClinicalSequenceBatch,
        concept_labels: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[ConceptBottleneckLossWeights] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute next-token + concept + orthogonality loss.

        ``concept_labels``/``concept_mask`` are ``(batch, num_concepts)`` --
        subject-level, supervising the bottleneck at each sequence's last
        non-padding position (see :func:`_pool_last_non_padding`). Assumes
        one whole patient sequence per row; use :meth:`compute_streaming_loss`
        for packed, chunked batches.
        """
        logits, bottleneck_out, _ = self(batch)
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

        return combined_loss(
            next_token_loss,
            pooled_concept_logits,
            concept_labels,
            pooled_concept_embeddings,
            pooled_unknown_embedding,
            observability_logits=pooled_observability_logits,
            concept_mask=concept_mask,
            weights=loss_weights,
        )

    def compute_streaming_loss(
        self,
        chunk: StreamingChunk,
        concept_labels: ConceptLabelDict,
        concept_mask: Optional[ConceptLabelDict] = None,
        *,
        state: Optional[TimeAwareState] = None,
        loss_weights: Optional[ConceptBottleneckLossWeights] = None,
        supervision: ConceptSupervision = "stay",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], TimeAwareState]:
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
        """
        logits, bottleneck_out, new_state = self(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        next_token_loss = self._streaming_next_token_loss(logits, chunk.targets)

        pool_mask = chunk.patient_end if supervision == "stay" else chunk.visit_end
        if not pool_mask.any():
            zero = next_token_loss.new_zeros(())
            components = {
                "task_loss": next_token_loss.detach(),
                "concept_loss": zero,
                "orthogonality_loss": zero,
                "observability_loss": zero,
            }
            return next_token_loss, components, new_state

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
            next_token_loss,
            pooled_concept_logits,
            labels_batch,
            pooled_concept_embeddings,
            pooled_unknown_embedding,
            observability_logits=pooled_observability_logits,
            concept_mask=mask_batch,
            weights=loss_weights,
        )
        return total, components, new_state
