"""Packed, whole-history batching for the stateless transformer backbone.

:class:`~odyssey.data.streaming.PackedLaneSampler` is built for a *stateful*
backbone: short chunks, persistent lanes, recurrent state carried across
calls. :class:`~odyssey.models.backbones.transformer.TransformerBackbone`
has no state to carry, so it needs the opposite regime: each patient's
*whole* history (or, if too long, its most recent ``max_context`` tokens)
inside one context window, processed in a single forward call. Packing
multiple patients into one row is a compute-efficiency requirement, not a
correctness one -- mean sequence length is ~301 events (README), so a
4096-token context window would be >90% padding waste for a typical
patient if each row held exactly one.

:class:`PackedContextSampler` packs patients back-to-back into rows up to
``max_context`` tokens, greedily (first-fit): a row keeps taking patients
until the next one would not fit, then starts a new row with that patient.
A patient is never split across two rows -- if it alone exceeds
``max_context``, it is truncated from the left (most recent tokens kept)
before packing, so every row's patients are always complete, truncated or
not. This makes the coverage guarantee simple: every (possibly truncated)
patient's every token appears as an input exactly once, in exactly one
row, over one pass through the patient iterator.

Cross-patient leakage: a packed patient's own hidden states must be
identical to processing that patient alone, both at the attention level
(no position may attend past its segment's start -- the block-diagonal
mask :func:`~odyssey.models.backbones.transformer._build_attn_mask` builds
from ``reset_mask``) and at the embeddings level
(:class:`~odyssey.models.embeddings.TimeEmbeddingLayer` computes
time-since-previous-event as a delta over the whole row's raw timestamps,
which would otherwise smear a segment boundary's delta across two
different patients' clocks). ``TransformerBackbone`` owns both
guarantees itself, from ``reset_mask`` alone
(:func:`~odyssey.models.backbones.transformer._rebase_time_stamps`) --
this sampler does not need to, and deliberately does not, pre-adjust
timestamps before packing: patients are concatenated with their own raw
values as :class:`~odyssey.data.sequences.PatientSequence` already
produced them, no seam correction here, so the no-leakage guarantee holds
for any caller of the backbone, not only this one sampler's output. See
``tests/odyssey/models/backbones/test_transformer.py`` for the
load-bearing test this has to survive: change a packed patient's
neighbor, assert their own logits are bit-identical.

Tail patients (truncated because they alone exceed ``max_context``):
``truncated_subject_ids`` accumulates every truncated subject id across
this sampler's lifetime (not reset per call) -- an eval harness reads it
once a pass completes and reports those subjects' metrics as a distinct
slice rather than pooling them into the headline number, since losing
distant history is itself part of what this backbone control measures,
not something to average away. ``truncation_boundaries`` accumulates
alongside it: each truncated subject's kept-window start, in that
subject's own original time frame (see
:attr:`PackedContextSampler.truncation_boundaries`) -- needed by any
caller that has to reconcile the packed path's rebased-to-0 timestamps
against something computed in the original frame, notably
``odyssey.inference.alerts``' landmark/outcome logic.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch

from odyssey.data.sequences import N_RECENCY_FAMILIES, NO_VISIT, PatientSequence
from odyssey.data.streaming import NO_SUBJECT, StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.vocabulary import PAD_ID


_NAN = float("nan")


def _truncate_head(seq: PatientSequence, max_context: int) -> PatientSequence:
    """Keep ``seq``'s most recent ``max_context`` tokens, timestamps rebased to 0.

    Rebasing (subtracting the kept window's own first timestamp) restores
    the "hours since this sequence's first event" convention
    :class:`PatientSequence` documents for its own ``time_stamps`` field --
    this patient is, from here on, effectively a fresh sequence starting
    at the truncation point. ``family_recency`` needs no equivalent
    adjustment: it already stores differences (hours since each family's
    last occurrence), invariant under adding the same constant to every
    timestamp.
    """
    n = len(seq)
    if n <= max_context:
        return seq

    def _tail(values: List[object]) -> List[object]:
        return values[-max_context:] if len(values) == n else []

    kept_times = seq.time_stamps[-max_context:]
    first = kept_times[0]
    rebased_times = [t - first for t in kept_times]

    return PatientSequence(
        subject_id=seq.subject_id,
        concept_ids=seq.concept_ids[-max_context:],
        type_ids=seq.type_ids[-max_context:],
        time_stamps=rebased_times,
        ages=seq.ages[-max_context:],
        visit_orders=seq.visit_orders[-max_context:],
        visit_segments=seq.visit_segments[-max_context:],
        visit_ids=_tail(list(seq.visit_ids)),  # type: ignore[arg-type]
        visit_ends=_tail(list(seq.visit_ends)),  # type: ignore[arg-type]
        static_mask=_tail(list(seq.static_mask)),  # type: ignore[arg-type]
        values=_tail(list(seq.values)),  # type: ignore[arg-type]
        family_recency=_tail(list(seq.family_recency)),  # type: ignore[arg-type]
    )


@dataclass
class _RowBuilder:
    """One packed row's not-yet-padded tokens, built by appending whole patients."""

    concept_ids: List[int] = field(default_factory=list)
    type_ids: List[int] = field(default_factory=list)
    time_stamps: List[float] = field(default_factory=list)
    ages: List[float] = field(default_factory=list)
    visit_orders: List[int] = field(default_factory=list)
    visit_segments: List[int] = field(default_factory=list)
    reset: List[bool] = field(default_factory=list)
    subject_ids: List[int] = field(default_factory=list)
    patient_end: List[bool] = field(default_factory=list)
    visit_ids: List[int] = field(default_factory=list)
    visit_end: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    static: List[bool] = field(default_factory=list)
    family_recency: List[List[float]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of tokens packed into this row so far."""
        return len(self.concept_ids)

    def append_patient(self, seq: PatientSequence) -> None:
        """Append one (whole or already-truncated) patient's tokens, as-is.

        No timestamp adjustment: ``time_stamps`` are each patient's own
        raw values, unchanged. The module docstring explains why that is
        safe (``TransformerBackbone`` corrects the segment-boundary delta
        itself, from ``reset``, below).
        """
        n = len(seq)
        if n == 0:
            return

        self.concept_ids.extend(seq.concept_ids)
        self.type_ids.extend(seq.type_ids)
        self.time_stamps.extend(seq.time_stamps)
        self.ages.extend(seq.ages)
        self.visit_orders.extend(seq.visit_orders)
        self.visit_segments.extend(seq.visit_segments)
        self.subject_ids.extend([seq.subject_id] * n)
        self.visit_ids.extend(
            seq.visit_ids if len(seq.visit_ids) == n else [NO_VISIT] * n
        )
        self.visit_end.extend(
            seq.visit_ends if len(seq.visit_ends) == n else [False] * n
        )
        self.values.extend(seq.values if len(seq.values) == n else [_NAN] * n)
        self.static.extend(
            seq.static_mask if len(seq.static_mask) == n else [False] * n
        )
        self.family_recency.extend(
            seq.family_recency
            if len(seq.family_recency) == n
            else [[_NAN] * N_RECENCY_FAMILIES] * n
        )

        resets = [False] * n
        resets[0] = True
        self.reset.extend(resets)
        patient_end = [False] * n
        patient_end[-1] = True
        self.patient_end.extend(patient_end)

    def pad_to(self, capacity: int) -> "_Row":
        """Return this row's content, right-padded to exactly ``capacity`` tokens."""
        n = len(self)
        pad = capacity - n
        if pad < 0:
            raise ValueError(f"row has {n} tokens, exceeds capacity {capacity}")
        return _Row(
            concept_ids=self.concept_ids + [PAD_ID] * pad,
            type_ids=self.type_ids + [0] * pad,
            time_stamps=self.time_stamps + [0.0] * pad,
            ages=self.ages + [0.0] * pad,
            visit_orders=self.visit_orders + [0] * pad,
            visit_segments=self.visit_segments + [0] * pad,
            reset=self.reset + [False] * pad,
            subject_ids=self.subject_ids + [NO_SUBJECT] * pad,
            patient_end=self.patient_end + [False] * pad,
            visit_ids=self.visit_ids + [NO_VISIT] * pad,
            visit_end=self.visit_end + [False] * pad,
            values=self.values + [_NAN] * pad,
            static=self.static + [False] * pad,
            family_recency=self.family_recency + [[_NAN] * N_RECENCY_FAMILIES] * pad,
            n_real=n,
        )


@dataclass
class _Row:
    """A padded, fixed-length packed row."""

    concept_ids: List[int]
    type_ids: List[int]
    time_stamps: List[float]
    ages: List[float]
    visit_orders: List[int]
    visit_segments: List[int]
    reset: List[bool]
    subject_ids: List[int]
    patient_end: List[bool]
    visit_ids: List[int]
    visit_end: List[bool]
    values: List[float]
    static: List[bool]
    family_recency: List[List[float]]
    n_real: int


class PackedContextSampler:
    """Greedy first-fit packer of whole patients into ``max_context``-token rows.

    ``patients`` should already be shuffled by the caller, matching
    :class:`~odyssey.data.streaming.PackedLaneSampler`'s convention -- this
    sampler does not shuffle. Unlike that sampler, there is no persistent
    per-lane state and no ``reset_prob`` (no synthetic mid-visit history
    resets): a row's patients are always complete, real segments, since
    this backbone is stateless and has no notion of "the rest of this
    patient's history was already seen in an earlier chunk" to simulate
    losing.
    """

    def __init__(
        self,
        patients: Iterator[PatientSequence],
        *,
        batch_size: int,
        max_context: int,
    ) -> None:
        """Initialize the sampler over an exhaustible iterator of patients."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_context < 2:
            raise ValueError(
                "max_context must be >= 2 (need room for at least one "
                "input/target pair)"
            )
        self._patients = patients
        self.batch_size = batch_size
        self.max_context = max_context
        self._held: Optional[PatientSequence] = None
        self._exhausted = False
        self.truncated_subject_ids: List[int] = []
        self.truncation_boundaries: Dict[int, float] = {}
        """subject_id -> the truncation boundary, in that subject's own
        original time frame ("hours since this sequence's first event",
        the same convention :func:`~odyssey.data.alert_events.origin_hours`
        uses) -- the raw time of the first token this sampler kept, before
        :func:`_truncate_head` rebases the kept window to start at 0. A
        caller needing to compare this subject's packed-path timestamps
        against anything computed in the original frame (e.g.
        ``odyssey.inference.alerts``' landmark/outcome logic) adds this
        back; nothing in this module needs to un-rebase, since the
        backbone only ever needs relative deltas within one row."""

    def _next_patient(self) -> Optional[PatientSequence]:
        if self._held is not None:
            patient = self._held
            self._held = None
            return patient
        if self._exhausted:
            return None
        try:
            patient = next(self._patients)
        except StopIteration:
            self._exhausted = True
            return None
        if len(patient) > self.max_context:
            self.truncated_subject_ids.append(patient.subject_id)
            self.truncation_boundaries[patient.subject_id] = patient.time_stamps[
                -self.max_context
            ]
            patient = _truncate_head(patient, self.max_context)
        return patient

    def next_chunk(self) -> Optional[StreamingChunk]:
        """Pack up to ``batch_size`` rows, each up to ``max_context`` tokens.

        Returns ``None`` once the patient queue is exhausted and no row
        got any real content -- the end of an epoch, matching
        :meth:`~odyssey.data.streaming.PackedLaneSampler.next_chunk`'s
        convention.
        """
        rows: List[_Row] = []
        any_real = False
        for _ in range(self.batch_size):
            row = _RowBuilder()
            while True:
                patient = self._next_patient()
                if patient is None:
                    break
                if len(row) == 0 or len(row) + len(patient) <= self.max_context:
                    row.append_patient(patient)
                else:
                    self._held = patient
                    break
            padded = row.pad_to(self.max_context)
            rows.append(padded)
            if padded.n_real > 0:
                any_real = True

        if not any_real:
            return None

        def _stack(field_name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor([getattr(r, field_name) for r in rows], dtype=dtype)

        concept_ids_full = _stack("concept_ids", torch.long)
        type_ids_full = _stack("type_ids", torch.long)
        time_stamps_full = _stack("time_stamps", torch.float)
        ages_full = _stack("ages", torch.float)
        visit_orders_full = _stack("visit_orders", torch.long)
        visit_segments_full = _stack("visit_segments", torch.long)
        reset_full = _stack("reset", torch.bool)
        subject_ids_full = _stack("subject_ids", torch.long)
        patient_end_full = _stack("patient_end", torch.bool)
        visit_ids_full = _stack("visit_ids", torch.long)
        visit_end_full = _stack("visit_end", torch.bool)
        values_full = _stack("values", torch.float)
        static_full = _stack("static", torch.bool)
        family_recency_full = _stack("family_recency", torch.float)

        # Same-window shift by one (not the +1-lookahead-token convention
        # PackedLaneSampler uses): max_context means exactly the window
        # size, so the window's own last position never has a real target.
        targets = torch.full_like(concept_ids_full, PAD_ID)
        targets[:, :-1] = concept_ids_full[:, 1:]
        not_target = patient_end_full.clone()
        not_target[:, :-1] |= static_full[:, 1:]
        not_target[:, -1] = True
        targets = targets.masked_fill(not_target, PAD_ID)
        real_mask = (subject_ids_full != NO_SUBJECT) & ~not_target

        batch = ClinicalSequenceBatch(
            concept_ids=concept_ids_full,
            aux=AuxiliaryInputs(
                type_ids=type_ids_full,
                time_stamps=time_stamps_full,
                ages=ages_full,
                visit_orders=visit_orders_full,
                visit_segments=visit_segments_full,
                values=values_full,
                family_recency=family_recency_full,
            ),
        )
        return StreamingChunk(
            batch=batch,
            targets=targets,
            reset_mask=reset_full,
            real_mask=real_mask,
            subject_ids=subject_ids_full,
            patient_end=patient_end_full,
            visit_ids=visit_ids_full,
            visit_end=visit_end_full,
        )

    def __iter__(self) -> Iterator[StreamingChunk]:
        """Yield packed chunks until the patient queue is exhausted."""
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                return
            yield chunk
