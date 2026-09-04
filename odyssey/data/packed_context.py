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
caller that has to know where a truncated subject's visible history
starts, notably ``odyssey.inference.alerts``' landmark verifier (timestamps
themselves stay in the original frame; nothing is rebased).
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from odyssey.data.sequences import NO_VISIT, PatientSequence
from odyssey.data.streaming import NO_SUBJECT, StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.vocabulary import PAD_ID


_NAN = float("nan")


def _truncate_head(seq: PatientSequence, max_context: int) -> PatientSequence:
    """Keep ``seq``'s most recent ``max_context`` tokens, timestamps UNCHANGED.

    The kept window keeps its true "hours since this subject's first
    event" timestamps. The backbone never needs a 0-based window
    (:class:`~odyssey.models.backbones.transformer.TransformerBackbone`
    derives deltas from ``reset_mask`` and
    :class:`~odyssey.models.embeddings.TimeEmbeddingLayer` reads deltas),
    and every consumer of ``time_hours`` downstream (landmark buckets,
    visit starts, outcomes) works in the true frame. An earlier version
    rebased the window to 0 and had the alerts harness add the boundary
    back; the float64 round trip ``(t - b) + b`` differed from ``t`` by
    ~1e-13 h, enough to flip a landmark bucket exactly on a boundary
    (~22 rows per real MIMIC shard, research journal entry 44's CPU
    integration pass). See :meth:`PatientSequence.tail`.
    """
    return seq.tail(max_context)


@dataclass
class _RowBuilder:
    """One packed row's not-yet-padded tokens, built by appending whole patients."""

    concept_ids: list[int] = field(default_factory=list)
    type_ids: list[int] = field(default_factory=list)
    time_stamps: list[float] = field(default_factory=list)
    ages: list[float] = field(default_factory=list)
    visit_orders: list[int] = field(default_factory=list)
    visit_segments: list[int] = field(default_factory=list)
    reset: list[bool] = field(default_factory=list)
    subject_ids: list[int] = field(default_factory=list)
    patient_end: list[bool] = field(default_factory=list)
    visit_ids: list[int] = field(default_factory=list)
    visit_end: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    static: list[bool] = field(default_factory=list)
    score: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of tokens packed into this row so far."""
        return len(self.concept_ids)

    def append_patient(self, seq: PatientSequence, *, score_from: int = 0) -> None:
        """Append one (whole, truncated, or windowed) patient's tokens, as-is.

        ``score_from`` marks where scoring may start inside this segment:
        positions before it are context that an earlier overlapping
        window already scored (sliding-window mode); 0 scores everything.

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

        resets = [False] * n
        resets[0] = True
        self.reset.extend(resets)
        patient_end = [False] * n
        patient_end[-1] = True
        self.patient_end.extend(patient_end)
        self.score.extend([i >= score_from for i in range(n)])

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
            score=self.score + [False] * pad,
            n_real=n,
        )


@dataclass
class _Row:
    """A padded, fixed-length packed row."""

    concept_ids: list[int]
    type_ids: list[int]
    time_stamps: list[float]
    ages: list[float]
    visit_orders: list[int]
    visit_segments: list[int]
    reset: list[bool]
    subject_ids: list[int]
    patient_end: list[bool]
    visit_ids: list[int]
    visit_end: list[bool]
    values: list[float]
    static: list[bool]
    score: list[bool]
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
        window_stride: int | None = None,
    ) -> None:
        """Initialize the sampler over an exhaustible iterator of patients.

        ``window_stride`` switches a too-long patient from tail truncation
        to sliding windows: ``max_context``-token windows starting every
        ``window_stride`` tokens, the last one ending at the record's end,
        each scoring only the positions the previous window did not (the
        first window scores all of its positions, every later window its
        final positions). Every position of every patient is then scored
        exactly once, each with at least ``max_context - window_stride``
        tokens of context, and no window is anchored to the record's end
        for the positions it scores. Nothing is recorded as truncated.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if window_stride is not None and not 1 <= window_stride <= max_context:
            raise ValueError("window_stride must be in [1, max_context]")
        if max_context < 2:
            raise ValueError(
                "max_context must be >= 2 (need room for at least one "
                "input/target pair)"
            )
        self._patients = patients
        self.batch_size = batch_size
        self.max_context = max_context
        self.window_stride = window_stride
        self._held: tuple[PatientSequence, int] | None = None
        self._pending: list[tuple[PatientSequence, int]] = []
        self._exhausted = False
        self.truncated_subject_ids: list[int] = []
        self.truncation_boundaries: dict[int, float] = {}
        """subject_id -> the truncation boundary, in that subject's own
        original time frame ("hours since this sequence's first event",
        the same convention :func:`~odyssey.data.alert_events.origin_hours`
        uses) -- the time of the first token this sampler kept. Timestamps
        are NOT rebased (see :func:`_truncate_head`); the boundary is kept
        for the tail-slice bookkeeping and the landmark verifier, which
        needs to know which landmarks a truncated subject legitimately
        lost."""

    def _windows(self, patient: PatientSequence) -> list[tuple[PatientSequence, int]]:
        """Split a too-long patient into overlapping windows with score offsets."""
        assert self.window_stride is not None
        n, width, stride = len(patient), self.max_context, self.window_stride
        out: list[tuple[PatientSequence, int]] = []
        scored_to = 0
        start = 0
        while True:
            end = min(start + width, n)
            if end == n:
                start = max(0, n - width)
            out.append((patient.window(start, start + width), scored_to - start))
            scored_to = min(start + width, n)
            if scored_to >= n:
                return out
            start += stride

    def _next_patient(self) -> tuple[PatientSequence, int] | None:
        if self._held is not None:
            held = self._held
            self._held = None
            return held
        if self._pending:
            return self._pending.pop(0)
        if self._exhausted:
            return None
        try:
            patient: PatientSequence = next(self._patients)
        except StopIteration:
            self._exhausted = True
            return None
        if len(patient) > self.max_context:
            if self.window_stride is not None:
                self._pending = self._windows(patient)
                return self._pending.pop(0)
            self.truncated_subject_ids.append(patient.subject_id)
            self.truncation_boundaries[patient.subject_id] = patient.time_stamps[
                -self.max_context
            ]
            patient = _truncate_head(patient, self.max_context)
        return (patient, 0)

    def next_chunk(self) -> StreamingChunk | None:
        """Pack up to ``batch_size`` rows, each up to ``max_context`` tokens.

        Returns ``None`` once the patient queue is exhausted and no row
        got any real content -- the end of an epoch, matching
        :meth:`~odyssey.data.streaming.PackedLaneSampler.next_chunk`'s
        convention.
        """
        rows: list[_Row] = []
        any_real = False
        for _ in range(self.batch_size):
            row = _RowBuilder()
            while True:
                item = self._next_patient()
                if item is None:
                    break
                patient, score_from = item
                if len(row) == 0 or len(row) + len(patient) <= self.max_context:
                    row.append_patient(patient, score_from=score_from)
                else:
                    self._held = item
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
        # double, not float: _landmark_mask/_unrebase_truncated_times and
        # the final IndexRow.time_hours all read this same tensor back for
        # exact comparison against _index_rows_from_events' float64 polars
        # computation -- float32 loses enough precision at real-data time
        # magnitudes (hundreds of hours since origin) to occasionally flip
        # the 6th decimal digit verify_packed_landmark_rows compares on,
        # confirmed via a real-eICU-data repro (root-caused 2026-08-23).
        # The model's own TimeEmbeddingLayer.forward() already does its
        # own .float() cast before use, so this is purely additive
        # precision for the landmark-bookkeeping path, not a model-input
        # dtype change.
        time_stamps_full = _stack("time_stamps", torch.double)
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
        score_full = _stack("score", torch.bool)

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
            score_mask=score_full if self.window_stride is not None else None,
        )

    def __iter__(self) -> Iterator[StreamingChunk]:
        """Yield packed chunks until the patient queue is exhausted."""
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                return
            yield chunk
