"""Streaming, packed batch sampler for training on full patient history.

Implements the training pipeline design in
``research_journal/02_sequence_scoping_methodology.html`` Section 05:
persistent per-lane streams, sequential non-overlapping chunk advancement,
patients packed back-to-back within a lane, and a per-position reset mask
marking every place the backbone's carried-forward recurrent state must be
zeroed -- either a real patient boundary or a synthetic mid-patient reset
that simulates missing history.

A lane is a continuous concatenation of :class:`~odyssey.data.sequences.PatientSequence`
objects pulled from a queue. :meth:`PackedLaneSampler.next_chunk` advances
every lane by up to ``chunk_size`` tokens per call and returns a
:class:`StreamingChunk`:

Backbones that carry recurrent state across chunks (e.g.
:class:`~odyssey.models.backbones.hybrid.EHRHybridBackbone`) only support a
state reset at position 0 of a chunk -- a mid-chunk reset would require
per-row varlen kernel support that hasn't been validated (see that
backbone's module docstring). Whenever a lane's next reset (patient
boundary or synthetic mid-visit reset) would otherwise land after
position 0, :meth:`next_chunk` stops that lane's window there and pads
the remainder instead of pulling in the next segment early; the
untaken tokens stay in the buffer and become position 0 of that lane's
next window, so every event still appears as a real *input* token
exactly once -- only some compute in that one chunk is spent on padding
instead. The one deliberate exception is on the *target* side: the
(input, target) pair that would predict a segment's first token from
the previous segment's last hidden state -- a prediction across the
very reset boundary that says no state carries over -- is never a
target in any chunk, consistent with the model having no information
to predict it from. This is a correctness improvement over naive
packing, not data loss: that pair was never a meaningful forecasting
target to begin with.

- ``batch``: a :class:`~odyssey.data.types.ClinicalSequenceBatch` of
  ``chunk_size`` input tokens per lane.
- ``targets``: the next-token prediction target for each input position
  (``batch.concept_ids`` shifted by one). Consecutive chunks in the same
  lane overlap by exactly one token, the seam every next-token training
  loop needs -- not the near-duplicate overlap decision (i) in the report
  rules out, since each lane still advances by a full ``chunk_size`` and
  every event is a prediction target exactly once per epoch.
- ``reset_mask``: ``True`` at a position means the backbone must not carry
  state from the previous position into this one.
- ``real_mask``: whether ``targets`` at that position is an actual event
  and not padding. ``False`` for the trailing token(s) of a lane once its
  buffer runs dry (the input side can still hold one real, final event
  with nothing after it to predict) and for every position once the
  patient queue and every lane's buffer are fully exhausted at epoch end.
- ``subject_ids``: which patient each input position belongs to (``-1`` for
  padding), so downstream code can pool the concept bottleneck per patient
  segment instead of once per lane (a chunk can contain fragments of
  several different patients).
- ``patient_end``: ``True`` at a position that is the last token of its
  patient's *entire* sequence, not merely the last token of that patient
  seen so far in this lane. Concept labels (``odyssey.data.concepts``) are
  derived over a patient's whole stay, so pooling for supervision is only
  valid here -- a patient whose history spans multiple chunks must not be
  supervised at a chunk boundary that isn't their actual last event, since
  events later in their stay could still change the true label.
"""

from dataclasses import dataclass, field
from typing import Iterator, List, NamedTuple, Optional

import torch

from odyssey.data.sequences import PatientSequence
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.vocabulary import PAD_ID


# Sentinel `subject_ids` value at padding positions.
NO_SUBJECT = -1


class StreamingChunk(NamedTuple):
    """One packed, chunked training step across every lane."""

    batch: ClinicalSequenceBatch
    targets: torch.Tensor
    reset_mask: torch.Tensor
    real_mask: torch.Tensor
    subject_ids: torch.Tensor
    patient_end: torch.Tensor


@dataclass
class _LaneBuffer:
    """One lane's not-yet-consumed tokens, with a per-token reset flag."""

    concept_ids: List[int] = field(default_factory=list)
    type_ids: List[int] = field(default_factory=list)
    time_stamps: List[float] = field(default_factory=list)
    ages: List[float] = field(default_factory=list)
    visit_orders: List[int] = field(default_factory=list)
    visit_segments: List[int] = field(default_factory=list)
    reset: List[bool] = field(default_factory=list)
    subject_ids: List[int] = field(default_factory=list)
    patient_end: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of unconsumed tokens in this lane."""
        return len(self.concept_ids)

    def append_patient(
        self, seq: PatientSequence, *, reset_prob: float, rng: torch.Generator
    ) -> None:
        """Append one patient's tokens, with reset flags at the boundaries.

        Position 0 of ``seq`` always resets (a real patient boundary).
        Positions at a new visit (``visit_orders`` changes from the
        previous token) reset with probability ``reset_prob``, simulating a
        patient whose prior history is unavailable at that admission. The
        last token of ``seq`` is marked ``patient_end``.
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

        resets = [False] * n
        resets[0] = True
        if reset_prob > 0.0:
            for t in range(1, n):
                if seq.visit_orders[t] == seq.visit_orders[t - 1]:
                    continue
                if torch.rand(1, generator=rng).item() < reset_prob:
                    resets[t] = True
        self.reset.extend(resets)

        patient_end = [False] * n
        patient_end[-1] = True
        self.patient_end.extend(patient_end)

    def pop_front(self, n: int) -> None:
        """Drop the first ``n`` tokens (already consumed by a chunk)."""
        if n <= 0:
            return
        del self.concept_ids[:n]
        del self.type_ids[:n]
        del self.time_stamps[:n]
        del self.ages[:n]
        del self.visit_orders[:n]
        del self.visit_segments[:n]
        del self.reset[:n]
        del self.subject_ids[:n]
        del self.patient_end[:n]

    def peek_padded(self, k: int) -> "_Window":
        """Return the first ``k`` tokens, padded if fewer than ``k`` remain."""
        have = len(self)
        n_real = min(k, have)
        pad = k - n_real
        return _Window(
            concept_ids=self.concept_ids[:n_real] + [PAD_ID] * pad,
            type_ids=self.type_ids[:n_real] + [0] * pad,
            time_stamps=self.time_stamps[:n_real] + [0.0] * pad,
            ages=self.ages[:n_real] + [0.0] * pad,
            visit_orders=self.visit_orders[:n_real] + [0] * pad,
            visit_segments=self.visit_segments[:n_real] + [0] * pad,
            reset=self.reset[:n_real] + [False] * pad,
            subject_ids=self.subject_ids[:n_real] + [NO_SUBJECT] * pad,
            patient_end=self.patient_end[:n_real] + [False] * pad,
            n_real=n_real,
        )


def _repad(window: "_Window", real_len: int) -> "_Window":
    """Return ``window`` with only its first ``real_len`` positions real.

    Used to truncate a window early (at a mid-chunk reset position) while
    keeping its fixed length -- the same padding convention
    :meth:`_LaneBuffer.peek_padded` uses for end-of-buffer padding.
    """
    k = len(window.concept_ids)
    pad = k - real_len
    return _Window(
        concept_ids=window.concept_ids[:real_len] + [PAD_ID] * pad,
        type_ids=window.type_ids[:real_len] + [0] * pad,
        time_stamps=window.time_stamps[:real_len] + [0.0] * pad,
        ages=window.ages[:real_len] + [0.0] * pad,
        visit_orders=window.visit_orders[:real_len] + [0] * pad,
        visit_segments=window.visit_segments[:real_len] + [0] * pad,
        reset=window.reset[:real_len] + [False] * pad,
        subject_ids=window.subject_ids[:real_len] + [NO_SUBJECT] * pad,
        patient_end=window.patient_end[:real_len] + [False] * pad,
        n_real=real_len,
    )


@dataclass
class _Window:
    """A padded, fixed-length peek into a lane's buffer."""

    concept_ids: List[int]
    type_ids: List[int]
    time_stamps: List[float]
    ages: List[float]
    visit_orders: List[int]
    visit_segments: List[int]
    reset: List[bool]
    subject_ids: List[int]
    patient_end: List[bool]
    n_real: int


class PackedLaneSampler:
    """Persistent-lane, packed, chunked sampler over a stream of patients.

    ``patients`` should already be shuffled by the caller (subject order
    matters for what ends up packed together and where synthetic resets
    fall, but the sampler itself does not shuffle).
    """

    def __init__(
        self,
        patients: Iterator[PatientSequence],
        *,
        num_lanes: int,
        chunk_size: int,
        reset_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        """Initialize the sampler over an exhaustible iterator of patients."""
        if num_lanes < 1:
            raise ValueError("num_lanes must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self._patients = patients
        self.num_lanes = num_lanes
        self.chunk_size = chunk_size
        self.reset_prob = reset_prob
        self._rng = torch.Generator().manual_seed(seed)
        self._lanes = [_LaneBuffer() for _ in range(num_lanes)]
        self._queue_exhausted = False

    def _refill(self, lane: _LaneBuffer, min_len: int) -> None:
        while len(lane) < min_len and not self._queue_exhausted:
            try:
                seq = next(self._patients)
            except StopIteration:
                self._queue_exhausted = True
                return
            lane.append_patient(seq, reset_prob=self.reset_prob, rng=self._rng)

    def next_chunk(self) -> Optional[StreamingChunk]:
        """Advance every lane by up to ``chunk_size`` tokens.

        A lane advances by fewer than ``chunk_size`` tokens (the rest of
        its window padded) when a reset would otherwise land after
        position 0 -- see the module docstring.

        Returns ``None`` once the patient queue is exhausted and every
        lane's buffer is empty, marking the end of an epoch.
        """
        k = self.chunk_size + 1
        windows = []
        any_real = False
        for lane in self._lanes:
            self._refill(lane, k)
            window = lane.peek_padded(k)
            # reset[0] is always allowed (it becomes this window's own
            # position-0 reset); a reset at any input position after that
            # (indices 1..chunk_size-1) would ask the backbone to reset
            # mid-chunk, which it does not support -- stop the real
            # content there instead and leave the rest for next call.
            truncate_at = next(
                (idx for idx in range(1, self.chunk_size) if window.reset[idx]), None
            )
            if truncate_at is not None:
                window = _repad(window, truncate_at)
                lane.pop_front(truncate_at)
            else:
                lane.pop_front(min(self.chunk_size, len(lane)))
            windows.append(window)
            if window.n_real > 0:
                any_real = True

        if not any_real:
            return None

        def _stack(field_name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor([getattr(w, field_name) for w in windows], dtype=dtype)

        concept_ids_full = _stack("concept_ids", torch.long)
        type_ids_full = _stack("type_ids", torch.long)
        time_stamps_full = _stack("time_stamps", torch.float)
        ages_full = _stack("ages", torch.float)
        visit_orders_full = _stack("visit_orders", torch.long)
        visit_segments_full = _stack("visit_segments", torch.long)
        reset_full = _stack("reset", torch.bool)
        subject_ids_full = _stack("subject_ids", torch.long)
        patient_end_full = _stack("patient_end", torch.bool)

        input_ids = concept_ids_full[:, :-1]
        targets = concept_ids_full[:, 1:]
        # Real means "this position has a valid next-token target", not
        # merely "the input token itself is real" -- the last genuine token
        # in an exhausted lane is real input but has no real target.
        real_mask = subject_ids_full[:, 1:] != NO_SUBJECT

        batch = ClinicalSequenceBatch(
            concept_ids=input_ids,
            aux=AuxiliaryInputs(
                type_ids=type_ids_full[:, :-1],
                time_stamps=time_stamps_full[:, :-1],
                ages=ages_full[:, :-1],
                visit_orders=visit_orders_full[:, :-1],
                visit_segments=visit_segments_full[:, :-1],
            ),
        )
        return StreamingChunk(
            batch=batch,
            targets=targets,
            reset_mask=reset_full[:, :-1],
            real_mask=real_mask,
            subject_ids=subject_ids_full[:, :-1],
            patient_end=patient_end_full[:, :-1],
        )

    def __iter__(self) -> Iterator[StreamingChunk]:
        """Yield chunks until the epoch ends."""
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                return
            yield chunk
