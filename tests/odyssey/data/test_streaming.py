"""Tests for the packed, chunked, persistent-lane training sampler."""

from typing import Iterator, List, Optional

import torch

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler
from odyssey.data.vocabulary import PAD_ID


def _seq(
    subject_id: int, n: int, *, visit_orders: Optional[List[int]] = None
) -> PatientSequence:
    """Build a patient with ``n`` events, token ids ``subject_id * 1000 + i``."""
    if visit_orders is None:
        visit_orders = [0] * n
    return PatientSequence(
        subject_id=subject_id,
        concept_ids=[subject_id * 1000 + i for i in range(n)],
        type_ids=[1] * n,
        time_stamps=[float(i) for i in range(n)],
        ages=[30.0] * n,
        visit_orders=visit_orders,
        visit_segments=[0] * n,
    )


def _patients(seqs: List[PatientSequence]) -> Iterator[PatientSequence]:
    return iter(seqs)


# ---------------------------------------------------------------------------
# Basic chunk advancement
# ---------------------------------------------------------------------------


def test_single_lane_advances_sequentially_without_overlap() -> None:
    patients = _patients([_seq(1, 10)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    c2 = sampler.next_chunk()
    c3 = sampler.next_chunk()

    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, 1002]
    assert c1.targets[0].tolist() == [1001, 1002, 1003]
    # c2 starts exactly where c1's targets left off: no re-visited spans.
    assert c2.batch.concept_ids[0].tolist() == [1003, 1004, 1005]
    assert c2.targets[0].tolist() == [1004, 1005, 1006]
    assert c3.batch.concept_ids[0].tolist() == [1006, 1007, 1008]


def test_every_event_is_a_target_exactly_once() -> None:
    patients = _patients([_seq(1, 12)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    seen_targets: List[int] = []
    for chunk in sampler:
        seen_targets.extend(
            t
            for t, real in zip(chunk.targets[0].tolist(), chunk.real_mask[0].tolist())
            if real
        )
    # events 1001..1011 (event 1000 is never a target -- it's the first input token)
    assert seen_targets == list(range(1001, 1012))
    assert len(seen_targets) == len(set(seen_targets))


# ---------------------------------------------------------------------------
# Packing multiple patients into one lane
# ---------------------------------------------------------------------------
#
# Patients are still packed back-to-back across the *stream* -- every event
# is eventually consumed in order, nothing is dropped -- but never within
# the same *chunk*: a chunk's real content stops at the first reset after
# position 0, and the untaken tokens (including the next patient's own
# position-0 reset) carry over to start the following chunk instead. See
# the module docstring for why (the backbone can't resume mid-chunk state
# for more than one segment).


def test_a_second_patient_never_starts_within_the_same_chunk() -> None:
    patients = _patients([_seq(1, 2), _seq(2, 2), _seq(3, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    # patient 1's 2 events, then padding -- patient 2 is deferred to c2
    # rather than starting mid-chunk.
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, PAD_ID]
    assert c1.subject_ids[0].tolist() == [1, 1, NO_SUBJECT]

    c2 = sampler.next_chunk()
    # patient 2 now starts fresh at position 0 of its own chunk.
    assert c2.batch.concept_ids[0].tolist() == [2000, 2001, PAD_ID]
    assert c2.subject_ids[0].tolist() == [2, 2, NO_SUBJECT]
    assert c2.reset_mask[0].tolist() == [True, False, False]


def test_patient_boundary_within_a_chunk_truncates_it() -> None:
    patients = _patients([_seq(1, 2), _seq(2, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    # position 0: first token of patient 1 -> reset (allowed). Patient 2's
    # own reset would otherwise land at position 2 -- not allowed mid-chunk,
    # so the chunk is truncated there instead, and that position is padding.
    assert c1.reset_mask[0].tolist() == [True, False, False]
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, PAD_ID]


# ---------------------------------------------------------------------------
# patient_end: where concept-label supervision is valid
# ---------------------------------------------------------------------------


def test_patient_end_marks_a_patients_true_last_token() -> None:
    # patient 1 (2 events) fully fits; patient 2 is deferred to the next
    # chunk entirely rather than starting mid-chunk here.
    patients = _patients([_seq(1, 2), _seq(2, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    assert c1.subject_ids[0].tolist() == [1, 1, NO_SUBJECT]
    # patient 1's last token (position 1) is a true end; position 2 is padding.
    assert c1.patient_end[0].tolist() == [False, True, False]


def test_patient_end_is_false_throughout_a_chunk_when_patient_continues() -> None:
    # a single long patient spanning two chunks: patient_end must be False
    # everywhere in the first chunk, since we haven't reached their true
    # last event yet.
    patients = _patients([_seq(1, 10)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    c1 = sampler.next_chunk()
    assert not c1.patient_end[0].any()


def test_patient_end_fires_on_the_correct_later_chunk() -> None:
    patients = _patients([_seq(1, 10)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    chunks = list(sampler)
    # 10 events, chunk_size 4: chunks cover events [0:4], [4:8], [8:10]+pad.
    # patient 1's true last event (index 9) falls in the third chunk.
    assert not chunks[0].patient_end[0].any()
    assert not chunks[1].patient_end[0].any()
    assert chunks[-1].patient_end[0].any()


def test_a_second_patient_end_never_shares_a_chunk_with_the_first() -> None:
    # four single-event patients: each patient_end now fires alone, in its
    # own chunk, rather than all four landing in one packed chunk -- a
    # second patient's reset can't share a chunk with the first (see
    # test_a_second_patient_never_starts_within_the_same_chunk).
    patients = _patients([_seq(i, 1) for i in range(1, 5)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    chunks = [sampler.next_chunk() for _ in range(4)]
    for chunk in chunks:
        assert chunk.patient_end[0].tolist() == [True, False, False, False]


def test_patient_end_is_false_on_padding() -> None:
    patients = _patients([_seq(1, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=5)
    c1 = sampler.next_chunk()
    assert c1.patient_end[0].tolist() == [False, True, False, False, False]


def test_no_padding_waste_when_patient_length_divides_chunk_size() -> None:
    # each patient's length exactly matches chunk_size, so every reset
    # lands precisely at position 0 of some chunk -- zero padding waste
    # on the *input* side. A third patient keeps c2 away from the true
    # end of the stream (whose last token has no real target of its own,
    # a separate, unrelated edge case -- see
    # test_padding_uses_no_subject_sentinel).
    patients = _patients([_seq(1, 4), _seq(2, 4), _seq(3, 4)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    c1 = sampler.next_chunk()
    c2 = sampler.next_chunk()
    # every input position is a real token in both chunks...
    assert (c1.subject_ids[0] != NO_SUBJECT).all()
    assert (c2.subject_ids[0] != NO_SUBJECT).all()
    # ...but each chunk's final target would be the *next* patient's first
    # token -- a prediction across the reset boundary, so it is not real.
    assert c1.real_mask[0].tolist() == [True, True, True, False]
    assert c2.real_mask[0].tolist() == [True, True, True, False]
    assert c2.reset_mask[0].tolist() == [True, False, False, False]


def test_patient_boundary_at_exact_chunk_end_is_never_a_target() -> None:
    # Patient 1's length exactly fills the first chunk, so patient 2's
    # first token sits at the window's final, target-only position. That
    # (input, target) pair crosses the reset boundary: the model has no
    # carried state to predict it from, so it must be masked out of both
    # targets (as PAD, for the loss's ignore_index) and real_mask (for
    # metric consumers) -- not silently trained on.
    patients = _patients([_seq(1, 4), _seq(2, 3)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    c1 = sampler.next_chunk()
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, 1002, 1003]
    assert c1.targets[0].tolist() == [1001, 1002, 1003, PAD_ID]
    assert c1.real_mask[0].tolist() == [True, True, True, False]

    # patient 2 still starts intact at position 0 of the next chunk, with
    # its own reset -- no event was lost, only the meaningless target.
    c2 = sampler.next_chunk()
    assert c2.batch.concept_ids[0, :3].tolist() == [2000, 2001, 2002]
    assert bool(c2.reset_mask[0, 0])


def test_padding_absorbs_a_patient_transition_within_a_chunk() -> None:
    # patients shorter than chunk_size still waste some chunk space at the
    # transition -- every event still appears as a real input token, but a
    # target that would predict a new patient's first token from the
    # previous patient's last hidden state (a prediction across the reset
    # boundary) is correctly never trained on, in any chunk.
    patients = _patients([_seq(i, 2) for i in range(1, 6)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    c1 = sampler.next_chunk()
    assert not c1.real_mask[0].all()
    chunks = [c1, *sampler]
    seen = [
        t
        for chunk in chunks
        for t, real in zip(chunk.targets[0].tolist(), chunk.real_mask[0].tolist())
        if real
    ]
    assert len(seen) == len(set(seen))
    # one valid within-patient target per patient (predicting each
    # patient's 2nd event from its 1st) -- the 5 cross-patient-boundary
    # pairs are correctly excluded.
    assert seen == [1001, 2001, 3001, 4001, 5001]


# ---------------------------------------------------------------------------
# Multiple lanes: independence and parallel refill
# ---------------------------------------------------------------------------


def test_lanes_are_independent_streams() -> None:
    patients = _patients([_seq(1, 6), _seq(2, 6)])
    sampler = PackedLaneSampler(patients, num_lanes=2, chunk_size=3)

    c1 = sampler.next_chunk()
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, 1002]
    assert c1.batch.concept_ids[1].tolist() == [2000, 2001, 2002]
    assert c1.subject_ids[0].tolist() == [1, 1, 1]
    assert c1.subject_ids[1].tolist() == [2, 2, 2]


def test_a_lane_running_dry_does_not_stall_other_lanes() -> None:
    # lane 0 gets a long patient, lane 1's patient is short and the queue
    # then runs out -- lane 1 should pad, lane 0 should keep going.
    patients = _patients([_seq(1, 20), _seq(2, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=2, chunk_size=4)

    c1 = sampler.next_chunk()
    c2 = sampler.next_chunk()
    assert c1.real_mask[0].all()  # lane 0 fully real
    assert c2.real_mask[0].all()  # lane 0 still going
    assert c2.real_mask[1].tolist() == [False, False, False, False]  # lane 1 exhausted


# ---------------------------------------------------------------------------
# Epoch end
# ---------------------------------------------------------------------------


def test_next_chunk_returns_none_once_everything_is_exhausted() -> None:
    patients = _patients([_seq(1, 3)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    assert sampler.next_chunk() is not None
    assert sampler.next_chunk() is None


def test_iteration_stops_cleanly() -> None:
    patients = _patients([_seq(1, 5), _seq(2, 5)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)
    chunks = list(sampler)
    assert len(chunks) > 0
    seen_targets = [
        t
        for chunk in chunks
        for t, real in zip(chunk.targets[0].tolist(), chunk.real_mask[0].tolist())
        if real
    ]
    # 10 total events across both patients; the very first event of patient
    # 1 is never a target (no predecessor), and predicting patient 2's
    # first event (2000) from patient 1's last hidden state is a
    # cross-reset-boundary prediction, also correctly excluded -- see
    # test_padding_absorbs_a_patient_transition_within_a_chunk.
    assert seen_targets == [1001, 1002, 1003, 1004, 2001, 2002, 2003, 2004]


def test_padding_uses_no_subject_sentinel() -> None:
    patients = _patients([_seq(1, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=5)
    c1 = sampler.next_chunk()
    assert c1.subject_ids[0].tolist() == [1, 1, NO_SUBJECT, NO_SUBJECT, NO_SUBJECT]
    # 2 real events (tokens 1000, 1001) give exactly 1 valid (input, target)
    # pair: predicting 1001 from 1000. Token 1001 has no successor to
    # predict, so real_mask is True only at that one position.
    assert c1.real_mask[0].tolist() == [True, False, False, False, False]


# ---------------------------------------------------------------------------
# Synthetic missing-history resets
# ---------------------------------------------------------------------------


def test_reset_prob_zero_never_synthesizes_mid_patient_resets() -> None:
    seq = _seq(1, 6, visit_orders=[0, 0, 1, 1, 2, 2])
    sampler = PackedLaneSampler(
        _patients([seq]), num_lanes=1, chunk_size=6, reset_prob=0.0
    )
    c1 = sampler.next_chunk()
    assert c1.reset_mask[0].tolist() == [True, False, False, False, False, False]


def test_reset_prob_one_always_resets_at_visit_boundaries() -> None:
    # A synthetic mid-visit reset is subject to the same mid-chunk
    # restriction as a real patient boundary: each visit's 2 events get
    # its own chunk, truncated at the next visit's reset rather than
    # packing all 3 visits into the one chunk_size=6 window.
    seq = _seq(1, 6, visit_orders=[0, 0, 1, 1, 2, 2])
    sampler = PackedLaneSampler(
        _patients([seq]), num_lanes=1, chunk_size=6, reset_prob=1.0
    )
    c1 = sampler.next_chunk()
    c2 = sampler.next_chunk()
    c3 = sampler.next_chunk()
    assert c1.reset_mask[0].tolist() == [True, False, False, False, False, False]
    assert c2.reset_mask[0].tolist() == [True, False, False, False, False, False]
    assert c3.reset_mask[0].tolist() == [True, False, False, False, False, False]
    assert c1.batch.concept_ids[0, :2].tolist() == [1000, 1001]
    assert c2.batch.concept_ids[0, :2].tolist() == [1002, 1003]
    assert c3.batch.concept_ids[0, :2].tolist() == [1004, 1005]


def test_reset_prob_is_deterministic_given_a_seed() -> None:
    seq = _seq(1, 20, visit_orders=list(range(20)))  # a new visit every event
    s1 = PackedLaneSampler(
        _patients([seq]), num_lanes=1, chunk_size=20, reset_prob=0.5, seed=42
    )
    s2 = PackedLaneSampler(
        _patients([_seq(1, 20, visit_orders=list(range(20)))]),
        num_lanes=1,
        chunk_size=20,
        reset_prob=0.5,
        seed=42,
    )
    assert s1.next_chunk().reset_mask.tolist() == s2.next_chunk().reset_mask.tolist()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_rejects_zero_lanes() -> None:
    try:
        PackedLaneSampler(_patients([]), num_lanes=0, chunk_size=4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_rejects_zero_chunk_size() -> None:
    try:
        PackedLaneSampler(_patients([]), num_lanes=1, chunk_size=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_empty_patient_stream_produces_no_chunks() -> None:
    sampler = PackedLaneSampler(_patients([]), num_lanes=2, chunk_size=4)
    assert sampler.next_chunk() is None


def test_zero_length_patient_is_skipped_without_corrupting_the_lane() -> None:
    empty = _seq(99, 0)
    sampler = PackedLaneSampler(
        _patients([_seq(1, 2), empty, _seq(2, 2)]), num_lanes=1, chunk_size=5
    )
    c1 = sampler.next_chunk()
    # patient 2 is deferred to its own chunk rather than starting mid-chunk;
    # the empty patient in between contributes nothing and corrupts nothing.
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, PAD_ID, PAD_ID, PAD_ID]
    assert c1.subject_ids[0].tolist() == [1, 1, NO_SUBJECT, NO_SUBJECT, NO_SUBJECT]

    c2 = sampler.next_chunk()
    assert c2.batch.concept_ids[0, :2].tolist() == [2000, 2001]
    assert c2.subject_ids[0, :2].tolist() == [2, 2]


def test_next_chunk_keeps_returning_none_after_exhaustion() -> None:
    sampler = PackedLaneSampler(_patients([_seq(1, 3)]), num_lanes=1, chunk_size=3)
    assert sampler.next_chunk() is not None
    assert sampler.next_chunk() is None
    # calling again after exhaustion must not raise or resurrect a chunk
    assert sampler.next_chunk() is None
    assert sampler.next_chunk() is None


def test_iterating_an_exhausted_sampler_a_second_time_yields_nothing() -> None:
    sampler = PackedLaneSampler(_patients([_seq(1, 3)]), num_lanes=1, chunk_size=3)
    first_pass = list(sampler)
    second_pass = list(sampler)
    assert len(first_pass) > 0
    assert second_pass == []


def test_chunk_size_one_advances_one_token_at_a_time() -> None:
    sampler = PackedLaneSampler(_patients([_seq(1, 3)]), num_lanes=1, chunk_size=1)
    chunks = list(sampler)
    assert [c.batch.concept_ids[0].item() for c in chunks] == [1000, 1001, 1002]
    assert [c.targets[0].item() for c in chunks] == [1001, 1002, 0]
    assert [c.real_mask[0].item() for c in chunks] == [True, True, False]


def test_batch_and_targets_are_long_tensors() -> None:
    sampler = PackedLaneSampler(_patients([_seq(1, 5)]), num_lanes=1, chunk_size=3)
    chunk = sampler.next_chunk()
    assert chunk.batch.concept_ids.dtype == torch.long
    assert chunk.targets.dtype == torch.long
    assert chunk.reset_mask.dtype == torch.bool
    assert chunk.real_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Fast-forward resume: odyssey.training.train's steps_into_epoch mechanism
# discards chunks from a fresh, identically-seeded sampler to reach the
# position a checkpoint was taken at, rather than resuming that epoch
# from its own beginning. These tests validate the core assumption that
# makes that correct: a fresh sampler, fast-forwarded N chunks, produces
# exactly the same subsequent chunks as one that was simply run for N
# chunks and then continued -- same patients, seed, num_lanes, chunk_size.
# ---------------------------------------------------------------------------


def _many_patients(n: int, events_each: int) -> List[PatientSequence]:
    return [_seq(i, events_each) for i in range(n)]


def test_fast_forward_matches_continuing_the_original_run() -> None:
    def make_sampler() -> PackedLaneSampler:
        return PackedLaneSampler(
            _patients(_many_patients(60, 20)),
            num_lanes=4,
            chunk_size=5,
            reset_prob=0.3,
            seed=42,
        )

    original = make_sampler()
    for _ in range(10):
        original.next_chunk()
    continued_tail = [c.batch.concept_ids.tolist() for c in original]

    fast_forwarded = make_sampler()
    for _ in range(10):
        fast_forwarded.next_chunk()
    fast_forwarded_tail = [c.batch.concept_ids.tolist() for c in fast_forwarded]

    assert continued_tail == fast_forwarded_tail
    assert len(continued_tail) > 0  # otherwise this test proves nothing


def test_fast_forward_past_the_end_of_the_epoch_yields_nothing() -> None:
    def make_sampler() -> PackedLaneSampler:
        return PackedLaneSampler(
            _patients(_many_patients(3, 4)), num_lanes=1, chunk_size=3, seed=0
        )

    sampler = make_sampler()
    total_chunks = len(list(sampler))

    fast_forwarded = make_sampler()
    for _ in range(total_chunks):
        fast_forwarded.next_chunk()
    assert fast_forwarded.next_chunk() is None
