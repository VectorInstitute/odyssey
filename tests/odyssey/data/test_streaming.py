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


def test_short_patients_are_packed_back_to_back_in_one_lane() -> None:
    patients = _patients([_seq(1, 2), _seq(2, 2), _seq(3, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    # patient 1 (2 events) + first event of patient 2, packed into one chunk
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, 2000]
    assert c1.subject_ids[0].tolist() == [1, 1, 2]


def test_patient_boundary_within_a_chunk_sets_reset_flag() -> None:
    patients = _patients([_seq(1, 2), _seq(2, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    # position 0: first token of patient 1 -> reset. position 2: first
    # token of patient 2 -> reset. position 1: mid-patient-1 -> no reset.
    assert c1.reset_mask[0].tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# patient_end: where concept-label supervision is valid
# ---------------------------------------------------------------------------


def test_patient_end_marks_a_patients_true_last_token() -> None:
    # patient 1 (2 events) fully fits, patient 2 (2 events) starts but only
    # its first token fits in this chunk.
    patients = _patients([_seq(1, 2), _seq(2, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=3)

    c1 = sampler.next_chunk()
    assert c1.subject_ids[0].tolist() == [1, 1, 2]
    # patient 1's last token (position 1) is a true end; patient 2's first
    # token (position 2) is not, since patient 2 has one more event left.
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


def test_patient_end_can_fire_multiple_times_in_one_chunk_with_short_patients() -> None:
    patients = _patients([_seq(i, 1) for i in range(1, 5)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    c1 = sampler.next_chunk()
    # every single-event patient's one token is also its last token
    assert c1.patient_end[0].tolist() == [True, True, True, True]


def test_patient_end_is_false_on_padding() -> None:
    patients = _patients([_seq(1, 2)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=5)
    c1 = sampler.next_chunk()
    assert c1.patient_end[0].tolist() == [False, True, False, False, False]


def test_no_padding_waste_when_enough_patients_are_queued() -> None:
    patients = _patients([_seq(i, 2) for i in range(1, 6)])
    sampler = PackedLaneSampler(patients, num_lanes=1, chunk_size=4)

    c1 = sampler.next_chunk()
    c2 = sampler.next_chunk()
    assert c1.real_mask[0].all()
    assert c2.real_mask[0].all()


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
    # 1 is never a target, giving 9 valid targets total.
    assert seen_targets == [1001, 1002, 1003, 1004, 2000, 2001, 2002, 2003, 2004]


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
    seq = _seq(1, 6, visit_orders=[0, 0, 1, 1, 2, 2])
    sampler = PackedLaneSampler(
        _patients([seq]), num_lanes=1, chunk_size=6, reset_prob=1.0
    )
    c1 = sampler.next_chunk()
    # resets at position 0 (patient start) and at every visit-order change
    assert c1.reset_mask[0].tolist() == [True, False, True, False, True, False]


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
    assert c1.batch.concept_ids[0].tolist() == [1000, 1001, 2000, 2001, PAD_ID]
    assert c1.subject_ids[0].tolist() == [1, 1, 2, 2, NO_SUBJECT]


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
