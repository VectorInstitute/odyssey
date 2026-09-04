"""Tests for PackedContextSampler: whole-history packing for the transformer backbone.

Complements ``tests/odyssey/models/backbones/test_transformer.py``: this
file tests packing/coverage/truncation logic in isolation (no backbone
involved), that file tests the backbone's own use of what this sampler
produces (segment isolation, no leakage) end to end.
"""

from collections.abc import Iterator

import pytest
import torch

from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import NO_SUBJECT
from odyssey.data.vocabulary import PAD_ID


def _seq(
    subject_id: int, n: int, *, visit_orders: list[int] | None = None
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


def _patients(seqs: list[PatientSequence]) -> Iterator[PatientSequence]:
    return iter(seqs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        PackedContextSampler(_patients([]), batch_size=0, max_context=16)


def test_max_context_must_allow_a_target() -> None:
    with pytest.raises(ValueError, match="max_context"):
        PackedContextSampler(_patients([]), batch_size=1, max_context=1)


def test_empty_patient_stream_returns_none_immediately() -> None:
    sampler = PackedContextSampler(_patients([]), batch_size=2, max_context=16)

    assert sampler.next_chunk() is None


# ---------------------------------------------------------------------------
# Packing: multiple patients per row, first-fit
# ---------------------------------------------------------------------------


def test_two_small_patients_pack_into_one_row() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 4)]), batch_size=1, max_context=16
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    assert chunk.batch.concept_ids.shape == (1, 16)
    subject_ids = chunk.subject_ids[0]
    assert (subject_ids[:3] == 1).all()
    assert (subject_ids[3:7] == 2).all()
    assert (subject_ids[7:] == NO_SUBJECT).all()


def test_reset_mask_marks_every_patient_start() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 4)]), batch_size=1, max_context=16
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    reset = chunk.reset_mask[0]
    assert reset[0].item() is True
    assert reset[3].item() is True
    assert not reset[1].item()
    assert not reset[2].item()
    assert not reset[4:].any().item()


def test_patient_end_marks_each_patients_true_last_token() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 4)]), batch_size=1, max_context=16
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    patient_end = chunk.patient_end[0]
    assert patient_end[2].item() is True  # patient 1's last real token
    assert patient_end[6].item() is True  # patient 2's last real token
    assert patient_end.sum().item() == 2


def test_patient_too_big_for_remaining_space_starts_a_new_row() -> None:
    # max_context=5: patient 1 (3 tokens) + patient 2 (4 tokens) = 7 > 5,
    # so patient 2 must start row 2, not spill into row 1's padding.
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 4)]), batch_size=2, max_context=5
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    assert (chunk.subject_ids[0, :3] == 1).all()
    assert (chunk.subject_ids[0, 3:] == NO_SUBJECT).all()
    assert (chunk.subject_ids[1, :4] == 2).all()
    assert (chunk.subject_ids[1, 4:] == NO_SUBJECT).all()


def test_a_lone_oversized_patient_still_starts_and_fills_its_own_row() -> None:
    """A single patient exactly filling capacity, no truncation needed, is fine."""
    sampler = PackedContextSampler(_patients([_seq(1, 6)]), batch_size=1, max_context=6)

    chunk = sampler.next_chunk()

    assert chunk is not None
    assert (chunk.subject_ids[0] == 1).all()


# ---------------------------------------------------------------------------
# Truncation: a patient longer than max_context
# ---------------------------------------------------------------------------


def test_patient_longer_than_max_context_is_truncated_from_the_left() -> None:
    long_patient = _seq(1, 10)
    sampler = PackedContextSampler(
        _patients([long_patient]), batch_size=1, max_context=4
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    kept_ids = chunk.batch.concept_ids[0].tolist()
    # subject 1's tokens are 1000..1009; keeping the most recent 4 means
    # 1006, 1007, 1008, 1009 survive, in order.
    assert kept_ids == [1006, 1007, 1008, 1009]


def test_truncated_patients_keep_their_true_time_stamps() -> None:
    """The kept tail is NOT rebased to 0: true times flow to every consumer.

    An earlier version rebased the window and had the alerts harness add the
    boundary back; the float64 round trip flipped landmark buckets on exact
    boundaries (research journal entry 44). The backbone derives deltas from
    reset_mask, so it never needed the rebase.
    """
    long_patient = _seq(1, 10)  # time_stamps 0..9
    sampler = PackedContextSampler(
        _patients([long_patient]), batch_size=1, max_context=4
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    kept_times = chunk.batch.aux.time_stamps[0].tolist()
    assert kept_times == [6.0, 7.0, 8.0, 9.0]
    assert sampler.truncation_boundaries == {1: 6.0}


def test_truncated_subject_ids_records_the_truncated_patient() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 10), _seq(2, 3)]), batch_size=2, max_context=4
    )

    sampler.next_chunk()

    assert sampler.truncated_subject_ids == [1]


def test_untruncated_patients_are_not_recorded() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 4)]), batch_size=2, max_context=16
    )

    sampler.next_chunk()

    assert sampler.truncated_subject_ids == []


def test_truncation_boundaries_records_the_original_frame_kept_start() -> None:
    """The boundary is in the patient's OWN original time (pre-rebase), not 0.

    _seq(1, 10) has time_stamps 0..9; keeping the most recent 4 (indices
    6..9) means the kept window starts at original time 6.0 -- not the
    0.0 the row's own (rebased) time_stamps show it as.
    """
    sampler = PackedContextSampler(
        _patients([_seq(1, 10)]), batch_size=1, max_context=4
    )

    sampler.next_chunk()

    assert sampler.truncation_boundaries == {1: 6.0}


def test_truncation_boundaries_empty_when_nothing_truncated() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 4)]), batch_size=2, max_context=16
    )

    sampler.next_chunk()

    assert sampler.truncation_boundaries == {}


def test_truncated_subject_ids_accumulates_across_multiple_calls() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 10), _seq(2, 12)]), batch_size=1, max_context=4
    )

    sampler.next_chunk()
    sampler.next_chunk()

    assert sampler.truncated_subject_ids == [1, 2]


# ---------------------------------------------------------------------------
# Targets and real_mask
# ---------------------------------------------------------------------------


def test_targets_are_next_token_within_a_patient() -> None:
    sampler = PackedContextSampler(_patients([_seq(1, 4)]), batch_size=1, max_context=8)

    chunk = sampler.next_chunk()

    assert chunk is not None
    # input: [1000, 1001, 1002, 1003, PAD, PAD, PAD, PAD]
    # targets for positions 0,1,2 are 1001,1002,1003; position 3 (patient's
    # own last token) and everything padded has no valid target.
    targets = chunk.targets[0].tolist()
    assert targets[:3] == [1001, 1002, 1003]
    assert targets[3] == PAD_ID


def test_real_mask_true_only_where_a_next_token_target_exists() -> None:
    sampler = PackedContextSampler(_patients([_seq(1, 4)]), batch_size=1, max_context=8)

    chunk = sampler.next_chunk()

    assert chunk is not None
    real = chunk.real_mask[0].tolist()
    assert real == [True, True, True, False, False, False, False, False]


def test_target_does_not_cross_a_patient_boundary() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 3)]), batch_size=1, max_context=6
    )

    chunk = sampler.next_chunk()

    assert chunk is not None
    # position 2 is patient 1's last real token; its target must not be
    # patient 2's first token (2000), even though that token immediately
    # follows it in the row.
    assert chunk.targets[0, 2].item() == PAD_ID
    assert not chunk.real_mask[0, 2].item()
    # position 5 (patient 2's own last token) also has no target: no
    # lookahead token exists past the window.
    assert chunk.targets[0, 5].item() == PAD_ID
    assert not chunk.real_mask[0, 5].item()


def test_last_window_position_never_has_a_target_even_when_full() -> None:
    """max_context means exactly the window size: no +1 lookahead token exists."""
    sampler = PackedContextSampler(_patients([_seq(1, 6)]), batch_size=1, max_context=6)

    chunk = sampler.next_chunk()

    assert chunk is not None
    assert chunk.targets[0, -1].item() == PAD_ID
    assert not chunk.real_mask[0, -1].item()


# ---------------------------------------------------------------------------
# Coverage: every position appears as input exactly once, over an epoch
# ---------------------------------------------------------------------------


def test_packing_round_trip_covers_every_position_exactly_once() -> None:
    # every length here is <= max_context, so nothing gets truncated --
    # truncation's own effect on coverage is tested separately above.
    patients = [_seq(i, n) for i, n in enumerate([3, 5, 2, 6, 4, 6, 1], start=1)]
    sampler = PackedContextSampler(_patients(patients), batch_size=2, max_context=6)

    seen: dict[int, list[int]] = {}
    for chunk in sampler:
        subject_ids = chunk.subject_ids
        concept_ids = chunk.batch.concept_ids
        real = subject_ids != NO_SUBJECT
        for subj, tok in zip(
            subject_ids[real].tolist(), concept_ids[real].tolist(), strict=True
        ):
            seen.setdefault(subj, []).append(tok)

    expected = {p.subject_id: p.concept_ids for p in patients}
    assert seen == expected


def test_epoch_ends_when_patient_queue_and_every_row_are_exhausted() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 3)]), batch_size=2, max_context=16
    )

    chunk1 = sampler.next_chunk()
    chunk2 = sampler.next_chunk()

    assert chunk1 is not None
    assert chunk2 is None


def test_iter_stops_cleanly_at_epoch_end() -> None:
    # max_context=3 forces each 3-token patient into its own row/chunk
    # (batch_size=1: no room for a second patient to join); with more
    # headroom, greedy packing would correctly combine them into fewer
    # chunks -- that behavior is exactly what the packing tests above
    # check, this test is about __iter__ terminating cleanly.
    patients = [_seq(1, 3), _seq(2, 3), _seq(3, 3)]
    sampler = PackedContextSampler(_patients(patients), batch_size=1, max_context=3)

    chunks = list(sampler)

    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_packing_is_deterministic_given_the_same_patient_order() -> None:
    patients_1 = [_seq(1, 3), _seq(2, 5), _seq(3, 2)]
    patients_2 = [_seq(1, 3), _seq(2, 5), _seq(3, 2)]
    sampler_1 = PackedContextSampler(_patients(patients_1), batch_size=2, max_context=6)
    sampler_2 = PackedContextSampler(_patients(patients_2), batch_size=2, max_context=6)

    chunk_1 = sampler_1.next_chunk()
    chunk_2 = sampler_2.next_chunk()

    assert chunk_1 is not None
    assert chunk_2 is not None
    assert torch.equal(chunk_1.batch.concept_ids, chunk_2.batch.concept_ids)
    assert torch.equal(chunk_1.reset_mask, chunk_2.reset_mask)
    assert torch.equal(chunk_1.subject_ids, chunk_2.subject_ids)


# ---------------------------------------------------------------------------
# Time precision (real-data scale)
# ---------------------------------------------------------------------------


def test_time_stamps_survive_real_data_magnitude_at_double_precision() -> None:
    """time_stamps must stay float64 through packing, not silently narrow to float32.

    Regression test for a real bug: verify_packed_landmark_rows disagreed
    with _index_rows_from_events' model-free ground truth on real eICU
    data -- 38,424 missing / 40,727 extra landmark rows on one alert
    event alone. Root cause, confirmed via a local repro (tiny CPU
    TransformerBackbone + PackedContextSampler over real eICU held-out
    shards): every single mismatch paired up as the same (subject, visit,
    bucket) landmark at nearly the same time, off by ~1e-6 hours -- e.g.
    434.416667 (the float64 ground truth, computed by polars) vs
    434.416656 (the packed path's own value). That gap is exactly
    float32's precision loss at real-data time magnitudes (hundreds of
    hours since a subject's origin): torch.tensor(434.416667,
    dtype=torch.float) rounds to 434.4166564941406, a different value at
    the 6th decimal place, which is exactly what
    ``_landmark_key_set``'s round-to-6-decimals comparison guard is
    sensitive to. 434.416667 here is the literal value observed in that
    real-data repro, not a hand-picked adversarial one -- ordinary
    patient timelines hit this given long enough follow-up.
    """
    real_data_hours = 434.416667
    seq = PatientSequence(
        subject_id=1,
        concept_ids=[1000, 1001],
        type_ids=[1, 1],
        time_stamps=[0.0, real_data_hours],
        ages=[30.0, 30.0],
        visit_orders=[0, 0],
        visit_segments=[0, 0],
    )
    sampler = PackedContextSampler(_patients([seq]), batch_size=1, max_context=8)

    chunk = sampler.next_chunk()

    assert chunk is not None
    assert chunk.batch.aux.time_stamps.dtype == torch.double
    recovered = float(chunk.batch.aux.time_stamps[0, 1])
    assert round(recovered, 6) == real_data_hours, (
        f"time_stamps lost precision in packing: {real_data_hours} -> {recovered}"
    )


# ---------------------------------------------------------------------------
# Sliding windows (window_stride)
# ---------------------------------------------------------------------------


def _scored_ids(sampler: PackedContextSampler) -> list[int]:
    ids: list[int] = []
    for chunk in sampler:
        assert chunk.score_mask is not None
        mask = chunk.score_mask & (chunk.subject_ids != NO_SUBJECT)
        ids.extend(chunk.batch.concept_ids[mask].tolist())
    return ids


def test_window_stride_scores_every_position_of_a_long_patient_exactly_once() -> None:
    long_patient = _seq(1, 10)
    sampler = PackedContextSampler(
        _patients([long_patient]), batch_size=1, max_context=4, window_stride=2
    )

    assert sorted(_scored_ids(sampler)) == [1000 + i for i in range(10)]
    assert sampler.truncated_subject_ids == []
    assert sampler.truncation_boundaries == {}


def test_window_stride_later_windows_score_only_their_new_positions() -> None:
    long_patient = _seq(1, 7)
    sampler = PackedContextSampler(
        _patients([long_patient]), batch_size=1, max_context=4, window_stride=2
    )
    chunks = list(sampler)
    # windows [0,4) scoring all, [2,6) scoring 4..5, and the tail [3,7) scoring 6
    windows = []
    for chunk in chunks:
        for row in range(chunk.batch.concept_ids.shape[0]):
            real = chunk.subject_ids[row] != NO_SUBJECT
            if not bool(real.any()):
                continue
            ids = chunk.batch.concept_ids[row][real].tolist()
            assert chunk.score_mask is not None
            scored = chunk.batch.concept_ids[row][chunk.score_mask[row] & real].tolist()
            windows.append((ids, scored))
    assert windows == [
        ([1000, 1001, 1002, 1003], [1000, 1001, 1002, 1003]),
        ([1002, 1003, 1004, 1005], [1004, 1005]),
        ([1003, 1004, 1005, 1006], [1006]),
    ]


def test_window_stride_leaves_short_patients_and_targets_alone() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 3), _seq(2, 2)]),
        batch_size=1,
        max_context=8,
        window_stride=4,
    )
    chunk = sampler.next_chunk()
    assert chunk is not None
    assert chunk.score_mask is not None
    real = chunk.subject_ids != NO_SUBJECT
    assert bool(chunk.score_mask[real].all())
    assert not bool(chunk.score_mask[~real].any())


def test_window_stride_is_none_by_default_and_score_mask_absent() -> None:
    sampler = PackedContextSampler(
        _patients([_seq(1, 10)]), batch_size=1, max_context=4
    )
    chunk = sampler.next_chunk()
    assert chunk is not None
    assert chunk.score_mask is None
    assert sampler.truncated_subject_ids == [1]


def test_window_stride_must_fit_the_window() -> None:
    with pytest.raises(ValueError, match="window_stride"):
        PackedContextSampler(
            _patients([]), batch_size=1, max_context=4, window_stride=5
        )
