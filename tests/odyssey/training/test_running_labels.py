"""Tests for odyssey.training.running_labels."""

import torch

from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.training.running_labels import position_running_labels


def _chunk(subject_id: int, n: int) -> StreamingChunk:
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.arange(10, 10 + n).unsqueeze(0),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(1, n, dtype=torch.long),
                time_stamps=torch.arange(n, dtype=torch.float32).unsqueeze(0),
                ages=torch.full((1, n), 40.0),
                visit_orders=torch.zeros(1, n, dtype=torch.long),
                visit_segments=torch.zeros(1, n, dtype=torch.long),
            ),
        ),
        targets=torch.arange(11, 11 + n).unsqueeze(0),
        reset_mask=torch.ones(1, n, dtype=torch.bool),
        real_mask=torch.ones(1, n, dtype=torch.bool),
        subject_ids=torch.full((1, n), subject_id, dtype=torch.long),
        patient_end=torch.zeros(1, n, dtype=torch.bool),
        visit_ids=torch.full((1, n), -1, dtype=torch.long),
        visit_end=torch.zeros(1, n, dtype=torch.bool),
    )


def test_position_running_labels_passes_retrospective_label_through_when_no_first_times() -> (
    None
):
    # Pins the -inf sentinel's actual purpose (verified correct by
    # inspection: local torch.Generator elsewhere is unrelated, this is
    # about the *default* used when a key has no first-trigger info) --
    # deliberately the OPPOSITE convention from
    # odyssey.training.data.NEVER_TRIGGERED (+inf, "never triggered").
    # With concept_first_times=None, `now >= first_times` must be true at
    # every position so the retrospective label passes through completely
    # unchanged everywhere, not just wherever a real first-trigger time
    # happens to be supplied. A future "fix" making this +inf instead
    # (to match NEVER_TRIGGERED) would silently zero out every label for
    # every caller that doesn't supply concept_first_times -- this test
    # breaks loudly if that ever happens.
    chunk = _chunk(subject_id=1, n=4)
    concept_labels = {1: torch.tensor([1.0, 0.0])}
    concept_mask = {1: torch.tensor([1.0, 1.0])}

    labels, observed = position_running_labels(
        chunk,
        concept_labels,
        concept_mask,
        concept_first_times=None,
        supervision="stay",
        num_concepts=2,
    )

    assert observed.squeeze(0).tolist() == [[1.0, 1.0]] * 4
    assert labels.squeeze(0).tolist() == [[1.0, 0.0]] * 4


def test_position_running_labels_gates_on_first_trigger_time_when_supplied() -> None:
    # Contrast case: when a real first-trigger time IS supplied, the label
    # is false before it and true from it onward -- the actual "running"
    # semantics this module exists for, not the -inf passthrough default.
    chunk = _chunk(subject_id=1, n=4)  # time_stamps = [0, 1, 2, 3]
    concept_labels = {1: torch.tensor([1.0])}
    concept_mask = {1: torch.tensor([1.0])}
    concept_first_times = {1: torch.tensor([2.0])}

    labels, _ = position_running_labels(
        chunk,
        concept_labels,
        concept_mask,
        concept_first_times,
        supervision="stay",
        num_concepts=1,
    )

    assert labels.squeeze(0).squeeze(-1).tolist() == [0.0, 0.0, 1.0, 1.0]
