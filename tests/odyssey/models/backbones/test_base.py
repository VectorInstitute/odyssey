"""Tests for the SequenceBackbone interface: TimeAwareState, resolve_prev_time_stamps.

Both are pure CPU/tensor logic with no dependency on any concrete backbone,
so every edge case is directly testable here rather than through a full
model stack.
"""

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.base import (
    SequenceBackbone,
    TimeAwareState,
    resolve_prev_time_stamps,
)


def _batch(time_stamps: torch.Tensor) -> ClinicalSequenceBatch:
    batch, seq_len = time_stamps.shape
    return ClinicalSequenceBatch(
        concept_ids=torch.ones(batch, seq_len, dtype=torch.long),
        aux=AuxiliaryInputs(
            type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            time_stamps=time_stamps,
            ages=torch.zeros(batch, seq_len),
            visit_orders=torch.zeros(batch, seq_len, dtype=torch.long),
            visit_segments=torch.zeros(batch, seq_len, dtype=torch.long),
        ),
    )


def _state(prev_time_stamps: torch.Tensor) -> TimeAwareState:
    return TimeAwareState(recurrent=None, prev_time_stamps=prev_time_stamps)


# ---------------------------------------------------------------------------
# resolve_prev_time_stamps
# ---------------------------------------------------------------------------


def test_no_state_returns_none_regardless_of_reset_mask() -> None:
    batch = _batch(torch.tensor([[10.0, 11.0]]))
    reset_mask = torch.zeros(1, 2, dtype=torch.bool)

    assert resolve_prev_time_stamps(None, batch, reset_mask) is None
    assert resolve_prev_time_stamps(None, batch, None) is None


def test_no_reset_mask_passes_carried_timestamp_through_unchanged() -> None:
    state = _state(torch.tensor([5.0, 6.0]))
    batch = _batch(torch.tensor([[10.0, 11.0], [20.0, 21.0]]))

    result = resolve_prev_time_stamps(state, batch, None)

    assert result is not None
    assert torch.equal(result, state.prev_time_stamps)


def test_reset_mask_with_no_rows_set_passes_carried_timestamp_through() -> None:
    state = _state(torch.tensor([5.0, 6.0]))
    batch = _batch(torch.tensor([[10.0, 11.0], [20.0, 21.0]]))
    reset_mask = torch.zeros(2, 2, dtype=torch.bool)

    result = resolve_prev_time_stamps(state, batch, reset_mask)

    assert result is not None
    assert torch.equal(result, state.prev_time_stamps)


def test_reset_mask_with_all_rows_set_overrides_every_row() -> None:
    state = _state(torch.tensor([5.0, 6.0]))
    batch = _batch(torch.tensor([[10.0, 11.0], [20.0, 21.0]]))
    reset_mask = torch.ones(2, 2, dtype=torch.bool)

    result = resolve_prev_time_stamps(state, batch, reset_mask)

    assert result is not None
    assert torch.equal(result, batch.aux.time_stamps[:, 0])


def test_reset_mask_overrides_only_the_reset_rows() -> None:
    state = _state(torch.tensor([5.0, 6.0, 7.0]))
    batch = _batch(torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]))
    reset_mask = torch.zeros(3, 2, dtype=torch.bool)
    reset_mask[1, 0] = True  # only row 1 resets

    result = resolve_prev_time_stamps(state, batch, reset_mask)

    assert result is not None
    assert result[0].item() == 5.0
    assert result[1].item() == batch.aux.time_stamps[1, 0].item()
    assert result[2].item() == 7.0


def test_reset_mask_only_looks_at_position_zero() -> None:
    # a reset later in the chunk (not position 0) must not affect the
    # cross-chunk carried timestamp at all -- that's the recurrent state's
    # job (each backbone's own reset handling), not this helper's.
    state = _state(torch.tensor([5.0]))
    batch = _batch(torch.tensor([[10.0, 11.0, 12.0]]))
    reset_mask = torch.zeros(1, 3, dtype=torch.bool)
    reset_mask[0, 1] = True

    result = resolve_prev_time_stamps(state, batch, reset_mask)

    assert result is not None
    assert result[0].item() == 5.0


def test_reset_mask_does_not_mutate_the_carried_state_in_place() -> None:
    prev = torch.tensor([5.0, 6.0])
    state = _state(prev)
    batch = _batch(torch.tensor([[10.0, 11.0], [20.0, 21.0]]))
    reset_mask = torch.zeros(2, 2, dtype=torch.bool)
    reset_mask[0, 0] = True

    resolve_prev_time_stamps(state, batch, reset_mask)

    assert torch.equal(state.prev_time_stamps, prev)
    assert state.prev_time_stamps[0].item() == 5.0


def test_reset_mask_with_mismatched_batch_size_raises() -> None:
    state = _state(torch.tensor([5.0, 6.0, 7.0]))
    batch = _batch(torch.tensor([[10.0, 11.0], [20.0, 21.0]]))
    reset_mask = torch.zeros(2, 2, dtype=torch.bool)  # only 2 rows, state has 3
    reset_mask[0, 0] = True

    with pytest.raises(IndexError):
        resolve_prev_time_stamps(state, batch, reset_mask)


def test_reset_mask_with_too_few_dimensions_raises() -> None:
    state = _state(torch.tensor([5.0]))
    batch = _batch(torch.tensor([[10.0, 11.0]]))
    reset_mask = torch.zeros(1, dtype=torch.bool)  # missing the seq_len axis

    with pytest.raises(IndexError):
        resolve_prev_time_stamps(state, batch, reset_mask)


# ---------------------------------------------------------------------------
# SequenceBackbone
# ---------------------------------------------------------------------------


def test_sequence_backbone_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        SequenceBackbone()  # type: ignore[abstract]
