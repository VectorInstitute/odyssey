"""Tests for TinyGRUBackbone: the CPU stand-in for the real hybrid backbone.

Complements ``tests/odyssey/models/test_sequence_model.py`` and
``tests/odyssey/models/test_streaming_training.py`` (which exercise
TinyGRUBackbone through the full model stack) by testing the backbone in
isolation, including exact numerical equivalence properties that only
matter at this level.
"""

import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone


VOCAB_SIZE = 30
HIDDEN_SIZE = 16
PADDING_IDX = 0


def _make_batch(batch: int, seq_len: int) -> ClinicalSequenceBatch:
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, VOCAB_SIZE, (batch, seq_len)),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len)),
            time_stamps=torch.arange(seq_len).float().unsqueeze(0).expand(batch, -1)
            + torch.rand(batch, seq_len),
            ages=torch.rand(batch, seq_len) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len)),
            visit_segments=torch.randint(0, 3, (batch, seq_len)),
        ),
    )


def _make_backbone(num_layers: int = 2) -> TinyGRUBackbone:
    return TinyGRUBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=num_layers,
        padding_idx=PADDING_IDX,
    )


def _slice(batch: ClinicalSequenceBatch, start: int, end: int) -> ClinicalSequenceBatch:
    return ClinicalSequenceBatch(
        concept_ids=batch.concept_ids[:, start:end],
        aux=AuxiliaryInputs(
            *(
                None if v is None else v[:, start:end]
                for v in (getattr(batch.aux, f) for f in batch.aux._fields)
            )
        ),
    )


# ---------------------------------------------------------------------------
# Shapes / plumbing
# ---------------------------------------------------------------------------


def test_forward_shape_and_state_layer_count() -> None:
    backbone = _make_backbone(num_layers=3)
    batch = _make_batch(batch=4, seq_len=7)

    hidden_states, state = backbone(batch)

    assert hidden_states.shape == (4, 7, HIDDEN_SIZE)
    assert len(state.recurrent) == 3
    for layer_state in state.recurrent:
        assert layer_state.shape == (4, HIDDEN_SIZE)
    assert torch.equal(state.prev_time_stamps, batch.aux.time_stamps[:, -1])


def test_batch_size_one() -> None:
    backbone = _make_backbone()
    batch = _make_batch(batch=1, seq_len=5)

    hidden_states, state = backbone(batch)

    assert hidden_states.shape == (1, 5, HIDDEN_SIZE)
    assert state.recurrent[0].shape == (1, HIDDEN_SIZE)


def test_single_layer_output_equals_its_own_final_hidden_state() -> None:
    backbone = _make_backbone(num_layers=1)
    batch = _make_batch(batch=2, seq_len=6)

    hidden_states, state = backbone(batch)

    assert torch.equal(hidden_states[:, -1, :], state.recurrent[0])


def test_deeper_layer_uses_shallower_layer_output_as_its_input() -> None:
    # a 2-layer backbone's output must differ from a 1-layer one built from
    # the same first-layer cell -- otherwise layer 2 wouldn't be wired in.
    torch.manual_seed(0)
    two_layer = _make_backbone(num_layers=2)
    one_layer = _make_backbone(num_layers=1)
    one_layer.embeddings.load_state_dict(two_layer.embeddings.state_dict())
    one_layer.cells[0].load_state_dict(two_layer.cells[0].state_dict())

    batch = _make_batch(batch=2, seq_len=5)
    two_layer.eval()
    one_layer.eval()
    with torch.no_grad():
        out_two, _ = two_layer(batch)
        out_one, _ = one_layer(batch)

    assert not torch.allclose(out_two, out_one)


# ---------------------------------------------------------------------------
# State carrying across chunks
# ---------------------------------------------------------------------------


def test_chunked_forward_with_carried_state_matches_one_shot_forward() -> None:
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    chunk_len = 5
    full_batch = _make_batch(batch=3, seq_len=2 * chunk_len)

    with torch.no_grad():
        full_hidden, _ = backbone(full_batch)

        first = _slice(full_batch, 0, chunk_len)
        second = _slice(full_batch, chunk_len, 2 * chunk_len)
        hidden1, state1 = backbone(first)
        hidden2, _ = backbone(second, state=state1)

    assert torch.allclose(full_hidden[:, :chunk_len], hidden1, atol=1e-6)
    assert torch.allclose(full_hidden[:, chunk_len:], hidden2, atol=1e-6)


def test_state_carrying_is_not_a_no_op() -> None:
    # sanity check for the equivalence test above: carrying state must
    # actually change the second chunk's output relative to a fresh start,
    # otherwise the equivalence test would pass trivially.
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    chunk1 = _make_batch(batch=2, seq_len=4)
    chunk2 = _make_batch(batch=2, seq_len=4)

    with torch.no_grad():
        _, state1 = backbone(chunk1)
        with_state, _ = backbone(chunk2, state=state1)
        fresh, _ = backbone(chunk2, state=None)

    assert not torch.allclose(with_state, fresh)


def test_reset_at_position_zero_matches_fresh_state_other_rows_keep_carried_state() -> (
    None
):
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    chunk1 = _make_batch(batch=2, seq_len=4)
    chunk2 = _make_batch(batch=2, seq_len=4)

    reset_mask = torch.zeros(2, 4, dtype=torch.bool)
    reset_mask[0, 0] = True

    with torch.no_grad():
        _, state1 = backbone(chunk1)
        with_reset, _ = backbone(chunk2, state=state1, reset_mask=reset_mask)
        fresh, _ = backbone(chunk2, state=None)

    assert torch.allclose(with_reset[0], fresh[0], atol=1e-6)
    assert not torch.allclose(with_reset[1], fresh[1])


def test_reset_mid_chunk_cuts_off_state_from_before_it() -> None:
    # TinyGRUBackbone supports resets at any position, not just position 0
    # of a chunk (unlike the real hybrid backbone -- see hybrid.py's
    # NotImplementedError for resets past position 0). Two sequences that
    # differ only before the reset position must produce identical output
    # from the reset position onward.
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    seq_len = 6
    reset_at = 3

    batch_a = _make_batch(batch=1, seq_len=seq_len)
    concept_ids_b = batch_a.concept_ids.clone()
    concept_ids_b[0, :reset_at] = torch.randint(1, VOCAB_SIZE, (reset_at,))
    batch_b_full = batch_a._replace(concept_ids=concept_ids_b)

    reset_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    reset_mask[0, reset_at] = True

    with torch.no_grad():
        out_a, _ = backbone(batch_a, reset_mask=reset_mask)
        out_b, _ = backbone(batch_b_full, reset_mask=reset_mask)

    assert torch.allclose(out_a[0, reset_at:], out_b[0, reset_at:], atol=1e-6)


def test_all_rows_reset_matches_completely_fresh_computation() -> None:
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    chunk1 = _make_batch(batch=2, seq_len=4)
    chunk2 = _make_batch(batch=2, seq_len=4)
    reset_mask = torch.zeros(2, 4, dtype=torch.bool)
    reset_mask[:, 0] = True  # every row resets at the chunk boundary

    with torch.no_grad():
        _, state1 = backbone(chunk1)
        with_reset, _ = backbone(chunk2, state=state1, reset_mask=reset_mask)
        fresh, _ = backbone(chunk2, state=None)

    assert torch.allclose(with_reset, fresh, atol=1e-6)
