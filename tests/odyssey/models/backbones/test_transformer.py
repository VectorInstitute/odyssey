"""Tests for TransformerBackbone: the pure-attention control (roadmap Track A item 5).

Complements the packed-context sampler tests
(``tests/odyssey/data/test_packed_context.py``): those test packing logic
in isolation, these test the backbone's own reinterpretation of
``reset_mask`` as block-diagonal attention -- including the load-bearing
no-cross-patient-leakage property, end to end through embeddings and every
attention layer.
"""

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone, TimeAwareState
from odyssey.models.backbones.transformer import TransformerBackbone


VOCAB_SIZE = 40
HIDDEN_SIZE = 16
NUM_HEADS = 4
PADDING_IDX = 0


def _make_batch(batch: int, seq_len: int, *, seed: int = 0) -> ClinicalSequenceBatch:
    gen = torch.Generator().manual_seed(seed)
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, VOCAB_SIZE, (batch, seq_len), generator=gen),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len), generator=gen),
            time_stamps=torch.cumsum(torch.rand(batch, seq_len, generator=gen), dim=1),
            ages=torch.rand(batch, seq_len, generator=gen) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len), generator=gen),
            visit_segments=torch.randint(0, 3, (batch, seq_len), generator=gen),
        ),
    )


def _make_backbone(num_layers: int = 2) -> TransformerBackbone:
    """Build a backbone in eval mode.

    Every test here checks exact/near-exact numerical properties
    (determinism, causality, no leakage), which the shared
    ClinicalEventEmbeddings' own dropout (train-mode only, like any other
    backbone built on it -- see test_tiny_gru.py's identical convention)
    would otherwise inject per-call noise into.
    """
    return TransformerBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=num_layers,
        num_heads=NUM_HEADS,
        padding_idx=PADDING_IDX,
    ).eval()


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


def test_is_a_sequence_backbone() -> None:
    assert isinstance(_make_backbone(), SequenceBackbone)


def test_forward_shape_and_hidden_size_attribute() -> None:
    backbone = _make_backbone(num_layers=3)
    batch = _make_batch(batch=4, seq_len=7)

    hidden_states, state = backbone(batch)

    assert hidden_states.shape == (4, 7, HIDDEN_SIZE)
    assert backbone.hidden_size == HIDDEN_SIZE
    assert isinstance(state, TimeAwareState)
    assert torch.equal(state.prev_time_stamps, batch.aux.time_stamps[:, -1])


def test_state_argument_is_accepted_and_ignored() -> None:
    """A stateless backbone must not error, or change output, if state is passed."""
    backbone = _make_backbone()
    batch = _make_batch(batch=2, seq_len=5)

    out_no_state, _ = backbone(batch, state=None)
    dummy_state = TimeAwareState(
        recurrent="anything, this backbone must never read it",
        prev_time_stamps=torch.zeros(2),
    )
    out_with_state, _ = backbone(batch, state=dummy_state)

    assert torch.equal(out_no_state, out_with_state)


def test_batch_size_one() -> None:
    backbone = _make_backbone()
    batch = _make_batch(batch=1, seq_len=5)

    hidden_states, _ = backbone(batch)

    assert hidden_states.shape == (1, 5, HIDDEN_SIZE)


def test_reset_mask_none_defaults_to_one_segment_per_row() -> None:
    """No reset_mask (the ordinary, one-patient-per-row case) must not crash or NaN."""
    backbone = _make_backbone()
    batch = _make_batch(batch=2, seq_len=6)

    hidden_states, _ = backbone(batch, reset_mask=None)

    assert torch.isfinite(hidden_states).all()


def test_num_heads_not_dividing_hidden_size_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerBackbone(
            vocab_size=VOCAB_SIZE, hidden_size=10, num_heads=3, num_hidden_layers=1
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_repeated_calls() -> None:
    backbone = _make_backbone(num_layers=2)
    batch = _make_batch(batch=3, seq_len=9)

    out1, _ = backbone(batch)
    out2, _ = backbone(batch)

    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Causality: position t's logits must not depend on positions > t
# ---------------------------------------------------------------------------


def test_causal_position_invariant_to_future_token_changes() -> None:
    backbone = _make_backbone(num_layers=2)
    batch = _make_batch(batch=2, seq_len=8)

    hidden_states, _ = backbone(batch)

    changed_concept_ids = batch.concept_ids.clone()
    changed_concept_ids[:, 5:] = torch.randint(
        1, VOCAB_SIZE, changed_concept_ids[:, 5:].shape
    )
    changed_batch = ClinicalSequenceBatch(
        concept_ids=changed_concept_ids, aux=batch.aux
    )
    changed_hidden_states, _ = backbone(changed_batch)

    # Positions 0..4 (strictly before the changed region) must be untouched;
    # position 5 onward may (and generally will) differ.
    assert torch.equal(hidden_states[:, :5], changed_hidden_states[:, :5])
    assert not torch.equal(hidden_states[:, 5:], changed_hidden_states[:, 5:])


def test_causal_time_and_age_changes_also_respect_position_t() -> None:
    """Causality must hold for every input channel, not just token identity."""
    backbone = _make_backbone(num_layers=2)
    batch = _make_batch(batch=2, seq_len=8)

    hidden_states, _ = backbone(batch)

    changed_ages = batch.aux.ages.clone()
    changed_ages[:, 4:] += 50.0
    changed_batch = ClinicalSequenceBatch(
        concept_ids=batch.concept_ids,
        aux=batch.aux._replace(ages=changed_ages),
    )
    changed_hidden_states, _ = backbone(changed_batch)

    assert torch.equal(hidden_states[:, :4], changed_hidden_states[:, :4])


# ---------------------------------------------------------------------------
# No cross-patient leakage in packed (multi-segment) rows -- load-bearing
# ---------------------------------------------------------------------------


def test_no_cross_patient_leakage_changing_neighbor_leaves_patient_bit_identical() -> (
    None
):
    """Pack two patients (A then B) in one row; changing B must not move A at all."""
    backbone = _make_backbone(num_layers=3)
    len_a, len_b = 5, 4
    seq_len = len_a + len_b

    patient_a = _make_batch(batch=1, seq_len=len_a, seed=1)
    patient_b = _make_batch(batch=1, seq_len=len_b, seed=2)
    other_patient_b = _make_batch(batch=1, seq_len=len_b, seed=99)

    def _pack(
        a: ClinicalSequenceBatch, b: ClinicalSequenceBatch
    ) -> ClinicalSequenceBatch:
        concept_ids = torch.cat([a.concept_ids, b.concept_ids], dim=1)
        aux_fields = {}
        for name in a.aux._fields:
            va, vb = getattr(a.aux, name), getattr(b.aux, name)
            aux_fields[name] = None if va is None else torch.cat([va, vb], dim=1)
        return ClinicalSequenceBatch(
            concept_ids=concept_ids,
            aux=AuxiliaryInputs(**aux_fields),  # type: ignore[arg-type]
        )

    reset_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    reset_mask[0, 0] = True
    reset_mask[0, len_a] = True  # patient B starts here

    packed = _pack(patient_a, patient_b)
    packed_other_b = _pack(patient_a, other_patient_b)

    out, _ = backbone(packed, reset_mask=reset_mask)
    out_other_b, _ = backbone(packed_other_b, reset_mask=reset_mask)

    assert torch.equal(out[:, :len_a], out_other_b[:, :len_a])
    # sanity: B's own region generally does change (otherwise the test would
    # be vacuous -- the backbone ignoring all its inputs would also pass).
    assert not torch.equal(out[:, len_a:], out_other_b[:, len_a:])


def test_no_cross_patient_leakage_when_neighbor_precedes() -> None:
    """The same guarantee with the roles reversed: B changes, A comes after B."""
    backbone = _make_backbone(num_layers=3)
    len_a, len_b = 4, 6
    seq_len = len_a + len_b

    patient_b = _make_batch(batch=1, seq_len=len_b, seed=3)
    other_patient_b = _make_batch(batch=1, seq_len=len_b, seed=42)
    patient_a = _make_batch(batch=1, seq_len=len_a, seed=4)

    def _pack(
        b: ClinicalSequenceBatch, a: ClinicalSequenceBatch
    ) -> ClinicalSequenceBatch:
        concept_ids = torch.cat([b.concept_ids, a.concept_ids], dim=1)
        aux_fields = {}
        for name in b.aux._fields:
            vb, va = getattr(b.aux, name), getattr(a.aux, name)
            aux_fields[name] = None if vb is None else torch.cat([vb, va], dim=1)
        return ClinicalSequenceBatch(
            concept_ids=concept_ids,
            aux=AuxiliaryInputs(**aux_fields),  # type: ignore[arg-type]
        )

    reset_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    reset_mask[0, 0] = True
    reset_mask[0, len_b] = True  # patient A starts here

    packed = _pack(patient_b, patient_a)
    packed_other_b = _pack(other_patient_b, patient_a)

    out, _ = backbone(packed, reset_mask=reset_mask)
    out_other_b, _ = backbone(packed_other_b, reset_mask=reset_mask)

    # allclose, not equal: A's positions are later in the row here, so the
    # SDPA kernel's internal reduction still numerically touches B's
    # (masked-to-zero-contribution) key/value tiles while computing A's
    # attention output -- float32 non-associativity gives ~1e-7 max
    # difference, far below this tolerance, not a real leakage signal (see
    # the companion test above, where A precedes B and gets bit-exact
    # equality, for the case that has no such kernel-internal contact).
    assert torch.allclose(out[:, len_b:], out_other_b[:, len_b:], atol=1e-5)


def test_packed_patient_matches_processing_alone() -> None:
    """A patient's own hidden states, packed after another, equal processing it alone.

    The strongest form of the no-leakage property: not just "unaffected by
    a neighbor's *content*", but numerically identical to that patient
    never having had a neighbor at all -- position ids and attention are
    both fully segment-local.
    """
    backbone = _make_backbone(num_layers=3)
    len_a, len_b = 5, 3
    seq_len = len_a + len_b

    patient_a = _make_batch(batch=1, seq_len=len_a, seed=7)
    patient_b = _make_batch(batch=1, seq_len=len_b, seed=8)

    concept_ids = torch.cat([patient_a.concept_ids, patient_b.concept_ids], dim=1)
    aux_fields = {}
    for name in patient_a.aux._fields:
        va, vb = getattr(patient_a.aux, name), getattr(patient_b.aux, name)
        aux_fields[name] = None if va is None else torch.cat([va, vb], dim=1)
    packed = ClinicalSequenceBatch(
        concept_ids=concept_ids,
        aux=AuxiliaryInputs(**aux_fields),  # type: ignore[arg-type]
    )
    reset_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    reset_mask[0, 0] = True
    reset_mask[0, len_a] = True

    packed_out, _ = backbone(packed, reset_mask=reset_mask)
    alone_out, _ = backbone(patient_a)

    assert torch.allclose(packed_out[:, :len_a], alone_out, atol=1e-5)
