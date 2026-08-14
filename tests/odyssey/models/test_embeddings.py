"""Tests for the clinical event embedding layers.

No dedicated test file existed for this module before -- coverage was
entirely incidental, through :class:`~odyssey.models.backbones.tiny_gru.TinyGRUBackbone`
inside the sequence-model/streaming tests. This adds direct unit coverage,
in particular for :class:`TimeEmbeddingLayer`'s ``prev_value`` handling:
the subject of a real, previously-shipped chunk-boundary bug (see
``odyssey/models/embeddings.py``'s module docstring and the git history of
this file), so it deserves a test that isolates it from the rest of the
embedding stack rather than relying only on end-to-end streaming tests.
"""

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs
from odyssey.models.embeddings import (
    CachedEHREmbeddings,
    ClinicalEventEmbeddings,
    TimeEmbeddingLayer,
    VisitEmbedding,
)


torch.manual_seed(0)

VOCAB_SIZE = 30
HIDDEN_SIZE = 16
PADDING_IDX = 0


def _aux(batch: int, seq_len: int) -> AuxiliaryInputs:
    return AuxiliaryInputs(
        type_ids=torch.randint(0, 9, (batch, seq_len)),
        time_stamps=torch.rand(batch, seq_len).cumsum(dim=-1) * 10,
        ages=torch.rand(batch, seq_len) * 90,
        visit_orders=torch.randint(0, 5, (batch, seq_len)),
        visit_segments=torch.randint(0, 3, (batch, seq_len)),
    )


# ---------------------------------------------------------------------------
# TimeEmbeddingLayer: the chunk-boundary-sensitive delta computation
# ---------------------------------------------------------------------------


def test_time_embedding_zeroes_first_delta_without_prev_value() -> None:
    layer = TimeEmbeddingLayer(embedding_size=4, is_time_delta=True)
    out = layer(torch.tensor([[5.0, 8.0, 12.0]]))
    # A zeroed first delta feeds w*0 + phi into sin(), same as the bias alone.
    assert torch.allclose(out[:, 0], torch.sin(layer.phi))


def test_time_embedding_uses_prev_value_for_first_delta() -> None:
    """``prev_value`` replaces the implicit "sequence just started" zero.

    Isolates the exact math the chunk-boundary fix depends on: with
    ``prev_value``, position 0's delta must be ``time_stamps[:, 0] -
    prev_value``, not 0.
    """
    delta_layer = TimeEmbeddingLayer(embedding_size=4, is_time_delta=True)
    raw_layer = TimeEmbeddingLayer(embedding_size=4, is_time_delta=False)
    delta_layer.w.data.copy_(raw_layer.w.data)
    delta_layer.phi.data.copy_(raw_layer.phi.data)

    time_stamps = torch.tensor([[10.0, 12.0]])
    prev_value = torch.tensor([4.0])

    out_with_prev = delta_layer(time_stamps, prev_value=prev_value)
    expected_deltas = torch.tensor([[6.0, 2.0]])  # [10 - 4, 12 - 10]
    assert torch.allclose(out_with_prev, raw_layer(expected_deltas))


def test_time_embedding_prev_value_equal_to_first_timestamp_matches_omitting_it() -> (
    None
):
    """A caller resetting a lane sets ``prev_value`` to that row's own first timestamp.

    This must reproduce exactly the zero-delta convention that omitting
    ``prev_value`` entirely gives a genuinely fresh sequence -- the
    contract :func:`~odyssey.models.backbones.base.resolve_prev_time_stamps`
    relies on for reset lanes.
    """
    layer = TimeEmbeddingLayer(embedding_size=4, is_time_delta=True)
    time_stamps = torch.tensor([[10.0, 12.0]])

    out_with_prev = layer(time_stamps, prev_value=time_stamps[:, 0])
    out_without_prev = layer(time_stamps)
    assert torch.allclose(out_with_prev, out_without_prev)


def test_time_embedding_non_delta_mode_ignores_prev_value() -> None:
    # Ages are absolute, not delta-based (is_time_delta=False by default);
    # prev_value must have no effect on them.
    layer = TimeEmbeddingLayer(embedding_size=4)
    time_stamps = torch.tensor([[10.0, 12.0]])
    out_a = layer(time_stamps, prev_value=torch.tensor([999.0]))
    out_b = layer(time_stamps)
    assert torch.allclose(out_a, out_b)


def test_time_embedding_single_timestep_does_not_crash() -> None:
    layer = TimeEmbeddingLayer(embedding_size=4, is_time_delta=True)
    out = layer(torch.tensor([[7.0]]), prev_value=torch.tensor([3.0]))
    assert torch.isfinite(out).all()
    assert out.shape == (1, 1, 4)


# ---------------------------------------------------------------------------
# VisitEmbedding
# ---------------------------------------------------------------------------


def test_visit_embedding_shape() -> None:
    layer = VisitEmbedding(visit_order_size=3, embedding_size=8)
    out = layer(torch.randint(0, 3, (2, 5)))
    assert out.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# ClinicalEventEmbeddings
# ---------------------------------------------------------------------------


def test_clinical_event_embeddings_shape() -> None:
    layer = ClinicalEventEmbeddings(VOCAB_SIZE, HIDDEN_SIZE, PADDING_IDX)
    batch, seq_len = 3, 6
    input_ids = torch.randint(1, VOCAB_SIZE, (batch, seq_len))
    out = layer(input_ids, _aux(batch, seq_len))
    assert out.shape == (batch, seq_len, HIDDEN_SIZE)


def test_clinical_event_embeddings_single_token_sequence() -> None:
    layer = ClinicalEventEmbeddings(VOCAB_SIZE, HIDDEN_SIZE, PADDING_IDX)
    input_ids = torch.randint(1, VOCAB_SIZE, (2, 1))
    out = layer(input_ids, _aux(2, 1))
    assert out.shape == (2, 1, HIDDEN_SIZE)
    assert torch.isfinite(out).all()


def test_clinical_event_embeddings_prev_time_stamps_changes_only_first_position() -> (
    None
):
    """Only position 0's contribution should move when ``prev_time_stamps`` changes.

    Positions 1+ compute their delta from ``time_stamps`` alone, so an
    otherwise-identical forward pass with a different ``prev_time_stamps``
    must leave them untouched.
    """
    layer = ClinicalEventEmbeddings(VOCAB_SIZE, HIDDEN_SIZE, PADDING_IDX)
    layer.eval()
    batch, seq_len = 2, 5
    input_ids = torch.randint(1, VOCAB_SIZE, (batch, seq_len))
    aux = _aux(batch, seq_len)

    with torch.no_grad():
        out_a = layer(input_ids, aux, prev_time_stamps=aux.time_stamps[:, 0] - 1.0)
        out_b = layer(input_ids, aux, prev_time_stamps=aux.time_stamps[:, 0] - 5.0)

    assert not torch.allclose(out_a[:, 0], out_b[:, 0])
    assert torch.allclose(out_a[:, 1:], out_b[:, 1:])


def test_gradients_flow_to_every_embedding_parameter() -> None:
    layer = ClinicalEventEmbeddings(VOCAB_SIZE, HIDDEN_SIZE, PADDING_IDX)
    input_ids = torch.randint(1, VOCAB_SIZE, (3, 6))
    out = layer(input_ids, _aux(3, 6))
    out.sum().backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


# ---------------------------------------------------------------------------
# CachedEHREmbeddings: bridges the two-argument API into single-argument
# backbones (e.g. mamba_ssm's MixerModel calling self.embedding(input_ids)).
# ---------------------------------------------------------------------------


def test_cached_embeddings_forward_raises_without_set_aux_inputs() -> None:
    cached = CachedEHREmbeddings(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, padding_idx=PADDING_IDX
    )
    input_ids = torch.randint(1, VOCAB_SIZE, (2, 4))
    with pytest.raises(RuntimeError, match="set_aux_inputs"):
        cached(input_ids)


def test_cached_embeddings_matches_direct_call_and_consumes_aux_once() -> None:
    cached = CachedEHREmbeddings(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, padding_idx=PADDING_IDX
    )
    direct = ClinicalEventEmbeddings(VOCAB_SIZE, HIDDEN_SIZE, PADDING_IDX)
    direct.load_state_dict(cached.embeddings.state_dict())
    cached.eval()
    direct.eval()

    input_ids = torch.randint(1, VOCAB_SIZE, (2, 4))
    aux = _aux(2, 4)

    cached.set_aux_inputs(aux)
    with torch.no_grad():
        out_cached = cached(input_ids)
        out_direct = direct(input_ids, aux)
    assert torch.allclose(out_cached, out_direct)

    # set_aux_inputs is one-shot: a second forward without re-setting it
    # must raise rather than silently reusing stale aux inputs.
    with pytest.raises(RuntimeError, match="set_aux_inputs"):
        cached(input_ids)


def test_cached_embeddings_forwards_prev_time_stamps() -> None:
    cached = CachedEHREmbeddings(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, padding_idx=PADDING_IDX
    )
    cached.eval()
    input_ids = torch.randint(1, VOCAB_SIZE, (2, 3))
    aux = _aux(2, 3)

    with torch.no_grad():
        cached.set_aux_inputs(aux, prev_time_stamps=aux.time_stamps[:, 0])
        out_reset = cached(input_ids)
        cached.set_aux_inputs(aux, prev_time_stamps=aux.time_stamps[:, 0] - 100.0)
        out_with_gap = cached(input_ids)

    assert not torch.allclose(out_reset[:, 0], out_with_gap[:, 0])
