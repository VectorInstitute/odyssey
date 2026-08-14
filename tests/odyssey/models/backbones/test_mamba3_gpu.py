"""GPU-only tests for the real EHR-Mamba3 backbone.

`mamba-ssm` requires a CUDA build and its kernels only run on CUDA tensors,
so these tests are meaningless (and mostly can't even import) anywhere
else. They auto-skip unless both `mamba-ssm` is installed and a CUDA
device is visible -- exercised on a GPU host, not local/CPU CI.
"""

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch  # noqa: E402
from odyssey.models.backbones.mamba3 import EHRMamba3Backbone  # noqa: E402
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel  # noqa: E402


VOCAB_SIZE = 40
HIDDEN_SIZE = 64  # must be divisible by headdim
NUM_CONCEPTS = 4
EMBEDDING_DIM = 6
PADDING_IDX = 0
# Mamba3's MIMO kernels require seq_len % chunk_size == 0, and headdim=64
# is the smallest value that reliably compiles for both fwd and bwd
# TileLang kernels in mamba-ssm 2.3.2 -- headdim=16/32 hit warp-partitioning
# InternalErrors in the backward kernel for this shape. chunk_size=16
# matches the module's documented recommendation of 64 / mimo_rank (4).
MAMBA3_HEADDIM = 64
MAMBA3_CHUNK_SIZE = 16
SEQ_LEN = 16  # multiple of MAMBA3_CHUNK_SIZE


def _make_batch(batch: int, seq_len: int, device: str) -> ClinicalSequenceBatch:
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, VOCAB_SIZE, (batch, seq_len), device=device),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len), device=device),
            time_stamps=torch.rand(batch, seq_len, device=device) * 100,
            ages=torch.rand(batch, seq_len, device=device) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len), device=device),
            visit_segments=torch.randint(0, 3, (batch, seq_len), device=device),
        ),
    )


@cuda_required
def test_ehr_mamba3_backbone_forward_shape() -> None:
    backbone = EHRMamba3Backbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        state_size=16,
        num_hidden_layers=2,
        headdim=MAMBA3_HEADDIM,
        chunk_size=MAMBA3_CHUNK_SIZE,
    ).cuda()
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    hidden_states, state = backbone(batch)

    assert hidden_states.shape == (2, SEQ_LEN, HIDDEN_SIZE)
    assert hidden_states.is_cuda
    assert isinstance(state, dict)
    assert len(state) == 2  # one cache entry per layer (num_hidden_layers=2)


@cuda_required
def test_ehr_mamba3_backbone_through_concept_bottleneck_end_to_end() -> None:
    """Confirm EHRMamba3Backbone satisfies the SequenceBackbone contract.

    This is what CPU tests (TinyGRUBackbone) can't cover: the real
    backbone, through the full concept-bottleneck model, with a real
    backward pass against actual mamba_ssm kernels.
    """
    backbone = EHRMamba3Backbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        state_size=16,
        num_hidden_layers=2,
        headdim=MAMBA3_HEADDIM,
        chunk_size=MAMBA3_CHUNK_SIZE,
    )
    model = ConceptBottleneckSequenceModel(
        backbone=backbone,
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PADDING_IDX,
    ).cuda()

    batch = _make_batch(batch=3, seq_len=SEQ_LEN, device="cuda")
    concept_labels = torch.randint(0, 2, (3, NUM_CONCEPTS), device="cuda").float()

    total, components = model.compute_loss(batch, concept_labels)
    assert torch.isfinite(total)

    total.backward()
    assert model.bottleneck.context_proj.weight.grad is not None
    assert torch.any(model.bottleneck.context_proj.weight.grad != 0)
    for name in ("task_loss", "concept_loss", "orthogonality_loss"):
        assert torch.isfinite(components[name])


# ---------------------------------------------------------------------------
# Chunk-boundary state passing (odyssey/data/streaming.py's TBTT design).
#
# These validate the confidence-noted claim in mamba3.py's module
# docstring: chunk-boundary state passing via InferenceParams was
# confirmed correct by reading the mamba_ssm kernel source directly, but
# never executed, since this machine has no CUDA. Run these on the GCP A100
# VM before trusting that design for real training.
# ---------------------------------------------------------------------------


def _make_backbone(num_hidden_layers: int = 2) -> EHRMamba3Backbone:
    return EHRMamba3Backbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        state_size=16,
        num_hidden_layers=num_hidden_layers,
        headdim=MAMBA3_HEADDIM,
        chunk_size=MAMBA3_CHUNK_SIZE,
    ).cuda()


@cuda_required
def test_chunked_forward_with_carried_state_matches_one_shot_forward() -> None:
    """Chunked forward with carried state must match a one-shot forward.

    Splitting a sequence into two TBTT chunks is supposed to be exactly
    equivalent to processing it in one call. This is the core
    numerical-equivalence assumption the whole streaming training design
    (decision (i)/(j), entry 02) depends on for Mamba-3 specifically. If
    this fails, chunked training silently computes something different
    from what a full-history forward pass would.
    """
    torch.manual_seed(0)
    backbone = _make_backbone()
    full_len = 2 * MAMBA3_CHUNK_SIZE
    full_batch = _make_batch(batch=2, seq_len=full_len, device="cuda")

    hidden_full, _ = backbone(full_batch)

    def _slice_batch(batch: ClinicalSequenceBatch, start: int, end: int) -> ClinicalSequenceBatch:
        return ClinicalSequenceBatch(
            concept_ids=batch.concept_ids[:, start:end],
            aux=AuxiliaryInputs(
                type_ids=batch.aux.type_ids[:, start:end],
                time_stamps=batch.aux.time_stamps[:, start:end],
                ages=batch.aux.ages[:, start:end],
                visit_orders=batch.aux.visit_orders[:, start:end],
                visit_segments=batch.aux.visit_segments[:, start:end],
            ),
        )

    first_half = _slice_batch(full_batch, 0, MAMBA3_CHUNK_SIZE)
    second_half = _slice_batch(full_batch, MAMBA3_CHUNK_SIZE, full_len)

    hidden1, state1 = backbone(first_half)
    hidden2, _ = backbone(second_half, state=state1)

    assert torch.allclose(hidden_full[:, :MAMBA3_CHUNK_SIZE], hidden1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(hidden_full[:, MAMBA3_CHUNK_SIZE:], hidden2, atol=1e-3, rtol=1e-3)


@cuda_required
def test_reset_row_ignores_carried_state_other_rows_keep_it() -> None:
    """A per-row reset must zero only that row's carried state.

    Row 0 gets reset at the start of chunk 2: its output there must match
    a fresh (state=None) computation on the same chunk-2 content alone,
    not depend on chunk 1 at all. Row 1 is not reset: its chunk-2 output
    must differ from the fresh computation, since it legitimately carries
    chunk-1 context forward.
    """
    torch.manual_seed(0)
    backbone = _make_backbone()
    chunk1 = _make_batch(batch=2, seq_len=MAMBA3_CHUNK_SIZE, device="cuda")
    chunk2 = _make_batch(batch=2, seq_len=MAMBA3_CHUNK_SIZE, device="cuda")

    _, state1 = backbone(chunk1)
    reset_mask = torch.zeros(2, MAMBA3_CHUNK_SIZE, dtype=torch.bool, device="cuda")
    reset_mask[0, 0] = True
    hidden2_with_reset, _ = backbone(chunk2, state=state1, reset_mask=reset_mask)

    hidden2_fresh, _ = backbone(chunk2)

    assert torch.allclose(
        hidden2_with_reset[0], hidden2_fresh[0], atol=1e-3, rtol=1e-3
    )
    assert not torch.allclose(
        hidden2_with_reset[1], hidden2_fresh[1], atol=1e-3, rtol=1e-3
    )


@cuda_required
def test_reset_past_position_zero_raises_not_implemented() -> None:
    backbone = _make_backbone()
    batch = _make_batch(batch=1, seq_len=MAMBA3_CHUNK_SIZE, device="cuda")
    reset_mask = torch.zeros(1, MAMBA3_CHUNK_SIZE, dtype=torch.bool, device="cuda")
    reset_mask[0, 1] = True  # a mid-chunk reset, not just position 0

    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        backbone(batch, reset_mask=reset_mask)
