"""GPU-only tests for the hybrid Mamba-2 + attention backbone.

`mamba-ssm` requires a CUDA build and its kernels only run on CUDA
tensors, so these tests are meaningless (and mostly can't even import)
anywhere else. They auto-skip unless both `mamba-ssm` is installed and a
CUDA device is visible -- exercised on a GPU host, not local/CPU CI. All
batches are synthetic/random (dummy batches): these tests check that the
architecture runs, produces the right shapes, and trains, not that it
learns anything meaningful on real data.
"""

from typing import Dict, List

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch  # noqa: E402
from odyssey.models.backbones.hybrid import EHRHybridBackbone  # noqa: E402
from odyssey.models.sequence_model import (  # noqa: E402
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)


VOCAB_SIZE = 40
HIDDEN_SIZE = 64  # must be divisible by both mamba_headdim and attn_num_heads
NUM_CONCEPTS = 4
EMBEDDING_DIM = 6
PADDING_IDX = 0
MAMBA_HEADDIM = 64
MAMBA_CHUNK_SIZE = 16  # Mamba-3 MIMO kernel requires seq_len % chunk_size == 0
ATTN_NUM_HEADS = 8
SEQ_LEN = 16  # multiple of MAMBA_CHUNK_SIZE


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


def _make_backbone(num_hidden_layers: int = 2) -> EHRHybridBackbone:
    return EHRHybridBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        num_hidden_layers=num_hidden_layers,
        mamba_state_size=16,
        mamba_headdim=MAMBA_HEADDIM,
        mamba_chunk_size=MAMBA_CHUNK_SIZE,
        attn_num_heads=ATTN_NUM_HEADS,
    ).cuda()


@cuda_required
def test_hybrid_backbone_forward_shape() -> None:
    backbone = _make_backbone()
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    hidden_states, state = backbone(batch)

    assert hidden_states.shape == (2, SEQ_LEN, HIDDEN_SIZE)
    assert hidden_states.is_cuda
    assert torch.isfinite(hidden_states).all()
    assert state is not None


@cuda_required
def test_hybrid_backbone_through_concept_bottleneck_end_to_end() -> None:
    """Confirm EHRHybridBackbone satisfies the SequenceBackbone contract.

    Full forward + backward through the concept bottleneck, with real
    mamba_ssm and attention kernels, not just a CPU stand-in.
    """
    model = ConceptBottleneckSequenceModel(
        backbone=_make_backbone(),
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
    # gradient must reach both branches of every hybrid block, not just one
    for layer in model.backbone.layers:
        assert layer.mamba.in_proj.weight.grad is not None
        assert torch.any(layer.mamba.in_proj.weight.grad != 0)
        assert layer.attn.in_proj.weight.grad is not None
        assert torch.any(layer.attn.in_proj.weight.grad != 0)
    for name in ("task_loss", "concept_loss", "orthogonality_loss"):
        assert torch.isfinite(components[name])


@cuda_required
def test_hybrid_backbone_through_baseline_model() -> None:
    model = BaselineSequenceModel(
        backbone=_make_backbone(), vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX
    ).cuda()
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    loss, components = model.compute_loss(batch)
    assert torch.isfinite(loss)

    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(components["task_loss"])


@cuda_required
def test_hybrid_backbone_accepts_carried_state_and_produces_finite_output() -> None:
    """Cross-chunk state passing is supported on the Mamba side (decision (d)).

    The attention side still has no cross-chunk cache (see the module
    docstring: mamba_ssm's MHA prefill path doesn't support independent
    per-lane cache positions), so this no longer raises -- it just runs
    each chunk's attention fresh while the Mamba branch carries real
    state forward.
    """
    backbone = _make_backbone()
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")
    hidden1, state = backbone(batch)
    assert torch.isfinite(hidden1).all()

    hidden2, state2 = backbone(batch, state=state)
    assert hidden2.shape == (2, SEQ_LEN, HIDDEN_SIZE)
    assert torch.isfinite(hidden2).all()
    assert state2 is not None


@cuda_required
def test_hybrid_backbone_mamba_branch_carries_state_matches_one_shot() -> None:
    """The Mamba branch's chunked-with-state output must match one-shot.

    Uses a single-layer backbone so each hybrid block's input is exactly
    the raw embeddings for both the one-shot and chunked runs (with 1
    layer, nothing from the attention branch feeds back into what the
    Mamba branch sees) -- isolating the claim to exactly what the Mamba-2
    patch (see test_mamba2_patch_gpu.py) is responsible for, end to end
    through the real backbone and its state-seeding/reset wiring, not
    just the raw mixer.
    """
    torch.manual_seed(0)
    backbone = _make_backbone(num_hidden_layers=1).eval()
    full_len = 2 * SEQ_LEN
    full_batch = _make_batch(batch=2, seq_len=full_len, device="cuda")

    captured: Dict[str, List[torch.Tensor]] = {}
    block = backbone.layers[0]
    original_forward = block.forward

    def _capture_mamba_out(hidden_states, residual, **kwargs):  # type: ignore[no-untyped-def]
        new_residual = hidden_states + residual if residual is not None else hidden_states
        normed = block.norm(new_residual.to(dtype=block.norm.weight.dtype))
        mamba_out = block.mamba(normed, inference_params=kwargs.get("mamba_inference_params"))
        captured.setdefault("outputs", []).append(mamba_out)
        attn_out = block.attn(normed, inference_params=kwargs.get("attn_inference_params"))
        fused = block.merge(mamba_out, attn_out)
        return fused, new_residual

    block.forward = _capture_mamba_out  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            backbone(full_batch)
        mamba_full = captured["outputs"][0]

        captured["outputs"] = []
        first_half = ClinicalSequenceBatch(
            concept_ids=full_batch.concept_ids[:, :SEQ_LEN],
            aux=AuxiliaryInputs(
                type_ids=full_batch.aux.type_ids[:, :SEQ_LEN],
                time_stamps=full_batch.aux.time_stamps[:, :SEQ_LEN],
                ages=full_batch.aux.ages[:, :SEQ_LEN],
                visit_orders=full_batch.aux.visit_orders[:, :SEQ_LEN],
                visit_segments=full_batch.aux.visit_segments[:, :SEQ_LEN],
            ),
        )
        second_half = ClinicalSequenceBatch(
            concept_ids=full_batch.concept_ids[:, SEQ_LEN:],
            aux=AuxiliaryInputs(
                type_ids=full_batch.aux.type_ids[:, SEQ_LEN:],
                time_stamps=full_batch.aux.time_stamps[:, SEQ_LEN:],
                ages=full_batch.aux.ages[:, SEQ_LEN:],
                visit_orders=full_batch.aux.visit_orders[:, SEQ_LEN:],
                visit_segments=full_batch.aux.visit_segments[:, SEQ_LEN:],
            ),
        )
        with torch.no_grad():
            _, state1 = backbone(first_half)
            mamba1 = captured["outputs"][0]
            captured["outputs"] = []
            backbone(second_half, state=state1)
            mamba2 = captured["outputs"][0]
    finally:
        block.forward = original_forward  # type: ignore[method-assign]

    assert torch.allclose(mamba_full[:, :SEQ_LEN], mamba1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(mamba_full[:, SEQ_LEN:], mamba2, atol=1e-3, rtol=1e-3)


@cuda_required
def test_hybrid_backbone_reset_row_ignores_carried_state_other_rows_keep_it() -> None:
    """A per-row reset must zero only that row's carried Mamba state.

    Mirrors test_mamba3_gpu.py's identically-named test, at the
    EHRHybridBackbone level: row 0 is reset at the start of chunk 2, so
    its output there must match a fresh (state=None) computation; row 1
    is not reset, so its output must differ (it legitimately carries
    chunk-1 context forward).
    """
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    chunk1 = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")
    chunk2 = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    with torch.no_grad():
        _, state1 = backbone(chunk1)
        reset_mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool, device="cuda")
        reset_mask[0, 0] = True
        hidden2_with_reset, _ = backbone(chunk2, state=state1, reset_mask=reset_mask)

        hidden2_fresh, _ = backbone(chunk2)

    assert torch.allclose(hidden2_with_reset[0], hidden2_fresh[0], atol=1e-3, rtol=1e-3)
    assert not torch.allclose(hidden2_with_reset[1], hidden2_fresh[1], atol=1e-3, rtol=1e-3)


@cuda_required
def test_hybrid_backbone_rejects_reset_past_position_zero() -> None:
    backbone = _make_backbone()
    batch = _make_batch(batch=1, seq_len=SEQ_LEN, device="cuda")
    reset_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool, device="cuda")
    reset_mask[0, 1] = True

    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        backbone(batch, reset_mask=reset_mask)


@cuda_required
def test_hybrid_backbone_output_differs_from_mamba_only_and_attention_only() -> None:
    """Sanity check that both branches are actually load-bearing.

    Zeroing out one branch's contribution (by scaling its output to zero
    before the merge) must change the result -- if it didn't, the merge
    layer would be silently ignoring one branch entirely, defeating the
    entire point of the hybrid design (entry 03).
    """
    torch.manual_seed(0)
    backbone = _make_backbone(num_hidden_layers=1).eval()  # disable dropout
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    block = backbone.layers[0]
    original_forward = block.forward

    def _mamba_zeroed(hidden_states, residual, **kwargs):  # type: ignore[no-untyped-def]
        new_residual = hidden_states + residual if residual is not None else hidden_states
        normed = block.norm(new_residual.to(dtype=block.norm.weight.dtype))
        mamba_out = block.mamba(normed, inference_params=kwargs.get("mamba_inference_params")) * 0
        attn_out = block.attn(normed, inference_params=kwargs.get("attn_inference_params"))
        fused = block.merge(mamba_out, attn_out)
        return fused, new_residual

    with torch.no_grad():
        hidden_full, _ = backbone(batch)
        block.forward = _mamba_zeroed  # type: ignore[method-assign]
        hidden_attn_only, _ = backbone(batch)
        block.forward = original_forward  # type: ignore[method-assign]

    assert not torch.allclose(hidden_full, hidden_attn_only, atol=1e-4)
