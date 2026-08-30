"""GPU-only tests for the hybrid Mamba-2 + attention backbone.

`mamba-ssm` requires a CUDA build and its kernels only run on CUDA
tensors, so these tests are meaningless (and mostly can't even import)
anywhere else. They auto-skip unless both `mamba-ssm` is installed and a
CUDA device is visible -- exercised on a GPU host, not local/CPU CI. All
batches are synthetic/random (dummy batches): these tests check that the
architecture runs, produces the right shapes, and trains, not that it
learns anything meaningful on real data.
"""

import contextlib
from collections.abc import Iterator
from typing import cast

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch  # noqa: E402
from odyssey.models.backbones.hybrid import (  # noqa: E402
    EHRHybridBackbone,
    HybridBlock,
)
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
def test_hybrid_backbone_does_not_mutate_the_state_it_was_given() -> None:
    """Passing a state into forward() must not corrupt the caller's copy.

    EHRHybridBackbone's own reset-zeroing loop and the Mamba2 mixer's
    ssm_state.copy_(last_state) write-back both mutate cache tensors in
    place; without cloning them first, calling backbone(chunk2,
    state=state1) would silently corrupt state1 as a side effect -- a
    caller holding onto a state for checkpointing or replay would see it
    change out from under them. A fresh (state=None) forward on the same
    input, after state has supposedly been "used", must still match a
    fresh forward computed before state was ever touched.
    """
    torch.manual_seed(0)
    backbone = _make_backbone().eval()
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    with torch.no_grad():
        _, state = backbone(batch)
        snapshot = {
            layer_idx: tuple(t.clone() for t in cached)
            for layer_idx, cached in state.recurrent.mamba_states.items()
        }

        backbone(batch, state=state)  # a second chunk; must not mutate `state`

        for layer_idx, cached in state.recurrent.mamba_states.items():
            for original, after in zip(snapshot[layer_idx], cached):
                assert torch.equal(original, after)


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

    captured: dict[str, list[torch.Tensor]] = {}
    block = cast(HybridBlock, backbone.layers[0])
    original_forward = block.forward

    def _capture_mamba_out(hidden_states, residual, **kwargs):  # type: ignore[no-untyped-def]
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        normed = block.norm(new_residual.to(dtype=block.norm.weight.dtype))
        mamba_out = block.mamba(
            normed, inference_params=kwargs.get("mamba_inference_params")
        )
        captured.setdefault("outputs", []).append(mamba_out)
        attn_out = block.attn(
            normed, inference_params=kwargs.get("attn_inference_params")
        )
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

    Row 0 is reset at the start of chunk 2, so its output there must
    match a fresh (state=None) computation; row 1 is not reset, so its
    output must differ (it legitimately carries chunk-1 context
    forward).
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
    assert not torch.allclose(
        hidden2_with_reset[1], hidden2_fresh[1], atol=1e-3, rtol=1e-3
    )


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

    block = cast(HybridBlock, backbone.layers[0])
    original_forward = block.forward

    def _mamba_zeroed(hidden_states, residual, **kwargs):  # type: ignore[no-untyped-def]
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        normed = block.norm(new_residual.to(dtype=block.norm.weight.dtype))
        mamba_out = (
            block.mamba(normed, inference_params=kwargs.get("mamba_inference_params"))
            * 0
        )
        attn_out = block.attn(
            normed, inference_params=kwargs.get("attn_inference_params")
        )
        fused = block.merge(mamba_out, attn_out)
        return fused, new_residual

    with torch.no_grad():
        hidden_full, _ = backbone(batch)
        block.forward = _mamba_zeroed  # type: ignore[method-assign]
        hidden_attn_only, _ = backbone(batch)
        block.forward = original_forward  # type: ignore[method-assign]

    assert not torch.allclose(hidden_full, hidden_attn_only, atol=1e-4)


@cuda_required
def test_hybrid_backbone_output_also_differs_when_attention_is_zeroed() -> None:
    """The symmetric half of the "both branches contribute" claim.

    The adjacent test zeroes the Mamba branch and shows the result
    differs from the full output -- proving attention alone isn't
    carrying the whole thing, but never checks the other direction. This
    zeroes the attention branch instead (mamba-only) and shows THAT also
    differs from the full output: if it didn't, the merge layer would be
    silently ignoring the attention branch entirely.
    """
    torch.manual_seed(0)
    backbone = _make_backbone(num_hidden_layers=1).eval()
    batch = _make_batch(batch=2, seq_len=SEQ_LEN, device="cuda")

    block = cast(HybridBlock, backbone.layers[0])
    original_forward = block.forward

    def _attn_zeroed(hidden_states, residual, **kwargs):  # type: ignore[no-untyped-def]
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        normed = block.norm(new_residual.to(dtype=block.norm.weight.dtype))
        mamba_out = block.mamba(
            normed, inference_params=kwargs.get("mamba_inference_params")
        )
        attn_out = (
            block.attn(normed, inference_params=kwargs.get("attn_inference_params")) * 0
        )
        fused = block.merge(mamba_out, attn_out)
        return fused, new_residual

    with torch.no_grad():
        hidden_full, _ = backbone(batch)
        block.forward = _attn_zeroed  # type: ignore[method-assign]
        hidden_mamba_only, _ = backbone(batch)
        block.forward = original_forward  # type: ignore[method-assign]

    assert not torch.allclose(hidden_full, hidden_mamba_only, atol=1e-4)


@contextlib.contextmanager
def _capturing_mamba_out(block: HybridBlock) -> Iterator[list[torch.Tensor]]:
    """Monkeypatch a HybridBlock to record its pre-merge Mamba output per call.

    Same technique as test_hybrid_backbone_mamba_branch_carries_state_
    matches_one_shot, factored out so a chunk-size sweep can reuse it
    without re-deriving the patch for every split.
    """
    captured: list[torch.Tensor] = []
    original_forward = block.forward

    def _capture(hidden_states, residual, **kwargs):  # type: ignore[no-untyped-def]
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        normed = block.norm(new_residual.to(dtype=block.norm.weight.dtype))
        mamba_out = block.mamba(
            normed, inference_params=kwargs.get("mamba_inference_params")
        )
        captured.append(mamba_out)
        attn_out = block.attn(
            normed, inference_params=kwargs.get("attn_inference_params")
        )
        fused = block.merge(mamba_out, attn_out)
        return fused, new_residual

    block.forward = _capture  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        block.forward = original_forward  # type: ignore[method-assign]


def _slice_batch(
    batch: ClinicalSequenceBatch, start: int, end: int
) -> ClinicalSequenceBatch:
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


@cuda_required
def test_hybrid_backbone_mamba_branch_matches_one_shot_across_chunk_size_splits() -> (
    None
):
    """Chunk-size invariance, TinyGRU's exact discipline, on the real backbone.

    test_hybrid_backbone_mamba_branch_carries_state_matches_one_shot
    already proves this for one fixed 2-way split at chunk_size==SEQ_LEN.
    This is the property _make_mamba2_with_state_cls exists to provide
    (initial_states actually wired into the chunk-scan kernel across
    forward() calls), and it's exactly the kind of bug that's invisible
    to every downstream check (loss curves, eval metrics all still look
    plausible) since it only shows up as numerical drift accumulating
    across streaming-chunk boundaries -- so it's worth pinning across
    genuinely different split granularities, not just one. Uses a small
    mamba_chunk_size (kernel-internal chunking; a forward() call's own
    seq_len must be a multiple of it) so a truly small streaming
    increment is a valid call, then sweeps small (4) through huge (32,
    i.e. only two calls) splits of the same 64-token sequence against the
    same one-shot (single-call, no state) ground truth.
    """
    torch.manual_seed(0)
    small_chunk_backbone = (
        EHRHybridBackbone(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            padding_idx=PADDING_IDX,
            num_hidden_layers=1,
            mamba_state_size=16,
            mamba_headdim=MAMBA_HEADDIM,
            mamba_chunk_size=4,
            attn_num_heads=ATTN_NUM_HEADS,
        )
        .cuda()
        .eval()
    )
    block = cast(HybridBlock, small_chunk_backbone.layers[0])

    full_len = 64
    full_batch = _make_batch(batch=2, seq_len=full_len, device="cuda")

    with torch.no_grad():
        with _capturing_mamba_out(block) as captured:
            small_chunk_backbone(full_batch)
        mamba_full = captured[0]

        for split_len in (4, 8, 32):  # small through huge, all multiples of 4
            state = None
            pieces = []
            with _capturing_mamba_out(block) as captured:
                for start in range(0, full_len, split_len):
                    chunk = _slice_batch(full_batch, start, start + split_len)
                    _, state = small_chunk_backbone(chunk, state=state)
                    pieces.append(captured[-1])
            streamed = torch.cat(pieces, dim=1)
            assert torch.allclose(streamed, mamba_full, atol=1e-3, rtol=1e-3), (
                f"chunk_size={split_len} diverged from the one-shot ground truth"
            )


@cuda_required
def test_hybrid_backbone_real_positions_are_invariant_to_trailing_padding() -> None:
    """The concrete, checkable form of "masking correct at padding".

    Padding is always appended at the end (packed_context.py's pad_to:
    PAD_ID + zeroed aux fields, right-padded, never interior) -- so real
    positions' outputs must not change depending on how much trailing
    padding follows them, in either the causal-attention branch (whose
    mask must not let padding leak backward) or the Mamba branch (whose
    parallel SSD-chunked kernel could in principle be shape-dependent even
    though the recurrence itself is causal). Not that padding positions
    look sensible (nothing downstream reads them -- loss is masked via
    PAD_ID/ignore_index), but that real positions are unaffected by
    whether/how much padding follows.
    """
    torch.manual_seed(0)
    backbone = _make_backbone().eval()

    real_len = SEQ_LEN
    pad_len = SEQ_LEN
    real_batch = _make_batch(batch=2, seq_len=real_len, device="cuda")

    padded_batch = ClinicalSequenceBatch(
        concept_ids=torch.cat(
            [
                real_batch.concept_ids,
                torch.full((2, pad_len), PADDING_IDX, device="cuda"),
            ],
            dim=1,
        ),
        aux=AuxiliaryInputs(
            type_ids=torch.cat(
                [real_batch.aux.type_ids, torch.zeros(2, pad_len, device="cuda")],
                dim=1,
            ).long(),
            time_stamps=torch.cat(
                [real_batch.aux.time_stamps, torch.zeros(2, pad_len, device="cuda")],
                dim=1,
            ),
            ages=torch.cat(
                [real_batch.aux.ages, torch.zeros(2, pad_len, device="cuda")], dim=1
            ),
            visit_orders=torch.cat(
                [real_batch.aux.visit_orders, torch.zeros(2, pad_len, device="cuda")],
                dim=1,
            ).long(),
            visit_segments=torch.cat(
                [
                    real_batch.aux.visit_segments,
                    torch.zeros(2, pad_len, device="cuda"),
                ],
                dim=1,
            ).long(),
        ),
    )

    with torch.no_grad():
        hidden_real, _ = backbone(real_batch)
        hidden_padded, _ = backbone(padded_batch)

    assert torch.allclose(
        hidden_padded[:, :real_len], hidden_real, atol=1e-3, rtol=1e-3
    )
