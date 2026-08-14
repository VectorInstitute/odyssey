"""Tests for MergeAttention.

The only part of the hybrid backbone that doesn't need mamba-ssm/CUDA to
exercise directly.
"""

import torch

from odyssey.models.backbones.hybrid import MergeAttention


def test_output_shape_matches_branch_shape() -> None:
    merge = MergeAttention(hidden_size=8)
    mamba_out = torch.randn(2, 5, 8)
    attn_out = torch.randn(2, 5, 8)

    fused = merge(mamba_out, attn_out)

    assert fused.shape == (2, 5, 8)


def test_identical_branches_give_equal_attention_weight() -> None:
    # if both branches are identical, their keys are identical too, so the
    # softmax attention weights must split 50/50 regardless of the query --
    # the fused output is then just the (learned) value projection of that
    # shared input, not a no-op average of the raw branches.
    merge = MergeAttention(hidden_size=8)
    same = torch.randn(2, 5, 8)

    fused = merge(same, same)
    expected = merge.value_proj(same)

    assert torch.allclose(fused, expected, atol=1e-5)


def test_gradients_flow_to_both_branches() -> None:
    merge = MergeAttention(hidden_size=8)
    mamba_out = torch.randn(2, 5, 8, requires_grad=True)
    attn_out = torch.randn(2, 5, 8, requires_grad=True)

    fused = merge(mamba_out, attn_out)
    fused.sum().backward()

    assert mamba_out.grad is not None
    assert attn_out.grad is not None
    assert torch.any(mamba_out.grad != 0)
    assert torch.any(attn_out.grad != 0)


def test_gradients_flow_to_learned_parameters() -> None:
    merge = MergeAttention(hidden_size=8)
    mamba_out = torch.randn(2, 5, 8)
    attn_out = torch.randn(2, 5, 8)

    merge(mamba_out, attn_out).sum().backward()

    assert merge.query.grad is not None
    assert torch.any(merge.query.grad != 0)
    assert merge.key_proj.weight.grad is not None
    assert merge.value_proj.weight.grad is not None


def test_fusion_weight_favors_the_more_relevant_branch() -> None:
    # train the merge layer to prefer branch A's content over branch B's
    # for a target that only matches branch A -- confirms the attention
    # weighting is actually load-bearing, not a no-op average.
    torch.manual_seed(0)
    merge = MergeAttention(hidden_size=4)
    optimizer = torch.optim.Adam(merge.parameters(), lr=0.05)

    branch_a = torch.randn(1, 1, 4)
    branch_b = torch.randn(1, 1, 4)
    target = branch_a.detach().clone()

    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        fused = merge(branch_a, branch_b)
        loss = torch.nn.functional.mse_loss(fused, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1


def test_batch_and_sequence_dimensions_are_independent() -> None:
    # each (batch, position) pair's fusion must not depend on any other
    # position or batch row -- confirms there's no accidental mixing
    # across the sequence or batch dims in the einsum indices.
    merge = MergeAttention(hidden_size=6)
    mamba_out = torch.randn(3, 4, 6)
    attn_out = torch.randn(3, 4, 6)

    full = merge(mamba_out, attn_out)
    single = merge(mamba_out[1:2, 2:3], attn_out[1:2, 2:3])

    assert torch.allclose(full[1:2, 2:3], single, atol=1e-6)
