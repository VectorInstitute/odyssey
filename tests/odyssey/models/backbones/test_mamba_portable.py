"""The pure-PyTorch ``mamba-ssm`` stand-ins (CPU; see the GPU parity test too).

The reference for the scan is the SSM recurrence itself, stepped one token
at a time; the references for the norms and attention are their textbook
formulas. Parameter names are pinned to ``mamba-ssm`` 2.3.0's so a GPU
checkpoint loads unchanged.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.models.backbones.mamba_portable import (
    MHA,
    InferenceParams,
    Mamba2WithState,
    RMSNorm,
    RMSNormGated,
    ssd_chunk_scan,
)


def _naive_scan(  # noqa: PLR0913, PLR0917
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``h_t = exp(dt_t a) h_{t-1} + dt_t x_t b_t^T``; ``y_t = h_t c_t + d x_t``."""
    heads = x.shape[2]
    b = b.repeat_interleave(heads // b.shape[2], dim=2)
    c = c.repeat_interleave(heads // c.shape[2], dim=2)
    dt = F.softplus(dt + dt_bias)
    ys = []
    for t in range(x.shape[1]):
        decay = torch.exp(dt[:, t] * a)[:, :, None, None]
        update = dt[:, t, :, None, None] * x[:, t, :, :, None] * b[:, t, :, None, :]
        state = decay * state + update
        ys.append(torch.einsum("bhpn,bhn->bhp", state, c[:, t]) + d[:, None] * x[:, t])
    return torch.stack(ys, dim=1), state


def _scan_inputs(
    seqlen: int, ngroups: int = 1, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    batch, heads, headdim, dstate = 2, 4, 3, 5

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen)

    return (
        randn(batch, seqlen, heads, headdim),
        randn(batch, seqlen, heads),
        -torch.rand(heads, generator=gen) * 2,
        randn(batch, seqlen, ngroups, dstate),
        randn(batch, seqlen, ngroups, dstate),
        randn(heads),
        randn(heads),
        randn(batch, heads, headdim, dstate),
    )


@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "ngroups"),
    [(16, 8, 1), (13, 8, 1), (5, 8, 1), (13, 4, 2)],
)
def test_chunk_scan_matches_the_stepwise_recurrence(
    seqlen: int, chunk_size: int, ngroups: int
) -> None:
    x, dt, a, b, c, d, dt_bias, state = _scan_inputs(seqlen, ngroups)
    y, final = ssd_chunk_scan(x, dt, a, b, c, chunk_size, d, dt_bias, state)
    y_ref, final_ref = _naive_scan(x, dt, a, b, c, d, dt_bias, state)
    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(final, final_ref, atol=1e-4, rtol=1e-4)


def test_no_initial_state_means_a_zero_state() -> None:
    x, dt, a, b, c, d, dt_bias, state = _scan_inputs(9)
    y_none, final_none = ssd_chunk_scan(x, dt, a, b, c, 4, d, dt_bias, None)
    y_zero, final_zero = ssd_chunk_scan(
        x, dt, a, b, c, 4, d, dt_bias, torch.zeros_like(state)
    )
    torch.testing.assert_close(y_none, y_zero)
    torch.testing.assert_close(final_none, final_zero)


def _mixer(chunk_size: int = 8, seed: int = 0) -> Mamba2WithState:
    torch.manual_seed(seed)
    mixer = Mamba2WithState(
        32, d_state=8, headdim=8, chunk_size=chunk_size, layer_idx=0
    )
    with torch.no_grad():
        for param in mixer.parameters():
            param.normal_(std=0.2)
    return mixer.eval()


def test_carried_state_continues_a_split_sequence_exactly() -> None:
    mixer = _mixer()
    u = torch.randn(2, 23, 32)
    whole = mixer(u, inference_params=InferenceParams(23, 2))
    params = InferenceParams(23, 2)
    parts = [
        mixer(u[:, s], inference_params=params)
        for s in (slice(0, 2), slice(2, 11), slice(11, 23))
    ]
    torch.testing.assert_close(torch.cat(parts, dim=1), whole, atol=1e-5, rtol=1e-5)


def test_output_does_not_depend_on_the_scan_chunk_size() -> None:
    u = torch.randn(1, 19, 32)
    small, large = _mixer(chunk_size=4), _mixer(chunk_size=64)
    torch.testing.assert_close(small(u), large(u), atol=1e-5, rtol=1e-5)


def test_single_token_decode_is_refused() -> None:
    params = InferenceParams(4, 1, seqlen_offset=1)
    with pytest.raises(NotImplementedError, match="decode"):
        _mixer()(torch.randn(1, 1, 32), inference_params=params)


def test_a_carried_state_needs_a_layer_index() -> None:
    mixer = Mamba2WithState(32, d_state=8, headdim=8)
    with pytest.raises(ValueError, match="layer_idx"):
        mixer(torch.randn(1, 3, 32), inference_params=InferenceParams(3, 1))


def test_rms_norm_matches_its_formula() -> None:
    norm = RMSNorm(6, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(1.0, 7.0))
    x = torch.randn(3, 6)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    torch.testing.assert_close(norm(x), expected)


@pytest.mark.parametrize("norm_before_gate", [False, True])
def test_gated_rms_norm_gates_on_the_documented_side(norm_before_gate: bool) -> None:
    norm = RMSNormGated(8, eps=1e-5, group_size=4, norm_before_gate=norm_before_gate)
    x, z = torch.randn(2, 8), torch.randn(2, 8)

    def grouped_rms(t: torch.Tensor) -> torch.Tensor:
        g = t.reshape(2, 2, 4)
        return (g / torch.sqrt(g.pow(2).mean(-1, keepdim=True) + 1e-5)).reshape(2, 8)

    gate = F.silu(z)
    expected = grouped_rms(x) * gate if norm_before_gate else grouped_rms(x * gate)
    torch.testing.assert_close(norm(x, z), expected)


@pytest.mark.parametrize("num_heads_kv", [4, 2, 1])
def test_attention_is_causal_multi_head_with_grouped_kv(num_heads_kv: int) -> None:
    torch.manual_seed(0)
    attn = MHA(16, num_heads=4, num_heads_kv=num_heads_kv).eval()
    x = torch.randn(2, 7, 16)
    head_dim = 4
    q, k, v = attn.in_proj(x).split(
        [4 * head_dim, num_heads_kv * head_dim, num_heads_kv * head_dim], dim=-1
    )
    q = q.reshape(2, 7, 4, head_dim).transpose(1, 2)
    k = k.reshape(2, 7, num_heads_kv, head_dim).transpose(1, 2)
    v = v.reshape(2, 7, num_heads_kv, head_dim).transpose(1, 2)
    k = k.repeat_interleave(4 // num_heads_kv, dim=1)
    v = v.repeat_interleave(4 // num_heads_kv, dim=1)
    scores = q @ k.transpose(-1, -2) / head_dim**0.5
    scores = scores.masked_fill(torch.ones(7, 7).triu(1).bool(), float("-inf"))
    context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 16)
    torch.testing.assert_close(attn(x), attn.out_proj(context), atol=1e-5, rtol=1e-5)


def test_attention_refuses_a_kv_cache() -> None:
    with pytest.raises(NotImplementedError):
        MHA(16, num_heads=4)(torch.randn(1, 2, 16), inference_params=object())


def test_parameter_names_and_shapes_match_mamba_ssm() -> None:
    """Pinned to mamba-ssm 2.3.0 so GPU checkpoints load without remapping."""
    d_model, d_state, headdim = 32, 8, 8
    d_inner, nheads = 2 * d_model, 2 * d_model // headdim
    conv_dim = d_inner + 2 * d_state
    mixer = Mamba2WithState(d_model, d_state=d_state, headdim=headdim)
    assert {k: tuple(v.shape) for k, v in mixer.state_dict().items()} == {
        "in_proj.weight": (2 * d_inner + 2 * d_state + nheads, d_model),
        "conv1d.weight": (conv_dim, 1, 4),
        "conv1d.bias": (conv_dim,),
        "dt_bias": (nheads,),
        "A_log": (nheads,),
        "D": (nheads,),
        "norm.weight": (d_inner,),
        "out_proj.weight": (d_model, d_inner),
    }
    assert set(MHA(16, num_heads=4).state_dict()) == {
        "in_proj.weight",
        "in_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
    }
    assert set(RMSNorm(8).state_dict()) == {"weight"}
