"""Pure-PyTorch stand-ins for the ``mamba-ssm`` modules the hybrid backbone uses.

``mamba-ssm`` needs CUDA and Triton, so :class:`EHRHybridBackbone` cannot be
built on a laptop (CPU or Apple MPS). This module re-implements the four
pieces the backbone takes from it -- ``Mamba2`` (with the carried-state fix
of :func:`odyssey.models.backbones.hybrid._make_mamba2_with_state_cls`),
``MHA``, the plain ``RMSNorm`` and ``InferenceParams`` -- in plain PyTorch,
following the reference (``*_ref``) code paths of ``mamba-ssm`` 2.3.0.

Parameter and buffer names match the upstream modules exactly, so a
checkpoint trained on the GPU loads with ``load_state_dict`` unchanged.

Scope: inference only. The chunked SSD scan is the "minimal SSD" algorithm
(Listing 1 of the Mamba-2 paper) computed in float32; it is not memory-
or speed-optimized and has no backward kernel worth training with. Only
the configuration the backbone uses is supported: ``ngroups=1`` style
grouping, ``rmsnorm=True``, ``norm_before_gate=False``, no ``d_mlp``
split, no rotary embedding or conv in attention, no varlen
(``cu_seqlens``/``seq_idx``) inputs and no single-token ``step()`` decode.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


@dataclass
class InferenceParams:
    """The fields of ``mamba_ssm.utils.generation.InferenceParams`` used here."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict[int, Any] = field(default_factory=dict)
    lengths_per_sample: torch.Tensor | None = None


class RMSNorm(nn.Module):
    """``mamba_ssm.ops.triton.layer_norm.RMSNorm`` without the fused kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        """Initialize with a unit weight, as upstream does."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by the root mean square of the last dimension."""
        dtype = x.dtype
        xf = x.float()
        rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rstd * self.weight.float()).to(dtype)


class RMSNormGated(nn.Module):
    """``mamba_ssm.ops.triton.layernorm_gated.RMSNorm`` (``rms_norm_ref`` path)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
    ) -> None:
        """Initialize with a unit weight, as upstream does."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``norm(x * silu(z))`` (or ``norm(x) * silu(z)``), in float32."""
        dtype = x.dtype
        xf = x.float()
        zf = z.float() if z is not None else None
        if zf is not None and not self.norm_before_gate:
            xf = xf * F.silu(zf)
        group = self.group_size or xf.shape[-1]
        xg = xf.reshape(*xf.shape[:-1], -1, group)
        rstd = torch.rsqrt(xg.square().mean(dim=-1, keepdim=True) + self.eps)
        out = (xg * rstd).flatten(-2) * self.weight.float()
        if zf is not None and self.norm_before_gate:
            out = out * F.silu(zf)
        result: torch.Tensor = out.to(dtype)
        return result


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: ``out[..., i, j] = sum(x[..., j+1 : i+1])``, -inf above."""
    t = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, t)
    lower = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~lower, 0)
    out = torch.cumsum(x, dim=-2)
    keep = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=0)
    return out.masked_fill(~keep, -torch.inf)


def ssd_chunk_scan(  # noqa: PLR0913, PLR0917
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    chunk_size: int,
    d: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked SSD scan, the math of ``mamba_chunk_scan_combined`` (z=None).

    Shapes: ``x`` (b, l, h, p), ``dt`` (b, l, h), ``a`` (h,), ``b``/``c``
    (b, l, g, n), ``d`` (h,), ``dt_bias`` (h,), ``initial_states``
    (b, h, p, n). Returns ``(y, final_state)`` with ``y`` (b, l, h, p).
    Computed in float32 throughout; ``dt`` gets ``softplus(dt + dt_bias)``.
    """
    batch, seqlen, nheads, _ = x.shape
    out_dtype = x.dtype
    ngroups = b.shape[2]
    dt = F.softplus(dt.float() + dt_bias.float())
    xf = x.float()
    bf = b.float().repeat_interleave(nheads // ngroups, dim=2)
    cf = c.float().repeat_interleave(nheads // ngroups, dim=2)
    pad = (-seqlen) % chunk_size
    if pad:
        # Zero dt on the padding keeps the state unchanged (decay exp(0)=1,
        # input dt*x = 0), so the final state is the true last-token state.
        dt = F.pad(dt, (0, 0, 0, pad))
        xf = F.pad(xf, (0, 0, 0, 0, 0, pad))
        bf = F.pad(bf, (0, 0, 0, 0, 0, pad))
        cf = F.pad(cf, (0, 0, 0, 0, 0, pad))
    n_chunks = (seqlen + pad) // chunk_size

    def chunked(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch, n_chunks, chunk_size, *t.shape[2:])

    xdt = chunked(xf * dt.unsqueeze(-1))  # (b, c, l, h, p)
    a_dt = chunked(dt * a.float()).permute(0, 3, 1, 2)  # (b, h, c, l)
    bc, cc = chunked(bf), chunked(cf)  # (b, c, l, h, n)
    a_cumsum = torch.cumsum(a_dt, dim=-1)

    decay = torch.exp(_segsum(a_dt))  # (b, h, c, l, s)
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", cc, bc, decay, xdt)

    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", bc, decay_states, xdt)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([initial_states.float().unsqueeze(1), states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(a_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", cc, states, torch.exp(a_cumsum))
    y = (y_diag + y_off).reshape(batch, n_chunks * chunk_size, nheads, -1)[:, :seqlen]
    y = y + xf[:, :seqlen] * d.float().view(1, 1, nheads, 1)
    return y.to(out_dtype), final_state


class Mamba2WithState(nn.Module):
    """``Mamba2`` + the backbone's carried-state fix, in plain PyTorch.

    Same parameters as ``mamba_ssm.modules.mamba2.Mamba2`` (``in_proj``,
    ``conv1d``, ``dt_bias``, ``A_log``, ``D``, ``norm``, ``out_proj``) and
    the same ``(conv_state, ssm_state)`` cache layout, so a chunk's carried
    state continues into the next chunk exactly as on the GPU.
    """

    def __init__(  # noqa: PLR0913
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        layer_idx: int | None = None,
    ) -> None:
        """Build the layer with upstream's shapes; values come from the checkpoint."""
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.d_ssm = self.d_inner
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_ssm // headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        conv_dim = self.d_ssm + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=True,
        )
        self.act = nn.SiLU()
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(
            self.d_ssm,
            eps=1e-5,
            group_size=self.d_ssm // ngroups,
            norm_before_gate=False,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _get_states_from_cache(
        self, inference_params: InferenceParams, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx is None:
            raise ValueError("a carried state needs layer_idx")
        layer = self.layer_idx
        cache = inference_params.key_value_memory_dict
        if layer not in cache:
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            cache[layer] = (conv_state, ssm_state)
        states: tuple[torch.Tensor, torch.Tensor] = cache[layer]
        return states

    def forward(
        self, u: torch.Tensor, inference_params: InferenceParams | None = None
    ) -> torch.Tensor:
        """Prefill one chunk ``u`` (b, l, d), reading and writing the carried state."""
        batch, seqlen, _ = u.shape
        conv_state = ssm_state = incoming_conv = None
        if inference_params is not None:
            if inference_params.seqlen_offset > 0:
                raise NotImplementedError("single-token decode is not ported")
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            incoming_conv = conv_state.clone()

        zxbcdt = self.in_proj(u)
        z, xbc, dt = torch.split(
            zxbcdt,
            [self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        xbc_t = xbc.transpose(1, 2)  # (b, d, l)
        if conv_state is not None:
            conv_state.copy_(F.pad(xbc_t, (self.d_conv - xbc_t.shape[-1], 0)))
        if incoming_conv is not None and self.d_conv > 1:
            padded = torch.cat(
                [incoming_conv[:, :, -(self.d_conv - 1) :], xbc_t], dim=-1
            )
            conv_out = F.conv1d(
                padded, self.conv1d.weight, self.conv1d.bias, groups=padded.shape[1]
            )
        else:
            conv_out = self.conv1d(xbc_t)[:, :, : -(self.d_conv - 1)]
        xbc = self.act(conv_out.transpose(1, 2))
        x, b, c = torch.split(
            xbc,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        y, final_state = ssd_chunk_scan(
            x.reshape(batch, seqlen, self.nheads, self.headdim),
            dt,
            -torch.exp(self.A_log.float()),
            b.reshape(batch, seqlen, self.ngroups, self.d_state),
            c.reshape(batch, seqlen, self.ngroups, self.d_state),
            self.chunk_size,
            self.D,
            self.dt_bias,
            ssm_state,
        )
        if ssm_state is not None:
            ssm_state.copy_(final_state)
        y = self.norm(y.flatten(-2), z)
        out: torch.Tensor = self.out_proj(y)
        return out


class MHA(nn.Module):
    """``mamba_ssm.modules.mha.MHA`` for the configuration the backbone uses.

    Causal self-attention with optional grouped-query heads; no rotary
    embedding, no conv, no MLP split, and no KV cache (the backbone always
    passes ``inference_params=None`` to attention).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_kv: int | None = None,
        causal: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        """Build the layer with upstream's shapes; values come from the checkpoint."""
        super().__init__()
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.layer_idx = layer_idx
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        self.in_proj = nn.Linear(embed_dim, qkv_dim, bias=True)
        self.out_proj = nn.Linear(self.head_dim * num_heads, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, inference_params: Any = None) -> torch.Tensor:  # noqa: ANN401
        """Attend over the chunk ``x`` (b, l, d)."""
        if inference_params is not None:
            raise NotImplementedError("attention KV caching is not ported")
        qkv = self.in_proj(x)
        q, kv = qkv.split(
            [self.num_heads * self.head_dim, 2 * self.num_heads_kv * self.head_dim],
            dim=-1,
        )
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)
        k, v = kv.reshape(*kv.shape[:-1], 2, self.num_heads_kv, self.head_dim).unbind(
            dim=-3
        )
        repeats = self.num_heads // self.num_heads_kv
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
        context = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.causal,
        ).transpose(1, 2)
        out: torch.Tensor = self.out_proj(context.flatten(-2))
        return out


def mamba_ssm_available() -> bool:
    """Return whether the CUDA ``mamba-ssm`` build can be imported."""
    try:
        import mamba_ssm  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


__all__ = [
    "MHA",
    "InferenceParams",
    "Mamba2WithState",
    "RMSNorm",
    "RMSNormGated",
    "mamba_ssm_available",
    "ssd_chunk_scan",
]
