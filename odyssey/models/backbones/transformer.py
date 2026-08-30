"""Modern-vanilla decoder-only transformer backbone: the pure-attention control.

Roadmap Track A item 5 (see README): swapped in behind the same
tokenization, losses and heads as the hybrid Mamba-2 + attention backbone,
at a matched parameter and compute budget, to price the architecture
choice the way the no-bottleneck baseline prices the concept bottleneck.
Subset scale first; full scale only if the subset result is interesting in
either direction. If it matches the hybrid, simplicity wins and the
project switches; if it loses, the architecture choice finally has a
measured receipt instead of an assumption.

"Modern-vanilla" here means: RoPE, pre-norm RMSNorm, SwiGLU, and torch's
built-in ``scaled_dot_product_attention`` for causal self-attention. It
deliberately excludes anything that would make this the *best possible*
transformer rather than a fair, plain reference point -- no mixture of
experts, no grouped-query or sliding-window attention, no dropout, no
weight tying, no separate warmup/cooldown recipe from the rest of this
project's runs, no LLM pre/post-training extras of any kind. The arm's job
is to price the backbone choice, not to build the best transformer.

Stateless by design: unlike :class:`~odyssey.models.backbones.hybrid.EHRHybridBackbone`
(recurrent Mamba state carried across TBTT chunks) or
:class:`~odyssey.models.backbones.tiny_gru.TinyGRUBackbone`, this backbone
has nothing to carry across calls -- self-attention over the tokens
actually present in one forward call is the whole mechanism; there is no
compressed summary of earlier tokens to hand off. ``state`` is accepted
(interface conformance with :class:`~odyssey.models.backbones.base.SequenceBackbone`)
and always ignored; the returned state is an inert sentinel
(``recurrent=None``). This also means training with this backbone cannot
use the TBTT/``PackedLaneSampler`` chunking regime the recurrent backbones
need (deliberately short chunks, state carried lane to lane): it needs the
*whole* history it will ever see for a patient inside one context window,
or it silently forgets anything before the window -- exactly why this
module ships alongside :mod:`odyssey.data.packed_context`, which packs
multiple whole (or head-truncated) patients into one ``max_context``-token
window instead of chunking one patient across many.

Packed multi-patient windows and ``reset_mask``: this backbone
reinterprets ``reset_mask`` (defined generically by ``SequenceBackbone`` as
"state must be zeroed before this position") as a segment boundary:
cumulative-summing it along the sequence axis gives every position a
segment id, and self-attention is restricted to same-segment, causal pairs
(a block-diagonal causal mask) -- the only way a stateless attention
mechanism can express the same guarantee a stateful backbone gets from
zeroing its recurrent state at a reset: never attend across it. Position
ids for RoPE reset to 0 at every segment boundary too, so a patient's own
attention pattern is identical whether it is packed alongside others or
processed alone -- packing changes only compute efficiency, never the
represented computation for any one patient. See
:mod:`odyssey.data.packed_context` for the token-level argument (the
running time-offset trick) that makes the *embeddings* side of this
equally leakage-free, not just the attention side.

Tail patients (a single patient's own history longer than ``max_context``):
:class:`~odyssey.data.packed_context.PackedContextSampler` truncates from
the left, keeping the most recent ``max_context`` tokens (near-term
history is more relevant to near-term forecasting than distant history,
the same asymmetry :func:`~odyssey.data.sequences.build_patient_sequence`'s
own ``max_seq_len`` truncation already uses). Callers scoring this
backbone must treat truncated-tail patients as a distinct evaluation slice
rather than silently averaging them in with untruncated ones, since
whether losing distant history costs this backbone accuracy relative to
the hybrid's unbounded recurrent state is itself part of what this control
measures. The hook: ``PackedContextSampler.truncated_subject_ids``
accumulates every subject id truncated across the sampler's lifetime; an
eval harness reads it after a pass completes and reports those subjects'
metrics separately rather than pooling them into the headline number.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone, TimeAwareState
from odyssey.models.embeddings import CachedEHREmbeddings


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-centering, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        """Initialize the norm's learned scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x`` by its RMS, then scale."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class SwiGLU(nn.Module):
    """Gated MLP: ``down(silu(gate(x)) * up(x))``, the Llama-family FFN."""

    def __init__(self, hidden_size: int, ffn_mult: int = 4) -> None:
        """Initialize the gated MLP's three projections.

        ``ffn_mult`` scales the inner width before the standard 2/3
        SwiGLU correction (so the parameter count roughly matches a plain
        ``ffn_mult``-wide ReLU MLP with two projections instead of three).
        """
        super().__init__()
        inner = max(1, int(hidden_size * ffn_mult * 2 / 3))
        self.gate_proj = nn.Linear(hidden_size, inner, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP."""
        result: torch.Tensor = self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )
        return result


def _segment_ids(reset_mask: torch.Tensor) -> torch.Tensor:
    """Return a per-position segment id: increments at every reset.

    Only equality between two positions' ids is meaningful (whether they
    belong to the same packed patient); the absolute values carry no
    other information.
    """
    return torch.cumsum(reset_mask.long(), dim=1)


def _position_ids(reset_mask: torch.Tensor) -> torch.Tensor:
    """Return each position's index since its most recent reset (for RoPE).

    Position 0 of every segment gets id 0, whether or not that segment is
    the first in the row -- a packed patient's rotary angles are identical
    to processing that same patient alone.
    """
    seq_len = reset_mask.shape[1]
    idx = (
        torch.arange(seq_len, device=reset_mask.device)
        .unsqueeze(0)
        .expand_as(reset_mask)
    )
    segment_start = torch.where(reset_mask, idx, torch.zeros_like(idx))
    segment_start = torch.cummax(segment_start, dim=1).values
    return idx - segment_start


def _rebase_time_stamps(
    time_stamps: torch.Tensor, reset_mask: torch.Tensor
) -> torch.Tensor:
    """Force every segment boundary's time delta to exactly 0, elsewhere unchanged.

    :class:`~odyssey.models.embeddings.TimeEmbeddingLayer` computes
    time-since-previous-event as a delta over the *whole row's* raw
    timestamps, uniformly -- it has no notion of a packed segment
    boundary. Left alone, a packed segment's own first position would get
    whatever ``time_stamps[boundary] - time_stamps[boundary - 1]`` happens
    to be: a value that depends on the *previous* segment's absolute
    timestamps, exactly the cross-patient leakage this backbone must not
    have. Zeroing the delta at every reset (matching the "fresh sequence
    start" convention :class:`TimeEmbeddingLayer` already uses at row
    position 0 when no ``prev_value`` is given) and reconstructing the
    series by cumulative sum makes every non-boundary delta come out
    identical to the original (nothing but the boundary deltas changes),
    so this is a correction, not an approximation. Doing this inside the
    backbone -- from ``reset_mask`` alone -- rather than trusting a caller
    to have pre-shifted timestamps means the no-leakage guarantee holds
    for *any* valid ``reset_mask``, not only ones a particular sampler
    happens to construct carefully.
    """
    deltas = time_stamps[:, 1:] - time_stamps[:, :-1]
    deltas = deltas.masked_fill(reset_mask[:, 1:], 0.0)
    first = time_stamps[:, :1]
    return torch.cat([first, first + torch.cumsum(deltas, dim=1)], dim=1)


def _build_attn_mask(reset_mask: torch.Tensor) -> torch.Tensor:
    """Return a ``(batch, 1, seq, seq)`` bool mask: True where attention is allowed.

    Causal (position i may attend to position j <= i) AND same-segment
    (i and j belong to the same packed patient) -- a block-diagonal causal
    mask, block boundaries given by ``reset_mask``.
    """
    batch, seq_len = reset_mask.shape
    segment = _segment_ids(reset_mask)
    same_segment = segment.unsqueeze(2) == segment.unsqueeze(1)
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=reset_mask.device)
    )
    mask = same_segment & causal.unsqueeze(0)
    return mask.unsqueeze(1).expand(batch, 1, seq_len, seq_len)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    position_ids: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)``, each ``(batch, seq, head_dim)``."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim)
    )
    freqs = position_ids.unsqueeze(-1).float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x``, shape ``(batch, heads, seq, dim)``."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE, via ``scaled_dot_product_attention``."""

    def __init__(
        self, hidden_size: int, num_heads: int, rope_theta: float = 10000.0
    ) -> None:
        """Initialize projections; ``hidden_size`` must divide by ``num_heads``."""
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_theta = rope_theta
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the attention output, same shape as ``x``."""
        batch, seq_len, hidden = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = _rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(  # noqa: PLC0415 (torch >= 2.0 API)
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        result: torch.Tensor = self.out_proj(out)
        return result


class TransformerBlock(nn.Module):
    """One pre-norm transformer block: attention, then a SwiGLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        norm_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        ffn_mult: int = 4,
    ) -> None:
        """Initialize the block's norms, attention and MLP."""
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_epsilon)
        self.attn = CausalSelfAttention(hidden_size, num_heads, rope_theta=rope_theta)
        self.norm2 = RMSNorm(hidden_size, eps=norm_epsilon)
        self.mlp = SwiGLU(hidden_size, ffn_mult=ffn_mult)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the block: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""
        x = x + self.attn(self.norm1(x), position_ids, attn_mask)
        result: torch.Tensor = x + self.mlp(self.norm2(x))
        return result


class TransformerBackbone(SequenceBackbone):
    """A stack of modern-vanilla pre-norm transformer blocks. Stateless."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 256,
        padding_idx: int = 0,
        num_hidden_layers: int = 8,
        num_heads: int = 8,
        ffn_mult: int = 4,
        norm_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        **embedding_kwargs: object,
    ) -> None:
        """Initialize the transformer backbone.

        Defaults mirror :class:`~odyssey.models.backbones.hybrid.EHRHybridBackbone`'s
        ``hidden_size``/``num_hidden_layers`` so the two backbones start
        from a comparable depth/width before either is deliberately tuned
        to match the other's parameter count.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    norm_epsilon=norm_epsilon,
                    rope_theta=rope_theta,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm_f = RMSNorm(hidden_size, eps=norm_epsilon)

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: TimeAwareState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TimeAwareState]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        ``state`` is always ignored (this backbone is stateless; see the
        module docstring). ``reset_mask`` marks packed-patient segment
        boundaries within the row, not a cross-call recurrent-state reset:
        ``None`` (or an all-``False`` mask) is treated as "the whole row
        is one segment, starting at position 0" -- the ordinary,
        one-patient-per-row case every existing test batch already uses.
        """
        batch_size, seq_len = batch.concept_ids.shape

        resolved_reset_mask: torch.Tensor = (
            batch.concept_ids.new_zeros(batch_size, seq_len, dtype=torch.bool)
            if reset_mask is None
            else reset_mask
        )
        if seq_len > 0 and not bool(resolved_reset_mask[:, 0].all()):
            resolved_reset_mask = resolved_reset_mask.clone()
            resolved_reset_mask[:, 0] = True

        aux = batch.aux
        if seq_len > 0:
            aux = aux._replace(
                time_stamps=_rebase_time_stamps(aux.time_stamps, resolved_reset_mask)
            )
        self.embeddings.set_aux_inputs(aux, prev_time_stamps=None)
        hidden_states = self.embeddings(batch.concept_ids)

        position_ids = _position_ids(resolved_reset_mask)
        attn_mask = _build_attn_mask(resolved_reset_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attn_mask)
        hidden_states = self.norm_f(hidden_states)

        new_state = TimeAwareState(
            recurrent=None, prev_time_stamps=batch.aux.time_stamps[:, -1]
        )
        return hidden_states, new_state
