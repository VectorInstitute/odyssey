"""Hybrid Mamba-3 + attention backbone with parallel fusion.

Implements the architecture decided in
``research_journal/03_backbone_architecture.html``: decision (a) hybrid,
not pure Mamba; decision (b) parallel fusion, not sequential interleaving;
decision (c) dense, not Mixture-of-Experts.

Each :class:`HybridBlock` runs a Mamba-3 mixer and an attention mixer on
the *same* input in parallel (not one feeding the other), then combines
their outputs with :class:`MergeAttention`, a small learned attention over
the two branch outputs. This follows the "parallel hybrid with
merge-attention" design Lee et al. 2025 (arXiv:2510.26912) report
performing best for long-context recall specifically -- see entry 03,
Section 05. ``MergeAttention`` here is our own implementation in that
spirit, not a reproduction of their exact published architecture, since
the paper does not give enough implementation detail to reproduce exactly.

Requires `mamba-ssm`, which needs a CUDA/`nvcc` build; see
``mamba3.py``'s module docstring for the same constraint and for why the
block stack is built directly instead of through ``mamba_ssm``'s
high-level dispatcher.

State passing across chunks (for :mod:`odyssey.data.streaming`'s TBTT
training):

- **Mamba side**: identical mechanism to
  :class:`~odyssey.models.backbones.mamba3.EHRMamba3Backbone`, confirmed
  against the ``mamba_ssm.utils.generation.InferenceParams``/
  ``key_value_memory_dict`` cache. Per-row (per-lane) resets work by
  zeroing that row's cached state tensors.
- **Attention side**: chunk-boundary state carrying is NOT supported yet.
  ``mamba_ssm.modules.mha.MHA``'s prefill path
  (``_update_kv_cache``) addresses its cache with a single scalar
  ``inference_params.seqlen_offset`` shared across the whole batch, not
  the per-row ``lengths_per_sample`` tensor the dataclass also defines --
  confirmed by reading ``mha.py`` directly, where ``lengths_per_sample``
  is only read in the fast ``flash_attn_with_kvcache`` single-token decode
  path, and a ``# TODO: this only uses seqlen_offset and not
  lengths_per_sample`` comment in the library's own prefill path
  acknowledges the gap. Packed lanes need independent per-row cache
  positions (different lanes reset at different times), which this does
  not provide without custom cache-indexing code. Passing carried
  attention state across chunks raises ``NotImplementedError`` rather
  than silently attending to the wrong content. Each chunk's attention is
  still real, full attention over that chunk's own tokens (bounded by
  ``chunk_size`, not compressed into a fixed-size state the way the Mamba
  branch is), which is already a real recall improvement over a pure SSM
  even without cross-chunk carry-over -- see entry 03's decision (a).
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone, TimeAwareState
from odyssey.models.embeddings import CachedEHREmbeddings


MambaStateDict = Dict[int, Tuple[torch.Tensor, ...]]


class MergeAttention(nn.Module):
    """Learned fusion of two per-position branch outputs via a small attention.

    Treats each position's ``(mamba_out, attn_out)`` pair as a 2-token
    sequence and runs one learned query's attention over it, producing a
    single fused output per position. A learned query (rather than, say, a
    fixed average) lets the model decide per-position, per-channel how
    much to trust each branch.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize the merge-attention layer."""
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, 1, hidden_size))
        nn.init.xavier_uniform_(self.query.view(1, -1))
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**-0.5

    def forward(self, mamba_out: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """Fuse two ``(batch, seq_len, hidden_size)`` branch outputs into one."""
        branches = torch.stack([mamba_out, attn_out], dim=-2)  # (b, s, 2, h)
        keys = self.key_proj(branches)
        values = self.value_proj(branches)
        query = self.query.expand(mamba_out.shape[0], mamba_out.shape[1], 1, -1)
        scores = torch.einsum("bsqd,bskd->bsqk", query, keys) * self.scale
        weights = scores.softmax(dim=-1)
        fused = torch.einsum("bsqk,bskd->bsqd", weights, values)
        return fused.squeeze(-2)


class HybridState:
    """Opaque carried state for :class:`EHRHybridBackbone`: Mamba side only.

    The attention side has no cross-chunk state yet -- see the module
    docstring. ``mamba_states[i]`` is layer ``i``'s
    ``key_value_memory_dict``-style cache.
    """

    __slots__ = ("mamba_states",)

    def __init__(self, mamba_states: Dict[int, MambaStateDict]) -> None:
        """Initialize with one Mamba cache dict per hybrid layer index."""
        self.mamba_states = mamba_states


class HybridBlock(nn.Module):
    """One parallel Mamba + attention block, fused by :class:`MergeAttention`.

    Prenorm residual, matching ``mamba_ssm.modules.block.Block``'s
    structure conceptually (norm before the mixers, residual added after),
    but without ``Block``'s fused add-norm CUDA kernel optimization --
    correctness first; that is a later performance pass, not a
    correctness requirement.
    """

    def __init__(
        self,
        hidden_size: int,
        mamba_mixer_cls: Any,
        attn_mixer_cls: Any,
        norm_cls: Any,
    ) -> None:
        """Initialize one hybrid block."""
        super().__init__()
        self.norm = norm_cls(hidden_size)
        self.mamba = mamba_mixer_cls(hidden_size)
        self.attn = attn_mixer_cls(hidden_size)
        self.merge = MergeAttention(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        *,
        mamba_inference_params: Any = None,
        attn_inference_params: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(block_output, new_residual)``."""
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        normed = self.norm(new_residual.to(dtype=self.norm.weight.dtype))
        mamba_out = self.mamba(normed, inference_params=mamba_inference_params)
        attn_out = self.attn(normed, inference_params=attn_inference_params)
        fused = self.merge(mamba_out, attn_out)
        return fused, new_residual


class EHRHybridBackbone(SequenceBackbone):
    """A stack of parallel Mamba-3 + attention hybrid blocks."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 768,
        padding_idx: int = 0,
        num_hidden_layers: int = 8,
        mamba_state_size: int = 128,
        mamba_headdim: int = 64,
        mamba_is_mimo: bool = True,
        mamba_mimo_rank: int = 4,
        mamba_chunk_size: int = 256,
        attn_num_heads: int = 8,
        attn_num_heads_kv: Optional[int] = None,
        norm_epsilon: float = 1e-5,
        **embedding_kwargs: object,
    ) -> None:
        """Initialize the hybrid backbone.

        ``attn_num_heads_kv`` defaults to ``attn_num_heads`` (plain
        multi-head attention); set it lower for grouped-query attention,
        as Nemotron-H does (entry 03, Section 03).
        """
        try:
            # Deferred: mamba-ssm needs CUDA. See the module docstring.
            from mamba_ssm.modules.mamba3 import Mamba3  # noqa: PLC0415
            from mamba_ssm.modules.mha import MHA  # noqa: PLC0415
            from mamba_ssm.ops.triton.layer_norm import RMSNorm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "EHRHybridBackbone requires mamba-ssm, which needs a CUDA "
                "build: `uv sync --extra cuda --no-build-isolation`. Use "
                "odyssey.models.backbones.tiny_gru.TinyGRUBackbone for "
                "CPU development instead."
            ) from exc

        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )

        def _make_block(layer_idx: int) -> HybridBlock:
            mamba_cls = partial(
                Mamba3,
                layer_idx=layer_idx,
                d_state=mamba_state_size,
                headdim=mamba_headdim,
                is_mimo=mamba_is_mimo,
                mimo_rank=mamba_mimo_rank,
                chunk_size=mamba_chunk_size,
            )
            attn_cls = partial(
                MHA,
                num_heads=attn_num_heads,
                num_heads_kv=attn_num_heads_kv,
                layer_idx=layer_idx,
                causal=True,
            )
            norm_cls = partial(RMSNorm, eps=norm_epsilon)
            return HybridBlock(hidden_size, mamba_cls, attn_cls, norm_cls)

        self.layers = nn.ModuleList(
            [_make_block(i) for i in range(num_hidden_layers)]
        )
        self.norm_f = RMSNorm(hidden_size, eps=norm_epsilon)

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: Optional[TimeAwareState] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TimeAwareState]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        Raises ``NotImplementedError`` if ``reset_mask`` has any reset
        past position 0 of a chunk (packed multi-patient chunks) -- same
        constraint as :class:`~odyssey.models.backbones.mamba3.EHRMamba3Backbone`,
        for the same reason (see that module's docstring). Also raises if
        ``state`` is given at all, since cross-chunk attention state isn't
        supported yet (see the module docstring); the returned state is
        still tracked correctly, ready for whenever that's fixed.
        """
        from mamba_ssm.utils.generation import InferenceParams  # noqa: PLC0415

        if state is not None:
            raise NotImplementedError(
                "EHRHybridBackbone does not yet support carrying state "
                "across chunks: the attention side has no working "
                "chunk-boundary cache mechanism yet. See this module's "
                "docstring for the lengths_per_sample gap in mamba_ssm's "
                "MHA prefill path that causes this."
            )

        if reset_mask is not None and reset_mask.shape[1] > 1 and reset_mask[:, 1:].any():
            raise NotImplementedError(
                "EHRHybridBackbone does not yet support resets after "
                "position 0 of a chunk (packed multi-patient chunks). See "
                "odyssey.models.backbones.mamba3's module docstring for "
                "the cu_seqlens path this needs on the Mamba side; the "
                "attention side has the same gap."
            )

        self.embeddings.set_aux_inputs(batch.aux)  # state is always None here
        hidden_states = self.embeddings(batch.concept_ids)
        batch_size, seq_len, _ = hidden_states.shape

        mamba_ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch_size)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                mamba_inference_params=mamba_ip,
                attn_inference_params=None,
            )

        residual = hidden_states + residual if residual is not None else hidden_states
        result: torch.Tensor = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        new_state = TimeAwareState(
            recurrent=HybridState(
                mamba_states=cast(Dict[int, MambaStateDict], mamba_ip.key_value_memory_dict)
            ),
            prev_time_stamps=batch.aux.time_stamps[:, -1],
        )
        return result, new_state
