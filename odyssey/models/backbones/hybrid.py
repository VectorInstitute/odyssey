"""Hybrid Mamba-2 + attention backbone with parallel fusion.

Implements the architecture decided in
``research_journal/03_backbone_architecture.html``: decision (a) hybrid,
not pure Mamba; decision (b) parallel fusion, not sequential interleaving;
decision (c) dense, not Mixture-of-Experts; decision (d) Mamba-2 kernel,
not Mamba-3, for the SSM branch.

Each :class:`HybridBlock` runs a Mamba mixer and an attention mixer on
the *same* input in parallel (not one feeding the other), then combines
their outputs with :class:`MergeAttention`, a small learned attention over
the two branch outputs. This follows the "parallel hybrid with
merge-attention" design Lee et al. 2025 (arXiv:2510.26912) report
performing best for long-context recall specifically -- see entry 03,
Section 05. ``MergeAttention`` here is our own implementation in that
spirit, not a reproduction of their exact published architecture, since
the paper does not give enough implementation detail to reproduce exactly.

Requires `mamba-ssm`, which needs a CUDA/`nvcc` build. The block stack is
built directly (each :class:`HybridBlock` constructed by hand) rather
than through ``mamba_ssm``'s high-level ``MixerModel`` dispatcher: that
dispatcher only builds a sequential stack of single-mixer blocks, with no
way to express this architecture's per-block parallel Mamba+attention
fusion.

Why Mamba-2, not Mamba-3, for the Mamba branch (entry 03, decision (d)):
GPU validation found that neither of Mamba-3's two kernel variants can
carry state across chunks correctly. MIMO's ``mamba3_mimo_forward`` has
no parameter to seed an initial state at all (confirmed by reading its
signature). SISO's kernel (``mamba3_siso_combined``) does have a real
state-seeding parameter, but an independently reported GitHub issue
(mamba-ssm #1017, filed the day before this investigation, unrelated
user) found 3-7.5% numerical deviation in that same kernel and explicitly
implicated per-chunk boundary state passing as a suspected cause. Mamba-2's
``mamba_chunk_scan_combined`` kernel is the same SSD formulation, mature
and production-proven since 2024, with no analogous compile-fragility or
numerical-deviation reports. It has the same *wiring* gap as Mamba-3 (its
``Mamba2.forward`` never passes ``initial_states`` into the kernel either,
despite the kernel supporting it) but that gap is a one-line fix in
application code, not a kernel-level limitation -- see
:func:`_make_mamba2_with_state_cls` below for the patched subclass.

State passing across chunks (for :mod:`odyssey.data.streaming`'s TBTT
training):

- **Mamba side**: supported. :func:`_make_mamba2_with_state_cls` builds a
  ``Mamba2`` subclass that seeds ``initial_states`` from the carried-over
  ``key_value_memory_dict`` cache, and this backbone's ``forward`` zeroes
  a row's cached state before the layer call when that row's
  ``reset_mask`` fires. See that function's docstring for the SSM-state
  and causal-conv left-context fixes, and the one remaining,
  deliberately-not-fixed gap (the fast Triton ``causal_conv1d_fn`` path,
  dead code in every environment this was validated on).
- **Attention side**: chunk-boundary state carrying is NOT supported.
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
  not provide without custom cache-indexing code. Rather than block all
  state passing on this (the Mamba-3 era's design), each chunk's
  attention now always runs fresh, real, full attention over just that
  chunk's own tokens (``attn_inference_params=None``, always) -- bounded
  recall within the chunk, no memory of earlier chunks. This is a
  deliberate, documented trade-off, not a bug: the whole point of this
  hybrid design is that the Mamba branch is responsible for compressed
  long-range recall across the full sequence, while the attention branch
  is responsible for precise, uncompressed local recall -- see entry 03's
  decision (a). Passing carried attention state is simply not attempted;
  there is nothing to opt into or reject.
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import (
    SequenceBackbone,
    TimeAwareState,
    resolve_prev_time_stamps,
)
from odyssey.models.embeddings import CachedEHREmbeddings


MambaStateDict = Dict[int, Tuple[torch.Tensor, ...]]


def _make_mamba2_with_state_cls(mamba2_cls: Any) -> Any:
    """Build a ``Mamba2`` subclass that seeds ``initial_states`` for TBTT.

    Deferred (needs the real ``Mamba2`` class from an installed,
    CUDA-built `mamba-ssm`) so this stays a plain function rather than
    a module-level class definition, matching how ``Mamba2``/``MHA`` are
    imported lazily elsewhere in this module.

    Upstream ``Mamba2.forward`` already *writes* the final SSM state back
    into the cache (``return_final_states=ssm_state is not None`` +
    ``ssm_state.copy_(last_state)``), but never *reads* a carried-over
    state back in: its call to ``mamba_chunk_scan_combined`` never passes
    ``initial_states``, even though that kernel genuinely supports it --
    confirmed by reading ``mamba_ssm.modules.mamba2.Mamba2.forward``
    directly (source available locally even without a CUDA build to
    import it). This subclass copies that method's body and adds the one
    missing keyword argument.

    This backbone always constructs an ``InferenceParams`` and passes it
    into every layer, even on the very first chunk of a sequence (state
    is ``None`` from the caller's point of view), so ``ssm_state`` is
    always a real tensor from ``_get_states_from_cache`` -- all zeros on
    a fresh cache entry, real carried values afterward. So a *clone* of
    it is passed as ``initial_states`` on every call, including the
    first; on a fresh cache this is an explicit all-zero tensor rather
    than ``None``, which is the same "start from zero" behavior the
    kernel gives for ``initial_states=None`` -- this class never changes
    behavior relative to upstream for anything a one-shot (non-chunked)
    forward call already did, only adds correct continuation when a
    non-empty cache is carried in. The clone (not the raw cache tensor)
    matters for training: ``mamba_chunk_scan_combined`` saves
    ``initial_states`` for its backward pass, and this same method writes
    the chunk's *final* state back into the cache tensor in place
    (``ssm_state.copy_(last_state)``, upstream behavior, unchanged here)
    -- passing the raw tensor as both the saved-for-backward input and
    the in-place write target corrupts autograd's saved version and
    raises "modified by an inplace operation" on ``.backward()``,
    confirmed by hitting exactly that error before adding the clone.

    ``conv_state`` (the short causal-conv left-context) gets the same
    treatment, in the plain ``nn.Conv1d`` fallback path (``causal_conv1d``
    not installed -- true in every environment this was validated on;
    confirmed via ``mamba_ssm.modules.mamba2.causal_conv1d_fn is None``).
    Upstream only *writes* ``conv_state`` here (for a future ``step()``
    call), never *reads* one back in for this prefill path -- it relies
    on ``nn.Conv1d``'s own built-in zero-padding at the start of every
    call. Initially this was assumed to be a small, ~3-position artifact
    and left unfixed, matching the "working and reliable, not perfect"
    bar for this backbone (entry 03); GPU testing
    (``test_chunked_forward_with_carried_state_matches_one_shot_forward``
    in ``test_mamba2_patch_gpu.py``) showed that assumption was wrong --
    the corrupted conv output at a chunk's first few positions feeds the
    *recurrent* SSM state, so the error persists and decays across the
    whole chunk rather than staying confined to a few positions, large
    enough to fail a 1e-3 equivalence check well into the second half of
    a 16-token chunk. So this class snapshots the incoming ``conv_state``
    (before upstream's write-back overwrites it in place) and manually
    pads the plain-``nn.Conv1d`` branch with it instead of relying on that
    module's zero-padding. Same "purely additive on a fresh cache"
    guarantee as ``ssm_state`` above: an all-zero incoming ``conv_state``
    produces the same padding as before.

    Known, deliberately-not-fixed residual gap: the *fast* Triton
    ``causal_conv1d_fn`` branch (taken only if the ``causal_conv1d``
    package is installed, which it is not in any environment this was
    tested on) still has the original gap -- it is dead code here, so
    fixing it blind, without being able to run it, was judged not worth
    the risk. If that package is ever installed, this class's fast-path
    branch will silently regress to the original chunk-boundary conv
    discontinuity; there is no runtime guard against this, so revisit if
    ``causal_conv1d`` becomes a dependency.
    """

    class Mamba2WithState(mamba2_cls):  # type: ignore[misc]
        """``Mamba2`` with ``initial_states`` wired into the chunk-scan call."""

        def forward(  # noqa: PLR0912, PLR0915
            self,
            u: torch.Tensor,
            seqlen: Optional[int] = None,
            seq_idx: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            inference_params: Any = None,
        ) -> torch.Tensor:
            """Identical to upstream except for the ``initial_states=`` fix."""
            from einops import rearrange  # noqa: PLC0415
            from mamba_ssm.modules import mamba2 as mamba2_module  # noqa: PLC0415

            seqlen_og = seqlen
            if seqlen is None:
                batch, seqlen, _ = u.shape
            else:
                batch_seqlen, _ = u.shape
                batch = batch_seqlen // seqlen

            conv_state, ssm_state = None, None
            incoming_conv_state = None
            if inference_params is not None:
                inference_batch = (
                    cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
                )
                conv_state, ssm_state = self._get_states_from_cache(
                    inference_params, inference_batch
                )
                if inference_params.seqlen_offset > 0:
                    out_step, _, _ = self.step(u, conv_state, ssm_state)
                    return cast(torch.Tensor, out_step)
                # Snapshot the incoming (previous-chunk) conv left-context
                # before the write-back below overwrites conv_state in
                # place with THIS chunk's own trailing values (kept for a
                # future step() call, upstream behavior, unchanged here).
                incoming_conv_state = conv_state.clone() if conv_state is not None else None

            zxbcdt = self.in_proj(u)
            if seqlen_og is not None:
                zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
            A = -torch.exp(self.A_log.float())  # noqa: N806
            dt_limit_kwargs = (
                {}
                if self.dt_limit == (0.0, float("inf"))
                else {"dt_limit": self.dt_limit}
            )

            # Unlike upstream, never take the fast use_mem_eff_path branch
            # (mamba_split_conv1d_scan_combined): that fused kernel has no
            # initial_states parameter at all, so it can't support chunk
            # continuation. This backbone always passes inference_params
            # (even on a state=None first chunk, to obtain ssm_state for
            # writing), so upstream's own `if self.use_mem_eff_path and
            # inference_params is None` guard already skips that branch
            # in every case this backbone exercises -- this is not a
            # behavior change.
            d_mlp = (
                zxbcdt.shape[-1]
                - 2 * self.d_ssm
                - 2 * self.ngroups * self.d_state
                - self.nheads
            ) // 2
            z0, x0, z, xBC, dt = torch.split(  # noqa: N806
                zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.d_ssm,
                    self.d_ssm + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")  # noqa: N806
                    conv_state.copy_(
                        F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                    )
                else:
                    assert mamba2_module.causal_conv1d_varlen_states is not None, (
                        "varlen inference requires causal_conv1d package"
                    )
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = mamba2_module.causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            causal_conv1d_fn = mamba2_module.causal_conv1d_fn
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                if incoming_conv_state is not None and self.d_conv > 1:
                    # Real left-context instead of nn.Conv1d's implicit
                    # zero-padding: without this, every chunk after the
                    # first sees a spurious "sequence start" at its own
                    # position 0, which this backbone's GPU tests showed
                    # is NOT a localized, 3-position artifact -- the
                    # corrupted conv output feeds the recurrent SSM state,
                    # so the error persists and decays across the whole
                    # chunk rather than staying confined to a few
                    # positions. On a fresh (all-zero) cache this is
                    # numerically identical to the zero-padding it
                    # replaces, so this is purely additive for a one-shot
                    # (non-chunked) forward call, same as the SSM-state
                    # fix above.
                    left_context = incoming_conv_state[:, :, -(self.d_conv - 1) :]
                    xBC_padded = torch.cat(  # noqa: N806
                        [left_context, xBC.transpose(1, 2)], dim=-1
                    )
                    xBC = self.act(  # noqa: N806
                        F.conv1d(
                            xBC_padded,
                            self.conv1d.weight,
                            self.conv1d.bias,
                            groups=xBC_padded.shape[1],
                        ).transpose(1, 2)
                    )
                else:
                    xBC = self.act(  # noqa: N806
                        self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[
                            :, : -(self.d_conv - 1)
                        ]
                    )
            else:
                xBC = causal_conv1d_fn(  # noqa: N806
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(  # noqa: N806
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1,
            )
            y = mamba2_module.mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
                if not self.rmsnorm
                else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                # .clone(): initial_states is saved for backward by the
                # kernel's autograd Function; the return_final_states
                # write-back below (ssm_state.copy_(last_state)) mutates
                # this same cache tensor in place. Passing the raw
                # ssm_state here would let that in-place write corrupt
                # what autograd saved, raising "modified by an inplace
                # operation" on .backward() -- confirmed by hitting
                # exactly that error before adding this clone.
                initial_states=ssm_state.clone() if ssm_state is not None else None,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out: torch.Tensor = self.out_proj(y)
            return out

    return Mamba2WithState


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

    The attention side has no cross-chunk state -- see the module
    docstring. ``mamba_states[i]`` is layer ``i``'s
    ``key_value_memory_dict``-style cache.
    """

    __slots__ = ("mamba_states",)

    def __init__(self, mamba_states: MambaStateDict) -> None:
        """Initialize with one Mamba ``key_value_memory_dict``-style cache."""
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
    """A stack of parallel Mamba-2 + attention hybrid blocks."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 768,
        padding_idx: int = 0,
        num_hidden_layers: int = 8,
        mamba_state_size: int = 128,
        mamba_headdim: int = 64,
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
            from mamba_ssm.modules.mamba2 import Mamba2  # noqa: PLC0415
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

        mamba2_with_state_cls = _make_mamba2_with_state_cls(Mamba2)

        def _make_block(layer_idx: int) -> HybridBlock:
            mamba_cls = partial(
                mamba2_with_state_cls,
                layer_idx=layer_idx,
                d_state=mamba_state_size,
                headdim=mamba_headdim,
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

        ``state.recurrent``, if given, must be a :class:`HybridState` (as
        returned by a previous call to this same method): its
        ``mamba_states`` seed the Mamba branch's cache; the attention
        branch always runs fresh over just this chunk (see the module
        docstring). Raises ``NotImplementedError`` if ``reset_mask`` has
        any reset past position 0 (packed multi-patient chunks): the
        kernel exposes a ``cu_seqlens`` varlen path built for exactly
        this, but its batch-dimension semantics haven't been validated
        against this backbone.
        """
        from mamba_ssm.utils.generation import InferenceParams  # noqa: PLC0415

        typed_state: Optional[MambaStateDict] = (
            None
            if state is None
            else cast(HybridState, state.recurrent).mamba_states
        )

        if reset_mask is not None and reset_mask.shape[1] > 1 and reset_mask[:, 1:].any():
            raise NotImplementedError(
                "EHRHybridBackbone does not yet support resets after "
                "position 0 of a chunk (packed multi-patient chunks). The "
                "kernel exposes a cu_seqlens varlen path built for "
                "exactly this on the Mamba side, but its batch-dimension "
                "semantics haven't been validated here; the attention "
                "side has the same gap."
            )

        prev_time_stamps = resolve_prev_time_stamps(state, batch, reset_mask)
        self.embeddings.set_aux_inputs(batch.aux, prev_time_stamps=prev_time_stamps)
        hidden_states = self.embeddings(batch.concept_ids)
        batch_size, seq_len, _ = hidden_states.shape

        mamba_ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch_size)
        if typed_state is not None:
            # Clone rather than reuse the caller's tensors directly: this
            # method's own reset-zeroing loop below, and Mamba2WithState's
            # ssm_state.copy_(last_state) write-back inside the layer
            # loop, both mutate these tensors in place. Without the
            # clone, calling backbone(chunk2, state=state1) would
            # silently corrupt state1 as a side effect -- the same class
            # of aliasing bug already fixed for the ssm_state/
            # initial_states pair and the conv_state snapshot in
            # _make_mamba2_with_state_cls, just one layer further out.
            mamba_ip.key_value_memory_dict = {
                layer_idx: tuple(tensor.clone() for tensor in cached)
                for layer_idx, cached in typed_state.items()
            }
            if reset_mask is not None:
                reset_rows = reset_mask[:, 0]
                if reset_rows.any():
                    for cached in mamba_ip.key_value_memory_dict.values():
                        for tensor in cached:
                            tensor[reset_rows] = 0

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
                mamba_states=cast(MambaStateDict, mamba_ip.key_value_memory_dict)
            ),
            prev_time_stamps=batch.aux.time_stamps[:, -1],
        )
        return result, new_state
