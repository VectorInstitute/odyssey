"""EHR-Mamba3 backbone: a stack of real ``mamba_ssm`` Mamba-3 blocks.

Requires `mamba-ssm`, which needs a CUDA/`nvcc` build:

    uv sync --extra cuda --no-build-isolation

The import is deferred to ``__init__`` so this module -- and the rest of
``odyssey.models`` -- stays importable and type-checkable on machines
without CUDA (e.g. local Mac development). Instantiating
:class:`EHRMamba3Backbone` itself still requires `mamba-ssm` to be
installed; use :class:`~odyssey.models.backbones.tiny_gru.TinyGRUBackbone`
for CPU development instead.

Note: ``mamba_ssm``'s high-level ``MambaLMHeadModel``/``MixerModel``
convenience wrapper only dispatches ``ssm_cfg={"layer": ...}`` to Mamba1 or
Mamba2 (as of mamba-ssm 2.3.2) even though the package ships real Mamba-3
kernels (``mamba_ssm.modules.mamba3.Mamba3``) -- that dispatcher just
hasn't been updated yet. So this backbone builds the block stack directly,
mirroring exactly what ``MixerModel`` does internally (see
``mamba_ssm.models.mixer_seq_simple.create_block``/``MixerModel``), with
``Mamba3`` as the mixer instead of going through the string-based dispatch.

Chunk-boundary state passing (for :mod:`odyssey.data.streaming`'s TBTT
training) uses ``mamba_ssm.utils.generation.InferenceParams`` and each
``Mamba3`` layer's ``key_value_memory_dict`` cache, the same mechanism the
library uses for autoregressive decoding -- confirmed by reading
``mamba_ssm.modules.mamba3.Mamba3.forward``/``_get_states_from_cache``
directly (source available locally even without a CUDA build to import
it), not assumed. A carried-over state dict is reused as-is when present
(``_get_states_from_cache`` only zeros it if asked to); this backbone
zeros the batch rows that need a reset before the layer call, and relies
on ``inference_params.seqlen_offset`` staying ``0`` so every call takes the
full multi-token kernel path, never the single-token autoregressive
``step()`` path.

Resets *after* position 0 of a chunk (a packed multi-patient chunk's
mid-chunk patient boundary) are not handled here: the kernel exposes a
``cu_seqlens`` varlen path built for exactly this
(``mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen``), but its
batch-dimension semantics haven't been validated against this backbone
without CUDA access, and ``seq_idx`` -- a same-purpose-looking parameter
``Mamba3.forward`` accepts -- is never actually read inside the method, so
it does nothing in this version despite being present in the signature.
Passing a ``reset_mask`` with any reset after position 0 raises rather
than silently computing the wrong thing.
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import nn

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone
from odyssey.models.embeddings import CachedEHREmbeddings


StateDict = Dict[int, Tuple[torch.Tensor, ...]]


class EHRMamba3Backbone(SequenceBackbone):
    """A stack of Mamba-3 blocks enriched with clinical embeddings."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 768,
        padding_idx: int = 0,
        state_size: int = 128,
        num_hidden_layers: int = 32,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 256,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        **embedding_kwargs: object,
    ) -> None:
        """Initialize the EHR-Mamba3 backbone."""
        try:
            # Deferred: mamba-ssm needs CUDA and isn't installed on CPU-only
            # dev machines. See the module docstring.
            from mamba_ssm.modules.block import Block  # noqa: PLC0415
            from mamba_ssm.modules.mamba3 import Mamba3  # noqa: PLC0415
            from mamba_ssm.ops.triton.layer_norm import (  # noqa: PLC0415
                RMSNorm,
                layer_norm_fn,
            )
        except ImportError as exc:
            raise ImportError(
                "EHRMamba3Backbone requires mamba-ssm, which needs a CUDA "
                "build: `uv sync --extra cuda --no-build-isolation`. Use "
                "odyssey.models.backbones.tiny_gru.TinyGRUBackbone for "
                "CPU development instead."
            ) from exc

        super().__init__()
        self.hidden_size = hidden_size
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self._layer_norm_fn = layer_norm_fn

        ssm_cfg: Dict[str, Any] = {
            "d_state": state_size,
            "headdim": headdim,
            "is_mimo": is_mimo,
            "mimo_rank": mimo_rank,
            "chunk_size": chunk_size,
        }

        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )

        def _make_block(layer_idx: int) -> nn.Module:
            mixer_cls = partial(Mamba3, layer_idx=layer_idx, **ssm_cfg)
            norm_cls = partial(RMSNorm, eps=norm_epsilon)
            block: nn.Module = Block(
                hidden_size,
                mixer_cls,
                nn.Identity,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )
            return block

        self.layers = nn.ModuleList([_make_block(i) for i in range(num_hidden_layers)])
        self.norm_f = RMSNorm(hidden_size, eps=norm_epsilon)

    def forward(
        self,
        batch: ClinicalSequenceBatch,
        state: Optional[object] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Return ``(hidden_states, new_state)``; see the base class docstring.

        ``state``, if given, must be a ``StateDict`` (as returned by a
        previous call to this same method). Raises ``NotImplementedError``
        if ``reset_mask`` has any reset past position 0 -- see the module
        docstring's "Resets after position 0" section for why that path
        isn't wired up yet.
        """
        from mamba_ssm.utils.generation import InferenceParams  # noqa: PLC0415

        typed_state: Optional[StateDict] = (
            None if state is None else cast(StateDict, state)
        )

        if reset_mask is not None and reset_mask.shape[1] > 1 and reset_mask[:, 1:].any():
            raise NotImplementedError(
                "EHRMamba3Backbone does not yet support resets after "
                "position 0 of a chunk (packed multi-patient chunks). See "
                "this module's docstring for the cu_seqlens path this "
                "needs and why it isn't wired up yet."
            )

        self.embeddings.set_aux_inputs(batch.aux)
        hidden_states = self.embeddings(batch.concept_ids)
        batch_size, seq_len, _ = hidden_states.shape

        inference_params = InferenceParams(
            max_seqlen=seq_len, max_batch_size=batch_size
        )
        if typed_state is not None:
            inference_params.key_value_memory_dict = typed_state
            if reset_mask is not None:
                reset_rows = reset_mask[:, 0]
                if reset_rows.any():
                    for cached in typed_state.values():
                        for tensor in cached:
                            tensor[reset_rows] = 0

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            result: torch.Tensor = self.norm_f(
                residual.to(dtype=self.norm_f.weight.dtype)
            )
            return result, inference_params.key_value_memory_dict

        normed: torch.Tensor = self._layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=True,
        )
        return normed, inference_params.key_value_memory_dict
