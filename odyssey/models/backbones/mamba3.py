"""EHR-Mamba3 backbone: ``mamba_ssm.MambaLMHeadModel`` with a Mamba-3 SSM config.

Requires `mamba-ssm`, which needs a CUDA/`nvcc` build:

    uv sync --extra cuda --no-build-isolation

The import is deferred to ``__init__`` so this module — and the rest of
``odyssey.models`` — stays importable and type-checkable on machines
without CUDA (e.g. local Mac development). Instantiating
:class:`EHRMamba3Backbone` itself still requires `mamba-ssm` to be
installed; use :class:`~odyssey.models.backbones.tiny_gru.TinyGRUBackbone`
for CPU development instead.
"""

from typing import Any, Dict

import torch

from odyssey.data.types import ClinicalSequenceBatch
from odyssey.models.backbones.base import SequenceBackbone
from odyssey.models.embeddings import CachedEHREmbeddings


class EHRMamba3Backbone(SequenceBackbone):
    """Mamba-3 SSM backbone enriched with clinical embeddings."""

    def __init__(  # noqa: PLR0917
        self,
        vocab_size: int,
        hidden_size: int = 768,
        padding_idx: int = 0,
        state_size: int = 128,
        num_hidden_layers: int = 32,
        d_intermediate: int = 0,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 256,
        **embedding_kwargs: object,
    ) -> None:
        """Initialize the EHR-Mamba3 backbone."""
        try:
            # Deferred: mamba-ssm needs CUDA and isn't installed on CPU-only
            # dev machines. See the module docstring.
            from mamba_ssm.models.config_mamba import (  # noqa: PLC0415
                MambaConfig as MambaSsmConfig,
            )
            from mamba_ssm.models.mixer_seq_simple import (  # noqa: PLC0415
                MambaLMHeadModel,
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

        ssm_cfg: Dict[str, Any] = {
            "layer": "Mamba3",
            "d_state": state_size,
            "headdim": headdim,
            "is_mimo": is_mimo,
            "mimo_rank": mimo_rank,
            "chunk_size": chunk_size,
        }
        backbone_cfg = MambaSsmConfig(
            d_model=hidden_size,
            d_intermediate=d_intermediate,
            n_layer=num_hidden_layers,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
        )
        self.model = MambaLMHeadModel(backbone_cfg)

        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            **embedding_kwargs,
        )
        # Replace the backbone's plain Embedding with the EHR-enriched one.
        self.model.backbone.embedding = self.embeddings

    def forward(self, batch: ClinicalSequenceBatch) -> torch.Tensor:
        """Return hidden states of shape ``(batch, seq_len, hidden_size)``."""
        self.embeddings.set_aux_inputs(batch.aux)
        return self.model.backbone(batch.concept_ids)  # type: ignore[no-any-return]
