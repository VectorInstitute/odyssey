"""Embedding layers for the EHR-Mamba3 model."""

from typing import Any, Optional

import torch
from torch import nn


class TimeEmbeddingLayer(nn.Module):
    """Sinusoidal embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        """Initialize the time embedding layer."""
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> torch.Tensor:
        """Apply time embedding to the input time stamps."""
        if self.is_time_delta:
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi
        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Learned embedding layer for visit segments."""

    def __init__(self, visit_order_size: int, embedding_size: int):
        """Initialize the visit embedding layer."""
        super().__init__()
        self.embedding = nn.Embedding(visit_order_size, embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> torch.Tensor:
        """Apply visit embedding to the input visit segments."""
        return self.embedding(visit_segments)  # type: ignore[no-any-return]


class CachedEHREmbeddings(nn.Module):
    """Bridges EHR-specific embeddings into the mamba_ssm backbone.

    The mamba_ssm MixerModel calls ``self.embedding(input_ids)`` with a single
    argument.  This wrapper caches the auxiliary clinical inputs (token types,
    time stamps, ages, visit orders/segments) so they are available when the
    backbone invokes the embedding call.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """Initialize the cached EHR embeddings."""
        super().__init__()

        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.visit_order_embeddings = nn.Embedding(max_num_visits, hidden_size)
        self.visit_segment_embeddings = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=hidden_size,
        )
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embeddings = TimeEmbeddingLayer(embedding_size=time_embeddings_size)
        self.scale_back_concat_layer = nn.Linear(
            hidden_size + 2 * time_embeddings_size, hidden_size
        )
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Auxiliary inputs cached before each backbone forward call
        self._type_ids: Optional[torch.Tensor] = None
        self._time_stamps: Optional[torch.Tensor] = None
        self._ages: Optional[torch.Tensor] = None
        self._visit_orders: Optional[torch.Tensor] = None
        self._visit_segments: Optional[torch.Tensor] = None

    def set_aux_inputs(
        self,
        type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> None:
        """Cache auxiliary clinical inputs before the backbone forward call."""
        self._type_ids = type_ids
        self._time_stamps = time_stamps
        self._ages = ages
        self._visit_orders = visit_orders
        self._visit_segments = visit_segments

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Produce EHR-enriched embeddings, consuming cached auxiliary inputs."""
        word_embeds = self.word_embeddings(input_ids)

        if self._type_ids is not None:
            time_embeds = self.time_embeddings(self._time_stamps)
            age_embeds = self.age_embeddings(self._ages)
            visit_seg_embeds = self.visit_segment_embeddings(self._visit_segments)
            visit_order_embeds = self.visit_order_embeddings(self._visit_orders)
            token_type_embeds = self.token_type_embeddings(self._type_ids)

            # Clear cache
            self._type_ids = self._time_stamps = self._ages = None
            self._visit_orders = self._visit_segments = None

            fused = torch.cat([word_embeds, time_embeds, age_embeds], dim=-1)
            fused = self.tanh(self.scale_back_concat_layer(fused))
            embeddings = (
                fused + token_type_embeds + visit_order_embeds + visit_seg_embeds
            )
        else:
            embeddings = word_embeds

        return self.LayerNorm(self.dropout(embeddings))
