import math

import torch
import torch.nn as nn


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(
            self,
            embedding_size: int,
            is_time_delta: bool = False
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> torch.Tensor:
        """Applies time embedding to the input time stamps."""
        if self.is_time_delta:
            # If the time_stamps represent time deltas, we calculate the deltas.
            # This is equivalent to the difference between consecutive elements.
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]), dim=-1
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi

        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Embedding layer for visit segments."""

    def __init__(
            self,
            visit_order_size: int,
            embedding_size: int,
    ):
        super().__init__()
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.visit_order_size, self.embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> torch.Tensor:
        """Applies visit embedding to the input visit segments."""
        return self.embedding(visit_segments)


class ConceptEmbedding(nn.Module):
    """Embedding layer for event concepts."""

    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            padding_idx: int = None,
    ):
        super(ConceptEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=padding_idx)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies concept embedding to the input concepts."""
        return self.embedding(inputs)


class PositionalEmbedding(nn.Module):
    """Positional embedding layer."""

    def __init__(self, embedding_size, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_size, 2).float() \
                    * -(math.log(10000.0) / embedding_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, visit_orders: torch.Tensor) -> torch.Tensor:
        """Applies positional embedding to the input visit orders."""
        first_visit_concept_orders = visit_orders[:, 0:1]
        normalized_visit_orders = \
            torch.clamp(visit_orders - first_visit_concept_orders, 0, self.pe.size(0) - 1)
        return self.pe[normalized_visit_orders]


class Embeddings(nn.Module):
    """Embeddings for CEHR-BERT."""

    def __init__(
            self,
            vocab_size: int,
            embedding_size: int = 128,
            time_embedding_size: int = 16,
            visit_order_size: int = 3,
            max_len: int = 512,
            layer_norm_eps: float = 1e-12,
            dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.concept_embedding = ConceptEmbedding(
            num_embeddings=vocab_size, embedding_size=embedding_size
        )
        self.time_embedding = TimeEmbeddingLayer(embedding_size=time_embedding_size)
        self.age_embedding = TimeEmbeddingLayer(embedding_size=time_embedding_size)
        self.positional_embedding = PositionalEmbedding(
            embedding_size=time_embedding_size, max_len=max_len
        )
        self.visit_embedding = VisitEmbedding(
            visit_order_size=visit_order_size, embedding_size=embedding_size
        )
        self.scale_back_concat_layer = nn.Linear(
            embedding_size + 3 * time_embedding_size, embedding_size
        )  # Assuming 4 input features are concatenated
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
            self,
            concept_ids: torch.Tensor,
            time_stamps: torch.Tensor,
            ages: torch.Tensor,
            visit_orders: torch.Tensor,
            visit_segments: torch.Tensor,
    ) -> torch.Tensor:
        """Applies embeddings to the input features."""
        concept_embed = self.concept_embedding(concept_ids)
        time_embed = self.time_embedding(time_stamps)
        age_embed = self.age_embedding(ages)
        positional_embed = self.positional_embedding(visit_orders)
        visit_segment_embed = self.visit_embedding(visit_segments)

        embeddings = torch.cat((concept_embed, time_embed, age_embed, positional_embed), dim=-1)
        embeddings = self.tanh(self.scale_back_concat_layer(embeddings))
        embeddings = visit_segment_embed + embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
