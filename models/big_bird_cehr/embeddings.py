import math

import torch
from torch import nn


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
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
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
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
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_size,
            padding_idx=padding_idx,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies concept embedding to the input concepts."""
        return self.embedding(inputs)


class PositionalEmbedding(nn.Module):
    """Positional embedding layer."""

    def __init__(self, embedding_size, max_len=2048):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embedding_size, 2).float()
            * -(math.log(10000.0) / embedding_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, visit_orders: torch.Tensor) -> torch.Tensor:
        """Applies positional embedding to the input visit orders."""
        first_visit_concept_orders = visit_orders[:, 0:1]
        normalized_visit_orders = torch.clamp(
            visit_orders - first_visit_concept_orders,
            0,
            self.pe.size(0) - 1,
        )
        return self.pe[normalized_visit_orders]


class Embeddings(nn.Module):
    """Embeddings for CEHR-BERT."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 128,
        time_embedding_size: int = 16,
        type_vocab_size: int = 8,
        visit_order_size: int = 3,
        max_len: int = 2048,
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 0.1,
        padding_idx: int = 1,
    ):
        super().__init__()
        self.concept_embedding = ConceptEmbedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            padding_idx=padding_idx,
        )
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        self.time_embedding = TimeEmbeddingLayer(
            embedding_size=time_embedding_size,
            is_time_delta=True,
        )
        self.age_embedding = TimeEmbeddingLayer(embedding_size=time_embedding_size)
        self.positional_embedding = PositionalEmbedding(
            embedding_size=embedding_size,
            max_len=max_len,
        )
        self.visit_embedding = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=embedding_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            embedding_size + 2 * time_embedding_size,
            embedding_size,
        )  # Assuming 4 input features are concatenated
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        concept_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> torch.Tensor:
        """Applies embeddings to the input features."""
        concept_embed = self.concept_embedding(concept_ids)
        # type_embed = self.token_type_embeddings(type_ids)
        time_embed = self.time_embedding(time_stamps)
        age_embed = self.age_embedding(ages)
        # positional_embed = self.positional_embedding(visit_orders)
        visit_segment_embed = self.visit_embedding(visit_segments)

        embeddings = torch.cat((concept_embed, time_embed, age_embed), dim=-1)
        embeddings = self.tanh(self.scale_back_concat_layer(embeddings))
        # embeddings = embeddings + type_embed + positional_embed + visit_segment_embed
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)

        return embeddings


class BigBirdEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config, cehr_embedding):
        super().__init__()
        self.word_embeddings = None
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.visit_segment_embed = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config,
            "position_embedding_type",
            "absolute",
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )
        # End copy

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :,
                past_key_values_length : seq_length + past_key_values_length,
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually
        # occurs when its auto-generated, registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0],
                    seq_length,
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings
