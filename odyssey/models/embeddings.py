"""Clinical embedding layers, backbone-agnostic.

Fuses token identity with the auxiliary structure of a patient event
sequence (token type, time since previous event, patient age, visit
order/segment) into a single embedding consumed by any sequence backbone.
"""

from typing import Optional

import torch
from torch import nn

from odyssey.data.types import AuxiliaryInputs


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

    def forward(
        self, time_stamps: torch.Tensor, prev_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply time embedding to the input time stamps.

        ``prev_value`` is ``(batch,)``: the true value immediately
        preceding ``time_stamps[:, 0]``, from a previous chunk in the same
        lane (see ``odyssey/data/streaming.py``). Without it, position 0's
        delta is zeroed, correct for a genuinely fresh sequence but wrong
        for a chunk continuing an earlier one -- confirmed on real
        hardware to produce materially different hidden states (up to
        ~30% relative) purely from this, independent of any actual
        recurrent-state bug. Callers continuing a lane must zero
        ``prev_value`` at real reset positions themselves (e.g. by setting
        it equal to that row's own ``time_stamps[:, 0]``, which makes the
        computed delta ``0`` the same way omitting it would).
        """
        if self.is_time_delta:
            if prev_value is not None:
                first_delta = time_stamps[:, 0:1] - prev_value.unsqueeze(-1)
            else:
                first_delta = time_stamps[:, 0:1] * 0
            time_stamps = torch.cat(
                (first_delta, time_stamps[:, 1:] - time_stamps[:, :-1]),
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


class ClinicalEventEmbeddings(nn.Module):
    """Fuses token identity with clinical sequence structure.

    Parameters
    ----------
    vocab_size : int
        Size of the event-token vocabulary.
    hidden_size : int
        Output embedding dimensionality (must match the backbone's input).
    padding_idx : int
        Token id used for padding.
    type_vocab_size : int
        Number of distinct token types (e.g. diagnosis/med/lab/procedure).
    max_num_visits : int
        Maximum number of visits per patient the visit-order embedding
        table needs to cover.
    time_embeddings_size : int
        Dimensionality of each sinusoidal time/age embedding, before
        projection back to ``hidden_size``.
    visit_order_size : int
        Number of distinct visit segment values (e.g. first/middle/last).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int,
        *,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """Initialize the clinical event embeddings."""
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

    def forward(
        self,
        input_ids: torch.Tensor,
        aux: AuxiliaryInputs,
        prev_time_stamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse token identity with clinical sequence structure.

        ``prev_time_stamps`` is ``(batch,)``, see :meth:`TimeEmbeddingLayer.forward`.
        Ages are not delta-based (absolute age at each event), so they are
        unaffected by chunk boundaries and need no equivalent parameter.
        """
        word_embeds = self.word_embeddings(input_ids)
        time_embeds = self.time_embeddings(aux.time_stamps, prev_value=prev_time_stamps)
        age_embeds = self.age_embeddings(aux.ages)
        visit_seg_embeds = self.visit_segment_embeddings(aux.visit_segments)
        visit_order_embeds = self.visit_order_embeddings(aux.visit_orders)
        token_type_embeds = self.token_type_embeddings(aux.type_ids)

        fused = torch.cat([word_embeds, time_embeds, age_embeds], dim=-1)
        fused = self.tanh(self.scale_back_concat_layer(fused))
        embeddings = fused + token_type_embeds + visit_order_embeds + visit_seg_embeds

        result: torch.Tensor = self.LayerNorm(self.dropout(embeddings))
        return result


class CachedEHREmbeddings(nn.Module):
    """Bridges :class:`ClinicalEventEmbeddings` into single-argument backbones.

    Some backbones (e.g. ``mamba_ssm``'s ``MixerModel``) call
    ``self.embedding(input_ids)`` with a single argument. This wrapper
    caches the auxiliary clinical inputs so they're available when the
    backbone invokes the embedding call it owns.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the cached EHR embeddings."""
        super().__init__()
        self.embeddings = ClinicalEventEmbeddings(*args, **kwargs)  # type: ignore[arg-type]
        self._aux: Optional[AuxiliaryInputs] = None
        self._prev_time_stamps: Optional[torch.Tensor] = None

    def set_aux_inputs(
        self, aux: AuxiliaryInputs, prev_time_stamps: Optional[torch.Tensor] = None
    ) -> None:
        """Cache auxiliary clinical inputs before the backbone forward call."""
        self._aux = aux
        self._prev_time_stamps = prev_time_stamps

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Produce EHR-enriched embeddings, consuming cached auxiliary inputs."""
        if self._aux is None:
            raise RuntimeError("set_aux_inputs must be called before forward")
        embeddings: torch.Tensor = self.embeddings(
            input_ids, self._aux, prev_time_stamps=self._prev_time_stamps
        )
        self._aux = None
        self._prev_time_stamps = None
        return embeddings
