"""Shared synthetic batches and backbones for the hybrid-backbone tests."""

import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.hybrid import EHRHybridBackbone


VOCAB_SIZE = 40
HIDDEN_SIZE = 64  # divisible by both MAMBA_HEADDIM and ATTN_NUM_HEADS


def make_batch(
    batch: int, seq_len: int, device: str = "cpu", seed: int = 0
) -> ClinicalSequenceBatch:
    """Return a random batch with increasing time stamps, as real records have."""
    gen = torch.Generator().manual_seed(seed)

    def ints(high: int) -> torch.Tensor:
        return torch.randint(0, high, (batch, seq_len), generator=gen)

    times = torch.rand(batch, seq_len, generator=gen).cumsum(dim=1) * 10
    aux = AuxiliaryInputs(
        type_ids=ints(9),
        time_stamps=times.double(),
        ages=torch.rand(batch, seq_len, generator=gen) * 90,
        visit_orders=ints(5),
        visit_segments=ints(3),
    )
    batch_cpu = ClinicalSequenceBatch(concept_ids=ints(VOCAB_SIZE - 1) + 1, aux=aux)
    return ClinicalSequenceBatch(
        concept_ids=batch_cpu.concept_ids.to(device),
        aux=AuxiliaryInputs(
            *(t.to(device) if t is not None else None for t in batch_cpu.aux)
        ),
    )


def make_backbone(
    chunk_size: int = 16, num_hidden_layers: int = 2
) -> EHRHybridBackbone:
    """Return a small hybrid backbone; the environment picks the mixers."""
    return EHRHybridBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=num_hidden_layers,
        mamba_state_size=16,
        mamba_headdim=32,
        mamba_chunk_size=chunk_size,
        attn_num_heads=8,
    )
