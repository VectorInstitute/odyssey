"""GPU-only tests for the real EHR-Mamba3 backbone.

`mamba-ssm` requires a CUDA build and its kernels only run on CUDA tensors,
so these tests are meaningless (and mostly can't even import) anywhere
else. They auto-skip unless both `mamba-ssm` is installed and a CUDA
device is visible -- exercised on a GPU host, not local/CPU CI.
"""

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch  # noqa: E402
from odyssey.models.backbones.mamba3 import EHRMamba3Backbone  # noqa: E402
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel  # noqa: E402


VOCAB_SIZE = 40
HIDDEN_SIZE = 64  # must be divisible by headdim
NUM_CONCEPTS = 4
RESIDUAL_DIM = 6
PADDING_IDX = 0


def _make_batch(batch: int, seq_len: int, device: str) -> ClinicalSequenceBatch:
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, VOCAB_SIZE, (batch, seq_len), device=device),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len), device=device),
            time_stamps=torch.rand(batch, seq_len, device=device) * 100,
            ages=torch.rand(batch, seq_len, device=device) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len), device=device),
            visit_segments=torch.randint(0, 3, (batch, seq_len), device=device),
        ),
    )


@cuda_required
def test_ehr_mamba3_backbone_forward_shape() -> None:
    backbone = EHRMamba3Backbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        state_size=16,
        num_hidden_layers=2,
        headdim=32,
        # Mamba3's MIMO kernel recommends chunk_size = 64 / mimo_rank
        # (mimo_rank defaults to 4); untested headdim/chunk_size
        # combinations can fail in the TileLang kernel's warp partitioning.
        chunk_size=16,
    ).cuda()
    batch = _make_batch(batch=2, seq_len=10, device="cuda")

    hidden_states = backbone(batch)

    assert hidden_states.shape == (2, 10, HIDDEN_SIZE)
    assert hidden_states.is_cuda


@cuda_required
def test_ehr_mamba3_backbone_through_concept_bottleneck_end_to_end() -> None:
    """Confirm EHRMamba3Backbone satisfies the SequenceBackbone contract.

    This is what CPU tests (TinyGRUBackbone) can't cover: the real
    backbone, through the full concept-bottleneck model, with a real
    backward pass against actual mamba_ssm kernels.
    """
    backbone = EHRMamba3Backbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
        state_size=16,
        num_hidden_layers=2,
        headdim=32,
        # Mamba3's MIMO kernel recommends chunk_size = 64 / mimo_rank
        # (mimo_rank defaults to 4); untested headdim/chunk_size
        # combinations can fail in the TileLang kernel's warp partitioning.
        chunk_size=16,
    )
    model = ConceptBottleneckSequenceModel(
        backbone=backbone,
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        residual_dim=RESIDUAL_DIM,
        padding_idx=PADDING_IDX,
    ).cuda()

    batch = _make_batch(batch=3, seq_len=12, device="cuda")
    concept_labels = torch.randint(0, 2, (3, NUM_CONCEPTS), device="cuda").float()

    total, components = model.compute_loss(batch, concept_labels)
    assert torch.isfinite(total)

    total.backward()
    assert model.bottleneck.concept_proj.weight.grad is not None
    assert torch.any(model.bottleneck.concept_proj.weight.grad != 0)
    for name in ("task_loss", "concept_loss", "orthogonality_loss"):
        assert torch.isfinite(components[name])
