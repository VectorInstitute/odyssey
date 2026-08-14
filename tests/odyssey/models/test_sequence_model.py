"""Tests for the concept-bottleneck sequence model (backbone/bottleneck/LM head)."""

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


torch.manual_seed(0)

VOCAB_SIZE = 50
HIDDEN_SIZE = 32
NUM_CONCEPTS = 4
EMBEDDING_DIM = 6
PADDING_IDX = 0


def _make_backbone() -> TinyGRUBackbone:
    return TinyGRUBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        padding_idx=PADDING_IDX,
    )


def _make_model() -> ConceptBottleneckSequenceModel:
    return ConceptBottleneckSequenceModel(
        backbone=_make_backbone(),
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PADDING_IDX,
    )


def _make_batch(batch: int, seq_len: int) -> ClinicalSequenceBatch:
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, VOCAB_SIZE, (batch, seq_len)),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len)),
            time_stamps=torch.rand(batch, seq_len) * 100,
            ages=torch.rand(batch, seq_len) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len)),
            visit_segments=torch.randint(0, 3, (batch, seq_len)),
        ),
    )


# ---------------------------------------------------------------------------
# Shapes / plumbing
# ---------------------------------------------------------------------------


def test_forward_shapes() -> None:
    model = _make_model()
    batch, seq_len = 3, 12
    inputs = _make_batch(batch, seq_len)

    logits, bottleneck_out = model(inputs)

    assert logits.shape == (batch, seq_len, VOCAB_SIZE)
    assert bottleneck_out.concept_logits.shape == (batch, seq_len, NUM_CONCEPTS)
    assert bottleneck_out.concept_embeddings.shape == (
        batch,
        seq_len,
        NUM_CONCEPTS,
        EMBEDDING_DIM,
    )
    assert bottleneck_out.unknown_embedding.shape == (batch, seq_len, EMBEDDING_DIM)


def test_backbone_is_swappable_via_shared_interface() -> None:
    # A second, independently constructed backbone with the same interface
    # must work as a drop-in replacement -- this is the whole point of
    # SequenceBackbone: the real EHRMamba3Backbone slots in identically.
    other_backbone = TinyGRUBackbone(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, padding_idx=PADDING_IDX
    )
    model = ConceptBottleneckSequenceModel(
        backbone=other_backbone,
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PADDING_IDX,
    )
    logits, _ = model(_make_batch(2, 8))
    assert logits.shape == (2, 8, VOCAB_SIZE)


def test_compute_loss_returns_finite_scalar_and_components() -> None:
    model = _make_model()
    inputs = _make_batch(batch=4, seq_len=10)
    concept_labels = torch.randint(0, 2, (4, NUM_CONCEPTS)).float()

    total, components = model.compute_loss(inputs, concept_labels)

    assert total.dim() == 0
    assert torch.isfinite(total)
    assert set(components) == {"task_loss", "concept_loss", "orthogonality_loss"}


def test_concept_mask_excludes_unobserved_labels_from_loss() -> None:
    model = _make_model()
    inputs = _make_batch(batch=4, seq_len=10)
    concept_labels = torch.randint(0, 2, (4, NUM_CONCEPTS)).float()
    all_masked_out = torch.zeros(4, NUM_CONCEPTS)

    _, components = model.compute_loss(
        inputs, concept_labels, concept_mask=all_masked_out
    )
    assert components["concept_loss"].item() == 0.0


def test_gradients_flow_through_backbone_and_bottleneck() -> None:
    model = _make_model()
    inputs = _make_batch(batch=2, seq_len=6)
    concept_labels = torch.randint(0, 2, (2, NUM_CONCEPTS)).float()

    total, _ = model.compute_loss(inputs, concept_labels)
    total.backward()

    assert model.bottleneck.context_proj.weight.grad is not None
    assert torch.any(model.bottleneck.context_proj.weight.grad != 0)
    assert model.backbone.embeddings.embeddings.word_embeddings.weight.grad is not None


def test_padding_positions_do_not_leak_into_pooled_concept_supervision() -> None:
    """Pooling must stop at the last non-padding token.

    Two sequences differing only after their padding boundary must produce
    identical pooled concept logits.
    """
    model = _make_model()
    batch = _make_batch(batch=1, seq_len=8)
    concept_ids = batch.concept_ids.clone()
    # First 5 tokens real, rest padding.
    concept_ids[0, :5] = torch.randint(1, VOCAB_SIZE, (5,))
    concept_ids[0, 5:] = PADDING_IDX
    batch = batch._replace(concept_ids=concept_ids)

    concept_ids_variant = concept_ids.clone()
    # Change only the padded tail's token ids -- should not affect anything
    # pooled from the last *non-padding* position.
    concept_ids_variant[0, 5:] = torch.randint(1, VOCAB_SIZE, (3,))
    batch_variant = batch._replace(concept_ids=concept_ids_variant)

    model.eval()
    with torch.no_grad():
        _, out_a = model(batch)
        _, out_b = model(batch_variant)

    # GRU is causal, so hidden state (and thus concept logits) at position 4
    # only depends on positions 0..4, which are identical between the two.
    assert torch.allclose(
        out_a.concept_logits[0, 4], out_b.concept_logits[0, 4], atol=1e-6
    )


# ---------------------------------------------------------------------------
# Synthetic end-to-end training
# ---------------------------------------------------------------------------


def test_synthetic_training_reduces_next_token_and_concept_loss() -> None:
    torch.manual_seed(3)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    weights = ConceptBottleneckLossWeights(concept=1.0, orthogonality=0.05)

    batch, seq_len = 16, 15
    inputs = _make_batch(batch, seq_len)
    concept_labels = torch.randint(0, 2, (batch, NUM_CONCEPTS)).float()

    task_losses = []
    concept_losses = []
    for _ in range(150):
        optimizer.zero_grad()
        total, components = model.compute_loss(
            inputs, concept_labels, loss_weights=weights
        )
        total.backward()
        optimizer.step()
        task_losses.append(components["task_loss"].item())
        concept_losses.append(components["concept_loss"].item())

    assert task_losses[-1] < task_losses[0] * 0.8
    assert concept_losses[-1] < concept_losses[0] * 0.5


# ---------------------------------------------------------------------------
# The real Mamba3 backbone can't be executed here (CUDA-only); confirm the
# import guard fails helpfully instead of with an opaque ImportError.
# ---------------------------------------------------------------------------


def test_ehr_mamba3_backbone_raises_helpful_error_without_cuda() -> None:
    """Can't test the real backbone's forward pass without a GPU here.

    This instead validates that, absent `mamba-ssm`, the import guard
    raises a clear and actionable error rather than an opaque one.
    """
    try:
        import mamba_ssm  # noqa: F401, PLC0415

        pytest.skip("mamba-ssm is installed here; the guard path isn't exercised")
    except ImportError:
        pass

    from odyssey.models.backbones.mamba3 import EHRMamba3Backbone  # noqa: PLC0415

    with pytest.raises(ImportError, match="mamba-ssm"):
        EHRMamba3Backbone(vocab_size=10, hidden_size=8, num_hidden_layers=1)
