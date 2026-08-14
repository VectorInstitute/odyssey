"""Tests for the concept bottleneck layer."""

import pytest
import torch
from torch import nn

from odyssey.models.concept_bottleneck import (
    ConceptBottleneck,
    ConceptBottleneckLossWeights,
    combined_loss,
    concept_loss,
    orthogonality_loss,
)


torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Shape / plumbing
# ---------------------------------------------------------------------------


def test_forward_shapes_sequence_input() -> None:
    batch, seq_len, hidden, num_concepts, embedding_dim = 4, 10, 32, 5, 8
    layer = ConceptBottleneck(hidden, num_concepts, embedding_dim)
    hidden_states = torch.randn(batch, seq_len, hidden)

    out = layer(hidden_states)

    assert out.concept_logits.shape == (batch, seq_len, num_concepts)
    assert out.concept_probs.shape == (batch, seq_len, num_concepts)
    assert out.concept_embeddings.shape == (batch, seq_len, num_concepts, embedding_dim)
    assert out.unknown_embedding.shape == (batch, seq_len, embedding_dim)
    assert out.bottleneck.shape == (batch, seq_len, (num_concepts + 1) * embedding_dim)


def test_forward_shapes_pooled_input() -> None:
    batch, hidden, num_concepts, embedding_dim = 4, 16, 3, 4
    layer = ConceptBottleneck(hidden, num_concepts, embedding_dim)
    hidden_states = torch.randn(batch, hidden)

    out = layer(hidden_states)

    assert out.concept_logits.shape == (batch, num_concepts)
    assert out.concept_embeddings.shape == (batch, num_concepts, embedding_dim)
    assert out.unknown_embedding.shape == (batch, embedding_dim)
    assert out.bottleneck.shape == (batch, (num_concepts + 1) * embedding_dim)


def test_concept_probs_are_valid_probabilities() -> None:
    layer = ConceptBottleneck(hidden_size=8, num_concepts=4, embedding_dim=4)
    out = layer(torch.randn(6, 8))
    assert torch.all(out.concept_probs >= 0.0)
    assert torch.all(out.concept_probs <= 1.0)


def test_bottleneck_is_concat_of_concept_and_unknown_embeddings() -> None:
    layer = ConceptBottleneck(hidden_size=8, num_concepts=3, embedding_dim=4)
    out = layer(torch.randn(5, 8))
    expected = torch.cat(
        [out.concept_embeddings.reshape(5, -1), out.unknown_embedding], dim=-1
    )
    assert torch.allclose(out.bottleneck, expected)


def test_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        ConceptBottleneck(hidden_size=8, num_concepts=0, embedding_dim=4)
    with pytest.raises(ValueError):
        ConceptBottleneck(hidden_size=8, num_concepts=4, embedding_dim=0)


def test_gradients_flow_to_both_branches() -> None:
    layer = ConceptBottleneck(hidden_size=8, num_concepts=4, embedding_dim=4)
    hidden_states = torch.randn(6, 8, requires_grad=True)

    out = layer(hidden_states)
    out.bottleneck.sum().backward()

    assert layer.context_proj.weight.grad is not None
    assert layer.prob_weight.grad is not None
    assert torch.any(layer.context_proj.weight.grad != 0)
    assert torch.any(layer.prob_weight.grad != 0)


# ---------------------------------------------------------------------------
# concept_loss
# ---------------------------------------------------------------------------


def test_concept_loss_zero_for_perfect_confident_predictions() -> None:
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = (labels * 2 - 1) * 20.0  # +-20 => sigmoid saturates to the label
    loss = concept_loss(logits, labels)
    assert loss.item() < 1e-6


def test_concept_loss_mask_excludes_unobserved_labels() -> None:
    # Two entries: one correct-if-scored, one wildly wrong but masked out.
    logits = torch.tensor([[10.0, -10.0]])
    labels = torch.tensor([[1.0, 1.0]])  # second entry is a "wrong" label
    mask_all = torch.tensor([[1.0, 1.0]])
    mask_observed_only_first = torch.tensor([[1.0, 0.0]])

    loss_all = concept_loss(logits, labels, mask_all)
    loss_masked = concept_loss(logits, labels, mask_observed_only_first)

    assert loss_masked.item() < loss_all.item()
    assert loss_masked.item() < 1e-4


def test_concept_loss_all_masked_out_is_finite() -> None:
    logits = torch.randn(3, 4)
    labels = torch.randint(0, 2, (3, 4)).float()
    mask = torch.zeros(3, 4)
    loss = concept_loss(logits, labels, mask)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# orthogonality_loss
# ---------------------------------------------------------------------------


def test_orthogonality_loss_zero_when_unknown_orthogonal_to_all_concepts() -> None:
    # 2 known concepts in a 2D embedding space spanning the x/y axes; the
    # unknown embedding points along z -- exactly orthogonal to both.
    concept_embeddings = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # (1, 2, 3)
    unknown_embedding = torch.tensor([[0.0, 0.0, 1.0]])  # (1, 3)
    loss = orthogonality_loss(concept_embeddings, unknown_embedding)
    assert loss.item() < 1e-6


def test_orthogonality_loss_maximal_when_unknown_equals_a_concept() -> None:
    concept_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    unknown_embedding = torch.tensor([[1.0, 0.0]])  # identical to first concept
    loss = orthogonality_loss(concept_embeddings, unknown_embedding)
    assert loss.item() > 0.49  # mean of (1.0, 0.0) == 0.5


def test_orthogonality_loss_higher_when_correlated() -> None:
    n, k, m = 200, 3, 4
    concept_embeddings = torch.randn(n, k, m)
    # Correlated: unknown is a noisy copy of the first concept.
    correlated_unknown = concept_embeddings[:, 0, :] + 0.01 * torch.randn(n, m)
    # Uncorrelated: independent noise.
    uncorrelated_unknown = torch.randn(n, m)

    loss_correlated = orthogonality_loss(concept_embeddings, correlated_unknown)
    loss_uncorrelated = orthogonality_loss(concept_embeddings, uncorrelated_unknown)

    assert loss_correlated.item() > loss_uncorrelated.item()


# ---------------------------------------------------------------------------
# combined_loss
# ---------------------------------------------------------------------------


def test_combined_loss_components_and_weighting() -> None:
    layer = ConceptBottleneck(hidden_size=8, num_concepts=3, embedding_dim=4)
    hidden_states = torch.randn(5, 8)
    out = layer(hidden_states)
    labels = torch.randint(0, 2, (5, 3)).float()
    task_loss = torch.tensor(2.0)

    weights = ConceptBottleneckLossWeights(concept=1.0, orthogonality=0.0)
    total, components = combined_loss(
        task_loss,
        out.concept_logits,
        labels,
        out.concept_embeddings,
        out.unknown_embedding,
        weights=weights,
    )

    expected = task_loss + components["concept_loss"]
    assert torch.allclose(total, expected)
    assert set(components) == {"task_loss", "concept_loss", "orthogonality_loss"}
    # Components are detached (no grad_fn) so logging them doesn't retain the graph.
    assert not components["concept_loss"].requires_grad


# ---------------------------------------------------------------------------
# Synthetic end-to-end training: does the mechanism actually do what it
# claims, not just "does the code run"?
# ---------------------------------------------------------------------------


class _TinyBackboneWithHead(nn.Module):
    """A minimal stand-in for EHR-Mamba3: encoder + bottleneck + task head.

    Mamba-3 requires CUDA (mamba-ssm) to build, so it can't run on a Mac
    dev machine. This substitutes a small MLP encoder that exposes the same
    hidden-state interface the concept bottleneck consumes, so the
    bottleneck/loss wiring can be validated end-to-end on CPU. Swapping in
    the real EHR-Mamba3 backbone (on the GCP A100 host) only changes what
    produces ``hidden_states`` — everything downstream is unchanged.
    """

    def __init__(
        self, input_dim: int, hidden: int, num_concepts: int, embedding_dim: int
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.bottleneck = ConceptBottleneck(hidden, num_concepts, embedding_dim)
        self.task_head = nn.Linear((num_concepts + 1) * embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        hidden_states = self.encoder(x)
        out = self.bottleneck(hidden_states)
        prediction = self.task_head(out.bottleneck).squeeze(-1)
        return prediction, out


def _make_synthetic_dataset(
    n: int, input_dim: int, num_concepts: int, num_true_residual_factors: int
) -> tuple:
    """Build a task solvable only via both concepts and uncovered factors.

    So a correct implementation must learn concept accuracy AND retain
    unknown-concept capacity to solve the task, not just collapse
    everything into one branch.
    """
    z_concepts = torch.randn(n, num_concepts)
    z_residual_true = torch.randn(n, num_true_residual_factors)
    mixing = torch.randn(input_dim, num_concepts + num_true_residual_factors)
    z = torch.cat([z_concepts, z_residual_true], dim=-1)
    x = z @ mixing.T + 0.05 * torch.randn(n, input_dim)

    concept_labels = (z_concepts > 0).float()
    # Task target depends on concepts AND the residual-only factors.
    y = z_concepts.sum(dim=-1) + z_residual_true.sum(dim=-1)
    return x, concept_labels, y


def test_synthetic_training_learns_concepts_and_task() -> None:
    torch.manual_seed(1)
    input_dim, hidden, num_concepts, embedding_dim = 12, 16, 4, 6
    x, concept_labels, y = _make_synthetic_dataset(
        n=256,
        input_dim=input_dim,
        num_concepts=num_concepts,
        num_true_residual_factors=3,
    )

    model = _TinyBackboneWithHead(input_dim, hidden, num_concepts, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    weights = ConceptBottleneckLossWeights(concept=1.0, orthogonality=0.05)

    losses = []
    for _ in range(300):
        optimizer.zero_grad()
        prediction, out = model(x)
        task_loss = nn.functional.mse_loss(prediction, y)
        total, components = combined_loss(
            task_loss,
            out.concept_logits,
            concept_labels,
            out.concept_embeddings,
            out.unknown_embedding,
            weights=weights,
        )
        total.backward()
        optimizer.step()
        losses.append(components)

    final = losses[-1]
    initial = losses[0]

    # Both the task and the concept-supervision objective actually improve.
    assert final["task_loss"].item() < initial["task_loss"].item() * 0.3
    assert final["concept_loss"].item() < initial["concept_loss"].item() * 0.5

    # Concept predictions are actually accurate, not just "loss went down".
    with torch.no_grad():
        _, out = model(x)
        predicted_concepts = out.concept_probs > 0.5
        accuracy = (predicted_concepts == concept_labels.bool()).float().mean().item()
    assert accuracy > 0.9


def test_orthogonality_penalty_reduces_concept_unknown_entanglement() -> None:
    """Directly tests the paper's central claim about the orthogonality term.

    It should suppress the unknown concept re-encoding the known ones.
    Train the same architecture on the same data with the penalty on vs.
    off and confirm the trained bottleneck's concept/unknown cosine
    similarity is lower with it on.
    """

    def train(orthogonality_weight: float) -> float:
        torch.manual_seed(2)
        input_dim, hidden, num_concepts, embedding_dim = 12, 16, 4, 6
        x, concept_labels, y = _make_synthetic_dataset(
            n=256,
            input_dim=input_dim,
            num_concepts=num_concepts,
            num_true_residual_factors=3,
        )
        model = _TinyBackboneWithHead(input_dim, hidden, num_concepts, embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        weights = ConceptBottleneckLossWeights(
            concept=1.0, orthogonality=orthogonality_weight
        )

        for _ in range(300):
            optimizer.zero_grad()
            prediction, out = model(x)
            task_loss = nn.functional.mse_loss(prediction, y)
            total, _ = combined_loss(
                task_loss,
                out.concept_logits,
                concept_labels,
                out.concept_embeddings,
                out.unknown_embedding,
                weights=weights,
            )
            total.backward()
            optimizer.step()

        with torch.no_grad():
            _, out = model(x)
            return orthogonality_loss(
                out.concept_embeddings, out.unknown_embedding
            ).item()

    entanglement_with_penalty = train(orthogonality_weight=0.5)
    entanglement_without_penalty = train(orthogonality_weight=0.0)

    assert entanglement_with_penalty < entanglement_without_penalty
