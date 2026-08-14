"""Tests for evaluation metrics."""

import torch

from odyssey.data.vocabulary import PAD_ID, UNK_ID, Vocabulary
from odyssey.models.concept_bottleneck import orthogonality_loss
from odyssey.training.metrics import (
    compute_completeness,
    compute_concept_metrics,
    compute_observability_metrics,
    compute_task_metrics,
    compute_task_metrics_by_code_type,
    orthogonality_diagnostic,
)


# ---------------------------------------------------------------------------
# compute_task_metrics
# ---------------------------------------------------------------------------


def test_task_metrics_perfect_predictions_have_zero_cross_entropy() -> None:
    targets = torch.tensor([2, 5, 1])
    logits = torch.full((3, 10), -20.0)
    logits[torch.arange(3), targets] = 20.0

    metrics = compute_task_metrics(logits, targets, ignore_index=0)

    assert metrics.cross_entropy < 1e-6
    assert metrics.perplexity < 1.001
    assert metrics.top1_accuracy == 1.0
    assert metrics.top5_accuracy == 1.0
    assert metrics.n_predictions == 3


def test_task_metrics_excludes_ignore_index_positions() -> None:
    targets = torch.tensor([2, 0, 0, 5])  # two padding positions
    logits = torch.randn(4, 10)
    logits[0, 2] = 20.0  # make position 0 correct
    logits[3, 5] = 20.0  # make position 3 correct

    metrics = compute_task_metrics(logits, targets, ignore_index=0)

    assert metrics.n_predictions == 2  # only the two non-padding positions
    assert metrics.top1_accuracy == 1.0


def test_task_metrics_top5_more_lenient_than_top1() -> None:
    torch.manual_seed(0)
    targets = torch.tensor([3])
    logits = torch.zeros(1, 10)
    # Rank the true class 3rd-highest: not top-1, but within top-5.
    logits[0, 3] = 1.0
    logits[0, 7] = 3.0
    logits[0, 8] = 2.0

    metrics = compute_task_metrics(logits, targets, ignore_index=0, top_k=(1, 5))
    assert metrics.top1_accuracy == 0.0
    assert metrics.top5_accuracy == 1.0


def test_task_metrics_raises_when_everything_is_ignored() -> None:
    targets = torch.tensor([0, 0, 0])
    logits = torch.randn(3, 5)
    try:
        compute_task_metrics(logits, targets, ignore_index=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_task_metrics_by_code_type_splits_correctly() -> None:
    vocab = Vocabulary(
        token_to_id={
            "[PAD]": PAD_ID,
            "[UNK]": UNK_ID,
            "DIAGNOSIS//A": 2,
            "MEDICATION//B": 3,
            "LAB//220045//": 4,
        }
    )
    # Two diagnosis targets, one medication, one padding (excluded).
    targets = torch.tensor([2, 2, 3, 0])
    logits = torch.randn(4, 5)
    logits[:, :] = -20.0
    logits[torch.arange(4), targets] = 20.0

    by_type = compute_task_metrics_by_code_type(logits, targets, vocab, ignore_index=0)

    assert set(by_type) == {"diagnosis", "medication"}
    assert by_type["diagnosis"].n_predictions == 2
    assert by_type["medication"].n_predictions == 1
    assert by_type["diagnosis"].top1_accuracy == 1.0


# ---------------------------------------------------------------------------
# compute_concept_metrics
# ---------------------------------------------------------------------------


def test_concept_metrics_perfect_separation_gives_auroc_one() -> None:
    probs = torch.tensor([[0.9], [0.8], [0.1], [0.2]])
    labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    mask = torch.ones(4, 1)

    metrics = compute_concept_metrics(probs, labels, mask, ["c"])

    assert metrics[0].auroc == 1.0
    assert metrics[0].accuracy_at_0_5 == 1.0
    assert metrics[0].n_observed == 4
    assert metrics[0].prevalence == 0.5


def test_concept_metrics_excludes_unobserved_subjects() -> None:
    # Subject 2 is unobserved and would (if included) be a wildly wrong,
    # confident prediction -- must not affect the score at all.
    probs = torch.tensor([[0.9], [0.1], [0.99]])
    labels = torch.tensor([[1.0], [0.0], [0.0]])
    mask = torch.tensor([[1.0], [1.0], [0.0]])

    metrics = compute_concept_metrics(probs, labels, mask, ["c"])

    assert metrics[0].n_observed == 2
    assert metrics[0].auroc == 1.0


def test_concept_metrics_degenerate_single_class_returns_none() -> None:
    probs = torch.tensor([[0.9], [0.1], [0.5]])
    labels = torch.tensor([[1.0], [1.0], [1.0]])  # every observed label is positive
    mask = torch.ones(3, 1)

    metrics = compute_concept_metrics(probs, labels, mask, ["c"])

    assert metrics[0].auroc is None
    assert metrics[0].auprc is None
    assert metrics[0].brier_score is None
    assert metrics[0].accuracy_at_0_5 is None
    assert metrics[0].n_observed == 3  # still reported, just no separability metric


def test_concept_metrics_zero_observed_subjects() -> None:
    probs = torch.tensor([[0.9], [0.1]])
    labels = torch.tensor([[1.0], [0.0]])
    mask = torch.zeros(2, 1)

    metrics = compute_concept_metrics(probs, labels, mask, ["c"])

    assert metrics[0].n_observed == 0
    assert metrics[0].auroc is None


def test_concept_metrics_multiple_concepts_are_independent() -> None:
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    labels = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask = torch.ones(2, 2)

    metrics = compute_concept_metrics(probs, labels, mask, ["good", "bad"])

    assert metrics[0].name == "good"
    assert metrics[0].auroc == 1.0
    assert metrics[1].name == "bad"
    assert metrics[1].auroc == 0.0  # perfectly anti-correlated


# ---------------------------------------------------------------------------
# compute_observability_metrics
# ---------------------------------------------------------------------------


def test_observability_metrics_no_masking_every_subject_counts() -> None:
    probs = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
    observed = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    metrics = compute_observability_metrics(probs, observed, ["c"])

    assert metrics[0].n_subjects == 4
    assert metrics[0].auroc == 1.0
    assert metrics[0].observed_rate == 0.5


# ---------------------------------------------------------------------------
# orthogonality_diagnostic
# ---------------------------------------------------------------------------


def test_orthogonality_diagnostic_matches_orthogonality_loss() -> None:
    concept_embeddings = torch.randn(5, 3, 4)
    unknown_embedding = torch.randn(5, 4)

    diagnostic = orthogonality_diagnostic(concept_embeddings, unknown_embedding)
    expected = orthogonality_loss(concept_embeddings, unknown_embedding).item()

    assert abs(diagnostic - expected) < 1e-6


# ---------------------------------------------------------------------------
# compute_completeness
# ---------------------------------------------------------------------------


def test_completeness_perfect_probe_matches_full_model() -> None:
    torch.manual_seed(0)
    n = 200
    # Task label is exactly concept 0 -- a probe should recover it perfectly.
    concept_probs = torch.rand(n, 3)
    concept_probs[:, 0] = torch.cat(
        [torch.rand(n // 2) * 0.1, torch.rand(n - n // 2) * 0.1 + 0.9]
    )
    task_labels = (concept_probs[:, 0] > 0.5).float()
    perm = torch.randperm(n)
    concept_probs, task_labels = concept_probs[perm], task_labels[perm]

    train_probs, test_probs = concept_probs[:100], concept_probs[100:]
    train_labels, test_labels = task_labels[:100], task_labels[100:]

    result = compute_completeness(
        train_probs, train_labels, test_probs, test_labels, full_model_accuracy=1.0
    )

    assert result.concepts_only_accuracy > 0.95
    assert result.completeness_score > 0.9


def test_completeness_uninformative_concepts_score_near_zero() -> None:
    torch.manual_seed(1)
    n = 200
    concept_probs = torch.rand(n, 3)  # pure noise, unrelated to the task
    task_labels = torch.randint(0, 2, (n,)).float()

    train_probs, test_probs = concept_probs[:100], concept_probs[100:]
    train_labels, test_labels = task_labels[:100], task_labels[100:]

    result = compute_completeness(
        train_probs, train_labels, test_probs, test_labels, full_model_accuracy=0.9
    )

    assert result.completeness_score < 0.5


def test_completeness_degenerate_training_labels_falls_back_to_baseline() -> None:
    n = 50
    concept_probs = torch.rand(n, 3)
    train_labels = torch.zeros(50)  # only one class in training data
    test_labels = torch.randint(0, 2, (n,)).float()

    result = compute_completeness(
        concept_probs, train_labels, concept_probs, test_labels, full_model_accuracy=0.9
    )

    assert result.concepts_only_accuracy == result.random_baseline_accuracy
