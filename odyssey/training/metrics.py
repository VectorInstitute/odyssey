"""Evaluation metrics for the concept-bottleneck sequence model.

Three distinct questions, per research_journal/04_concept_pipeline.html
and research_journal/05_missingness.html, and each gets its own metric
family here rather than being folded into a single number:

1. **Forecasting quality** (:func:`compute_task_metrics`): is the model
   good at predicting what happens next, independent of interpretability
   at all -- the same question
   :class:`~odyssey.models.sequence_model.BaselineSequenceModel` exists
   to answer without a bottleneck in the way, for comparison.
2. **Concept quality** (:func:`compute_concept_metrics`,
   :func:`compute_observability_metrics`): are the individual known
   concepts, and the new observability head (entry 05), actually
   accurate against real labels -- evaluated only on observed subjects,
   the same masking discipline :func:`odyssey.data.concepts.label_concepts`
   uses for supervision.
3. **Concept usefulness** (:func:`compute_completeness`): do the concepts
   collectively explain the model's forecasting behavior, or are they a
   decorative side-channel the task loss routes around via the unknown
   embedding -- a directly runnable version of entry 04 decision (d)'s
   completeness/marginal-contribution filter, blocked at the time that
   entry was written on a trained model existing.

All three are needed together: a model can have perfectly accurate
concepts (2) that the task barely uses (3), or a task that forecasts
well (1) while the concepts are decorative (3) -- neither failure mode
is visible from the other two metric families alone.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.models.concept_bottleneck import orthogonality_loss


# ---------------------------------------------------------------------------
# 1. Forecasting quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskMetrics:
    """Next-token forecasting quality over a set of predictions."""

    cross_entropy: float
    perplexity: float
    top1_accuracy: float
    top5_accuracy: float
    n_predictions: int
    """How many non-ignored positions this was computed over."""

    set_top1_accuracy: Optional[float] = None
    """Top-1 scored against the target's same-timestamp event block: a
    prediction counts if it names any event recorded at the same instant
    as the true next event. Exact-next scoring grades the model on the
    arbitrary within-block charting order (a discharge's ~22 diagnosis
    codes share one timestamp, capping exact-next diagnosis accuracy near
    6% even for a perfect model guessing uniformly within blocks); set
    scoring measures whether the model knows what happens next, not where
    the ETL put it inside the block. None where the evaluation path
    cannot see block structure."""

    n_set_predictions: Optional[int] = None
    """Positions the set-based metric covered (block membership is only
    visible within a chunk, so chunk-boundary positions are excluded)."""


def compute_task_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int,
    top_k: Sequence[int] = (1, 5),
) -> TaskMetrics:
    """Compute next-token cross-entropy, perplexity, and top-k accuracy.

    ``logits`` is ``(N, vocab_size)``, ``targets`` is ``(N,)`` -- flatten
    any batch/sequence dims before calling. Positions where
    ``targets == ignore_index`` (padding) are excluded entirely, matching
    :func:`~odyssey.models.sequence_model._SequenceModelBase._whole_sequence_next_token_loss`'s
    ``ignore_index`` semantics.
    """
    mask = targets != ignore_index
    logits = logits[mask]
    targets = targets[mask]
    n = targets.shape[0]
    if n == 0:
        raise ValueError("no non-ignored predictions to compute metrics over")

    cross_entropy = F.cross_entropy(logits, targets, reduction="mean").item()
    perplexity = float(torch.exp(torch.tensor(cross_entropy)))

    max_k = max(top_k)
    top_k_preds = logits.topk(max_k, dim=-1).indices  # (n, max_k)
    hits = top_k_preds == targets.unsqueeze(-1)  # (n, max_k)
    accuracies = {k: hits[:, :k].any(dim=-1).float().mean().item() for k in top_k}

    return TaskMetrics(
        cross_entropy=cross_entropy,
        perplexity=perplexity,
        top1_accuracy=accuracies.get(1, float("nan")),
        top5_accuracy=accuracies.get(5, float("nan")),
        n_predictions=n,
    )


def compute_task_metrics_by_code_type(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: Vocabulary,
    *,
    ignore_index: int,
) -> Dict[str, TaskMetrics]:
    """:func:`compute_task_metrics`, broken down by the target token's code type.

    Diagnosis/medication/procedure/lab codes have wildly different
    predictability and vocabulary sizes; a single aggregate accuracy
    number hides that. Uses :func:`odyssey.data.vocabulary.code_type` on
    each target's decoded code, so the breakdown always matches the same
    type taxonomy tokenization already uses.
    """
    type_names = {
        0: "pad",
        1: "diagnosis",
        2: "medication",
        3: "procedure",
        4: "lab",
        5: "visit",
        6: "demographic",
        7: "billing",
        8: "other",
    }
    mask = targets != ignore_index
    logits, targets = logits[mask], targets[mask]
    target_types = torch.tensor(
        [code_type(vocab.decode(int(t))) for t in targets], dtype=torch.long
    )

    out: Dict[str, TaskMetrics] = {}
    for type_id, name in type_names.items():
        if type_id == 0:
            continue
        type_mask = target_types == type_id
        if not type_mask.any():
            continue
        out[name] = compute_task_metrics(
            logits[type_mask], targets[type_mask], ignore_index=ignore_index
        )
    return out


# ---------------------------------------------------------------------------
# 2. Concept quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptMetrics:
    """Accuracy of one concept's predicted probability, among observed subjects."""

    name: str
    n_observed: int
    prevalence: float
    """Fraction of observed subjects where the true label is 1."""

    auroc: Optional[float]
    """None if degenerate (only one class present among observed subjects)."""

    auprc: Optional[float]
    brier_score: Optional[float]
    accuracy_at_0_5: Optional[float]


def _binary_metrics(
    probs: torch.Tensor, labels: torch.Tensor
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Shared (auroc, auprc, brier, accuracy) computation for one binary column.

    Returns all-``None`` if there are no examples, or only one class is
    present (AUROC/AUPRC are undefined then -- sklearn raises rather than
    returning NaN, so this checks explicitly instead of catching broadly).
    """
    if probs.numel() == 0 or torch.unique(labels).numel() < 2:
        return None, None, None, None
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    auroc = float(roc_auc_score(labels_np, probs_np))
    auprc = float(average_precision_score(labels_np, probs_np))
    brier = float(brier_score_loss(labels_np, probs_np))
    accuracy = float(((probs >= 0.5).float() == labels).float().mean())
    return auroc, auprc, brier, accuracy


def compute_concept_metrics(
    concept_probs: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_mask: torch.Tensor,
    concept_names: Sequence[str],
) -> List[ConceptMetrics]:
    """Per-concept AUROC/AUPRC/Brier score/accuracy, restricted to observed subjects.

    ``concept_probs``/``concept_labels``/``concept_mask`` are all
    ``(n_subjects, num_concepts)``; ``concept_mask`` is the same
    ``{name}_observed`` semantics as everywhere else in this project (1 =
    observed). Evaluating on unobserved subjects would score the model
    against a label that was never real to begin with.
    """
    out = []
    for i, name in enumerate(concept_names):
        observed = concept_mask[:, i] > 0
        probs_i = concept_probs[observed, i]
        labels_i = concept_labels[observed, i]
        n_observed = int(observed.sum())
        prevalence = float(labels_i.float().mean()) if n_observed > 0 else float("nan")
        auroc, auprc, brier, accuracy = _binary_metrics(probs_i, labels_i)
        out.append(
            ConceptMetrics(
                name=name,
                n_observed=n_observed,
                prevalence=prevalence,
                auroc=auroc,
                auprc=auprc,
                brier_score=brier,
                accuracy_at_0_5=accuracy,
            )
        )
    return out


@dataclass(frozen=True)
class ObservabilityMetrics:
    """How well the entry-05 observability head predicts real measurement patterns."""

    name: str
    n_subjects: int
    observed_rate: float
    auroc: Optional[float]
    accuracy_at_0_5: Optional[float]


def compute_observability_metrics(
    observability_probs: torch.Tensor,
    observed_mask: torch.Tensor,
    concept_names: Sequence[str],
) -> List[ObservabilityMetrics]:
    """Per-concept accuracy of the observability head against real ``observed_mask``.

    Unlike :func:`compute_concept_metrics`, every subject has a real
    target here (whether a lab was drawn is never itself missing), so
    there's no masking -- this is entry 05's "still open" validation
    that the head learned something real, not just wired correctly.
    """
    out = []
    for i, name in enumerate(concept_names):
        probs_i = observability_probs[:, i]
        labels_i = observed_mask[:, i]
        n = probs_i.shape[0]
        observed_rate = float(labels_i.float().mean()) if n > 0 else float("nan")
        auroc, _, _, accuracy = _binary_metrics(probs_i, labels_i)
        out.append(
            ObservabilityMetrics(
                name=name,
                n_subjects=n,
                observed_rate=observed_rate,
                auroc=auroc,
                accuracy_at_0_5=accuracy,
            )
        )
    return out


def orthogonality_diagnostic(
    concept_embeddings: torch.Tensor, unknown_embedding: torch.Tensor
) -> float:
    """Test-set mean |cosine similarity| between known and unknown concept embeddings.

    Same computation as :func:`odyssey.models.concept_bottleneck.orthogonality_loss`,
    reported here as a plain float diagnostic rather than a training loss
    term -- confirms the regularization actually held on held-out data,
    not just on the data it was optimized against.
    """
    return float(orthogonality_loss(concept_embeddings, unknown_embedding))


# ---------------------------------------------------------------------------
# 3. Concept usefulness: completeness (entry 04, decision (d))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletenessResult:
    """How much of a binary task outcome the known concepts alone explain.

    A simple probe (logistic regression) trained on concept_probs alone
    -> the task outcome, evaluated against how much of the full model's
    own accuracy that recovers. ``completeness_score`` of 1.0 means the
    concepts alone are as good as the full model at this task; 0.0 means
    they're no better than the random baseline -- the concepts would be
    decorative. Directly implements the completeness idea from Yeh, Kim,
    Yen, Ravikumar (NeurIPS 2020), simplified to skip their concept-
    discovery machinery since the concept set here is already fixed and
    supervised (entry 04, Section 03).
    """

    full_model_accuracy: float
    concepts_only_accuracy: float
    random_baseline_accuracy: float
    completeness_score: float


def compute_completeness(
    concept_probs_train: torch.Tensor,
    task_labels_train: torch.Tensor,
    concept_probs_test: torch.Tensor,
    task_labels_test: torch.Tensor,
    full_model_accuracy: float,
) -> CompletenessResult:
    """Fit a probe on ``concept_probs_train`` -> ``task_labels_train``, score on test.

    ``task_labels_{train,test}`` is a binary ``(n_subjects,)`` outcome
    (e.g. "an adverse-event-relevant code appears in this patient's next
    window") -- not the multi-class next-token target, a single
    forecasting-relevant binary question the probe can actually be
    trained on with a modest number of subjects. ``full_model_accuracy``
    is supplied by the caller (computed however the real task accuracy
    for that same binary question is defined) rather than recomputed
    here, since that requires running the full model, not just its
    concept outputs.
    """
    random_baseline = max(
        float(task_labels_test.float().mean()),
        1.0 - float(task_labels_test.float().mean()),
    )
    if torch.unique(task_labels_train).numel() < 2:
        # Can't fit a probe with only one class in the training labels;
        # report the random baseline as the probe's own score rather than
        # raising -- a genuinely uninformative training split, not a bug.
        concepts_only_accuracy = random_baseline
    else:
        probe = LogisticRegression(max_iter=1000)
        probe.fit(concept_probs_train.numpy(), task_labels_train.numpy())
        concepts_only_accuracy = float(
            probe.score(concept_probs_test.numpy(), task_labels_test.numpy())
        )

    denom = full_model_accuracy - random_baseline
    completeness_score = (
        (concepts_only_accuracy - random_baseline) / denom
        if denom > 1e-9
        else float("nan")
    )
    return CompletenessResult(
        full_model_accuracy=full_model_accuracy,
        concepts_only_accuracy=concepts_only_accuracy,
        random_baseline_accuracy=random_baseline,
        completeness_score=completeness_score,
    )


__all__ = [
    "TaskMetrics",
    "compute_task_metrics",
    "compute_task_metrics_by_code_type",
    "ConceptMetrics",
    "compute_concept_metrics",
    "ObservabilityMetrics",
    "compute_observability_metrics",
    "orthogonality_diagnostic",
    "CompletenessResult",
    "compute_completeness",
]
