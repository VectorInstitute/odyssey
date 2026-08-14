"""Run a trained :class:`ConceptBottleneckSequenceModel` over held-out data.

Streams held-out patients through the model exactly the way training did
(:class:`~odyssey.data.streaming.PackedLaneSampler`, carried recurrent
state across chunks), rather than the whole-sequence-per-row path: a
held-out patient's full stay can be far longer than any single training
chunk, and streaming is the only path that scales to that without an
enormous padded batch. ``reset_prob=0.0`` here, unlike training -- at
inference time we want the model to see a patient's true full history,
not synthetic missing-history resets.

Produces one :class:`InferenceResults`, covering all three eval
questions from ``odyssey/training/metrics.py``: forecasting quality,
concept quality, and (via :func:`orthogonality_diagnostic`) whether the
known/unknown concept split held on data the model never trained on.
Concept usefulness (completeness) is intentionally not computed here --
see ``research_journal`` for why a binary task-outcome label for that
probe still needs a real design decision, not implemented yet.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
import torch

from odyssey.data.concepts import CONCEPTS
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    _gather_by_subject,
    _pool_patient_ends,
)
from odyssey.training.data import (
    build_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.metrics import (
    ConceptMetrics,
    ObservabilityMetrics,
    TaskMetrics,
    compute_concept_metrics,
    compute_observability_metrics,
    compute_task_metrics,
    compute_task_metrics_by_code_type,
    orthogonality_diagnostic,
)
from odyssey.training.train import TrainingConfig, _move_chunk_to_device, build_model


@dataclass(frozen=True)
class InferenceResults:
    """Everything scored from one streaming pass over a held-out split."""

    task_metrics: TaskMetrics
    task_metrics_by_code_type: Dict[str, TaskMetrics]
    concept_metrics: List[ConceptMetrics]
    observability_metrics: List[ObservabilityMetrics]
    orthogonality: float
    n_patient_ends_scored: int


def _latest_checkpoint(run_dir: Path) -> Path:
    """Return ``checkpoint_final.pt`` if present, else the highest-step periodic one.

    Lets evaluation run against an in-progress training run (e.g. to
    sanity-check the pipeline before a long run finishes), not only a
    fully completed one.
    """
    final = run_dir / "checkpoint_final.pt"
    if final.exists():
        return final
    candidates = list(run_dir.glob("checkpoint_[0-9]*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_*.pt found in {run_dir}")
    return max(candidates, key=lambda p: int(p.stem.split("_")[-1]))


def load_run(
    run_dir: Union[str, Path], *, device: str = "cuda"
) -> Tuple[ConceptBottleneckSequenceModel, Vocabulary, QuantileBinner, TrainingConfig]:
    """Reconstruct a trained model and its tokenization artifacts from a run dir.

    ``run_dir`` is a :func:`~odyssey.training.train.train` output
    directory: reads ``config.json`` (architecture hyperparameters --
    the training-only fields it also contains, e.g. ``learning_rate``,
    are simply unused by :func:`~odyssey.training.train.build_model`),
    ``vocabulary.json``, ``quantile_binner.json``, and the latest
    available checkpoint (see :func:`_latest_checkpoint`).
    """
    run_dir = Path(run_dir)
    config = TrainingConfig(**json.loads((run_dir / "config.json").read_text()))
    vocab = Vocabulary.load(run_dir / "vocabulary.json")
    binner = QuantileBinner.load(run_dir / "quantile_binner.json")

    model = build_model(config, vocab_size=len(vocab), num_concepts=len(CONCEPTS))
    checkpoint = torch.load(_latest_checkpoint(run_dir), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model, vocab, binner, config


def load_and_bin_held_out(
    shard_dir: Union[str, Path],
    binner: QuantileBinner,
    *,
    max_shards: Optional[int] = None,
) -> pl.DataFrame:
    """Load a held-out MEDS split and apply the *train-fit* binner to it.

    Never re-fits the binner here -- using the train split's own
    quantile boundaries on held-out data is the whole point of
    evaluating on genuinely unseen data.
    """
    events = load_meds_shards(shard_dir, max_shards=max_shards)
    return add_value_tokens(events, binner)


def run_streaming_inference(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_labels: Dict[int, torch.Tensor],
    concept_mask: Dict[int, torch.Tensor],
    *,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
) -> InferenceResults:
    """Stream held-out patients through ``model`` and score every eval question.

    ``concept_labels``/``concept_mask`` follow
    :func:`~odyssey.training.data.build_concept_label_dicts`'s
    ``subject_id -> (num_concepts,)`` shape, built from the *unbinned*
    held-out events (concept labeling never looks at value tokens).
    """
    model.eval()
    patients = iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    end_subject_ids: List[torch.Tensor] = []
    end_concept_probs: List[torch.Tensor] = []
    end_observability_probs: List[torch.Tensor] = []
    end_concept_embeddings: List[torch.Tensor] = []
    end_unknown_embedding: List[torch.Tensor] = []

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            logits, bottleneck_out, state = model(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )

            real = chunk.real_mask
            if real.any():
                all_logits.append(logits[real].cpu())
                all_targets.append(chunk.targets[real].cpu())

            if chunk.patient_end.any():
                end_subject_ids.append(
                    _pool_patient_ends(chunk.subject_ids, chunk.patient_end).cpu()
                )
                end_concept_probs.append(
                    _pool_patient_ends(
                        bottleneck_out.concept_probs, chunk.patient_end
                    ).cpu()
                )
                end_observability_probs.append(
                    _pool_patient_ends(
                        bottleneck_out.observability_probs, chunk.patient_end
                    ).cpu()
                )
                end_concept_embeddings.append(
                    _pool_patient_ends(
                        bottleneck_out.concept_embeddings, chunk.patient_end
                    ).cpu()
                )
                end_unknown_embedding.append(
                    _pool_patient_ends(
                        bottleneck_out.unknown_embedding, chunk.patient_end
                    ).cpu()
                )

    task_logits = torch.cat(all_logits)
    task_targets = torch.cat(all_targets)
    task_metrics = compute_task_metrics(task_logits, task_targets, ignore_index=PAD_ID)
    task_metrics_by_code_type = compute_task_metrics_by_code_type(
        task_logits, task_targets, vocab, ignore_index=PAD_ID
    )

    subject_ids = torch.cat(end_subject_ids)
    concept_probs = torch.cat(end_concept_probs)
    observability_probs = torch.cat(end_observability_probs)
    concept_embeddings = torch.cat(end_concept_embeddings)
    unknown_embedding = torch.cat(end_unknown_embedding)

    concept_names = [c.name for c in CONCEPTS]
    labels = _gather_by_subject(subject_ids, concept_labels)
    masks = _gather_by_subject(subject_ids, concept_mask)
    observed_mask = masks > 0

    concept_metrics = compute_concept_metrics(
        concept_probs, labels, masks, concept_names
    )
    observability_metrics = compute_observability_metrics(
        observability_probs, observed_mask.float(), concept_names
    )
    orthogonality = orthogonality_diagnostic(concept_embeddings, unknown_embedding)

    return InferenceResults(
        task_metrics=task_metrics,
        task_metrics_by_code_type=task_metrics_by_code_type,
        concept_metrics=concept_metrics,
        observability_metrics=observability_metrics,
        orthogonality=orthogonality,
        n_patient_ends_scored=int(subject_ids.shape[0]),
    )


def evaluate_run(
    run_dir: Union[str, Path],
    held_out_shard_dir: Union[str, Path],
    *,
    max_shards: Optional[int] = None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: Optional[str] = None,
) -> InferenceResults:
    """End-to-end: load a trained run, score it against a held-out split."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, _ = load_run(run_dir, device=device)

    print(f"[inference] loading held-out shards from {held_out_shard_dir}")
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    events_binned = add_value_tokens(raw_events, binner)

    print("[inference] labeling concepts")
    concept_labels, concept_mask = build_concept_label_dicts(raw_events, CONCEPTS)
    del raw_events

    print("[inference] running streaming inference")
    return run_streaming_inference(
        model,
        events_binned,
        vocab,
        concept_labels,
        concept_mask,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
    )


__all__ = [
    "InferenceResults",
    "load_run",
    "load_and_bin_held_out",
    "run_streaming_inference",
    "evaluate_run",
]
