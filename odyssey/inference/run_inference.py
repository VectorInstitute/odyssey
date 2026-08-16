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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import CONCEPTS
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
    _gather_by_subject,
    _gather_by_visit,
    _pool_patient_ends,
)
from odyssey.training.data import (
    build_concept_label_dicts,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.metrics import (
    ConceptMetrics,
    ObservabilityMetrics,
    TaskMetrics,
    compute_concept_metrics,
    compute_observability_metrics,
    orthogonality_diagnostic,
)
from odyssey.training.train import TrainingConfig, _move_chunk_to_device, build_model


logger = logging.getLogger(__name__)

_CODE_TYPE_NAMES = {
    1: "diagnosis",
    2: "medication",
    3: "procedure",
    4: "lab",
    5: "visit",
    6: "demographic",
    7: "billing",
    8: "other",
}


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
    run_dir: Union[str, Path],
    *,
    device: str = "cuda",
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Tuple[ConceptBottleneckSequenceModel, Vocabulary, QuantileBinner, TrainingConfig]:
    """Reconstruct a trained model and its tokenization artifacts from a run dir.

    ``run_dir`` is a :func:`~odyssey.training.train.train` output
    directory: reads ``config.json`` (architecture hyperparameters --
    the training-only fields it also contains, e.g. ``learning_rate``,
    are simply unused by :func:`~odyssey.training.train.build_model`),
    ``vocabulary.json``, ``quantile_binner.json``, and a checkpoint --
    ``checkpoint_path`` if given (e.g. ``run_dir / "checkpoint_best.pt"``
    to evaluate the lowest-val-loss checkpoint rather than wherever
    training happened to stop), else the latest available one (see
    :func:`_latest_checkpoint`).
    """
    run_dir = Path(run_dir)
    config = TrainingConfig(**json.loads((run_dir / "config.json").read_text()))
    vocab = Vocabulary.load(run_dir / "vocabulary.json")
    binner = QuantileBinner.load(run_dir / "quantile_binner.json")

    model = build_model(config, vocab_size=len(vocab), num_concepts=len(CONCEPTS))
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path else _latest_checkpoint(run_dir)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def _build_type_lookup(vocab: Vocabulary, device: str) -> torch.Tensor:
    """``(vocab_size,)`` token id -> code-type id, for a vectorized per-chunk lookup.

    Precomputed once rather than decoding each target token on every
    chunk (real held-out passes have hundreds of thousands of real
    positions).
    """
    lookup = torch.zeros(len(vocab), dtype=torch.long)
    for token_id, token in vocab.id_to_token.items():
        lookup[token_id] = code_type(token)
    return lookup.to(device)


@dataclass
class _RunningBucket:
    """Cross-entropy/top-k sums for one slice of targets, updated chunk by chunk.

    Sums (not means) so weighted-averaging across chunks of different
    sizes reduces to a single division at the end -- exactly what
    ``F.cross_entropy(..., reduction="sum")`` plus a running count gives.
    """

    ce_sum: float = 0.0
    hit_sums: Dict[int, int] = field(default_factory=dict)
    n: int = 0
    set_hit_sum: int = 0
    n_set: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        top_k: Sequence[int],
        set_valid: Optional[torch.Tensor] = None,
        set_hit: Optional[torch.Tensor] = None,
    ) -> None:
        """``logits`` is ``(n, vocab_size)``, ``targets`` is ``(n,)``: one chunk."""
        if targets.numel() == 0:
            return
        self.n += int(targets.numel())
        if set_valid is not None and set_hit is not None:
            self.n_set += int(set_valid.sum().item())
            self.set_hit_sum += int(set_hit.sum().item())
        self.ce_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        top_k_preds = logits.topk(max(top_k), dim=-1).indices
        hits = top_k_preds == targets.unsqueeze(-1)
        for k in top_k:
            self.hit_sums[k] = self.hit_sums.get(k, 0) + int(
                hits[:, :k].any(dim=-1).sum().item()
            )

    def finalize(self) -> TaskMetrics:
        """Combine the running sums into one :class:`TaskMetrics`."""
        if self.n == 0:
            raise ValueError("no non-ignored predictions to compute metrics over")
        cross_entropy = self.ce_sum / self.n
        return TaskMetrics(
            cross_entropy=cross_entropy,
            perplexity=float(torch.exp(torch.tensor(cross_entropy))),
            top1_accuracy=self.hit_sums.get(1, 0) / self.n,
            top5_accuracy=self.hit_sums.get(5, 0) / self.n,
            n_predictions=self.n,
            set_top1_accuracy=(self.set_hit_sum / self.n_set if self.n_set else None),
            n_set_predictions=self.n_set or None,
        )


def _block_set_hits(
    top1: torch.Tensor,
    targets: torch.Tensor,
    *,
    times: torch.Tensor,
    subject_ids: torch.Tensor,
    real_mask: torch.Tensor,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-position "top-1 named some event in the target's time block".

    Sequences are time-sorted per subject, so a same-timestamp event block
    is a contiguous run; block membership is recoverable from the chunk's
    own input timestamps (the target at position ``j`` is the input token
    at ``j+1``). Fully vectorized: each position gets a composite
    ``block_id * vocab_size + token`` key, and ``torch.isin`` of predicted
    keys against target keys answers membership with no Python loops. The
    final position of each lane has no in-chunk target timestamp and is
    excluded (`n_set_predictions` counts what remains); blocks never span
    subjects because a subject change starts a new block.
    """
    lanes, chunk = targets.shape
    tgt_t = times[:, 1:]
    tgt_s = subject_ids[:, 1:]
    tgt = targets[:, : chunk - 1]
    pred = top1[:, : chunk - 1]
    valid = real_mask[:, : chunk - 1]

    new_block = torch.ones_like(tgt_s, dtype=torch.bool)
    new_block[:, 1:] = (tgt_t[:, 1:] != tgt_t[:, :-1]) | (tgt_s[:, 1:] != tgt_s[:, :-1])
    lane_offset = torch.arange(lanes, device=targets.device).unsqueeze(1) * (chunk + 1)
    block_id = new_block.long().cumsum(dim=1) + lane_offset

    member_keys = torch.where(
        valid,
        block_id * vocab_size + tgt,
        torch.full_like(tgt, -1),
    )
    query_keys = block_id * vocab_size + pred
    hit = torch.isin(query_keys, member_keys) & valid

    set_valid = torch.zeros_like(real_mask)
    set_hit = torch.zeros_like(real_mask)
    set_valid[:, : chunk - 1] = valid
    set_hit[:, : chunk - 1] = hit
    return set_valid, set_hit


class _RunningTaskMetrics:
    """Streaming equivalent of ``compute_task_metrics``/``..._by_code_type``.

    Those two functions need the full ``(N, vocab_size)`` logits tensor
    materialized at once -- fine for one training batch, but for a real
    held-out split with hundreds of thousands of real positions, holding
    onto every chunk's logits until the very end doesn't scale.
    Confirmed the hard way: exactly this accumulation pattern (an
    earlier version of :func:`run_streaming_inference`) OOM-killed the
    actual training job it happened to be running alongside, evaluating
    against only 5 real held-out shards. This accumulates the same
    quantities incrementally instead, holding only running scalars
    (never more than one chunk's logits at a time -- the same transient
    cost the model's own forward pass already pays).
    """

    def __init__(
        self, vocab: Vocabulary, *, device: str, top_k: Sequence[int] = (1, 5)
    ) -> None:
        self._top_k = top_k
        self._type_lookup = _build_type_lookup(vocab, device)
        self.overall = _RunningBucket()
        self.by_type: Dict[str, _RunningBucket] = {}

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        set_valid: Optional[torch.Tensor] = None,
        set_hit: Optional[torch.Tensor] = None,
    ) -> None:
        """Fold in one chunk's real-position ``(logits, targets)``."""
        if targets.numel() == 0:
            return
        self.overall.update(logits, targets, self._top_k, set_valid, set_hit)
        target_types = self._type_lookup[targets]
        for type_id, name in _CODE_TYPE_NAMES.items():
            type_mask = target_types == type_id
            if type_mask.any():
                self.by_type.setdefault(name, _RunningBucket()).update(
                    logits[type_mask],
                    targets[type_mask],
                    self._top_k,
                    set_valid[type_mask] if set_valid is not None else None,
                    set_hit[type_mask] if set_hit is not None else None,
                )

    def finalize(self) -> Tuple[TaskMetrics, Dict[str, TaskMetrics]]:
        return self.overall.finalize(), {
            name: bucket.finalize() for name, bucket in self.by_type.items()
        }


def run_streaming_inference(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    supervision: ConceptSupervision = "stay",
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

    task_stats = _RunningTaskMetrics(vocab, device=device)
    end_subject_ids: List[torch.Tensor] = []
    end_visit_ids: List[torch.Tensor] = []
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
                set_valid, set_hit = _block_set_hits(
                    logits.argmax(dim=-1),
                    chunk.targets,
                    times=chunk.batch.aux.time_stamps,
                    subject_ids=chunk.subject_ids,
                    real_mask=real,
                    vocab_size=logits.shape[-1],
                )
                task_stats.update(
                    logits[real],
                    chunk.targets[real],
                    set_valid[real],
                    set_hit[real],
                )

            pool_mask = chunk.patient_end if supervision == "stay" else chunk.visit_end
            if pool_mask.any():
                end_subject_ids.append(
                    _pool_patient_ends(chunk.subject_ids, pool_mask).cpu()
                )
                end_visit_ids.append(
                    _pool_patient_ends(chunk.visit_ids, pool_mask).cpu()
                )
                end_concept_probs.append(
                    _pool_patient_ends(bottleneck_out.concept_probs, pool_mask).cpu()
                )
                end_observability_probs.append(
                    _pool_patient_ends(
                        bottleneck_out.observability_probs, pool_mask
                    ).cpu()
                )
                end_concept_embeddings.append(
                    _pool_patient_ends(
                        bottleneck_out.concept_embeddings, pool_mask
                    ).cpu()
                )
                end_unknown_embedding.append(
                    _pool_patient_ends(
                        bottleneck_out.unknown_embedding, pool_mask
                    ).cpu()
                )

    task_metrics, task_metrics_by_code_type = task_stats.finalize()

    if not end_subject_ids:
        # No chunk ever had a real pool_mask position -- e.g. supervision
        # is "visit" but nothing in this split has a real hadm_id, so
        # chunk.visit_end never fires. Forecasting quality (task_metrics
        # above) is still valid, since it never depends on pooling; only
        # the pooled concept/observability/orthogonality questions have
        # nothing to score.
        logger.warning(
            "[inference] no %s-scoped pool positions were ever produced -- "
            "skipping concept/observability/orthogonality metrics",
            supervision,
        )
        return InferenceResults(
            task_metrics=task_metrics,
            task_metrics_by_code_type=task_metrics_by_code_type,
            concept_metrics=[],
            observability_metrics=[],
            orthogonality=float("nan"),
            n_patient_ends_scored=0,
        )

    subject_ids = torch.cat(end_subject_ids)
    concept_probs = torch.cat(end_concept_probs)
    observability_probs = torch.cat(end_observability_probs)
    concept_embeddings = torch.cat(end_concept_embeddings)
    unknown_embedding = torch.cat(end_unknown_embedding)

    concept_names = [c.name for c in CONCEPTS]
    if supervision == "visit":
        visit_ids = torch.cat(end_visit_ids)
        labels = _gather_by_visit(subject_ids, visit_ids, concept_labels)  # type: ignore[arg-type]
        masks = _gather_by_visit(subject_ids, visit_ids, concept_mask)  # type: ignore[arg-type]
    else:
        labels = _gather_by_subject(subject_ids, concept_labels)  # type: ignore[arg-type]
        masks = _gather_by_subject(subject_ids, concept_mask)  # type: ignore[arg-type]
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
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> InferenceResults:
    """End-to-end: load a trained run, score it against a held-out split."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )

    logger.info("[inference] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events, enabled=getattr(config, "normalize_medications", False)
    )
    events_binned = add_value_tokens(raw_events, binner)

    supervision = getattr(config, "concept_supervision", "stay")
    logger.info("[inference] labeling concepts (%s-scoped)", supervision)
    concept_labels: ConceptLabelDict
    concept_mask: ConceptLabelDict
    if supervision == "visit":
        concept_labels, concept_mask = build_visit_concept_label_dicts(
            raw_events, CONCEPTS
        )
    else:
        concept_labels, concept_mask = build_concept_label_dicts(raw_events, CONCEPTS)
    del raw_events

    logger.info("[inference] running streaming inference")
    return run_streaming_inference(
        model,
        events_binned,
        vocab,
        concept_labels,
        concept_mask,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
        supervision=supervision,  # type: ignore[arg-type]
    )


def results_to_dict(results: InferenceResults) -> Dict[str, object]:
    """Plain-JSON-able view of :class:`InferenceResults` (already all plain types)."""
    from dataclasses import asdict  # noqa: PLC0415

    return asdict(results)


@dataclass(frozen=True)
class _CliArgs:
    """Parsed CLI args for :func:`evaluate_run`, mirroring ``training.train``'s CLI."""

    run_dir: Path
    held_out_shard_dir: str
    output_json: Path
    checkpoint_path: Path
    max_shards: Optional[int]
    num_lanes: int
    chunk_size: int


def _parse_args() -> _CliArgs:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename within --run-dir (default: checkpoint_best.pt).",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    return _CliArgs(
        run_dir=run_dir,
        held_out_shard_dir=args.held_out_shard_dir,
        output_json=Path(args.output_json),
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        max_shards=args.max_shards,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli_args = _parse_args()
    results = evaluate_run(
        cli_args.run_dir,
        cli_args.held_out_shard_dir,
        max_shards=cli_args.max_shards,
        num_lanes=cli_args.num_lanes,
        chunk_size=cli_args.chunk_size,
        checkpoint_path=cli_args.checkpoint_path,
    )
    cli_args.output_json.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output_json.write_text(json.dumps(results_to_dict(results), indent=2))
    logger.info("[inference] wrote results to %s", cli_args.output_json)


__all__ = [
    "InferenceResults",
    "load_run",
    "load_and_bin_held_out",
    "run_streaming_inference",
    "evaluate_run",
    "results_to_dict",
]
