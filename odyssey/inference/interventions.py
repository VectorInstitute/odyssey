"""Concept intervention and completeness evaluation.

The one architectural claim a concept bottleneck makes beyond ordinary
sequence modeling is that the supervised concepts *mediate* prediction:
the task head reads a mixture steered by the concept probabilities, so
editing those probabilities should causally move the forecasts. This
module tests that claim directly, CEM/CBGM-style, by re-running the
streaming next-event evaluation under do()-style edits inside the
bottleneck (:class:`~odyssey.models.concept_bottleneck.BottleneckIntervention`)
and comparing task metrics across modes:

- ``none`` -- the unedited baseline; must reproduce the standard
  evaluation's numbers.
- ``truth`` -- replace each known concept's mixing probability with its
  ground-truth rule label wherever that label is observed. If concepts
  causally steer prediction, perfect concept information should *help*
  (or at minimum not hurt) next-event accuracy; a model that ignores its
  bottleneck shows no movement.
- ``flip`` -- feed ``1 - label`` on the same positions. The mirror
  image: reliance on the concept channel shows up as damage.
- ``random`` -- feed coin-flip values on the same positions. Separates
  "any perturbation hurts" from "wrong information hurts": a gap
  between ``random`` and ``flip`` means the model reads the *direction*
  of the concept values, not just their stability.
- ``zero_known`` / ``zero_unknown`` -- zero the known concepts' (resp.
  the unknown channel's) mixed embeddings. The completeness probe: how
  the task signal is apportioned between the supervised, interpretable
  channel and the unsupervised one. A bottleneck whose entire task
  performance survives ``zero_known`` is interpretable-in-name-only --
  the concepts would be a decorative side channel.

Intervened values are applied per position from the same visit- or
stay-scoped label dictionaries used for supervision, gated by the
observed mask (unobserved concepts keep the model's own probability, in
every mode -- there is no ground truth to feed there).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.inference.run_inference import (
    _CODE_TYPE_NAMES,
    _build_type_lookup,
    load_run,
)
from odyssey.models.concept_bottleneck import BottleneckIntervention
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
)
from odyssey.training.data import (
    build_concept_label_dicts,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

INTERVENTION_MODES = (
    "none",
    "truth",
    "flip",
    "random",
    "zero_known",
    "zero_unknown",
)


@dataclass(frozen=True)
class InterventionResult:
    """Task metrics for one intervention mode over the held-out stream."""

    mode: str
    n_predictions: int
    top1_accuracy: float
    mean_task_loss: float
    top1_by_code_type: Dict[str, float] = field(default_factory=dict)
    n_by_code_type: Dict[str, int] = field(default_factory=dict)
    n_intervened_positions: int = 0
    """Positions where at least one concept's mixing probability was
    actually replaced (0 for none/zero_* modes, which edit embeddings
    or nothing)."""


def _position_labels(
    chunk: StreamingChunk,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-position ground-truth labels and observed-masks for one chunk.

    Returns ``(labels, observed)``, each ``(lanes, T, num_concepts)``.
    Positions with no dictionary entry (padding lanes, events outside
    any visit under visit scoping) get ``observed = 0`` everywhere, so
    no intervention applies there and the model's own probability is
    kept.
    """
    sid = chunk.subject_ids
    lanes, chunk_len = sid.shape
    labels_out = torch.zeros(lanes, chunk_len, num_concepts)
    observed_out = torch.zeros(lanes, chunk_len, num_concepts)

    if supervision == "visit":
        keys = torch.stack([sid, chunk.visit_ids], dim=-1).reshape(-1, 2)
    else:
        keys = sid.reshape(-1, 1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    unique_labels = torch.zeros(unique_keys.shape[0], num_concepts)
    unique_observed = torch.zeros(unique_keys.shape[0], num_concepts)
    for i, key in enumerate(unique_keys.tolist()):
        lookup = (key[0], key[1]) if supervision == "visit" else key[0]
        label = concept_labels.get(lookup)  # type: ignore[arg-type]
        if label is None:
            continue
        unique_labels[i] = label.float().cpu()
        mask = concept_mask.get(lookup)  # type: ignore[arg-type]
        if mask is not None:
            unique_observed[i] = mask.float().cpu()

    labels_out = unique_labels[inverse].view(lanes, chunk_len, num_concepts)
    observed_out = unique_observed[inverse].view(lanes, chunk_len, num_concepts)
    return labels_out, observed_out


def _chunk_intervention(
    chunk: StreamingChunk,
    mode: str,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
    device: str,
    rng: torch.Generator,
) -> Optional[BottleneckIntervention]:
    """Build the per-position intervention for one chunk, or None."""
    if mode == "none":
        return None
    if mode == "zero_known":
        return BottleneckIntervention(zero_known=True)
    if mode == "zero_unknown":
        return BottleneckIntervention(zero_unknown=True)

    labels, observed = _position_labels(
        chunk,
        concept_labels,
        concept_mask,
        supervision=supervision,
        num_concepts=num_concepts,
    )
    if mode == "truth":
        values = labels
    elif mode == "flip":
        values = 1.0 - labels
    elif mode == "random":
        values = (torch.rand(labels.shape, generator=rng) < 0.5).float()
    else:
        raise ValueError(f"unknown intervention mode: {mode!r}")
    return BottleneckIntervention(
        probs=values.to(device), probs_mask=observed.bool().to(device)
    )


def run_streaming_intervention(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    mode: str,
    supervision: ConceptSupervision = "stay",
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    seed: int = 0,
) -> InterventionResult:
    """Score next-event prediction under one intervention mode.

    The identical streaming pass as
    :func:`~odyssey.inference.run_inference.run_streaming_inference`
    (same sampler, same state carrying), with the bottleneck edited per
    :data:`INTERVENTION_MODES`. Deterministic for a given ``seed``
    (which only the ``random`` mode consumes).
    """
    if mode not in INTERVENTION_MODES:
        raise ValueError(
            f"unknown intervention mode {mode!r}; known: {INTERVENTION_MODES}"
        )
    model.eval()
    num_concepts = model.bottleneck.num_concepts
    patients = iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    rng = torch.Generator().manual_seed(seed)
    type_lookup = _build_type_lookup(vocab, device)

    n = 0
    top1_hits = 0
    loss_sum = 0.0
    n_intervened = 0
    type_n: Dict[int, int] = {}
    type_hits: Dict[int, int] = {}

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            intervention = _chunk_intervention(
                chunk,
                mode,
                concept_labels,
                concept_mask,
                supervision=supervision,
                num_concepts=num_concepts,
                device=device,
                rng=rng,
            )
            logits, _, state = model(
                chunk.batch,
                state=state,
                reset_mask=chunk.reset_mask,
                intervention=intervention,
            )
            real = chunk.real_mask
            if intervention is not None and intervention.probs_mask is not None:
                input_real = chunk.subject_ids != NO_SUBJECT
                n_intervened += int(
                    (intervention.probs_mask.any(dim=-1) & input_real).sum().item()
                )
            if not real.any():
                continue
            real_logits = logits[real]
            real_targets = chunk.targets[real]
            n += int(real_targets.shape[0])
            preds = real_logits.argmax(dim=-1)
            hits = preds == real_targets
            top1_hits += int(hits.sum().item())
            loss_sum += float(
                F.cross_entropy(
                    real_logits, real_targets, ignore_index=PAD_ID, reduction="sum"
                ).item()
            )
            target_types = type_lookup[real_targets]
            for type_id in torch.unique(target_types).tolist():
                sel = target_types == type_id
                type_n[type_id] = type_n.get(type_id, 0) + int(sel.sum().item())
                type_hits[type_id] = type_hits.get(type_id, 0) + int(
                    hits[sel].sum().item()
                )

    return InterventionResult(
        mode=mode,
        n_predictions=n,
        top1_accuracy=top1_hits / n if n else float("nan"),
        mean_task_loss=loss_sum / n if n else float("nan"),
        top1_by_code_type={
            _CODE_TYPE_NAMES[tid]: type_hits[tid] / type_n[tid]
            for tid in sorted(type_n)
            if tid in _CODE_TYPE_NAMES
        },
        n_by_code_type={
            _CODE_TYPE_NAMES[tid]: type_n[tid]
            for tid in sorted(type_n)
            if tid in _CODE_TYPE_NAMES
        },
        n_intervened_positions=n_intervened,
    )


def evaluate_interventions(
    run_dir: Union[str, Path],
    held_out_shard_dir: Union[str, Path],
    *,
    modes: Sequence[str] = INTERVENTION_MODES,
    max_shards: Optional[int] = None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    seed: int = 0,
) -> List[InterventionResult]:
    """End-to-end: load a trained run, score every intervention mode.

    Data preparation matches
    :func:`~odyssey.inference.run_inference.evaluate_run` exactly (same
    normalization, binning, and label scoping from the run's own
    config), so the ``none`` mode is directly comparable to the standard
    evaluation and every other mode is directly comparable to ``none``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )

    logger.info("[interventions] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events, enabled=getattr(config, "normalize_medications", False)
    )
    source = getattr(config, "source", "mimic_iv")
    concepts = concepts_for_source(source)
    events_binned = add_value_tokens(raw_events, binner, source=source)

    supervision: ConceptSupervision = getattr(config, "concept_supervision", "stay")
    if supervision == "visit":
        concept_labels, concept_mask = build_visit_concept_label_dicts(
            raw_events, concepts
        )
    else:
        concept_labels, concept_mask = build_concept_label_dicts(raw_events, concepts)
    del raw_events

    results = []
    for mode in modes:
        logger.info("[interventions] scoring mode %r", mode)
        results.append(
            run_streaming_intervention(
                model,
                events_binned,
                vocab,
                concept_labels,
                concept_mask,
                mode=mode,
                supervision=supervision,
                num_lanes=num_lanes,
                chunk_size=chunk_size,
                device=device,
                seed=seed,
            )
        )
        baseline = results[0]
        latest = results[-1]
        logger.info(
            "[interventions] %s: top1 %.4f (delta vs none %+0.4f), loss %.4f",
            mode,
            latest.top1_accuracy,
            latest.top1_accuracy - baseline.top1_accuracy,
            latest.mean_task_loss,
        )
    return results


def _main() -> None:
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
    parser.add_argument("--modes", nargs="*", default=list(INTERVENTION_MODES))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = evaluate_interventions(
        run_dir,
        args.held_out_shard_dir,
        modes=args.modes,
        max_shards=args.max_shards,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info("[interventions] wrote %d modes to %s", len(results), out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _main()


__all__ = [
    "INTERVENTION_MODES",
    "InterventionResult",
    "run_streaming_intervention",
    "evaluate_interventions",
]
