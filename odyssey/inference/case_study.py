"""Per-patient detail traces and diverse case selection for qualitative eval.

Two separate concerns, kept in one module because they're always used
together:

- :func:`select_diverse_cases` picks a small, deliberately varied set of
  held-out subjects (mundane and acute, short and long stays) from cheap,
  already-computed signals (sequence length, which known concepts
  triggered) -- no model forward pass needed, so this runs over every
  held-out subject.
- :func:`extract_patient_case` runs the model over *one* selected
  patient's sequence through the exact chunked, state-carrying
  streaming path training and quantitative evaluation use (a single
  lane, no synthetic resets), and returns a rich, per-timestep trace:
  predicted next-token distribution, concept/observability
  probabilities, and where the true next token ranked in the model's
  own prediction. An earlier version ran one whole-sequence
  full-attention forward pass instead, on the theory that chunking is
  a training-time concern; that was backwards -- the model was only
  ever trained on ``chunk_size``-token windows with carried recurrent
  state, so an un-chunked pass over a long stay is a regime it never
  saw, and the traces it produced were not evidence about deployed
  behavior. Chunk boundaries are seamless by construction (consecutive
  windows overlap by exactly one token), so the stitched trace still
  covers every position exactly once.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import polars as pl
import torch

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import PatientSequence, build_patient_sequence
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import probability_within
from odyssey.training.data import build_concept_label_dicts, load_meds_shards
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatientCaseTrace:
    """A full per-timestep model trace over one patient's tokenized stay."""

    subject_id: int
    times: List[float]
    """Hours since this patient's first event, one per position."""

    input_codes: List[str]
    predicted_top_k: List[List[Tuple[str, float]]]
    """Per position, the model's top-k (code, probability) predictions
    for the *next* token -- empty for the last position (no next token)."""

    true_next_code: List[Optional[str]]
    """The actual next code, or None at the last position."""

    true_next_rank: List[Optional[int]]
    """0-indexed rank of the true next code among the model's own
    predicted probabilities (0 = the model's top prediction was
    correct); None where there is no true next token."""

    concept_probs: List[List[float]]
    """Per position, per known concept -- the bottleneck's running
    concept-activation probability at that point in the stay."""

    observability_probs: List[List[float]]
    concept_names: List[str]
    concept_labels: List[float]
    """This patient's final, whole-stay concept labels (1.0/0.0)."""

    concept_observed: List[float]
    """Whether each concept label is real (1.0) or never-observed (0.0)."""

    event_risk_names: List[str] = field(default_factory=list)
    """Alert events with a hazard head in this model (empty otherwise)."""

    event_risk_24h: List[List[float]] = field(default_factory=list)
    """Per position, per alert event: the head's P(event within 24h) --
    the alert curve a clinician would watch over the stay."""


def extract_patient_case(
    model: ConceptBottleneckSequenceModel,
    seq: PatientSequence,
    vocab: Vocabulary,
    concept_names: Sequence[str],
    *,
    concept_labels: Optional[torch.Tensor] = None,
    concept_mask: Optional[torch.Tensor] = None,
    device: str = "cuda",
    top_k: int = 5,
    chunk_size: int = 256,
) -> PatientCaseTrace:
    """Trace one patient position-by-position under the streaming regime.

    Streams the sequence through the model exactly the way training and
    :func:`~odyssey.inference.run_inference.run_streaming_inference`
    do -- ``chunk_size``-token windows with carried recurrent state, one
    lane, no synthetic resets -- so every per-position probability in
    the trace comes from the operating regime the model was actually
    trained in. Pass the training run's own ``chunk_size``.
    """
    model.eval()
    sampler = PackedLaneSampler(
        iter([seq]), num_lanes=1, chunk_size=chunk_size, reset_prob=0.0
    )

    predicted_top_k: List[List[Tuple[str, float]]] = []
    true_next_code: List[Optional[str]] = []
    true_next_rank: List[Optional[int]] = []
    concept_probs: List[List[float]] = []
    observability_probs: List[List[float]] = []
    event_heads = getattr(model, "event_heads", None)
    event_risk: List[List[float]] = []

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            fwd = model.forward_with_features(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            logits, bottleneck_out, state = fwd.logits, fwd.bottleneck, fwd.state
            # Case traces read concept/observability probabilities, which
            # only the bottleneck variant produces (fwd.bottleneck is None
            # for the baseline model).
            assert bottleneck_out is not None, (  # noqa: S101
                "extract_patient_case requires a concept-bottleneck model"
            )
            # One lane, one patient, no resets: real input positions are a
            # contiguous prefix (padding only where the lane runs out).
            input_real = chunk.subject_ids[0] != NO_SUBJECT
            n_real = int(input_real.sum().item())
            assert bool(input_real[:n_real].all())  # noqa: S101
            probs = torch.softmax(logits[0, :n_real], dim=-1)  # (n_real, vocab)
            top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)
            # real_mask means "has a valid next-token target": false only
            # at the final position of the whole sequence.
            has_target = chunk.real_mask[0, :n_real]
            targets = chunk.targets[0, :n_real]
            for i in range(n_real):
                if not bool(has_target[i]):
                    predicted_top_k.append([])
                    true_next_code.append(None)
                    true_next_rank.append(None)
                    continue
                predicted_top_k.append(
                    [
                        (vocab.decode(int(tok_id)), float(prob))
                        for tok_id, prob in zip(
                            top_k_ids[i].tolist(), top_k_probs[i].tolist()
                        )
                    ]
                )
                true_id = int(targets[i].item())
                true_next_code.append(vocab.decode(true_id))
                # argsort descending -> position of true_id is its rank.
                rank = int((probs[i] > probs[i, true_id]).sum().item())
                true_next_rank.append(rank)
            concept_probs.extend(bottleneck_out.concept_probs[0, :n_real].tolist())
            observability_probs.extend(
                bottleneck_out.observability_probs[0, :n_real].tolist()
            )
            if event_heads is not None:
                hazards = event_heads(fwd.features[0, :n_real])
                event_risk.extend(
                    probability_within(hazards, event_heads.edges, 24.0).tolist()
                )

    assert len(concept_probs) == len(seq)  # noqa: S101 -- every position, once

    zeros = [0.0] * len(concept_names)
    return PatientCaseTrace(
        subject_id=seq.subject_id,
        times=list(seq.time_stamps),
        input_codes=[vocab.decode(c) for c in seq.concept_ids],
        predicted_top_k=predicted_top_k,
        true_next_code=true_next_code,
        true_next_rank=true_next_rank,
        concept_probs=concept_probs,
        observability_probs=observability_probs,
        concept_names=list(concept_names),
        concept_labels=(
            concept_labels.tolist() if concept_labels is not None else zeros
        ),
        concept_observed=(concept_mask.tolist() if concept_mask is not None else zeros),
        event_risk_names=(
            list(event_heads.event_names) if event_heads is not None else []
        ),
        event_risk_24h=event_risk,
    )


def select_diverse_cases(
    events: pl.DataFrame,
    concept_labels: Dict[int, torch.Tensor],
    *,
    n_cases: int = 15,
    min_events: int = 10,
    seed: int = 0,
) -> List[int]:
    """Pick a deliberately varied set of held-out ``subject_id``\\ s.

    Stratifies by (sequence-length tertile) x (concepts triggered: 0 /
    1-2 / 3+), then samples roughly evenly across the non-empty strata
    -- the same spirit as the earlier, hand-picked 15-case selection in
    ``research_journal/01_patient_sequences.html`` (acute presentations
    down to routine ones, deliberately contrasted), made reproducible
    and automatic here. Only uses signals already on hand (event counts,
    concept labels), not a model forward pass, so this can run over
    every held-out subject cheaply.
    """
    # sort("subject_id"): group_by's own row order isn't guaranteed, and
    # this feeds a seeded shuffle below -- without a canonical order
    # here, the same seed would shuffle a differently-ordered list each
    # call and silently stop being deterministic (see
    # build_patient_sequence's maintain_order=True fix for the same
    # class of issue).
    event_counts = (
        events.group_by("subject_id")
        .len()
        .rename({"len": "n_events"})
        .sort("subject_id")
    )
    subject_ids = event_counts.filter(pl.col("n_events") >= min_events)[
        "subject_id"
    ].to_list()
    if not subject_ids:
        return []

    counts_by_subject = dict(
        zip(event_counts["subject_id"].to_list(), event_counts["n_events"].to_list())
    )
    lengths = sorted(counts_by_subject[sid] for sid in subject_ids)
    tertile_1 = lengths[len(lengths) // 3]
    tertile_2 = lengths[2 * len(lengths) // 3]

    def _length_bucket(sid: int) -> int:
        n = counts_by_subject[sid]
        if n <= tertile_1:
            return 0
        if n <= tertile_2:
            return 1
        return 2

    def _concept_bucket(sid: int) -> int:
        labels = concept_labels.get(sid)
        if labels is None:
            return 0
        n_triggered = int((labels > 0).sum().item())
        if n_triggered == 0:
            return 0
        if n_triggered <= 2:
            return 1
        return 2

    strata: Dict[Tuple[int, int], List[int]] = {}
    for sid in subject_ids:
        key = (_length_bucket(sid), _concept_bucket(sid))
        strata.setdefault(key, []).append(sid)

    rng = random.Random(seed)
    for bucket in strata.values():
        rng.shuffle(bucket)

    selected: List[int] = []
    stratum_keys = sorted(strata.keys())
    i = 0
    while len(selected) < n_cases and any(strata[k] for k in stratum_keys):
        key = stratum_keys[i % len(stratum_keys)]
        if strata[key]:
            selected.append(strata[key].pop())
        i += 1

    return selected


def build_case_studies(
    run_dir: Union[str, Path],
    held_out_shard_dir: Union[str, Path],
    *,
    n_cases: int = 15,
    max_shards: Optional[int] = None,
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> List[PatientCaseTrace]:
    """End-to-end: load a trained run, pick diverse held-out cases, trace each.

    Mirrors :func:`~odyssey.inference.run_inference.evaluate_run`'s
    load-a-run pattern, but for the qualitative path: selects
    ``n_cases`` diverse subjects (see :func:`select_diverse_cases`) and
    streams each through the model with the run's own trained
    ``chunk_size`` (see :func:`extract_patient_case`).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            "this evaluation needs a concept bottleneck; the run's model_kind is "
            f"{getattr(config, 'model_kind', 'bottleneck')!r}"
        )
    if getattr(config, "backbone", "hybrid") == "transformer":
        raise NotImplementedError(
            "case_study is not yet wired for backbone='transformer': this is "
            "concept-bottleneck-lever tooling, not needed for the backbone "
            "control's own subset-scale comparison (unlike run_inference and "
            "alerts, which are). Extend it only if the transformer backbone "
            "earns longer-term status."
        )

    logger.info("[case_study] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events,
        enabled=getattr(config, "normalize_medications", False),
        source=getattr(config, "source", "mimic_iv"),
    )
    raw_events = maybe_history_recap(
        raw_events, enabled=getattr(config, "history_recap", False)
    )

    source = getattr(config, "source", "mimic_iv")
    concepts = concepts_for_source(source)
    logger.info("[case_study] labeling concepts (source=%s)", source)
    concept_labels, concept_mask = build_concept_label_dicts(raw_events, concepts)

    logger.info("[case_study] selecting %d diverse cases", n_cases)
    subject_ids = select_diverse_cases(raw_events, concept_labels, n_cases=n_cases)

    logger.info("[case_study] binning values")
    events_binned = add_value_tokens(raw_events, binner, source=source)
    del raw_events

    concept_names = [c.name for c in concepts]
    traces: List[PatientCaseTrace] = []
    for subject_id in subject_ids:
        logger.info("[case_study] tracing subject %d", subject_id)
        subject_events = events_binned.filter(pl.col("subject_id") == subject_id)
        seq = build_patient_sequence(subject_events, vocab)
        traces.append(
            extract_patient_case(
                model,
                seq,
                vocab,
                concept_names,
                concept_labels=concept_labels.get(subject_id),
                concept_mask=concept_mask.get(subject_id),
                device=device,
                chunk_size=getattr(config, "chunk_size", 256),
            )
        )
    return traces


@dataclass(frozen=True)
class _CliArgs:
    """Parsed CLI args for :func:`build_case_studies`."""

    run_dir: Path
    held_out_shard_dir: str
    output_json: Path
    checkpoint_path: Path
    n_cases: int
    max_shards: Optional[int]


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
    parser.add_argument("--n-cases", type=int, default=15)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    return _CliArgs(
        run_dir=run_dir,
        held_out_shard_dir=args.held_out_shard_dir,
        output_json=Path(args.output_json),
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        n_cases=args.n_cases,
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli_args = _parse_args()
    case_traces = build_case_studies(
        cli_args.run_dir,
        cli_args.held_out_shard_dir,
        n_cases=cli_args.n_cases,
        max_shards=cli_args.max_shards,
        checkpoint_path=cli_args.checkpoint_path,
    )
    cli_args.output_json.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output_json.write_text(
        json.dumps([asdict(trace) for trace in case_traces], indent=2)
    )
    logger.info(
        "[case_study] wrote %d cases to %s", len(case_traces), cli_args.output_json
    )


__all__ = [
    "PatientCaseTrace",
    "extract_patient_case",
    "select_diverse_cases",
    "build_case_studies",
]
