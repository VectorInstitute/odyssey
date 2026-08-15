"""Per-patient detail traces and diverse case selection for qualitative eval.

Two separate concerns, kept in one module because they're always used
together:

- :func:`select_diverse_cases` picks a small, deliberately varied set of
  held-out subjects (mundane and acute, short and long stays) from cheap,
  already-computed signals (sequence length, which known concepts
  triggered) -- no model forward pass needed, so this runs over every
  held-out subject.
- :func:`extract_patient_case` runs the model over *one* selected
  patient's full sequence (a single whole-sequence forward pass, not
  streaming: a single patient's stay is short enough to not need
  chunking, and a clean, uninterrupted trace is what a qualitative case
  study visualization wants -- streaming's synthetic resets and
  chunk-boundary state carrying are training-time concerns, not
  relevant here) and returns a rich, per-timestep trace: predicted
  next-token distribution, concept/observability probabilities, and
  where the true next token ranked in the model's own prediction.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl
import torch

from odyssey.data.sequences import PatientSequence, collate_patient_sequences
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.train import _move_chunk_to_device


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
) -> PatientCaseTrace:
    """Run ``model`` over one patient's full sequence and trace every position."""
    model.eval()
    batch = collate_patient_sequences([seq], padding_idx=PAD_ID)
    batch = _move_chunk_to_device(batch, device)

    with torch.no_grad():
        logits, bottleneck_out, _ = model(batch)

    n = len(seq)
    probs = torch.softmax(logits[0, :n], dim=-1)  # (n, vocab_size)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)  # (n, top_k) each

    predicted_top_k: List[List[Tuple[str, float]]] = []
    true_next_code: List[Optional[str]] = []
    true_next_rank: List[Optional[int]] = []
    for i in range(n):
        if i == n - 1:
            predicted_top_k.append([])
            true_next_code.append(None)
            true_next_rank.append(None)
            continue
        predicted_top_k.append(
            [
                (vocab.decode(int(tok_id)), float(prob))
                for tok_id, prob in zip(top_k_ids[i].tolist(), top_k_probs[i].tolist())
            ]
        )
        true_id = seq.concept_ids[i + 1]
        true_next_code.append(vocab.decode(true_id))
        # argsort descending -> position of true_id is its rank.
        rank = int((probs[i] > probs[i, true_id]).sum().item())
        true_next_rank.append(rank)

    concept_probs = bottleneck_out.concept_probs[0, :n].tolist()
    observability_probs = bottleneck_out.observability_probs[0, :n].tolist()

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


__all__ = ["PatientCaseTrace", "extract_patient_case", "select_diverse_cases"]
