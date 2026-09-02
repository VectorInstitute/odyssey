"""Lifted tokens: the events that say "this concept is present".

Steerling's express target (their 10.2.4) is the set of vocabulary tokens
with the highest lift, ``P(token | c) / P(token)``, under a minimum-support
filter. Here the tags are the concept rules' running labels, so "under the
concept" means "at a position where the concept has triggered". Used by
the steering benchmark to read whether a push moves the next-event
distribution toward the concept, and by steering training as the target
the express loss pushes mass onto.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import PackedLaneSampler, move_to_device
from odyssey.models.sequence_model import ConceptLabelDict, ConceptSupervision
from odyssey.training.running_labels import position_running_labels


def lifted_token_sets(
    patients: Iterable[PatientSequence],
    *,
    vocab_size: int,
    num_concepts: int,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    supervision: ConceptSupervision,
    top_k: int = 25,
    min_count: int = 20,
    min_share: float = 0.005,
    min_lift: float = 2.0,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
) -> dict[int, list[int]]:
    """Per concept, the ``top_k`` next-event tokens with the highest lift.

    Lift is ``P(token | concept active) / P(token)`` over target positions
    of a labeled stream, with running labels so "active" means "has
    triggered by this position"; Steerling defines the express target the
    same way over its chunk tags, with a minimum-support filter. Tokens
    seen fewer than ``min_count`` times under the concept are ignored so a
    rare code cannot top the list on two occurrences, and the floor rises
    with prevalence: a token must also account for at least ``min_share``
    of the concept's positions. Without that, the highest-lift tokens for
    the MAP-hypotension concept on MIMIC-IV were pressure-ulcer
    measurements seen a few dozen
    times. Only tokens with lift above 1 are kept.
    """
    total = torch.zeros(vocab_size, dtype=torch.float64)
    per_concept = torch.zeros(num_concepts, vocab_size, dtype=torch.float64)
    sampler = PackedLaneSampler(
        iter(patients), num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    for chunk in sampler:
        chunk = move_to_device(chunk, device)  # noqa: PLW2901
        labels, observed = position_running_labels(
            chunk,
            concept_labels,
            concept_mask,
            concept_first_times,
            supervision=supervision,
            num_concepts=num_concepts,
        )
        real = chunk.real_mask
        if not real.any():
            continue
        targets = chunk.targets[real]
        active = (labels[real] * observed[real]).to(torch.float64)  # (N, k)
        # Scatter-add rather than a dense one-hot: at 64 lanes x 512 positions
        # the one-hot over a 20k vocabulary is 5 GB and does not fit beside
        # a running eval chain.
        total += torch.bincount(targets, minlength=vocab_size).to(torch.float64).cpu()
        per_concept_dev = torch.zeros(
            num_concepts, vocab_size, dtype=torch.float64, device=targets.device
        )
        per_concept_dev.index_add_(1, targets, active.T)
        per_concept += per_concept_dev.cpu()
    return rank_by_lift(
        total,
        per_concept,
        top_k=top_k,
        min_count=min_count,
        min_share=min_share,
        min_lift=min_lift,
    )


def rank_by_lift(
    total: torch.Tensor,
    per_concept: torch.Tensor,
    *,
    top_k: int,
    min_count: int,
    min_share: float = 0.0,
    min_lift: float = 1.0,
) -> dict[int, list[int]]:
    """Top-``top_k`` tokens by ``P(token | c) / P(token)`` per concept.

    ``total`` is ``(vocab,)`` target counts over the stream and
    ``per_concept`` is ``(num_concepts, vocab)`` counts at positions where
    each concept is active. A token needs at least ``min_count``
    occurrences under the concept and at least ``min_share`` of the
    concept's positions, and lift above ``min_lift`` (at least above 1): a
    token barely over-represented under a concept says nothing about it,
    and with a share floor alone the sets filled with generic ICU care-plan
    items shared by every concept.
    """
    base = total / total.sum().clamp_min(1.0)
    sets: dict[int, list[int]] = {}
    for c in range(per_concept.shape[0]):
        counts = per_concept[c]
        support = max(float(min_count), min_share * float(counts.sum()))
        cond = counts / counts.sum().clamp_min(1.0)
        lift = torch.where(
            counts >= support, cond / base.clamp_min(1e-12), torch.zeros_like(cond)
        )
        threshold = max(min_lift, 1.0)
        keep = int(min(top_k, int((lift > threshold).sum().item())))
        sets[c] = (
            [int(i) for i in torch.topk(lift, k=keep).indices.tolist()] if keep else []
        )
    return sets


__all__ = ["lifted_token_sets", "rank_by_lift"]
