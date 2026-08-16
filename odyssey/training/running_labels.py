"""Per-position *running* concept labels for one streaming chunk.

Concept labels are visit- (or stay-) scoped and retrospective: "did this
happen at some point during the visit". A concept's value at a given
moment is a different thing -- false before the concept first triggers,
true from then on. Anything that injects concept values into the model
position by position (CEM-style interventions at evaluation time,
:mod:`odyssey.inference.interventions`; intervention-aware RandInt
training in ``ConceptBottleneckSequenceModel.compute_streaming_loss``)
needs the running value, which this module derives from the retrospective
label plus each concept's first-trigger time
(:func:`~odyssey.training.data.build_visit_concept_first_times`).

Injecting the retrospective label at every position instead would feed
the bottleneck a fact about the future that the model's running concept
state has no business knowing; an earlier evaluation harness did exactly
that, and "truth" hurt more than "flip" purely because, before the event,
the flipped label was the accurate one.
"""

from typing import Optional, Tuple

import torch

from odyssey.data.streaming import StreamingChunk
from odyssey.models.concept_bottleneck import BottleneckIntervention
from odyssey.models.sequence_model import ConceptLabelDict, ConceptSupervision


def position_running_labels(
    chunk: StreamingChunk,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: Optional[ConceptLabelDict],
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive running ground-truth labels and observed-masks per chunk position.

    Returns ``(labels, observed)``, each ``(lanes, T, num_concepts)`` on
    the chunk's device. A concept is labeled true at a position only
    from its first-trigger time onward (``concept_first_times``, hours on
    the sequence's own time origin, ``inf`` for never). Positions with no
    dictionary entry (padding lanes, events outside any visit under visit
    scoping) get ``observed = 0`` everywhere, so no injection applies
    there. With ``concept_first_times=None`` (or a key missing from it)
    the retrospective label passes through unchanged -- only valid for
    concepts that are constant across the sequence; callers that need
    running semantics must supply first times.
    """
    sid = chunk.subject_ids
    lanes, chunk_len = sid.shape

    if supervision == "visit":
        keys = torch.stack([sid, chunk.visit_ids], dim=-1).reshape(-1, 2)
    else:
        keys = sid.reshape(-1, 1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    # Everything lives on the chunk's device: indexing with a
    # device-mismatched index tensor raises, and this path only runs
    # once a chunk actually has an intervention to apply, so a CPU
    # default here would be silent until a real CUDA run.
    unique_labels = torch.zeros(unique_keys.shape[0], num_concepts, device=sid.device)
    unique_observed = torch.zeros(unique_keys.shape[0], num_concepts, device=sid.device)
    unique_first = torch.full(
        (unique_keys.shape[0], num_concepts), float("-inf"), device=sid.device
    )
    first_times_dict = concept_first_times or {}
    for i, key in enumerate(unique_keys.tolist()):
        lookup = (key[0], key[1]) if supervision == "visit" else key[0]
        label = concept_labels.get(lookup)  # type: ignore[arg-type]
        if label is None:
            continue
        unique_labels[i] = label.float().to(sid.device)
        mask = concept_mask.get(lookup)  # type: ignore[arg-type]
        if mask is not None:
            unique_observed[i] = mask.float().to(sid.device)
        first = first_times_dict.get(lookup)  # type: ignore[arg-type]
        if first is not None:
            unique_first[i] = first.float().to(sid.device)

    labels_scoped = unique_labels[inverse].view(lanes, chunk_len, num_concepts)
    observed = unique_observed[inverse].view(lanes, chunk_len, num_concepts)
    first_times = unique_first[inverse].view(lanes, chunk_len, num_concepts)
    now = chunk.batch.aux.time_stamps.unsqueeze(-1)  # (lanes, T, 1) hours
    labels = labels_scoped * (now >= first_times).float()
    return labels, observed


def randint_intervention(
    chunk: StreamingChunk,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
    prob: float,
    generator: Optional[torch.Generator] = None,
) -> Optional[BottleneckIntervention]:
    """Build one training step's RandInt intervention, or None if ``prob <= 0``.

    CEM's intervention-aware training (Espinosa Zarlenga et al., 2022,
    Section 4): at each position, each observed concept's mixing
    probability is replaced by its running ground-truth value with
    probability ``prob``, independently. The task head therefore sees
    trustworthy concept values often enough to learn to rely on them,
    which is what makes test-time interventions effective; without it,
    task information can route through the concept embeddings while the
    calibrated probability stays causally inert (the magnitude-controlled
    intervention test showed exactly that on the retrained subset run).
    Unobserved concepts are never substituted -- there is no ground truth
    to feed there.
    """
    if prob <= 0.0:
        return None
    labels, observed = position_running_labels(
        chunk,
        concept_labels,
        concept_mask,
        concept_first_times,
        supervision=supervision,
        num_concepts=num_concepts,
    )
    coin = torch.rand(labels.shape, generator=generator, device=labels.device)
    return BottleneckIntervention(
        probs=labels, probs_mask=(observed > 0) & (coin < prob)
    )
