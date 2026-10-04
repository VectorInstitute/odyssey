"""Stream one patient through a trained model, position by position.

Every per-patient reader (qualitative case traces, counterfactual re-scoring,
the clinician demo) needs the same loop: one lane, one patient, no synthetic
resets, ``chunk_size``-token windows with the recurrent state carried
between them -- the regime training and quantitative evaluation use, so
every per-position output is evidence about deployed behavior. That loop
lives here once instead of being copied into each reader.

A hybrid backbone's attention is chunk-local, so ``chunk_size`` must be the
training run's own value (``config.chunk_size``); a different value
changes the numbers, not just the speed.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler
from odyssey.models.sequence_model import ForwardWithFeatures, SequenceModel
from odyssey.models.time_to_event import probability_within
from odyssey.training.train import _move_chunk_to_device


@dataclass(frozen=True)
class StreamSpan:
    """One chunk's real positions of a single-patient stream.

    ``fwd`` holds the full lane-0 chunk outputs; positions ``[:n_real]`` of
    it are real, and they are absolute sequence positions
    ``[start, start + n_real)``.
    """

    start: int
    n_real: int
    fwd: ForwardWithFeatures
    has_target: torch.Tensor
    """``(n_real,)`` bool: the position has a next-token target (false only at
    the sequence's final position)."""
    targets: torch.Tensor
    """``(n_real,)`` next-token ids."""


def stream_patient(
    model: SequenceModel,
    seq: PatientSequence,
    *,
    device: str,
    chunk_size: int,
    stop_after: int | None = None,
) -> Iterator[StreamSpan]:
    """Yield the model's outputs over ``seq``, one chunk at a time.

    ``stop_after`` ends the stream once position ``stop_after`` has been
    yielded, so a caller that only needs a prefix (an index time, one visit)
    does not pay for the rest of the record. Forward passes run under
    ``torch.no_grad``; callers that compute on the yielded tensors should
    do the same.
    """
    sampler = PackedLaneSampler(
        iter([seq]), num_lanes=1, chunk_size=chunk_size, reset_prob=0.0
    )
    state = None
    offset = 0
    for raw_chunk in sampler:
        chunk = _move_chunk_to_device(raw_chunk, device)
        with torch.no_grad():
            fwd = model.forward_with_features(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
        state = fwd.state
        # One lane, one patient, no resets: real input positions are a
        # contiguous prefix (padding only where the lane runs out).
        input_real = chunk.subject_ids[0] != NO_SUBJECT
        n_real = int(input_real.sum().item())
        assert bool(input_real[:n_real].all())  # noqa: S101
        yield StreamSpan(
            start=offset,
            n_real=n_real,
            fwd=fwd,
            has_target=chunk.real_mask[0, :n_real],
            targets=chunk.targets[0, :n_real],
        )
        offset += n_real
        if stop_after is not None and offset > stop_after:
            return


def risk_within(
    hazard_logits: torch.Tensor, edges: Sequence[float], horizons: Sequence[float]
) -> torch.Tensor:
    """``P(event within h)`` for each horizon, stacked on a new last axis.

    ``hazard_logits`` is ``(..., num_bins)`` (e.g. ``(n, E, B)`` from
    :class:`~odyssey.models.time_to_event.EventHazardHeads`); the result is
    ``(..., len(horizons))``. Horizons should be bin edges for an exact
    answer (see :func:`~odyssey.models.time_to_event.probability_within`).
    No horizons gives an empty last axis rather than an error.
    """
    if not horizons:
        return hazard_logits.new_zeros((*hazard_logits.shape[:-1], 0))
    return torch.stack(
        [probability_within(hazard_logits, edges, h) for h in horizons], dim=-1
    )


__all__ = ["StreamSpan", "risk_within", "stream_patient"]
