"""Frozen pre/post-bottleneck embeddings at alert index positions.

Shared foundation for every frozen-probe diagnostic in this project: the
recency-recovery precedent that got `recency_features`/`signal_channels`
removed (commit cd96842), `scripts/probe_bottleneck_signal.py`'s per-task
AUROC probes, `scripts/probe_counting_signal.py`'s count-recovery probes,
and :mod:`odyssey.inference.probe_baseline`'s EHRSHOT-style benchmark all
need the same thing: the backbone hidden state (pre-bottleneck) and the
bottleneck output (post-bottleneck, what every trained head actually reads)
at the same landmark/visit-end positions the alerts harness scores.

This module captures those two embeddings in one streaming pass instead of
scoring heads, reusing :mod:`odyssey.inference.alerts`' own landmark
selection so labels and positions line up exactly with the alerts harness
(see ``test_collect_model_scores_and_index_rows_from_events_agree_on_landmark_times``
in ``tests/odyssey/inference/test_alerts.py`` for the invariant this
depends on: the model-driven and model-free landmark selectors produce the
identical (subject, visit, time) row set for the same ``landmark_hours``).
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
import torch

from odyssey.data.alert_events import AlertEvent, EventTimes
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.alerts import IndexRow, LandmarkState, _select_index_positions
from odyssey.inference.alerts import outcome_at_horizon as _outcome_at_horizon
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.data import iter_patient_sequences
from odyssey.training.train import _move_chunk_to_device


Key = tuple[int, int, float]


def collect_embeddings(
    model: ConceptBottleneckSequenceModel,
    binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    landmark_alerts: Sequence[AlertEvent],
    visit_end_alerts: Sequence[AlertEvent],
    visit_start: dict[tuple[int, int], float],
    landmark_hours: float,
    num_lanes: int,
    chunk_size: int,
    device: str,
) -> tuple[list[Key], np.ndarray, np.ndarray, list[Key], np.ndarray, np.ndarray]:
    """One streaming pass: (keys, pre, post) for landmark and visit-end rows.

    ``landmark_alerts``/``visit_end_alerts`` only gate WHICH index-position
    scan runs (landmark selection itself is event-independent, see
    :func:`~odyssey.inference.alerts._select_index_positions`) -- pass any
    non-empty sequence to collect landmark rows regardless of which alert
    tasks are actually being probed; the same embeddings serve every task
    that shares the same landmark cadence.
    """
    model.eval()
    patients = iter_patient_sequences(binned, vocab)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    lm_keys: list[Key] = []
    lm_pre_blocks: list[np.ndarray] = []
    lm_post_blocks: list[np.ndarray] = []
    ve_keys: list[Key] = []
    ve_pre_blocks: list[np.ndarray] = []
    ve_post_blocks: list[np.ndarray] = []

    state = None
    landmark_state: LandmarkState | None = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            hidden_states, state = model.backbone(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            bottleneck_out = model.bottleneck(hidden_states)
            sids = chunk.subject_ids
            vids = chunk.visit_ids
            times = chunk.batch.aux.time_stamps

            keys = torch.stack([sids, vids], dim=-1).reshape(-1, 2)
            unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
            unique_starts = torch.tensor(
                [
                    visit_start.get((int(s), int(v)), 0.0)
                    for s, v in unique_keys.tolist()
                ],
                dtype=times.dtype,
                device=times.device,
            )
            starts = unique_starts[inverse].view_as(times)

            if landmark_alerts:
                keep, landmark_state = _select_index_positions(
                    "landmark",
                    chunk,
                    times=times,
                    sids=sids,
                    vids=vids,
                    landmark_hours=landmark_hours,
                    starts=starts,
                    landmark_state=landmark_state,
                )
                if keep.any():
                    idx = keep.nonzero(as_tuple=False)
                    b, t = idx[:, 0], idx[:, 1]
                    lm_pre_blocks.append(
                        hidden_states[b, t].detach().float().cpu().numpy()
                    )
                    lm_post_blocks.append(
                        bottleneck_out.bottleneck[b, t].detach().float().cpu().numpy()
                    )
                    sel_sids = sids[b, t].tolist()
                    sel_vids = vids[b, t].tolist()
                    sel_times = times[b, t].tolist()
                    lm_keys.extend(zip(sel_sids, sel_vids, sel_times))

            if visit_end_alerts:
                keep_ve, _ = _select_index_positions(
                    "visit_end",
                    chunk,
                    times=times,
                    sids=sids,
                    vids=vids,
                    landmark_hours=landmark_hours,
                    starts=starts,
                    landmark_state=None,
                )
                if keep_ve.any():
                    idx = keep_ve.nonzero(as_tuple=False)
                    b, t = idx[:, 0], idx[:, 1]
                    ve_pre_blocks.append(
                        hidden_states[b, t].detach().float().cpu().numpy()
                    )
                    ve_post_blocks.append(
                        bottleneck_out.bottleneck[b, t].detach().float().cpu().numpy()
                    )
                    sel_sids = sids[b, t].tolist()
                    sel_vids = vids[b, t].tolist()
                    sel_times = times[b, t].tolist()
                    ve_keys.extend(zip(sel_sids, sel_vids, sel_times))
            del hidden_states, bottleneck_out

    def _cat(blocks: list[np.ndarray], dim: int) -> np.ndarray:
        return (
            np.concatenate(blocks, axis=0)
            if blocks
            else np.zeros((0, dim), dtype=np.float32)
        )

    hidden_dim = model.backbone.hidden_size
    bottleneck_dim = model.bottleneck.output_dim
    return (
        lm_keys,
        _cat(lm_pre_blocks, hidden_dim),
        _cat(lm_post_blocks, bottleneck_dim),
        ve_keys,
        _cat(ve_pre_blocks, hidden_dim),
        _cat(ve_post_blocks, bottleneck_dim),
    )


def labels_for(keys: Sequence[Key], times: EventTimes, horizon: float) -> np.ndarray:
    """Binary outcome-at-horizon label for each (subject, visit, time) key.

    ``np.nan`` where :func:`~odyssey.inference.alerts.outcome_at_horizon`
    returns ``None`` (censored, or not at risk) -- callers mask with
    ``~np.isnan(...)`` before fitting/scoring, the same convention every
    caller of ``outcome_at_horizon`` already follows.
    """
    labels = np.full(len(keys), np.nan, dtype=np.float64)
    for i, (sid, vid, t) in enumerate(keys):
        row = IndexRow(sid, vid, t)
        y = _outcome_at_horizon(row, times, horizon)
        if y is not None:
            labels[i] = y
    return labels


__all__ = ["collect_embeddings", "labels_for", "Key"]
