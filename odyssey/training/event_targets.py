"""Per-position time-to-event targets for the per-event hazard heads.

For every alert event (:data:`odyssey.data.alert_events.ALERT_EVENTS`) and
every position of a streaming chunk: how long until the event first
occurs, or, if it does not occur before follow-up ends, how long until
follow-up ends (right-censored). Positions at or after the event's onset
are not at risk and are masked out. Times are hours on the sequence time
origin, exactly like chunk time stamps, so the gap is a plain subtraction.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from odyssey.data.alert_events import EventTimes
from odyssey.data.streaming import StreamingChunk


@dataclass
class EventHazardTargets:
    """``(lanes, T, num_events)`` gap / observed / at-risk tensors for one chunk."""

    gap_hours: torch.Tensor
    observed: torch.Tensor
    at_risk: torch.Tensor


class EventTimeTables:
    """Onset and censoring times per event, indexed by (subject, visit).

    Built once per split from :func:`odyssey.data.alert_events.all_event_times`
    and queried per chunk. Subject-scoped events (death) are keyed by
    subject alone.
    """

    def __init__(
        self, times: Dict[str, EventTimes], event_names: Sequence[str]
    ) -> None:
        self.event_names = list(event_names)
        self.times = [times[name] for name in self.event_names]

    def lookup(self, subject_id: int, visit_id: int) -> Tuple[List[float], List[float]]:
        """``(onsets, censors)`` per event for one key; ``inf`` when unknown."""
        onsets: List[float] = []
        censors: List[float] = []
        for et in self.times:
            key = (subject_id, -1 if et.subject_scoped else visit_id)
            onsets.append(et.onset.get(key, float("inf")))
            censors.append(et.censor.get(key, float("-inf")))
        return onsets, censors


def event_hazard_targets(
    chunk: StreamingChunk, tables: EventTimeTables
) -> EventHazardTargets:
    """Compute :class:`EventHazardTargets` for one chunk.

    For position with time ``t``: if the event's onset ``o`` satisfies
    ``o > t``, the target is observed with gap ``o - t``; if there is no
    onset but follow-up ends at ``c > t``, the target is censored with gap
    ``c - t``; otherwise (onset already passed, or no follow-up
    information for the key) the position is not at risk.
    """
    sids = chunk.subject_ids
    vids = chunk.visit_ids
    lanes, chunk_len = sids.shape
    n_events = len(tables.event_names)
    keys = torch.stack([sids, vids], dim=-1).reshape(-1, 2)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    onset = torch.full(
        (unique_keys.shape[0], n_events), float("inf"), device=sids.device
    )
    censor = torch.full(
        (unique_keys.shape[0], n_events), float("-inf"), device=sids.device
    )
    for i, (s, v) in enumerate(unique_keys.tolist()):
        o, c = tables.lookup(int(s), int(v))
        onset[i] = torch.tensor(o, device=sids.device)
        censor[i] = torch.tensor(c, device=sids.device)
    onset_pos = onset[inverse].view(lanes, chunk_len, n_events)
    censor_pos = censor[inverse].view(lanes, chunk_len, n_events)
    now = chunk.batch.aux.time_stamps.unsqueeze(-1)
    observed = torch.isfinite(onset_pos) & (onset_pos > now)
    censored = ~torch.isfinite(onset_pos) & (censor_pos > now)
    at_risk = observed | censored
    gap = torch.where(observed, onset_pos - now, censor_pos - now)
    gap = torch.where(at_risk, gap, torch.zeros_like(gap))
    return EventHazardTargets(gap_hours=gap, observed=observed, at_risk=at_risk)


__all__ = ["EventHazardTargets", "EventTimeTables", "event_hazard_targets"]
