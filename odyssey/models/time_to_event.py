"""Time-to-next-bundle forecasting as a discrete-time hazard.

The sequence models forecast *what* comes next; this head forecasts
*when*. Events in a record arrive in bundles at one timestamp (a lab
panel, a medication order set), separated by gaps from minutes to weeks,
so the natural target at every position is the gap to the next event:
zero while the current bundle continues, positive when it ends. That gap
is modeled as a discrete-time hazard over log-spaced bins (Tutz &
Schmid, *Modeling Discrete Time-to-Event Data*): the head emits one logit
per bin, ``sigmoid(h_b)`` is the probability the next event lands in bin
``b`` given it has not landed earlier, the survival function is the
running product of ``1 - sigmoid``, and the likelihood of an observed gap
is the standard discrete-time survival likelihood. Everything an alert
needs -- ``P(next event within 6 hours)``, a survival curve over the
coming day -- reads straight off the hazards, and the same formulation
extends to per-event-type hazards (time to vasopressor start, to ICU
transfer) with censoring handled by construction.

Bin 0 is the "same instant" bin (gap of zero): whether the current
bundle continues is itself a forecast, and one the set-valued next-event
objective in :mod:`odyssey.models.sequence_model` needs.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


# Right edges of the positive-gap bins, in hours. Bin 0 is the exact-zero
# gap; bin b in 1..len(edges) covers (edges[b-2], edges[b-1]] (with an
# implicit 0 before the first edge); bin len(edges)+1 is open (anything
# longer than the last edge). len(edges) + 2 bins in total.
DEFAULT_TIME_BIN_EDGES_HOURS: Tuple[float, ...] = (
    1.0 / 60,  # 1 minute
    5.0 / 60,
    15.0 / 60,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    12.0,
    24.0,
    48.0,
    72.0,
    24.0 * 7,
    24.0 * 30,
)


def gap_to_bin(gap_hours: torch.Tensor, edges: Sequence[float]) -> torch.Tensor:
    """Map non-negative gaps (hours) to bin indices ``0 .. len(edges) + 1``.

    Exactly-zero gaps go to bin 0; a positive gap ``g`` goes to the first
    bin whose right edge is ``>= g``, or to the open final bin
    ``len(edges) + 1``.
    """
    edge_tensor = torch.as_tensor(edges, dtype=gap_hours.dtype, device=gap_hours.device)
    positive = gap_hours > 0
    # searchsorted with right=False gives the first edge >= g.
    idx = torch.searchsorted(edge_tensor, gap_hours.reshape(-1), right=False).reshape(
        gap_hours.shape
    )
    return torch.where(positive, idx + 1, torch.zeros_like(idx))


class TimeToEventHead(nn.Module):
    """Linear hazard head: input features -> one hazard logit per time bin."""

    def __init__(
        self, in_features: int, edges: Sequence[float] = DEFAULT_TIME_BIN_EDGES_HOURS
    ) -> None:
        """Initialize with ``len(edges) + 2`` bins (zero, per edge, open tail)."""
        super().__init__()
        self.edges: List[float] = list(edges)
        self.num_bins = len(self.edges) + 2
        self.proj = nn.Linear(in_features, self.num_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(..., num_bins)`` hazard logits."""
        out: torch.Tensor = self.proj(features)
        return out

    def bins(self, gap_hours: torch.Tensor) -> torch.Tensor:
        """Map gaps to this head's bin indices (see :func:`gap_to_bin`)."""
        return gap_to_bin(gap_hours, self.edges)


def hazard_log_likelihood(
    hazard_logits: torch.Tensor, target_bin: torch.Tensor
) -> torch.Tensor:
    """Per-position log-likelihood of an observed (uncensored) gap.

    ``log P(bin = B) = log h_B + sum_{b < B} log(1 - h_b)`` with
    ``h_b = sigmoid(logit_b)``. Uses ``logsigmoid`` throughout so it is
    stable for extreme logits. Shapes: ``hazard_logits (..., num_bins)``,
    ``target_bin (...)`` long, result ``(...)``.
    """
    log_h = F.logsigmoid(hazard_logits)
    log_1mh = F.logsigmoid(-hazard_logits)
    num_bins = hazard_logits.shape[-1]
    bins = torch.arange(num_bins, device=hazard_logits.device)
    before = (bins < target_bin.unsqueeze(-1)).to(log_1mh.dtype)
    survive_before = (log_1mh * before).sum(dim=-1)
    hit = log_h.gather(-1, target_bin.unsqueeze(-1)).squeeze(-1)
    return survive_before + hit


def censored_hazard_log_likelihood(
    hazard_logits: torch.Tensor, target_bin: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    """Log-likelihood with right censoring, per position.

    Observed rows (``observed`` True): the event landed in ``target_bin``,
    scored by :func:`hazard_log_likelihood`. Censored rows: follow-up ran
    out inside ``target_bin`` without the event, so all that is known is
    survival through every earlier bin: ``sum_{b < B} log(1 - h_b)``
    (the standard discrete-time survival contribution; the partially
    observed bin itself is not credited).
    """
    log_1mh = F.logsigmoid(-hazard_logits)
    num_bins = hazard_logits.shape[-1]
    bins = torch.arange(num_bins, device=hazard_logits.device)
    before = (bins < target_bin.unsqueeze(-1)).to(log_1mh.dtype)
    survive_before = (log_1mh * before).sum(dim=-1)
    full = hazard_log_likelihood(hazard_logits, target_bin)
    return torch.where(observed, full, survive_before)


class EventHazardHeads(nn.Module):
    """One discrete-time hazard per named event: features -> (events, bins).

    The alert-grade counterpart of :class:`TimeToEventHead`: instead of
    "when is the next event of any kind", each head answers "when does
    *this* event (vasopressor start, ICU admission, ...) first occur",
    trained with right censoring at the end of follow-up, so calibrated
    ``P(event within h)`` and full survival curves read off it directly.
    """

    def __init__(
        self,
        in_features: int,
        event_names: Sequence[str],
        edges: Sequence[float] = DEFAULT_TIME_BIN_EDGES_HOURS,
    ) -> None:
        """Initialize one hazard vector per event over ``len(edges) + 2`` bins."""
        super().__init__()
        self.event_names: List[str] = list(event_names)
        self.edges: List[float] = list(edges)
        self.num_bins = len(self.edges) + 2
        self.proj = nn.Linear(in_features, len(self.event_names) * self.num_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(..., num_events, num_bins)`` hazard logits."""
        out: torch.Tensor = self.proj(features)
        return out.view(*features.shape[:-1], len(self.event_names), self.num_bins)


def event_hazard_nll(
    hazard_logits: torch.Tensor,
    gap_hours: torch.Tensor,
    observed: torch.Tensor,
    at_risk: torch.Tensor,
    edges: Sequence[float],
) -> torch.Tensor:
    """Mean censored NLL over at-risk (position, event) entries; zero-graph if none.

    ``hazard_logits`` ``(..., E, B)``; ``gap_hours`` ``(..., E)`` is the
    time from the position to the event's onset (observed) or to the end
    of follow-up (censored); ``observed`` and ``at_risk`` are ``(..., E)``
    bools. Entries not at risk (the event already happened, or no
    follow-up information) contribute nothing.
    """
    if not bool(at_risk.any()):
        return hazard_logits.sum() * 0.0
    target_bin = gap_to_bin(gap_hours.clamp_min(0.0), edges)
    ll = censored_hazard_log_likelihood(hazard_logits, target_bin, observed)
    weight = at_risk.to(ll.dtype)
    return -(ll * weight).sum() / weight.sum()


def hazard_nll(
    hazard_logits: torch.Tensor,
    gap_hours: torch.Tensor,
    valid: torch.Tensor,
    edges: Sequence[float],
) -> torch.Tensor:
    """Mean negative log-likelihood over ``valid`` positions; zero-graph if none.

    ``gap_hours`` is the observed gap to the next event at each position;
    positions whose next event is unknown (last position of a lane, a
    reset boundary, padding) must be masked out via ``valid`` -- they are
    not censored observations here, simply unobserved.
    """
    if not bool(valid.any()):
        return hazard_logits.sum() * 0.0
    target_bin = gap_to_bin(gap_hours.clamp_min(0.0), edges)
    ll = hazard_log_likelihood(hazard_logits, target_bin)
    return -(ll * valid.to(ll.dtype)).sum() / valid.sum().to(ll.dtype)


def survival_curve(hazard_logits: torch.Tensor) -> torch.Tensor:
    """``S(b) = P(next event lands after bin b)`` for each bin, ``(..., num_bins)``.

    ``S(b) = prod_{b' <= b} (1 - h_{b'})``; ``1 - S(b)`` is the CDF at
    the bin's right edge, i.e. ``P(next event within edges[b])`` for
    ``b >= 1`` and ``P(same instant)`` for ``b = 0``.
    """
    log_1mh = F.logsigmoid(-hazard_logits)
    return torch.exp(log_1mh.cumsum(dim=-1))


def probability_within(
    hazard_logits: torch.Tensor, edges: Sequence[float], horizon_hours: float
) -> torch.Tensor:
    """``P(next event within horizon_hours)``, ``(...)``.

    Uses the largest bin whose right edge is ``<= horizon_hours`` (the
    horizon should be one of the bin edges for an exact answer).
    """
    edge_tensor = torch.as_tensor(edges, dtype=hazard_logits.dtype)
    covered = int((edge_tensor <= horizon_hours).sum().item())  # bins 1..covered
    cdf = 1.0 - survival_curve(hazard_logits)
    within: torch.Tensor = cdf[..., covered]
    return within


def gap_survival_valid_mask(
    time_stamps: torch.Tensor, real_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-position gap targets and validity for one streaming chunk.

    ``time_stamps`` are the chunk's input times ``(lanes, T)``; the gap
    at position ``i`` is ``time[i+1] - time[i]``, known only when the
    position has a real in-chunk next token (``real_mask[i]`` and
    ``i < T-1``). Returns ``(gap_hours, valid)``, both ``(lanes, T)``.
    """
    lanes, chunk = time_stamps.shape
    gap = torch.zeros_like(time_stamps)
    gap[:, : chunk - 1] = time_stamps[:, 1:] - time_stamps[:, :-1]
    valid = real_mask.clone()
    valid[:, chunk - 1] = False
    return gap, valid


__all__: List[str] = [
    "DEFAULT_TIME_BIN_EDGES_HOURS",
    "EventHazardHeads",
    "TimeToEventHead",
    "censored_hazard_log_likelihood",
    "event_hazard_nll",
    "gap_to_bin",
    "hazard_log_likelihood",
    "hazard_nll",
    "survival_curve",
    "probability_within",
    "gap_survival_valid_mask",
]
