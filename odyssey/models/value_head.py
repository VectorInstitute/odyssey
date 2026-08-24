"""Quantile forecasting of the next event's magnitude: not just when and which.

The tuned GBM comparator's edge over our hazard heads localizes to raw
numeric access: stratified error analysis shows the AKI gap tracking the
staleness and magnitude of the last creatinine, and a KDIGO stage-1 rise
(+0.3 mg/dL in 48h) is invisible inside our own NORMAL bin token. The
input half of this was already fixed --
:class:`~odyssey.models.embeddings.ClinicalEventEmbeddings` has fed
per-code standardized values (``aux.values``, i.e. ``numeric_z``
from :meth:`~odyssey.data.value_binning.QuantileBinner.standardize`) into
the token embedding since v7. The output half was untouched: the model
predicts a bin TOKEN, so nothing in the training objective forces the
model's internal state to retain magnitude at all, only which quantile
bucket a value fell in.

This is not "teach a language model to spell digits". The sequence model
is completing a marked temporal point process: WHEN the next event
happens (:mod:`odyssey.models.time_to_event`'s hazard head), WHICH event
it is (the token head), and now HOW MUCH -- the mark's continuous
magnitude. Magnitude is modeled as a DISTRIBUTION, never a point
estimate: a clinician needs "creatinine 2.4, 80% interval 1.9-3.1", not
a bare number with no sense of how confident the model is.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


# Quantile levels this head predicts by default: 0.1 .. 0.9 in steps of
# 0.1 (K=9), giving symmetric 80%/60%/40%/20% central intervals plus the
# median, without predicting the (unstable, rarely load-bearing) tails.
DEFAULT_QUANTILE_LEVELS: Tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


class ValueQuantileHead(nn.Module):
    """Predicts K quantiles of the next event's standardized value.

    Conditioned on the TARGET code, not just the shared hidden state:
    head input is ``[head_features, target_token_embedding]``, reusing
    the model's own input embedding table (no second table is added --
    see how callers look up ``target_token_embedding``, e.g.
    :meth:`odyssey.models.sequence_model._SequenceModelBase._streaming_value_loss`).
    Sodium and creatinine share the same standardized scale (both go
    through :meth:`~odyssey.data.value_binning.QuantileBinner.standardize`)
    but not a predictive shape -- a fixed, code-agnostic distribution
    would average over that, so the head must see which code it is
    predicting for.

    Monotone by construction, not by penalty: the projection emits
    ``q_0.1`` directly, then ``K - 1`` non-negative increments (via
    softplus) that are cumulatively summed to produce ``q_0.2 .. q_0.9``.
    This guarantees ``q_0.1 <= q_0.2 <= ... <= q_0.9`` for every input,
    with no separate sorting or clamping step and no risk of crossing
    quantiles during training.
    """

    def __init__(
        self,
        in_features: int,
        target_embedding_dim: int,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        hidden: int = 0,
    ) -> None:
        """Initialize with ``len(quantile_levels)`` output quantiles.

        ``hidden > 0`` puts a one-hidden-layer MLP in front of the quantile
        projection, the same option the per-event hazard heads have. The
        linear default is what arm B ran (2026-08-24) and it fit the value
        distribution poorly -- mid-quantile coverage 0.243 against a nominal
        0.3, creatinine median absolute error 0.50 SD -- and a head that
        predicts magnitude badly cannot force the shared state to retain
        magnitude, which is the whole mechanism under test.
        """
        super().__init__()
        self.quantile_levels: List[float] = list(quantile_levels)
        self.num_quantiles = len(self.quantile_levels)
        in_dim = in_features + target_embedding_dim
        self.proj: nn.Module = (
            nn.Linear(in_dim, self.num_quantiles)
            if hidden <= 0
            else nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.num_quantiles),
            )
        )

    def forward(
        self, features: torch.Tensor, target_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return ``(..., num_quantiles)`` predicted quantiles, non-decreasing."""
        combined = torch.cat([features, target_embedding], dim=-1)
        raw: torch.Tensor = self.proj(combined)
        first = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        return torch.cat([first, first + increments.cumsum(dim=-1)], dim=-1)


def pinball_loss(
    quantiles: torch.Tensor, target: torch.Tensor, levels: Sequence[float]
) -> torch.Tensor:
    """Mean pinball (quantile) loss over the last dim, ``(...)`` from ``(..., K)``.

    ``target`` broadcasts against ``quantiles``' last dimension. Standard
    pinball loss at level ``tau``: ``tau * (y - q)`` when ``y >= q``,
    ``(1 - tau) * (q - y)`` otherwise -- the asymmetric loss whose
    minimizer at level ``tau`` is exactly the ``tau``-quantile of the
    target distribution.
    """
    levels_t = torch.as_tensor(levels, dtype=quantiles.dtype, device=quantiles.device)
    diff = target.unsqueeze(-1) - quantiles
    loss = torch.maximum(levels_t * diff, (levels_t - 1.0) * diff)
    return loss.mean(dim=-1)


def value_target_valid_mask(
    values: torch.Tensor, real_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-position target standardized value and validity, one streaming chunk.

    Mirrors :func:`odyssey.models.time_to_event.gap_survival_valid_mask`'s
    shift trick exactly: ``values`` is the chunk's INPUT-side standardized
    value ``(lanes, T)`` (``chunk.batch.aux.values``, already shifted to
    align with the input tokens by the streaming sampler); the value
    belonging to position ``i``'s target token is ``values[i + 1]``, known
    only when position ``i`` has a real in-chunk next token
    (``real_mask[i]`` and ``i < T - 1``) AND that next position's value is
    not NaN (the target token carries no numeric value at all -- most
    tokens, e.g. every diagnosis code). Returns ``(target_value, valid)``,
    both ``(lanes, T)``.
    """
    lanes, chunk = values.shape
    target_value = values.new_full((lanes, chunk), float("nan"))
    target_value[:, : chunk - 1] = values[:, 1:]
    valid = real_mask & ~torch.isnan(target_value)
    return target_value, valid


def value_quantile_loss(
    quantiles: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    levels: Sequence[float],
) -> torch.Tensor:
    """Mean pinball loss over ``valid`` positions; zero-graph if none.

    Positions whose target token carries no numeric value (``valid`` is
    False, from :func:`value_target_valid_mask`) contribute nothing --
    not a zero loss averaged in, excluded from the mean entirely, so a
    record with few valued targets does not get diluted gradient from
    positions the head was never asked to predict.

    ``target`` is NaN at exactly the invalid positions (that is the
    "no value here" signal ``value_target_valid_mask`` reads). Sanitized
    to 0 before scoring, not merely masked afterward: ``NaN * 0`` is
    ``NaN``, not ``0``, so multiplying an unsanitized NaN-derived loss by
    ``weight`` would poison the whole mean with NaN despite ``valid``
    correctly marking the position as excluded -- confirmed the hard way
    (a real forward/backward smoke test produced ``total loss = nan``
    with mask-derived gradient dropped to nowhere) before this fix.
    """
    if not bool(valid.any()):
        return quantiles.sum() * 0.0
    safe_target = torch.nan_to_num(target, nan=0.0)
    loss = pinball_loss(quantiles, safe_target, levels)
    weight = valid.to(loss.dtype)
    return (loss * weight).sum() / weight.sum()


def crps_from_quantiles(
    quantiles: torch.Tensor, target: torch.Tensor, levels: Sequence[float]
) -> torch.Tensor:
    """Quantile-based CRPS estimate, ``(...)`` from ``(..., K)`` quantiles.

    The continuous ranked probability score is a proper scoring rule for
    a full predictive distribution; with only ``K`` quantiles available
    (not a closed-form CDF), this uses the standard trapezoidal
    pinball-loss estimator (Gneiting & Raftery 2007, and the estimator
    used by ``properscoring``/``scoringRules``' quantile-based CRPS):
    twice the mean pinball loss across levels approximates the CRPS
    integral ``\\int (F(x) - 1{x >= y})^2 dx`` when the levels are
    equally spaced, which :data:`DEFAULT_QUANTILE_LEVELS` is (0.1 step).
    Coarser than a closed-form CRPS but proper and consistent as
    ``K -> infinity``.
    """
    return 2.0 * pinball_loss(quantiles, target, levels)


def quantile_coverage(quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-level empirical coverage: fraction of ``target <= quantiles[..., k]``.

    A well-calibrated head has ``coverage[k] ~= levels[k]`` -- the
    "predicted 80% interval actually contains the true value ~80% of the
    time" check, done per level (not just the outer interval) since a
    head can get the median right while its tails are miscalibrated.
    ``quantiles`` is ``(N, K)`` (already restricted to valid positions,
    the way :class:`~odyssey.inference.run_inference._RunningTimeMetrics`
    restricts its own inputs to ``valid`` before scoring -- coverage over
    positions with no true value is undefined, not zero), ``target`` is
    ``(N,)``; returns ``(K,)``.
    """
    covered = (target.unsqueeze(-1) <= quantiles).to(quantiles.dtype)
    return covered.mean(dim=0)


def median_absolute_error(
    quantiles: torch.Tensor, target: torch.Tensor, levels: Sequence[float]
) -> torch.Tensor:
    """Mean absolute error of the predicted median against ``target``, ``()``.

    Uses the quantile level closest to 0.5 (exactly 0.5 for the default
    levels) as the point estimate, since the head has no other single
    "the" prediction -- a distribution, not a point, is the actual
    output; this is a summary statistic of it, not the head's target.
    ``quantiles`` is ``(N, K)`` restricted to valid positions, ``target``
    is ``(N,)``.
    """
    median_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    median = quantiles[..., median_idx]
    return (median - target).abs().mean()


__all__: List[str] = [
    "DEFAULT_QUANTILE_LEVELS",
    "ValueQuantileHead",
    "crps_from_quantiles",
    "median_absolute_error",
    "pinball_loss",
    "quantile_coverage",
    "value_quantile_loss",
    "value_target_valid_mask",
]
