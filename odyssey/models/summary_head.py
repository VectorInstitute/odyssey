"""Summary head: report window statistics of the chart from the state.

An auxiliary, training-only readout. At landmark positions it predicts the
standardized window-summary panel of
:mod:`odyssey.training.summary_targets` (per-signal window min/max/mean,
change from the visit's first value, per-family occurrence counts) from
the same features the hazard heads read. The loss is a masked Huber loss:
robust to the residual heavy tails of clinical values, and zero (with a
live graph) where no target applies, so it can be summed unconditionally.
The head is never used at inference; it exists to shape the state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SummaryHead(nn.Module):
    """Features -> one standardized value per summary target."""

    def __init__(
        self, in_features: int, num_targets: int, hidden_size: int = 0
    ) -> None:
        """Initialize a linear readout, or a GELU MLP when ``hidden_size`` > 0."""
        super().__init__()
        self.num_targets = int(num_targets)
        self.hidden_size = int(hidden_size)
        self.proj: nn.Module = (
            nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_targets),
            )
            if self.hidden_size > 0
            else nn.Linear(in_features, self.num_targets)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(..., num_targets)`` predictions."""
        out: torch.Tensor = self.proj(features)
        return out


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber loss averaged over the ``True`` entries of ``mask``.

    ``prediction``/``target``/``mask`` share the shape ``(..., K)``.
    ``target_weights`` (``(K,)``, optional) reweights the per-target terms
    inside the average, so targets the state finds hardest (changes from
    baseline, counts) can carry more of the gradient than the levels it
    learns anyway. Returns ``0 * prediction.sum()`` when the mask is empty,
    so the result always carries a graph and can be added unconditionally.
    """
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    per_entry = F.huber_loss(
        prediction.float(), target.float(), reduction="none", delta=delta
    )
    weights = mask.to(per_entry.dtype)
    if target_weights is not None:
        weights = weights * target_weights.to(per_entry.dtype).to(weights.device)
    return (per_entry * weights).sum() / weights.sum()


__all__ = ["SummaryHead", "masked_huber_loss"]
