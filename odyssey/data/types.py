"""Shared batch types for clinical event sequences.

Bundling these fields into NamedTuples (rather than passing them as
parallel positional arguments through every layer) means adding a new
auxiliary input is a one-line change here instead of a signature change
across embeddings, every backbone, and the sequence model.
"""

from typing import NamedTuple, Optional

import torch


class AuxiliaryInputs(NamedTuple):
    """Per-token clinical structure alongside a token id."""

    type_ids: torch.Tensor
    time_stamps: torch.Tensor
    ages: torch.Tensor
    visit_orders: torch.Tensor
    visit_segments: torch.Tensor
    values: Optional[torch.Tensor] = None
    """Standardized numeric value per token (see
    :meth:`~odyssey.data.value_binning.QuantileBinner.standardize`), NaN
    where the event carries no value; ``None`` on batches built without a
    value channel. Read only by embeddings configured with
    ``use_values=True``."""
    family_recency: Optional[torch.Tensor] = None
    """(..., 8) hours since the previous event of each code family (NaN if
    never), per token; ``None`` on batches built without the channel. Read
    only by models with ``recency_features`` on, at the heads."""
    signal_state: Optional[torch.Tensor] = None
    """(..., 2 * N_PANEL_SIGNALS) per-token panel-signal state: hours since
    each curated signal's previous observation, then that observation's
    standardized value (NaN where unseen); ``None`` on batches built
    without the channel. Read only by models with ``signal_channels`` on,
    at the time/event heads."""


class ClinicalSequenceBatch(NamedTuple):
    """One batch of patient event sequences."""

    concept_ids: torch.Tensor
    aux: AuxiliaryInputs
