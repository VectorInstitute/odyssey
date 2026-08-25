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


class ClinicalSequenceBatch(NamedTuple):
    """One batch of patient event sequences."""

    concept_ids: torch.Tensor
    aux: AuxiliaryInputs
