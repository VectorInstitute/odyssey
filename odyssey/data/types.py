"""Shared batch types for clinical event sequences.

Bundling these fields into NamedTuples (rather than passing them as
parallel positional arguments through every layer) means adding a new
auxiliary input is a one-line change here instead of a signature change
across embeddings, every backbone, and the sequence model.
"""

from typing import NamedTuple

import torch


class AuxiliaryInputs(NamedTuple):
    """Per-token clinical structure alongside a token id."""

    type_ids: torch.Tensor
    time_stamps: torch.Tensor
    ages: torch.Tensor
    visit_orders: torch.Tensor
    visit_segments: torch.Tensor


class ClinicalSequenceBatch(NamedTuple):
    """One batch of patient event sequences."""

    concept_ids: torch.Tensor
    aux: AuxiliaryInputs
