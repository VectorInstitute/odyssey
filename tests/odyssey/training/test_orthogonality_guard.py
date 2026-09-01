"""The orthogonality penalty must not go inert silently.

``orthogonality_loss`` returns exactly zero whenever the unknown slot's
width differs from ``embedding_dim``, because cosine similarity between
different-width vectors is undefined. That is deliberate -- the width cap
replaces the penalty -- but it means a config can carry a nonzero
``orthogonality_weight`` while optimizing nothing, and the run logs a
tidy ``orthogonality_loss: 0.0`` as if the term were live.

That silence confounded the L1/L3/L4 comparison: narrowing the residual
also switched the penalty off, so those arms differ in two ways rather
than one. These tests pin the warning that makes it audible.
"""

import logging

import pytest
import torch

from odyssey.models.concept_bottleneck import orthogonality_loss
from odyssey.training.train import TrainingConfig, _checked_unknown_dim


def _config(**kw) -> TrainingConfig:
    """Build a minimal config; only the three path fields are required."""
    return TrainingConfig(
        train_shard_dir="/tmp/train",
        tuning_shard_dir="/tmp/tuning",
        output_dir="/tmp/out",
        **kw,
    )


def test_penalty_is_genuinely_zero_at_mismatched_width() -> None:
    """The behaviour the warning is about, pinned directly."""
    concepts = torch.randn(4, 3, 32)
    narrow = torch.randn(4, 8)
    assert orthogonality_loss(concepts, narrow).item() == 0.0
    # and non-zero when the widths agree, so the test is not vacuous
    wide = torch.randn(4, 32)
    assert orthogonality_loss(concepts, wide).item() > 0.0


def test_warns_when_weight_is_set_but_width_makes_it_inert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _config(embedding_dim=32, unknown_dim=8, orthogonality_weight=0.1)
    with caplog.at_level(logging.WARNING):
        assert _checked_unknown_dim(cfg) == 8
    assert any("INERT" in r.message for r in caplog.records), caplog.text


def test_silent_when_the_penalty_is_actually_active() -> None:
    cfg = _config(embedding_dim=32, unknown_dim=32, orthogonality_weight=0.1)
    assert _checked_unknown_dim(cfg) == 32


def test_silent_when_the_weight_is_deliberately_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Zeroing the weight everywhere is the correct way to run the sweep."""
    cfg = _config(embedding_dim=32, unknown_dim=8, orthogonality_weight=0.0)
    with caplog.at_level(logging.WARNING):
        _checked_unknown_dim(cfg)
    assert not any("INERT" in r.message for r in caplog.records)


def test_silent_when_unknown_dim_is_unset() -> None:
    """unknown_dim=None defaults to embedding_dim, so the penalty is live."""
    cfg = _config(embedding_dim=32, unknown_dim=None, orthogonality_weight=0.1)
    assert _checked_unknown_dim(cfg) is None
