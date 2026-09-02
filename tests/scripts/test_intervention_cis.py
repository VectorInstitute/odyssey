"""Tests for the paired intervention-mode CI helper."""

import numpy as np
import pytest

from scripts.intervention_cis import DEFAULT_PAIRS, paired_accuracy_delta


def _counts(n_subjects: int, acc: float, seed: int, per: int = 40) -> dict:
    rng = np.random.default_rng(seed)
    return {s: [int(rng.binomial(per, acc)), per] for s in range(n_subjects)}


def test_identical_modes_give_zero_delta_and_no_separation() -> None:
    a = _counts(50, 0.4, seed=1)
    res = paired_accuracy_delta(a, a, n_boot=200, seed=0)
    assert res["point"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["separated"] is False


def test_paired_delta_detects_a_consistent_per_subject_gap() -> None:
    """Mode a is mode b plus one extra hit per subject: fully paired signal."""
    rng = np.random.default_rng(0)
    b = {s: [int(rng.binomial(40, 0.4)), 40] for s in range(60)}
    a = {s: [min(h + 1, 40), n] for s, (h, n) in b.items()}
    res = paired_accuracy_delta(a, b, n_boot=500, seed=0)
    assert res["point"] == pytest.approx(1 / 40)
    assert res["separated"] is True and res["ci_low"] > 0


def test_mismatched_subject_sets_raise_rather_than_intersect() -> None:
    a = _counts(10, 0.5, seed=2)
    b = _counts(11, 0.5, seed=3)
    with pytest.raises(ValueError, match="differ between modes"):
        paired_accuracy_delta(a, b)


def test_residual_pairs_are_requested_for_the_decomposed_bottleneck() -> None:
    """The residual channel needs intervals, not just a point estimate.

    The chain can now produce zero_residual, but the paper bolds only
    what a paired bootstrap separates, so a residual number without a CI
    is unusable. Both pairings are required: against none for the size of
    the effect, and against zero_known for the comparative claim that the
    task routed around the named channel.
    """
    assert ("zero_residual", "none") in DEFAULT_PAIRS
    assert ("zero_residual", "zero_known") in DEFAULT_PAIRS


def test_pairs_are_unique_and_never_self_paired() -> None:
    """A duplicated pair would silently overwrite its own CI entry."""
    assert len(DEFAULT_PAIRS) == len(set(DEFAULT_PAIRS))
    assert all(a != b for a, b in DEFAULT_PAIRS)
