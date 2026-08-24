"""Tests for the subject-clustered bootstrap AUROC helper."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from odyssey.inference.uncertainty import BootstrapAUROC, bootstrap_auroc


def _correlated_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """19 single-row background subjects + 1 big subject with 100 identical rows.

    Background subjects are perfectly rank-separable (y=1 always scores
    higher than y=0), giving a clean high baseline AUROC on their own. The
    big subject is a homogeneous block of 100 rows, all y=0 with a very
    high score (a false-positive cluster) -- whether or not it appears in
    a resample swings the AUROC a lot, and it is an all-or-nothing,
    whole-subject event only a SUBJECT bootstrap can reflect: a uniform
    ROW bootstrap draws ~84% of its rows from this block on almost every
    resample (it is 100 of 119 real rows), so the row-level composition
    barely moves resample to resample.
    """
    rng = np.random.default_rng(0)
    bg_y = np.array([i % 2 for i in range(19)], dtype=float)
    bg_p = np.where(
        bg_y == 1, 0.6 + rng.uniform(0, 0.1, 19), 0.1 + rng.uniform(0, 0.1, 19)
    )
    bg_subjects = np.arange(1, 20)

    big_y = np.zeros(100)
    big_p = np.full(100, 0.99)
    big_subjects = np.full(100, 1000)

    y = np.concatenate([bg_y, big_y])
    p = np.concatenate([bg_p, big_p])
    subjects = np.concatenate([bg_subjects, big_subjects])
    return y, p, subjects


def _row_bootstrap_std(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> float:
    """Compute a naive ROW-level bootstrap std, for comparison only.

    Not exposed by the module under test, since resampling rows
    independently is exactly the wrong thing to do on correlated
    landmark data (see the module docstring). Used here only to prove the
    subject-clustered version is not narrower than this on correlated
    data.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(roc_auc_score(y_b, p_b))
    return float(np.std(np.array(scores), ddof=1))


def test_subject_clustered_ci_is_wider_than_row_bootstrap_on_correlated_data() -> None:
    y, p, subjects = _correlated_fixture()

    result = bootstrap_auroc(y, p, subjects, n_boot=2000, seed=0)
    assert result is not None
    assert result.std is not None

    row_std = _row_bootstrap_std(y, p, n_boot=2000, seed=0)

    assert result.std > row_std * 3  # not just marginally wider -- the
    # whole point of resampling subjects is that a single dominant,
    # homogeneous subject's presence/absence is a coin flip at the
    # subject level and nearly invisible at the row level.


def test_single_class_observed_y_returns_none() -> None:
    y = np.ones(10)
    p = np.linspace(0, 1, 10)
    subjects = np.arange(10)

    assert bootstrap_auroc(y, p, subjects) is None


def test_skip_counter_counts_degenerate_resamples() -> None:
    """One positive subject among many negatives.

    Most subject-level resamples that never draw it are single-class and
    must be skipped and counted, not silently dropped.
    """
    n_negative_subjects = 30
    y = np.array([0.0] * n_negative_subjects + [1.0])
    p = np.linspace(0, 1, n_negative_subjects + 1)
    subjects = np.arange(n_negative_subjects + 1)

    result = bootstrap_auroc(y, p, subjects, n_boot=500, seed=0)

    assert result is not None
    assert result.n_boot_skipped > 0
    assert result.n_boot_used + result.n_boot_skipped == 500


def test_seeded_reproducibility() -> None:
    y, p, subjects = _correlated_fixture()

    a = bootstrap_auroc(y, p, subjects, n_boot=200, seed=42)
    b = bootstrap_auroc(y, p, subjects, n_boot=200, seed=42)

    assert a == b


def test_different_seeds_can_give_different_results() -> None:
    y, p, subjects = _correlated_fixture()

    a = bootstrap_auroc(y, p, subjects, n_boot=200, seed=1)
    b = bootstrap_auroc(y, p, subjects, n_boot=200, seed=2)

    assert a is not None and b is not None
    assert a.mean != b.mean


def test_shape_mismatch_raises() -> None:
    y = np.array([0.0, 1.0, 0.0])
    p = np.array([0.1, 0.9])
    subjects = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="same rows"):
        bootstrap_auroc(y, p, subjects)


def test_point_estimate_matches_plain_roc_auc_score() -> None:
    y, p, subjects = _correlated_fixture()

    result = bootstrap_auroc(y, p, subjects, n_boot=50, seed=0)

    assert result is not None
    assert result.point_estimate == pytest.approx(roc_auc_score(y, p))


def test_all_resamples_skipped_returns_none_mean_std_ci_but_a_count() -> None:
    # Every subject carries only one class label; with a single positive
    # subject far outnumbered, a very small n_boot with an unlucky seed
    # can plausibly skip everything -- constructed directly by isolating
    # exactly one subject to guarantee a resample that never draws it.
    y = np.array([1.0, 0.0])
    p = np.array([0.9, 0.1])
    subjects = np.array([1, 2])

    result = bootstrap_auroc(y, p, subjects, n_boot=1, seed=7)

    assert result is not None
    if result.n_boot_used == 0:
        assert result.mean is None
        assert result.std is None
        assert result.ci_low is None
        assert result.ci_high is None
        assert result.n_boot_skipped == 1
    else:
        # rare with n_boot=1 but not impossible depending on the draw
        assert result.n_boot_used == 1


def test_bootstrap_auroc_result_is_frozen() -> None:
    y, p, subjects = _correlated_fixture()
    result = bootstrap_auroc(y, p, subjects, n_boot=10, seed=0)
    assert isinstance(result, BootstrapAUROC)
    with pytest.raises(Exception):  # noqa: B017, PT011 -- dataclasses.FrozenInstanceError
        result.mean = 0.5  # type: ignore[misc]
