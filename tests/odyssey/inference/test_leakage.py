"""Concept-channel leakage metrics (CTL, ICL): probes recover planted leaks.

Synthetic-bank tests, same convention as
tests/odyssey/inference/test_time_head_probe.py: build a :class:`LeakageBank`
directly (no model, no streaming) with a signal planted in a specific slot,
and check the probes find it -- and report ~0 when a slot is pure noise.
"""

from typing import Optional, Tuple

import torch

from odyssey.inference.leakage import (
    LeakageBank,
    _fit_categorical_probe,
    _score_categorical_probe,
    compute_ctl,
    compute_icl,
)


NUM_CONCEPTS = 4
EMBEDDING_DIM = 6
UNKNOWN_DIM = 6
NUM_CLASSES = 5  # small closed family taxonomy, classes 0..4
NAMES = tuple(f"concept_{i}" for i in range(NUM_CONCEPTS))
# Adam's step size is ~lr regardless of gradient magnitude, so what matters
# for convergence is total optimizer steps (epochs * ceil(n / batch_size)),
# not epochs alone -- these small synthetic banks need a small batch_size
# (and enough epochs/patience) to get the same step count real, much
# larger banks get for free at the default batch_size=4096.
FIT_KW = {"epochs": 150, "batch_size": 64, "patience": 20}


def _synthetic_bank(
    n: int,
    *,
    seed: int,
    leak_slot: Optional[int] = None,
    leak_strength: float = 8.0,
) -> LeakageBank:
    """Build a bank with independent random probs/embeddings/labels.

    If ``leak_slot`` is given, that concept's embedding dim 0 is overwritten
    with a strongly scaled, signed copy of the next-token family label (one
    dimension is enough for a linear probe to recover a class from -- see
    ``test_fit_categorical_probe_recovers_planted_signal``), and its dim 1
    with a strongly scaled, signed copy of concept ``(leak_slot + 1) %
    NUM_CONCEPTS``'s own running label -- the ICL plant. Neither plant
    touches ``concept_probs``, so any probe using only the probability (CTL's
    ``probs_only``, ICL's probability baseline) sees none of it.
    """
    g = torch.Generator().manual_seed(seed)
    concept_probs = torch.rand(n, NUM_CONCEPTS, generator=g)
    concept_embeddings = 0.1 * torch.randn(n, NUM_CONCEPTS, EMBEDDING_DIM, generator=g)
    unknown_embedding = 0.1 * torch.randn(n, UNKNOWN_DIM, generator=g)
    family_labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    concept_labels = (torch.rand(n, NUM_CONCEPTS, generator=g) < 0.5).float()
    concept_observed = torch.ones(n, NUM_CONCEPTS, dtype=torch.bool)

    if leak_slot is not None:
        signed_family = (family_labels.float() - (NUM_CLASSES - 1) / 2) * leak_strength
        concept_embeddings[:, leak_slot, 0] = signed_family
        target_j = (leak_slot + 1) % NUM_CONCEPTS
        signed_label = (concept_labels[:, target_j] * 2 - 1) * leak_strength
        concept_embeddings[:, leak_slot, 1] = signed_label

    return LeakageBank(
        concept_probs.half(),
        concept_embeddings.half(),
        unknown_embedding.half(),
        family_labels,
        concept_labels,
        concept_observed,
        NAMES,
        n_positions_seen=n,
        sample_rate=1.0,
    )


def _three_splits(
    *, leak_slot: Optional[int] = None
) -> Tuple[LeakageBank, LeakageBank, LeakageBank]:
    return (
        _synthetic_bank(3000, seed=0, leak_slot=leak_slot),
        _synthetic_bank(1000, seed=1, leak_slot=leak_slot),
        _synthetic_bank(2000, seed=2, leak_slot=leak_slot),
    )


# ---------------------------------------------------------------------------
# LeakageBank mechanics
# ---------------------------------------------------------------------------


def test_leakage_bank_len_to_and_concat() -> None:
    a = _synthetic_bank(50, seed=0)
    b = _synthetic_bank(30, seed=1)
    assert len(a) == 50
    moved = a.to("cpu")
    assert moved.concept_probs.device.type == "cpu"

    combined = LeakageBank.concat([a, b], max_positions=None)
    assert len(combined) == 80
    assert combined.n_positions_seen == 80
    assert combined.concept_names == NAMES

    capped = LeakageBank.concat([a, b], max_positions=10)
    assert len(capped) == 10


# ---------------------------------------------------------------------------
# Probe-level: recovers a planted signal, ~chance on pure noise
# ---------------------------------------------------------------------------


def test_fit_categorical_probe_recovers_planted_signal() -> None:
    g = torch.Generator().manual_seed(0)
    n = 4000
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    signed = (y.float() - (NUM_CLASSES - 1) / 2) * 8.0
    x = signed.unsqueeze(-1) + 0.01 * torch.randn(n, 1, generator=g)

    head, _ = _fit_categorical_probe(
        1,
        NUM_CLASSES,
        train_x=x[:3000],
        train_y=y[:3000],
        tune_x=x[3000:],
        tune_y=y[3000:],
        epochs=150,
        batch_size=64,
        patience=20,
        seed=0,
    )
    acc, _ = _score_categorical_probe(head, x[3000:], y[3000:])
    assert acc > 0.95


def test_fit_categorical_probe_near_chance_on_pure_noise() -> None:
    g = torch.Generator().manual_seed(0)
    n = 4000
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    x = torch.randn(n, 8, generator=g)  # independent of y

    head, _ = _fit_categorical_probe(
        8,
        NUM_CLASSES,
        train_x=x[:3000],
        train_y=y[:3000],
        tune_x=x[3000:],
        tune_y=y[3000:],
        epochs=150,
        batch_size=64,
        patience=20,
        seed=0,
    )
    acc, _ = _score_categorical_probe(head, x[3000:], y[3000:])
    assert acc < 1.0 / NUM_CLASSES + 0.1


# ---------------------------------------------------------------------------
# CTL
# ---------------------------------------------------------------------------


def test_compute_ctl_detects_embedding_leak_beyond_probs() -> None:
    train, tune, held = _three_splits(leak_slot=0)
    result = compute_ctl(train, tune, held, seed=0, **FIT_KW)

    assert result.embeddings_only.accuracy > 0.9
    assert result.probs_only.accuracy < 1.0 / NUM_CLASSES + 0.15
    assert result.ctl_accuracy > 0.5
    assert result.ctl_cross_entropy < -0.5  # embeddings' CE is much lower
    assert result.n_held_out == len(held)
    assert result.probs_only.n == len(held)


def test_compute_ctl_near_zero_when_neither_channel_carries_the_task() -> None:
    train, tune, held = _three_splits(leak_slot=None)
    result = compute_ctl(train, tune, held, seed=0, **FIT_KW)

    assert abs(result.ctl_accuracy) < 0.15
    assert result.embeddings_only.accuracy < 1.0 / NUM_CLASSES + 0.15
    assert result.probs_only.accuracy < 1.0 / NUM_CLASSES + 0.15


# ---------------------------------------------------------------------------
# ICL
# ---------------------------------------------------------------------------


def test_compute_icl_excludes_the_diagonal() -> None:
    train, tune, held = _three_splits(leak_slot=None)
    result = compute_icl(train, tune, held, seed=0, **FIT_KW)
    assert len(result.pairs) == NUM_CONCEPTS * (NUM_CONCEPTS - 1)
    assert all(p.concept_i != p.concept_j for p in result.pairs)


def test_compute_icl_detects_the_planted_pair_and_ignores_others() -> None:
    leak_slot = 0
    target_j = (leak_slot + 1) % NUM_CONCEPTS
    train, tune, held = _three_splits(leak_slot=leak_slot)
    result = compute_icl(train, tune, held, seed=0, **FIT_KW)

    by_pair = {(p.concept_i, p.concept_j): p for p in result.pairs}
    planted = by_pair[(NAMES[leak_slot], NAMES[target_j])]
    assert planted.auroc_embedding is not None and planted.auroc_embedding > 0.9
    assert planted.auroc_probability is not None and planted.auroc_probability < 0.65
    assert planted.icl_raw is not None and planted.icl_raw > 0.3
    assert planted.icl == planted.icl_raw  # already positive, clip is a no-op

    # a pair with no planted signal: small (near-zero) leakage in both directions
    other = by_pair[(NAMES[(leak_slot + 2) % NUM_CONCEPTS], NAMES[leak_slot])]
    assert other.icl is not None and other.icl < 0.2


def test_compute_icl_raw_can_be_negative_but_icl_clips_at_zero() -> None:
    train, tune, held = _three_splits(leak_slot=None)
    result = compute_icl(train, tune, held, seed=0, **FIT_KW)
    for pair in result.pairs:
        if pair.icl_raw is not None:
            assert pair.icl is not None
            assert pair.icl >= 0.0
            assert pair.icl == max(0.0, pair.icl_raw)


def test_compute_icl_pair_is_none_when_target_never_observed() -> None:
    train, tune, held = _three_splits(leak_slot=None)
    held.concept_observed[:, 1] = False
    result = compute_icl(train, tune, held, epochs=5, seed=0)
    for pair in result.pairs:
        if pair.concept_j == NAMES[1]:
            assert pair.auroc_embedding is None
            assert pair.auroc_probability is None
            assert pair.icl is None
            assert pair.n == 0


def test_compute_icl_top_pairs_are_sorted_and_capped() -> None:
    train, tune, held = _three_splits(leak_slot=0)
    result = compute_icl(train, tune, held, seed=0, top_k=3, **FIT_KW)
    assert len(result.top_pairs) <= 3
    raws = [p.icl_raw for p in result.top_pairs if p.icl_raw is not None]
    assert raws == sorted(raws, reverse=True)
