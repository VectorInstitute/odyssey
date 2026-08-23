"""The distributional time-head probe (N2): heads normalize, fit, and score."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.time_head_probe import (
    DEFAULT_HEADS,
    CategoricalProbe,
    FeatureBank,
    HazardProbe,
    LogNormalMixtureProbe,
    bin_nll,
    collect_feature_bank,
    fit_head,
    make_head,
    run_probe,
    score_head,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import BaselineSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS, gap_to_bin


EDGES = DEFAULT_TIME_BIN_EDGES_HOURS
T0 = datetime(2024, 1, 1)


def _synthetic_bank(n: int, seed: int = 0) -> FeatureBank:
    """Features carry the gap regime: x[0] > 0 -> long gaps, x[0] < 0 -> short."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    regime = x[:, 0] > 0
    zero = torch.rand(n, generator=g) < 0.3
    log_gap = torch.where(
        regime,
        math.log(24.0) + 0.3 * torch.randn(n, generator=g),
        math.log(0.5) + 0.3 * torch.randn(n, generator=g),
    )
    gaps = torch.where(zero, torch.zeros(n), log_gap.exp())
    return FeatureBank(x, gaps, n_positions_seen=n, sample_rate=1.0)


@pytest.mark.parametrize("name", DEFAULT_HEADS)
def test_heads_produce_normalized_bin_distributions(name: str) -> None:
    head = make_head(name, 4, EDGES)
    x = torch.randn(16, 4)
    logp = head.log_bin_probs(x)
    assert logp.shape == (16, len(EDGES) + 2)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(16), atol=1e-4)


def test_fitting_beats_the_untrained_head_and_learns_the_regimes() -> None:
    train, tune, held = (
        _synthetic_bank(6000, 0),
        _synthetic_bank(1000, 1),
        _synthetic_bank(2000, 2),
    )
    for name in DEFAULT_HEADS:
        head = make_head(name, 4, EDGES)
        before = bin_nll(head, held)
        trace = fit_head(
            head, train, tune, epochs=40, batch_size=256, lr=2e-2, patience=4
        )
        metrics = score_head(head, held, name)
        assert metrics.bin_nll < before - 0.3, name
        assert trace.best_epoch >= 0 and len(trace.tuning_nll) >= 1
        # regime is linearly readable: P(within 1h) should be high for the
        # short-gap half and low for the long-gap half
        logp = head.log_bin_probs(held.features)
        cdf = logp.exp().cumsum(-1)
        covered = int((torch.tensor(EDGES) <= 1.0).sum())
        within_1h = cdf[:, covered]
        short = held.features[:, 0] < 0
        assert within_1h[short].mean() > within_1h[~short].mean() + 0.2, name
        assert 0.25 < metrics.same_instant_accuracy <= 1.0
        assert set(metrics.calibration) == {"1h", "8h", "24h"}


def test_lognormal_mixture_continuous_and_median_are_sensible() -> None:
    train, tune, held = (
        _synthetic_bank(6000, 0),
        _synthetic_bank(1000, 1),
        _synthetic_bank(2000, 2),
    )
    head = LogNormalMixtureProbe(4, EDGES, k=2)
    fit_head(head, train, tune, epochs=40, batch_size=256, lr=2e-2, patience=4)
    metrics = score_head(head, held, "lognormal2")
    assert (
        metrics.continuous_nll is not None
        and metrics.median_gap_abs_error_hours is not None
    )
    long = held.features[:, 0] > 0
    med = head.median_positive_gap(held.features)
    # long-gap regime median ~24h, short ~0.5h (within a factor of 2)
    assert 12 < med[long].median().item() < 48
    assert 0.25 < med[~long].median().item() < 1.0


def test_hazard_probe_matches_the_production_head_likelihood() -> None:
    """The refit hazard head's bin log-probs equal the production likelihood."""
    from odyssey.models.time_to_event import hazard_log_likelihood  # noqa: PLC0415

    head = HazardProbe(4, EDGES)
    x = torch.randn(8, 4)
    gaps = torch.tensor([0.0, 0.1, 1.5, 30.0, 0.0, 5.0, 800.0, 2.0])
    target = gap_to_bin(gaps, EDGES)
    ours = head.log_bin_probs(x).gather(-1, target[:, None]).squeeze(-1)
    theirs = hazard_log_likelihood(head.net(x), target)
    closed = target < len(EDGES) + 1
    assert torch.allclose(ours[closed], theirs[closed], atol=1e-5)
    # The open bin differs by design: the probe gives it ALL remaining mass
    # (a proper distribution), the production likelihood scores it as
    # h_last * S(last-1) and leaves (1 - h_last) * S(last-1) unassigned.
    assert (ours[~closed] >= theirs[~closed]).all()
    assert torch.allclose(head.log_bin_probs(x).exp().sum(-1), torch.ones(8), atol=1e-5)
    assert isinstance(make_head("categorical", 4, EDGES), CategoricalProbe)


def test_collect_feature_bank_and_run_probe_on_a_tiny_model() -> None:
    rows = []
    for sid in range(1, 7):
        t = T0
        for k in range(30):
            t = t + timedelta(hours=0 if k % 3 == 1 else (1 if sid % 2 else 6))
            rows.append((sid, "LAB//220045//bpm", t, 80.0 + k, 100 + sid))
    events = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )
    binned = add_value_tokens(events)
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=EDGES,
    )
    full = collect_feature_bank(model, binned, vocab, num_lanes=2, chunk_size=16)
    assert full.features.shape[1] == 8 and len(full) == full.n_positions_seen
    # <= 29 valid gaps per subject: the last position has no target, and a
    # position whose next token sits in the next chunk is not scored (the
    # training loss's own in-chunk rule, gap_survival_valid_mask)
    assert 6 * 20 < len(full) <= 6 * 29
    half = collect_feature_bank(
        model, binned, vocab, num_lanes=2, chunk_size=16, sample_rate=0.5, seed=1
    )
    assert 0 < len(half) < len(full) and half.n_positions_seen == len(full)
    capped = collect_feature_bank(
        model, binned, vocab, num_lanes=2, chunk_size=16, max_positions=20
    )
    assert len(capped) == 20
    results = run_probe(
        model,
        vocab,
        {"train": full, "tuning": full, "held_out": full},
        heads=("hazard", "lognormal2"),
        epochs=2,
    )
    assert [r.head for r in results] == ["hazard", "lognormal2"]
    assert all(r.n_positions == len(full) for r in results)
    assert results[1].continuous_nll is not None
