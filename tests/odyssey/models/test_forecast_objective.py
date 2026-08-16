"""Tests for the bundle-invariant forecasting objective and the time head."""

import math
from typing import List

import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import PackedLaneSampler
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
    ForecastObjective,
    _bundle_log_likelihood,
)
from odyssey.models.time_to_event import (
    DEFAULT_TIME_BIN_EDGES_HOURS,
    gap_to_bin,
    hazard_log_likelihood,
    probability_within,
    survival_curve,
)


VOCAB = 12


def _seq(subject_id: int, codes: List[int], times: List[float]) -> PatientSequence:
    n = len(codes)
    return PatientSequence(
        subject_id=subject_id,
        concept_ids=codes,
        type_ids=[1] * n,
        time_stamps=times,
        ages=[50.0] * n,
        visit_orders=[0] * n,
        visit_segments=[0] * n,
    )


def _chunk(seqs, chunk_size=16, num_lanes=1):
    sampler = PackedLaneSampler(
        iter(seqs), num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    return sampler.next_chunk()


def _slow_bundle_ll(logp, targets, times, subject_ids, real):
    """Compute the reference bundle log-likelihood with plain Python loops."""
    lanes, chunk = targets.shape
    out = torch.zeros(lanes, chunk)
    for lane in range(lanes):
        for i in range(chunk):
            if not real[lane, i]:
                continue
            if i == chunk - 1:
                out[lane, i] = logp[lane, i, targets[lane, i]]
                continue
            key = (subject_ids[lane, i + 1].item(), times[lane, i + 1].item())
            seen = set()
            total = 0.0
            for j in range(i, chunk - 1):
                if not real[lane, j]:
                    continue
                if (subject_ids[lane, j + 1].item(), times[lane, j + 1].item()) != key:
                    continue
                tok = targets[lane, j].item()
                if tok in seen:
                    continue
                seen.add(tok)
                total += math.exp(logp[lane, i, tok].item())
            out[lane, i] = math.log(total)
    return out


def test_bundle_ll_matches_slow_reference_with_duplicates_and_two_lanes() -> None:
    torch.manual_seed(0)
    # subject 1: bundle at t=1 (codes 3,4,3 duplicate), t=2 (5), t=3 (6,7)
    s1 = _seq(1, [2, 3, 4, 3, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    s2 = _seq(2, [8, 9, 9, 10], [0.0, 0.0, 0.0, 5.0])
    chunk = _chunk([s1, s2], chunk_size=8, num_lanes=2)
    logits = torch.randn(*chunk.targets.shape, VOCAB)
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    fast = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    slow = _slow_bundle_ll(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    assert torch.allclose(fast, slow, atol=1e-5), (fast, slow)
    # never exceeds log 1 (dedup keeps credited mass <= 1)
    assert (fast <= 1e-6).all()


def test_bundle_loss_equals_cross_entropy_on_singleton_bundles() -> None:
    torch.manual_seed(0)
    s = _seq(1, [2, 3, 4, 5, 6], [0.0, 1.0, 2.0, 3.0, 4.0])
    chunk = _chunk([s], chunk_size=8)
    logits = torch.randn(*chunk.targets.shape, VOCAB)
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    ll = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    ce = logp.gather(-1, chunk.targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(ll[real], ce[real], atol=1e-6)


def test_bundle_loss_lower_bounds_cross_entropy() -> None:
    torch.manual_seed(1)
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chunk = _chunk([s], chunk_size=8)
    logits = torch.randn(*chunk.targets.shape, VOCAB)
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    ll = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    ce = logp.gather(-1, chunk.targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    assert (ll[real] >= ce[real] - 1e-6).all()
    assert (ll[real] > ce[real] + 1e-6).any()


def _model(time: bool = False) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS if time else None,
    )


def test_default_objective_reproduces_original_loss() -> None:
    model = _model()
    model.eval()  # concept dropout would otherwise differ between passes
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chunk = _chunk([s], chunk_size=8)
    labels = {1: torch.tensor([1.0, 0.0])}
    a, comp_a, _ = model.compute_streaming_loss(chunk, labels)
    b, comp_b, _ = model.compute_streaming_loss(
        chunk, labels, objective=ForecastObjective()
    )
    assert torch.allclose(a, b)
    assert comp_a["time_loss"].item() == 0.0  # no time head


def test_family_weights_reweight_and_are_normalized() -> None:
    model = _model()
    model.eval()
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    chunk = _chunk([s], chunk_size=8)
    labels = {1: torch.tensor([1.0, 0.0])}
    token_types = torch.zeros(VOCAB, dtype=torch.long)
    token_types[[3, 5]] = 1  # family 1
    uniform = ForecastObjective(
        family_weights=torch.tensor([1.0, 1.0]), token_types=token_types
    )
    boosted = ForecastObjective(
        family_weights=torch.tensor([1.0, 5.0]), token_types=token_types
    )
    plain, _, _ = model.compute_streaming_loss(chunk, labels)
    u, _, _ = model.compute_streaming_loss(chunk, labels, objective=uniform)
    b, cb, _ = model.compute_streaming_loss(chunk, labels, objective=boosted)
    assert torch.allclose(plain, u, atol=1e-6)
    assert not torch.allclose(plain, b)
    assert torch.isfinite(cb["task_loss"])


def test_time_head_trains_and_survival_is_well_formed() -> None:
    model = _model(time=True)
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 0.0, 0.5, 6.0, 6.0, 30.0])
    chunk = _chunk([s], chunk_size=8)
    labels = {1: torch.tensor([1.0, 0.0])}
    obj = ForecastObjective(time_weight=1.0)
    total, comp, _ = model.compute_streaming_loss(chunk, labels, objective=obj)
    assert comp["time_loss"].item() > 0.0
    total.backward()
    assert model.time_head is not None
    assert model.time_head.proj.weight.grad is not None
    logits = torch.randn(4, model.time_head.num_bins)
    surv = survival_curve(logits)
    assert (surv[:, 1:] <= surv[:, :-1] + 1e-6).all()
    p_bins = torch.stack(
        [
            hazard_log_likelihood(logits, torch.full((4,), b)).exp()
            for b in range(model.time_head.num_bins)
        ],
        -1,
    )
    assert torch.allclose(p_bins.sum(-1) + surv[:, -1], torch.ones(4), atol=1e-5)
    assert (probability_within(logits, DEFAULT_TIME_BIN_EDGES_HOURS, 24.0) <= 1.0).all()


def test_gap_to_bin_edges() -> None:
    edges = DEFAULT_TIME_BIN_EDGES_HOURS
    g = torch.tensor([0.0, 1 / 60, 0.9 / 60, 24.0, 24.01, 1e6])
    assert gap_to_bin(g, edges).tolist() == [0, 1, 1, 10, 11, len(edges) + 1]


def test_baseline_model_shares_objective_and_time_head() -> None:
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
    )
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chunk = _chunk([s], chunk_size=8)
    total, comp, _ = model.compute_streaming_loss(
        chunk, objective=ForecastObjective(bundle_invariant=True, time_weight=0.5)
    )
    assert set(comp) == {"task_loss", "time_loss"}
    total.backward()
