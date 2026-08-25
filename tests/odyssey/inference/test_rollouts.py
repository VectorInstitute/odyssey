"""Sampled forward rollouts: sampling mechanics, structure, and the hazard check."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from odyssey.data.alert_events import ALERT_EVENTS
from odyssey.data.sequences import build_patient_sequence
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary, code_type
from odyssey.inference.rollouts import (
    _sample_code,
    _sample_gap_hours,
    hazard_probabilities_at,
    rollout_from_position,
    summarize_rollouts,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import BaselineSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2024, 1, 1)
EDGES = DEFAULT_TIME_BIN_EDGES_HOURS


def _events(n_subjects: int = 3, n_events: int = 30) -> pl.DataFrame:
    rows = []
    for sid in range(1, n_subjects + 1):
        for k in range(n_events):
            rows.append(
                (
                    sid,
                    "LAB//220045//bpm" if k % 3 else "MEDICATION//norepinephrine//X",
                    T0 + timedelta(hours=k),
                    80.0,
                    100 + sid,
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def _model(vocab: Vocabulary) -> BaselineSequenceModel:
    torch.manual_seed(0)
    return BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=EDGES,
        event_names=["vasopressor_start", "death"],
    )


def test_gap_sampling_respects_the_hazard_and_the_same_instant_bin() -> None:
    """Bin 0 draws exactly 0; a bin's draw lands inside that bin's range."""
    g = torch.Generator().manual_seed(0)
    n_bins = len(EDGES) + 2
    # all mass on bin 0 (same instant): every draw is exactly 0
    same = torch.full((n_bins,), -20.0)
    same[0] = 20.0
    assert all(_sample_gap_hours(same, EDGES, g) == 0.0 for _ in range(20))
    # all mass on bin 3, i.e. the range (edges[1], edges[2]]
    later = torch.full((n_bins,), -20.0)
    later[3] = 20.0
    draws = [_sample_gap_hours(later, EDGES, g) for _ in range(50)]
    assert all(EDGES[1] <= d <= EDGES[2] for d in draws)
    assert len(set(draws)) > 1  # uniform within the bin, not a constant
    # the open tail bin draws beyond the last edge
    tail = torch.full((n_bins,), -20.0)
    tail[-1] = 20.0
    assert all(_sample_gap_hours(tail, EDGES, g) > EDGES[-1] for _ in range(10))


def test_code_sampling_excludes_padding_and_honors_top_k() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.zeros(10)
    logits[PAD_ID] = 100.0  # would dominate if not excluded
    assert all(
        _sample_code(logits.clone(), g, temperature=1.0, top_k=None) != PAD_ID
        for _ in range(30)
    )
    peaked = torch.zeros(10)
    peaked[7] = 10.0
    assert {
        _sample_code(peaked.clone(), g, temperature=1.0, top_k=1) for _ in range(10)
    } == {7}
    # temperature -> 0 is argmax
    soft = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert {
        _sample_code(soft.clone(), g, temperature=1e-4, top_k=None) for _ in range(5)
    } == {3}


def test_rollout_is_seeded_bounded_and_advances_structure() -> None:
    events = _events()
    binned = add_value_tokens(events)
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    model = _model(vocab)
    seq = build_patient_sequence(binned.filter(pl.col("subject_id") == 1), vocab)
    position = len(seq) - 5
    index_time = seq.time_stamps[position]
    common = {
        "position": position,
        "horizon_hours": 24.0,
        "n_samples": 4,
        "max_steps": 20,
        "device": "cpu",
    }
    a = rollout_from_position(model, seq, vocab, seed=0, **common)
    b = rollout_from_position(model, seq, vocab, seed=0, **common)
    c = rollout_from_position(model, seq, vocab, seed=1, **common)
    assert [s.codes for s in a] == [s.codes for s in b]  # seeded, reproducible
    assert [s.codes for s in a] != [s.codes for s in c]  # different seed differs
    for s in a:
        assert len(s.codes) == len(s.times) <= 20
        assert all(t >= index_time for t in s.times)  # never before the index
        assert all(t <= index_time + 24.0 for t in s.times)  # nor past the horizon
        assert s.times == sorted(s.times)  # time never runs backwards
        assert all(c != "[PAD]" for c in s.codes)


def test_summary_reports_event_fractions_and_family_counts() -> None:
    events = _events()
    binned = add_value_tokens(events)
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    model = _model(vocab)
    seq = build_patient_sequence(binned.filter(pl.col("subject_id") == 1), vocab)
    position = len(seq) - 5
    samples = rollout_from_position(
        model,
        seq,
        vocab,
        position=position,
        horizon_hours=24.0,
        n_samples=8,
        max_steps=24,
        seed=0,
        device="cpu",
    )
    summary = summarize_rollouts(
        samples,
        ALERT_EVENTS,
        vocab,
        subject_id=seq.subject_id,
        index_time_hours=seq.time_stamps[position],
        horizons=(8.0, 24.0),
    )
    assert summary.n_samples == 8
    assert set(summary.event_probability) == {a.name for a in ALERT_EVENTS}
    for per_horizon in summary.event_probability.values():
        assert set(per_horizon) == {"8h", "24h"}
        assert all(0.0 <= v <= 1.0 for v in per_horizon.values())
        # a longer horizon can only include more sampled events
        assert per_horizon["24h"] >= per_horizon["8h"]
    assert set(summary.family_counts) == {"8h", "24h"}
    total_24h = sum(summary.family_counts["24h"].values())
    assert total_24h == pytest.approx(
        sum(len(s.events_within(summary.index_time_hours, 24.0)) for s in samples) / 8
    )
    # family keys are code_type ids of the sampled codes
    sampled_families = {
        str(code_type(c))
        for s in samples
        for c in s.events_within(summary.index_time_hours, 24.0)
    }
    assert set(summary.family_counts["24h"]) == sampled_families


def test_hazard_probabilities_are_available_for_the_same_position() -> None:
    """The closed-form head output the sampled fractions are checked against."""
    events = _events()
    binned = add_value_tokens(events)
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    model = _model(vocab)
    features = torch.randn(model.time_head.proj.in_features)
    hz = hazard_probabilities_at(model, features, ALERT_EVENTS, (8.0, 24.0))
    assert set(hz) == {"vasopressor_start", "death"}  # only events with heads
    for per_horizon in hz.values():
        assert set(per_horizon) == {"8h", "24h"}
        assert all(0.0 <= v <= 1.0 for v in per_horizon.values())
        assert per_horizon["24h"] >= per_horizon["8h"]  # monotone in the horizon
    # a model without event heads reports nothing rather than raising
    bare = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=EDGES,
    )
    assert hazard_probabilities_at(bare, features, ALERT_EVENTS, (8.0,)) == {}
