"""Tests for the concept intervention / completeness harness (CPU)."""

from datetime import datetime, timedelta
from typing import Dict, Tuple

import polars as pl
import torch

from odyssey.data.concepts import concepts_for_source
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.interventions import (
    INTERVENTION_MODES,
    run_streaming_intervention,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.data import (
    build_visit_concept_first_times,
    iter_patient_sequences,
)
from odyssey.training.running_labels import position_running_labels


T0 = datetime(2024, 1, 1)
NUM_CONCEPTS = 3
CODES = [f"LAB//{i}//" for i in range(10)]


def _events() -> pl.DataFrame:
    rows = []
    for sid in (1, 2, 3):
        for i in range(20):
            rows.append(
                (
                    sid,
                    CODES[(sid * 3 + i) % len(CODES)],
                    T0 + timedelta(hours=i),
                    None,
                    100 + sid,
                )
            )
    return pl.DataFrame(
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


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({c: i + 2 for i, c in enumerate(CODES)})
    return Vocabulary(tokens)


def _model(vocab_size: int) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        padding_idx=0,
    )


def _labels_and_masks() -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    labels = {
        1: torch.tensor([1.0, 0.0, 1.0]),
        2: torch.tensor([0.0, 0.0, 0.0]),
        3: torch.tensor([1.0, 1.0, 0.0]),
    }
    masks = {sid: torch.ones(NUM_CONCEPTS) for sid in labels}
    return labels, masks


def _run(mode: str, seed: int = 0):
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    return run_streaming_intervention(
        _model(len(vocab)),
        _events(),
        vocab,
        labels,
        masks,
        mode=mode,
        supervision="stay",
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        seed=seed,
    )


def test_every_mode_runs_and_scores_every_real_position() -> None:
    for mode in INTERVENTION_MODES:
        result = _run(mode)
        # 3 subjects x 20 events, minus one position per subject (the
        # final event has no next-token target).
        assert result.n_predictions == 3 * 19, mode
        assert 0.0 <= result.top1_accuracy <= 1.0, mode
        assert result.mean_task_loss > 0.0, mode


def test_probs_modes_move_the_logits_and_count_intervened_positions() -> None:
    baseline = _run("none")
    truth = _run("truth")
    flip = _run("flip")
    # Every position belongs to a fully-observed subject, so every
    # input position is intervened in truth/flip modes.
    assert truth.n_intervened_positions == 3 * 20
    assert baseline.n_intervened_positions == 0
    # An untrained model has no meaningful direction, but forcing
    # extreme (0/1) mixing probabilities must move the task loss.
    assert truth.mean_task_loss != baseline.mean_task_loss
    assert flip.mean_task_loss != truth.mean_task_loss


def test_zero_modes_change_task_loss() -> None:
    baseline = _run("none")
    assert _run("zero_known").mean_task_loss != baseline.mean_task_loss
    assert _run("zero_unknown").mean_task_loss != baseline.mean_task_loss


def test_random_mode_is_deterministic_per_seed() -> None:
    a = _run("random", seed=7)
    b = _run("random", seed=7)
    assert a.mean_task_loss == b.mean_task_loss
    assert a.top1_accuracy == b.top1_accuracy


def test_unobserved_concepts_are_never_intervened() -> None:
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    # Nothing observed anywhere: truth mode must reduce to the baseline.
    masks = {sid: torch.zeros(NUM_CONCEPTS) for sid in masks}
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
    }
    baseline = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="none", **kwargs
    )
    truth = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="truth", **kwargs
    )
    assert truth.n_intervened_positions == 0
    assert truth.mean_task_loss == baseline.mean_task_loss
    assert truth.top1_accuracy == baseline.top1_accuracy


def test_running_labels_are_false_before_first_trigger() -> None:
    """A concept is injected as 1 only from its first-trigger time on."""
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    # Subject 1's concept 0 first triggers at hour 10 (events are hourly),
    # concept 2 at hour 0; concept 1 never (label 0, inf).
    first = {
        1: torch.tensor([10.0, float("inf"), 0.0]),
        2: torch.tensor([float("inf")] * 3),
        3: torch.tensor([0.0, 5.0, float("inf")]),
    }
    seqs = iter_patient_sequences(_events(), vocab)
    sampler = PackedLaneSampler(seqs, num_lanes=1, chunk_size=64, reset_prob=0.0)
    chunk = next(iter(sampler))
    pos_labels, observed = position_running_labels(
        chunk, labels, masks, first, supervision="stay", num_concepts=NUM_CONCEPTS
    )
    sid = chunk.subject_ids[0]
    times = chunk.batch.aux.time_stamps[0]
    s1 = sid == 1
    # concept 0 for subject 1: 0 before hour 10, 1 from hour 10 on
    assert torch.equal(pos_labels[0, s1, 0], (times[s1] >= 10.0).float())
    # concept 2 for subject 1: 1 everywhere (triggered at hour 0)
    assert pos_labels[0, s1, 2].eq(1.0).all()
    # concept 1 for subject 1: label 0 -> 0 everywhere
    assert pos_labels[0, s1, 1].eq(0.0).all()
    assert observed[0, s1].eq(1.0).all()


def test_first_time_builders_align_with_sequence_time_origin() -> None:
    concepts = [c for c in concepts_for_source("mimic_iv") if c.name == "tachycardia"]
    events = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "code": ["LAB//50912//x", "LAB//220045//bpm", "LAB//220045//bpm"],
            "numeric_value": [1.0, 80.0, 130.0],
            "time": [T0, T0 + timedelta(hours=2), T0 + timedelta(hours=7)],
            "hadm_id": [10, 10, 10],
        }
    )
    first = build_visit_concept_first_times(events, concepts)
    # 7 hours after the subject's first event (the creatinine at T0).
    assert first[(1, 10)].tolist() == [7.0]


def test_uncertain_band_limits_replacement_and_reports_displacement() -> None:
    """A band replaces only entries near p=0.5 and reports their displacement."""
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
    }
    full = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="truth", **kwargs
    )
    banded = run_streaming_intervention(
        model,
        _events(),
        vocab,
        labels,
        masks,
        mode="truth",
        uncertain_band=0.05,
        **kwargs,
    )
    assert full.mean_abs_displacement is not None
    assert 0.0 < full.mean_abs_displacement <= 1.0
    assert banded.uncertain_band == 0.05
    assert banded.n_intervened_positions <= full.n_intervened_positions
    # Inside a +/-0.05 band around 0.5 no displacement can exceed 0.55.
    if banded.mean_abs_displacement is not None:
        assert banded.mean_abs_displacement <= 0.55
