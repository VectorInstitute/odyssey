"""Tests for the concept intervention / completeness harness (CPU)."""

from datetime import datetime, timedelta
from typing import Dict, Tuple

import polars as pl
import torch

from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.interventions import (
    INTERVENTION_MODES,
    run_streaming_intervention,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


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
