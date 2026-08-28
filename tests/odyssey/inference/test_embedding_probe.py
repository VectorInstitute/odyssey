"""Tests for odyssey.inference.embedding_probe.

collect_embeddings/labels_for were promoted out of
scripts/probe_bottleneck_signal.py on 2026-08-28 with zero prior test
coverage (an ad hoc diagnostic script, run manually). This is the first
real test of the streaming embedding-collection pass itself.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl
import torch

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.alerts import IndexRow, _visit_starts, outcome_at_horizon
from odyssey.inference.embedding_probe import collect_embeddings, labels_for
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


T0 = datetime(2024, 1, 1)
_EventRow = tuple[int, str, datetime, Optional[float], int]


def _events(n_subjects: int = 8) -> pl.DataFrame:
    """Hourly heart-rate readings; every other subject starts a vasopressor at 14h."""
    rows: list[_EventRow] = []
    for sid in range(1, n_subjects + 1):
        hadm = 1000 + sid
        for h in range(24):
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), 80.0, hadm))
        if sid % 2 == 0:
            rows.append(
                (
                    sid,
                    "MEDICATION//norepinephrine//Administered",
                    T0 + timedelta(hours=14),
                    None,
                    hadm,
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


def _vocab(events_binned: pl.DataFrame) -> Vocabulary:
    return Vocabulary.build(events_binned["code"].to_list(), min_count=1)


def _model(vocab_size: int, num_concepts: int) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=num_concepts,
        embedding_dim=4,
        padding_idx=0,
    )


def test_collect_embeddings_landmark_keys_match_the_model_free_row_count() -> None:
    """~6 landmark rows per 24h visit at a 4h cadence, one embedding per row."""
    events = _events(8)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    model = _model(len(vocab), num_concepts=3)

    lm_keys, lm_pre, lm_post, ve_keys, ve_pre, ve_post = collect_embeddings(
        model,
        binned,
        vocab,
        landmark_alerts=ALERT_EVENTS,
        visit_end_alerts=[],
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
    )

    assert len(lm_keys) >= 8 * 5  # 24h / 4h buckets, minus edge effects
    assert lm_pre.shape == (len(lm_keys), model.backbone.hidden_size)
    assert lm_post.shape == (len(lm_keys), model.bottleneck.output_dim)
    # no visit_end_alerts requested -> the visit-end pass never ran
    assert ve_keys == []
    assert ve_pre.shape == (0, model.backbone.hidden_size)
    assert ve_post.shape == (0, model.bottleneck.output_dim)


def test_collect_embeddings_visit_end_pass_is_gated_by_visit_end_alerts() -> None:
    """visit_end_alerts non-empty -> one row per visit, at its last event."""
    events = _events(4)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    model = _model(len(vocab), num_concepts=3)
    readmission = [a for a in ALERT_EVENTS if a.name == "death"]  # any AlertEvent

    _, _, _, ve_keys, ve_pre, ve_post = collect_embeddings(
        model,
        binned,
        vocab,
        landmark_alerts=[],
        visit_end_alerts=readmission,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
    )

    assert len(ve_keys) == 4  # one row per subject's single visit
    assert ve_pre.shape == (4, model.backbone.hidden_size)
    assert ve_post.shape == (4, model.bottleneck.output_dim)


def test_labels_for_matches_outcome_at_horizon_row_by_row() -> None:
    """labels_for is exactly outcome_at_horizon over each key, nan for None."""
    events = _events(8)
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")["vasopressor_start"]
    keys = [(2, 1002, 2.0), (2, 1002, 10.0), (2, 1002, 15.0), (1, 1001, 20.0)]

    labels = labels_for(keys, times, 8.0)

    expected = [outcome_at_horizon(IndexRow(s, v, t), times, 8.0) for s, v, t in keys]
    for label, exp in zip(labels, expected):
        if exp is None:
            assert np.isnan(label)
        else:
            assert label == exp
