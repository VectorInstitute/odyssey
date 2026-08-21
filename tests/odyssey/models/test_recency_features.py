"""The recency (staleness) channel: computation, threading, heads, learning."""

import math
from datetime import datetime, timedelta

import polars as pl
import torch

from odyssey.data.sequences import (
    N_RECENCY_FAMILIES,
    build_patient_sequence,
)
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.vocabulary import LAB_TYPE, MEDICATION_TYPE, Vocabulary
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    RECENCY_DIM,
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)


T0 = datetime(2024, 1, 1)


def _events() -> pl.DataFrame:
    rows = [
        (1, "LAB//A", T0, 1.0, 10),
        (1, "MEDICATION//X", T0 + timedelta(hours=2), None, 10),
        (1, "LAB//A", T0 + timedelta(hours=5), 2.0, 10),
        (1, "DIAGNOSIS//D", T0 + timedelta(hours=6), None, 10),
    ]
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


def test_family_recency_is_hours_since_previous_family_event() -> None:
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab)
    rec = seq.family_recency
    assert len(rec) == 4 and all(len(r) == N_RECENCY_FAMILIES for r in rec)
    lab, med = LAB_TYPE - 1, MEDICATION_TYPE - 1
    assert all(math.isnan(v) for v in rec[0])  # nothing seen before first event
    assert math.isnan(rec[1][med]) and rec[1][lab] == 2.0  # lab was 2h ago
    assert rec[2][lab] == 5.0 and rec[2][med] == 3.0  # prior lab at t=0, med at t=2
    assert rec[3][lab] == 1.0  # lab refreshed at t=5, now t=6


def test_recency_survives_chunking_and_padding() -> None:
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab)
    chunks = list(
        PackedLaneSampler(iter([seq]), num_lanes=1, chunk_size=3, reset_prob=0.0)
    )
    got = torch.cat([c.batch.aux.family_recency[0] for c in chunks], dim=0)
    lab = LAB_TYPE - 1
    # positions 0..2 are inputs across chunks; values match the per-patient truth
    assert got.shape[1] == N_RECENCY_FAMILIES
    assert torch.isnan(got[0]).all()
    assert got[1][lab].item() == 2.0
    assert got[2][lab].item() == 5.0


def _mk(model_cls, recency: bool, **kw):
    torch.manual_seed(0)
    backbone = TinyGRUBackbone(vocab_size=12, hidden_size=8, num_layers=1, padding_idx=0)
    return model_cls(
        backbone=backbone,
        vocab_size=12,
        padding_idx=0,
        time_bin_edges=(1.0, 8.0),
        event_names=["death"],
        recency_features=recency,
        **kw,
    )


def test_head_widths_and_forward_shapes() -> None:
    base_off = _mk(BaselineSequenceModel, False)
    base_on = _mk(BaselineSequenceModel, True)
    assert base_on.time_head.proj.in_features == 8 + RECENCY_DIM
    assert base_off.time_head.proj.in_features == 8
    cb_on = _mk(
        ConceptBottleneckSequenceModel, True, num_concepts=3, embedding_dim=4
    )
    assert cb_on.time_head.proj.in_features == cb_on.bottleneck.output_dim + RECENCY_DIM
    # forward_with_features returns augmented features that fit the heads
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab)
    chunk = next(
        iter(PackedLaneSampler(iter([seq]), num_lanes=1, chunk_size=8, reset_prob=0.0))
    )
    for m in (base_on, cb_on):
        m.eval()
        fwd = m.forward_with_features(chunk.batch, state=None, reset_mask=None)
        out = m.time_head(fwd.features)
        assert torch.isfinite(out).all()
    # a batch without the channel still works (zeros block)
    aux = chunk.batch.aux._replace(family_recency=None)
    fwd = base_on.forward_with_features(
        chunk.batch._replace(aux=aux), state=None, reset_mask=None
    )
    assert fwd.features.shape[-1] == 8 + RECENCY_DIM
