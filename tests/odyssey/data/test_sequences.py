"""Tests for converting raw MEDS events into patient token sequences."""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from odyssey.data.sequences import (
    HOURS_PER_YEAR,
    build_patient_sequence,
    collate_patient_sequences,
)
from odyssey.data.vocabulary import PAD_ID, UNK_ID, Vocabulary
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


def _events(rows: list) -> pl.DataFrame:
    """rows: list of (subject_id, time_or_None, code, hadm_id_or_None)."""
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


VOCAB = Vocabulary.build(
    ["DIAGNOSIS//A", "LAB//220045//bpm", "MEDICATION//X"], min_count=1
)
T0 = datetime(2180, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# build_patient_sequence
# ---------------------------------------------------------------------------


def test_events_sorted_by_time_regardless_of_input_order() -> None:
    events = _events(
        [
            (1, T0 + timedelta(hours=2), "LAB//220045//bpm", None),
            (1, T0, "DIAGNOSIS//A", None),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.concept_ids == [
        VOCAB.encode("DIAGNOSIS//A"),
        VOCAB.encode("MEDICATION//X"),
        VOCAB.encode("LAB//220045//bpm"),
    ]
    assert seq.time_stamps == [0.0, 1.0, 2.0]


def test_static_timeless_facts_are_dropped() -> None:
    events = _events(
        [
            (1, None, "GENDER//F", None),
            (1, T0, "DIAGNOSIS//A", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert len(seq) == 1


def test_meds_birth_computes_ages_and_is_excluded_from_tokens() -> None:
    birth = T0 - timedelta(days=30 * 365.25)  # ~30 years before first event
    events = _events(
        [
            (1, birth, "MEDS_BIRTH", None),
            (1, T0, "DIAGNOSIS//A", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert len(seq) == 1  # MEDS_BIRTH itself isn't a sequence token
    assert abs(seq.ages[0] - 30.0) < 0.1


def test_ages_are_zero_without_a_birth_event() -> None:
    events = _events([(1, T0, "DIAGNOSIS//A", None)])
    seq = build_patient_sequence(events, VOCAB)
    assert seq.ages == [0.0]


def test_unknown_code_maps_to_unk_token() -> None:
    events = _events([(1, T0, "NEVER_SEEN//CODE", None)])
    seq = build_patient_sequence(events, VOCAB)
    assert seq.concept_ids == [UNK_ID]


def test_max_seq_len_keeps_most_recent_events() -> None:
    events = _events(
        [(1, T0 + timedelta(hours=i), f"DIAGNOSIS//{i}", None) for i in range(5)]
    )
    vocab = Vocabulary.build([f"DIAGNOSIS//{i}" for i in range(5)], min_count=1)
    seq = build_patient_sequence(events, vocab, max_seq_len=2)
    assert len(seq) == 2
    assert seq.time_stamps == [3.0, 4.0]  # kept the last two, times unshifted


def test_empty_events_produce_empty_sequence() -> None:
    events = _events([])
    seq = build_patient_sequence(events, VOCAB)
    assert len(seq) == 0


# ---------------------------------------------------------------------------
# Visit derivation
# ---------------------------------------------------------------------------


def test_same_hadm_id_forms_one_visit_with_first_middle_last_segments() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 100),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", 100),
            (1, T0 + timedelta(hours=2), "LAB//220045//bpm", 100),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 0, 0]
    assert seq.visit_segments == [0, 1, 2]


def test_events_without_hadm_id_each_get_their_own_visit() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", None),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 1]
    assert seq.visit_segments == [0, 0]  # each is a single-event "visit"


def test_distinct_hadm_ids_get_distinct_visit_orders() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 100),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", 200),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 1]


def test_visit_order_capped_at_max_num_visits() -> None:
    events = _events(
        [(1, T0 + timedelta(hours=i), "DIAGNOSIS//A", 100 + i) for i in range(5)]
    )
    seq = build_patient_sequence(events, VOCAB, max_num_visits=3)
    assert max(seq.visit_orders) == 2  # capped at max_num_visits - 1


# ---------------------------------------------------------------------------
# collate_patient_sequences
# ---------------------------------------------------------------------------


def test_collate_pads_to_longest_sequence() -> None:
    events_a = _events([(1, T0, "DIAGNOSIS//A", None)])
    events_b = _events(
        [
            (2, T0, "DIAGNOSIS//A", None),
            (2, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq_a = build_patient_sequence(events_a, VOCAB)
    seq_b = build_patient_sequence(events_b, VOCAB)

    batch = collate_patient_sequences([seq_a, seq_b])

    assert batch.concept_ids.shape == (2, 2)
    assert batch.concept_ids[0, 1].item() == PAD_ID  # padded position
    assert batch.concept_ids[0, 0].item() == VOCAB.encode("DIAGNOSIS//A")
    assert batch.concept_ids[1, 0].item() == VOCAB.encode("DIAGNOSIS//A")
    assert batch.concept_ids[1, 1].item() == VOCAB.encode("MEDICATION//X")


def test_collate_output_types_and_aux_shapes_match() -> None:
    seq = build_patient_sequence(_events([(1, T0, "DIAGNOSIS//A", None)]), VOCAB)
    batch = collate_patient_sequences([seq])

    assert batch.concept_ids.dtype == torch.long
    assert batch.aux.type_ids.shape == batch.concept_ids.shape
    assert batch.aux.time_stamps.shape == batch.concept_ids.shape
    assert batch.aux.ages.shape == batch.concept_ids.shape
    assert batch.aux.visit_orders.shape == batch.concept_ids.shape
    assert batch.aux.visit_segments.shape == batch.concept_ids.shape


def test_collate_empty_list_produces_empty_batch() -> None:
    batch = collate_patient_sequences([])
    assert batch.concept_ids.shape == (0, 0)


def test_hours_per_year_constant_is_a_julian_year() -> None:
    assert abs(HOURS_PER_YEAR - 24.0 * 365.25) < 1e-9


# ---------------------------------------------------------------------------
# Real MEDS data, end to end: extraction -> vocab -> sequences -> model
# ---------------------------------------------------------------------------


@pytest.mark.integration_test
def test_real_meds_data_tokenizes_and_runs_through_the_model(tmp_path: Path) -> None:
    """The full path a training script will actually take, on real data.

    Runs the real meds-extract pipeline against the public MIMIC-IV demo,
    builds a vocabulary from the resulting events, tokenizes a batch of
    real patients, and runs that batch through
    ConceptBottleneckSequenceModel end to end -- proving the tokenization
    output is actually shaped and typed the way the model expects, not
    just that each piece works in isolation.
    """
    output_dir = tmp_path / "meds_demo"
    result = subprocess.run(
        [
            "meds-extract-run",
            "spec=MIMIC-IV",
            f"output_dir={output_dir}",
            "dataset_key=demo",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    shards = sorted((output_dir / "data" / "train").glob("*.parquet"))[:3]
    events = pl.concat([pl.read_parquet(s) for s in shards])
    events = events.select(["subject_id", "time", "code", "hadm_id"])

    vocab = Vocabulary.build(events["code"].to_list(), min_count=2, max_size=2000)
    assert len(vocab) > 2  # more than just PAD/UNK

    subject_ids = events["subject_id"].unique().to_list()[:8]
    sequences = [
        build_patient_sequence(
            events.filter(pl.col("subject_id") == sid), vocab, max_seq_len=64
        )
        for sid in subject_ids
    ]
    sequences = [s for s in sequences if len(s) > 0]
    assert sequences, "expected at least one non-empty real patient sequence"

    batch = collate_patient_sequences(sequences)
    assert batch.concept_ids.shape[0] == len(sequences)
    assert batch.concept_ids.max().item() < len(vocab)
    assert batch.aux.visit_segments.max().item() <= 2

    backbone = TinyGRUBackbone(
        vocab_size=len(vocab), hidden_size=16, padding_idx=PAD_ID
    )
    model = ConceptBottleneckSequenceModel(
        backbone=backbone,
        vocab_size=len(vocab),
        num_concepts=4,
        embedding_dim=4,
        padding_idx=PAD_ID,
    )
    concept_labels = torch.zeros(len(sequences), 4)
    total, _ = model.compute_loss(batch, concept_labels)
    assert torch.isfinite(total)
    total.backward()
    assert model.bottleneck.context_proj.weight.grad is not None
