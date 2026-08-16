"""Tests for the MEDS shard -> training-ready data pipeline."""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest
import torch

from odyssey.data.concepts import ConceptDefinition, ConceptRule
from odyssey.data.vocabulary import PAD_ID, UNK_ID
from odyssey.training.data import (
    _shuffle_buffered,
    build_concept_label_dicts,
    build_visit_concept_label_dicts,
    build_vocabulary,
    count_subjects,
    family_loss_weights,
    iter_patient_sequences,
    load_meds_shards,
)


T0 = datetime(2024, 1, 1, 0, 0)


_EventRow = Tuple[int, str, Optional[datetime], Optional[float], Optional[int]]


def _events(rows: List[_EventRow]) -> pl.DataFrame:
    """rows: (subject_id, code, time, numeric_value_or_None, hadm_id_or_None)."""
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


# ---------------------------------------------------------------------------
# load_meds_shards
# ---------------------------------------------------------------------------


def test_load_meds_shards_concatenates_in_numeric_filename_order(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _events([(1, "A", T0, None, None)]).write_parquet(shard_dir / "0.parquet")
    _events([(2, "B", T0, None, None)]).write_parquet(shard_dir / "1.parquet")
    _events([(3, "C", T0, None, None)]).write_parquet(shard_dir / "10.parquet")

    out = load_meds_shards(shard_dir)

    assert out.height == 3
    assert set(out["subject_id"].to_list()) == {1, 2, 3}


def test_load_meds_shards_respects_max_shards(tmp_path: Path) -> None:
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _events([(1, "A", T0, None, None)]).write_parquet(shard_dir / "0.parquet")
    _events([(2, "B", T0, None, None)]).write_parquet(shard_dir / "1.parquet")

    out = load_meds_shards(shard_dir, max_shards=1)

    assert out.height == 1
    assert out["subject_id"].to_list() == [1]


def test_load_meds_shards_raises_when_directory_has_no_shards(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_meds_shards(empty_dir)


def test_load_meds_shards_drops_unused_columns(tmp_path: Path) -> None:
    # Real MIMIC-IV MEDS shards carry ~16 extra columns (drg_severity,
    # icustay_id, order_id, ...) nothing in this pipeline reads -- loading
    # them anyway was the dominant real memory cost at full-extraction
    # scale (OOM-confirmed via dmesg), not row count. Widen the schema
    # here and confirm they get projected away before collecting, not
    # just left unused.
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    pl.DataFrame(
        {
            "subject_id": [1],
            "code": ["A"],
            "time": [T0],
            "numeric_value": [None],
            "hadm_id": [None],
            "icustay_id": [999],
            "drg_severity": [1],
            "text_value": ["unused"],
        },
        schema_overrides={"numeric_value": pl.Float32, "hadm_id": pl.Int64},
    ).write_parquet(shard_dir / "0.parquet")

    out = load_meds_shards(shard_dir)

    assert set(out.columns) == {
        "subject_id",
        "code",
        "time",
        "numeric_value",
        "hadm_id",
    }


# ---------------------------------------------------------------------------
# iter_patient_sequences
# ---------------------------------------------------------------------------


def test_iter_patient_sequences_yields_one_per_subject() -> None:
    events = _events(
        [
            (1, "DIAGNOSIS//A", T0, None, None),
            (1, "MEDICATION//B", T0 + timedelta(hours=1), None, None),
            (2, "DIAGNOSIS//C", T0, None, None),
        ]
    )
    vocab = build_vocabulary(events, min_count=1, max_size=100)

    seqs = list(iter_patient_sequences(events, vocab))

    assert {s.subject_id for s in seqs} == {1, 2}
    subject_1 = next(s for s in seqs if s.subject_id == 1)
    assert len(subject_1) == 2


def test_iter_patient_sequences_skips_subjects_with_no_real_tokens() -> None:
    # Subject 2 has only a static, timeless fact (null time) -- dropped by
    # build_patient_sequence, leaving an empty sequence that must not be
    # yielded (PackedLaneSampler already handles empty patients defensively,
    # but this keeps the sampler's queue free of no-op entries).
    events = _events(
        [
            (1, "DIAGNOSIS//A", T0, None, None),
            (2, "GENDER//F", None, None, None),
        ]
    )
    vocab = build_vocabulary(events, min_count=1, max_size=100)

    seqs = list(iter_patient_sequences(events, vocab))

    assert {s.subject_id for s in seqs} == {1}


def test_iter_patient_sequences_shuffle_seed_is_deterministic() -> None:
    events = _events([(i, f"DIAGNOSIS//{i}", T0, None, None) for i in range(20)])
    vocab = build_vocabulary(events, min_count=1, max_size=100)

    order_a = [
        s.subject_id for s in iter_patient_sequences(events, vocab, shuffle_seed=7)
    ]
    order_b = [
        s.subject_id for s in iter_patient_sequences(events, vocab, shuffle_seed=7)
    ]
    order_unshuffled = [s.subject_id for s in iter_patient_sequences(events, vocab)]

    assert order_a == order_b
    assert sorted(order_a) == sorted(order_unshuffled)
    assert order_a != order_unshuffled  # exceedingly unlikely to coincide by chance


# ---------------------------------------------------------------------------
# _shuffle_buffered
# ---------------------------------------------------------------------------


def test_shuffle_buffered_yields_every_item_exactly_once_when_over_buffer_size() -> (
    None
):
    # 500 items through a buffer of only 16 -- exercises the yield-and-
    # replace branch many times over, not just the final drain.
    items = list(range(500))
    out = list(_shuffle_buffered(iter(items), buffer_size=16, rng=random.Random(0)))
    assert sorted(out) == items
    assert out != items  # exceedingly unlikely to coincide by chance


def test_shuffle_buffered_is_deterministic_given_a_seed() -> None:
    items = list(range(500))
    out_a = list(_shuffle_buffered(iter(items), buffer_size=16, rng=random.Random(3)))
    out_b = list(_shuffle_buffered(iter(items), buffer_size=16, rng=random.Random(3)))
    assert out_a == out_b


def test_shuffle_buffered_fully_shuffles_when_input_is_smaller_than_buffer() -> None:
    items = list(range(10))
    out = list(_shuffle_buffered(iter(items), buffer_size=4096, rng=random.Random(0)))
    assert sorted(out) == items


# ---------------------------------------------------------------------------
# build_concept_label_dicts
# ---------------------------------------------------------------------------


def test_build_concept_label_dicts_shapes_and_values() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        ),
    ]
    events = _events(
        [
            (1, "LAB//220045//bpm", T0, 120.0, None),  # triggers
            (2, "LAB//220045//bpm", T0, 80.0, None),  # does not trigger
            (3, "MEDICATION//X", T0, None, None),  # never observed
        ]
    )

    labels, masks = build_concept_label_dicts(events, concepts)

    assert labels[1].tolist() == [1.0]
    assert labels[2].tolist() == [0.0]
    assert masks[1].tolist() == [1.0]
    assert masks[3].tolist() == [0.0]


def test_build_concept_label_dicts_multiple_concepts_preserve_order() -> None:
    concepts = [
        ConceptDefinition("c1", [ConceptRule("LAB//A//", 1.0, "above")], "c1"),
        ConceptDefinition("c2", [ConceptRule("LAB//B//", 1.0, "above")], "c2"),
    ]
    events = _events(
        [
            (1, "LAB//A//", T0, 2.0, None),
            (1, "LAB//B//", T0, 0.5, None),
        ]
    )

    labels, _ = build_concept_label_dicts(events, concepts)

    assert labels[1].tolist() == [1.0, 0.0]  # c1 triggers, c2 does not


# ---------------------------------------------------------------------------
# count_subjects / build_vocabulary
# ---------------------------------------------------------------------------


def test_count_subjects() -> None:
    events = _events(
        [(1, "A", T0, None, None), (1, "B", T0, None, None), (2, "C", T0, None, None)]
    )
    assert count_subjects(events) == 2


def test_build_vocabulary_reserves_pad_and_unk() -> None:
    events = _events([(1, "DIAGNOSIS//A", T0, None, None)])
    vocab = build_vocabulary(events, min_count=1, max_size=100)
    assert vocab.encode("[PAD]") == PAD_ID
    assert vocab.encode("[UNK]") == UNK_ID
    assert vocab.encode("DIAGNOSIS//A") not in (PAD_ID, UNK_ID)


def test_build_vocabulary_respects_real_code_frequencies() -> None:
    # build_vocabulary counts via a vectorized Polars group_by rather than
    # materializing the full column as a Python list (see its docstring on
    # why) -- this checks that path actually produces min_count-correct
    # frequency filtering, not just that it runs.
    rows: List[_EventRow] = []
    for i in range(10):
        rows.append((i % 3, "DIAGNOSIS//frequent", T0, None, None))
    rows.append((0, "DIAGNOSIS//rare", T0, None, None))
    events = _events(rows)
    vocab = build_vocabulary(events, min_count=5, max_size=100)
    assert vocab.encode("DIAGNOSIS//frequent") not in (PAD_ID, UNK_ID)
    assert vocab.encode("DIAGNOSIS//rare") == UNK_ID


# ---------------------------------------------------------------------------
# build_visit_concept_label_dicts
# ---------------------------------------------------------------------------


def test_build_visit_concept_label_dicts_keys_and_shapes() -> None:

    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2],
            "time": [datetime(2020, 1, 1)] * 4,
            "code": ["LAB//220045//bpm"] * 4,
            "numeric_value": [130.0, 80.0, 90.0, 120.0],
            "hadm_id": [10, 11, None, 20],
        }
    )
    labels, masks = build_visit_concept_label_dicts(events, concepts)
    assert set(labels.keys()) == {(1, 10), (1, 11), (2, 20)}  # no solo entry
    assert labels[(1, 10)].tolist() == [1.0]
    assert labels[(1, 11)].tolist() == [0.0]
    assert labels[(2, 20)].tolist() == [1.0]
    assert masks[(1, 10)].tolist() == [1.0]
    assert labels[(1, 10)].shape == (len(concepts),)


def test_family_loss_weights_follow_family_shares() -> None:
    events = pl.DataFrame(
        {
            "code": ["LAB//1//"] * 850
            + ["MEDICATION//x"] * 120
            + ["DIAGNOSIS//ICD//10//I50"] * 20
            + ["PROCEDURE//ICD//10//0BH"] * 5
            + ["DRG//1"] * 5
        }
    )
    w = family_loss_weights(events, alpha=1.0, cap=100.0)
    # families: 1 diagnosis, 2 medication, 3 procedure, 4 lab, 7 billing
    lab, med, diag, proc, billing = w[4], w[2], w[1], w[3], w[7]
    assert lab < med < diag < proc == billing
    # inverse-frequency exactly, normalized so sum(share * w) == 1
    shares = torch.tensor([0.02, 0.12, 0.005, 0.85, 0.005])
    assert torch.allclose(
        (shares * torch.stack([diag, med, proc, lab, billing])).sum(),
        torch.tensor(1.0),
        atol=1e-5,
    )
    uniform = family_loss_weights(events, alpha=0.0)
    assert torch.allclose(uniform, torch.ones_like(uniform))
    # families absent from the events (here 5, 6, 8) still get an entry,
    # with the neutral weight 1, when n_families asks for them
    padded = family_loss_weights(events, alpha=1.0, n_families=9)
    assert padded.shape == (9,)
    assert padded[8].item() == 1.0
