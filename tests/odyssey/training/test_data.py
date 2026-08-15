"""Tests for the MEDS shard -> training-ready data pipeline."""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest

from odyssey.data.concepts import ConceptDefinition, ConceptRule
from odyssey.data.vocabulary import PAD_ID, UNK_ID
from odyssey.training.data import (
    _shuffle_buffered,
    build_concept_label_dicts,
    build_vocabulary,
    count_subjects,
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

    assert set(out.columns) == {"subject_id", "code", "time", "numeric_value", "hadm_id"}


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
