"""The numeric value channel: binner statistics -> numeric_z -> sequences -> chunks."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch

from odyssey.data.sequences import build_patient_sequence, collate_patient_sequences
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import VALUE_Z_COL, QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary


T0 = datetime(2024, 1, 1)


def _events() -> pl.DataFrame:
    rows = []
    # creatinine 0.5..2.9 (curated clinical range) and an uncurated lab 10..40
    for i in range(120):
        rows.append(
            (
                1,
                "LAB//RESULT//50912//mg/dL",
                T0 + timedelta(hours=i),
                0.5 + 0.02 * i,
                10,
            )
        )
        rows.append(
            (1, "LAB//RESULT//99999//x", T0 + timedelta(hours=i), 10.0 + 0.25 * i, 10)
        )
    rows.append((1, "DIAGNOSIS//ICD//10//I50", T0 + timedelta(hours=200), None, 10))
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


def test_binner_fits_value_stats_and_standardizes() -> None:
    events = _events()
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    assert set(binner.value_stats) == {
        "LAB//RESULT//50912//mg/dL",
        "LAB//RESULT//99999//x",
    }
    center, scale = binner.value_stats["LAB//RESULT//50912//mg/dL"]
    assert (
        center
        == events.filter(pl.col("code") == "LAB//RESULT//50912//mg/dL")[
            "numeric_value"
        ].median()
    )
    assert scale > 0
    z = binner.standardize(events)
    assert z.dtype == pl.Float32
    assert z.null_count() == 1  # the diagnosis row
    assert abs(float(z[0]) - (0.5 - center) / scale) < 1e-5
    assert z.drop_nulls().min() >= -5.0 and z.drop_nulls().max() <= 5.0


def test_add_value_tokens_adds_numeric_z_next_to_bin_tokens() -> None:
    events = _events()
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    binned = add_value_tokens(events, binner, source="mimic_iv")
    assert VALUE_Z_COL in binned.columns
    creat = binned.filter(pl.col("code").str.starts_with("LAB//RESULT//50912"))
    # clinical bin on the token, continuous value alongside
    assert creat["code"][0].endswith("::NORMAL")
    assert creat[VALUE_Z_COL].null_count() == 0
    assert (
        binned.filter(pl.col("code").str.starts_with("DIAGNOSIS"))[
            VALUE_Z_COL
        ].null_count()
        == 1
    )
    # a binner without stats (older runs) adds no column
    old = QuantileBinner(boundaries=binner.boundaries, n_bins=5)
    assert VALUE_Z_COL not in add_value_tokens(events, old, source="mimic_iv").columns


def test_binner_json_roundtrip_and_legacy_files(tmp_path: Path) -> None:
    events = _events()
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    binner.save(tmp_path / "b.json")
    loaded = QuantileBinner.load(tmp_path / "b.json")
    assert loaded.value_stats == binner.value_stats
    (tmp_path / "legacy.json").write_text(json.dumps({"n_bins": 5, "boundaries": {}}))
    assert QuantileBinner.load(tmp_path / "legacy.json").value_stats == {}


def test_values_flow_into_sequences_collate_and_chunks() -> None:
    events = _events()
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    binned = add_value_tokens(events, binner, source="mimic_iv")
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    seq = build_patient_sequence(binned, vocab)
    assert len(seq.values) == len(seq)
    assert math.isnan(seq.values[-1])  # the diagnosis
    assert not math.isnan(seq.values[0])
    batch = collate_patient_sequences([seq])
    assert (
        batch.aux.values is not None
        and batch.aux.values.shape == batch.concept_ids.shape
    )
    assert torch.isnan(batch.aux.values[0, -1])
    chunks = list(
        PackedLaneSampler(iter([seq]), num_lanes=1, chunk_size=64, reset_prob=0.0)
    )
    assert chunks
    first = chunks[0].batch.aux.values
    assert first is not None and first.shape == chunks[0].batch.concept_ids.shape
    assert torch.isclose(first[0, 0], torch.tensor(seq.values[0]))
    # padding positions carry NaN, real positions carry the sequence's values
    last = chunks[-1]
    assert torch.isnan(last.batch.aux.values[0][~(last.subject_ids >= 0)[0]]).all()


def test_sequences_without_numeric_z_have_no_values() -> None:
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab)
    assert seq.values == []
    chunks = list(
        PackedLaneSampler(iter([seq]), num_lanes=1, chunk_size=64, reset_prob=0.0)
    )
    assert torch.isnan(chunks[0].batch.aux.values).all()
