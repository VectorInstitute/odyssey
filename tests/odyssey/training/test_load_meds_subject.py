"""load_meds_subject: one subject from one shard, projected and in shard order."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from odyssey.training.data import load_meds_shard, load_meds_subject


def test_returns_only_that_subject_in_shard_order(tmp_path: Path) -> None:
    t = datetime(2024, 1, 1)
    shard = pl.DataFrame(
        {
            "subject_id": [1, 2, 1, 2, 1],
            "time": [t, t, t, t, t],
            "code": ["A", "B", "C", "D", "E"],
            "numeric_value": [None, 1.0, 2.0, None, 3.0],
            "hadm_id": [10, 20, 10, 20, 10],
            "unused": ["x"] * 5,
        }
    )
    path = tmp_path / "0.parquet"
    shard.write_parquet(path)
    one = load_meds_subject(path, 1)
    assert one["code"].to_list() == ["A", "C", "E"]
    assert one.columns == load_meds_shard(path).columns
    assert "unused" not in one.columns
    assert load_meds_subject(path, 99).height == 0


def test_shard_without_hadm_id_loads_the_columns_it_has(tmp_path: Path) -> None:
    path = tmp_path / "3.parquet"
    pl.DataFrame(
        {
            "subject_id": [5, 5],
            "time": [datetime(2024, 1, 1), None],
            "code": ["A", "GENDER//M"],
            "numeric_value": [1.0, None],
        }
    ).write_parquet(path)
    one = load_meds_subject(path, 5)
    assert one.columns == ["subject_id", "time", "code", "numeric_value"]
    assert one.height == 2


def test_missing_shard_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_meds_subject(tmp_path / "nope.parquet", 1)
