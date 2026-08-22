"""The MEDS conformance validator: the gate at the pipeline's narrow waist."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import pytest

from odyssey.data.meds_validation import (
    Finding,
    raise_on_errors,
    validate_meds_dataset,
)


def _events(
    subject_ids: list[int], *, subject_dtype: pl.DataType = pl.Int64
) -> pl.DataFrame:
    rows = [
        (sid, datetime(2024, 1, 1, hour), f"LAB//{hour}", float(hour))
        for sid in subject_ids
        for hour in range(3)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": subject_dtype,
            "time": pl.Datetime("us"),
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
        },
        orient="row",
    )


def _write_dataset(
    root: Path,
    *,
    with_metadata: bool = True,
    train_subjects: Optional[list[int]] = None,
    held_out_subjects: Optional[list[int]] = None,
) -> None:
    train_subjects = train_subjects if train_subjects is not None else [1, 2]
    held_out_subjects = held_out_subjects if held_out_subjects is not None else [3]
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "held_out").mkdir(parents=True)
    _events(train_subjects).write_parquet(root / "data" / "train" / "0.parquet")
    _events(held_out_subjects).write_parquet(root / "data" / "held_out" / "0.parquet")
    if with_metadata:
        metadata = root / "metadata"
        metadata.mkdir()
        (metadata / "dataset.json").write_text(json.dumps({"dataset_name": "test"}))
        pl.DataFrame({"code": ["LAB//0"]}).write_parquet(metadata / "codes.parquet")
        pl.DataFrame(
            {
                "subject_id": train_subjects + held_out_subjects,
                "split": ["train"] * len(train_subjects)
                + ["held_out"] * len(held_out_subjects),
            }
        ).write_parquet(metadata / "subject_splits.parquet")


def _codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings}


def test_conformant_dataset_passes_clean(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    assert validate_meds_dataset(tmp_path, deep=True) == []


def test_string_subject_id_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([9]).with_columns(
        pl.col("subject_id").cast(pl.Utf8).str.replace("9", "hash-9")
    ).write_parquet(tmp_path / "data" / "train" / "1.parquet")
    findings = validate_meds_dataset(tmp_path)
    assert "subject-id-dtype" in _codes(findings)
    with pytest.raises(ValueError, match="subject-id-dtype"):
        raise_on_errors(findings)


def test_missing_metadata_and_splits_reported(tmp_path: Path) -> None:
    _write_dataset(tmp_path, with_metadata=False)
    codes = _codes(validate_meds_dataset(tmp_path))
    assert "missing-metadata" in codes


def test_missing_required_column_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([4]).drop("numeric_value").write_parquet(
        tmp_path / "data" / "train" / "1.parquet"
    )
    assert "missing-column" in _codes(validate_meds_dataset(tmp_path))


def test_float64_numeric_value_is_a_warning_not_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([4]).with_columns(pl.col("numeric_value").cast(pl.Float64)).write_parquet(
        tmp_path / "data" / "train" / "1.parquet"
    )
    findings = validate_meds_dataset(tmp_path)
    assert "numeric-value-width" in _codes(findings)
    raise_on_errors(findings)  # warnings alone must not raise


def test_deep_catches_unsorted_shard(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([5]).reverse().write_parquet(tmp_path / "data" / "train" / "1.parquet")
    assert "unsorted-shard" not in _codes(validate_meds_dataset(tmp_path))
    assert "unsorted-shard" in _codes(validate_meds_dataset(tmp_path, deep=True))


def test_deep_catches_subject_spanning_shards(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([1]).write_parquet(tmp_path / "data" / "train" / "1.parquet")
    assert "subject-spans-shards" in _codes(validate_meds_dataset(tmp_path, deep=True))


def test_deep_catches_undeclared_split_membership(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([99]).write_parquet(tmp_path / "data" / "held_out" / "1.parquet")
    findings = validate_meds_dataset(tmp_path, deep=True)
    assert "split-membership" in _codes(findings)


def test_missing_data_dir_short_circuits(tmp_path: Path) -> None:
    assert _codes(validate_meds_dataset(tmp_path)) == {"missing-data-dir"}


# ---------------------------------------------------------------------------
# _check_shard_schema: every dtype/readability guard, one at a time
# ---------------------------------------------------------------------------


def test_unreadable_shard_is_an_error_not_a_crash(tmp_path: Path) -> None:
    """A corrupt/truncated parquet file must produce a finding, never raise."""
    _write_dataset(tmp_path)
    (tmp_path / "data" / "train" / "1.parquet").write_bytes(b"not a parquet file")
    findings = validate_meds_dataset(tmp_path)
    assert "unreadable-shard" in _codes(findings)
    with pytest.raises(ValueError, match="unreadable-shard"):
        raise_on_errors(findings)


def test_wrong_time_dtype_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _events([4]).with_columns(pl.col("time").cast(pl.Utf8)).write_parquet(
        tmp_path / "data" / "train" / "1.parquet"
    )
    assert "time-dtype" in _codes(validate_meds_dataset(tmp_path))


def test_wrong_code_dtype_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    bad = _events([4]).with_columns(pl.Series("code", [0, 1, 2], dtype=pl.Int64))
    bad.write_parquet(tmp_path / "data" / "train" / "1.parquet")
    assert "code-dtype" in _codes(validate_meds_dataset(tmp_path))


def test_non_float_numeric_value_dtype_is_an_error(tmp_path: Path) -> None:
    """A numeric_value that isn't even a float type (not the Float64-width warning)."""
    _write_dataset(tmp_path)
    _events([4]).with_columns(pl.col("numeric_value").cast(pl.Utf8)).write_parquet(
        tmp_path / "data" / "train" / "1.parquet"
    )
    findings = validate_meds_dataset(tmp_path)
    assert "numeric-value-dtype" in _codes(findings)
    assert "numeric-value-width" not in _codes(findings)


# ---------------------------------------------------------------------------
# _check_metadata: dataset.json / codes.parquet / subject_splits.parquet
# ---------------------------------------------------------------------------


def test_missing_dataset_json_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (tmp_path / "metadata" / "dataset.json").unlink()
    assert "missing-dataset-json" in _codes(validate_meds_dataset(tmp_path))


def test_invalid_dataset_json_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (tmp_path / "metadata" / "dataset.json").write_text("{not valid json")
    assert "invalid-dataset-json" in _codes(validate_meds_dataset(tmp_path))


def test_missing_codes_parquet_is_a_warning(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (tmp_path / "metadata" / "codes.parquet").unlink()
    findings = validate_meds_dataset(tmp_path)
    assert "missing-codes-parquet" in _codes(findings)
    raise_on_errors(findings)  # a warning alone must not raise


def test_codes_parquet_without_code_column_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    pl.DataFrame({"other": [1]}).write_parquet(tmp_path / "metadata" / "codes.parquet")
    assert "codes-parquet-shape" in _codes(validate_meds_dataset(tmp_path))


def test_missing_subject_splits_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (tmp_path / "metadata" / "subject_splits.parquet").unlink()
    assert "missing-subject-splits" in _codes(validate_meds_dataset(tmp_path))


def test_subject_splits_missing_a_required_column_is_an_error(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(  # no "split" column
        tmp_path / "metadata" / "subject_splits.parquet"
    )
    assert "subject-splits-shape" in _codes(validate_meds_dataset(tmp_path))


# ---------------------------------------------------------------------------
# deep=True: empty shards and the split-membership short-circuits
# ---------------------------------------------------------------------------


def test_deep_check_shard_skips_an_empty_shard_without_crashing(tmp_path: Path) -> None:
    """A zero-row shard has nothing to sort-check; must not raise on frame.height==0."""
    _write_dataset(tmp_path)
    _events([]).write_parquet(tmp_path / "data" / "train" / "1.parquet")
    findings = validate_meds_dataset(tmp_path, deep=True)
    assert "unsorted-shard" not in _codes(findings)
    assert "subject-spans-shards" not in _codes(findings)


def test_deep_split_membership_skips_when_subject_splits_already_missing(
    tmp_path: Path,
) -> None:
    """Already reported by _check_metadata -- must not also crash trying to read it."""
    _write_dataset(tmp_path)
    (tmp_path / "metadata" / "subject_splits.parquet").unlink()
    findings = validate_meds_dataset(tmp_path, deep=True)
    assert "missing-subject-splits" in _codes(findings)
    assert "split-membership" not in _codes(findings)


def test_deep_split_membership_skips_when_subject_splits_lacks_required_columns(
    tmp_path: Path,
) -> None:
    """Already reported by _check_metadata's shape check -- same short-circuit."""
    _write_dataset(tmp_path)
    pl.DataFrame({"subject_id": [1, 2, 3]}).write_parquet(
        tmp_path / "metadata" / "subject_splits.parquet"
    )
    findings = validate_meds_dataset(tmp_path, deep=True)
    assert "subject-splits-shape" in _codes(findings)
    assert "split-membership" not in _codes(findings)


# ---------------------------------------------------------------------------
# validate_meds_dataset: the data/ directory's own structural guards
# ---------------------------------------------------------------------------


def test_no_split_subdirectories_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    assert "no-split-dirs" in _codes(validate_meds_dataset(tmp_path))


def test_unconventional_split_name_is_a_warning(tmp_path: Path) -> None:
    (tmp_path / "data" / "custom_split").mkdir(parents=True)
    _events([1]).write_parquet(tmp_path / "data" / "custom_split" / "0.parquet")
    findings = validate_meds_dataset(tmp_path)
    matches = [f for f in findings if f.code == "unconventional-splits"]
    assert matches and matches[0].severity == "warning"


def test_split_dir_with_no_shards_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "data" / "train").mkdir(parents=True)
    assert "no-shards" in _codes(validate_meds_dataset(tmp_path))
