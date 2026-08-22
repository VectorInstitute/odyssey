"""Tests for scripts/gemini/finalize_meds.py.

No real database beyond ``odyssey.data.gemini.db.query`` (monkeypatched) is
used, and no real filesystem beyond ``tmp_path``. Extraction-shaped fixture
output is built directly with ``extract_meds.MedsShardWriter`` rather than
hand-rolled, so these tests exercise the real flat-layout shape finalize
actually has to read.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)
pytest.importorskip("pyarrow", reason=_SKIP_REASON)
pytest.importorskip("polars", reason=_SKIP_REASON)

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from odyssey.data.meds_validation import validate_meds_dataset  # noqa: E402


def _load_module(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "gemini" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_hospital_query(
    monkeypatch: pytest.MonkeyPatch, mod: ModuleType, hadm_ids: list[int]
) -> None:
    monkeypatch.setattr(
        mod,
        "_fetch_hadm_id_hospital",
        lambda: pd.DataFrame(
            {"genc_id": hadm_ids, "hospital_num": [1] * len(hadm_ids)}
        ),
    )


def _write_fixture_extraction(tmp_path: Path, n_subjects: int) -> None:
    """Build a small, real extract_meds.py-shaped flat layout under tmp_path."""
    extract_meds = _load_module("extract_meds")
    shard_by_subject = extract_meds.assign_shards(
        [f"subj-{i}" for i in range(n_subjects)], subjects_per_shard=1000
    )
    writer = extract_meds.MedsShardWriter(tmp_path, shard_by_subject)
    rows = []
    for i in range(n_subjects):
        subject = f"subj-{i}"
        for j in range(2):
            rows.append(
                {
                    "subject_id": subject,
                    "time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=j),
                    "code": f"CODE{j}",
                    "numeric_value": None,
                    "hadm_id": i * 10 + j,
                }
            )
    writer.write_batch("admdad_subset", pd.DataFrame(rows))
    writer.close()
    (tmp_path / "extract_manifest.json").write_text(
        json.dumps({"admdad_subset": "complete"})
    )


# --- _input_shard_files / _dataset_size_bytes ------------------------------


def test_input_shard_files_groups_base_and_part_files(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    (tmp_path / "shard_0000.parquet").write_bytes(b"a")
    (tmp_path / "shard_0000_part1.parquet").write_bytes(b"bb")
    (tmp_path / "shard_0001.parquet").write_bytes(b"ccc")
    (tmp_path / "not_a_shard.parquet").write_bytes(b"ignored")

    groups = mod._input_shard_files(tmp_path)

    assert set(groups) == {0, 1}
    assert {p.name for p in groups[0]} == {
        "shard_0000.parquet",
        "shard_0000_part1.parquet",
    }
    assert {p.name for p in groups[1]} == {"shard_0001.parquet"}
    assert mod._dataset_size_bytes(groups) == 1 + 2 + 3


# --- preflight checks -------------------------------------------------------


def test_preflight_disk_raises_with_exact_byte_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module("finalize_meds")
    (tmp_path / "shard_0000.parquet").write_bytes(b"x" * 1000)
    shard_files = mod._input_shard_files(tmp_path)

    class _FakeUsage:
        free = 10  # far less than the 1000-byte dataset

    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _p: _FakeUsage())

    with pytest.raises(RuntimeError, match="GB free"):
        mod._preflight_disk(tmp_path, shard_files)


def test_preflight_disk_passes_when_space_is_sufficient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module("finalize_meds")
    (tmp_path / "shard_0000.parquet").write_bytes(b"x" * 1000)
    shard_files = mod._input_shard_files(tmp_path)

    class _FakeUsage:
        free = 10**12

    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _p: _FakeUsage())

    mod._preflight_disk(tmp_path, shard_files)  # must not raise


def test_preflight_nofile_raises_with_the_exact_ulimit_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module("finalize_meds")
    monkeypatch.setattr(mod.resource, "getrlimit", lambda _which: (256, 256))
    monkeypatch.setattr(mod.resource, "setrlimit", lambda _which, _limits: None)

    with pytest.raises(RuntimeError, match=r"ulimit -n \d+"):
        mod._preflight_nofile(1_000_000)


def test_preflight_nofile_passes_within_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module("finalize_meds")
    monkeypatch.setattr(mod.resource, "getrlimit", lambda _which: (10_000, 10_000))
    mod._preflight_nofile(100)  # must not raise


# --- crash semantics ---------------------------------------------------


def test_wipe_partial_output_is_a_noop_when_nothing_exists(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    mod._wipe_partial_output(tmp_path)  # must not raise


def test_wipe_partial_output_deletes_a_dead_attempt(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    (tmp_path / "data" / "train").mkdir(parents=True)
    (tmp_path / "data" / "train" / "shard_0000.parquet").write_bytes(b"stale")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "dataset.json").write_text("{}")

    mod._wipe_partial_output(tmp_path)

    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "metadata").exists()


def test_wipe_partial_output_refuses_when_sentinel_present(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / ".finalize_complete").write_text("2020-01-01")

    with pytest.raises(RuntimeError, match="already has a completed finalize"):
        mod._wipe_partial_output(tmp_path)


# --- subject universe / id mapping -----------------------------------------


def test_collect_subject_universe_unions_across_shards(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    pl.DataFrame({"subject_id": ["a", "b"]}).write_parquet(
        tmp_path / "shard_0000.parquet"
    )
    pl.DataFrame({"subject_id": ["b", "c"]}).write_parquet(
        tmp_path / "shard_0000_part1.parquet"
    )
    pl.DataFrame({"subject_id": ["d"]}).write_parquet(tmp_path / "shard_0001.parquet")
    shard_files = mod._input_shard_files(tmp_path)

    subjects = mod._collect_subject_universe(shard_files)

    assert subjects == ["a", "b", "c", "d"]


def test_build_subject_id_mapping_is_deterministic_and_nonnegative() -> None:
    mod = _load_module("finalize_meds")
    subjects = [f"patient-{i}" for i in range(500)]

    first = mod._build_subject_id_mapping(subjects)
    second = mod._build_subject_id_mapping(subjects)

    assert first == second
    assert set(first) == set(subjects)
    assert all(v >= 0 for v in first.values())
    assert len(set(first.values())) == len(subjects)  # no collisions at this scale


def test_build_subject_id_mapping_raises_loudly_on_a_real_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module("finalize_meds")

    class _FakeDigest:
        def hexdigest(self) -> str:
            return "0" * 64  # every subject collides

    monkeypatch.setattr(mod.hashlib, "sha256", lambda _b: _FakeDigest())

    with pytest.raises(RuntimeError, match="collision"):
        mod._build_subject_id_mapping(["subj-a", "subj-b"])


# --- split assignment --------------------------------------------------


def test_assign_splits_covers_every_subject_exactly_once_and_is_deterministic() -> None:
    mod = _load_module("finalize_meds")
    subject_ids = list(range(1000))

    first = mod._assign_splits(subject_ids, seed=0)
    second = mod._assign_splits(subject_ids, seed=0)

    assert first == second
    assert set(first) == set(subject_ids)
    assert set(first.values()) <= {"train", "tuning", "held_out"}


def test_assign_splits_matches_the_meds_extract_default_proportions() -> None:
    mod = _load_module("finalize_meds")
    subject_ids = list(range(10_000))

    assignment = mod._assign_splits(subject_ids, seed=0)

    counts = dict.fromkeys(mod.FINALIZE_SPLIT_FRACS, 0)
    for split in assignment.values():
        counts[split] += 1
    assert counts["train"] == 8000
    assert counts["tuning"] == 1000
    assert counts["held_out"] == 1000


def test_assign_splits_different_seed_gives_a_different_assignment() -> None:
    mod = _load_module("finalize_meds")
    subject_ids = list(range(1000))

    seed_0 = mod._assign_splits(subject_ids, seed=0)
    seed_1 = mod._assign_splits(subject_ids, seed=1)

    assert seed_0 != seed_1


def test_assign_output_shards_is_deterministic_and_per_split() -> None:
    mod = _load_module("finalize_meds")
    split_by_subject = {i: ("train" if i < 800 else "held_out") for i in range(1000)}

    shard_by_subject, n_shards_by_split = mod._assign_output_shards(
        split_by_subject, subjects_per_shard=100
    )

    assert n_shards_by_split == {"train": 8, "held_out": 2}
    assert set(shard_by_subject) == set(split_by_subject)
    for subject_id, split in split_by_subject.items():
        assert 0 <= shard_by_subject[subject_id] < n_shards_by_split[split]
    # deterministic
    again, _ = mod._assign_output_shards(split_by_subject, subjects_per_shard=100)
    assert again == shard_by_subject


# --- hospital lookup ------------------------------------------------------


def test_fetch_hadm_id_hospital_queries_admdad_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module("finalize_meds")
    captured = {}

    def fake_query(sql: str, params: object = None) -> pd.DataFrame:
        captured["sql"] = sql
        return pd.DataFrame({"genc_id": [1, 2], "hospital_num": [10, 20]})

    monkeypatch.setattr(mod.db, "query", fake_query)

    result = mod._fetch_hadm_id_hospital()

    assert "admdad_subset" in captured["sql"]
    assert result["hospital_num"].tolist() == [10, 20]


# --- repartition + sort -----------------------------------------------


def test_repartition_pass_partitions_rows_by_split_and_out_shard(
    tmp_path: Path,
) -> None:
    mod = _load_module("finalize_meds")
    frame = pl.DataFrame(
        {
            "subject_id": ["a", "b"],
            "time": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01")],
            "code": ["X", "Y"],
            "numeric_value": [1.0, None],
            "hadm_id": [1, 2],
        }
    )
    frame.write_parquet(tmp_path / "shard_0000.parquet")
    shard_files = mod._input_shard_files(tmp_path)
    subject_id_mapping = {"a": 111, "b": 222}
    split_by_subject = {111: "train", 222: "tuning"}
    output_shard_by_subject = {111: 0, 222: 0}
    tmp_dir = tmp_path / "repartition_tmp"
    tmp_dir.mkdir()

    paths = mod._repartition_pass(
        shard_files,
        subject_id_mapping,
        output_shard_by_subject,
        split_by_subject,
        tmp_dir,
    )

    assert set(paths) == {("train", 0), ("tuning", 0)}
    train_frame = pl.concat([pl.read_parquet(p) for p in paths[("train", 0)]])
    assert train_frame["subject_id"].to_list() == [111]
    assert train_frame["subject_id"].dtype == pl.Int64
    assert list(train_frame.columns) == mod.MEDS_COLUMNS  # __split/__out_shard dropped


def test_repartition_pass_raises_loudly_on_an_unmapped_subject(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    frame = pl.DataFrame(
        {
            "subject_id": ["unknown"],
            "time": [pd.Timestamp("2020-01-01")],
            "code": ["X"],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )
    frame.write_parquet(tmp_path / "shard_0000.parquet")
    shard_files = mod._input_shard_files(tmp_path)
    tmp_dir = tmp_path / "repartition_tmp"
    tmp_dir.mkdir()

    with pytest.raises(RuntimeError, match="not in the collected universe"):
        mod._repartition_pass(shard_files, {}, {}, {}, tmp_dir)


def test_repartition_pass_error_names_the_actual_raw_subject_id(
    tmp_path: Path,
) -> None:
    """The error message must name the real offending raw id, not a null.

    Regression guard for a real bug hit while building this: by the point
    the null-check runs, "subject_id" has already been overwritten with
    the (null, for an unmapped subject) remapped value -- the error path
    has to read the example from a column that still holds the pre-remap
    raw value, not re-read "subject_id" itself.
    """
    mod = _load_module("finalize_meds")
    frame = pl.DataFrame(
        {
            "subject_id": ["totally-unmapped-subject"],
            "time": [pd.Timestamp("2020-01-01")],
            "code": ["X"],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )
    frame.write_parquet(tmp_path / "shard_0000.parquet")
    shard_files = mod._input_shard_files(tmp_path)
    tmp_dir = tmp_path / "repartition_tmp"
    tmp_dir.mkdir()

    with pytest.raises(RuntimeError, match="totally-unmapped-subject"):
        mod._repartition_pass(shard_files, {}, {}, {}, tmp_dir)


def test_sort_and_finalize_shard_sorts_by_subject_then_time_nulls_first(
    tmp_path: Path,
) -> None:
    mod = _load_module("finalize_meds")
    unsorted = pl.DataFrame(
        {
            "subject_id": [2, 1, 1],
            "time": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), None],
            "code": ["C2", "C1b", "C1a"],
            "numeric_value": [None, None, None],
            "hadm_id": [1, 2, 3],
        }
    )
    tmp_path_file = tmp_path / "unsorted.parquet"
    unsorted.write_parquet(tmp_path_file)
    final_path = tmp_path / "data" / "train" / "shard_0000.parquet"

    n_rows, code_counts = mod._sort_and_finalize_shard([tmp_path_file], final_path)

    assert n_rows == 3
    result = pl.read_parquet(final_path)
    # subject 1's null-time row sorts before its real-time row; subject 2 last.
    assert result["subject_id"].to_list() == [1, 1, 2]
    assert result["code"].to_list() == ["C1a", "C1b", "C2"]
    assert code_counts == {"C1a": 1, "C1b": 1, "C2": 1}


def test_sort_and_finalize_shard_concatenates_multiple_input_files(
    tmp_path: Path,
) -> None:
    """_sort_and_finalize_shard must merge every part file for a destination.

    The sink can in principle split one destination across >1 physical
    file (see _repartition_pass's docstring) -- _sort_and_finalize_shard
    must read and merge every one of them, not just the first.
    """
    mod = _load_module("finalize_meds")
    part0 = pl.DataFrame(
        {
            "subject_id": [2],
            "time": [pd.Timestamp("2020-01-01")],
            "code": ["C2"],
            "numeric_value": [None],
            "hadm_id": [1],
        }
    )
    part1 = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [pd.Timestamp("2020-01-02")],
            "code": ["C1"],
            "numeric_value": [None],
            "hadm_id": [2],
        }
    )
    path0 = tmp_path / "train__0000_000.parquet"
    path1 = tmp_path / "train__0000_001.parquet"
    part0.write_parquet(path0)
    part1.write_parquet(path1)
    final_path = tmp_path / "data" / "train" / "shard_0000.parquet"

    n_rows, code_counts = mod._sort_and_finalize_shard([path0, path1], final_path)

    assert n_rows == 2
    result = pl.read_parquet(final_path)
    assert result["subject_id"].to_list() == [1, 2]
    assert code_counts == {"C1": 1, "C2": 1}


# --- the lazy translation's own focused test --------------------------------


def test_lazy_replace_strict_matches_eager_replace_strict_for_the_same_lookups() -> (
    None
):
    """Focused test for the one real translation risk in the rewrite.

    The prior implementation built subject_id/split/out_shard as three
    plain eager pl.Series via Series.replace_strict, all derived from one
    original Series; _repartition_pass now builds the same three via
    Expr.replace_strict chained from one pl.col("subject_id") expression,
    inside a LazyFrame. This isolates just that translation -- eager
    Series-level replace_strict vs. the identical lookups expressed as
    lazy expressions -- independent of the sink/partitioning machinery
    the broader A/B test below also covers.
    """
    subject_id_mapping = {"a": 111, "b": 222, "c": 333}
    split_by_subject = {111: "train", 222: "train", 333: "tuning"}
    output_shard_by_subject = {111: 0, 222: 1, 333: 0}
    raw_ids = ["a", "b", "c", "a", "unmapped"]

    eager_series = pl.Series("subject_id", raw_ids)
    eager_remapped = eager_series.replace_strict(
        subject_id_mapping, default=None, return_dtype=pl.Int64
    )
    eager_split = eager_remapped.replace_strict(
        split_by_subject, default=None, return_dtype=pl.Utf8
    )
    eager_out_shard = eager_remapped.replace_strict(
        output_shard_by_subject, default=None, return_dtype=pl.Int64
    )

    lazy_remapped_expr = pl.col("subject_id").replace_strict(
        subject_id_mapping, default=None, return_dtype=pl.Int64
    )
    lazy_result = (
        pl.LazyFrame({"subject_id": raw_ids})
        .with_columns(
            lazy_remapped_expr.alias("subject_id"),
            lazy_remapped_expr.replace_strict(
                split_by_subject, default=None, return_dtype=pl.Utf8
            ).alias("__split"),
            lazy_remapped_expr.replace_strict(
                output_shard_by_subject, default=None, return_dtype=pl.Int64
            ).alias("__out_shard"),
        )
        .collect()
    )

    assert lazy_result["subject_id"].to_list() == eager_remapped.to_list()
    assert lazy_result["__split"].to_list() == eager_split.to_list()
    assert lazy_result["__out_shard"].to_list() == eager_out_shard.to_list()
    # The unmapped "unmapped" entry must come through as null on all three,
    # not silently coerced or dropped -- same contract on both sides.
    assert lazy_result["subject_id"][4] is None
    assert eager_remapped[4] is None


# --- A/B equality against a reference buffered implementation --------------


def _reference_buffered_repartition(
    shard_files: dict[int, list[Path]],
    subject_id_mapping: dict[str, int],
    output_shard_by_subject: dict[int, int],
    split_by_subject: dict[int, str],
    tmp_dir: Path,
    *,
    meds_columns: list[str],
    flush_threshold: int = 250_000,
) -> dict[tuple[str, int], Path]:
    """Run the pre-rewrite buffered-scatter algorithm.

    Preserved here (only) as the ground truth an A/B equality check runs
    against -- not imported from finalize_meds.py, which no longer has
    this code path at all.
    """
    arrow_schema = pa.schema(
        [
            ("subject_id", pa.int64()),
            ("time", pa.timestamp("us")),
            ("code", pa.string()),
            ("numeric_value", pa.float64()),
            ("hadm_id", pa.int64()),
        ]
    )
    buffers: dict[tuple[str, int], list[pl.DataFrame]] = {}
    buffer_counts: dict[tuple[str, int], int] = {}
    writers: dict[tuple[str, int], "pq.ParquetWriter"] = {}
    paths: dict[tuple[str, int], Path] = {}

    def flush(key: tuple[str, int]) -> None:
        frames = buffers.pop(key, None)
        buffer_counts[key] = 0
        if not frames:
            return
        combined = frames[0] if len(frames) == 1 else pl.concat(frames)
        if key not in writers:
            split_name, out_idx = key
            path = tmp_dir / f"ref__{split_name}__{out_idx:04d}.parquet"
            writers[key] = pq.ParquetWriter(str(path), arrow_schema)
            paths[key] = path
        arrow_table = pa.Table.from_pandas(
            combined.select(meds_columns).to_pandas(),
            schema=arrow_schema,
            preserve_index=False,
        )
        writers[key].write_table(arrow_table)

    for shard_index in sorted(shard_files):
        files = shard_files[shard_index]
        frame = (
            pl.concat([pl.read_parquet(p) for p in files])
            if len(files) > 1
            else pl.read_parquet(files[0])
        )
        if frame.height == 0:
            continue
        subject_int = frame["subject_id"].replace_strict(
            subject_id_mapping, default=None, return_dtype=pl.Int64
        )
        split = subject_int.replace_strict(
            split_by_subject, default=None, return_dtype=pl.Utf8
        )
        out_shard = subject_int.replace_strict(
            output_shard_by_subject, default=None, return_dtype=pl.Int64
        )
        frame = frame.with_columns(
            subject_int.alias("subject_id"),
            split.alias("__split"),
            out_shard.alias("__out_shard"),
        )
        for (split_name, out_idx), group in frame.group_by(["__split", "__out_shard"]):
            key = (str(split_name), int(out_idx))
            buffers.setdefault(key, []).append(group)
            buffer_counts[key] = buffer_counts.get(key, 0) + group.height
            if buffer_counts[key] >= flush_threshold:
                flush(key)

    for key in list(buffers):
        flush(key)
    for writer in writers.values():
        writer.close()
    return paths


def test_repartition_pass_matches_a_reference_buffered_implementation(
    tmp_path: Path,
) -> None:
    """Graduated from the scratch benchmark's exhaustive A/B check.

    The new native-sink _repartition_pass and the old buffered algorithm
    must produce byte-identical row sets for every destination, not just
    matching row counts. Small scale here (a real test, not a benchmark)
    but the same exhaustive-not-sampled comparison.
    """
    mod = _load_module("finalize_meds")
    extraction_dir = tmp_path / "extraction"
    extraction_dir.mkdir()
    # _write_fixture_extraction's own subjects_per_shard=1000 (extract_meds'
    # convention, not finalize's) means >1000 subjects is what it takes to
    # get more than one *input* file here -- separate from output-shard
    # sizing below, which uses finalize's own subjects_per_shard.
    _write_fixture_extraction(extraction_dir, n_subjects=2500)
    shard_files = mod._input_shard_files(extraction_dir)
    assert len(shard_files) > 1  # exercise the multi-input-file scan for real

    subjects = mod._collect_subject_universe(shard_files)
    subject_id_mapping = mod._build_subject_id_mapping(subjects)
    split_by_subject = mod._assign_splits(list(subject_id_mapping.values()))
    output_shard_by_subject, n_shards_by_split = mod._assign_output_shards(
        split_by_subject, subjects_per_shard=20
    )
    assert sum(n_shards_by_split.values()) > 1  # exercise multiple destinations

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()

    new_paths = mod._repartition_pass(
        shard_files,
        subject_id_mapping,
        output_shard_by_subject,
        split_by_subject,
        new_dir,
    )
    ref_paths = _reference_buffered_repartition(
        shard_files,
        subject_id_mapping,
        output_shard_by_subject,
        split_by_subject,
        ref_dir,
        meds_columns=mod.MEDS_COLUMNS,
    )

    assert set(new_paths) == set(ref_paths), (
        f"destination sets differ: new-only={set(new_paths) - set(ref_paths)}, "
        f"ref-only={set(ref_paths) - set(new_paths)}"
    )
    for key in sorted(new_paths):
        new_frame = pl.concat([pl.read_parquet(p) for p in new_paths[key]]).select(
            mod.MEDS_COLUMNS
        )
        ref_frame = pl.read_parquet(ref_paths[key]).select(mod.MEDS_COLUMNS)
        assert new_frame.height == ref_frame.height, f"row count differs at {key}"
        assert new_frame.sort(mod.MEDS_COLUMNS).equals(
            ref_frame.sort(mod.MEDS_COLUMNS)
        ), f"row content differs at {key}"


# --- PartitionBy API-surface guard ------------------------------------------


def test_partition_by_multi_key_no_include_key_custom_provider_api_surface(
    tmp_path: Path,
) -> None:
    """Exercise the exact pl.PartitionBy surface _repartition_pass depends on.

    Multi-column key, include_key=False, file_path_provider -- directly,
    independent of finalize_meds.py itself. polars marks PartitionBy
    unstable (may change without a semver-breaking bump); this is the
    guard that's supposed to catch that on a routine `uv sync`/CI run,
    not on the GEMINI node mid-finalize. If this test starts failing after
    a polars bump, the fix belongs in _repartition_pass's own
    sink_parquet(PartitionBy(...)) call, not here.
    """
    lf = pl.LazyFrame(
        {
            "value": [1, 2, 3, 4],
            "split": ["train", "train", "tuning", "tuning"],
            "shard": [0, 1, 0, 1],
        }
    )

    def file_path_provider(args: "pl.FileProviderArgs") -> str:
        split = args.partition_keys["split"][0]
        shard = args.partition_keys["shard"][0]
        return str(
            tmp_path / f"out__{split}__{shard}_{args.index_in_partition:03d}.parquet"
        )

    lf.sink_parquet(
        pl.PartitionBy(
            str(tmp_path),
            key=["split", "shard"],
            include_key=False,
            file_path_provider=file_path_provider,
        ),
        mkdir=True,
    )
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == [
        "out__train__0_000.parquet",
        "out__train__1_000.parquet",
        "out__tuning__0_000.parquet",
        "out__tuning__1_000.parquet",
    ]
    one = pl.read_parquet(tmp_path / "out__train__0_000.parquet")
    # include_key=False: the partition columns are not in the output.
    assert list(one.columns) == ["value"]
    assert one["value"].to_list() == [1]


# --- run_finalize (integration) -----------------------------------------


def test_run_finalize_end_to_end_produces_a_conformant_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module("finalize_meds")
    _write_fixture_extraction(tmp_path, n_subjects=30)
    _fake_hospital_query(
        monkeypatch, mod, hadm_ids=[i * 10 + j for i in range(30) for j in range(2)]
    )

    summary = mod.run_finalize(output_dir=tmp_path)

    assert summary["n_subjects"] == 30
    assert sum(summary["splits"].values()) == 30
    assert (tmp_path / "metadata" / ".finalize_complete").is_file()
    assert not list(tmp_path.glob("shard_*.parquet"))  # old flat layout deleted
    assert not (tmp_path / "extract_manifest.json").exists()

    findings = validate_meds_dataset(tmp_path, deep=True)
    assert all(f.severity != "error" for f in findings)


def test_run_finalize_raises_when_extraction_is_not_complete(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    (tmp_path / "extract_manifest.json").write_text(
        json.dumps({"admdad_subset": "complete", "lab_subset": "in_progress"})
    )

    with pytest.raises(RuntimeError, match="not fully complete"):
        mod.run_finalize(output_dir=tmp_path)


def test_run_finalize_raises_when_no_manifest_exists(tmp_path: Path) -> None:
    mod = _load_module("finalize_meds")
    with pytest.raises(RuntimeError, match="not found"):
        mod.run_finalize(output_dir=tmp_path)


def test_run_finalize_refuses_to_redo_a_completed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module("finalize_meds")
    _write_fixture_extraction(tmp_path, n_subjects=10)
    _fake_hospital_query(
        monkeypatch, mod, hadm_ids=[i * 10 + j for i in range(10) for j in range(2)]
    )
    mod.run_finalize(output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="already has a completed finalize"):
        mod.run_finalize(output_dir=tmp_path)


def test_run_finalize_wipes_a_partial_attempt_before_redoing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Real scenario this guards against: a finalize run that died after
    # writing some of data/+metadata/ but before the sentinel -- a re-run
    # must not try to resume or merge with that partial state.
    mod = _load_module("finalize_meds")
    _write_fixture_extraction(tmp_path, n_subjects=10)
    (tmp_path / "data" / "train").mkdir(parents=True)
    (tmp_path / "data" / "train" / "shard_9999.parquet").write_bytes(b"stale-partial")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "dataset.json").write_text("{}")
    _fake_hospital_query(
        monkeypatch, mod, hadm_ids=[i * 10 + j for i in range(10) for j in range(2)]
    )

    summary = mod.run_finalize(output_dir=tmp_path)

    assert summary["n_subjects"] == 10
    assert not (tmp_path / "data" / "train" / "shard_9999.parquet").exists()
