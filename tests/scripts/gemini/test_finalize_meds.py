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

import pandas as pd
import pytest

from odyssey.data.meds_validation import validate_meds_dataset


_SKIP_REASON = "gemini extra not installed (uv sync --extra gemini)"
pytest.importorskip("sqlalchemy", reason=_SKIP_REASON)
pytest.importorskip("pandas", reason=_SKIP_REASON)
pytest.importorskip("pyarrow", reason=_SKIP_REASON)
pytest.importorskip("polars", reason=_SKIP_REASON)

import polars as pl  # noqa: E402


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


def test_repartition_pass_buffers_and_flushes_across_input_shards(
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
        flush_threshold=1_000_000,  # never triggers mid-pass -- flushed at end
    )

    assert set(paths) == {("train", 0), ("tuning", 0)}
    train_frame = pl.read_parquet(paths[("train", 0)])
    assert train_frame["subject_id"].to_list() == [111]
    assert train_frame["subject_id"].dtype == pl.Int64


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

    n_rows, code_counts = mod._sort_and_finalize_shard(tmp_path_file, final_path)

    assert n_rows == 3
    result = pl.read_parquet(final_path)
    # subject 1's null-time row sorts before its real-time row; subject 2 last.
    assert result["subject_id"].to_list() == [1, 1, 2]
    assert result["code"].to_list() == ["C1a", "C1b", "C2"]
    assert code_counts == {"C1a": 1, "C1b": 1, "C2": 1}


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
