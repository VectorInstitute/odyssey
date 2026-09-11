"""PatientStore: shard index, lazy per-subject loads, visits and header facts."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from apps.clinician_demo.patient_store import (
    PatientStore,
    UnknownPatientError,
    build_shard_index,
    discover_shards,
    load_splits,
)


T0 = datetime(2150, 3, 1, 8, 0)
SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _subject(sid: int, *, visits: int = 2) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = [
        (sid, None, "GENDER//F", None, None),
        (sid, T0 - timedelta(days=365.25 * 70), "MEDS_BIRTH", None, None),
    ]
    for v in range(visits):
        start = T0 + timedelta(days=30 * v)
        hadm = sid * 10 + v
        rows += [
            (
                sid,
                start,
                "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM HOSPITAL",
                None,
                hadm,
            ),
            (sid, start + timedelta(hours=1), "LAB//220045//bpm", 88.0, hadm),
            (
                sid,
                start + timedelta(hours=2),
                "MEDICATION//Norepinephrine 4mg//Administered",
                None,
                hadm,
            ),
            (sid, start + timedelta(hours=30), "HOSPITAL_DISCHARGE//HOME", None, hadm),
        ]
    return rows


def _write(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "held_out").mkdir(parents=True)
    (data / "train").mkdir()
    (data / ".meds_extract_run").mkdir()
    pl.DataFrame(
        _subject(1) + _subject(2, visits=1), schema=SCHEMA, orient="row"
    ).write_parquet(data / "held_out" / "0.parquet")
    pl.DataFrame(_subject(3), schema=SCHEMA, orient="row").write_parquet(
        data / "held_out" / "10.parquet"
    )
    pl.DataFrame(_subject(4), schema=SCHEMA, orient="row").write_parquet(
        data / "train" / "2.parquet"
    )
    # extractor working state must never be mistaken for a shard
    pl.DataFrame(_subject(9), schema=SCHEMA, orient="row").write_parquet(
        data / ".meds_extract_run" / "0.parquet"
    )
    return data


def _store(data: Path, **kwargs: object) -> PatientStore:
    return PatientStore(
        build_shard_index(discover_shards(data)),
        source="mimic_iv",
        normalize_medications=True,
        **kwargs,  # type: ignore[arg-type]
    )


def test_discovery_is_recursive_ordered_and_skips_hidden_dirs(tmp_path: Path) -> None:
    data = _write(tmp_path)
    shards = discover_shards(data)
    assert [p.relative_to(data).as_posix() for p in shards] == [
        "held_out/0.parquet",
        "held_out/10.parquet",
        "train/2.parquet",
    ]
    assert len(discover_shards(data, max_shards=1)) == 1
    with pytest.raises(FileNotFoundError):
        discover_shards(tmp_path / "empty_does_not_exist")


def test_index_maps_every_subject_to_its_shard_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    data = _write(tmp_path)
    index = build_shard_index(discover_shards(data))
    assert {sid: p.name for sid, p in index.items()} == {
        1: "0.parquet",
        2: "0.parquet",
        3: "10.parquet",
        4: "2.parquet",
    }
    dup = data / "train" / "3.parquet"
    pl.DataFrame(_subject(1), schema=SCHEMA, orient="row").write_parquet(dup)
    with pytest.raises(ValueError, match="subject 1 is in both"):
        build_shard_index(discover_shards(data))


def test_raw_events_load_one_subject_normalized_and_cached(tmp_path: Path) -> None:
    store = _store(_write(tmp_path), cache_size=1)
    events = store.raw_events(1)
    assert set(events["subject_id"].to_list()) == {1}
    meds = events.filter(pl.col("code").str.starts_with("MEDICATION"))["code"].to_list()
    assert meds and all(c.startswith("MEDICATION//norepinephrine") for c in meds), meds
    assert store.raw_events(1) is events  # cached
    store.raw_events(2)  # evicts subject 1 (cache_size=1)
    assert store.raw_events(1) is not events
    assert 1 in store and 99 not in store and len(store) == 4
    assert store.subject_ids == [1, 2, 3, 4]
    with pytest.raises(UnknownPatientError):
        store.raw_events(99)


def test_cache_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_size"):
        _store(_write(tmp_path), cache_size=0)


def test_visits_are_time_ordered_with_readable_admission_and_counts(
    tmp_path: Path,
) -> None:
    store = _store(_write(tmp_path), describe=lambda code: code.split("//")[1].title())
    visits = store.visits(1)
    assert [v.visit_id for v in visits] == [10, 11]
    assert visits[0].admission == "Urgent"
    assert visits[0].n_events == 4
    assert visits[0].start_hours == pytest.approx(0.0)
    assert visits[0].end_hours == pytest.approx(30.0)
    assert visits[1].start_hours == pytest.approx(30 * 24.0)


def test_visit_without_an_admission_code_gets_a_generic_label(tmp_path: Path) -> None:
    data = tmp_path / "d"
    data.mkdir()
    rows = [
        (5, T0, "LAB//220045//bpm", 90.0, 50),
        (5, T0 + timedelta(hours=3), "LAB//220045//bpm", 95.0, 50),
    ]
    pl.DataFrame(rows, schema=SCHEMA, orient="row").write_parquet(data / "0.parquet")
    store = _store(data)
    (visit,) = store.visits(5)
    assert visit.admission == "Admission" and visit.end_hours == pytest.approx(3.0)


def test_summary_reports_sex_age_split_and_training_flag(tmp_path: Path) -> None:
    splits = tmp_path / "splits.parquet"
    pl.DataFrame(
        {"subject_id": [1, 2, 3], "split": ["train", "tuning", "held_out"]}
    ).write_parquet(splits)
    store = _store(_write(tmp_path), splits=load_splits(splits))
    s1 = store.summary(1)
    assert s1.sex == "F" and s1.age_years == pytest.approx(70.0, abs=0.01)
    assert s1.split == "train" and s1.seen_in_training
    assert store.summary(2).seen_in_training  # tuning counts as seen
    assert not store.summary(3).seen_in_training
    s4 = store.summary(4)
    assert s4.split is None and not s4.seen_in_training


def test_summary_without_birth_or_sex_is_none(tmp_path: Path) -> None:
    data = tmp_path / "d"
    data.mkdir()
    pl.DataFrame(
        [(6, T0, "LAB//220045//bpm", 90.0, 60)], schema=SCHEMA, orient="row"
    ).write_parquet(data / "0.parquet")
    summary = _store(data).summary(6)
    assert summary.sex is None and summary.age_years is None


def test_load_splits_handles_missing_files(tmp_path: Path) -> None:
    assert load_splits(None) == {}
    assert load_splits(tmp_path / "nope.parquet") == {}
