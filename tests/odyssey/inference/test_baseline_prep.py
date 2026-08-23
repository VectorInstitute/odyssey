"""Streaming baseline prep: equivalence with the whole-frame path, and memory shape.

The module's reason to exist is a confirmed OOM incident (whole-split
frames held across three fits, plus a second full raw load for event
times). These tests pin the two properties that make the streaming
replacement safe: it computes EXACTLY what the whole-frame path computes,
and it never holds more than one shard's frames alive at a time.
"""

import gc
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pytest

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import _index_rows_from_events, features_for_events
from odyssey.inference.baseline_prep import (
    BaselineData,
    _verify_matching_origins,
    prepare_baseline_data,
)


T0 = datetime(2024, 1, 1)


def _shard(subject_ids: List[int]) -> pl.DataFrame:
    """Hourly heart-rate readings; even subjects deteriorate and get pressors."""
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in subject_ids:
        hadm = 1000 + sid
        for h in range(24):
            hr = 130.0 if sid % 2 == 0 and h >= 12 else 80.0
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), hr, hadm))
        if sid % 2 == 0:
            rows.append(
                (
                    sid,
                    "MEDICATION//norepinephrine//Administered",
                    T0 + timedelta(hours=14),
                    None,
                    hadm,
                )
            )
        if sid % 4 == 0:
            rows.append(
                (sid, "ICU_ADMISSION//MICU", T0 + timedelta(hours=6), None, hadm)
            )
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


def _fake_shards() -> Dict[Path, pl.DataFrame]:
    """Two shards, disjoint subjects (the MEDS invariant the module relies on)."""
    return {
        Path("shard_0"): _shard([1, 2, 3, 4, 5, 6]),
        Path("shard_1"): _shard([7, 8, 9, 10, 11, 12]),
    }


def _identity(events: pl.DataFrame) -> pl.DataFrame:
    return events


def _streamed(**kwargs: object) -> BaselineData:
    shards = _fake_shards()
    return prepare_baseline_data(
        list(shards),
        _identity,
        None,
        alerts=ALERT_EVENTS,
        feature_sets=("strong", "basic"),
        source="mimic_iv",
        loader=lambda p: shards[p],
        **kwargs,  # type: ignore[arg-type]
    )


def _key(r: object) -> Tuple[int, int, float]:
    return (r.subject_id, r.visit_id, r.time_hours)  # type: ignore[attr-defined]


def test_matches_whole_frame_path_exactly() -> None:
    """Rows, times, and every feature value equal the single-frame computation."""
    shards = _fake_shards()
    whole_raw = pl.concat(list(shards.values()), how="vertical")
    # Times come from prepared-raw events; rows/features from binned events --
    # mirror the helper's own pipeline stages or the comparison is vacuous.
    whole_times = all_event_times(whole_raw, ALERT_EVENTS, "mimic_iv")
    whole = add_value_tokens(whole_raw, None, source="mimic_iv")
    whole_rows = _index_rows_from_events(whole, ALERT_EVENTS, landmark_hours=4.0)

    data = _streamed()

    assert set(data.rows) == set(whole_rows)
    for event in whole_rows:
        assert sorted(map(_key, data.rows[event])) == sorted(
            map(_key, whole_rows[event])
        )
    assert set(data.times) == set(whole_times)
    for event, times in whole_times.items():
        assert data.times[event].onset == times.onset
        assert data.times[event].censor == times.censor
        assert data.times[event].subject_scoped == times.subject_scoped

    for feature_set in ("strong", "basic"):
        whole_feats = features_for_events(
            whole, whole_rows, source="mimic_iv", feature_set=feature_set
        )
        for event, feats in whole_feats.items():
            by_key_whole = {_key(r): feats[i] for i, r in enumerate(whole_rows[event])}
            streamed = data.features[feature_set][event]
            assert streamed.dtype == np.float32
            assert streamed.shape[0] == len(data.rows[event])
            for i, r in enumerate(data.rows[event]):
                assert np.array_equal(
                    streamed[i],
                    by_key_whole[_key(r)].astype(np.float32),
                    equal_nan=True,
                )


def test_shared_grid_features_are_aliased_not_copied() -> None:
    """Events share the landmark grid, so they must share ONE feature array."""
    data = _streamed()
    for feature_set in ("strong", "basic"):
        arrays = list(data.features[feature_set].values())
        assert len(arrays) > 1
        assert all(a is arrays[0] for a in arrays[1:])


def test_at_most_one_shard_alive_at_a_time() -> None:
    """The streaming loop must release each shard's frame before the next load."""
    shards = _fake_shards()
    alive: List[weakref.ref] = []
    max_alive = 0

    def loader(path: Path) -> pl.DataFrame:
        nonlocal max_alive
        gc.collect()
        live_now = sum(1 for ref in alive if ref() is not None)
        max_alive = max(max_alive, live_now)
        frame = shards[path].clone()
        alive.append(weakref.ref(frame))
        return frame

    prepare_baseline_data(
        list(shards),
        _identity,
        None,
        alerts=ALERT_EVENTS,
        feature_sets=("strong",),
        source="mimic_iv",
        loader=loader,
    )
    # At the moment shard N loads, shard N-1's raw frame must already be
    # collectable -- a regression here recreates the all-shards-resident
    # footprint this module exists to prevent.
    assert max_alive == 0


def test_unknown_feature_set_refuses() -> None:
    shards = _fake_shards()
    with pytest.raises(ValueError, match="unknown baseline feature set"):
        prepare_baseline_data(
            list(shards),
            _identity,
            None,
            alerts=ALERT_EVENTS,
            feature_sets=("bogus",),
            loader=lambda p: shards[p],
        )


# ---------------------------------------------------------------------------
# Missingness stress protocol glue: degraded_shard_dir
# (docs/missingness_protocol.md; odyssey.data.degrade produces the shards)
# ---------------------------------------------------------------------------


def _degraded_shard(subject_ids: List[int]) -> pl.DataFrame:
    """Build a degraded stand-in for _shard's subjects.

    Same subjects/visits as _shard, but every heart-rate value flattened
    to a single constant -- a stand-in for a real degrade.py transform,
    distinguishable in the resulting features without needing the real
    module here.
    """
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in subject_ids:
        hadm = 1000 + sid
        for h in range(24):
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), 100.0, hadm))
        if sid % 2 == 0:
            rows.append(
                (
                    sid,
                    "MEDICATION//norepinephrine//Administered",
                    T0 + timedelta(hours=14),
                    None,
                    hadm,
                )
            )
        if sid % 4 == 0:
            rows.append(
                (sid, "ICU_ADMISSION//MICU", T0 + timedelta(hours=6), None, hadm)
            )
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


def test_degraded_shard_dir_scores_features_from_the_degraded_copy() -> None:
    """Score against the degraded copy while keeping clean labels.

    rows/times/labels come from the clean split; features come from the
    degraded one -- the whole point of the hook.
    """
    clean = _fake_shards()
    degraded = {
        Path("degraded") / p.name: _degraded_shard(list(range(1 + i * 6, 7 + i * 6)))
        for i, p in enumerate(clean)
    }
    all_frames = {**clean, **degraded}
    data = prepare_baseline_data(
        list(clean),
        _identity,
        None,
        alerts=ALERT_EVENTS,
        feature_sets=("basic",),
        source="mimic_iv",
        loader=lambda p: all_frames[p],
        degraded_shard_dir=Path("degraded"),
    )
    clean_only = prepare_baseline_data(
        list(clean),
        _identity,
        None,
        alerts=ALERT_EVENTS,
        feature_sets=("basic",),
        source="mimic_iv",
        loader=lambda p: all_frames[p],
    )
    # Same landmark rows either way (clean split defines them; sorted since
    # polars group_by order isn't guaranteed stable across separate calls)...
    for event in clean_only.rows:
        assert sorted(map(_key, data.rows[event])) == sorted(
            map(_key, clean_only.rows[event])
        )
    # ...but the features differ, because they were built from the degraded
    # copy (every heart rate flattened to 100.0) instead of the clean one.
    assert not np.array_equal(
        data.features["basic"]["death"], clean_only.features["basic"]["death"]
    )


def test_verify_matching_origins_passes_when_origin_unchanged() -> None:
    clean = _shard([1, 2])
    # drop everything except the earliest row per subject and one extra --
    # origin (each subject's first non-birth timed row) is unchanged.
    degraded = clean.filter(
        pl.col("code") != "MEDICATION//norepinephrine//Administered"
    )
    _verify_matching_origins(clean, degraded, context="test")  # no raise


def test_verify_matching_origins_raises_when_a_subjects_earliest_row_is_gone() -> None:
    clean = _shard([1, 2])
    origin_row = clean.sort("time").filter(pl.col("subject_id") == 1).head(1)
    degraded = clean.filter(
        ~(
            (pl.col("subject_id") == 1)
            & (pl.col("time") == origin_row["time"][0])
            & (pl.col("code") == origin_row["code"][0])
        )
    )
    with pytest.raises(RuntimeError, match="shifted or"):
        _verify_matching_origins(clean, degraded, context="test")
