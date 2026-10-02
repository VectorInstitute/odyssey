"""Tests for the self-supervised window-summary targets and their chunk lookup."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.training.summary_targets import (
    COUNT_STATS,
    SIGNAL_STATS,
    SummaryTargetStats,
    SummaryTargetTables,
    compute_summary_targets,
    count_target_mask,
    fit_summary_stats,
    landmark_rows,
    load_summary_tables,
    summary_target_names,
    summary_targets_for_chunk,
)


T0 = datetime(2024, 1, 1)


def _frame(rows):
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


def _creat(h: float, v: float, sid: int = 1, hadm: int = 10):
    return (sid, "LAB//RESULT//50912//mg/dL::HIGH", T0 + timedelta(hours=h), v, hadm)


def _events() -> pl.DataFrame:
    return _frame(
        [
            (1, "MEDS_BIRTH", T0 - timedelta(days=365.25 * 60), None, None),
            (1, "GENDER//F", None, None, None),
            _creat(0.0, 1.0),
            _creat(5.0, 1.2),
            _creat(9.0, 2.2),
            (1, "MEDICATION//norepinephrine", T0 + timedelta(hours=8.5), None, 10),
            (1, "MEDICATION//norepinephrine", T0 + timedelta(hours=9.0), None, 10),
        ]
    )


def test_target_names_cover_signals_and_counts() -> None:
    names = summary_target_names()
    assert len(names) == len(set(names))
    assert "creatinine.delta_visit_first" in names
    assert "drug.vasopressor.n_6h" in names
    assert "family.lab.n_24h" in names
    assert all(n.rsplit(".", 1)[-1] in SIGNAL_STATS + COUNT_STATS for n in names)
    counts = count_target_mask(names)
    assert counts.sum() == sum(n.rsplit(".", 1)[-1] in COUNT_STATS for n in names)


def test_landmark_rows_are_every_4h_at_the_last_event_time() -> None:
    sids, vids, times = landmark_rows(_events(), landmark_hours=4.0)
    # visit spans 0..9 h: landmarks at 0, 4, 8 -> last event at or before: 0, 0, 5
    assert sids == [1, 1]  # the 0 h and 4 h landmarks both map to the 0 h event
    assert vids == [10, 10]
    assert times == [0.0, 5.0]


def test_compute_summary_targets_reports_baseline_change_and_counts() -> None:
    frame = compute_summary_targets(_events(), landmark_hours=4.0)
    assert frame.columns[:3] == ["subject_id", "visit_id", "time_hours"]
    at5 = frame.filter(pl.col("time_hours") == 5.0)
    assert at5.height == 1
    assert at5["creatinine.delta_visit_first"][0] == pytest.approx(0.2)
    assert at5["creatinine.max_24h"][0] == pytest.approx(1.2)
    assert at5["creatinine.min_24h"][0] == pytest.approx(1.0)
    assert at5["drug.vasopressor.n_6h"][0] == 0.0
    assert at5["family.lab.n_24h"][0] == 2.0
    at0 = frame.filter(pl.col("time_hours") == 0.0)
    assert at0["creatinine.delta_visit_first"][0] == pytest.approx(0.0)
    # a signal never measured is NaN, never a fake zero
    assert np.isnan(at0["lactate.min_6h"][0])


def test_compute_summary_targets_on_empty_frame_has_the_full_schema() -> None:
    frame = compute_summary_targets(_events().head(0))
    assert frame.height == 0
    assert set(summary_target_names()) <= set(frame.columns)


def test_stats_standardize_counts_through_log1p_and_winsorize() -> None:
    names = summary_target_names()
    j_count = names.index("drug.vasopressor.n_6h")
    j_delta = names.index("creatinine.delta_visit_first")
    n = 1000
    raw = np.full((n, len(names)), np.nan)
    rng = np.random.default_rng(0)
    raw[:, j_count] = rng.poisson(2.0, n)
    raw[:, j_delta] = rng.normal(0.0, 1.0, n)
    raw[0, j_delta] = 1e6  # a sentinel that must not set the scale
    frame = pl.DataFrame(
        {
            "subject_id": [1] * n,
            "visit_id": [1] * n,
            "time_hours": np.arange(n, dtype=float),
        }
    )
    frame = frame.with_columns(
        [pl.Series(name, raw[:, i], dtype=pl.Float32) for i, name in enumerate(names)]
    )
    stats = fit_summary_stats([frame])
    assert stats.std[j_delta] < 2.0
    z = stats.transform(raw)
    assert abs(float(np.nanmean(z[:, j_delta]))) < 0.2
    assert float(z[0, j_delta]) < 5.0  # clipped, not 1e6 standard deviations
    assert np.isnan(z[:, names.index("lactate.min_6h")]).all()
    # counts: log1p(0) -> the smallest standardized value
    zero = stats.transform(
        np.where(np.arange(len(names)) == j_count, 0.0, np.nan)[None, :]
    )
    assert float(zero[0, j_count]) < 0.0


def test_stats_round_trip_through_json(tmp_path) -> None:
    stats = fit_summary_stats([compute_summary_targets(_events())])
    stats.save(tmp_path / "stats.json")
    back = SummaryTargetStats.load(tmp_path / "stats.json")
    assert back.names == stats.names
    np.testing.assert_allclose(back.mean, stats.mean)
    np.testing.assert_allclose(back.std, stats.std)


def _chunk(subject_ids, visit_ids, times, real=None) -> StreamingChunk:
    lanes, length = np.asarray(times).shape
    real_mask = (
        torch.ones(lanes, length, dtype=torch.bool)
        if real is None
        else torch.tensor(real)
    )
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.ones(lanes, length, dtype=torch.long),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(lanes, length, dtype=torch.long),
                time_stamps=torch.tensor(times, dtype=torch.float32),
                ages=torch.full((lanes, length), 40.0),
                visit_orders=torch.zeros(lanes, length, dtype=torch.long),
                visit_segments=torch.zeros(lanes, length, dtype=torch.long),
            ),
        ),
        targets=torch.ones(lanes, length, dtype=torch.long),
        reset_mask=torch.zeros(lanes, length, dtype=torch.bool),
        real_mask=real_mask,
        subject_ids=torch.tensor(subject_ids),
        patient_end=torch.zeros(lanes, length, dtype=torch.bool),
        visit_ids=torch.tensor(visit_ids),
        visit_end=torch.zeros(lanes, length, dtype=torch.bool),
    )


def _tables() -> SummaryTargetTables:
    frame = compute_summary_targets(_events())
    stats = fit_summary_stats([frame])
    tables = SummaryTargetTables(stats)
    tables.add_frame(frame)
    return tables


def test_chunk_targets_land_on_bundle_ends_of_landmark_rows() -> None:
    tables = _tables()
    assert len(tables) == 1
    # lane 0: patient 1, visit 10; the 5.0 h bundle has two tokens (a
    # panel), the target must sit on the LAST of them. lane 1: unknown patient.
    chunk = _chunk(
        subject_ids=[[1, 1, 1, 1], [7, 7, 7, 7]],
        visit_ids=[[10, 10, 10, 10], [1, 1, 1, 1]],
        times=[[0.0, 5.0, 5.0, 9.0], [0.0, 4.0, 8.0, 12.0]],
    )
    out = summary_targets_for_chunk(chunk, tables)
    assert out is not None
    hit = out.mask.any(dim=-1)
    assert hit.tolist() == [[True, False, True, False], [False, False, False, False]]
    k = tables.num_targets
    assert out.values.shape == (2, 4, k)
    j = tables.stats.names.index("creatinine.delta_visit_first")
    # the 5 h row's standardized delta is above the 0 h row's (0.2 vs 0.0)
    assert out.values[0, 2, j] > out.values[0, 0, j]
    # NaN targets are masked out and zero-filled, never passed as NaN
    assert torch.isfinite(out.values).all()
    assert not out.mask[0, 2, tables.stats.names.index("lactate.min_6h")]


def test_chunk_targets_ignore_padding_and_return_none_when_nothing_matches() -> None:
    tables = _tables()
    chunk = _chunk(
        subject_ids=[[1, 1]],
        visit_ids=[[10, 10]],
        times=[[5.0, 5.0]],
        real=[[True, False]],
    )
    out = summary_targets_for_chunk(chunk, tables)
    assert out is not None
    assert out.mask.any(dim=-1).tolist() == [[True, False]]
    assert (
        summary_targets_for_chunk(
            _chunk(subject_ids=[[1]], visit_ids=[[10]], times=[[3.0]]), tables
        )
        is None
    )


def test_tables_load_from_a_directory_of_parquets(tmp_path) -> None:
    frame = compute_summary_targets(_events())
    (tmp_path / "train").mkdir()
    frame.write_parquet(tmp_path / "train" / "shard_0.parquet")
    fit_summary_stats([frame]).save(tmp_path / "stats.json")
    tables = load_summary_tables(tmp_path / "train")
    assert len(tables) == 1
    times, values = tables.lookup(1, 10)
    assert times.tolist() == [0.0, 5.0]
    assert values.shape == (2, tables.num_targets)
    assert tables.lookup(1, 11) is None
