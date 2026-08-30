"""Tests for the strong (best-effort) baseline feature builder."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from odyssey.inference.baseline_features import (
    CONTEXT_FEATURES,
    DRUG_CLASSES,
    DRUG_STATS,
    FAMILY_LABELS,
    FAMILY_STATS,
    SIGNAL_PANEL,
    SIGNAL_STATS,
    StrongFeatureBuilder,
    _reduce_windows,
    feature_names,
)


T0 = datetime(2024, 1, 1)


def _frame(
    rows: list[tuple[int, str, datetime | None, float | None, int | None]],
):
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


def _hr(h: float, v: float, sid: int = 1, hadm: int = 10):
    return (sid, "LAB//220045//bpm::HIGH", T0 + timedelta(hours=h), v, hadm)


def _events() -> pl.DataFrame:
    rows = [
        (1, "MEDS_BIRTH", T0 - timedelta(days=365.25 * 60), None, None),
        (1, "GENDER//F", None, None, None),
        _hr(0.0, 80.0),
        _hr(1.0, 90.0),
        _hr(2.0, 120.0),
        _hr(30.0, 70.0),
        # creatinine (LOINC 2160-0), Fahrenheit temperature (8310-5, unit F)
        (1, "LAB//RESULT//50912//mg/dL::NORMAL", T0 + timedelta(hours=0.5), 1.0, 10),
        (1, "LAB//RESULT//50912//mg/dL::HIGH", T0 + timedelta(hours=20.0), 2.0, 10),
        (1, "LAB//223761//F", T0 + timedelta(hours=1.5), 98.6, 10),
        # drugs: vasopressor at 1.5h, antibiotic at 0.2h
        (
            1,
            "MEDICATION//norepinephrine//Administered",
            T0 + timedelta(hours=1.5),
            None,
            10,
        ),
        (1, "MEDICATION//START//vancomycin//123", T0 + timedelta(hours=0.2), None, 10),
        (1, "ICU_ADMISSION//MICU", T0 + timedelta(hours=1.0), None, 10),
        (1, "ICU_DISCHARGE//MICU", T0 + timedelta(hours=25.0), None, 10),
        # a second, later visit
        (1, "HOSPITAL_ADMISSION//EW EMER.", T0 + timedelta(hours=100.0), None, 11),
        _hr(100.5, 60.0, hadm=11),
        # subject 2: male, no values at all
        (2, "GENDER//M", None, None, None),
        (2, "DIAGNOSIS//ICD//10//I50", T0 + timedelta(hours=1.0), None, 20),
    ]
    return _frame(rows)


def _col(name: str) -> int:
    return feature_names().index(name)


def test_feature_names_cover_every_block() -> None:
    names = feature_names()
    assert len(names) == (
        len(CONTEXT_FEATURES)
        + len(SIGNAL_PANEL) * len(SIGNAL_STATS)
        + len(DRUG_CLASSES) * len(DRUG_STATS)
        + len(FAMILY_LABELS) * len(FAMILY_STATS)
    )
    assert len(set(names)) == len(names)


def test_reduce_windows_matches_a_python_loop() -> None:
    values = np.array([5.0, 3.0, 8.0, 1.0, 9.0])
    lo = np.array([0, 1, 3, 2, 5])
    hi = np.array([2, 4, 3, 5, 5])
    got_min = _reduce_windows(values, lo, hi, np.minimum)
    got_max = _reduce_windows(values, lo, hi, np.maximum)
    for i in range(len(lo)):
        seg = values[lo[i] : hi[i]]
        if len(seg) == 0:
            assert np.isnan(got_min[i]) and np.isnan(got_max[i])
        else:
            assert got_min[i] == seg.min() and got_max[i] == seg.max()


def test_signal_stats_include_the_index_instant() -> None:
    """Protocol v4: the reading AT the index time is visible, (t-w, t] windows.

    Before v4 the baseline saw only events strictly before t while the
    model-side scorers had already consumed the token at t -- the fairness
    asymmetry the 2026-08-30 leakage review flagged. One shared boundary
    now: at hour 2.0 the reading charted at 2.0 (120) IS the last value.
    """
    builder = StrongFeatureBuilder(_events(), source="mimic_iv")
    x = builder.features([1, 1], [10, 10], [2.0, 2.5])
    hr = "heart_rate"
    assert x[0, _col(f"{hr}.last")] == pytest.approx(120.0)
    assert x[0, _col(f"{hr}.hours_since_last")] == pytest.approx(0.0)
    assert x[0, _col(f"{hr}.n_24h")] == 3
    assert x[0, _col(f"{hr}.mean_24h")] == pytest.approx((80.0 + 90.0 + 120.0) / 3)
    assert x[0, _col(f"{hr}.delta_prev")] == pytest.approx(30.0)
    # at 2.5 the same three readings are visible, from half an hour later
    assert x[1, _col(f"{hr}.last")] == pytest.approx(120.0)
    assert x[1, _col(f"{hr}.hours_since_last")] == pytest.approx(0.5)
    assert x[1, _col(f"{hr}.max_24h")] == pytest.approx(120.0)
    assert x[1, _col(f"{hr}.min_6h")] == pytest.approx(80.0)
    assert x[1, _col(f"{hr}.delta_visit_first")] == pytest.approx(40.0)
    assert x[1, _col(f"{hr}.ratio_visit_min")] == pytest.approx(1.5)


def test_windows_expire_and_visit_baseline_resets_per_visit() -> None:
    builder = StrongFeatureBuilder(_events(), source="mimic_iv")
    x = builder.features([1, 1], [10, 11], [30.5, 101.0])
    hr = "heart_rate"
    # at 30.5 only the 30.0 reading is inside 24h; last-prev spans the gap
    assert x[0, _col(f"{hr}.n_24h")] == 1
    assert x[0, _col(f"{hr}.mean_24h")] == pytest.approx(70.0)
    assert x[0, _col(f"{hr}.delta_prev")] == pytest.approx(-50.0)
    # second visit: baseline is the visit's own first reading (60), not 80
    assert x[1, _col(f"{hr}.last")] == pytest.approx(60.0)
    assert x[1, _col(f"{hr}.delta_visit_first")] == pytest.approx(0.0)
    assert x[1, _col("n_prior_visits")] == 1
    assert x[1, _col("hours_into_visit")] == pytest.approx(1.0)


def test_unit_conversion_and_creatinine_ratio() -> None:
    builder = StrongFeatureBuilder(_events(), source="mimic_iv")
    x = builder.features([1], [10], [21.0])
    assert x[0, _col("temperature.last")] == pytest.approx(37.0, abs=0.01)
    assert x[0, _col("creatinine.last")] == pytest.approx(2.0)
    assert x[0, _col("creatinine.ratio_visit_min")] == pytest.approx(2.0)
    assert x[0, _col("creatinine.delta_visit_first")] == pytest.approx(1.0)


def test_drug_classes_context_and_families() -> None:
    builder = StrongFeatureBuilder(_events(), source="mimic_iv")
    x = builder.features([1, 1, 2], [10, 10, 20], [1.0, 26.0, 2.0])
    assert x[0, _col("drug.antibiotic.n_6h")] == 1
    assert x[0, _col("drug.vasopressor.n_24h")] == 0  # starts at 1.5h
    assert x[1, _col("drug.vasopressor.ever_visit")] == 1
    assert x[1, _col("drug.vasopressor.hours_since_last")] == pytest.approx(24.5)
    assert x[1, _col("drug.vasopressor.n_24h")] == 0
    # ICU: admitted at 1h, discharged at 25h. Protocol v4: the admission
    # event AT the index instant is visible.
    assert x[0, _col("in_icu")] == 1
    assert x[0, _col("hours_since_icu_admission")] == pytest.approx(0.0)
    assert x[1, _col("in_icu")] == 0  # discharged at 25h < 26h
    x2 = builder.features([1], [10], [3.0])
    assert x2[0, _col("in_icu")] == 1
    assert x2[0, _col("hours_since_icu_admission")] == pytest.approx(2.0)
    # demographics
    assert x[0, _col("sex_female")] == 1
    assert x[2, _col("sex_female")] == 0
    assert x[0, _col("age_years")] == pytest.approx(60.0, abs=0.01)
    assert np.isnan(x[2, _col("age_years")])
    # family counts: subject 2 has one diagnosis event before 2.0
    assert x[2, _col("family.diagnosis.n_24h")] == 1
    assert x[2, _col("family.lab.n_visit")] == 0
    # unknown subject rows stay NaN
    x3 = builder.features([99], [1], [1.0])
    assert np.isnan(x3).all()


def test_eicu_source_resolves_its_own_prefixes() -> None:
    rows = [
        (5, "VITALS//PERIODIC//HEARTRATE::HIGH", T0 + timedelta(hours=1.0), 130.0, 50),
        (5, "VITALS//PERIODIC//TEMPERATURE", T0 + timedelta(hours=1.0), 38.5, 50),
        (5, "INFUSION_DRUG//norepinephrine", T0 + timedelta(hours=1.5), 5.0, 50),
        (5, "MEDICATION//STARTED//vancomycin", T0 + timedelta(hours=1.0), None, 50),
    ]
    builder = StrongFeatureBuilder(_frame(rows), source="eicu")
    x = builder.features([5], [50], [2.0])
    assert x[0, _col("heart_rate.last")] == pytest.approx(130.0)
    assert x[0, _col("temperature.last")] == pytest.approx(38.5)  # Celsius already
    assert x[0, _col("drug.vasopressor.n_6h")] == 1
    assert x[0, _col("drug.antibiotic.n_6h")] == 1
