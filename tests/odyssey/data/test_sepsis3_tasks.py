"""Task set v2: SOFA scoring, the Sepsis-3 concept, readmission onsets, sidecars."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from odyssey.data.alert_events import (
    ALERT_EVENTS,
    ALERT_EVENTS_V1,
    ALERT_EVENTS_V2,
    alert_events_for,
    all_event_times,
    event_times,
)
from odyssey.data.concepts import (
    TASK_SETS,
    Sepsis3Rule,
    concepts_for_source,
    label_concepts_by_visit,
)
from odyssey.data.sidecars import (
    MICROBIOLOGY,
    activate_sidecars,
    active_sidecar,
    sidecar_context,
    sidecar_root_for,
)
from odyssey.data.sofa import component_observations, sofa_timeseries


T0 = datetime(2024, 1, 1, 0, 0)
H = timedelta(hours=1)

# MIMIC-IV code prefixes (code_mapping's mimic_iv table)
CREAT = "LAB//RESULT//50912//mg/dL"
PLT = "LAB//RESULT//51265//K/uL"
BILI = "LAB//RESULT//50885//mg/dL"
MAP_NI = "LAB//220181//mmHg"
PAO2 = "LAB//RESULT//50821//mm Hg"
FIO2 = "LAB//223835//UNK"
GCS_E, GCS_V, GCS_M = "LAB//220739//UNK", "LAB//223900//UNK", "LAB//223901//UNK"
URINE = "SUBJECT_FLUID_OUTPUT//226559//mL"
NOREPI_START, NOREPI_END = "INFUSION_START//221906", "INFUSION_END//221906"
VENT_START, VENT_END = "PROCEDURE//START//225792", "PROCEDURE//END//225792"
ABX = "MEDICATION//START//Vancomycin//67457033950"


def _frame(rows, *, with_route=False):
    schema = {
        "subject_id": pl.Int64,
        "hadm_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.Utf8,
        "numeric_value": pl.Float64,
    }
    if with_route:
        schema["route"] = pl.Utf8
    return pl.DataFrame(rows, schema=schema, orient="row")


# ---------------------------------------------------------------------------
# SOFA
# ---------------------------------------------------------------------------


def test_sofa_component_scores_follow_the_bands() -> None:
    rows = [
        (1, 10, T0, PLT, 90.0),  # coag 2
        (1, 10, T0 + H, BILI, 6.5),  # liver 3
        (1, 10, T0 + 2 * H, CREAT, 2.1),  # renal 2
        (1, 10, T0 + 3 * H, MAP_NI, 65.0),  # cardio 1
        (1, 10, T0 + 4 * H, GCS_E, 3.0),
        (1, 10, T0 + 4 * H, GCS_V, 4.0),
        (1, 10, T0 + 4 * H, GCS_M, 5.0),  # total 12 -> cns 2
        (1, 10, T0 + 5 * H, FIO2, 50.0),
        (1, 10, T0 + 6 * H, PAO2, 120.0),  # PF 240, not ventilated -> 2
    ]
    obs = component_observations(_frame(rows), key="subject_id")
    got = {
        (r["component"], r["score"])
        for r in obs.select("component", "score").to_dicts()
    }
    assert got == {
        ("coagulation", 2),
        ("liver", 3),
        ("renal", 2),
        ("cardiovascular", 1),
        ("cns", 2),
        ("respiration", 2),
    }
    ts = sofa_timeseries(_frame(rows), key="subject_id")
    # worst-in-24h accumulates: by the last instant every component counts
    assert ts["sofa"].to_list()[-1] == 2 + 3 + 2 + 1 + 2 + 2
    assert ts["sofa"].to_list()[0] == 2  # only platelets at t0


def test_sofa_vasopressor_rates_and_ventilation_and_window_decay() -> None:
    rows = [
        (1, 10, T0, NOREPI_START, 0.05),  # norepi <= 0.1 -> 3, active until END
        (1, 10, T0 + 3 * H, NOREPI_END, None),
        (1, 10, T0 + 40 * H, MAP_NI, 80.0),  # normal MAP, >24h later
        (1, 10, T0 + 41 * H, VENT_START, None),
        (1, 10, T0 + 42 * H, FIO2, 100.0),
        (1, 10, T0 + 43 * H, PAO2, 90.0),  # PF 90 ventilated -> 4
        (1, 10, T0 + 44 * H, VENT_END, None),
    ]
    # The grid is made of abnormal-reading instants (a normal MAP adds no
    # point); ask for T0+40h explicitly to see the window decay.
    probe = pl.DataFrame({"subject_id": [1], "time": [T0 + 40 * H]}).cast(
        {"subject_id": pl.Int64, "time": pl.Datetime("us")}
    )
    ts = sofa_timeseries(_frame(rows), key="subject_id", grid_times=probe)
    by_time = dict(zip(ts["time"].to_list(), ts["sofa"].to_list()))
    assert by_time[T0] == 3  # norepi on
    assert by_time[T0 + 40 * H] == 0  # infusion ended 37h ago: outside the 24h window
    assert by_time[T0 + 43 * H] == 4  # ventilated PF < 100


def test_sofa_urine_output_needs_24h_of_record() -> None:
    rows = [(1, 10, T0, CREAT, 0.8)] + [
        (1, 10, T0 + k * H, URINE, 10.0) for k in range(1, 30)
    ]
    obs = component_observations(_frame(rows), key="subject_id")
    renal = obs.filter(pl.col("component") == "renal")
    assert renal.height > 0
    # no renal (urine) score before 24h into the record
    assert renal["time"].min() >= T0 + 24 * H
    assert set(renal["score"].to_list()) == {3}  # 10 mL/h * 24h = 240: < 500, >= 200


# ---------------------------------------------------------------------------
# Sepsis-3 concept
# ---------------------------------------------------------------------------


def _septic_visit(*, abx_offset_h: float = 2.0, sofa: bool = True, hadm: int = 10):
    """Culture at T0+1h; antibiotic at T0+abx_offset; SOFA 2 (platelets 90) at T0+5h."""
    rows = [
        (1, hadm, T0, "HOSPITAL_ADMISSION//EMERGENCY", None, None),
        (1, hadm, T0 + abx_offset_h * H, ABX, None, "IV"),
        (1, hadm, T0 + 30 * H, "HOSPITAL_DISCHARGE//HOME", None, None),
    ]
    if sofa:
        rows.append((1, hadm, T0 + 5 * H, PLT, 90.0, None))
    else:
        rows.append((1, hadm, T0 + 5 * H, PLT, 300.0, None))
    return _frame(rows, with_route=True)


def _cultures(time: datetime, hadm=None):
    return pl.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": pl.Series([hadm], dtype=pl.Int64),
            "time": [time],
            "spec_type_desc": ["BLOOD CULTURE"],
            "positive_culture": [False],
            "micro_specimen_id": [1],
        }
    )


def test_sepsis3_fires_with_culture_antibiotic_and_sofa() -> None:
    concepts = [
        c for c in concepts_for_source("mimic_iv", task_set="v2") if c.name == "sepsis3"
    ]
    assert len(concepts) == 1 and isinstance(concepts[0].rules[0], Sepsis3Rule)
    with sidecar_context({MICROBIOLOGY: _cultures(T0 + 1 * H)}):
        labeled = label_concepts_by_visit(
            _septic_visit(), concepts, include_first_time=True
        )
    row = labeled.to_dicts()[0]
    assert row["sepsis3"] == 1 and row["sepsis3_observed"] == 1
    # suspicion = min(culture 1h, antibiotic 2h) = 1h; SOFA>=2 at 5h -> onset 5h
    assert row["sepsis3_first_time"] == T0 + 5 * H


def test_sepsis3_negative_without_sofa_rise_or_outside_the_window() -> None:
    concepts = [
        c for c in concepts_for_source("mimic_iv", task_set="v2") if c.name == "sepsis3"
    ]
    with sidecar_context({MICROBIOLOGY: _cultures(T0 + 1 * H)}):
        no_sofa = label_concepts_by_visit(_septic_visit(sofa=False), concepts)
        # antibiotic 100h after the culture: not "suspected infection"
        late_abx = label_concepts_by_visit(_septic_visit(abx_offset_h=100.0), concepts)
    assert no_sofa.to_dicts()[0]["sepsis3"] == 0
    assert no_sofa.to_dicts()[0]["sepsis3_observed"] == 1  # SOFA was assessable
    assert late_abx.to_dicts()[0]["sepsis3"] == 0


def test_sepsis3_culture_attribution_by_hadm_or_by_time_span() -> None:
    concepts = [
        c for c in concepts_for_source("mimic_iv", task_set="v2") if c.name == "sepsis3"
    ]
    # culture row carries a DIFFERENT hadm_id: not this visit's, even though in span
    with sidecar_context({MICROBIOLOGY: _cultures(T0 + 1 * H, hadm=99)}):
        other = label_concepts_by_visit(_septic_visit(), concepts)
    assert other.to_dicts()[0]["sepsis3"] == 0
    # hadm-less culture outside the visit span: not attributed
    with sidecar_context({MICROBIOLOGY: _cultures(T0 - 48 * H)}):
        outside = label_concepts_by_visit(_septic_visit(), concepts)
    assert outside.to_dicts()[0]["sepsis3"] == 0


def test_sepsis3_route_exclusion_and_missing_sidecar() -> None:
    concepts = [
        c for c in concepts_for_source("mimic_iv", task_set="v2") if c.name == "sepsis3"
    ]
    topical = _septic_visit().with_columns(
        pl.when(pl.col("code") == ABX)
        .then(pl.lit("TP"))
        .otherwise(pl.col("route"))
        .alias("route")
    )
    with sidecar_context({MICROBIOLOGY: _cultures(T0 + 1 * H)}):
        assert label_concepts_by_visit(topical, concepts).to_dicts()[0]["sepsis3"] == 0
    with sidecar_context({}):
        row = label_concepts_by_visit(_septic_visit(), concepts).to_dicts()[0]
    assert (
        row["sepsis3"] == 0 and row["sepsis3_observed"] == 0
    )  # unobserved, not negative


# ---------------------------------------------------------------------------
# Task sets and alert events
# ---------------------------------------------------------------------------


def test_task_sets_are_versioned_and_backward_compatible() -> None:
    v1 = [c.name for c in concepts_for_source("mimic_iv")]
    assert v1 == list(TASK_SETS["v1"]) and len(v1) == 15
    v2 = [c.name for c in concepts_for_source("mimic_iv", task_set="v2")]
    assert v2 == v1 + ["sepsis3"]
    assert [c.name for c in concepts_for_source("eicu", task_set="v2")] == [
        c.name for c in concepts_for_source("eicu")
    ]  # no SOFA ingredients on eICU: sepsis3 dropped, v2 == v1 there
    with pytest.raises(ValueError, match="unknown task_set"):
        concepts_for_source("mimic_iv", task_set="v9")
    assert ALERT_EVENTS is ALERT_EVENTS_V1 and alert_events_for("v1") is ALERT_EVENTS_V1
    assert [a.name for a in ALERT_EVENTS_V2] == [
        "vasopressor_start",
        "icu_admission",
        "acute_kidney_injury",
        "death",
        "sepsis3",
        "readmission_30d",
    ]


def test_readmission_onset_is_next_admission_and_censor_is_end_of_record() -> None:
    rows = [
        (1, 10, T0, "HOSPITAL_ADMISSION//EMERGENCY", None),
        (1, 10, T0 + 48 * H, "HOSPITAL_DISCHARGE//HOME", None),
        (1, 20, T0 + 48 * H + 10 * 24 * H, "HOSPITAL_ADMISSION//EMERGENCY", None),
        (1, 20, T0 + 48 * H + 12 * 24 * H, "HOSPITAL_DISCHARGE//HOME", None),
        (1, None, T0 + 100 * 24 * H, "LAB//RESULT//50912//mg/dL", 1.0),
    ]
    readmit = [a for a in ALERT_EVENTS_V2 if a.name == "readmission_30d"][0]
    times = event_times(_frame(rows), readmit)
    # visit 10: readmitted 10 days after discharge; visit 20: never
    assert times.onset == {(1, 10): pytest.approx(48.0 + 240.0)}
    # censoring for BOTH visits is the subject's last event (100 days), not discharge
    assert times.censor[(1, 10)] == pytest.approx(100 * 24.0)
    assert times.censor[(1, 20)] == pytest.approx(100 * 24.0)
    all_times = all_event_times(
        _frame(rows), ALERT_EVENTS_V2, "mimic_iv", task_set="v2"
    )
    assert set(all_times) == {a.name for a in ALERT_EVENTS_V2}


# ---------------------------------------------------------------------------
# Sidecar discovery
# ---------------------------------------------------------------------------


def test_sidecar_discovery_from_a_split_directory(tmp_path: Path) -> None:
    root = tmp_path / "mimiciv_3.1_v1"
    (root / "data" / "train").mkdir(parents=True)
    (root / "sidecars").mkdir()
    _cultures(T0).write_parquet(root / "sidecars" / "microbiology.parquet")
    assert sidecar_root_for(root / "data" / "train") == root / "sidecars"
    try:
        assert activate_sidecars(root / "data" / "train") == ["microbiology"]
        assert active_sidecar(MICROBIOLOGY) is not None
        assert activate_sidecars(root / "data" / "held_out") == [
            "microbiology"
        ]  # cached
    finally:
        activate_sidecars(None)
    assert active_sidecar(MICROBIOLOGY) is None
    assert activate_sidecars(tmp_path / "nowhere" / "data" / "train") == []
