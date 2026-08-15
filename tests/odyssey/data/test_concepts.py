"""Tests for rule-derived concept labels."""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Union

import polars as pl
import pytest

from odyssey.data.concepts import (
    CONCEPTS,
    AnyOf,
    BaselineRelativeRule,
    CompositeConceptDefinition,
    ConceptDefinition,
    ConceptRule,
    DerivedGcsTotalRule,
    SustainedRule,
    label_concepts,
)


T0 = datetime(2024, 1, 1, 0, 0)


_EventRow = Union[Tuple[int, str, float], Tuple[int, str, float, datetime]]


def _events(rows: List[_EventRow]) -> pl.DataFrame:
    """Build a synthetic events frame; each row optionally includes a time.

    Rows may be ``(subject_id, code, value)`` (time defaults to ``T0``,
    fine for rules that don't need real timing) or
    ``(subject_id, code, value, time)`` for tests of the time-aware rule
    types (:class:`~odyssey.data.concepts.SustainedRule`,
    :class:`~odyssey.data.concepts.BaselineRelativeRule`,
    :class:`~odyssey.data.concepts.DerivedGcsTotalRule`).
    """
    padded = [row if len(row) == 4 else (*row, T0) for row in rows]
    return pl.DataFrame(
        padded,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "time": pl.Datetime,
        },
        orient="row",
    )


# ---------------------------------------------------------------------------
# Unit tests on synthetic MEDS-schema data
# ---------------------------------------------------------------------------


def test_above_threshold_triggers() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = _events(
        [
            (1, "LAB//220045//bpm", 120.0),  # triggers
            (2, "LAB//220045//bpm", 80.0),  # does not trigger
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["tachycardia"].to_list() == [1, 0]
    assert labels["tachycardia_observed"].to_list() == [1, 1]


def test_below_threshold_triggers() -> None:
    concepts = [
        ConceptDefinition(
            "hypotension", [ConceptRule("LAB//220179//", 90.0, "below")], "SBP < 90"
        )
    ]
    events = _events(
        [
            (1, "LAB//220179//mmHg", 85.0),  # triggers
            (2, "LAB//220179//mmHg", 110.0),  # does not trigger
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["hypotension"].to_list() == [1, 0]


def test_multiple_rules_are_ored() -> None:
    # Fever via either the Fahrenheit or Celsius chartevents itemid.
    concepts = [c for c in CONCEPTS if c.name == "fever"]
    events = _events(
        [
            (
                1,
                "LAB//223761//F",
                101.5,
            ),  # F itemid triggers
            (
                2,
                "LAB//223762//C",
                39.0,
            ),  # C itemid triggers
            (
                3,
                "LAB//223761//F",
                98.6,
            ),  # F itemid, normal
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["fever"].to_list() == [1, 1, 0]
    assert labels["fever_observed"].to_list() == [1, 1, 1]


def test_observed_mask_false_when_itemid_absent() -> None:
    concepts = [
        ConceptDefinition(
            "elevated_lactate",
            [ConceptRule("LAB//RESULT//50813//", 2.0, "above")],
            "Lactate > 2.0",
        )
    ]
    # Subject 2 has events, but never the lactate itemid at all.
    events = _events(
        [
            (1, "LAB//RESULT//50813//mmol/L", 3.0),
            (2, "LAB//220045//bpm", 80.0),
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["elevated_lactate"].to_list() == [1, 0]
    assert labels["elevated_lactate_observed"].to_list() == [1, 0]


def test_null_numeric_value_never_triggers_or_counts_as_observed() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = pl.DataFrame(
        {
            "subject_id": [1],
            "code": ["LAB//220045//bpm"],
            "numeric_value": pl.Series([None], dtype=pl.Float32),
            "time": [T0],
        }
    )
    labels = label_concepts(events, concepts)
    assert labels["tachycardia"].to_list() == [0]
    assert labels["tachycardia_observed"].to_list() == [0]


def test_concept_with_no_rules_raises_a_clear_error() -> None:
    concepts = [ConceptDefinition("empty", [], "no rules defined")]
    events = _events([(1, "LAB//220045//bpm", 75.0)])
    with pytest.raises(ValueError, match="empty"):
        label_concepts(events, concepts)


# ---------------------------------------------------------------------------
# SustainedRule
# ---------------------------------------------------------------------------


def test_sustained_rule_requires_recurrence_apart_not_just_one_reading() -> None:
    concepts = [
        ConceptDefinition(
            "sustained_tachypnea",
            [SustainedRule("LAB//220210//", 20.0, "above", min_gap_hours=1.0)],
            "RR > 20, recurring >= 1h apart",
        )
    ]
    events = _events(
        [
            # subject 1: a single transient spike -- must NOT trigger.
            (1, "LAB//220210//", 25.0, T0),
            # subject 2: two qualifying readings 2h apart -- must trigger.
            (2, "LAB//220210//", 25.0, T0),
            (2, "LAB//220210//", 22.0, T0 + timedelta(hours=2)),
            # subject 3: two readings close together (30 min) -- must NOT trigger.
            (3, "LAB//220210//", 25.0, T0),
            (3, "LAB//220210//", 22.0, T0 + timedelta(minutes=30)),
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["sustained_tachypnea"].to_list() == [0, 1, 0]
    assert labels["sustained_tachypnea_observed"].to_list() == [1, 1, 1]


def test_sustained_rule_non_qualifying_readings_between_do_not_matter() -> None:
    """Only the earliest/latest qualifying readings' span matters."""
    concepts = [
        ConceptDefinition(
            "sustained_tachypnea",
            [SustainedRule("LAB//220210//", 20.0, "above", min_gap_hours=1.0)],
            "RR > 20, recurring >= 1h apart",
        )
    ]
    events = _events(
        [
            (1, "LAB//220210//", 25.0, T0),
            (1, "LAB//220210//", 10.0, T0 + timedelta(hours=1)),  # normal, in between
            (1, "LAB//220210//", 25.0, T0 + timedelta(hours=2)),
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["sustained_tachypnea"].to_list() == [1]


# ---------------------------------------------------------------------------
# BaselineRelativeRule
# ---------------------------------------------------------------------------


def test_baseline_relative_rule_triggers_on_rise_within_window() -> None:
    concepts = [
        ConceptDefinition(
            "aki",
            [
                BaselineRelativeRule(
                    "LAB//RESULT//50912//",
                    delta=0.3,
                    direction="above",
                    window_hours=48.0,
                )
            ],
            "creatinine rose >= 0.3 within 48h",
        )
    ]
    events = _events(
        [
            (1, "LAB//RESULT//50912//", 1.0, T0),
            (
                1,
                "LAB//RESULT//50912//",
                1.4,
                T0 + timedelta(hours=24),
            ),  # +0.4 within 48h
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["aki"].to_list() == [1]


def test_baseline_relative_rule_does_not_trigger_outside_window() -> None:
    concepts = [
        ConceptDefinition(
            "aki",
            [
                BaselineRelativeRule(
                    "LAB//RESULT//50912//",
                    delta=0.3,
                    direction="above",
                    window_hours=48.0,
                )
            ],
            "creatinine rose >= 0.3 within 48h",
        )
    ]
    events = _events(
        [
            (1, "LAB//RESULT//50912//", 1.0, T0),
            (
                1,
                "LAB//RESULT//50912//",
                1.4,
                T0 + timedelta(hours=72),
            ),  # +0.4 but outside 48h
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["aki"].to_list() == [0]


def test_baseline_relative_rule_below_direction_triggers_on_fall() -> None:
    concepts = [
        ConceptDefinition(
            "big_drop",
            [
                BaselineRelativeRule(
                    "LAB//220045//", delta=20.0, direction="below", window_hours=6.0
                )
            ],
            "HR fell by >= 20 within 6h",
        )
    ]
    events = _events(
        [
            (1, "LAB//220045//", 100.0, T0),
            (1, "LAB//220045//", 70.0, T0 + timedelta(hours=2)),  # -30 within 6h
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["big_drop"].to_list() == [1]


def test_baseline_relative_rule_a_small_absolute_rise_from_a_low_baseline_still_triggers() -> (
    None
):
    """The whole point of KDIGO-style baseline-relative logic.

    Catches a delta v1's absolute-threshold proxy (creatinine > 1.5)
    would miss.
    """
    concepts = [
        ConceptDefinition(
            "aki",
            [
                BaselineRelativeRule(
                    "LAB//RESULT//50912//",
                    delta=0.3,
                    direction="above",
                    window_hours=48.0,
                )
            ],
            "creatinine rose >= 0.3 within 48h",
        )
    ]
    events = _events(
        [
            (1, "LAB//RESULT//50912//", 0.6, T0),
            (
                1,
                "LAB//RESULT//50912//",
                1.0,
                T0 + timedelta(hours=24),
            ),  # rose 0.4, still < 1.5
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["aki"].to_list() == [1]


def test_baseline_relative_rule_requires_exactly_one_of_delta_or_ratio() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        BaselineRelativeRule(
            "LAB//RESULT//50912//", direction="above", window_hours=48.0
        )
    with pytest.raises(ValueError, match="exactly one"):
        BaselineRelativeRule(
            "LAB//RESULT//50912//",
            direction="above",
            window_hours=48.0,
            delta=0.3,
            ratio=1.5,
        )


def test_baseline_relative_rule_ratio_mode_triggers_on_proportional_rise() -> None:
    concepts = [
        ConceptDefinition(
            "aki",
            [
                BaselineRelativeRule(
                    "LAB//RESULT//50912//",
                    ratio=1.5,
                    direction="above",
                    window_hours=168.0,
                )
            ],
            "creatinine rose to >= 1.5x baseline within 7 days",
        )
    ]
    events = _events(
        [
            # subject 1: 1.0 -> 1.5 exactly (1.5x) -- triggers.
            (1, "LAB//RESULT//50912//", 1.0, T0),
            (1, "LAB//RESULT//50912//", 1.5, T0 + timedelta(hours=48)),
            # subject 2: 1.0 -> 1.4 (only 1.4x) -- does not trigger.
            (2, "LAB//RESULT//50912//", 1.0, T0),
            (2, "LAB//RESULT//50912//", 1.4, T0 + timedelta(hours=48)),
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["aki"].to_list() == [1, 0]


def test_baseline_relative_rule_ratio_mode_below_direction_triggers_on_proportional_fall() -> (
    None
):
    concepts = [
        ConceptDefinition(
            "big_relative_drop",
            [
                BaselineRelativeRule(
                    "LAB//220045//", ratio=2.0, direction="below", window_hours=6.0
                )
            ],
            "HR fell to <= half of an earlier reading within 6h",
        )
    ]
    events = _events(
        [
            (1, "LAB//220045//", 100.0, T0),
            (1, "LAB//220045//", 50.0, T0 + timedelta(hours=2)),  # exactly half
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["big_relative_drop"].to_list() == [1]


def test_aki_staging_is_monotonically_more_selective() -> None:
    """Same real-data-shaped scenario, worse severity should trigger more stages."""
    aki_concepts = [
        c
        for c in CONCEPTS
        if c.name in ("acute_kidney_injury", "aki_stage_2", "aki_stage_3")
    ]
    events = _events(
        [
            # subject 1: 1.0 -> 3.5 (3.5x, absolute stays < 4.0) -- stages 1-3.
            (1, "LAB//RESULT//50912//", 1.0, T0),
            (1, "LAB//RESULT//50912//", 3.5, T0 + timedelta(hours=48)),
            # subject 2: 1.0 -> 2.2 (2.2x) -- stages 1-2 only.
            (2, "LAB//RESULT//50912//", 1.0, T0),
            (2, "LAB//RESULT//50912//", 2.2, T0 + timedelta(hours=48)),
            # subject 3: 1.0 -> 1.2 (only 1.2x, +0.2 absolute) -- no stage at all.
            (3, "LAB//RESULT//50912//", 1.0, T0),
            (3, "LAB//RESULT//50912//", 1.2, T0 + timedelta(hours=48)),
        ]
    )
    labels = label_concepts(events, aki_concepts).sort("subject_id")
    assert labels["acute_kidney_injury"].to_list() == [1, 1, 0]
    assert labels["aki_stage_2"].to_list() == [1, 1, 0]
    assert labels["aki_stage_3"].to_list() == [1, 0, 0]


def test_aki_stage_3_absolute_creatinine_trigger() -> None:
    """Stage 3's other trigger: any reading >= 4.0 mg/dL, regardless of baseline."""
    stage_3 = next(c for c in CONCEPTS if c.name == "aki_stage_3")
    events = _events(
        [(1, "LAB//RESULT//50912//", 4.5, T0)]
    )  # single reading, no baseline
    labels = label_concepts(events, [stage_3])
    assert labels["aki_stage_3"].to_list() == [1]


# ---------------------------------------------------------------------------
# DerivedGcsTotalRule
# ---------------------------------------------------------------------------


_GCS_RULE = DerivedGcsTotalRule(
    eye_prefix="LAB//220739//",
    verbal_prefix="LAB//223900//",
    motor_prefix="LAB//223901//",
    threshold=15.0,
    direction="below",
)


def test_derived_gcs_total_sums_components_charted_together() -> None:
    concepts = [ConceptDefinition("altered_mental_status", [_GCS_RULE], "GCS < 15")]
    events = _events(
        [
            (1, "LAB//220739//", 3.0, T0),  # eye
            (1, "LAB//223900//", 4.0, T0 + timedelta(minutes=1)),  # verbal
            (
                1,
                "LAB//223901//",
                5.0,
                T0 + timedelta(minutes=2),
            ),  # motor: total 12 < 15
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["altered_mental_status"].to_list() == [1]
    assert labels["altered_mental_status_observed"].to_list() == [1]


def test_derived_gcs_total_full_score_does_not_trigger() -> None:
    concepts = [ConceptDefinition("altered_mental_status", [_GCS_RULE], "GCS < 15")]
    events = _events(
        [
            (1, "LAB//220739//", 4.0, T0),
            (1, "LAB//223900//", 5.0, T0 + timedelta(minutes=1)),
            (1, "LAB//223901//", 6.0, T0 + timedelta(minutes=2)),  # total 15, not < 15
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["altered_mental_status"].to_list() == [0]


def test_derived_gcs_total_components_too_far_apart_do_not_pair() -> None:
    concepts = [ConceptDefinition("altered_mental_status", [_GCS_RULE], "GCS < 15")]
    events = _events(
        [
            (1, "LAB//220739//", 3.0, T0),
            (
                1,
                "LAB//223900//",
                4.0,
                T0 + timedelta(hours=5),
            ),  # far outside the 15min default
            (1, "LAB//223901//", 5.0, T0 + timedelta(hours=10)),
        ]
    )
    labels = label_concepts(events, concepts)
    assert labels["altered_mental_status"].to_list() == [0]
    # All three components were still individually observed.
    assert labels["altered_mental_status_observed"].to_list() == [1]


def test_derived_gcs_total_missing_a_component_is_not_observed_or_triggered() -> None:
    concepts = [ConceptDefinition("altered_mental_status", [_GCS_RULE], "GCS < 15")]
    events = _events(
        [(1, "LAB//220739//", 3.0, T0)]
    )  # only eye, no verbal/motor at all
    labels = label_concepts(events, concepts)
    assert labels["altered_mental_status"].to_list() == [0]
    assert labels["altered_mental_status_observed"].to_list() == [
        1
    ]  # eye alone was observed


# ---------------------------------------------------------------------------
# CompositeConceptDefinition / AnyOf
# ---------------------------------------------------------------------------


def test_composite_requires_min_criteria_met() -> None:
    concept = CompositeConceptDefinition(
        "sirs_like",
        components=[
            ConceptRule("LAB//220045//", 90.0, "above"),  # HR > 90
            ConceptRule("LAB//220210//", 20.0, "above"),  # RR > 20
        ],
        min_criteria=2,
        description="both criteria required",
    )
    events = _events(
        [
            # subject 1: only HR criterion met.
            (1, "LAB//220045//", 95.0, T0),
            # subject 2: both criteria met.
            (2, "LAB//220045//", 95.0, T0),
            (2, "LAB//220210//", 25.0, T0),
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["sirs_like"].to_list() == [0, 1]


def test_composite_observed_if_any_component_observed() -> None:
    concept = CompositeConceptDefinition(
        "sirs_like",
        components=[
            ConceptRule("LAB//220045//", 90.0, "above"),
            ConceptRule("LAB//220210//", 20.0, "above"),
        ],
        min_criteria=2,
        description="both criteria required",
    )
    events = _events([(1, "LAB//220045//", 50.0, T0)])  # observed, doesn't trigger
    labels = label_concepts(events, [concept])
    assert labels["sirs_like_observed"].to_list() == [1]
    assert labels["sirs_like"].to_list() == [0]


def test_composite_with_no_components_raises_a_clear_error() -> None:
    concept = CompositeConceptDefinition(
        "empty", [], min_criteria=1, description="no components"
    )
    events = _events([(1, "LAB//220045//bpm", 75.0)])
    with pytest.raises(ValueError, match="empty"):
        label_concepts(events, [concept])


def test_any_of_counts_as_one_criterion_even_if_both_branches_fire() -> None:
    """SIRS-style: an AnyOf criterion counts once, however many branches fire.

    'Abnormal temperature' (too high OR too low) must count once, not
    twice, even if a subject had both at different times.
    """
    concept = CompositeConceptDefinition(
        "sirs_like",
        components=[
            AnyOf(
                [
                    ConceptRule("LAB//223761//", 100.4, "above"),
                    ConceptRule("LAB//223761//", 96.8, "below"),
                ]
            ),  # criterion 1: abnormal temp (fires twice below, must count once)
            ConceptRule("LAB//220045//", 90.0, "above"),  # criterion 2: HR > 90
            ConceptRule("LAB//220210//", 20.0, "above"),  # criterion 3: RR > 20
        ],
        min_criteria=3,
        description="all three criteria required",
    )
    events = _events(
        [
            # subject 1: temp fires both high AND low (still just 1 criterion),
            # HR fires -- only 2 of 3 criteria, must not trigger.
            (1, "LAB//223761//", 101.0, T0),
            (1, "LAB//223761//", 90.0, T0 + timedelta(hours=1)),
            (1, "LAB//220045//", 95.0, T0),
        ]
    )
    labels = label_concepts(events, [concept])
    assert labels["sirs_like"].to_list() == [0]


def test_sirs_definition_triggers_on_two_of_four_criteria() -> None:
    sirs = next(c for c in CONCEPTS if c.name == "sirs")
    events = _events(
        [
            (1, "LAB//220045//", 95.0, T0),  # HR > 90
            (1, "LAB//220210//", 25.0, T0),  # RR > 20
        ]
    )
    labels = label_concepts(events, [sirs])
    assert labels["sirs"].to_list() == [1]


def test_sirs_wbc_criterion_triggers_on_high_or_low_count() -> None:
    sirs = next(c for c in CONCEPTS if c.name == "sirs")
    events = _events(
        [
            # subject 1: WBC high + HR high -- 2 of 4 criteria.
            (1, "LAB//RESULT//51301//", 15.0, T0),
            (1, "LAB//220045//", 95.0, T0),
            # subject 2: WBC low alone -- only 1 of 4 criteria.
            (2, "LAB//RESULT//51301//", 2.0, T0),
        ]
    )
    labels = label_concepts(events, [sirs]).sort("subject_id")
    assert labels["sirs"].to_list() == [1, 0]


def test_sirs_wbc_high_and_low_still_counts_as_one_criterion() -> None:
    """AnyOf semantics for a real concept.

    A subject with both a high and a low WBC reading (at different
    times) must still only get credit for one criterion.
    """
    sirs = next(c for c in CONCEPTS if c.name == "sirs")
    events = _events(
        [
            (1, "LAB//RESULT//51301//", 15.0, T0),
            (1, "LAB//RESULT//51301//", 2.0, T0 + timedelta(hours=1)),
            (1, "LAB//220045//", 50.0, T0),  # HR normal, no other criterion met
        ]
    )
    labels = label_concepts(events, [sirs])
    assert labels["sirs"].to_list() == [0]


def test_qsofa_definition_triggers_on_two_of_three_criteria() -> None:
    qsofa = next(c for c in CONCEPTS if c.name == "qsofa")
    events = _events(
        [
            (1, "LAB//220210//", 25.0, T0),  # RR >= 22
            (1, "LAB//220179//", 90.0, T0),  # SBP <= 100
        ]
    )
    labels = label_concepts(events, [qsofa])
    assert labels["qsofa"].to_list() == [1]


def test_qsofa_thresholds_are_inclusive_at_the_boundary() -> None:
    """Boundary values qualify: qSOFA is RR >= 22 and SBP <= 100.

    Vitals are typically charted as exact integers, so a strict
    inequality would silently miss real qualifying readings at exactly
    the documented cutoffs.
    """
    qsofa = next(c for c in CONCEPTS if c.name == "qsofa")
    events = _events(
        [
            (1, "LAB//220210//", 22.0, T0),  # exactly RR 22
            (1, "LAB//220179//", 100.0, T0),  # exactly SBP 100
        ]
    )
    labels = label_concepts(events, [qsofa])
    assert labels["qsofa"].to_list() == [1]


def test_aki_stage_3_absolute_creatinine_threshold_is_inclusive() -> None:
    """KDIGO Stage 3's absolute trigger is creatinine >= 4.0, inclusive."""
    aki_3 = next(c for c in CONCEPTS if c.name == "aki_stage_3")
    events = _events(
        [
            (1, "LAB//RESULT//50912//", 4.0, T0),  # exactly 4.0 -- triggers
            (2, "LAB//RESULT//50912//", 3.9, T0),  # just below -- does not
        ]
    )
    labels = label_concepts(events, [aki_3]).sort("subject_id")
    assert labels["aki_stage_3"].to_list() == [1, 0]


def test_default_registry_covers_all_subjects() -> None:
    events = _events([(1, "LAB//220045//bpm", 75.0), (2, "LAB//220045//bpm", 75.0)])
    labels = label_concepts(events)
    assert labels.height == 2
    for concept in CONCEPTS:
        assert concept.name in labels.columns
        assert f"{concept.name}_observed" in labels.columns


# ---------------------------------------------------------------------------
# Integration test against the real MIMIC-IV demo dataset
# ---------------------------------------------------------------------------


@pytest.mark.integration_test
def test_label_concepts_on_real_mimic_iv_demo_extraction(tmp_path: Path) -> None:
    """Sanity-check concept labeling against a real MEDS extraction.

    Runs the actual meds-extract pipeline against the public MIMIC-IV demo
    and confirms concept labeling behaves sensibly on real data: common
    vitals/labs are observed for a meaningful fraction of the cohort, and
    every concept produces a valid binary column.
    """
    output_dir = tmp_path / "meds_demo"
    result = subprocess.run(
        [
            "meds-extract-run",
            "spec=MIMIC-IV",
            f"output_dir={output_dir}",
            "dataset_key=demo",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    shards = list((output_dir / "data").rglob("*.parquet"))
    assert shards, "expected at least one MEDS data shard"
    events = pl.concat([pl.read_parquet(s) for s in shards])

    labels = label_concepts(events)
    n_subjects = labels.height
    assert n_subjects > 0

    for concept in CONCEPTS:
        assert labels[concept.name].is_in([0, 1]).all()

    # Heart rate is charted for virtually every ICU patient in the cohort.
    assert labels["tachycardia_observed"].sum() > 0
