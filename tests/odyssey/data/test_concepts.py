"""Tests for rule-derived concept labels."""

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from odyssey.data import code_mapping
from odyssey.data.concepts import (
    _GLUCOSE,
    ANTIBIOTIC_ROUTE_EXCLUDE,
    CANONICAL_CONCEPTS,
    CONCEPTS,
    AnyOf,
    BaselineRelativeRule,
    CodeOccurrenceRule,
    CompositeConceptDefinition,
    ConceptDefinition,
    ConceptRule,
    DerivedGcsTotalRule,
    LoincBaselineRelative,
    LoincThreshold,
    SustainedRule,
    _expand_rule,
    concepts_for_source,
    label_concepts,
    label_concepts_by_visit,
)


T0 = datetime(2024, 1, 1, 0, 0)


_EventRow = tuple[int, str, float] | tuple[int, str, float, datetime]


def _events(rows: list[_EventRow]) -> pl.DataFrame:
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


def test_derived_gcs_first_time_is_when_all_components_are_known() -> None:
    """The trigger stamp is the LAST paired component's time, not the eye's.

    Regression test for the 2026-08-30 fix: "nearest" pairing can pull a
    verbal/motor reading charted minutes AFTER the eye reading, and
    stamping the trigger at the eye time put that many minutes of future
    information into first_times.
    """
    concepts = [ConceptDefinition("altered_mental_status", [_GCS_RULE], "GCS < 15")]
    events = _events(
        [
            (1, "LAB//220739//", 3.0, T0),  # eye
            (1, "LAB//223901//", 5.0, T0 + timedelta(minutes=5)),  # motor
            (1, "LAB//223900//", 4.0, T0 + timedelta(minutes=10)),  # verbal, last
        ]
    )
    labels = label_concepts(events, concepts, include_first_time=True)
    assert labels["altered_mental_status"].to_list() == [1]
    assert labels["altered_mental_status_first_time"].to_list() == [
        T0 + timedelta(minutes=10)
    ]


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


def test_composite_observed_needs_min_criteria_components_observed() -> None:
    """Observed only when >= min_criteria components were measurable.

    A subject with only one of two required criteria ever measured can
    never reach min_criteria=2 no matter what the readings said --
    supervising them as a negative (the pre-2026-08-30 any-component
    mask) was label noise, not a genuine negative.
    """
    concept = CompositeConceptDefinition(
        "sirs_like",
        components=[
            ConceptRule("LAB//220045//", 90.0, "above"),
            ConceptRule("LAB//220210//", 20.0, "above"),
        ],
        min_criteria=2,
        description="both criteria required",
    )
    # only one component ever measured: structurally unassessable
    events = _events([(1, "LAB//220045//", 50.0, T0)])
    labels = label_concepts(events, [concept])
    assert labels["sirs_like_observed"].to_list() == [0]
    assert labels["sirs_like"].to_list() == [0]
    # both components measured, neither triggering: a real negative
    events = _events([(1, "LAB//220045//", 50.0, T0), (1, "LAB//220210//", 10.0, T0)])
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


def test_code_occurrence_rule_triggers_on_matching_code() -> None:
    concept = ConceptDefinition(
        "on_vasopressors_test",
        [CodeOccurrenceRule(r"norepinephrine|vasopressin")],
        "test",
    )
    events = _events(
        [
            (1, "MEDICATION//Norepinephrine 8 mg/250 mL", 1.0),  # triggers
            (2, "MEDICATION//Acetaminophen 325 mg", 1.0),  # observed, negative
            (3, "LAB//220045//", 80.0),  # no medication data: unobserved
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["on_vasopressors_test"].to_list() == [1, 0, 0]
    assert labels["on_vasopressors_test_observed"].to_list() == [1, 1, 0]


def test_code_occurrence_rule_is_case_insensitive() -> None:
    concept = ConceptDefinition("vaso", [CodeOccurrenceRule(r"norepinephrine")], "test")
    events = _events([(1, "MEDICATION//NOREPINEPHRINE", 1.0)])
    labels = label_concepts(events, [concept])
    assert labels["vaso"].to_list() == [1]


def test_code_occurrence_rule_matches_text_value_when_opted_in() -> None:
    # eICU charts infusion drug names in text_value under a bare
    # INFUSION_DRUG code; the rule must reach them when opted in.
    concept = ConceptDefinition(
        "vaso",
        [CodeOccurrenceRule(r"vasopressin", match_text_value=True)],
        "test",
    )
    events = _events([(1, "INFUSION_DRUG", 1.0), (2, "INFUSION_DRUG", 1.0)])
    events = events.with_columns(
        pl.Series("text_value", ["Vasopressin 40 units", None])
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["vaso"].to_list() == [1, 0]
    assert labels["vaso_observed"].to_list() == [1, 1]


def test_on_vasopressors_is_registered_and_labels_cleanly() -> None:
    vaso = next(c for c in CONCEPTS if c.name == "on_vasopressors")
    events = _events(
        [
            (1, "MEDICATION//Levophed", 1.0),
            (2, "MEDICATION//Metoprolol Tartrate 25 mg", 1.0),
        ]
    )
    labels = label_concepts(events, [vaso]).sort("subject_id")
    assert labels["on_vasopressors"].to_list() == [1, 0]


def _events_with_visits(rows):  # noqa: ANN001, ANN202
    """rows: (subject_id, code, value, hadm_id_or_None)."""
    return pl.DataFrame(
        [(r[0], r[1], r[2], T0, r[3]) for r in rows],
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "time": pl.Datetime,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def test_label_concepts_by_visit_scopes_evidence_to_each_visit() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = _events_with_visits(
        [
            (1, "LAB//220045//bpm", 130.0, 10),  # visit 10: triggers
            (1, "LAB//220045//bpm", 80.0, 11),  # visit 11: normal
            (1, "LAB//220045//bpm", 140.0, None),  # solo event: excluded
        ]
    )
    labels = label_concepts_by_visit(events, concepts).sort("hadm_id")
    assert labels.height == 2  # solo event contributes no visit row
    assert labels["subject_id"].to_list() == [1, 1]
    assert labels["hadm_id"].to_list() == [10, 11]
    assert labels["tachycardia"].to_list() == [1, 0]
    assert labels["tachycardia_observed"].to_list() == [1, 1]


def test_label_concepts_by_visit_keeps_visits_of_different_subjects_apart() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = _events_with_visits(
        [
            (1, "LAB//220045//bpm", 130.0, 10),
            (2, "LAB//220045//bpm", 80.0, 20),
        ]
    )
    labels = label_concepts_by_visit(events, concepts).sort("subject_id")
    assert labels["tachycardia"].to_list() == [1, 0]


def test_label_concepts_by_visit_baseline_rule_does_not_cross_visits() -> None:
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
            "creatinine rose >= 0.3 in 48h",
        )
    ]
    events = pl.DataFrame(
        [
            # baseline in visit 10, rise in visit 11, within 48h of each
            # other: must NOT trigger, since the baseline is visit-scoped.
            (1, "LAB//RESULT//50912//", 1.0, T0, 10),
            (1, "LAB//RESULT//50912//", 1.6, T0 + timedelta(hours=10), 11),
        ],
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "time": pl.Datetime,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )
    labels = label_concepts_by_visit(events, concepts)
    assert labels["aki"].to_list() == [0, 0]


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


# ---------------------------------------------------------------------------
# Canonical registry expansion (concepts_for_source)
# ---------------------------------------------------------------------------


def test_mimic_expansion_matches_the_historical_registry() -> None:
    """CONCEPTS is the mimic_iv expansion, pinned against known structure.

    A regression guard for the canonical layer: expanding the LOINC-keyed
    registry for MIMIC-IV must reproduce exactly the prefix-keyed
    definitions the models to date were trained and evaluated with.
    """
    concepts = concepts_for_source("mimic_iv")
    assert [c.name for c in concepts] == [c.name for c in CONCEPTS]
    assert concepts == CONCEPTS

    by_name = {c.name: c for c in concepts}
    hypotension = by_name["hypotension"]
    assert isinstance(hypotension, ConceptDefinition)
    assert [r.code_prefix for r in hypotension.rules] == [
        "LAB//220179//",
        "LAB//220050//",
    ]
    fever = by_name["fever"]
    assert isinstance(fever, ConceptDefinition)
    assert {(r.code_prefix, r.threshold) for r in fever.rules} == {
        ("LAB//223761//", 100.4),  # Fahrenheit itemid gets the F threshold
        ("LAB//223762//", 38.0),  # Celsius itemid gets the C threshold
    }
    qsofa = by_name["qsofa"]
    assert isinstance(qsofa, CompositeConceptDefinition)
    assert len(qsofa.components) == 3
    assert any(isinstance(comp, DerivedGcsTotalRule) for comp in qsofa.components)


def test_eicu_expansion_translates_every_signal_it_can() -> None:
    concepts = concepts_for_source("eicu")
    by_name = {c.name: c for c in concepts}

    tachy = by_name["tachycardia"]
    assert isinstance(tachy, ConceptDefinition)
    assert [r.code_prefix for r in tachy.rules] == ["VITALS//PERIODIC//HEARTRATE"]

    # eICU charts temperature in Celsius only: one rule, the C threshold.
    fever = by_name["fever"]
    assert isinstance(fever, ConceptDefinition)
    assert [(r.code_prefix, r.threshold) for r in fever.rules] == [
        ("VITALS//PERIODIC//TEMPERATURE", 38.0)
    ]

    # Both SBP measurements exist, so hypotension ORs both prefixes.
    hypotension = by_name["hypotension"]
    assert isinstance(hypotension, ConceptDefinition)
    assert {r.code_prefix for r in hypotension.rules} == {
        "VITALS//APERIODIC//BP//NONINVASIVE_SYSTOLIC",
        "VITALS//PERIODIC//BP//SYSTEMIC_SYSTOLIC",
    }


def test_eicu_qsofa_has_all_three_criteria_since_spec_v2() -> None:
    """EICU qSOFA has RR, SBP and the derived GCS total since spec v2.

    Spec v2 extracts nurseCharting GCS as GCS//EYES|VERBAL|MOTOR.
    """
    qsofa = next(c for c in concepts_for_source("eicu") if c.name == "qsofa")
    assert isinstance(qsofa, CompositeConceptDefinition)
    assert len(qsofa.components) == 3
    assert qsofa.min_criteria == 2
    gcs = [comp for comp in qsofa.components if isinstance(comp, DerivedGcsTotalRule)]
    assert len(gcs) == 1
    assert gcs[0].eye_prefix == "GCS//EYES"


def test_unmapped_criterion_is_dropped_but_the_composite_survives(monkeypatch) -> None:
    """A source without a GCS mapping drops that criterion, composite survives.

    (eICU spec v1 had no GCS.) qSOFA keeps its other two criteria, which
    still satisfy min_criteria=2 -- the translation entry 06 did by hand.
    """
    without_gcs = {
        k: v for k, v in code_mapping.EICU_TO_LOINC.items() if not k.startswith("GCS//")
    }
    monkeypatch.setitem(code_mapping._SOURCE_TABLES, "eicu", without_gcs)
    qsofa = next(c for c in concepts_for_source("eicu") if c.name == "qsofa")
    assert isinstance(qsofa, CompositeConceptDefinition)
    assert len(qsofa.components) == 2
    assert qsofa.min_criteria == 2
    assert not any(isinstance(comp, DerivedGcsTotalRule) for comp in qsofa.components)


def test_multi_prefix_criterion_counts_once_inside_a_composite() -> None:
    """A LOINC resolving to several prefixes must OR inside AnyOf.

    If each prefix became its own criterion, a source charting SBP two
    ways could satisfy qSOFA's min_criteria=2 from low blood pressure
    alone -- a correctness property, not a style choice.
    """
    for source in ("mimic_iv", "eicu"):
        qsofa = next(c for c in concepts_for_source(source) if c.name == "qsofa")
        assert isinstance(qsofa, CompositeConceptDefinition)
        sbp_components = [comp for comp in qsofa.components if isinstance(comp, AnyOf)]
        assert len(sbp_components) == 1
        assert len(sbp_components[0].rules) == 2


def test_eicu_expansion_labels_real_shaped_events() -> None:
    """End-to-end: eICU-shaped events label correctly under the expansion."""
    concepts = [
        c for c in concepts_for_source("eicu") if c.name in ("fever", "tachycardia")
    ]
    events = _events(
        [
            (1, "VITALS//PERIODIC//TEMPERATURE", 38.5),
            (1, "VITALS//PERIODIC//HEARTRATE", 80.0),
            (2, "VITALS//PERIODIC//TEMPERATURE", 37.0),
            (2, "VITALS//PERIODIC//HEARTRATE", 120.0),
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["fever"].to_list() == [1, 0]
    assert labels["tachycardia"].to_list() == [0, 1]


# ---------------------------------------------------------------------------
# include_first_time: when did the concept become satisfied?
# ---------------------------------------------------------------------------


def test_first_time_is_the_earliest_qualifying_reading() -> None:
    concepts = [c for c in CONCEPTS if c.name == "tachycardia"]
    events = _events(
        [
            (1, "LAB//220045//bpm", 80.0, T0),
            (1, "LAB//220045//bpm", 120.0, T0 + timedelta(hours=5)),
            (1, "LAB//220045//bpm", 130.0, T0 + timedelta(hours=9)),
            (2, "LAB//220045//bpm", 70.0, T0),
        ]
    )
    labels = label_concepts(events, concepts, include_first_time=True).sort(
        "subject_id"
    )
    assert labels["tachycardia"].to_list() == [1, 0]
    assert labels["tachycardia_first_time"].to_list() == [
        T0 + timedelta(hours=5),
        None,
    ]


def test_sustained_first_time_is_when_the_gap_is_first_met() -> None:
    concepts = [c for c in CONCEPTS if c.name == "sustained_tachypnea"]
    events = _events(
        [
            (1, "LAB//220210//insp/min", 25.0, T0),
            (1, "LAB//220210//insp/min", 25.0, T0 + timedelta(minutes=30)),
            (1, "LAB//220210//insp/min", 25.0, T0 + timedelta(hours=2)),
            (1, "LAB//220210//insp/min", 25.0, T0 + timedelta(hours=3)),
        ]
    )
    labels = label_concepts(events, concepts, include_first_time=True)
    # Not at 30 min (span 0.5h < 1h); satisfied at the 2h reading.
    assert labels["sustained_tachypnea_first_time"][0] == T0 + timedelta(hours=2)


def test_composite_first_time_is_when_min_criteria_is_reached() -> None:
    sirs = next(c for c in CONCEPTS if c.name == "sirs")
    events = _events(
        [
            (1, "LAB//220045//bpm", 95.0, T0 + timedelta(hours=1)),  # HR > 90
            (1, "LAB//220210//insp/min", 25.0, T0 + timedelta(hours=6)),  # RR > 20
            (1, "LAB//223762//C", 39.0, T0 + timedelta(hours=9)),  # fever
        ]
    )
    labels = label_concepts(events, [sirs], include_first_time=True)
    assert labels["sirs"][0] == 1
    # Second criterion fires at 6h: that is when SIRS (>= 2 of 4) is met.
    assert labels["sirs_first_time"][0] == T0 + timedelta(hours=6)


def test_first_time_respects_visit_scoping() -> None:
    concepts = [c for c in CONCEPTS if c.name == "tachycardia"]
    events = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "code": ["LAB//220045//bpm"] * 4,
            "numeric_value": [120.0, 80.0, 80.0, 130.0],
            "time": [
                T0,
                T0 + timedelta(hours=1),
                T0 + timedelta(days=30),
                T0 + timedelta(days=30, hours=4),
            ],
            "hadm_id": [10, 10, 20, 20],
        }
    )
    labels = label_concepts_by_visit(events, concepts, include_first_time=True).sort(
        "hadm_id"
    )
    assert labels["tachycardia_first_time"].to_list() == [
        T0,
        T0 + timedelta(days=30, hours=4),
    ]


# ---------------------------------------------------------------------------
# GEMINI expansion
# ---------------------------------------------------------------------------


def test_gemini_source_resolves_all_but_gcs_dependent_concepts() -> None:
    """15 of 16 canonical concepts resolve on GEMINI; only GCS-dependent parts drop.

    Mirrors the eICU situation (no GCS source yet): qsofa keeps 2 of 3
    criteria, everything else resolves fully -- including sirs with all
    four criteria, which never resolved before the GEMINI mapping table
    existed (the smoke runs logged 20+ dropped-concept warnings).
    """
    names = sorted(c.name for c in concepts_for_source("gemini"))
    assert "sirs" in names and "qsofa" in names
    assert "acute_kidney_injury" in names and "aki_stage_3" in names
    assert len(names) == 15


def test_gemini_v3_resolves_the_lab_panel_concepts() -> None:
    """25 of 29 v3 concepts resolve once the electrolyte/CBC/INR ids are mapped.

    The four that stay out need signals the datacut does not chart:
    urine output (oliguria; none anywhere in the datacut), FiO2/PaO2
    (hypoxemic respiratory failure; FiO2 is unmapped free text), mean
    arterial pressure (shock; only systolic and diastolic are charted),
    and sepsis3's SOFA components.
    """
    names = {c.name for c in concepts_for_source("gemini", task_set="v3")}
    assert len(names) == 25
    assert {
        "hyperkalemia",
        "hypokalemia",
        "hyponatremia",
        "hypernatremia",
        "hypoglycemia",
        "hyperglycemia",
        "anemia",
        "thrombocytopenia",
        "coagulopathy",
        "metabolic_acidosis",
    } <= names
    assert names.isdisjoint(
        {"oliguria", "hypoxemic_respiratory_failure", "shock", "sepsis3"}
    )
    # metabolic acidosis keeps its bicarbonate criterion; the pH one drops
    acidosis = next(
        c
        for c in concepts_for_source("gemini", task_set="v3")
        if c.name == "metabolic_acidosis"
    )
    prefixes = {r.code_prefix for r in acidosis.rules}
    assert prefixes == {"LAB//3016293//"}


def test_gemini_shaped_events_label_the_si_lab_concepts() -> None:
    """SI values cross the SI cutoffs (mmol/L glucose, g/L hemoglobin)."""
    wanted = (
        "hyponatremia",
        "hyperkalemia",
        "hypoglycemia",
        "anemia",
        "thrombocytopenia",
        "coagulopathy",
    )
    concepts = [
        c for c in concepts_for_source("gemini", task_set="v3") if c.name in wanted
    ]
    events = _events(
        [
            (1, "LAB//3019550//mmol/l", 125.0),  # sodium 125 -> hyponatremia
            (1, "LAB//3023103//mmol/l", 6.1),  # potassium 6.1 -> hyperkalemia
            (
                1,
                "LAB//3013826//mmol/l",
                3.0,
            ),  # glucose 3.0 mmol/L (54 mg/dL) -> hypoglycemia
            (1, "LAB//3000963//g/l", 65.0),  # hemoglobin 65 g/L (6.5 g/dL) -> anemia
            (1, "LAB//3007461//x10e9/l", 80.0),  # platelets 80 -> thrombocytopenia
            (1, "LAB//3032080//inr", 2.1),  # INR 2.1 -> coagulopathy
            (2, "LAB//3019550//mmol/l", 140.0),
            (2, "LAB//3023103//mmol/l", 4.0),
            (
                2,
                "LAB//3013826//mmol/l",
                6.0,
            ),  # 108 mg/dL: would be hypoglycemic under the mg/dL cutoff
            (
                2,
                "LAB//3000963//g/l",
                130.0,
            ),  # 13 g/dL: would be anemic under the g/dL cutoff
            (2, "LAB//3007461//x10e9/l", 250.0),
            (2, "LAB//3032080//inr", 1.0),
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    for name in wanted:
        assert labels[name].to_list() == [1, 0], name


def test_aki_delta_is_unit_converted_per_source() -> None:
    """KDIGO's 0.3 mg/dL rise becomes 26.5 umol/L on GEMINI, unchanged elsewhere.

    Applying the mg/dL delta to umol/L values would make stage-1 AKI
    trigger on ordinary assay noise (0.3 umol/L is ~1/90th of the real
    criterion), so the unit_deltas override is load-bearing, not
    cosmetic.
    """
    delta_rules = [
        rr
        for c in CANONICAL_CONCEPTS
        for r in (getattr(c, "rules", None) or getattr(c, "components", []) or [])
        for rr in (getattr(r, "rules", None) or [r])
        if isinstance(rr, LoincBaselineRelative) and rr.delta == 0.3
    ]
    assert len(delta_rules) == 1
    rule = delta_rules[0]
    assert [e.delta for e in _expand_rule(rule, "mimic_iv")] == [0.3]
    assert [e.delta for e in _expand_rule(rule, "eicu")] == [0.3]
    assert [e.delta for e in _expand_rule(rule, "gemini")] == [26.5]


# ---------------------------------------------------------------------------
# v3: electrolyte/metabolic/hematologic concept widening (Track B item 11)
# ---------------------------------------------------------------------------

_V3_NEW_NAMES = (
    "hyperkalemia",
    "hypokalemia",
    "hyponatremia",
    "hypernatremia",
    "hypoglycemia",
    "hyperglycemia",
    "anemia",
    "thrombocytopenia",
    "coagulopathy",
    "metabolic_acidosis",
    "shock",
)
# Added later, in the same task set, but derived from SOFA signals rather
# than a plain LOINC threshold -- so, like sepsis3, they only expand on
# sources with SOFA's non-LOINC ingredients (see the eicu test below).
_V3_DERIVED_NAMES = ("hypoxemic_respiratory_failure", "oliguria")


def test_v1_and_v2_expansions_are_unchanged_by_the_v3_addition() -> None:
    """Adding v3 must not alter what v1/v2 runs would have trained/evaluated with."""
    assert concepts_for_source("mimic_iv", task_set="v1") == CONCEPTS
    assert [c.name for c in concepts_for_source("mimic_iv", task_set="v1")] == [
        "tachycardia",
        "bradycardia",
        "hypotension",
        "hypertension",
        "hypoxia",
        "fever",
        "hypothermia",
        "elevated_lactate",
        "sustained_tachypnea",
        "acute_kidney_injury",
        "aki_stage_2",
        "aki_stage_3",
        "sirs",
        "qsofa",
        "on_vasopressors",
    ]
    assert [c.name for c in concepts_for_source("mimic_iv", task_set="v2")] == [
        "tachycardia",
        "bradycardia",
        "hypotension",
        "hypertension",
        "hypoxia",
        "fever",
        "hypothermia",
        "elevated_lactate",
        "sustained_tachypnea",
        "acute_kidney_injury",
        "aki_stage_2",
        "aki_stage_3",
        "sirs",
        "qsofa",
        "on_vasopressors",
        "sepsis3",
    ]


def test_v3_adds_exactly_the_new_concepts_on_top_of_v2() -> None:
    v2_names = {c.name for c in concepts_for_source("mimic_iv", task_set="v2")}
    v3_names = [c.name for c in concepts_for_source("mimic_iv", task_set="v3")]
    assert set(v3_names) - v2_names == set(_V3_NEW_NAMES) | set(_V3_DERIVED_NAMES)
    assert len(v3_names) == len(set(v3_names))  # no accidental duplicate names


def _v3_concept(name: str, source: str = "mimic_iv") -> ConceptDefinition:
    concepts = concepts_for_source(source, task_set="v3")
    by_name = {c.name: c for c in concepts}
    concept = by_name[name]
    assert isinstance(concept, ConceptDefinition)
    return concept


def test_hyperkalemia_triggers_above_5_5() -> None:
    concept = _v3_concept("hyperkalemia")
    events = _events(
        [
            (1, "LAB//RESULT//50971//mEq/L", 5.6),  # triggers
            (2, "LAB//RESULT//50971//mEq/L", 4.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["hyperkalemia"].to_list() == [1, 0]


def test_hypokalemia_triggers_below_3_0() -> None:
    concept = _v3_concept("hypokalemia")
    events = _events(
        [
            (1, "LAB//RESULT//50971//mEq/L", 2.9),  # triggers
            (2, "LAB//RESULT//50971//mEq/L", 4.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["hypokalemia"].to_list() == [1, 0]


def test_hyponatremia_triggers_below_130() -> None:
    concept = _v3_concept("hyponatremia")
    events = _events(
        [
            (1, "LAB//RESULT//50983//mEq/L", 128.0),  # triggers
            (2, "LAB//RESULT//50983//mEq/L", 140.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["hyponatremia"].to_list() == [1, 0]


def test_hypernatremia_triggers_above_150() -> None:
    concept = _v3_concept("hypernatremia")
    events = _events(
        [
            (1, "LAB//RESULT//50983//mEq/L", 151.0),  # triggers
            (2, "LAB//RESULT//50983//mEq/L", 140.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["hypernatremia"].to_list() == [1, 0]


def test_hypoglycemia_triggers_below_70() -> None:
    concept = _v3_concept("hypoglycemia")
    events = _events(
        [
            (1, "LAB//RESULT//50931//mg/dL", 65.0),  # triggers
            (2, "LAB//RESULT//50931//mg/dL", 100.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["hypoglycemia"].to_list() == [1, 0]


def test_hyperglycemia_triggers_above_250() -> None:
    concept = _v3_concept("hyperglycemia")
    events = _events(
        [
            (1, "LAB//RESULT//50931//mg/dL", 260.0),  # triggers
            (2, "LAB//RESULT//50931//mg/dL", 140.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["hyperglycemia"].to_list() == [1, 0]


def test_anemia_triggers_below_7() -> None:
    concept = _v3_concept("anemia")
    events = _events(
        [
            (1, "LAB//RESULT//51222//g/dL", 6.5),  # triggers
            (2, "LAB//RESULT//51222//g/dL", 12.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["anemia"].to_list() == [1, 0]


def test_thrombocytopenia_triggers_below_100() -> None:
    concept = _v3_concept("thrombocytopenia")
    events = _events(
        [
            (1, "LAB//RESULT//51265//K/uL", 80.0),  # triggers
            (2, "LAB//RESULT//51265//K/uL", 250.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["thrombocytopenia"].to_list() == [1, 0]


def test_coagulopathy_triggers_above_1_5() -> None:
    concept = _v3_concept("coagulopathy")
    events = _events(
        [
            (1, "LAB//RESULT//51237//", 1.8),  # triggers
            (2, "LAB//RESULT//51237//", 1.0),  # does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["coagulopathy"].to_list() == [1, 0]


def test_metabolic_acidosis_triggers_on_either_bicarb_or_ph() -> None:
    concept = _v3_concept("metabolic_acidosis")
    events = _events(
        [
            (1, "LAB//RESULT//50882//mEq/L", 15.0),  # low bicarb triggers
            (2, "LAB//RESULT//50820//units", 7.2),  # low pH triggers
            (3, "LAB//RESULT//50882//mEq/L", 24.0),  # normal, does not
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["metabolic_acidosis"].to_list() == [1, 1, 0]


def test_shock_requires_recurring_low_map_not_one_reading() -> None:
    concept = _v3_concept("shock")
    events = _events(
        [
            # subject 1: two low-MAP readings 2h apart -- sustained, triggers
            (1, "LAB//220181//mmHg", 60.0, T0),
            (1, "LAB//220181//mmHg", 58.0, T0 + timedelta(hours=2)),
            # subject 2: a single low reading -- does not trigger
            (2, "LAB//220181//mmHg", 60.0, T0),
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["shock"].to_list() == [1, 0]


def test_shock_pools_cuff_and_arterial_map_for_the_recurrence_check() -> None:
    """A low MAP on the cuff and later on the arterial line IS sustained.

    Regression test for the 2026-08-30 fix: a multi-prefix LoincSustained
    used to expand to one SustainedRule per prefix, so recurrence was
    checked within each charting route separately and a cross-modality
    recurrence (clinically the same sustained hypotension) never
    triggered. The expansion is now a single pooled rule.
    """
    concept = _v3_concept("shock")
    assert len(concept.rules) == 1
    rule = concept.rules[0]
    assert isinstance(rule, SustainedRule)
    assert rule.extra_prefixes  # both MAP prefixes pooled into one rule
    events = _events(
        [
            # low MAP on the cuff, then on the arterial line 2h later
            (1, "LAB//220181//mmHg", 60.0, T0),
            (1, "LAB//220052//mmHg", 58.0, T0 + timedelta(hours=2)),
            # same two readings within one hour: gap not met, no trigger
            (2, "LAB//220181//mmHg", 60.0, T0),
            (2, "LAB//220052//mmHg", 58.0, T0 + timedelta(minutes=30)),
        ]
    )
    labels = label_concepts(events, [concept]).sort("subject_id")
    assert labels["shock"].to_list() == [1, 0]


def test_shock_is_not_defined_in_terms_of_on_vasopressors() -> None:
    """The dropped-redundancy check.

    shock's rules never reference a vasopressor code pattern -- see its
    CanonicalConcept description for why the original vasopressor-OR
    clause was left out.
    """
    concept = _v3_concept("shock")
    assert all(not isinstance(r, CodeOccurrenceRule) for r in concept.rules)


def test_v3_eicu_expansion_resolves_every_new_concept() -> None:
    """Confirm none of the new concepts are dropped for eicu.

    Every v3 LOINC-threshold concept is already mapped for eicu
    (code_mapping.py), so none of those should be dropped there. The
    SOFA-derived ones are dropped, like sepsis3: they need ventilation
    codes and urine-output mappings eicu's spec does not provide.
    """
    eicu_names = {c.name for c in concepts_for_source("eicu", task_set="v3")}
    for name in _V3_NEW_NAMES:
        assert name in eicu_names, f"{name} unexpectedly dropped for eicu"
    for name in _V3_DERIVED_NAMES:
        assert name not in eicu_names, f"{name} unexpectedly present for eicu"


# ---------------------------------------------------------------------------
# Sepsis-3 antibiotic route exclusion (real-data finding,
# research_journal/experiments/44_real_data_checks.html)
# ---------------------------------------------------------------------------


def test_antibiotic_route_exclude_misses_real_mimic_route_abbreviations() -> None:
    """ANTIBIOTIC_ROUTE_EXCLUDE must cover MIMIC-IV's route abbreviations.

    Originally it matched route *words* only -- confirmed on a real held-out shard: 174
    Mupirocin (topical/nasal MRSA-decolonization ointment, not systemic
    antibiotic therapy) rows in one shard alone -- 139 null route, 35
    route='NU' (nasal), 3 route='PR' (rectal) -- all pass through uncaught
    as systemic antibiotic starts feeding Sepsis-3's suspected-infection
    criterion (odyssey.data.concepts._sepsis3_ids, the
    ``~pl.col(route_col)...str.contains(route_exclude)`` filter). Every
    other drug in the real top-40 antibiotic-regex match list was a
    genuine systemic antibacterial -- this route-abbreviation gap is the
    one concrete false-positive source found.

    Fix proposal (not applied here -- flagging for review, not isolated
    enough to fix blind): add MIMIC-IV's actual route abbreviations (at
    least NU, PR; audit the rest of d_items.csv's route vocabulary rather
    than guessing the full set) to ANTIBIOTIC_ROUTE_EXCLUDE, alongside the
    existing word-form matches. Separately, note the null-route case is
    currently treated as NOT excluded (fill_null("") never matches the
    regex) -- worth an explicit policy decision (assume systemic if route
    is unknown, or exclude conservatively) rather than the current
    incidental default.
    """
    pattern = re.compile(ANTIBIOTIC_ROUTE_EXCLUDE, re.IGNORECASE)
    # Fixed (6cbff95 follow-up): abbreviations audited on the real orders table.
    assert pattern.search("NU"), (
        "'NU' (MIMIC-IV's real nasal-route abbreviation) should be excluded "
        "but the regex only matches the word 'nasal', not this abbreviation"
    )
    assert pattern.search("PR"), (
        "'PR' (MIMIC-IV's real rectal-route abbreviation) should be excluded "
        "but the regex only matches the word 'rectal', not this abbreviation"
    )
    # sanity: a genuine systemic route must NOT be excluded (the fix must
    # not overcorrect and start dropping real IV/PO antibiotic starts).
    assert not pattern.search("IV")
    assert not pattern.search("PO")


# ---------------------------------------------------------------------------
# Per-unit thresholds: a conventional-unit default plus SI overrides
# ---------------------------------------------------------------------------


def test_loinc_threshold_needs_a_default_or_per_unit_thresholds() -> None:
    with pytest.raises(ValueError, match="threshold, unit_thresholds, or both"):
        LoincThreshold(("2345-7",), "below")
    # either alone is fine, and so are both together
    LoincThreshold(("2345-7",), "below", 70.0)
    LoincThreshold(("8310-5",), "above", unit_thresholds=(("C", 38.0),))
    LoincThreshold(("2345-7",), "below", 70.0, unit_thresholds=(("mmol/L", 3.9),))


def test_untagged_prefix_takes_the_default_and_tagged_prefix_its_own_unit() -> None:
    """Glucose: mg/dL on the US sources, mmol/L wherever the prefix is tagged."""
    rule = LoincThreshold(_GLUCOSE, "below", 70.0, unit_thresholds=(("mmol/L", 3.9),))
    assert {r.threshold for r in _expand_rule(rule, "mimic_iv")} == {70.0}
    assert {r.threshold for r in _expand_rule(rule, "eicu")} == {70.0}


def test_tagged_prefix_without_an_entry_is_an_error_not_a_fallthrough() -> None:
    """Creatinine is tagged umol/L on GEMINI; a mg/dL-only rule must refuse it."""
    rule = LoincThreshold(("2160-0",), "above", 4.0)
    with pytest.raises(ValueError, match="umol/L"):
        _expand_rule(rule, "gemini")
    # the temperature form (per-unit only) still refuses an untagged prefix
    only_c = LoincThreshold(("8310-5",), "above", unit_thresholds=(("C", 38.0),))
    with pytest.raises(ValueError, match="unit tag 'F'"):
        _expand_rule(only_c, "mimic_iv")


def test_hypoglycemia_hyperglycemia_and_anemia_carry_si_cutoffs() -> None:
    by_name = {c.name: c for c in CANONICAL_CONCEPTS}
    glucose_low = by_name["hypoglycemia"].rules[0]
    glucose_high = by_name["hyperglycemia"].rules[0]
    hemoglobin = by_name["anemia"].rules[0]
    assert (glucose_low.threshold, glucose_low.unit_thresholds) == (
        70.0,
        (("mmol/L", 3.9),),
    )
    assert (glucose_high.threshold, glucose_high.unit_thresholds) == (
        250.0,
        (("mmol/L", 13.9),),
    )
    assert (hemoglobin.threshold, hemoglobin.unit_thresholds) == (7.0, (("g/L", 70.0),))


def test_stage_3_creatinine_cutoff_is_unit_converted_on_gemini() -> None:
    """KDIGO's absolute 4.0 mg/dL is 353.7 umol/L where creatinine is SI.

    Before per-unit thresholds this rule compared GEMINI's umol/L values
    against 4.0 and was true for essentially every creatinine result.
    """
    by_name = {c.name: c for c in CANONICAL_CONCEPTS}
    absolute = [
        r
        for r in by_name["aki_stage_3"].rules
        if isinstance(r, LoincThreshold) and r.threshold == 4.0
    ]
    assert len(absolute) == 1
    assert [e.threshold for e in _expand_rule(absolute[0], "mimic_iv")] == [4.0]
    assert [e.threshold for e in _expand_rule(absolute[0], "gemini")] == [353.7]
