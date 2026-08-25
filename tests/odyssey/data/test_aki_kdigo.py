"""KDIGO AKI staging beyond creatinine: the RRT and urine-output legs.

The creatinine legs (baseline-relative rises, the absolute >= 4.0 mg/dL
trigger, and stage ordering) are tested in ``test_concepts.py``; the plain
SOFA renal component and the absolute-volume ``oliguria`` concept in
``test_sepsis3_tasks.py``. This file covers what those two do not:

- renal-replacement-therapy initiation as an automatic Stage 3
  (:data:`~odyssey.data.concepts.RRT_ITEMIDS`), including mimic-code's two
  deliberate exclusions;
- KDIGO's weight-normalized urine-output rate
  (:func:`~odyssey.data.sofa.urine_output_rate`) at each of the three
  windows the stages use, its daily/admission weight fallback, and the
  weight-free anuria branch.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl
import pytest

from odyssey.data.concepts import (
    RRT_CODE_PATTERN,
    RRT_ITEMIDS,
    AnyConceptDefinition,
    CodeOccurrenceRule,
    ConceptDefinition,
    DerivedUrineRateRule,
    concepts_for_source,
    label_concepts,
)
from odyssey.data.sofa import urine_output_24h, urine_output_rate


T0 = datetime(2024, 1, 1, 0, 0)
H = timedelta(hours=1)

# MIMIC-IV code prefixes (code_mapping's mimic_iv table / SOFA_SOURCE_CONFIG)
CREAT = "LAB//RESULT//50912//mg/dL"
URINE = "SUBJECT_FLUID_OUTPUT//226559//mL"  # Foley
URINE_VOID = "SUBJECT_FLUID_OUTPUT//226560//mL"  # a second collection route
DAILY_WEIGHT = "LAB//224639//kg"
ADMISSION_WEIGHT = "LAB//226512//kg"
# mimic-code rrt.sql: active dialysis, and its two exclusions
DIALYSIS_CATHETER = "PROCEDURE//START//224270"
CRRT_FILTER_CHANGE = "PROCEDURE//START//225436"

_Row = Tuple[int, str, Optional[float], datetime]


def _events(rows: Sequence[_Row]) -> pl.DataFrame:
    """Build a synthetic MEDS events frame; every row carries a real time.

    Same shape as ``test_concepts.py``'s ``_events``, with the time
    mandatory: every rule exercised here is time-aware (trailing urine
    windows, asof-joined weights, first-occurrence onsets).
    """
    return pl.DataFrame(
        list(rows),
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "numeric_value": pl.Float64,
            "time": pl.Datetime("us"),
        },
        orient="row",
    )


def _aki(*names: str) -> List[AnyConceptDefinition]:
    """Return the named AKI concepts, as the mimic_iv expansion builds them."""
    return [c for c in concepts_for_source("mimic_iv") if c.name in names]


_ALL_STAGES = ("acute_kidney_injury", "aki_stage_2", "aki_stage_3")


def _labels(rows: Sequence[_Row], *names: str) -> Dict[int, Dict[str, object]]:
    """Label ``rows`` with the named concepts, keyed by subject id."""
    labeled = label_concepts(
        _events(rows), _aki(*(names or _ALL_STAGES)), include_first_time=True
    )
    return {row["subject_id"]: row for row in labeled.to_dicts()}


def _rates(
    rows: Sequence[_Row], *, window_hours: float, weight_normalized: bool = True
) -> Dict[datetime, float]:
    """``urine_output_rate`` for the single subject in ``rows``, time -> value."""
    frame = urine_output_rate(
        _events(rows),
        key="subject_id",
        window_hours=window_hours,
        weight_normalized=weight_normalized,
    )
    return dict(zip(frame["time"].to_list(), frame["value"].to_list()))


# ---------------------------------------------------------------------------
# Renal replacement therapy: an automatic Stage 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("itemid", RRT_ITEMIDS)
def test_rrt_fires_on_each_dialysis_itemid_individually(itemid: str) -> None:
    """Every one of mimic-code's six dialysis_active item ids stages a 3.

    Parametrized rather than spot-checked: a regex alternation that only
    really matched its first branch would pass a single-item test.
    """
    rows = [
        (1, CREAT, 1.0, T0),  # creatinine charted and flat: no creatinine trigger
        (1, f"PROCEDURE//START//{itemid}", None, T0 + 5 * H),
    ]
    row = _labels(rows)[1]
    assert row["aki_stage_3"] == 1
    assert row["aki_stage_3_first_time"] == T0 + 5 * H
    # RRT is a Stage 3 criterion only; the lower stages have their own.
    assert row["acute_kidney_injury"] == 0 and row["aki_stage_2"] == 0


@pytest.mark.parametrize("code", [DIALYSIS_CATHETER, CRRT_FILTER_CHANGE])
def test_rrt_excludes_catheter_placement_and_filter_change(code: str) -> None:
    """mimic-code's two dialysis_active = 0 rows must not stage anything.

    224270 is line placement (no therapy yet) and 225436 is maintenance on
    an already-running circuit, whose therapy start has its own row.
    """
    row = _labels([(1, CREAT, 1.0, T0), (1, code, None, T0 + 5 * H)])[1]
    assert row["aki_stage_3"] == 0
    # ... and the subject is still *observed*: procedure data exists, so
    # "no dialysis" is a real negative rather than missingness.
    assert row["aki_stage_3_observed"] == 1


def test_rrt_pattern_is_anchored_to_procedure_start_and_a_whole_itemid() -> None:
    """The alternation must not leak into END rows or longer item ids."""
    rows = [
        (1, CREAT, 1.0, T0),
        (1, "PROCEDURE//END//225441", None, T0 + H),  # the episode ending
        (2, CREAT, 1.0, T0),
        (2, "PROCEDURE//START//2254411", None, T0 + H),  # a longer id, not ours
        (3, CREAT, 1.0, T0),
        (3, "PROCEDURE//START//225441//1", None, T0 + H),  # a suffixed segment
    ]
    labeled = _labels(rows)
    assert labeled[1]["aki_stage_3"] == 0
    assert labeled[2]["aki_stage_3"] == 0
    assert labeled[3]["aki_stage_3"] == 1  # a real start row with a trailing segment


def test_rrt_onset_is_the_first_dialysis_event_not_every_one() -> None:
    rows = [
        (1, CREAT, 1.0, T0),
        (1, "PROCEDURE//START//225802", None, T0 + 10 * H),  # CRRT starts
        (1, "PROCEDURE//START//225803", None, T0 + 20 * H),  # switched to CVVHD
        (1, "PROCEDURE//START//225802", None, T0 + 30 * H),  # and back
    ]
    row = _labels(rows)[1]
    assert row["aki_stage_3"] == 1
    assert row["aki_stage_3_first_time"] == T0 + 10 * H


def test_rrt_ors_with_the_creatinine_legs_without_double_firing() -> None:
    """A patient already Stage 3 by creatinine keeps the earlier onset.

    The concept is one binary label whose rules are OR-ed, so a second
    satisfied criterion cannot change the label or move the onset later.
    """
    rows = [
        (1, CREAT, 5.0, T0 + 2 * H),  # >= 4.0: Stage 3 by creatinine
        (1, "PROCEDURE//START//225441", None, T0 + 40 * H),  # dialysis later
    ]
    row = _labels(rows)[1]
    assert row["aki_stage_3"] == 1
    assert row["aki_stage_3_first_time"] == T0 + 2 * H
    # The mirror case: dialysis first, creatinine crossing later.
    rows = [
        (2, "PROCEDURE//START//225441", None, T0 + 2 * H),
        (2, CREAT, 5.0, T0 + 40 * H),
    ]
    row = _labels(rows)[2]
    assert row["aki_stage_3"] == 1 and row["aki_stage_3_first_time"] == T0 + 2 * H


def test_rrt_rule_is_the_only_occurrence_rule_and_only_on_stage_3() -> None:
    """Stage 3 carries the RRT rule; the lower stages must not.

    KDIGO's automatic-Stage-3 fact is about dialysis, and it is expressed
    as one rule inside aki_stage_3 rather than as a separate named
    concept the bottleneck would need its own head for.
    """
    by_name = {c.name: c for c in _aki(*_ALL_STAGES)}
    stage_3 = by_name["aki_stage_3"]
    assert isinstance(stage_3, ConceptDefinition)
    occurrence = [r for r in stage_3.rules if isinstance(r, CodeOccurrenceRule)]
    assert [r.code_pattern for r in occurrence] == [RRT_CODE_PATTERN]
    assert occurrence[0].observed_families == ("PROCEDURE",)
    for name in ("acute_kidney_injury", "aki_stage_2"):
        concept = by_name[name]
        assert isinstance(concept, ConceptDefinition)
        assert not any(isinstance(r, CodeOccurrenceRule) for r in concept.rules)


# ---------------------------------------------------------------------------
# Urine-output rate: mL/kg/h arithmetic over KDIGO's windows
# ---------------------------------------------------------------------------

# 60 kg, a 6 h burst of 60 mL/h then a trickle of 6 mL/h: each window
# length sees a different rate, so a window mix-up cannot pass silently.
_BURST_THEN_TRICKLE: List[_Row] = (
    [(1, DAILY_WEIGHT, 60.0, T0)]
    + [(1, URINE, 60.0, T0 + k * H) for k in range(1, 7)]
    + [(1, URINE, 6.0, T0 + k * H) for k in range(7, 25)]
)


def test_urine_rate_arithmetic_at_each_kdigo_window() -> None:
    six = _rates(_BURST_THEN_TRICKLE, window_hours=6.0)
    twelve = _rates(_BURST_THEN_TRICKLE, window_hours=12.0)
    twentyfour = _rates(_BURST_THEN_TRICKLE, window_hours=24.0)
    # 6 h window at 6 h: 6 x 60 mL / 60 kg / 6 h = 1.0 mL/kg/h
    assert six[T0 + 6 * H] == pytest.approx(1.0)
    # 6 h window at 24 h: 6 x 6 mL / 60 kg / 6 h = 0.1
    assert six[T0 + 24 * H] == pytest.approx(0.1)
    # 12 h window at 24 h: 12 x 6 mL / 60 kg / 12 h = 0.1
    assert twelve[T0 + 24 * H] == pytest.approx(0.1)
    # 24 h window at 24 h: (360 + 108) mL / 60 kg / 24 h = 0.325
    assert twentyfour[T0 + 24 * H] == pytest.approx(0.325)


def test_urine_rate_excludes_windows_without_a_full_window_of_record() -> None:
    """Same partial-window exclusion as the 24 h absolute-volume form.

    A window narrower than its length sums less urine only because less
    time has passed, which would read as oliguria at every admission.
    """
    six = _rates(_BURST_THEN_TRICKLE, window_hours=6.0)
    twentyfour = _rates(_BURST_THEN_TRICKLE, window_hours=24.0)
    assert min(six) == T0 + 6 * H
    assert list(twentyfour) == [T0 + 24 * H]  # nothing earlier is assessable


def test_urine_rate_sums_every_collection_route() -> None:
    """Foley plus void, resolved through the LOINC layer, not one itemid."""
    rows: List[_Row] = [(1, DAILY_WEIGHT, 50.0, T0)] + [
        row
        for k in range(1, 7)
        for row in ((1, URINE, 20.0, T0 + k * H), (1, URINE_VOID, 5.0, T0 + k * H))
    ]
    # 6 x 25 mL / 50 kg / 6 h = 0.5
    assert _rates(rows, window_hours=6.0)[T0 + 6 * H] == pytest.approx(0.5)


def test_urine_output_24h_is_the_absolute_volume_special_case() -> None:
    """The pre-existing callers (SOFA renal, oliguria) must be unchanged."""
    frame = urine_output_24h(_events(_BURST_THEN_TRICKLE), key="subject_id")
    assert frame.columns == ["subject_id", "time", "value"]
    assert dict(zip(frame["time"].to_list(), frame["value"].to_list())) == {
        T0 + 24 * H: 468.0
    }
    assert frame.equals(
        urine_output_rate(
            _events(_BURST_THEN_TRICKLE),
            key="subject_id",
            window_hours=24.0,
            weight_normalized=False,
        )
    )


def test_only_the_weighted_form_needs_a_source_weight_config() -> None:
    """The absolute-volume form must keep working where weights are unknown.

    eICU has no :data:`~odyssey.data.sofa.SOFA_SOURCE_CONFIG` entry, and
    ``urine_output_24h`` delegates here with ``weight_normalized=False``:
    that path must not reach for a config it has no need of. Asking for
    the weighted form on such a source is a programming error and says
    so, the same way every other SOFA entry point does.
    """
    rows: List[_Row] = [(1, "URINE_OUTPUT//mL", 10.0, T0 + k * H) for k in range(0, 30)]
    absolute = urine_output_rate(
        _events(rows),
        source="eicu",
        key="subject_id",
        window_hours=24.0,
        weight_normalized=False,
    )
    # Hours 24-29 are the assessable ones; each 24 h window is left-open,
    # so it holds 24 of the hourly 10 mL readings, not 25.
    assert absolute["value"].to_list() == [240.0] * 6
    with pytest.raises(KeyError):
        urine_output_rate(_events(rows), source="eicu", key="subject_id")


# ---------------------------------------------------------------------------
# Weight: the fallback order, and abstaining when there is none
# ---------------------------------------------------------------------------


def test_daily_weight_is_preferred_over_admission_weight_once_charted() -> None:
    """Daily weight is the current weight; admission weight is the fallback.

    Subject 1 has both: windows ending before the first daily weight use
    the admission weight (a *later* reading must never be pulled
    backwards by the asof join), windows after it use the daily weight.
    """
    urine: List[Tuple[int, str, Optional[float], datetime]] = [
        (1, URINE, 30.0, T0 + k * H) for k in range(1, 13)
    ]
    both = [
        (1, ADMISSION_WEIGHT, 100.0, T0),
        (1, DAILY_WEIGHT, 50.0, T0 + 8 * H),
        *urine,
    ]
    rates = _rates(both, window_hours=6.0)
    # at 6 h the only weight charted yet is the admission one: 180/100/6
    assert rates[T0 + 6 * H] == pytest.approx(0.3)
    # at 12 h the daily weight (charted at 8 h) applies: 180/50/6
    assert rates[T0 + 12 * H] == pytest.approx(0.6)

    admission_only = [(2, ADMISSION_WEIGHT, 100.0, T0)] + [
        (2, code, value, time) for (_, code, value, time) in urine
    ]
    daily_only = [(3, DAILY_WEIGHT, 50.0, T0)] + [
        (3, code, value, time) for (_, code, value, time) in urine
    ]
    assert _rates(admission_only, window_hours=6.0)[T0 + 12 * H] == pytest.approx(0.3)
    assert _rates(daily_only, window_hours=6.0)[T0 + 12 * H] == pytest.approx(0.6)


def test_a_window_with_no_weight_at_all_is_dropped_not_defaulted() -> None:
    """No weight anywhere: the rate criterion is unassessable, full stop.

    Weight coverage in the real MIMIC-IV extraction is ~10-17% of
    subjects, so this is the common case, and defaulting a weight would
    manufacture gold-standard labels out of nothing.
    """
    rows: List[_Row] = [(1, URINE, 1.0, T0 + k * H) for k in range(1, 13)]
    assert _rates(rows, window_hours=6.0) == {}
    # The same rows without normalization are perfectly assessable. The
    # window origin is the key's first *event* (here the first urine
    # reading, at T0 + 1h), so the earliest full 6 h window ends at
    # T0 + 7h -- the same rule urine_output_24h has always applied.
    assert _rates(rows, window_hours=6.0, weight_normalized=False)[
        T0 + 7 * H
    ] == pytest.approx(6.0)
    # a weight charted only AFTER every window is no help either
    late = [*rows, (1, DAILY_WEIGHT, 70.0, T0 + 20 * H)]
    assert _rates(late, window_hours=6.0) == {}


def test_a_non_positive_charted_weight_is_dropped_not_divided_by() -> None:
    rows: List[_Row] = [(1, DAILY_WEIGHT, 0.0, T0)] + [
        (1, URINE, 1.0, T0 + k * H) for k in range(1, 13)
    ]
    assert _rates(rows, window_hours=6.0) == {}


def test_urine_rate_is_empty_without_urine_readings() -> None:
    """No urine charted at all: no rows, whatever the weight coverage."""
    rows: List[_Row] = [(1, DAILY_WEIGHT, 70.0, T0), (1, CREAT, 1.0, T0 + H)]
    assert _rates(rows, window_hours=6.0) == {}
    assert _rates(rows, window_hours=12.0, weight_normalized=False) == {}


def test_unassessable_urine_rate_is_unobserved_not_a_negative() -> None:
    """The observed-mask convention, isolated to the urine-rate rule.

    A urine-rate-only concept must report subjects it cannot score as
    unobserved (masked out of supervision), never as label 0.
    """
    concept = ConceptDefinition(
        "urine_stage_1",
        [
            DerivedUrineRateRule(
                threshold=0.5, direction="below", window_hours=6.0, source="mimic_iv"
            )
        ],
        "test",
    )
    rows: List[_Row] = [
        # subject 1: urine, no weight -- not assessable
        *[(1, URINE, 1.0, T0 + k * H) for k in range(1, 8)],
        # subject 2: urine and a weight -- assessable, and oliguric
        (2, DAILY_WEIGHT, 60.0, T0),
        *[(2, URINE, 1.0, T0 + k * H) for k in range(1, 8)],
        # subject 3: urine and a weight, voiding well -- assessable negative
        (3, DAILY_WEIGHT, 60.0, T0),
        *[(3, URINE, 100.0, T0 + k * H) for k in range(1, 8)],
    ]
    labeled = label_concepts(_events(rows), [concept]).sort("subject_id")
    assert labeled["urine_stage_1"].to_list() == [0, 1, 0]
    assert labeled["urine_stage_1_observed"].to_list() == [0, 1, 1]


# ---------------------------------------------------------------------------
# Thresholds: strict "<" on the rate, inclusive "<= 0" on anuria
# ---------------------------------------------------------------------------


def test_the_rate_threshold_is_strict_exactly_at_kdigos_number() -> None:
    """KDIGO says "less than 0.5 mL/kg/h", so exactly 0.5 does not stage.

    The mirror of the ``at_or_above`` convention on Stage 3's creatinine
    >= 4.0 trigger: the direction spelled in the rule is the direction
    the criterion is written with.
    """
    rows: List[_Row] = [
        (1, DAILY_WEIGHT, 50.0, T0),  # 25 mL/h / 50 kg = exactly 0.5
        *[(1, URINE, 25.0, T0 + k * H) for k in range(1, 8)],
        (2, DAILY_WEIGHT, 50.0, T0),  # 24 mL/h / 50 kg = 0.48
        *[(2, URINE, 24.0, T0 + k * H) for k in range(1, 8)],
    ]
    labeled = _labels(rows, "acute_kidney_injury", "aki_stage_2")
    assert labeled[1]["acute_kidney_injury"] == 0
    assert labeled[2]["acute_kidney_injury"] == 1
    assert labeled[2]["acute_kidney_injury_first_time"] == T0 + 6 * H


def test_stage_2_needs_twelve_hours_of_the_same_rate_stage_1_needs_six() -> None:
    """The stages differ only in window length at 0.5 mL/kg/h."""
    rows: List[_Row] = [
        # subject 1: oliguric for 8 h only -- Stage 1's 6 h window fires,
        # Stage 2's 12 h window never has 12 h of oliguria to average over.
        (1, DAILY_WEIGHT, 100.0, T0),
        *[(1, URINE, 10.0, T0 + k * H) for k in range(1, 9)],  # 0.1 mL/kg/h
        *[(1, URINE, 300.0, T0 + k * H) for k in range(9, 21)],  # 3.0 mL/kg/h
        # subject 2: oliguric throughout -- both stages fire.
        (2, DAILY_WEIGHT, 100.0, T0),
        *[(2, URINE, 10.0, T0 + k * H) for k in range(1, 21)],
    ]
    labeled = _labels(rows, "acute_kidney_injury", "aki_stage_2")
    assert labeled[1]["acute_kidney_injury"] == 1
    assert labeled[1]["acute_kidney_injury_first_time"] == T0 + 6 * H
    assert labeled[1]["aki_stage_2"] == 0
    assert labeled[2]["acute_kidney_injury"] == 1 and labeled[2]["aki_stage_2"] == 1
    assert labeled[2]["aki_stage_2_first_time"] == T0 + 12 * H


def test_stage_3_rate_leg_is_below_point_three_over_twenty_four_hours() -> None:
    rows: List[_Row] = [
        # subject 1: 0.2 mL/kg/h for a full day -- Stage 3 by rate
        (1, DAILY_WEIGHT, 100.0, T0),
        *[(1, URINE, 20.0, T0 + k * H) for k in range(1, 26)],
        # subject 2: 0.4 mL/kg/h -- Stages 1 and 2, but not 3's 0.3 rate
        (2, DAILY_WEIGHT, 100.0, T0),
        *[(2, URINE, 40.0, T0 + k * H) for k in range(1, 26)],
    ]
    labeled = _labels(rows)
    assert labeled[1]["aki_stage_3"] == 1
    assert labeled[1]["aki_stage_3_first_time"] == T0 + 24 * H
    assert labeled[2]["acute_kidney_injury"] == 1 and labeled[2]["aki_stage_2"] == 1
    assert labeled[2]["aki_stage_3"] == 0


def test_anuria_stages_a_3_with_no_weight_reading_anywhere() -> None:
    """0 mL over 12 h is 0 mL at any body weight, so it needs no weight.

    This is the leg that keeps Stage 3's urine criterion assessable for
    the ~85% of subjects with no charted weight.
    """
    rows: List[_Row] = [
        # subject 1: anuric, no weight charted at all
        *[(1, URINE, 0.0, T0 + k * H) for k in range(1, 14)],
        # subject 2: a trickle, not anuria; no weight, so the rate legs
        # cannot rescue it either -- Stage 3 must stay 0, not "unknown = yes"
        *[(2, URINE, 1.0, T0 + k * H) for k in range(1, 14)],
    ]
    labeled = _labels(rows, "aki_stage_3")
    assert labeled[1]["aki_stage_3"] == 1
    # First event is the urine reading at T0 + 1h, so the first full 12 h
    # window ends at T0 + 13h (the partial-window rule, not an off-by-one).
    assert labeled[1]["aki_stage_3_first_time"] == T0 + 13 * H
    assert labeled[2]["aki_stage_3"] == 0
    assert labeled[2]["aki_stage_3_observed"] == 1  # urine was assessable


def test_anuria_needs_a_full_twelve_hours_of_record() -> None:
    """Six hours of no urine is not yet 12 h of anuria."""
    rows: List[_Row] = [(1, URINE, 0.0, T0 + k * H) for k in range(1, 7)]
    assert _labels(rows, "aki_stage_3")[1]["aki_stage_3"] == 0


# ---------------------------------------------------------------------------
# Whole-concept integration and per-source expansion
# ---------------------------------------------------------------------------


def test_all_three_legs_compose_as_an_or_across_a_mixed_cohort() -> None:
    """Creatinine-only, urine-only, RRT-only and combined, in one pass."""
    rows: List[_Row] = [
        # 1: creatinine only, 1.0 -> 3.5 (3.5x): stages 1, 2 and 3
        (1, CREAT, 1.0, T0),
        (1, CREAT, 3.5, T0 + 48 * H),
        # 2: urine only, 0.2 mL/kg/h for a day on a charted weight: 1, 2, 3
        (2, DAILY_WEIGHT, 100.0, T0),
        *[(2, URINE, 20.0, T0 + k * H) for k in range(1, 26)],
        # 3: RRT only, creatinine flat and urine fine: 3 alone
        (3, CREAT, 1.0, T0),
        (3, ADMISSION_WEIGHT, 100.0, T0),
        *[(3, URINE, 200.0, T0 + k * H) for k in range(1, 26)],
        (3, "PROCEDURE//START//225809", None, T0 + 30 * H),
        # 4: creatinine +0.3 in 48 h (stage 1) and anuric 12 h (stage 3),
        # no weight: the rate legs abstain, the anuria leg does not
        (4, CREAT, 1.0, T0),
        (4, CREAT, 1.35, T0 + 24 * H),
        *[(4, URINE, 0.0, T0 + k * H) for k in range(1, 14)],
        # 5: nothing wrong anywhere -- an observed negative on all three
        (5, CREAT, 1.0, T0),
        (5, DAILY_WEIGHT, 80.0, T0),
        *[(5, URINE, 100.0, T0 + k * H) for k in range(1, 26)],
    ]
    labeled = _labels(rows)
    assert [labeled[i]["acute_kidney_injury"] for i in range(1, 6)] == [1, 1, 0, 1, 0]
    assert [labeled[i]["aki_stage_2"] for i in range(1, 6)] == [1, 1, 0, 0, 0]
    assert [labeled[i]["aki_stage_3"] for i in range(1, 6)] == [1, 1, 1, 1, 0]
    for i in range(1, 6):
        for name in _ALL_STAGES:
            assert labeled[i][f"{name}_observed"] == 1, (i, name)
    # Stage 3 does not require Stage 1 to have fired: the stages stay
    # independent binary concepts, as they were before this change.
    assert labeled[3]["acute_kidney_injury"] == 0 and labeled[3]["aki_stage_3"] == 1


def test_urine_legs_expand_only_where_the_weight_item_ids_are_known() -> None:
    """Like sepsis3: the derived legs need a source config, the rest travels.

    eICU has no :data:`~odyssey.data.sofa.SOFA_SOURCE_CONFIG` entry, so
    its AKI concepts keep the creatinine legs (and the harmless
    occurrence rule, which is source-agnostic) but not the urine ones,
    and the concept is never dropped for want of them.
    """
    stages = {
        source: {
            c.name: c for c in concepts_for_source(source) if c.name in _ALL_STAGES
        }
        for source in ("mimic_iv", "eicu", "gemini")
    }
    for source in ("mimic_iv", "eicu", "gemini"):
        assert set(stages[source]) == set(_ALL_STAGES)

    def _urine_rules(source: str, name: str) -> List[DerivedUrineRateRule]:
        concept = stages[source][name]
        assert isinstance(concept, ConceptDefinition)
        return [r for r in concept.rules if isinstance(r, DerivedUrineRateRule)]

    assert [
        (r.threshold, r.window_hours, r.weight_normalized)
        for r in _urine_rules("mimic_iv", "aki_stage_3")
    ] == [(0.3, 24.0, True), (0.0, 12.0, False)]
    assert [
        (r.threshold, r.window_hours) for r in _urine_rules("mimic_iv", "aki_stage_2")
    ] == [(0.5, 12.0)]
    assert [
        (r.threshold, r.window_hours)
        for r in _urine_rules("mimic_iv", "acute_kidney_injury")
    ] == [(0.5, 6.0)]
    for name in _ALL_STAGES:
        assert _urine_rules("eicu", name) == []
        assert _urine_rules("gemini", name) == []
