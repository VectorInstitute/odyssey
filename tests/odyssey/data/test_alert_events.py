"""Tests for odyssey.data.alert_events's guard branches.

event_times has two ways to be misconfigured: a concept-defined alert
called without the concept_first_times frame it needs, and an AlertEvent
that names neither a concept nor a code_prefix at all (a malformed
registry entry). Neither failure path had ever been exercised.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import polars as pl
import pytest

from odyssey.data.alert_events import (
    PROBE_EVENTS,
    AlertEvent,
    _next_visit_onsets,
    event_times,
    origin_hours,
    visit_envelope,
)


T0 = datetime(2024, 1, 1)

_EventRow = Tuple[int, str, datetime, Optional[float], int]


def _events() -> pl.DataFrame:
    rows: List[_EventRow] = [
        (1, "LAB//220045//bpm", T0, 80.0, 1001),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=1), 80.0, 1001),
    ]
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


def test_concept_alert_without_concept_first_times_raises() -> None:
    """A concept-defined alert needs the shared first-time frame; None must fail."""
    alert = AlertEvent("acute_kidney_injury", concept="acute_kidney_injury")

    with pytest.raises(ValueError, match="acute_kidney_injury"):
        event_times(_events(), alert, concept_first_times=None)


def test_alert_with_neither_concept_nor_code_prefix_raises() -> None:
    """A malformed registry entry (no onset rule at all) must fail, not no-op."""
    alert = AlertEvent("malformed_alert")

    with pytest.raises(ValueError, match="malformed_alert"):
        event_times(_events(), alert)


def test_next_visit_onsets_finds_the_genuine_next_admission_not_a_later_one() -> None:
    """A stray late row on a visit must not hide a real, earlier readmission.

    Real-data finding (research_journal/experiments/44_real_data_checks.html):
    on a held-out MIMIC-IV shard, readmission_30d disagreed with the raw
    admissions table in 9/859 comparable visits, all traced to this one
    mechanism: ``_next_visit_onsets`` anchors its "find the next admission
    after this visit" search on the visit's *last timed event*
    (``time.max()`` grouped by hadm_id), not on the true discharge time. A
    late-charted/late-attributed row on hadm_id 100 -- happens on real data,
    e.g. a lab result finalized long after discharge -- pushes that anchor
    past a real back-to-back readmission (hadm_id 101), so the asof-forward
    search skips right over it and lands on a much later admission (hadm_id
    102) instead.

    This is reduced to the minimal synthetic case: hadm_id 100's real
    discharge is at t=10h, but a stray same-hadm_id row lands at t=200h;
    hadm_id 101 (the genuine next admission) starts at t=50h -- between the
    real discharge and the stray event. The correct onset for hadm_id 100
    is hadm_id 101's t=50h, not hadm_id 102's t=300h.
    """
    rows: List[_EventRow] = [
        # hadm_id 100: real events end at t=10h...
        (1, "LAB//220045//bpm", T0, 80.0, 100),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=10), 80.0, 100),
        # ...but a stray row, still tagged hadm_id 100, lands much later.
        (1, "LAB//RESULT//50912//", T0 + timedelta(hours=200), 1.1, 100),
        # hadm_id 101: the genuine next admission, well before the stray row.
        (1, "HOSPITAL_ADMISSION//EMERGENCY", T0 + timedelta(hours=50), None, 101),
        # hadm_id 102: a later, unrelated admission -- must NOT be selected.
        (1, "HOSPITAL_ADMISSION//EMERGENCY", T0 + timedelta(hours=300), None, 102),
    ]
    events = pl.DataFrame(
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
    origins = origin_hours(events)
    onsets = _next_visit_onsets(events, "HOSPITAL_ADMISSION//", origins)

    assert onsets[(1, 100)] == 50.0, (
        f"expected hadm_id 100's next admission at t=50h (hadm_id 101), "
        f"got {onsets[(1, 100)]}h -- the stray t=200h row on hadm_id 100 "
        "likely pushed the search past the real readmission"
    )


def test_visit_envelope_reports_first_and_last_event_hours_per_visit() -> None:
    """(subject, visit) -> (start, end) in hours-since-origin, two visits."""
    rows: List[_EventRow] = [
        # subject 1, hadm 100: origin at t=0, spans 0h to 10h.
        (1, "LAB//220045//bpm", T0, 80.0, 100),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=10), 80.0, 100),
        # subject 1, hadm 101: a second, later, longer visit for the SAME
        # subject -- envelope must key by (subject, hadm), not subject alone.
        (1, "LAB//220045//bpm", T0 + timedelta(hours=50), 80.0, 101),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=250), 80.0, 101),
        # subject 2: origin at t=0 relative to ITS OWN first event (T0+5h in
        # wall-clock time), so its envelope hours are relative, not absolute.
        (2, "LAB//220045//bpm", T0 + timedelta(hours=5), 80.0, 200),
        (2, "LAB//220045//bpm", T0 + timedelta(hours=7), 80.0, 200),
    ]
    events = pl.DataFrame(
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

    envelope = visit_envelope(events)

    assert envelope[(1, 100)] == (0.0, 10.0)
    assert envelope[(1, 101)] == (50.0, 250.0)
    assert envelope[(2, 200)] == (0.0, 2.0)


def test_visit_envelope_ignores_rows_with_no_hadm_id() -> None:
    """Static/subject-scoped rows (no hadm_id) must not create a phantom visit."""
    rows: List[Tuple[int, str, datetime, Optional[float], Optional[int]]] = [
        (1, "LAB//220045//bpm", T0, 80.0, 100),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=10), 80.0, 100),
        (1, "GENDER//F", T0, None, None),
    ]
    events = pl.DataFrame(
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

    envelope = visit_envelope(events)

    assert envelope == {(1, 100): (0.0, 10.0)}


def test_probe_events_names_match_their_own_concept() -> None:
    """Every PROBE_EVENTS entry is a concept-triggered AlertEvent named after it.

    Guards against a typo silently turning a probe task into a permanently
    unfittable one (event_times raises on an unknown concept name, but only
    at run time against real concept-labeled data, not at import time).
    """
    expected = {
        "anemia",
        "hyperkalemia",
        "hypoglycemia",
        "hyponatremia",
        "thrombocytopenia",
    }
    names = {a.name for a in PROBE_EVENTS}
    assert names == expected
    for alert in PROBE_EVENTS:
        assert alert.concept == alert.name
        assert alert.code_prefix is None
        assert not alert.subject_scoped
        assert not alert.next_visit
