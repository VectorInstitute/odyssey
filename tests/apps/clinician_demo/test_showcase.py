"""Gallery selection: deterministic, episode-based leads, honest about misses."""

import polars as pl
import pytest

from apps.clinician_demo.schemas import Gallery, GallerySection
from apps.clinician_demo.showcase import (
    MAX_FEATURED_LEAD_HOURS,
    MAX_STAY_HOURS,
    VISIT_STATS_SCHEMA,
    build_gallery,
    empty_visit_stats,
    visit_stats_from_alert_rows,
)


DISPLAY = {"icu_admission": "ICU admission", "death": "Death"}


def _stats(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=VISIT_STATS_SCHEMA)


def _row(  # noqa: PLR0913 -- one keyword per stats column
    sid: int,
    vid: int,
    event: str,
    *,
    positive: bool,
    cross: float | None,
    alert: float | None = None,
    end: float,
    start: float = 0.0,
    max_risk: float = 0.5,
    threshold: float = 0.2,
) -> dict[str, object]:
    return {
        "subject_id": sid,
        "visit_id": vid,
        "event": event,
        "positive": positive,
        "first_cross_hours": cross,
        "alert_start_hours": alert,
        "end_hours": end,
        "start_hours": start,
        "max_risk": max_risk,
        "threshold": threshold,
    }


def _section(gallery: Gallery, kind: str) -> GallerySection:
    return next(s for s in gallery.sections if s.kind == kind)


# -- stats from banked landmark rows ------------------------------------------


def _rows(event: str, hazards: list[float], outcome: float = 1.0) -> pl.DataFrame:
    n = len(hazards)
    return pl.DataFrame(
        {
            "subject_id": [1.0] * n,
            "visit_id": [10.0] * n,
            "time_hours": [4.0 * i for i in range(n)],
            "event": [event] * n,
            "hazard@24h": hazards,
            "y@24h": [0.0] * (n - 1) + [outcome],
        }
    )


def test_episode_start_is_the_run_still_on_at_the_last_row() -> None:
    # on at 0, off at 4, back on from 8 to the last row (16): the episode began at 8
    stats = visit_stats_from_alert_rows(
        _rows("icu_admission", [0.5, 0.1, 0.3, 0.4, 0.6]), {"icu_admission": 0.2}
    )
    row = stats.row(0, named=True)
    assert row["positive"] and row["first_cross_hours"] == 0.0
    assert row["alert_start_hours"] == 8.0 and row["end_hours"] == 16.0


def test_no_episode_when_the_alert_was_off_at_the_last_row() -> None:
    stats = visit_stats_from_alert_rows(
        _rows("icu_admission", [0.5, 0.6, 0.1]), {"icu_admission": 0.2}
    )
    row = stats.row(0, named=True)
    assert row["first_cross_hours"] == 0.0 and row["alert_start_hours"] is None


def test_an_alert_on_throughout_starts_at_the_first_row() -> None:
    stats = visit_stats_from_alert_rows(
        _rows("icu_admission", [0.3, 0.3, 0.3]), {"icu_admission": 0.2}
    )
    assert stats.row(0, named=True)["alert_start_hours"] == 0.0


def test_negatives_never_carry_an_episode() -> None:
    stats = visit_stats_from_alert_rows(
        _rows("death", [0.5, 0.5], outcome=0.0), {"death": 0.2}
    )
    row = stats.row(0, named=True)
    assert not row["positive"] and row["alert_start_hours"] is None
    assert row["first_cross_hours"] == 0.0


def test_stats_cast_ids_and_ignore_unthresholded_events() -> None:
    rows = pl.concat(
        [_rows("icu_admission", [0.1, 0.4]), _rows("readmission_30d", [0.9, 0.9])]
    )
    stats = visit_stats_from_alert_rows(rows, {"icu_admission": 0.2})
    assert stats.schema == pl.Schema(VISIT_STATS_SCHEMA)
    assert stats["event"].to_list() == ["icu_admission"]
    assert (stats["subject_id"][0], stats["visit_id"][0]) == (1, 10)
    assert visit_stats_from_alert_rows(rows, {}).height == 0


# -- the gallery ----------------------------------------------------------------


def test_early_warnings_use_the_live_episode_not_the_first_touch() -> None:
    stats = _stats(
        [
            # touched the line at hour 1, but the live alert began at 40: lead 20
            _row(
                1, 10, "icu_admission", positive=True, cross=1.0, alert=40.0, end=60.0
            ),
            _row(
                2, 20, "icu_admission", positive=True, cross=0.0, alert=10.0, end=14.0
            ),
            _row(3, 30, "icu_admission", positive=True, cross=0.0, alert=0.0, end=12.0),
        ]
    )
    section = _section(build_gallery(stats, display=DISPLAY), "early_warning")
    assert [(c.subject_id, c.lead_hours) for c in section.cases] == [
        (1, 20.0),
        (3, 12.0),
    ]
    assert section.cases[0].headline == "ICU admission: alert on ≈20 h before it began"
    assert section.cases[0].lead_approximate
    assert "ICU admission 2 of 3 (67%)" in section.summary
    assert "Death" not in section.summary  # no Death events: left out, not "0 of 0"


def test_featured_leads_are_capped_but_still_counted() -> None:
    long_lead = MAX_FEATURED_LEAD_HOURS + 10
    stats = _stats(
        [
            _row(1, 10, "death", positive=True, cross=0.0, alert=0.0, end=long_lead),
            _row(2, 20, "death", positive=True, cross=0.0, alert=0.0, end=30.0),
            _row(
                3,
                30,
                "death",
                positive=True,
                cross=0.0,
                alert=0.0,
                end=MAX_STAY_HOURS + 1,
            ),
        ]
    )
    section = _section(build_gallery(stats, display=DISPLAY), "early_warning")
    assert [c.subject_id for c in section.cases] == [2]
    assert "Death 3 of 3 (100%)" in section.summary


def test_exact_leads_drop_the_approximation_mark_and_per_event_caps_apply() -> None:
    stats = _stats(
        [
            _row(i, i, "death", positive=True, cross=0.0, alert=0.0, end=10.0 + i)
            for i in range(1, 6)
        ]
    )
    gallery = build_gallery(
        stats, display=DISPLAY, per_event=2, approximate_leads=False
    )
    cases = _section(gallery, "early_warning").cases
    assert [c.subject_id for c in cases] == [5, 4]
    assert "≈" not in cases[0].headline and not cases[0].lead_approximate


def test_quiet_stays_need_every_event_low_no_positive_and_two_days() -> None:
    def quiet(
        sid: int, *, stay: float = 60.0, risk: float = 0.01, positive: bool = False
    ) -> list[dict[str, object]]:
        return [
            _row(
                sid,
                sid,
                e,
                positive=positive and e == "death",
                cross=None,
                end=stay,
                max_risk=risk,
            )
            for e in DISPLAY
        ]

    stats = _stats(
        quiet(1)
        + quiet(2, stay=30.0)  # too short
        + quiet(3, risk=0.06)  # 0.06 >= 0.25 * 0.2
        + quiet(4, positive=True)
        + [_row(5, 5, "death", positive=False, cross=None, end=90.0, max_risk=0.0)]
        + quiet(6, stay=100.0)
    )
    section = _section(build_gallery(stats, display=DISPLAY), "quiet")
    assert [c.subject_id for c in section.cases] == [6, 1]
    assert section.cases[0].event is None and section.cases[0].headline.endswith(
        "4.2 days"
    )
    assert section.summary.startswith("2 stays had none of the events")
    assert all(not c.lead_approximate for c in section.cases)


def test_misses_are_events_with_no_alert_on_and_false_alarms_count_pairs() -> None:
    stats = _stats(
        [
            _row(1, 1, "death", positive=True, cross=None, end=10.0, max_risk=0.05),
            # crossed earlier but the alert was off again at onset: still a miss
            _row(
                2,
                2,
                "death",
                positive=True,
                cross=2.0,
                alert=None,
                end=10.0,
                max_risk=0.3,
            ),
            _row(3, 3, "death", positive=True, cross=2.0, alert=2.0, end=10.0),
            _row(
                4, 4, "icu_admission", positive=False, cross=3.0, end=10.0, max_risk=0.9
            ),
            _row(5, 5, "icu_admission", positive=False, cross=None, end=10.0),
        ]
    )
    gallery = build_gallery(
        stats, display=DISPLAY, seen_in_training=lambda sid: sid == 2
    )
    misses = _section(gallery, "miss")
    assert [c.subject_id for c in misses.cases] == [1, 2]  # lowest peak risk first
    assert misses.cases[1].seen_in_training and not misses.cases[0].seen_in_training
    assert misses.summary == "2 of 3 events (67%) began with no alert on"
    assert misses.cases[0].headline == "Death began with no alert on"
    alarms = _section(gallery, "false_alarm")
    assert [c.subject_id for c in alarms.cases] == [4] and alarms.cases[
        0
    ].lead_hours is None
    assert alarms.summary.startswith("In 1 of 2 cases (50%) an event's alert came on")


def test_empty_stats_give_four_empty_sections_without_dividing_by_zero() -> None:
    gallery = build_gallery(empty_visit_stats(), display=DISPLAY)
    assert [s.kind for s in gallery.sections] == [
        "early_warning",
        "quiet",
        "miss",
        "false_alarm",
    ]
    assert all(s.cases == [] for s in gallery.sections)
    assert _section(gallery, "early_warning").summary == (
        "No stay in this data had one of the events."
    )
    assert "n/a" in _section(gallery, "miss").summary


def test_events_outside_the_display_map_are_ignored_and_selection_is_deterministic() -> (
    None
):
    stats = _stats(
        [
            _row(i, i, "sepsis3", positive=True, cross=0.0, alert=0.0, end=20.0)
            for i in range(3)
        ]
        + [_row(9, 9, "death", positive=True, cross=0.0, alert=0.0, end=20.0)]
    )
    one = build_gallery(stats, display=DISPLAY)
    two = build_gallery(stats.reverse(), display=DISPLAY)
    assert one == two
    assert all(c.event != "sepsis3" for s in one.sections for c in s.cases)


@pytest.mark.parametrize("kind", ["early_warning", "quiet", "miss", "false_alarm"])
def test_every_section_has_a_title_and_summary(kind: str) -> None:
    section = _section(build_gallery(empty_visit_stats(), display=DISPLAY), kind)
    assert section.title and section.summary
