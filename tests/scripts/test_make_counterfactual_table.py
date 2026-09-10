"""The counterfactual table derives counts from fractions and intervals from those."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.make_counterfactual_table import agree_count, render


def _results(**arms: Any) -> dict[str, Any]:
    return {"index_hours": 24.0, "edits": arms}


def _arm(n_edited: int, agreement: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {"n_subjects": 300, "n_edited": n_edited, "sign_agreement": agreement}


def test_agree_count_recovers_the_integer() -> None:
    assert agree_count(0.9545454545454546, 44) == 42
    assert agree_count(0.85, 40) == 34
    assert agree_count(0.522727272727272, 44) == 23


def test_agree_count_rejects_a_fraction_that_is_not_whole_subjects() -> None:
    # A fraction that does not resolve to a whole number of subjects means
    # the stored fraction and n_edited disagree; silently rounding it would
    # produce an interval for a cohort that was never scored.
    with pytest.raises(ValueError, match="not a whole"):
        agree_count(0.5, 45)


def test_render_reports_a_wilson_interval_per_cell() -> None:
    results = _results(
        lactate_x3=_arm(40, {"death": {"8h": 0.85, "24h": 0.85, "72h": 0.90}})
    )
    text = render(results)
    # 34/40 = 85%, Wilson 95% = 71 to 93.
    assert "85 (71--93)" in text


def test_render_shows_the_inert_control_spanning_chance() -> None:
    # The control exists to sit at chance; its interval must be allowed to
    # contain 50, which is the whole point of reporting one here.
    results = _results(
        normotension_6h=_arm(44, {"vasopressor_start": {"8h": 0.5227272727272727}})
    )
    text = render(results)
    assert "52 (38--66)" in text
    assert "Inert control" in text


def test_render_skips_arms_with_no_expected_direction() -> None:
    results = _results(
        lactate_x3=_arm(40, {"death": {"8h": 0.85}}),
        remove_labs_24h=_arm(277, {}),
    )
    text = render(results)
    assert "Lactate rise" in text
    assert "277" not in text


def test_render_marks_a_missing_horizon_rather_than_dropping_the_row() -> None:
    results = _results(
        creatinine_plus_1=_arm(246, {"acute_kidney_injury": {"8h": 0.5}})
    )
    text = render(results)
    row = next(r for r in text.splitlines() if r.startswith("AKI "))
    assert row.count("--") >= 2


def test_render_labels_each_arm_with_its_n() -> None:
    results = _results(hypotension_6h=_arm(44, {"death": {"8h": 0.9318181818181818}}))
    text = render(results)
    assert "$n=44$ edited" in text
    assert "Induced hypotension" in text


def test_render_takes_the_arrow_from_the_json_not_from_curated_prose() -> None:
    # The direction is whatever the harness actually scored against, so a
    # table can never claim an expectation the run did not use. The inert
    # control is the case that matters: it was scored against a FALL in
    # vasopressor hazard, not against "no expectation".
    up = _results(
        hypotension_6h=_arm(44, {"death": {"8h": 0.9318181818181818}})
        | {"edits": [{"expected_direction": {"death": 1}}]}
    )
    assert "Death $\\uparrow$" in render(up)
    down = _results(
        normotension_6h=_arm(44, {"vasopressor_start": {"8h": 0.5227272727272727}})
        | {"edits": [{"expected_direction": {"vasopressor_start": -1}}]}
    )
    assert "Vasopressor $\\downarrow$" in render(down)
