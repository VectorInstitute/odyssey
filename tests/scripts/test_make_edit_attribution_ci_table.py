"""The per-horizon CI table reports each horizon separately, with Wilson bounds."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.make_edit_attribution_ci_table import render, wilson_interval


def _run(
    concept: str, source: str, n: int, cells: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"concept": concept, "source": source, "n_subjects": n, "cells": cells}


def _cell(event: str, horizon: str, agree: int) -> dict[str, Any]:
    return {
        "event": event,
        "horizon": horizon,
        "agree": agree,
        "total": 50,
        "pct": agree * 2.0,
    }


def test_wilson_interval_brackets_the_point_estimate() -> None:
    lo, hi = wilson_interval(47, 50)
    assert lo < 94.0 < hi


def test_wilson_interval_stays_inside_zero_to_one_hundred() -> None:
    # The normal approximation would put 50/50 at 100 +- 0 and 0/50 below
    # zero; Wilson has to stay inside the unit interval at both ends.
    lo, hi = wilson_interval(50, 50)
    assert 90.0 < lo < 100.0
    assert hi == pytest.approx(100.0)
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0)
    assert 0.0 < hi < 10.0


def test_wilson_interval_rejects_empty_denominator() -> None:
    with pytest.raises(ValueError, match="total=0"):
        wilson_interval(0, 0)


def test_render_keeps_the_three_horizons_apart() -> None:
    results = {
        "runs": [
            _run(
                "sepsis3",
                "mimic_iv",
                50,
                [
                    _cell("death", "8h", 47),
                    _cell("death", "24h", 49),
                    _cell("death", "72h", 50),
                ],
            )
        ]
    }
    text = render(results)
    # The averaging table would collapse these to one 97% cell; this one
    # must show all three points on the death row.
    death_row = next(row for row in text.splitlines() if row.startswith("Death "))
    assert "94 (" in death_row
    assert "98 (" in death_row
    assert "100 (" in death_row


def test_render_marks_but_still_reports_the_degenerate_eicu_cells() -> None:
    results = {"runs": [_run("qsofa", "eicu", 50, [_cell("icu_admission", "24h", 12)])]}
    text = render(results)
    assert "ICU adm.$^\\dagger$" in text
    # Marked, not suppressed: the appendix shows why it is excluded.
    assert "24 (14--37)" in text


def test_render_does_not_mark_icu_admission_on_mimic() -> None:
    results = {
        "runs": [_run("qsofa", "mimic_iv", 50, [_cell("icu_admission", "24h", 44)])]
    }
    text = render(results)
    assert "$^\\dagger$" not in text


def test_render_missing_cell_is_a_dash_not_a_crash() -> None:
    results = {"runs": [_run("sepsis3", "mimic_iv", 50, [])]}
    text = render(results)
    assert "--" in text
    assert "Sepsis-3, MIMIC-IV" in text


def test_render_labels_each_run_with_its_n() -> None:
    results = {"runs": [_run("aki_stage_3", "eicu", 50, [])]}
    text = render(results)
    assert "AKI stage 3, eICU-CRD" in text
    assert "$n=50$" in text
