"""The edit-attribution table averages each event's three horizons into one cell."""

from __future__ import annotations

from scripts.make_edit_attribution_table import render


def _run(concept: str, source: str, n: int, cells: list[dict]) -> dict:
    return {"concept": concept, "source": source, "n_subjects": n, "cells": cells}


def test_render_averages_horizons_per_event() -> None:
    results = {
        "runs": [
            _run(
                "sepsis3",
                "mimic_iv",
                50,
                [
                    {
                        "event": "death",
                        "horizon": "8h",
                        "agree": 47,
                        "total": 50,
                        "pct": 94.0,
                    },
                    {
                        "event": "death",
                        "horizon": "24h",
                        "agree": 49,
                        "total": 50,
                        "pct": 98.0,
                    },
                    {
                        "event": "death",
                        "horizon": "72h",
                        "agree": 50,
                        "total": 50,
                        "pct": 100.0,
                    },
                ],
            )
        ]
    }
    text = render(results)
    # mean of 94, 98, 100 = 97.33 -> rounds to 97%
    assert "97\\%" in text


def test_render_marks_icu_admission_degenerate_on_eicu() -> None:
    results = {
        "runs": [
            _run(
                "qsofa",
                "eicu",
                50,
                [
                    {
                        "event": "icu_admission",
                        "horizon": "24h",
                        "agree": 12,
                        "total": 50,
                        "pct": 24.0,
                    },
                ],
            )
        ]
    }
    text = render(results)
    assert "--$^\\dagger$" in text
    assert "24\\%" not in text


def test_render_does_not_mark_icu_admission_degenerate_on_mimic() -> None:
    results = {
        "runs": [
            _run(
                "qsofa",
                "mimic_iv",
                50,
                [
                    {
                        "event": "icu_admission",
                        "horizon": "24h",
                        "agree": 44,
                        "total": 50,
                        "pct": 88.0,
                    },
                ],
            )
        ]
    }
    text = render(results)
    assert "88\\%" in text
    assert "--$^\\dagger$" not in text


def test_render_missing_event_shows_dash() -> None:
    results = {"runs": [_run("sepsis3", "mimic_iv", 50, [])]}
    text = render(results)
    # every event column should render as a plain missing dash, not a
    # crash, when a run has no cells for that event at all.
    assert text.count("--") >= 4


def test_render_includes_n_and_labels() -> None:
    results = {"runs": [_run("aki_stage_3", "eicu", 50, [])]}
    text = render(results)
    assert "AKI stage 3" in text
    assert "eICU-CRD" in text
    assert " 50 " in text or "& 50 &" in text


def _death_cells(agree: int) -> list[dict]:
    return [
        {
            "event": "death",
            "horizon": h,
            "agree": agree,
            "total": 50,
            "pct": 2.0 * agree,
        }
        for h in ("8h", "24h", "72h")
    ]


def _two_arm_results() -> dict:
    attributed = _run("sepsis3", "mimic_iv", 50, _death_cells(47))
    attributed["selection"] = "attributed"
    rnd = _run("sepsis3", "mimic_iv", 50, _death_cells(26))
    rnd["selection"] = "random"
    rnd["random_seed"] = 0
    # file order puts the random run first on purpose: the renderer must
    # place it under its attributed twin regardless
    return {"runs": [rnd, attributed]}


def test_render_default_drops_random_runs_and_has_no_codes_column() -> None:
    text = render(_two_arm_results())
    untagged = render({"runs": [_run("sepsis3", "mimic_iv", 50, _death_cells(47))]})
    assert text == untagged
    assert "Codes" not in text
    assert "random" not in text
    assert "94\\%" in text
    assert "52\\%" not in text


def test_render_with_random_adds_codes_column_and_pairs_rows() -> None:
    text = render(_two_arm_results(), with_random=True)
    assert "Concept & Dataset & Codes & $n$" in text
    assert "\\begin{tabular}{lllrrrrr}" in text
    rows = [line for line in text.splitlines() if line.startswith("Sepsis-3")]
    assert len(rows) == 2
    assert rows[0].startswith("Sepsis-3 & MIMIC-IV & attributed & 50 & 94\\%")
    assert rows[1].startswith("Sepsis-3 & MIMIC-IV & random & 50 & 52\\%")


def test_render_with_random_keeps_unpaired_random_run() -> None:
    rnd = _run("qsofa", "eicu", 50, _death_cells(30))
    rnd["selection"] = "random"
    text = render({"runs": [rnd]}, with_random=True)
    assert "qSOFA & eICU-CRD & random & 50 & 60\\%" in text
    assert "qSOFA" not in render({"runs": [rnd]})
