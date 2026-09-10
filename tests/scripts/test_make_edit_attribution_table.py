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
