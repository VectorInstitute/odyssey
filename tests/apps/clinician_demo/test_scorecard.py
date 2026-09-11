"""Scorecard: cells, intervals, computed headline and missing-file handling."""

import json
from pathlib import Path

import pytest

from apps.clinician_demo.schemas import ConceptInfo, ScoreCell
from apps.clinician_demo.scorecard import (
    build_scorecard,
    concept_readouts,
    headline,
    read_json,
    score_cells,
)


def _alert(
    event: str, h: float, scorer: str, auroc: float | None, **extra: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "event": event,
        "horizon_hours": h,
        "scorer": scorer,
        "auroc": auroc,
        "n_at_risk": 1000,
        "n_positive": 25,
        "calibration": [
            {"predicted": 0.01, "observed": 0.02, "n": 100},
            {"predicted": None, "observed": 0.1, "n": 5},
        ],
    }
    row.update(extra)
    return row


ALERTS = [
    _alert("death", 24.0, "hazard", 0.95),
    _alert("death", 24.0, "baseline_gbm", 0.93),
    _alert("death", 24.0, "concept", 0.6),
    _alert("acute_kidney_injury", 24.0, "hazard", 0.82),
    _alert("acute_kidney_injury", 24.0, "baseline_gbm", 0.88),
    _alert("acute_kidney_injury", 8.0, "hazard", 0.87),
    _alert("readmission_30d", 24.0, "hazard", 0.58),
]
CIS = {
    "cells": {
        "death@24h": {
            "scorers": {
                "hazard": {"auroc": {"point": 0.95, "ci_low": 0.94, "ci_high": 0.96}},
                "gbm": {"auroc": {"point": 0.93, "ci_low": 0.92, "ci_high": 0.94}},
            },
            "paired_deltas": {
                "hazard_minus_gbm": {
                    "auroc": {
                        "point": 0.02,
                        "ci_low": 0.01,
                        "ci_high": 0.03,
                        "separated": True,
                    }
                }
            },
        }
    }
}


def test_cells_pair_the_model_with_the_gbm_and_skip_other_scorers() -> None:
    cells = score_cells(ALERTS, CIS, ["death", "acute_kidney_injury"], [8.0, 24.0])
    assert [(c.event, c.horizon_hours) for c in cells] == [
        ("death", 24.0),
        ("acute_kidney_injury", 8.0),
        ("acute_kidney_injury", 24.0),
    ]
    death = cells[0]
    assert (death.hazard_auroc, death.gbm_auroc) == (0.95, 0.93)
    assert death.hazard_ci == [0.94, 0.96] and death.gbm_ci == [0.92, 0.94]
    assert (
        death.delta == 0.02
        and death.delta_ci == [0.01, 0.03]
        and death.separated is True
    )
    assert death.base_rate == pytest.approx(0.025)
    assert len(death.calibration) == 1  # the bin with a null prediction is dropped


def test_without_intervals_the_delta_is_the_point_difference_and_unseparated_unknown() -> (
    None
):
    aki = score_cells(ALERTS, None, ["acute_kidney_injury"], [24.0])[0]
    assert (
        aki.delta == pytest.approx(-0.06)
        and aki.delta_ci is None
        and aki.separated is None
    )
    only_model = score_cells(ALERTS, None, ["acute_kidney_injury"], [8.0])[0]
    assert only_model.gbm_auroc is None and only_model.delta is None


def test_readmission_is_only_shown_if_asked_for() -> None:
    assert all(
        c.event != "readmission_30d"
        for c in score_cells(ALERTS, CIS, ["death"], [24.0])
    )


def test_zero_at_risk_gives_a_zero_base_rate() -> None:
    rows = [_alert("death", 8.0, "hazard", 0.9, n_at_risk=0, n_positive=0)]
    assert score_cells(rows, None, ["death"], [8.0])[0].base_rate == 0.0


def _cell(delta: float | None, separated: bool | None) -> ScoreCell:
    return ScoreCell(
        "e", 24.0, 1, 0, 0.0, 0.9, None, 0.9, None, delta, None, separated, []
    )


def test_headline_counts_wins_and_clear_wins_each_way() -> None:
    text = headline(
        [_cell(-0.05, True), _cell(-0.01, False), _cell(0.02, True), _cell(None, None)]
    )
    assert text == (
        "Against the tuned GBM on 3 event-horizon cells: the GBM ranks better on 2 "
        "(1 clearly), this model on 1 (1 clearly)."
    )
    assert headline([]) == "No GBM comparison is available for this run."


def test_concept_readouts_fill_aurocs_by_slot_name() -> None:
    concepts = [
        ConceptInfo("shock", "sustained hypotension (MAP)", "d", None),
        ConceptInfo("fever", "fever", "d", 0.5),
    ]
    filled = concept_readouts(
        {
            "concept_metrics": [
                {"name": "shock", "auroc": 0.81},
                {"name": "x", "auroc": None},
            ]
        },
        concepts,
    )
    assert [c.readout_auroc for c in filled] == [0.81, 0.5]
    assert concept_readouts(None, concepts) == concepts


def test_build_scorecard_from_a_run_directory(tmp_path: Path) -> None:
    (tmp_path / "alerts.json").write_text(json.dumps(ALERTS))
    (tmp_path / "alerts_cis.json").write_text(json.dumps(CIS))
    (tmp_path / "inference_results.json").write_text(
        json.dumps({"concept_metrics": [{"name": "fever", "auroc": 0.9}]})
    )
    card = build_scorecard(
        tmp_path, ["death"], [24.0], [ConceptInfo("fever", "fever", "", None)]
    )
    assert len(card.cells) == 1 and card.concepts[0].readout_auroc == 0.9
    assert card.headline.startswith("Against the tuned GBM on 1")
    assert not any("no banked" in n.lower() for n in card.notes)


def test_missing_or_broken_files_degrade_with_a_note(tmp_path: Path) -> None:
    empty = build_scorecard(tmp_path, ["death"], [24.0], [])
    assert empty.cells == [] and "no banked alert evaluation" in empty.notes[0]
    (tmp_path / "alerts.json").write_text(json.dumps(ALERTS))
    (tmp_path / "alerts_cis.json").write_text("{broken")
    no_ci = build_scorecard(tmp_path, ["death"], [24.0], [])
    assert (
        "No bootstrap intervals" in no_ci.notes[0] and no_ci.cells[0].hazard_ci is None
    )
    assert read_json(tmp_path / "alerts_cis.json") is None
