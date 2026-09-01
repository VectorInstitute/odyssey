"""Tests for the comparator-table generator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.make_comparator_tables import (
    _bold_targets,
    build_rows,
    load_alerts,
    load_tabicl,
)


def _alerts_records(protocol: int = 4) -> list[dict]:
    """Two events x one horizon, hazard and GBM, GBM ahead on AKI only."""
    out = []
    for event, hazard, gbm in (
        ("acute_kidney_injury", 0.80, 0.90),
        ("death", 0.96, 0.94),
    ):
        for scorer, auroc in (("hazard", hazard), ("baseline_gbm", gbm)):
            out.append(
                {
                    "event": event,
                    "horizon_hours": 8.0,
                    "scorer": scorer,
                    "auroc": auroc,
                    "n_at_risk": 1000,
                    "n_positive": 50,
                    "landmark_protocol_version": protocol,
                }
            )
    return out


def _write(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_load_alerts_keys_cells_and_collects_protocol(tmp_path: Path) -> None:
    cells, protocols = load_alerts(_write(tmp_path, "a.json", _alerts_records()))
    assert set(cells) == {"acute_kidney_injury@8h", "death@8h"}
    assert protocols == {4}
    assert cells["death@8h"]["scores"]["hazard"] == 0.96
    assert cells["death@8h"]["n"] == 1000


def test_scorers_outside_hazard_and_gbm_are_ignored(tmp_path: Path) -> None:
    """alerts.json also carries `concept` and `next_mass` proxy scorers."""
    records = _alerts_records() + [
        {
            "event": "death",
            "horizon_hours": 8.0,
            "scorer": "next_mass",
            "auroc": 0.99,
            "n_at_risk": 1000,
            "n_positive": 50,
            "landmark_protocol_version": 4,
        }
    ]
    cells, _ = load_alerts(_write(tmp_path, "a.json", records))
    assert set(cells["death@8h"]["scores"]) == {"hazard", "baseline_gbm"}


def test_bold_goes_to_the_max_without_cis() -> None:
    assert _bold_targets({"hazard": 0.9, "baseline_gbm": 0.8}, None) == {"hazard"}


def test_bold_is_withheld_when_the_paired_delta_does_not_separate() -> None:
    """The captions promise bold means a separated cell, not a bigger number."""
    ci_cell = {
        "paired_deltas": {
            "hazard_minus_baseline_gbm": {"auroc": {"point": 0.002, "separated": False}}
        }
    }
    assert _bold_targets({"hazard": 0.9, "baseline_gbm": 0.898}, ci_cell) == set()


def test_bold_survives_a_separated_delta() -> None:
    ci_cell = {
        "paired_deltas": {
            "hazard_minus_baseline_gbm": {"auroc": {"point": 0.1, "separated": True}}
        }
    }
    assert _bold_targets({"hazard": 0.9, "baseline_gbm": 0.8}, ci_cell) == {"hazard"}


def test_rows_warn_about_missing_columns(tmp_path: Path) -> None:
    cells, _ = load_alerts(_write(tmp_path, "a.json", _alerts_records()))
    rows, notes = build_rows(cells, {}, {})
    assert any("TabICL" in n for n in notes)
    assert any("NO CIs" in n for n in notes)
    assert any("AKI" in r for r in rows)
    assert any("Death" in r for r in rows)
    # sepsis3 is absent from this arm and must simply not appear
    assert not any("Sepsis-3" in r for r in rows)


def test_tabicl_column_is_added_when_supplied(tmp_path: Path) -> None:
    cells, _ = load_alerts(_write(tmp_path, "a.json", _alerts_records()))
    tabicl_path = _write(
        tmp_path,
        "t.json",
        {"death@8h": {"tabicl": {"point_estimate": 0.97}}},
    )
    tabicl = load_tabicl(tabicl_path)
    rows, notes = build_rows(cells, tabicl, {})
    # The column is present, so no ABSENT note. (A PARTIAL note IS expected
    # here and is asserted separately: this fixture covers one of two cells.)
    assert not any("ABSENT" in n for n in notes)
    death_row = next(r for r in rows if "Death" in r)
    # TabICL leads this cell, so it takes the bold
    assert "\\textbf{0.970}" in death_row


def test_partial_tabicl_coverage_is_reported(tmp_path: Path) -> None:
    """A partial supplementary file is worse than a missing one.

    The column renders, uncovered cells quietly show "--", and the table
    looks finished. tabicl_strong_compare.py rewrites its output after every
    cell, so pointing at a still-running job produces exactly this.
    """
    cells, _ = load_alerts(_write(tmp_path, "a.json", _alerts_records()))
    tabicl = load_tabicl(
        _write(tmp_path, "t.json", {"death@8h": {"tabicl": {"point_estimate": 0.97}}})
    )
    _, notes = build_rows(cells, tabicl, {})
    partial = [n for n in notes if "PARTIAL" in n]
    assert len(partial) == 1
    assert "1 of 2 cells" in partial[0]
    assert "acute_kidney_injury@8h" in partial[0]


def test_complete_tabicl_coverage_reports_no_partial_warning(tmp_path: Path) -> None:
    cells, _ = load_alerts(_write(tmp_path, "a.json", _alerts_records()))
    tabicl = load_tabicl(
        _write(
            tmp_path,
            "t.json",
            {
                "death@8h": {"tabicl": {"point_estimate": 0.97}},
                "acute_kidney_injury@8h": {"tabicl": {"point_estimate": 0.85}},
            },
        )
    )
    _, notes = build_rows(cells, tabicl, {})
    assert not any("PARTIAL" in n for n in notes)


def test_absent_tabicl_reports_absent_not_partial(tmp_path: Path) -> None:
    cells, _ = load_alerts(_write(tmp_path, "a.json", _alerts_records()))
    _, notes = build_rows(cells, {}, {})
    assert any("ABSENT" in n for n in notes)
    assert not any("PARTIAL" in n for n in notes)


def test_mixed_landmark_protocols_are_refused(tmp_path: Path) -> None:
    """v4 and v1-v3 cells are not comparable and must never share a table."""
    records = _alerts_records(protocol=4) + _alerts_records(protocol=3)
    _, protocols = load_alerts(_write(tmp_path, "a.json", records))
    assert protocols == {3, 4}


def test_cells_with_no_auroc_render_as_placeholder(tmp_path: Path) -> None:
    records = _alerts_records()
    for rec in records:
        if rec["event"] == "death" and rec["scorer"] == "hazard":
            rec["auroc"] = None
    cells, _ = load_alerts(_write(tmp_path, "a.json", records))
    rows, _ = build_rows(cells, {}, {})
    death_row = next(r for r in rows if "Death" in r)
    assert "--" in death_row


@pytest.mark.parametrize("horizon", [8.0, 24.0, 72.0])
def test_horizon_labels_have_no_trailing_zeros(tmp_path: Path, horizon: float) -> None:
    records = _alerts_records()
    for rec in records:
        rec["horizon_hours"] = horizon
    cells, _ = load_alerts(_write(tmp_path, "a.json", records))
    rows, _ = build_rows(cells, {}, {})
    assert any(f"{horizon:g}h" in r for r in rows)
    assert not any(".0h" in r for r in rows)


def test_emits_a_complete_tabular_not_a_bare_row_body(tmp_path: Path) -> None:
    """The file must be a self-contained tabular.

    \\input-ing a bare row body inside a tabular breaks LaTeX's alignment
    scanning and raises "Misplaced \\noalign" at the following
    \\bottomrule, which silently corrupts the table. The generator owns the
    preamble because it is the thing that knows its own column count.
    """
    alerts = _write(tmp_path, "a.json", _alerts_records())
    out = tmp_path / "body.tex"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_comparator_tables.py",
            "--alerts",
            str(alerts),
            "--output-tex",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    text = out.read_text()
    assert "\\begin{tabular}{llrrr}" in text
    assert "\\toprule" in text and "\\bottomrule" in text
    assert text.rstrip().endswith("\\end{tabular}")
    assert "Event & $h$ & $n$ (pos) & Hazard & GBM" in text


def test_tabicl_column_widens_the_preamble_and_header(tmp_path: Path) -> None:
    """Six columns when TabICL is supplied, five when it is not."""
    alerts = _write(tmp_path, "a.json", _alerts_records())
    tab = _write(tmp_path, "t.json", {"death@8h": {"tabicl": {"point_estimate": 0.97}}})
    out = tmp_path / "body.tex"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_comparator_tables.py",
            "--alerts",
            str(alerts),
            "--tabicl",
            str(tab),
            "--output-tex",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    text = out.read_text()
    assert "\\begin{tabular}{llrrrr}" in text
    assert "TabICL" in text
