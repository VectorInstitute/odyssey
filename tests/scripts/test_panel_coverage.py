"""Tests for the signal-panel coverage report (scripts/panel_coverage.py)."""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from odyssey.data.signal_panel import N_PANEL_SIGNALS, SIGNAL_PANEL
from scripts import panel_coverage
from scripts.panel_coverage import OUTPUT_KEYS, SOURCES, format_table


@pytest.mark.parametrize("source", SOURCES)
def test_every_source_resolves_a_subset_of_the_panel(source: str) -> None:
    report = panel_coverage.panel_coverage(source)
    assert set(report) == set(OUTPUT_KEYS)
    assert report["source"] == source
    assert report["n_panel_signals"] == N_PANEL_SIGNALS == len(SIGNAL_PANEL)
    assert 0 < report["n_resolved"] <= report["n_panel_signals"]
    assert report["n_resolved"] == len(report["resolved"])
    assert len(report["resolved"]) + len(report["unresolved"]) == N_PANEL_SIGNALS
    assert not set(report["resolved"]) & set(report["unresolved"])
    assert set(report["prefixes"]) == set(report["resolved"])
    # Without an inventory the observed split is explicitly absent.
    assert report["codes_inventory"] is False
    assert report["resolved_observed"] is None


def test_gemini_lacks_the_noninvasive_blood_pressure_panel() -> None:
    """The review's guess, checked against the in-repo table."""
    report = panel_coverage.panel_coverage("gemini")
    assert "sbp_noninvasive" in report["unresolved"]
    assert "map_noninvasive" in report["unresolved"]
    assert "creatinine" in report["resolved"]
    assert "lactate" in report["resolved"]


def test_observed_codes_split_the_resolved_signals() -> None:
    """A resolved prefix that no charted code matches is reported as unobserved."""
    heart_rate = panel_coverage.panel_coverage("gemini")["prefixes"]["heart_rate"][0]
    report = panel_coverage.panel_coverage(
        "gemini", observed_codes=[heart_rate + "bpm::3", "SOMETHING_ELSE"]
    )
    assert report["codes_inventory"] is True
    assert report["resolved_observed"] == ["heart_rate"]
    assert "creatinine" in report["resolved_unobserved"]
    assert set(report["resolved_observed"]) | set(report["resolved_unobserved"]) == set(
        report["resolved"]
    )
    assert "no codes" in format_table(report)


def test_main_writes_json_from_a_codes_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    codes = tmp_path / "codes.parquet"
    pl.DataFrame(
        {"code": ["LAB//3020564//umol/L", "VITALS//3027018//"], "count": [5, 7]}
    ).write_parquet(codes)
    out = tmp_path / "panel_coverage.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "panel_coverage",
            "--source",
            "gemini",
            "--codes-parquet",
            str(codes),
            "--output-json",
            str(out),
        ],
    )
    panel_coverage.main()
    report = json.loads(out.read_text())
    assert set(report) == set(OUTPUT_KEYS)
    assert sorted(report["resolved_observed"]) == ["creatinine", "heart_rate"]
    # Only names leave: the inventory's counts never reach the output.
    assert "count" not in json.dumps(report)
    assert "resolved: " in capsys.readouterr().out
