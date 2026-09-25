"""The rebuttal steps' export whitelists match what their scripts write.

run.sh exports aggregate JSON through a top-level key whitelist that
refuses anything unknown. PR #238 was the second time a script gained
a field and the export refused on the node, where a retry costs a
session inside the secure environment. These tests read the shipped
run.sh and pin each step's key list to the script that produces the
file, so the mismatch fails here.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts import alerts_cis, cohort_counts, panel_coverage


REPO = Path(__file__).resolve().parents[3]
RUN_SH = REPO / "scripts" / "gemini" / "run.sh"
STEPS = ("alerts-cis", "panel-coverage", "cohort-counts")


def _keys(variable: str) -> set[str]:
    source = RUN_SH.read_text()
    match = re.search(rf'^\s*{variable}="([^"]+)"', source, re.MULTILINE)
    assert match, f"{variable} not found in run.sh"
    return set(match.group(1).split())


def _export_keys_used(export_name: str) -> str:
    """Return the key-list variable the export call for ``export_name`` passes."""
    source = RUN_SH.read_text()
    match = re.search(
        rf'_export_aggregate_json \\\n\s*"scripts/gemini/out/evals/\$\{{run_name\}}[^"]*{export_name}\.json" "\$OUTPUT_JSON" \\\n\s*"\$(\w+)"',
        source,
    )
    assert match, f"no _export_aggregate_json call for {export_name}"
    return match.group(1)


def test_steps_are_documented_dispatched_and_listed_as_unknown_step_hints() -> None:
    source = RUN_SH.read_text()
    usage = re.search(
        r"^# Usage.*\n#\s+scripts/gemini/run.sh \[(.*)\]$", source, re.MULTILINE
    )
    assert usage
    for step in STEPS:
        assert f"{step} <run-name>" in usage.group(1)
        assert re.search(
            rf'^\s+{re.escape(step)}\) run_\w+ "\$\{{2:-\}}" ;;$', source, re.MULTILINE
        )
        assert re.search(rf"^#   {re.escape(step)} <run-name>$", source, re.MULTILINE)
        assert re.search(rf"unknown step: .*\b{re.escape(step)}\b", source)


def test_panel_coverage_whitelist_matches_the_script() -> None:
    assert _export_keys_used("panel_coverage") == "PANEL_COVERAGE_JSON_KEYS"
    assert _keys("PANEL_COVERAGE_JSON_KEYS") == set(panel_coverage.OUTPUT_KEYS)
    assert set(panel_coverage.panel_coverage("gemini")) == set(
        panel_coverage.OUTPUT_KEYS
    )


def test_cohort_counts_whitelist_matches_the_script() -> None:
    assert _export_keys_used("cohort_counts") == "COHORT_COUNTS_JSON_KEYS"
    assert _keys("COHORT_COUNTS_JSON_KEYS") == set(cohort_counts.OUTPUT_KEYS)


def test_alerts_cis_whitelist_matches_what_main_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _export_keys_used("alerts_cis") == "ALERTS_CIS_JSON_KEYS"
    rng = np.random.default_rng(0)
    sids = np.repeat(np.arange(30), 3)
    y = np.repeat((np.arange(30) % 3 == 0).astype(float), 3)
    noise = rng.uniform(0, 1, len(sids))
    dump = tmp_path / "alerts_rows_allshards.parquet"
    pl.DataFrame(
        {
            "event": ["death"] * len(sids),
            "subject_id": sids,
            "visit_id": sids,
            "time_hours": np.arange(len(sids), dtype=float),
            "y@8h": y,
            "hazard@8h": np.where(y == 1, 0.4 + 0.5 * noise, 0.1 + 0.5 * noise),
            "gbm@8h": np.where(y == 1, 0.3 + 0.6 * noise, 0.1 + 0.6 * noise),
        }
    ).write_parquet(dump)
    out = tmp_path / "alerts_cis_allshards.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alerts_cis",
            "--dump",
            str(dump),
            "--output-json",
            str(out),
            "--scorers",
            "hazard",
            "gbm",
            "--n-boot",
            "20",
        ],
    )
    alerts_cis.main()
    written = json.loads(out.read_text())
    assert set(written) == _keys("ALERTS_CIS_JSON_KEYS")
    assert written["n_rows_in_dumps"] == written["n_rows_scored"] == 90
    assert written["n_subjects_in_dumps"] == written["n_subjects_scored"] == 30
    (row,) = written["summary"]
    assert row["event"] == "death" and row["horizon_hours"] == 8.0
    assert (
        row["n_at_risk"] == 90 and row["n_positive"] == 30 and row["n_subjects"] == 30
    )
    assert row["hazard_auroc"] is not None and row["gbm_auroc"] is not None
    delta = row["hazard_minus_gbm_auroc"]
    assert set(delta) == {"point", "ci_low", "ci_high", "separated"}
    assert delta["ci_low"] <= delta["point"] <= delta["ci_high"]
    # The export validator's forbidden patient-level keys never appear.
    forbidden = {"subject_id", "subject_ids", "rows", "per_subject"}

    def scan(node: object) -> None:
        if isinstance(node, dict):
            assert not forbidden & set(node)
            for value in node.values():
                scan(value)
        elif isinstance(node, list):
            for value in node:
                scan(value)

    scan(written)
