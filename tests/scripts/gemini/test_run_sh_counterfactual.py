"""The counterfactual step's sanitiser runs the shipped code, not a copy.

Nothing patient-level leaves the GEMINI node. ``_export_aggregate_json``
enforces that with a whitelist of TOP-LEVEL keys, which cannot see a
``per_subject`` list nested one level down inside an edit arm -- and
``CounterfactualSummary`` carries exactly such a list when the run is
given ``--keep-per-subject``. The step therefore runs its own sanitiser
first: it drops the run directory (a home-directory path) and refuses
outright if any arm still carries per-subject records.

This test extracts that heredoc from ``run.sh`` and runs it, so the
guard cannot drift away from the file it is meant to protect.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RUN_SH = REPO / "scripts" / "gemini" / "run.sh"


def _sanitiser_source() -> str:
    source = RUN_SH.read_text()
    match = re.search(
        r"local export_json=.*?python3 - \"\$OUTPUT_JSON\" \"\$export_json\" "
        r"<<'PY'\n(.*?)\nPY\n",
        source,
        re.DOTALL,
    )
    assert match, "counterfactual sanitiser heredoc not found in run.sh"
    body = match.group(1)
    assert "per_subject" in body
    return body


def _run(obj: dict, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    src = tmp_path / "counterfactual.json"
    dest = tmp_path / "counterfactual_export.json"
    src.write_text(json.dumps(obj))
    script = tmp_path / "sanitise.py"
    script.write_text(_sanitiser_source())
    return subprocess.run(  # noqa: S603
        [sys.executable, str(script), str(src), str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )


def _summary_shape() -> dict:
    """Build the shape counterfactual.py writes without --keep-per-subject."""

    def arm() -> dict:
        return {
            "n_subjects": 300,
            "n_edited": 284,
            "sign_agreement": {"death": 0.71, "acute_kidney_injury": 0.52},
            "mean_delta_hazard": {"death": 0.014},
            "mean_delta_concepts": {"acute_kidney_injury": 0.09},
            "per_subject": [],
        }

    return {
        "index_hours": 24.0,
        "run_dir": "/mnt/nfs/home/krishnanam/runs/gemini_full_v10_15c",
        "edits": {
            "creatinine_plus_1": arm(),
            "creatinine_normal": dict(arm(), sign_agreement={"death": 0.49}),
            "lactate_x3": arm(),
        },
    }


def test_the_run_directory_is_stripped(tmp_path: Path) -> None:
    result = _run(_summary_shape(), tmp_path)
    assert result.returncode == 0, result.stderr
    exported = json.loads((tmp_path / "counterfactual_export.json").read_text())
    assert "run_dir" not in exported
    assert exported["index_hours"] == 24.0
    assert set(exported["edits"]) == {
        "creatinine_plus_1",
        "creatinine_normal",
        "lactate_x3",
    }
    # The aggregate numbers the paper needs survive intact.
    assert exported["edits"]["creatinine_normal"]["sign_agreement"] == {"death": 0.49}


def test_per_subject_records_are_refused_not_quietly_dropped(tmp_path: Path) -> None:
    obj = _summary_shape()
    obj["edits"]["lactate_x3"]["per_subject"] = [
        {"subject_id": 1234, "delta_hazard": {"death": 0.02}}
    ]
    result = _run(obj, tmp_path)
    assert result.returncode != 0
    assert "lactate_x3" in result.stderr
    assert "refusing to export patient-level forecasts" in result.stderr
    assert not (tmp_path / "counterfactual_export.json").exists()


def test_an_empty_per_subject_list_is_removed_from_the_export(tmp_path: Path) -> None:
    """Empty is not a reason to keep the key: the export carries aggregates only."""
    result = _run(_summary_shape(), tmp_path)
    assert result.returncode == 0, result.stderr
    exported = json.loads((tmp_path / "counterfactual_export.json").read_text())
    for arm in exported["edits"].values():
        assert "per_subject" not in arm
