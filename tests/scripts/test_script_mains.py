"""The command-line entry points of the post-processing scripts."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts import alerts_cis, intervention_cis, make_atlas_table, make_steering_table
from tests.scripts.test_make_steering_table import _payload


def _run(module, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_steering_table_main_writes_before_and_after_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(_payload()))
    after.write_text(json.dumps(_payload("suppress")))
    out = tmp_path / "dials.tex"
    _run(
        make_steering_table,
        ["--before", str(before), "--after", str(after), "--output-tex", str(out)],
        monkeypatch,
    )
    text = out.read_text()
    assert "before" in text and "after steering training" in text
    assert text.count("aki stage 3") == 2


def test_steering_table_main_refuses_mismatched_event_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(_payload()))
    other = _payload()
    other["event_names"] = ["death"]
    after.write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="different event heads"):
        _run(
            make_steering_table,
            [
                "--before",
                str(before),
                "--after",
                str(after),
                "--output-tex",
                str(tmp_path / "x.tex"),
            ],
            monkeypatch,
        )


def _atlas() -> dict:
    def row(index: int, name: str) -> dict:
        return {
            "index": index,
            "name": name,
            "norm": 1.5,
            "mean_activation": 0.4 - 0.1 * index,
            "promotes": [
                {
                    "token": f"LAB//{i}//mg::Q{i}",
                    "name": f"lab {i}",
                    "shift": 1.0 - 0.1 * i,
                }
                for i in range(6)
            ],
            "suppresses": [
                {"token": f"MED//{i}", "name": f"med {i}", "shift": -1.0}
                for i in range(6)
            ],
        }

    return {
        "run_dir": "x",
        "source": "mimic_iv",
        "n_positions": 10,
        "contribution_share": {"named": 0.1, "unknown": 0.7, "residual": 0.2},
        "known": [row(0, "tachycardia"), row(1, "fever")],
        "unknown": [row(0, "unknown_0"), row(1, "unknown_1"), row(2, "unknown_2")],
    }


def test_atlas_table_main_writes_the_requested_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = tmp_path / "atlas.json"
    atlas.write_text(json.dumps(_atlas()))
    unknown, known = tmp_path / "unknown.tex", tmp_path / "known.tex"
    _run(
        make_atlas_table,
        [
            "--atlas",
            str(atlas),
            "--unknown",
            str(unknown),
            "--known",
            str(known),
            "--top-concepts",
            "2",
            "--top-events",
            "3",
        ],
        monkeypatch,
    )
    printed = capsys.readouterr().out
    assert "contribution shares: named 0.100" in printed
    assert unknown.exists() and known.exists()
    assert "unknown 0" in unknown.read_text() and "unknown 2" not in unknown.read_text()
    assert "tachycardia" in known.read_text()


def test_intervention_cis_main_pairs_the_standard_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    per_subject = {
        mode: {str(sid): [int(rng.integers(3, 8)), 8] for sid in range(20)}
        for mode in ("none", "truth", "flip")
    }
    src = tmp_path / "per_subject.json"
    src.write_text(json.dumps(per_subject))
    out = tmp_path / "cis.json"
    _run(
        intervention_cis,
        ["--per-subject", str(src), "--output-json", str(out), "--n-boot", "50"],
        monkeypatch,
    )
    result = json.loads(out.read_text())
    assert result["n_boot"] == 50
    assert set(result["pairs"]) == {
        "truth_minus_flip",
        "truth_minus_none",
        "flip_minus_none",
    }
    pair = result["pairs"]["truth_minus_flip"]
    assert pair["ci_low"] <= pair["point"] <= pair["ci_high"]
    assert pair["n_subjects"] == 20


def test_alerts_cis_main_accepts_the_alerts_json_scorer_name_and_subsamples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    n_subjects, rows_per = 40, 3
    sids = np.repeat(np.arange(n_subjects), rows_per)
    y = np.repeat((np.arange(n_subjects) % 3 == 0).astype(float), rows_per)
    noise = rng.uniform(0, 1, len(sids))
    frame = pl.DataFrame(
        {
            "event": ["death"] * len(sids),
            "subject_id": sids,
            "visit_id": sids,
            "time_hours": np.arange(len(sids), dtype=float),
            "y@8h": y,
            "hazard@8h": np.where(y == 1, 0.4 + 0.5 * noise, 0.1 + 0.5 * noise),
            "gbm@8h": np.where(y == 1, 0.3 + 0.6 * noise, 0.1 + 0.6 * noise),
        }
    )
    dump = tmp_path / "rows.parquet"
    frame.write_parquet(dump)
    out = tmp_path / "cis.json"
    _run(
        alerts_cis,
        [
            "--dump",
            str(dump),
            "--output-json",
            str(out),
            "--scorers",
            "hazard",
            "baseline_gbm",
            "--n-boot",
            "30",
            "--max-subjects",
            "25",
        ],
        monkeypatch,
    )
    result = json.loads(out.read_text())
    assert result["scorers"] == ["hazard", "gbm"]
    assert result["max_subjects"] == 25
    cell = result["cells"]["death@8h"]
    assert cell["n"] == 25 * rows_per
    assert set(cell["scorers"]) == {"hazard", "gbm"}
