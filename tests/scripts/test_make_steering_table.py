"""The dial table bolds only expected, separated shifts and flags wrong-way ones."""

import importlib.util
import json
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_steering_table", Path("scripts/make_steering_table.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _outcome(event, horizon, ratio, *, sign, separated, as_expected):
    return {
        "event": event,
        "horizon_hours": horizon,
        "relative_change": ratio,
        "expected_sign": sign,
        "separated": separated,
        "as_expected": as_expected,
        "delta": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_subjects": 3},
        "baseline_risk": 0.1,
        "steered_risk": 0.1 * ratio,
        "agreement": 0.9,
    }


def _payload(direction="amplify"):
    return {
        "site": "stream",
        "layer_index": 4,
        "tau": 1.0,
        "event_names": ["death", "icu_admission"],
        "summaries": [
            {
                "concept": "aki_stage_3",
                "direction": direction,
                "gamma": 1.0,
                "site": "stream",
                "n_subjects": 3,
                "respond_baseline": 0.21,
                "respond_steered": 0.31,
                "respond_delta": {
                    "point": 0.1,
                    "ci_low": 0.09,
                    "ci_high": 0.11,
                    "n_subjects": 3,
                },
                "express_baseline": 0.05,
                "express_steered": 0.05,
                "express_delta": {
                    "point": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_subjects": 3,
                },
                "outcomes": [
                    _outcome(
                        "death", 24.0, 1.35, sign=+1, separated=True, as_expected=True
                    ),
                    _outcome(
                        "icu_admission",
                        24.0,
                        0.90,
                        sign=+1,
                        separated=True,
                        as_expected=False,
                    ),
                    _outcome(
                        "death", 8.0, 1.10, sign=+1, separated=False, as_expected=True
                    ),
                ],
            }
        ],
    }


def test_bold_expected_dagger_wrong_way(tmp_path) -> None:
    mod = _load()
    tex = mod.render(_payload(), None, horizon=24.0)
    assert "\\textbf{1.35}" in tex
    assert "0.90$^\\dagger$" in tex
    assert "aki stage 3 & $\\uparrow$ & 0.21$\\to$0.31" in tex
    assert "Death" in tex and "ICU" in tex


def test_after_block_is_labelled_and_refuses_mismatched_heads(tmp_path) -> None:
    mod = _load()
    tex = mod.render(_payload(), _payload("suppress"), horizon=24.0)
    assert "before steering training" in tex and "after steering training" in tex
    assert "$\\downarrow$" in tex
    other = _payload()
    other["event_names"] = ["death"]
    before, after = tmp_path / "b.json", tmp_path / "a.json"
    before.write_text(json.dumps(_payload()))
    after.write_text(json.dumps(other))
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/make_steering_table.py",
            "--before",
            str(before),
            "--after",
            str(after),
            "--output-tex",
            str(tmp_path / "o.tex"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "different event heads" in proc.stderr
