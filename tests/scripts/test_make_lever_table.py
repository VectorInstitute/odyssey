"""The lever table: paired CIs where banked, point values where not."""

import json
from pathlib import Path

from scripts.make_lever_table import render


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _band(none: float, truth: float, flip: float) -> list[dict]:
    return [
        {"mode": "none", "top1_accuracy": none, "mean_task_loss": 3.0},
        {"mode": "truth", "top1_accuracy": truth, "mean_task_loss": 3.016},
        {"mode": "flip", "top1_accuracy": flip, "mean_task_loss": 3.017},
    ]


def test_ci_rows_are_bold_when_separated_and_point_rows_are_starred(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "a" / "intervention_cis.json",
        {
            "pairs": {
                "truth_minus_flip": {
                    "point": -0.00146,
                    "ci_low": -0.00156,
                    "ci_high": -0.00135,
                    "separated": True,
                },
                "truth_minus_none": {
                    "point": -0.00195,
                    "ci_low": -0.00201,
                    "ci_high": -0.00189,
                    "separated": True,
                },
                "truth_calibrated_minus_flip_calibrated": {
                    "point": 0.00019,
                    "ci_low": -0.00058,
                    "ci_high": 0.00109,
                    "separated": False,
                },
            }
        },
    )
    _write(tmp_path / "a" / "interventions_band15.json", _band(0.3725, 0.3706, 0.3720))
    _write(tmp_path / "b" / "interventions_band15.json", _band(0.1491, 0.1498, 0.1446))
    text = render(tmp_path, (("With CIs", "a"), ("Points only", "b"), ("Missing", "c")))
    lines = [
        line
        for line in text.splitlines()
        if line.endswith("\\\\") and "Arm &" not in line
    ]
    assert len(lines) == 2  # the arm with nothing banked is skipped
    with_cis, points = lines
    assert with_cis.startswith("With CIs & \\textbf{$-0.15$ [-0.16, -0.14]}")
    assert "$+0.02$ [-0.06, +0.11]" in with_cis and "\\textbf{$+0.02$" not in with_cis
    assert with_cis.rstrip(" \\").endswith("$+0.016$")  # loss delta from the band file
    assert points.startswith("Points only$^\\ast$ & $+0.52$ & $+0.07$ & --")
