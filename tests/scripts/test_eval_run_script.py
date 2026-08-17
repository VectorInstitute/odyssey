"""The canonical evaluation script plans the right stages for each model kind."""

import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "eval_run.sh"


def _plan(tmp_path: Path, model_kind: str) -> str:
    run_dir = tmp_path / f"run_{model_kind}"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"model_kind": model_kind}))
    data = tmp_path / "data"
    (data / "held_out").mkdir(parents=True)
    (data / "train").mkdir()
    env = dict(
        os.environ, DRY_RUN="1", PYTHON="python3", LANES="16", BASELINE_SHARDS="30"
    )
    out = subprocess.run(
        ["bash", str(SCRIPT), str(run_dir), str(data)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return out.stdout


def test_bottleneck_plan_runs_every_stage(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "bottleneck")
    for stage in ("eval", "interventions", "alerts", "cases", "report"):
        assert f"=== STAGE {stage} EXIT 0 (dry) ===" in plan, stage
    # printf %q quotes each argument, so check flag and value as tokens
    tokens = plan.replace("'", "").split()
    assert "--uncertain-band" in tokens and "0.15" in tokens
    assert "--max-baseline-shards" in tokens and "30" in tokens
    assert "--num-lanes" in tokens and "16" in tokens
    assert "--interventions" in tokens  # report gets the banded file


def test_baseline_plan_skips_bottleneck_only_stages(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "baseline")
    assert "=== STAGE interventions SKIPPED (baseline model) ===" in plan
    assert "=== STAGE cases SKIPPED" in plan
    for stage in ("eval", "alerts", "report"):
        assert f"=== STAGE {stage} EXIT 0 (dry) ===" in plan, stage
    assert "--interventions" not in plan
