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
    for stage in ("eval", "interventions", "attribution", "alerts", "cases", "report"):
        assert f"=== STAGE {stage} EXIT 0 (dry) ===" in plan, stage
    # printf %q quotes each argument, so check flag and value as tokens
    tokens = plan.replace("'", "").split()
    assert "--uncertain-band" in tokens and "0.15" in tokens
    # The Guide Labs comparison modes ride in the interventions stage.
    assert "flip_gated" in tokens
    assert "truth_calibrated" in tokens and "flip_calibrated" in tokens
    assert "--calibrated-tau" in tokens
    assert "--max-baseline-shards" in tokens and "30" in tokens
    assert "--num-lanes" in tokens and "16" in tokens
    assert "--interventions" in tokens  # report gets the banded file
    assert "--dump-rows" in tokens  # per-index-row table for error analysis


def test_overwrite_flag_defaults_off_and_never_appears_in_the_plan(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, "bottleneck")
    assert "--overwrite" not in plan


def test_overwrite_env_var_forwards_the_flag_to_every_stage(tmp_path: Path) -> None:
    """OVERWRITE=1 must reach eval/interventions/alerts's own --overwrite guard."""
    run_dir = tmp_path / "run_bottleneck"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"model_kind": "bottleneck"}))
    data = tmp_path / "data"
    (data / "held_out").mkdir(parents=True)
    (data / "train").mkdir()
    env = dict(os.environ, DRY_RUN="1", PYTHON="python3", OVERWRITE="1")

    out = subprocess.run(
        ["bash", str(SCRIPT), str(run_dir), str(data)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    for stage in ("eval", "interventions", "attribution", "alerts"):
        assert f"=== STAGE {stage} EXIT 0 (dry) ===" in out.stdout, stage
    tokens = out.stdout.replace("'", "").split()
    assert tokens.count("--overwrite") == 4  # eval, interventions, attribution, alerts


def test_baseline_plan_skips_bottleneck_only_stages(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "baseline")
    assert "=== STAGE interventions SKIPPED (baseline model) ===" in plan
    assert "=== STAGE cases SKIPPED" in plan
    for stage in ("eval", "alerts", "report"):
        assert f"=== STAGE {stage} EXIT 0 (dry) ===" in plan, stage
    assert "--interventions" not in plan


def test_missing_config_json_fails_loudly_instead_of_silently_skipping_stages(
    tmp_path: Path,
) -> None:
    # Real bug this guards against: MODEL_KIND used to be parsed via a
    # command substitution whose failure (missing/malformed config.json)
    # was never checked -- MODEL_KIND silently became "", and every
    # `[ "$MODEL_KIND" = "bottleneck" ]` check then evaluated false,
    # silently skipping interventions/cases for what may actually be a
    # bottleneck run, with no error surfaced at all.
    run_dir = tmp_path / "run_no_config"
    run_dir.mkdir()  # no config.json written
    data = tmp_path / "data"
    (data / "held_out").mkdir(parents=True)
    (data / "train").mkdir()
    env = dict(os.environ, DRY_RUN="1", PYTHON="python3")

    out = subprocess.run(
        ["bash", str(SCRIPT), str(run_dir), str(data)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert out.returncode != 0
    assert "could not read model_kind" in out.stderr
    assert "STAGE" not in out.stdout  # never got far enough to plan any stage


def test_odyssey_repo_override_is_used_for_the_logged_commit(tmp_path: Path) -> None:
    # ODYSSEY_REPO must actually be honored, not just the PYTHON venv
    # default -- it's also what the logged `commit=` hash is read from.
    run_dir = tmp_path / "run_bottleneck"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"model_kind": "bottleneck"}))
    data = tmp_path / "data"
    (data / "held_out").mkdir(parents=True)
    (data / "train").mkdir()
    fake_repo = tmp_path / "not_home_odyssey"
    fake_repo.mkdir()
    env = dict(
        os.environ,
        DRY_RUN="1",
        PYTHON="python3",
        ODYSSEY_REPO=str(fake_repo),
    )

    out = subprocess.run(
        ["bash", str(SCRIPT), str(run_dir), str(data)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    # fake_repo isn't a git repo, so the commit lookup correctly falls
    # back to "unknown" -- the point is it tried ODYSSEY_REPO, not $HOME/odyssey.
    assert "commit=unknown" in out.stdout
