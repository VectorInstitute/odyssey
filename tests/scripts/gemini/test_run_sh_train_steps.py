"""The GEMINI train steps build the configs their names promise.

Regression guard for a real gap (2026-08-31): every GEMINI train step
was ``model_kind=baseline`` with ``event_hazards=false``, so running any
of them produced a forecasting-only checkpoint carrying none of the
trust-audit results (readout, completeness, lever) GEMINI is in the
paper for -- G3 would have burned hours of H200 time on output it could
not use. The bottleneck steps are exercised here by running the step's
own config-writing block, not by re-encoding the expected config in the
test: the failure being guarded is exactly a config that says something
other than what the step is named for.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from odyssey.data.alert_events import alert_events_for
from odyssey.training.train import TrainingConfig


REPO = Path(__file__).resolve().parents[3]
RUN_SH = REPO / "scripts" / "gemini" / "run.sh"


def _cbm_config(
    tmp_path: Path, max_train_shards: str, bottleneck_kind: str | None = None
) -> dict:
    """Run run.sh's own embedded python config builder for the CBM step.

    Extracts the heredoc from ``run_train_cbm`` and executes it exactly as
    the step does, so the test reads the shipped source rather than a
    copy of it.
    """
    source = RUN_SH.read_text()
    body = source.split("run_train_cbm() {", 1)[1]
    match = re.search(r"python3 - .*?<<'PY'\n(.*?)\nPY\n", body, re.DOTALL)
    assert match, "run_train_cbm's config-builder heredoc not found"
    script = tmp_path / "build_config.py"
    script.write_text(match.group(1))
    out = tmp_path / "config.json"
    argv = ["python3", str(script), str(out), max_train_shards]
    if bottleneck_kind is not None:
        argv.append(bottleneck_kind)
    subprocess.run(argv, check=True, capture_output=True)
    return json.loads(out.read_text())


def test_cbm_step_trains_a_concept_bottleneck_with_hazard_heads(
    tmp_path: Path,
) -> None:
    config = _cbm_config(tmp_path, "")
    assert config["model_kind"] == "bottleneck"
    assert config["event_hazards"] is True
    assert config["source"] == "gemini"
    assert config["task_set"] == "v3"
    assert config["concept_supervision"] == "visit"
    # No cap at full scale: TrainingConfig's own default means every shard.
    assert "max_train_shards" not in config


def test_cbm_smoke_step_caps_shards_but_keeps_the_same_recipe(
    tmp_path: Path,
) -> None:
    smoke = _cbm_config(tmp_path, "5")
    full = _cbm_config(tmp_path, "")
    assert smoke["max_train_shards"] == 5
    assert {k: v for k, v in smoke.items() if k != "max_train_shards"} == full


def test_cbm_config_matches_the_mimic_eicu_flagship_geometry(
    tmp_path: Path,
) -> None:
    """Cross-dataset comparability with the flagship runs.

    R2/R6 ran 64 lanes x 512 chunk at reset_prob 0.0 with a 256-wide
    hazard readout.
    """
    config = _cbm_config(tmp_path, "")
    assert config["num_lanes"] == 64
    assert config["chunk_size"] == 512
    assert config["reset_prob"] == 0.0
    assert config["event_head_hidden"] == 256
    assert config["value_embeddings"] is True


def test_cbm_config_leaves_medication_normalization_off(tmp_path: Path) -> None:
    """GEMINI medication codes need no normalization.

    The normalizer targets MIMIC sig-line/NDC and eICU HICL shapes;
    GEMINI's codes are already clean 3-part strings.
    """
    assert _cbm_config(tmp_path, "")["normalize_medications"] is False


def test_cbm_config_is_accepted_by_trainingconfig(tmp_path: Path) -> None:
    """Every key must be a real TrainingConfig field.

    An unknown one raises on the node, hours into the operator's queue,
    rather than here.
    """
    config = _cbm_config(tmp_path, "")
    built = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        **config,
    )
    assert built.model_kind == "bottleneck"
    assert built.source == "gemini"


def test_cbm_run_resolves_hazard_events_to_real_gemini_codes(
    tmp_path: Path,
) -> None:
    """The heads this config builds must name codes GEMINI actually emits.

    GEMINI writes bare structural tokens (ICU_ADMISSION, ADMISSION); the
    MIMIC-shaped "ICU_ADMISSION//" / "HOSPITAL_ADMISSION//" prefixes match
    nothing there, which trains silently-censored heads rather than
    erroring.
    """
    config = _cbm_config(tmp_path, "")
    events = {
        a.name: a for a in alert_events_for(config["task_set"], source=config["source"])
    }
    assert events["icu_admission"].code_prefix == "ICU_ADMISSION"
    assert events["readmission_30d"].code_prefix == "ADMISSION"
    assert "sepsis3" not in events  # drops with its unresolved concept


@pytest.mark.parametrize("step", ["train-smoke-cbm", "train-full-cbm"])
def test_new_steps_are_dispatched_and_documented(step: str) -> None:
    source = RUN_SH.read_text()
    assert f"{step})" in source, f"{step} has no case branch"
    assert step in source.split("unknown step:")[1], f"{step} missing from usage error"


def test_dec_step_adds_the_v12_decomposition_block_and_nothing_else(
    tmp_path: Path,
) -> None:
    """train-full-dec is train-full-cbm plus the MIMIC/eICU v12 block."""
    mixture = _cbm_config(tmp_path, "")
    decomposed = _cbm_config(tmp_path, "", "decomposed")
    assert decomposed["bottleneck_kind"] == "decomposed"
    assert decomposed["unknown_ratio"] == 3
    assert decomposed["residual_dropout"] == 0.3
    assert decomposed["reconstruction_weight"] == 1.0
    assert decomposed["independence_weight"] == 1.0
    assert decomposed["teacher_known_end"] == 0.5
    assert decomposed["teacher_anneal_steps"] == 4500
    assert "bottleneck_kind" not in mixture
    for key, value in mixture.items():
        assert decomposed[key] == value, key
    built = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        **decomposed,
    )
    assert built.bottleneck_kind == "decomposed"


def test_unknown_bottleneck_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _cbm_config(tmp_path, "", "hybrid")
