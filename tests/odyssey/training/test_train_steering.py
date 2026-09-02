"""CPU end-to-end run of the decomposed bottleneck with Steerling's steering phases.

Exercises, on real (tiny) MEDS shards through the real training script:
streamed shard loading, the decomposition with teacher forcing annealed
across steps, the lifted-token count, the calibrated steering runtime,
the steering-loss branch of the loop, init_from, and then the steering
benchmark's ``prepare``/``evaluate_steering``/CLI against the saved run.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from odyssey.inference import steering as steering_cli
from odyssey.inference.steering import evaluate_steering, prepare
from odyssey.training.train import TrainingConfig, _needs_running_labels, train


T0 = datetime(2024, 1, 1, 0, 0)


def _write_shards(shard_dir: Path, n_subjects: int, n_events: int) -> None:
    """Heart rates that trigger tachycardia on even subjects, bradycardia on odd."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, str, datetime, float | None, int | None]] = []
    for subject_id in range(n_subjects):
        base = T0 + timedelta(days=subject_id)
        for i in range(n_events):
            t = base + timedelta(hours=i)
            if i % 3 == 0:
                rate = 125.0 if subject_id % 2 == 0 else 45.0
                rows.append((subject_id, "LAB//220045//bpm", t, rate, None))
            elif i % 3 == 1:
                rows.append(
                    (
                        subject_id,
                        "LAB//220210//insp_min",
                        t,
                        15.0 + (subject_id % 10),
                        None,
                    )
                )
            else:
                rows.append((subject_id, f"DIAGNOSIS//{i % 7}", t, None, None))
    frame = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )
    frame.write_parquet(shard_dir / "0.parquet")


def _config(
    train_dir: Path, tuning_dir: Path, output_dir: Path, **overrides: Any
) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "train_shard_dir": str(train_dir),
        "tuning_shard_dir": str(tuning_dir),
        "output_dir": str(output_dir),
        "backbone": "transformer",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "attn_num_heads": 4,
        "embedding_dim": 8,
        "vocab_min_count": 1,
        "quantile_min_count": 1,
        "num_lanes": 2,
        "chunk_size": 16,
        "max_context": 32,
        "num_epochs": 1,
        "log_every": 2,
        "eval_every": 1000,
        "eval_max_chunks": 1,
        "checkpoint_every": 1000,
        "model_kind": "bottleneck",
        # stay-level supervision: the synthetic shards carry no visit ids
        "concept_supervision": "stay",
        "bottleneck_kind": "decomposed",
        "unknown_ratio": 2,
        "residual_dropout": 0.3,
        "event_hazards": True,
        "teacher_known_start": 1.0,
        "teacher_known_end": 0.5,
        "teacher_unknown_start": 1.0,
        "teacher_unknown_end": 0.5,
        "teacher_anneal_steps": 4,
        "orthogonality_weight": 0.0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_needs_running_labels_for_randint_or_steering() -> None:
    base = {
        "train_shard_dir": "a",
        "tuning_shard_dir": "b",
        "output_dir": "c",
        "randint_prob": 0.0,
    }
    assert not _needs_running_labels(TrainingConfig(**base))
    assert _needs_running_labels(TrainingConfig(**{**base, "randint_prob": 0.5}))
    assert _needs_running_labels(TrainingConfig(**base, steering_phases=1))


@pytest.fixture(scope="module")
def steered_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("steer")
    train_dir, tuning_dir = root / "data" / "train", root / "data" / "tuning"
    _write_shards(train_dir, n_subjects=10, n_events=30)
    _write_shards(tuning_dir, n_subjects=4, n_events=30)
    output_dir = root / "run"
    config = _config(
        train_dir,
        tuning_dir,
        output_dir,
        stream_shards=True,
        steering_phases=2,
        steering_phase_steps=2,
        steering_warmup_steps=1,
        steering_tau=1.0,
        lifted_patients=6,
        lifted_min_count=1,
        lifted_min_share=0.0,
        lifted_min_lift=1.0,
    )
    assert train(config) == output_dir
    return {"train": train_dir, "tuning": tuning_dir, "run": output_dir}


def test_steering_phases_train_and_log_their_losses(
    steered_run: dict[str, Path],
) -> None:
    run = steered_run["run"]
    assert (run / "checkpoint_final.pt").exists()
    lines = [
        json.loads(line) for line in (run / "loss_log.jsonl").read_text().splitlines()
    ]
    train_lines = [rec for rec in lines if rec.get("split") == "train"]
    assert train_lines
    steered = [rec for rec in train_lines if "respond_loss" in rec]
    assert steered, "no logged step went through the steering-loss branch"
    assert all(rec["respond_loss"] >= 0.0 for rec in steered)


def test_init_from_starts_a_fresh_run_from_saved_weights(
    steered_run: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(
        steered_run["train"],
        steered_run["tuning"],
        tmp_path / "warm",
        init_from=str(steered_run["run"] / "checkpoint_final.pt"),
    )
    train(config)
    assert (tmp_path / "warm" / "checkpoint_final.pt").exists()
    both = _config(
        steered_run["train"],
        steered_run["tuning"],
        tmp_path / "bad",
        init_from=str(steered_run["run"] / "checkpoint_final.pt"),
        resume_from=str(steered_run["run"] / "checkpoint_final.pt"),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(both)


def test_prepare_and_evaluate_steering_on_the_saved_run(
    steered_run: dict[str, Path],
) -> None:
    prepared = prepare(
        steered_run["run"],
        steered_run["tuning"],
        steered_run["train"],
        max_shards=None,
        lift_shards=1,
        tau=1.0,
        device="cpu",
        checkpoint_path=None,
        num_lanes=2,
        chunk_size=16,
        min_share=0.0,
        min_lift=1.0,
    )
    assert "tachycardia" in prepared.concept_names
    assert prepared.tables is not None
    assert len(prepared.gammas) == len(prepared.concept_names)
    summaries = evaluate_steering(
        prepared,
        concepts=["tachycardia"],
        site="stream",
        layer_index=0,
        suppress_strength=None,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
        n_boot=10,
    )
    assert [s.direction for s in summaries] == ["amplify", "suppress"]
    assert summaries[0].n_subjects == 4


def test_steering_cli_writes_json_and_refuses_to_overwrite(
    steered_run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "steering.json"
    argv = [
        "steering",
        "--run-dir",
        str(steered_run["run"]),
        "--held-out-shard-dir",
        str(steered_run["tuning"]),
        "--lift-shard-dir",
        str(steered_run["train"]),
        "--output-json",
        str(out),
        "--concepts",
        "tachycardia",
        "--site",
        "bottleneck",
        "--num-lanes",
        "2",
        "--chunk-size",
        "16",
        "--n-boot",
        "10",
        "--min-share",
        "0",
        "--min-lift",
        "1",
        "--lift-shards",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    steering_cli._main()
    payload = json.loads(out.read_text())
    assert payload["site"] == "bottleneck"
    assert [s["concept"] for s in payload["summaries"]] == [
        "tachycardia",
        "tachycardia",
    ]
    assert "tachycardia" in payload["gammas"]
    with pytest.raises(SystemExit):
        steering_cli._main()
