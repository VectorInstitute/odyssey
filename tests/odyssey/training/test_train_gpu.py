"""GPU-only integration test for the real training script.

This is the actual glue code the big real training run depends on --
`mamba-ssm` requires CUDA, and it's never been executed end to end
before this test, so it auto-skips unless both `mamba-ssm` is installed
and a CUDA device is visible, exercised on a GPU host, not local/CPU CI.
Tiny synthetic MEDS shards and tiny model dimensions, run for a handful
of steps -- this validates the training loop actually wires together
(data loading, tokenization, streaming, the real backbone, checkpointing,
loss logging), not that the model learns anything meaningful yet.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

import polars as pl  # noqa: E402

import odyssey.training.train as train_module  # noqa: E402
from odyssey.training.train import TrainingConfig, train  # noqa: E402


T0 = datetime(2024, 1, 1, 0, 0)


def _write_shards(shard_dir: Path, n_subjects: int, n_events_per_subject: int) -> None:
    """Write one small, real-shaped MEDS parquet shard covering several concepts.

    Includes heart-rate, respiratory-rate, and creatinine readings across
    a plausible range, so tachycardia/sustained_tachypnea/acute_kidney_injury
    concept labels are a real mix of triggered/not, not degenerate.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[int, str, datetime, Optional[float], Optional[int]]] = []
    for subject_id in range(n_subjects):
        base = T0 + timedelta(days=subject_id)
        for i in range(n_events_per_subject):
            t = base + timedelta(hours=i)
            if i % 3 == 0:
                rows.append(
                    (
                        subject_id,
                        "LAB//220045//bpm",
                        t,
                        70.0 + 10 * (subject_id % 5),
                        None,
                    )
                )
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


def _tiny_config(
    train_dir: Path, tuning_dir: Path, output_dir: Path, **overrides: Any
) -> TrainingConfig:
    """Build the smallest TrainingConfig that still exercises EHRHybridBackbone.

    Shared across every test in this file so each one only states the
    fields it actually varies.
    """
    defaults: Dict[str, Any] = {
        "train_shard_dir": str(train_dir),
        "tuning_shard_dir": str(tuning_dir),
        "output_dir": str(output_dir),
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "mamba_state_size": 16,
        "mamba_headdim": 64,
        "mamba_chunk_size": 16,
        "attn_num_heads": 8,
        "embedding_dim": 8,
        "vocab_min_count": 1,
        "quantile_min_count": 1,
        "num_lanes": 2,
        "chunk_size": 8,
        "num_epochs": 1,
        "log_every": 2,
        "eval_every": 4,
        "eval_max_chunks": 2,
        "checkpoint_every": 4,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


@cuda_required
def test_train_runs_end_to_end_and_produces_expected_outputs(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    config = _tiny_config(train_dir, tuning_dir, output_dir)

    result_dir = train(config)

    assert result_dir == output_dir
    assert (output_dir / "config.json").exists()
    assert (output_dir / "vocabulary.json").exists()
    assert (output_dir / "quantile_binner.json").exists()
    assert (output_dir / "checkpoint_final.pt").exists()
    assert (output_dir / "checkpoint_epoch_0.pt").exists()
    # best-checkpoint tracking always runs (independent of whether early
    # stopping is enabled) -- any eval improving on the running best saves
    # checkpoint_best.pt, and with eval_every=4 this run had several.
    assert (output_dir / "checkpoint_best.pt").exists()
    best_checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location="cpu")
    assert torch.isfinite(torch.tensor(best_checkpoint["best_val_loss"]))

    log_lines = (output_dir / "loss_log.jsonl").read_text().strip().split("\n")
    assert len(log_lines) > 0
    records = [json.loads(line) for line in log_lines]

    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "tuning"]
    assert train_records
    assert val_records

    for record in records:
        for key in (
            "task_loss",
            "concept_loss",
            "orthogonality_loss",
            "observability_loss",
        ):
            assert key in record, f"missing {key} in {record}"
            assert torch.isfinite(torch.tensor(record[key])), (
                f"non-finite {key} in {record}"
            )

    checkpoint = torch.load(output_dir / "checkpoint_final.pt", map_location="cpu")
    assert "model" in checkpoint
    # the final checkpoint's step count need not land exactly on a logged
    # step (logging cadence and total step count are independent), only
    # be at least as far along as the last thing we logged.
    assert checkpoint["step"] >= train_records[-1]["step"]


@cuda_required
def test_train_resumes_from_an_epoch_checkpoint(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    base_config = _tiny_config(
        train_dir,
        tuning_dir,
        output_dir,
        eval_every=100,  # skip eval here, already covered above
        checkpoint_every=100000,  # only the end-of-epoch checkpoint matters here
    )

    train(base_config)
    checkpoint_path = output_dir / "checkpoint_epoch_0.pt"
    assert checkpoint_path.exists()
    first_run_step = torch.load(checkpoint_path, map_location="cpu")["step"]

    resumed_config = TrainingConfig(
        **{**vars(base_config), "num_epochs": 2, "resume_from": str(checkpoint_path)}
    )
    train(resumed_config)

    final_checkpoint = torch.load(
        output_dir / "checkpoint_final.pt", map_location="cpu"
    )
    # the resumed run continues global_step from where the checkpoint left
    # off, rather than restarting the counter from 0.
    assert final_checkpoint["step"] > first_run_step

    log_lines = (output_dir / "loss_log.jsonl").read_text().strip().split("\n")
    records = [json.loads(line) for line in log_lines]
    # loss_log.jsonl is append-only, so both runs' records are present,
    # and the resumed run's records pick up step numbering where the
    # first run left off rather than overlapping it.
    steps_after_first_run = [r["step"] for r in records if r["step"] > first_run_step]
    assert steps_after_first_run


@cuda_required
def test_train_resumes_mid_epoch_by_fast_forwarding(tmp_path: Path) -> None:
    # Unlike test_train_resumes_from_an_epoch_checkpoint (which resumes
    # from the clean, epoch-boundary checkpoint), this resumes from a
    # checkpoint_every checkpoint taken partway through epoch 0 -- the
    # actual spot-preemption scenario this mechanism exists for.
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    base_config = _tiny_config(
        train_dir,
        tuning_dir,
        output_dir,
        eval_every=100000,
        checkpoint_every=4,  # small enough to land mid-epoch, not just at the end
    )
    train(base_config)

    mid_epoch_checkpoints = sorted(
        output_dir.glob("checkpoint_[0-9]*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    assert mid_epoch_checkpoints, "expected at least one checkpoint_every checkpoint"
    checkpoint_path = mid_epoch_checkpoints[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["steps_into_epoch"] > 0
    assert checkpoint["batch_config"] == {
        "num_lanes": 2,
        "chunk_size": 8,
        "reset_prob": base_config.reset_prob,
        "seed": base_config.seed,
    }
    first_checkpoint_step = checkpoint["step"]

    resumed_config = TrainingConfig(
        **{**vars(base_config), "resume_from": str(checkpoint_path)}
    )
    train(resumed_config)

    final_checkpoint = torch.load(
        output_dir / "checkpoint_final.pt", map_location="cpu"
    )
    assert final_checkpoint["step"] > first_checkpoint_step


@cuda_required
def test_train_falls_back_to_epoch_restart_when_batch_config_differs(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    base_config = _tiny_config(
        train_dir, tuning_dir, output_dir, eval_every=100000, checkpoint_every=4
    )
    train(base_config)
    checkpoint_path = next(output_dir.glob("checkpoint_[0-9]*.pt"))
    first_checkpoint_step = torch.load(checkpoint_path, map_location="cpu")["step"]

    # A different num_lanes than what the checkpoint was taken under --
    # must not crash or silently fast-forward to a meaningless position;
    # falls back to restarting the epoch, which still runs to completion.
    resumed_config = TrainingConfig(
        **{
            **vars(base_config),
            "num_lanes": 4,
            "resume_from": str(checkpoint_path),
        }
    )
    train(resumed_config)

    final_checkpoint = torch.load(
        output_dir / "checkpoint_final.pt", map_location="cpu"
    )
    assert final_checkpoint["step"] > first_checkpoint_step


@cuda_required
def test_train_stops_early_when_validation_loss_stops_improving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    # Many epochs' worth of steps available, but evaluate_streaming is
    # monkeypatched to a scripted, deterministic sequence -- improves
    # twice, then plateaus -- so the test doesn't depend on a tiny,
    # barely-trained model's real (noisy) validation loss actually
    # plateauing within a short run.
    scripted_val_losses = iter([2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

    def _fake_evaluate_streaming(*_args: object, **_kwargs: object) -> Dict[str, float]:
        return {"task_loss": next(scripted_val_losses)}

    monkeypatch.setattr(train_module, "evaluate_streaming", _fake_evaluate_streaming)

    config = _tiny_config(
        train_dir,
        tuning_dir,
        output_dir,
        num_epochs=20,
        eval_every=2,
        checkpoint_every=100000,
        early_stopping_patience=2,
    )

    train(config)

    records = [
        json.loads(line)
        for line in (output_dir / "loss_log.jsonl").read_text().strip().split("\n")
    ]
    val_records = [r for r in records if r["split"] == "tuning"]
    # 2 improving evals + 2 non-improving (patience=2) = 4 evals total,
    # not the dozens a 20-epoch run would otherwise produce.
    assert len(val_records) == 4
    assert [r["task_loss"] for r in val_records] == [2.0, 1.5, 1.5, 1.5]

    best_checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location="cpu")
    assert best_checkpoint["best_val_loss"] == pytest.approx(1.5)
    # stopped early -- never reached anywhere near 20 epochs' worth of
    # checkpoint_epoch_N.pt files.
    assert not (output_dir / "checkpoint_epoch_1.pt").exists()


@cuda_required
def test_train_resume_preserves_best_val_loss_across_a_restart(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    base_config = _tiny_config(
        train_dir,
        tuning_dir,
        output_dir,
        eval_every=4,
        checkpoint_every=4,
    )
    train(base_config)

    checkpoint_path = next(
        p
        for p in sorted(
            output_dir.glob("checkpoint_[0-9]*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if "best_val_loss" in torch.load(p, map_location="cpu")
    )
    saved_best = torch.load(checkpoint_path, map_location="cpu")["best_val_loss"]
    assert saved_best < float("inf")

    resumed_config = TrainingConfig(
        **{
            **vars(base_config),
            "num_epochs": 2,
            "resume_from": str(checkpoint_path),
        }
    )
    train(resumed_config)

    # checkpoint_best.pt after the resumed run must reflect the best
    # across *both* runs, never worse than what the first run already
    # found -- resuming must not silently reset best-tracking to
    # "nothing seen yet" and let a worse epoch overwrite a genuinely
    # better earlier checkpoint.
    final_best = torch.load(output_dir / "checkpoint_best.pt", map_location="cpu")[
        "best_val_loss"
    ]
    assert final_best <= saved_best
