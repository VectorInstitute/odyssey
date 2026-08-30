"""GPU-only integration test for the inference/eval script.

Trains a tiny model for a handful of steps (reusing the same synthetic
shard shape as test_train_gpu.py), then runs evaluate_run against a
separate held-out split built from the same code vocabulary. Validates
that the full load-checkpoint -> stream -> score pipeline wires
together end to end, not that the tiny model's metrics are any good.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

import polars as pl  # noqa: E402

from odyssey.inference.run_inference import evaluate_run, load_run  # noqa: E402
from odyssey.training.train import TrainingConfig, train  # noqa: E402


T0 = datetime(2024, 1, 1, 0, 0)


def _write_shards(shard_dir: Path, n_subjects: int, n_events_per_subject: int) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, str, datetime, float | None, int | None]] = []
    for subject_id in range(n_subjects):
        base = T0 + timedelta(days=subject_id)
        # One admission per subject (real hadm_id, constant across their
        # events) -- visit-scoped concept supervision is the default now
        # (odyssey.training.train.TrainingConfig.concept_supervision), and
        # needs a real hadm_id somewhere to have any visit to supervise at
        # all; an all-None hadm_id column here previously produced zero
        # visits, so evaluate_run had nothing to pool and the pipeline
        # this test exists to validate was silently never exercised.
        hadm_id = 10_000 + subject_id
        for i in range(n_events_per_subject):
            t = base + timedelta(hours=i)
            if i % 3 == 0:
                rows.append(
                    (
                        subject_id,
                        "LAB//220045//bpm",
                        t,
                        70.0 + 10 * (subject_id % 5),
                        hadm_id,
                    )
                )
            elif i % 3 == 1:
                rows.append(
                    (
                        subject_id,
                        "LAB//220210//insp_min",
                        t,
                        15.0 + (subject_id % 10),
                        hadm_id,
                    )
                )
            else:
                rows.append((subject_id, f"DIAGNOSIS//{i % 7}", t, None, hadm_id))
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


@cuda_required
def test_evaluate_run_end_to_end(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    held_out_dir = tmp_path / "data" / "held_out"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)
    _write_shards(held_out_dir, n_subjects=6, n_events_per_subject=30)

    run_dir = tmp_path / "run"
    config = TrainingConfig(
        train_shard_dir=str(train_dir),
        tuning_shard_dir=str(tuning_dir),
        output_dir=str(run_dir),
        hidden_size=64,
        num_hidden_layers=2,
        mamba_state_size=16,
        mamba_headdim=64,
        mamba_chunk_size=16,
        attn_num_heads=8,
        embedding_dim=8,
        vocab_min_count=1,
        quantile_min_count=1,
        num_lanes=2,
        chunk_size=8,
        num_epochs=1,
        log_every=2,
        eval_every=100,
        checkpoint_every=100000,
    )
    train(config)
    assert (run_dir / "checkpoint_final.pt").exists()

    results = evaluate_run(run_dir, held_out_dir, num_lanes=2, chunk_size=8)

    assert results.n_patient_ends_scored > 0
    assert torch.isfinite(torch.tensor(results.task_metrics.cross_entropy))
    assert torch.isfinite(torch.tensor(results.task_metrics.perplexity))
    assert 0.0 <= results.task_metrics.top1_accuracy <= 1.0
    assert torch.isfinite(torch.tensor(results.orthogonality))
    assert len(results.concept_metrics) > 0
    assert len(results.observability_metrics) > 0
    for m in results.observability_metrics:
        # every subject has a real observability target -- unlike
        # concept_metrics, n_subjects should exactly match what was scored.
        assert m.n_subjects == results.n_patient_ends_scored


@cuda_required
def test_evaluate_run_honours_an_explicit_checkpoint_path(tmp_path: Path) -> None:
    # Real usage picks checkpoint_best.pt over wherever training happened
    # to stop -- prove checkpoint_path is actually the file that gets
    # loaded, not just accepted and ignored, by deleting every other
    # checkpoint in the run dir before evaluating: if load_run silently
    # fell back to _latest_checkpoint's default search, this would raise
    # FileNotFoundError instead of succeeding.
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    held_out_dir = tmp_path / "data" / "held_out"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)
    _write_shards(held_out_dir, n_subjects=6, n_events_per_subject=30)

    run_dir = tmp_path / "run"
    config = TrainingConfig(
        train_shard_dir=str(train_dir),
        tuning_shard_dir=str(tuning_dir),
        output_dir=str(run_dir),
        hidden_size=64,
        num_hidden_layers=2,
        mamba_state_size=16,
        mamba_headdim=64,
        mamba_chunk_size=16,
        attn_num_heads=8,
        embedding_dim=8,
        vocab_min_count=1,
        quantile_min_count=1,
        num_lanes=2,
        chunk_size=8,
        num_epochs=1,
        log_every=2,
        eval_every=5,
        checkpoint_every=100000,
    )
    train(config)
    best_path = run_dir / "checkpoint_best.pt"
    assert best_path.exists()

    for other in run_dir.glob("checkpoint_*.pt"):
        if other != best_path:
            other.unlink()

    results = evaluate_run(
        run_dir, held_out_dir, num_lanes=2, chunk_size=8, checkpoint_path=best_path
    )
    assert results.n_patient_ends_scored > 0

    _, _, _, _ = load_run(run_dir, checkpoint_path=best_path)
