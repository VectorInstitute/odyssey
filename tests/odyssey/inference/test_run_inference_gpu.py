"""GPU-only integration test for the inference/eval script.

Trains a tiny model for a handful of steps (reusing the same synthetic
shard shape as test_train_gpu.py), then runs evaluate_run against a
separate held-out split built from the same code vocabulary. Validates
that the full load-checkpoint -> stream -> score pipeline wires
together end to end, not that the tiny model's metrics are any good.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

import polars as pl  # noqa: E402

from odyssey.inference.run_inference import evaluate_run  # noqa: E402
from odyssey.training.train import TrainingConfig, train  # noqa: E402


T0 = datetime(2024, 1, 1, 0, 0)


def _write_shards(shard_dir: Path, n_subjects: int, n_events_per_subject: int) -> None:
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
