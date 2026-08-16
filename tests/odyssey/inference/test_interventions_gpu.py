"""GPU-only integration test for the concept-intervention harness.

Trains a tiny model (same synthetic shard shape as test_run_inference_gpu.py),
then runs evaluate_interventions against a held-out split with modes that
actually apply a per-position intervention ("truth"/"flip"), not just
"none" -- that's the code path a CPU-only test can't cover, since
concept_labels/concept_mask end up as real CUDA tensors here, and a
CPU-vs-CUDA device mismatch in that path (confirmed the hard way, on a
real held-out run) is invisible when everything is on CPU.
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

from odyssey.inference.interventions import evaluate_interventions  # noqa: E402
from odyssey.training.train import TrainingConfig, train  # noqa: E402


T0 = datetime(2024, 1, 1, 0, 0)


def _write_shards(shard_dir: Path, n_subjects: int, n_events_per_subject: int) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[int, str, datetime, Optional[float], Optional[int]]] = []
    for subject_id in range(n_subjects):
        base = T0 + timedelta(days=subject_id)
        # Real hadm_id (see test_run_inference_gpu.py's own note): visit-
        # scoped supervision is the default and needs one to have any
        # visit to intervene on at all.
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
def test_evaluate_interventions_end_to_end(tmp_path: Path) -> None:
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

    results = evaluate_interventions(
        run_dir,
        held_out_dir,
        modes=["none", "truth", "flip", "zero_known"],
        num_lanes=2,
        chunk_size=8,
    )

    assert [r.mode for r in results] == ["none", "truth", "flip", "zero_known"]
    for r in results:
        assert r.n_predictions > 0
        assert torch.isfinite(torch.tensor(r.top1_accuracy))
        assert torch.isfinite(torch.tensor(r.mean_task_loss))
    # truth/flip actually intervened somewhere; zero_known edits embeddings
    # directly and never touches per-position probs.
    assert results[1].n_intervened_positions > 0
    assert results[2].n_intervened_positions > 0
    assert results[3].n_intervened_positions == 0
