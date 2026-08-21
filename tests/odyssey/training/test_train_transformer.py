"""CPU-runnable end-to-end integration test for backbone="transformer".

Unlike test_train_gpu.py (EHRHybridBackbone needs CUDA/mamba-ssm, so that
file auto-skips off a GPU host), TransformerBackbone has no such
dependency: this exercises the *real* training script -- real MEDS shard
loading, tokenization, PackedContextSampler, checkpointing, loss logging,
and load_run -- on ordinary CI/dev hardware. Tiny synthetic shards and
tiny model dimensions, run for a handful of steps: this validates the
loop wires together and a checkpoint round-trips, not that the model
learns anything meaningful.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.inference.run_inference import load_run
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.training.train import TrainingConfig, train


T0 = datetime(2024, 1, 1, 0, 0)


def _write_shards(shard_dir: Path, n_subjects: int, n_events_per_subject: int) -> None:
    """Write one small, real-shaped MEDS parquet shard covering several concepts.

    Same fixture as test_train_gpu.py's, duplicated rather than imported
    since that module is gated behind pytest.importorskip("mamba_ssm") at
    collection time and must stay skippable independently of this file.
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
    """Build the smallest TrainingConfig that still exercises TransformerBackbone."""
    defaults: Dict[str, Any] = {
        "train_shard_dir": str(train_dir),
        "tuning_shard_dir": str(tuning_dir),
        "output_dir": str(output_dir),
        "backbone": "transformer",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "attn_num_heads": 4,
        "embedding_dim": 8,
        "vocab_min_count": 1,
        "quantile_min_count": 1,
        "num_lanes": 2,
        "max_context": 32,
        "num_epochs": 1,
        "log_every": 2,
        "eval_every": 4,
        "eval_max_chunks": 2,
        "checkpoint_every": 4,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


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
    assert (output_dir / "loss_log.jsonl").exists()


def test_loss_decreases_over_a_few_steps(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=16, n_events_per_subject=40)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=40)

    output_dir = tmp_path / "run"
    config = _tiny_config(
        train_dir, tuning_dir, output_dir, log_every=1, eval_every=1000
    )
    train(config)

    lines = (output_dir / "loss_log.jsonl").read_text().strip().split("\n")
    import json  # noqa: PLC0415

    task_losses = [
        json.loads(line)["task_loss"]
        for line in lines
        if json.loads(line).get("split") == "train"
    ]
    assert len(task_losses) >= 4
    # not monotone step-to-step (small batches, real noise) but the back
    # half should be lower on average than the front half if anything is
    # actually learning from the packed-transformer path.
    midpoint = len(task_losses) // 2
    front_mean = sum(task_losses[:midpoint]) / midpoint
    back_mean = sum(task_losses[midpoint:]) / (len(task_losses) - midpoint)
    assert back_mean < front_mean


def test_checkpoint_round_trips_through_load_run(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)

    output_dir = tmp_path / "run"
    config = _tiny_config(train_dir, tuning_dir, output_dir)
    train(config)

    model, vocab, binner, loaded_config = load_run(output_dir, device="cpu")

    assert loaded_config.backbone == "transformer"
    assert isinstance(model.backbone, TransformerBackbone)
    assert len(vocab) > 0
    assert binner is not None

    # the loaded model is genuinely usable, not just structurally present
    batch = torch.randint(1, len(vocab), (1, 5))
    clinical_batch = ClinicalSequenceBatch(
        concept_ids=batch,
        aux=AuxiliaryInputs(
            type_ids=torch.ones(1, 5, dtype=torch.long),
            time_stamps=torch.arange(5).float().unsqueeze(0),
            ages=torch.full((1, 5), 50.0),
            visit_orders=torch.zeros(1, 5, dtype=torch.long),
            visit_segments=torch.zeros(1, 5, dtype=torch.long),
        ),
    )
    with torch.no_grad():
        logits, _, _ = model(clinical_batch)
    assert torch.isfinite(logits).all()
