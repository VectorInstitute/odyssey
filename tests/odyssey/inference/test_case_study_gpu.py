"""GPU-only integration test for per-patient case-trace extraction."""

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

from odyssey.data.concepts import CONCEPTS  # noqa: E402
from odyssey.data.sequences import build_patient_sequence  # noqa: E402
from odyssey.inference.case_study import extract_patient_case  # noqa: E402
from odyssey.inference.run_inference import (  # noqa: E402
    load_and_bin_held_out,
    load_run,
)
from odyssey.training.data import build_concept_label_dicts  # noqa: E402
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
def test_extract_patient_case_end_to_end(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    tuning_dir = tmp_path / "data" / "tuning"
    held_out_dir = tmp_path / "data" / "held_out"
    _write_shards(train_dir, n_subjects=12, n_events_per_subject=30)
    _write_shards(tuning_dir, n_subjects=4, n_events_per_subject=30)
    _write_shards(held_out_dir, n_subjects=3, n_events_per_subject=25)

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
        eval_every=100000,
        checkpoint_every=100000,
    )
    train(config)

    model, vocab, binner, _ = load_run(run_dir)
    held_out_events_binned = load_and_bin_held_out(held_out_dir, binner)
    held_out_raw = held_out_events_binned  # concept labeling ignores value tokens
    labels, mask = build_concept_label_dicts(held_out_raw, CONCEPTS)

    subject_id = 0
    subject_events = held_out_events_binned.filter(pl.col("subject_id") == subject_id)
    seq = build_patient_sequence(subject_events, vocab)
    assert len(seq) > 0

    concept_names = [c.name for c in CONCEPTS]
    trace = extract_patient_case(
        model,
        seq,
        vocab,
        concept_names,
        concept_labels=labels.get(subject_id),
        concept_mask=mask.get(subject_id),
        top_k=5,
    )

    n = len(seq)
    assert trace.subject_id == subject_id
    assert len(trace.times) == n
    assert len(trace.input_codes) == n
    assert len(trace.predicted_top_k) == n
    assert len(trace.concept_probs) == n
    assert len(trace.observability_probs) == n
    assert all(len(p) == len(concept_names) for p in trace.concept_probs)

    # every position but the last has a real prediction + rank.
    assert trace.predicted_top_k[-1] == []
    assert trace.true_next_code[-1] is None
    assert trace.true_next_rank[-1] is None
    for i in range(n - 1):
        assert len(trace.predicted_top_k[i]) == 5
        assert trace.true_next_code[i] is not None
        rank = trace.true_next_rank[i]
        assert rank is not None
        assert 0 <= rank < 20000  # a valid vocab-sized rank, not garbage
        probs = [p for _, p in trace.predicted_top_k[i]]
        assert all(torch.isfinite(torch.tensor(probs)))
