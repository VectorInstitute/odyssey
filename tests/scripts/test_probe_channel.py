"""Tests for scripts/probe_channel.py on a tiny synthetic run (CPU)."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.concepts import concepts_for_source
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner
from odyssey.data.vocabulary import Vocabulary
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.data import iter_patient_sequences
from odyssey.training.train import TrainingConfig
from scripts import probe_channel


T0 = datetime(2024, 1, 1)
NUM_CONCEPTS = 3
CODES = [f"LAB//{i}//" for i in range(10)]
SUBJECTS = range(1, 13)
EVENTS_PER_SUBJECT = 24
SCHEMA = {
    "subject_id": pl.Int64,
    "code": pl.Utf8,
    "time": pl.Datetime,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _events(subjects: range) -> pl.DataFrame:
    # The next code is a function of the current one, so a trained model
    # forecasts it exactly and the full bottleneck carries that forecast.
    rows = [
        (
            sid,
            CODES[(sid * 3 + i) % len(CODES)],
            T0 + timedelta(hours=i),
            None,
            100 + sid,
        )
        for sid in subjects
        for i in range(EVENTS_PER_SUBJECT)
    ]
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({c: i + 2 for i, c in enumerate(CODES)})
    return Vocabulary(tokens)


def _trained_model(vocab: Vocabulary) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=16, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        padding_idx=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(6):
        state = None
        sampler = PackedLaneSampler(
            iter_patient_sequences(_events(SUBJECTS), vocab),
            num_lanes=2,
            chunk_size=16,
            reset_prob=0.0,
        )
        for chunk in sampler:
            opt.zero_grad()
            logits, _, state = model(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            real = chunk.real_mask
            loss = F.cross_entropy(logits[real], chunk.targets[real])
            loss.backward()
            opt.step()
            state = state.detach() if hasattr(state, "detach") else None
    model.eval()
    return model


def _write_shards(shard_dir: Path) -> None:
    shard_dir.mkdir()
    _events(range(1, 7)).write_parquet(shard_dir / "0.parquet")
    _events(range(7, 13)).write_parquet(shard_dir / "1.parquet")


def _config(tmp_path: Path, **overrides: Any) -> TrainingConfig:
    return TrainingConfig(
        train_shard_dir=str(tmp_path / "train"),
        tuning_shard_dir=str(tmp_path / "tuning"),
        output_dir=str(tmp_path / "run"),
        source="mimic_iv",
        task_set="v1",
        concept_supervision="visit",
        **overrides,
    )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, model: torch.nn.Module, config: TrainingConfig
) -> None:
    vocab = _vocab()
    binner = QuantileBinner(boundaries={}, n_bins=3)
    monkeypatch.setattr(
        probe_channel, "load_run", lambda *a, **k: (model, vocab, binner, config)
    )
    monkeypatch.setattr(
        probe_channel,
        "resolve_concepts_for_run",
        lambda *a, **k: concepts_for_source("mimic_iv", task_set="v1")[:NUM_CONCEPTS],
    )


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("probe")
    _write_shards(tmp_path / "held_out")
    with pytest.MonkeyPatch.context() as mp:
        _patch_run(mp, _trained_model(_vocab()), _config(tmp_path))
        out = tmp_path / "run" / "channel_probes.json"
        probe_channel.main(
            [
                "--run-dir",
                str(tmp_path / "run"),
                "--held-out-shard-dir",
                str(tmp_path / "held_out"),
                "--output-json",
                str(out),
                "--max-positions",
                "0",
                "--num-lanes",
                "2",
                "--chunk-size",
                "16",
                "--epochs",
                "40",
                "--patience",
                "5",
                "--n-boot",
                "50",
                "--ctl-epochs",
                "3",
                "--train-frac",
                "0.6",
                "--tune-frac",
                "0.15",
            ]
        )
        loaded: dict[str, Any] = json.loads(out.read_text())
        return loaded


def test_split_by_subject_has_no_subject_overlap() -> None:
    gen = torch.Generator().manual_seed(1)
    subject_ids = torch.randint(0, 40, (1000,), generator=gen)
    split = probe_channel.split_by_subject(subject_ids, seed=3)
    parts = {
        name: set(subject_ids[getattr(split, name)].tolist())
        for name in ("train", "tune", "test")
    }
    assert parts["train"] & parts["test"] == set()
    assert parts["train"] & parts["tune"] == set()
    assert parts["tune"] & parts["test"] == set()
    assert len(parts["train"] | parts["tune"] | parts["test"]) == 40
    total = split.train.numel() + split.tune.numel() + split.test.numel()
    assert total == 1000


def test_full_bottleneck_readout_is_at_least_as_accurate_as_k_only(
    payload: dict[str, Any],
) -> None:
    readouts = payload["readouts"]
    assert readouts["h_bar"]["top1_accuracy"] >= readouts["k_only"]["top1_accuracy"]
    # The trained model forecasts the deterministic cycle, and the full
    # bottleneck is exactly what its head reads.
    assert payload["model"]["top1_accuracy"] > payload["majority_class_accuracy"]
    assert readouts["h_bar"]["retained"] > 0.5


def test_payload_schema(payload: dict[str, Any]) -> None:
    for key in (
        "run_dir",
        "checkpoint",
        "concept_names",
        "protocol",
        "n_positions",
        "n_subjects",
        "majority_class_accuracy",
        "model",
        "readouts",
        "ctl",
        "hazards",
        "notes",
    ):
        assert key in payload, key
    assert set(payload["readouts"]) == set(probe_channel.READOUT_NAMES)
    for score in (payload["model"], *payload["readouts"].values()):
        for key in (
            "n_positions",
            "n_subjects",
            "top1_accuracy",
            "ci95",
            "retained",
            "retained_ci95",
            "completeness_score",
        ):
            assert key in score, key
        lo, hi = score["ci95"]
        assert lo <= score["top1_accuracy"] <= hi
    assert payload["model"]["retained"] == 1.0
    assert (
        payload["n_subjects"]["train"]
        + payload["n_subjects"]["tune"]
        + payload["n_subjects"]["test"]
        == payload["n_subjects"]["bank"]
    )
    assert payload["n_positions"]["bank"] == len(SUBJECTS) * (EVENTS_PER_SUBJECT - 1)
    assert payload["n_shards"] == 2
    assert payload["hazards"] is None
    assert payload["ctl"] is not None
    assert "probs_only" in payload["ctl"]
    assert len(payload["concept_names"]) == NUM_CONCEPTS


def test_refuses_a_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BaselineSequenceModel(
        TinyGRUBackbone(vocab_size=12, hidden_size=4), vocab_size=12
    )
    _patch_run(monkeypatch, model, _config(tmp_path, model_kind="baseline"))
    with pytest.raises(ValueError, match="needs a concept bottleneck"):
        probe_channel.run_channel_probes(tmp_path / "run", tmp_path / "held_out")


def test_refuses_a_decomposed_bottleneck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ConceptBottleneckSequenceModel(
        TinyGRUBackbone(vocab_size=12, hidden_size=4),
        vocab_size=12,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        bottleneck_kind="decomposed",
    )
    _patch_run(monkeypatch, model, _config(tmp_path, bottleneck_kind="decomposed"))
    with pytest.raises(ValueError, match="mixture bottleneck"):
        probe_channel.run_channel_probes(tmp_path / "run", tmp_path / "held_out")
