"""Environment fingerprint and numeric canary."""

import json
from pathlib import Path

import torch

from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import BaselineSequenceModel
from odyssey.utils.env_fingerprint import (
    CANARY_FILENAME,
    FINGERPRINT_FILENAME,
    check_canary,
    environment_fingerprint,
    numeric_canary,
    verify_run_provenance,
    write_run_provenance,
)


def _model(vocab_size: int = 32) -> BaselineSequenceModel:
    torch.manual_seed(0)
    return BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        padding_idx=0,
    )


def test_fingerprint_has_the_load_bearing_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"hello")
    fp = environment_fingerprint({"checkpoint": artifact})
    assert fp["torch"] == torch.__version__
    assert "mamba_ssm" in fp and "binaries_sha256" in fp["mamba_ssm"]
    assert fp["artifacts"]["checkpoint"] is not None
    json.dumps(fp)  # serializable


def test_canary_is_deterministic_and_detects_weight_changes() -> None:
    model = _model()
    a = numeric_canary(model, 32)
    b = numeric_canary(model, 32)
    assert a == b
    assert check_canary(a, b) == []
    with torch.no_grad():
        model.lm_head.weight.add_(0.05)
    c = numeric_canary(model, 32)
    assert check_canary(a, c) != []


def test_provenance_roundtrip_and_legacy_runs(tmp_path: Path) -> None:
    model = _model()
    # fingerprint-only call writes no canary (a random model's canary is meaningless)
    write_run_provenance(tmp_path, model, 32)
    assert (tmp_path / FINGERPRINT_FILENAME).exists()
    assert not (tmp_path / CANARY_FILENAME).exists()
    # per-checkpoint canaries
    write_run_provenance(tmp_path, model, 32, checkpoint_name="checkpoint_best.pt")
    assert (
        verify_run_provenance(tmp_path, model, 32, checkpoint_name="checkpoint_best.pt")
        == []
    )
    # a different checkpoint name has no stored canary: silent
    assert (
        verify_run_provenance(tmp_path, model, 32, checkpoint_name="checkpoint_9.pt")
        == []
    )
    with torch.no_grad():
        model.lm_head.weight.add_(0.05)
    assert (
        verify_run_provenance(tmp_path, model, 32, checkpoint_name="checkpoint_best.pt")
        != []
    )
    # a run predating provenance files verifies silently
    assert verify_run_provenance(tmp_path / "nope", model, 32) == []


def test_legacy_single_canary_files_are_skipped(tmp_path: Path) -> None:
    """Files written by the first (construction-time) wiring never fail evals."""
    model = _model()
    (tmp_path / CANARY_FILENAME).write_text(
        json.dumps({"mean": 0.0, "std": 1.0, "absmax": 1.0, "shape": [1]})
    )
    assert (
        verify_run_provenance(tmp_path, model, 32, checkpoint_name="checkpoint_best.pt")
        == []
    )
