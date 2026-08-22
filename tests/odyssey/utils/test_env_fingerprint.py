"""Environment fingerprint and numeric canary."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import BaselineSequenceModel
from odyssey.utils.env_fingerprint import (
    CANARY_FILENAME,
    FINGERPRINT_FILENAME,
    _file_sha256,
    _mamba_ssm_info,
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


def test_numeric_canary_does_not_consume_global_rng_state() -> None:
    # Protects a property confirmed by inspection (numeric_canary uses its
    # own local torch.Generator, not torch's global RNG) against a future
    # edit silently reintroducing global RNG use -- which would make the
    # canary's "identical env => identical values" guarantee depend on
    # unrelated global RNG consumption elsewhere in the same process,
    # breaking write_run_provenance/verify_run_provenance's reproducibility
    # check in a way no existing test would catch.
    model = _model()
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    numeric_canary(model, 32)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


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


# ---------------------------------------------------------------------------
# _file_sha256: the truncation cap and the not-readable path
# ---------------------------------------------------------------------------


def test_file_sha256_stops_reading_past_max_bytes(tmp_path: Path) -> None:
    """A large file still hashes (over a bounded prefix), never reads it whole."""
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * (5 * (1 << 20)))  # 5 MiB, several read chunks
    digest = _file_sha256(path, max_bytes=1 << 20)  # cap well under the file size
    assert digest is not None and len(digest) == 64  # a real sha256 hex digest


def test_file_sha256_returns_none_for_an_unreadable_path(tmp_path: Path) -> None:
    assert _file_sha256(tmp_path / "does-not-exist.bin") is None


# ---------------------------------------------------------------------------
# _mamba_ssm_info: the importable branch (this dev machine has no mamba_ssm,
# so the ImportError branch is what real runs here take -- inject a fake
# module to exercise the other one)
# ---------------------------------------------------------------------------


def test_mamba_ssm_info_hashes_installed_binaries_when_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg_dir = tmp_path / "fake_mamba_ssm"
    pkg_dir.mkdir()
    (pkg_dir / "kernel.so").write_bytes(b"fake shared object contents")
    fake_module = types.ModuleType("mamba_ssm")
    fake_module.__file__ = str(pkg_dir / "__init__.py")
    fake_module.__version__ = "9.9.9"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mamba_ssm", fake_module)

    info = _mamba_ssm_info()

    assert info["version"] == "9.9.9"
    assert info["binaries_sha256"] is not None
    # deterministic given the same binaries -- not just "some string"
    assert info == _mamba_ssm_info()


# ---------------------------------------------------------------------------
# environment_fingerprint: the git-subprocess OSError path and the GPU branch
# ---------------------------------------------------------------------------


def test_environment_fingerprint_survives_git_being_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No git binary on PATH (or any other OSError running it): commit is None."""

    def raise_os_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", raise_os_error)

    fp = environment_fingerprint()

    assert fp["git_commit"] is None


def test_environment_fingerprint_records_gpu_name_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gpu-populated branch, dead on this machine without mocking torch.cuda.

    This dev machine has no CUDA device.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _idx: (9, 0))

    fp = environment_fingerprint()

    assert fp["gpu"] == {"name": "Fake GPU", "capability": "9.0"}


# ---------------------------------------------------------------------------
# check_canary: the None-guard and the shape-mismatch branch
# ---------------------------------------------------------------------------


def test_check_canary_skips_a_stat_missing_from_either_side() -> None:
    """A stat missing on one side (e.g. an older file format) must be skipped.

    Skipped silently, not a crash on None arithmetic.
    """
    stored = {"mean": 1.0, "shape": [1, 8]}  # no "std"/"absmax"
    fresh = {"mean": 1.0, "std": 2.0, "absmax": 3.0, "shape": [1, 8]}
    assert check_canary(stored, fresh) == []


def test_check_canary_flags_a_shape_mismatch() -> None:
    stored = {"mean": 1.0, "std": 1.0, "absmax": 1.0, "shape": [2, 64]}
    fresh = {"mean": 1.0, "std": 1.0, "absmax": 1.0, "shape": [4, 64]}
    problems = check_canary(stored, fresh)
    assert any("shape" in p for p in problems)


# ---------------------------------------------------------------------------
# write_run_provenance: migrating a legacy single-canary file in place
# ---------------------------------------------------------------------------


def test_write_run_provenance_migrates_a_legacy_canary_file(tmp_path: Path) -> None:
    """A pre-per-checkpoint canary file at the target path must be replaced.

    Replaced, not merged with -- its top-level "mean" key would otherwise
    poison the new per-checkpoint dict structure.
    """
    model = _model()
    (tmp_path / CANARY_FILENAME).write_text(
        json.dumps({"mean": 0.0, "std": 1.0, "absmax": 1.0, "shape": [1]})
    )

    write_run_provenance(tmp_path, model, 32, checkpoint_name="checkpoint_best.pt")

    stored = json.loads((tmp_path / CANARY_FILENAME).read_text())
    assert "mean" not in stored  # the legacy top-level shape is gone
    assert "checkpoint_best.pt" in stored
    assert set(stored["checkpoint_best.pt"]) == {
        "seed",
        "shape",
        "mean",
        "std",
        "absmax",
        "first8",
    }
