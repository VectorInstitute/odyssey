"""CPU-testable pieces of the inference script.

The real streaming inference path needs EHRHybridBackbone/CUDA, see
test_run_inference_gpu.py.
"""

from pathlib import Path

import pytest

from odyssey.inference.run_inference import _latest_checkpoint


def test_latest_checkpoint_prefers_final(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_500.pt").touch()
    (tmp_path / "checkpoint_final.pt").touch()

    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint_final.pt"


def test_latest_checkpoint_picks_highest_step_when_no_final(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_500.pt").touch()
    (tmp_path / "checkpoint_2000.pt").touch()
    (tmp_path / "checkpoint_1000.pt").touch()

    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint_2000.pt"


def test_latest_checkpoint_raises_when_none_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _latest_checkpoint(tmp_path)
