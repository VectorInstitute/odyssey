"""Device selection and transfer across CUDA, Apple MPS and CPU."""

import pytest
import torch

from odyssey.utils import device as device_mod
from odyssey.utils.device import default_device, resolve_device, to_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_default_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_mod.torch.backends.mps, "is_available", lambda: mps)
    assert default_device() == expected


def test_resolve_device_maps_auto_and_none_and_keeps_explicit_choices() -> None:
    assert resolve_device("auto") == default_device()
    assert resolve_device(None) == default_device()
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_to_device_keeps_dtypes_off_mps() -> None:
    stamps = torch.tensor([1.0, 2.5], dtype=torch.float64)
    moved = to_device(stamps, "cpu")
    assert moved.dtype == torch.float64 and torch.equal(moved, stamps)
    ids = torch.tensor([1, 2])
    assert to_device(ids, torch.device("cpu")).dtype == torch.long


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_to_device_casts_float64_to_float32_on_mps_only() -> None:
    stamps = torch.tensor([123.25, 456.5], dtype=torch.float64)
    moved = to_device(stamps, "mps")
    assert moved.device.type == "mps" and moved.dtype == torch.float32
    assert torch.equal(moved.cpu(), stamps.float())
    assert to_device(torch.tensor([1, 2]), "mps").dtype == torch.long
