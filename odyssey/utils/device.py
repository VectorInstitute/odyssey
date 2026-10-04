"""Torch device selection and transfer for CUDA, Apple MPS and CPU.

Inference runs on an NVIDIA GPU (with the ``mamba-ssm`` kernels) or on a
laptop (Apple MPS or CPU, with :mod:`odyssey.models.backbones.mamba_portable`).
These helpers keep the device-specific rules in one place.
"""

import torch


def default_device() -> str:
    """Return the best available device: ``cuda``, then ``mps``, then ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | None) -> str:
    """Map ``None`` or ``"auto"`` to :func:`default_device`; pass others through."""
    return default_device() if device in {None, "auto"} else str(device)


def to_device(tensor: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Move ``tensor`` to ``device``, casting float64 to float32 on MPS.

    MPS has no float64. The only float64 model inputs are time stamps in
    hours, which the model casts to float32 before use, so the cast loses
    nothing the model would see.
    """
    if tensor.dtype == torch.float64 and torch.device(device).type == "mps":
        return tensor.to(device=device, dtype=torch.float32)
    return tensor.to(device)


__all__ = ["default_device", "resolve_device", "to_device"]
