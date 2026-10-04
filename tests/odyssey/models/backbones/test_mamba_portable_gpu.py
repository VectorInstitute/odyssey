"""GPU parity: the pure-PyTorch backbone against the ``mamba-ssm`` kernels.

Builds the hybrid backbone twice from one state dict -- once with the CUDA
mixers, once with :mod:`mamba_portable` -- and checks that both give the
same hidden states on the same chunked stream. Auto-skips without
``mamba-ssm`` and a CUDA device. Tolerances allow for TF32 matmuls and the
Triton kernels' own accumulation order.
"""

import pytest
import torch


pytest.importorskip("mamba_ssm", reason="mamba-ssm not installed (needs CUDA)")
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from odyssey.data.streaming import move_to_device  # noqa: E402
from odyssey.models.backbones import mamba_portable  # noqa: E402
from tests.odyssey.models.backbones.hybrid_helpers import (  # noqa: E402
    make_backbone,
    make_batch,
)


def _pair(monkeypatch: pytest.MonkeyPatch, device: str):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    cuda_backbone = make_backbone().to("cuda").eval()
    assert not cuda_backbone.portable
    with monkeypatch.context() as patch:
        patch.setattr(mamba_portable, "mamba_ssm_available", lambda: False)
        portable = make_backbone().eval()
    assert portable.portable
    portable.load_state_dict(cuda_backbone.state_dict())
    return cuda_backbone, portable.to(device)


@cuda_required
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_portable_backbone_matches_the_cuda_kernels(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    cuda_backbone, portable = _pair(monkeypatch, device)
    batch = make_batch(2, 48)
    halves = [(0, 20), (20, 48)]
    with torch.no_grad():
        state_cuda = state_port = None
        for start, stop in halves:
            chunk = type(batch)(
                concept_ids=batch.concept_ids[:, start:stop],
                aux=type(batch.aux)(
                    *(t[:, start:stop] if t is not None else None for t in batch.aux)
                ),
            )
            out_cuda, state_cuda = cuda_backbone(
                move_to_device(chunk, "cuda"), state=state_cuda
            )
            out_port, state_port = portable(
                move_to_device(chunk, device), state=state_port
            )
            torch.testing.assert_close(
                out_port.cpu(), out_cuda.cpu(), atol=2e-3, rtol=2e-3
            )
