"""GPU-only tests for the Mamba-2 state-carrying patch.

This is the actual unit of change behind the Mamba-3 -> Mamba-2 SSM-kernel
swap (entry 03, decision (d)):
``odyssey.models.backbones.hybrid._make_mamba2_with_state_cls`` patches
``Mamba2.forward`` to seed ``initial_states`` in the
``mamba_chunk_scan_combined`` call, which upstream ``mamba_ssm`` 2.3.2
never does. These tests exercise that patched class directly and in
isolation -- no embeddings, no attention branch, no ``HybridBlock`` -- so
a failure here points unambiguously at the patch itself, not at some
interaction with the rest of the hybrid backbone. See
``test_hybrid_gpu.py`` for the integrated, backbone-level tests.
"""

import pytest
import torch


mamba_ssm = pytest.importorskip(
    "mamba_ssm", reason="mamba-ssm not installed (needs CUDA)"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from odyssey.models.backbones.hybrid import _make_mamba2_with_state_cls  # noqa: E402


HIDDEN_SIZE = 64
STATE_SIZE = 16
HEADDIM = 64
CHUNK_SIZE = 16


def _make_mixer(layer_idx: int = 0):  # type: ignore[no-untyped-def]
    from mamba_ssm.modules.mamba2 import Mamba2  # noqa: PLC0415

    mamba2_with_state_cls = _make_mamba2_with_state_cls(Mamba2)
    return mamba2_with_state_cls(
        HIDDEN_SIZE,
        d_state=STATE_SIZE,
        headdim=HEADDIM,
        chunk_size=CHUNK_SIZE,
        layer_idx=layer_idx,
    ).cuda()


@cuda_required
def test_patched_forward_matches_unpatched_when_state_is_freshly_zero() -> None:
    """Regression guard: the patch must be a no-op for every case already exercised.

    ``EHRHybridBackbone`` always constructs an ``InferenceParams`` and
    passes it into every layer, even on a state=None first chunk, so
    upstream's own ``use_mem_eff_path`` fast-path guard
    (``inference_params is None``) was already False in every real call
    this backbone made *before* this patch existed too. So the correct
    "did the patch change anything" comparison is: same
    ``inference_params`` (forcing the slow path both sides already took),
    patched's explicit ``initial_states=<all-zero tensor>`` from a fresh
    cache vs. upstream's implicit ``initial_states=None`` -- both mean
    "start from zero", so outputs must match closely.
    """
    from mamba_ssm.modules.mamba2 import Mamba2  # noqa: PLC0415
    from mamba_ssm.utils.generation import InferenceParams  # noqa: PLC0415

    torch.manual_seed(0)
    mamba2_with_state_cls = _make_mamba2_with_state_cls(Mamba2)
    patched = (
        mamba2_with_state_cls(
            HIDDEN_SIZE, d_state=STATE_SIZE, headdim=HEADDIM, chunk_size=CHUNK_SIZE, layer_idx=0
        )
        .cuda()
        .eval()
    )
    unpatched = (
        Mamba2(HIDDEN_SIZE, d_state=STATE_SIZE, headdim=HEADDIM, chunk_size=CHUNK_SIZE, layer_idx=0)
        .cuda()
        .eval()
    )
    unpatched.load_state_dict(patched.state_dict())

    torch.manual_seed(1)
    u = torch.randn(2, CHUNK_SIZE, HIDDEN_SIZE, device="cuda")

    ip_patched = InferenceParams(max_seqlen=CHUNK_SIZE, max_batch_size=2)
    ip_unpatched = InferenceParams(max_seqlen=CHUNK_SIZE, max_batch_size=2)

    with torch.no_grad():
        out_patched = patched(u, inference_params=ip_patched)
        out_unpatched = unpatched(u, inference_params=ip_unpatched)

    assert torch.allclose(out_patched, out_unpatched, atol=1e-4, rtol=1e-4)


@cuda_required
def test_chunked_forward_with_carried_state_matches_one_shot_forward() -> None:
    """Chunked forward with carried state must match a one-shot forward.

    This is the core numerical-equivalence claim the whole streaming
    training design (decision (i)/(j), entry 02) depends on, and the
    exact assumption that failed for Mamba-3 (see
    ``test_mamba3_gpu.py::test_chunked_forward_with_carried_state_matches_one_shot_forward``,
    still failing there since ``mamba3_mimo_forward`` has no
    ``initial_states`` parameter at all -- a kernel-level limitation, not
    a wiring gap). This test validates that the wiring fix for Mamba-2
    actually closes that gap.
    """
    torch.manual_seed(0)
    mixer = _make_mixer().eval()

    from mamba_ssm.utils.generation import InferenceParams  # noqa: PLC0415

    full_len = 2 * CHUNK_SIZE
    torch.manual_seed(2)
    u_full = torch.randn(2, full_len, HIDDEN_SIZE, device="cuda")

    ip_full = InferenceParams(max_seqlen=full_len, max_batch_size=2)
    with torch.no_grad():
        out_full = mixer(u_full, inference_params=ip_full)

    u1, u2 = u_full[:, :CHUNK_SIZE], u_full[:, CHUNK_SIZE:]
    ip1 = InferenceParams(max_seqlen=CHUNK_SIZE, max_batch_size=2)
    with torch.no_grad():
        out1 = mixer(u1, inference_params=ip1)

    ip2 = InferenceParams(max_seqlen=CHUNK_SIZE, max_batch_size=2)
    ip2.key_value_memory_dict = ip1.key_value_memory_dict
    with torch.no_grad():
        out2 = mixer(u2, inference_params=ip2)

    assert torch.allclose(out_full[:, :CHUNK_SIZE], out1, atol=1e-3, rtol=1e-3)
    assert torch.allclose(out_full[:, CHUNK_SIZE:], out2, atol=1e-3, rtol=1e-3)
