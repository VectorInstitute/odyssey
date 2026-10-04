"""The hybrid backbone built from the pure-PyTorch mixers (CPU, any host).

``mamba_ssm_available`` is forced to ``False`` so these run the portable
path even on a GPU host that has ``mamba-ssm`` installed.
"""

import pytest
import torch

from odyssey.models.backbones import mamba_portable
from odyssey.models.backbones.base import TimeAwareState
from tests.odyssey.models.backbones.hybrid_helpers import (
    HIDDEN_SIZE,
    make_backbone,
    make_batch,
)


@pytest.fixture(autouse=True)
def _portable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mamba_portable, "mamba_ssm_available", lambda: False)


def _slice(batch, start: int, stop: int):  # type: ignore[no-untyped-def]
    return type(batch)(
        concept_ids=batch.concept_ids[:, start:stop],
        aux=type(batch.aux)(
            *(t[:, start:stop] if t is not None else None for t in batch.aux)
        ),
    )


def test_builds_without_mamba_ssm_and_uses_the_portable_mixers() -> None:
    backbone = make_backbone()
    assert backbone.portable
    block = backbone.layers[0]
    assert isinstance(block.mamba, mamba_portable.Mamba2WithState)
    assert isinstance(block.attn, mamba_portable.MHA)
    assert isinstance(backbone.norm_f, mamba_portable.RMSNorm)


def test_forward_shapes_and_carried_state() -> None:
    torch.manual_seed(0)
    backbone = make_backbone().eval()
    hidden, state = backbone(make_batch(2, 12))
    assert hidden.shape == (2, 12, HIDDEN_SIZE)
    assert isinstance(state, TimeAwareState)
    assert set(state.recurrent.mamba_states) == {0, 1}


def test_mamba_state_carries_across_chunks() -> None:
    """Chunked streaming equals one pass on the Mamba side.

    Attention has no cross-chunk memory by design (see hybrid's module
    docstring), so this compares a backbone whose attention branch is
    silenced: what remains must match exactly.
    """
    torch.manual_seed(0)
    backbone = make_backbone().eval()
    with torch.no_grad():
        for block in backbone.layers:
            block.attn.out_proj.weight.zero_()
            block.attn.out_proj.bias.zero_()
            block.merge.value_proj.weight.data[:] = torch.eye(HIDDEN_SIZE)
            block.merge.value_proj.bias.zero_()
            block.merge.key_proj.weight.zero_()
            block.merge.key_proj.bias.zero_()
    batch = make_batch(2, 20)
    with torch.no_grad():
        whole, _ = backbone(batch)
        first, state = backbone(_slice(batch, 0, 7))
        second, _ = backbone(_slice(batch, 7, 20), state=state)
    # Equal keys give the two branches equal weight, so attention's zero
    # output halves the fused value on both paths alike.
    torch.testing.assert_close(
        torch.cat([first, second], 1), whole, atol=1e-4, rtol=1e-4
    )


def test_carried_state_is_not_mutated_by_the_next_chunk() -> None:
    torch.manual_seed(0)
    backbone = make_backbone().eval()
    batch = make_batch(1, 10)
    with torch.no_grad():
        _, state = backbone(_slice(batch, 0, 5))
        before = {
            k: tuple(t.clone() for t in v)
            for k, v in state.recurrent.mamba_states.items()
        }
        backbone(_slice(batch, 5, 10), state=state)
    for layer, tensors in state.recurrent.mamba_states.items():
        for kept, now in zip(before[layer], tensors, strict=True):
            assert torch.equal(kept, now)


def test_a_state_dict_round_trips_between_two_portable_backbones() -> None:
    torch.manual_seed(0)
    source = make_backbone().eval()
    target = make_backbone().eval()
    target.load_state_dict(source.state_dict())
    batch = make_batch(2, 9)
    with torch.no_grad():
        torch.testing.assert_close(source(batch)[0], target(batch)[0])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_runs_on_apple_mps_and_matches_cpu() -> None:
    from odyssey.data.streaming import move_to_device  # noqa: PLC0415

    torch.manual_seed(0)
    backbone = make_backbone().eval()
    batch = make_batch(2, 24)
    with torch.no_grad():
        cpu_out, _ = backbone(batch)
        mps_out, _ = backbone.to("mps")(move_to_device(batch, "mps"))
    torch.testing.assert_close(mps_out.cpu(), cpu_out, atol=1e-4, rtol=1e-4)
