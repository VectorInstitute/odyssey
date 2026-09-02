"""move_to_device walks nested NamedTuples without knowing their fields."""

from typing import NamedTuple

import torch

from odyssey.data.streaming import move_to_device


class _Inner(NamedTuple):
    x: torch.Tensor
    label: str


class _Outer(NamedTuple):
    inner: _Inner
    y: torch.Tensor
    n: int


def test_moves_every_tensor_and_keeps_structure_and_non_tensors() -> None:
    outer = _Outer(_Inner(torch.ones(2), "keep"), torch.zeros(3), 7)
    moved = move_to_device(outer, "cpu")
    assert isinstance(moved, _Outer) and isinstance(moved.inner, _Inner)
    assert torch.equal(moved.inner.x, outer.inner.x)
    assert moved.inner.label == "keep" and moved.n == 7
    assert moved.y.device.type == "cpu"


def test_bare_tensor_and_plain_values_pass_through() -> None:
    assert torch.equal(move_to_device(torch.arange(3), "cpu"), torch.arange(3))
    assert move_to_device("untouched", "cpu") == "untouched"
    assert move_to_device((1, 2), "cpu") == (1, 2)  # a plain tuple is not a NamedTuple
