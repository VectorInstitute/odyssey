"""DemoConfig validation and the JSON contract serializer."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.clinician_demo.config import DemoConfig
from apps.clinician_demo.schemas import (
    FLOAT_DIGITS,
    NextEvent,
    TimelineEntry,
    VisitSummary,
    to_jsonable,
)


def _config(**overrides: object) -> DemoConfig:
    return DemoConfig(run_dir=Path("/runs/r"), data_dir=Path("/data"), **overrides)  # type: ignore[arg-type]


def test_defaults_are_safe_and_cache_lives_under_the_run() -> None:
    config = _config()
    assert config.host == "127.0.0.1" and config.data_mode == "credentialed"
    assert config.resolved_cache_dir == Path("/runs/r/demo_cache")
    assert config.run_name == "r"
    assert _config(cache_dir=Path("/tmp/c")).resolved_cache_dir == Path("/tmp/c")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host", "0.0.0.0", "loopback"),
        ("host", "10.0.0.5", "loopback"),
        ("data_mode", "public", "data_mode"),
        ("alert_rate", 0.0, "alert_rate"),
        ("alert_rate", 1.0, "alert_rate"),
        ("port", 70000, "port"),
        ("port", -1, "port"),
        ("max_shards", 0, "max_shards"),
        ("horizons", (), "horizons"),
        ("horizons", (24.0, -8.0), "horizons"),
    ],
)
def test_unsafe_or_invalid_fields_are_refused(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**{field: value})


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_every_loopback_host_is_accepted(host: str) -> None:
    assert _config(host=host).host == host


def test_floats_are_rounded_and_non_finite_become_null() -> None:
    payload = {"a": 1 / 3, "b": math.nan, "c": math.inf, "d": -math.inf, "e": 2}
    out = to_jsonable(payload)
    assert out == {
        "a": round(1 / 3, FLOAT_DIGITS),
        "b": None,
        "c": None,
        "d": None,
        "e": 2,
    }
    json.dumps(out, allow_nan=False)  # must be strict-JSON clean


def test_dataclasses_nest_and_tuples_become_lists() -> None:
    visit = VisitSummary(1, 0.0, 48.123456, "Admission", 10)
    out = to_jsonable(
        {"visit": visit, "pair": (1, 2.0), "entries": [TimelineEntry(1.0, "lab", "Na")]}
    )
    assert out["visit"]["end_hours"] == 48.12346
    assert out["pair"] == [1, 2.0]
    assert out["entries"][0] == {
        "t": 1.0,
        "category": "lab",
        "label": "Na",
        "value": None,
        "flag": None,
    }


def test_numpy_and_torch_scalars_unwrap_and_bools_stay_bools() -> None:
    out = to_jsonable(
        [np.float32(0.5), np.int64(3), torch.tensor(0.25), True, np.bool_(False)]
    )
    assert out == [0.5, 3, 0.25, True, False]
    assert isinstance(out[1], int)


def test_unserializable_values_raise_instead_of_stringifying() -> None:
    with pytest.raises(TypeError, match="cannot serialize"):
        to_jsonable({"x": object()})
    with pytest.raises(TypeError, match="class"):
        to_jsonable(NextEvent)


def test_dict_keys_become_strings() -> None:
    @dataclass(frozen=True)
    class Holder:
        by_id: dict[int, float]

    assert to_jsonable(Holder({7: 0.1})) == {"by_id": {"7": 0.1}}
