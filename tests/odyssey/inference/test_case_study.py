"""CPU-testable pieces of the case-study module.

extract_patient_case needs the real EHRHybridBackbone/CUDA, see
test_case_study_gpu.py.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest
import torch

from odyssey.inference.case_study import _parse_args, select_diverse_cases


T0 = datetime(2024, 1, 1, 0, 0)

_EventRow = Tuple[int, str, datetime, Optional[float], Optional[int]]


def _events(rows: List[_EventRow]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def _patient_events(subject_id: int, n_events: int) -> List[_EventRow]:
    return [
        (subject_id, f"DIAGNOSIS//{i}", T0 + timedelta(hours=i), None, None)
        for i in range(n_events)
    ]


def test_select_diverse_cases_respects_min_events() -> None:
    events = _events(_patient_events(1, 5) + _patient_events(2, 20))
    selected = select_diverse_cases(events, {}, n_cases=15, min_events=10)
    assert selected == [2]


def test_select_diverse_cases_returns_at_most_n_cases() -> None:
    rows: List[_EventRow] = []
    for sid in range(30):
        rows += _patient_events(sid, 15)
    events = _events(rows)
    selected = select_diverse_cases(events, {}, n_cases=15, min_events=10)
    assert len(selected) == 15
    assert len(set(selected)) == 15  # no duplicates


def test_select_diverse_cases_spans_both_short_and_long_stays() -> None:
    rows: List[_EventRow] = []
    for sid in range(10):
        rows += _patient_events(sid, 10)  # short
    for sid in range(10, 20):
        rows += _patient_events(sid, 500)  # long
    events = _events(rows)

    selected = select_diverse_cases(events, {}, n_cases=10, min_events=5)

    assert any(sid < 10 for sid in selected)
    assert any(sid >= 10 for sid in selected)


def test_select_diverse_cases_spans_concept_triggered_and_not() -> None:
    rows: List[_EventRow] = []
    for sid in range(20):
        rows += _patient_events(sid, 30)
    events = _events(rows)
    concept_labels = {
        sid: torch.tensor([1.0, 1.0, 1.0])
        if sid % 2 == 0
        else torch.tensor([0.0, 0.0, 0.0])
        for sid in range(20)
    }

    selected = select_diverse_cases(events, concept_labels, n_cases=10, min_events=5)

    assert any(sid % 2 == 0 for sid in selected)
    assert any(sid % 2 == 1 for sid in selected)


def test_select_diverse_cases_is_deterministic_given_a_seed() -> None:
    rows: List[_EventRow] = []
    for sid in range(20):
        rows += _patient_events(sid, 30)
    events = _events(rows)

    a = select_diverse_cases(events, {}, n_cases=10, min_events=5, seed=7)
    b = select_diverse_cases(events, {}, n_cases=10, min_events=5, seed=7)
    assert a == b


def test_select_diverse_cases_empty_when_nobody_meets_min_events() -> None:
    events = _events(_patient_events(1, 3))
    assert select_diverse_cases(events, {}, n_cases=15, min_events=10) == []


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_checkpoint_to_best_and_15_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            "/out/cases.json",
        ],
    )
    args = _parse_args()
    assert args.checkpoint_path == Path("/runs/x/checkpoint_best.pt")
    assert args.n_cases == 15
    assert args.max_shards is None
