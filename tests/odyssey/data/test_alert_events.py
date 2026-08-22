"""Tests for odyssey.data.alert_events's guard branches.

event_times has two ways to be misconfigured: a concept-defined alert
called without the concept_first_times frame it needs, and an AlertEvent
that names neither a concept nor a code_prefix at all (a malformed
registry entry). Neither failure path had ever been exercised.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import polars as pl
import pytest

from odyssey.data.alert_events import AlertEvent, event_times


T0 = datetime(2024, 1, 1)

_EventRow = Tuple[int, str, datetime, Optional[float], int]


def _events() -> pl.DataFrame:
    rows: List[_EventRow] = [
        (1, "LAB//220045//bpm", T0, 80.0, 1001),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=1), 80.0, 1001),
    ]
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


def test_concept_alert_without_concept_first_times_raises() -> None:
    """A concept-defined alert needs the shared first-time frame; None must fail."""
    alert = AlertEvent("acute_kidney_injury", concept="acute_kidney_injury")

    with pytest.raises(ValueError, match="acute_kidney_injury"):
        event_times(_events(), alert, concept_first_times=None)


def test_alert_with_neither_concept_nor_code_prefix_raises() -> None:
    """A malformed registry entry (no onset rule at all) must fail, not no-op."""
    alert = AlertEvent("malformed_alert")

    with pytest.raises(ValueError, match="malformed_alert"):
        event_times(_events(), alert)
