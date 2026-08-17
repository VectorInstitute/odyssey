"""Tests for the prior-diagnosis history recap transform."""

from datetime import datetime, timedelta

import polars as pl

from odyssey.data.history_recap import add_history_recap, maybe_history_recap
from odyssey.data.vocabulary import code_type


T0 = datetime(2024, 1, 1)


def _events() -> pl.DataFrame:
    rows = [
        (1, "HOSPITAL_ADMISSION//EW", T0, None, 10),
        (1, "LAB//220045//bpm", T0 + timedelta(hours=1), 80.0, 10),
        (1, "DIAGNOSIS//ICD//10//I5023", T0 + timedelta(days=3), None, 10),
        (1, "DIAGNOSIS//ICD//10//E119", T0 + timedelta(days=3), None, 10),
        (1, "DIAGNOSIS//ICD//10//I5022", T0 + timedelta(days=3), None, 10),
        (1, "HOSPITAL_ADMISSION//EW", T0 + timedelta(days=60), None, 11),
        (1, "DIAGNOSIS//ICD//10//N179", T0 + timedelta(days=63), None, 11),
        (1, "HOSPITAL_ADMISSION//EW", T0 + timedelta(days=120), None, 12),
        (2, "HOSPITAL_ADMISSION//EW", T0, None, 20),
        (2, "DIAGNOSIS//ICD//9//4280", T0 + timedelta(days=1), None, 20),
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


def test_recap_uses_only_prior_admissions_at_category_level() -> None:
    out = add_history_recap(_events())
    recap = out.filter(pl.col("code").str.starts_with("HISTORY//"))
    by_visit = {
        v: f["code"].to_list()
        for v, f in recap.group_by("hadm_id", maintain_order=True)
        for v in [v[0]]
    }
    # first admissions have no history
    assert 10 not in by_visit and 20 not in by_visit
    # second admission: the two categories from admission 1 (I50 twice -> once)
    assert sorted(by_visit[11]) == [
        "HISTORY//DIAGNOSIS//ICD//10//E11",
        "HISTORY//DIAGNOSIS//ICD//10//I50",
    ]
    # third admission: most recent (N17) first, then the older ones
    assert by_visit[12][0] == "HISTORY//DIAGNOSIS//ICD//10//N17"
    assert set(by_visit[12][1:]) == {
        "HISTORY//DIAGNOSIS//ICD//10//E11",
        "HISTORY//DIAGNOSIS//ICD//10//I50",
    }
    # recap rows sit at the admission time, in the admission's visit
    adm_time = T0 + timedelta(days=60)
    assert recap.filter(pl.col("hadm_id") == 11)["time"].to_list() == [adm_time] * 2


def test_recap_rows_follow_the_admission_token_and_keep_schema() -> None:
    events = _events()
    out = add_history_recap(events)
    assert out.schema == events.schema
    codes = out.filter(pl.col("subject_id") == 1)["code"].to_list()
    i = [k for k, c in enumerate(codes) if c.startswith("HOSPITAL_ADMISSION")][1]
    assert codes[i + 1].startswith("HISTORY//")
    # HISTORY tokens are their own family ("other"), never diagnosis
    assert code_type("HISTORY//DIAGNOSIS//ICD//10//I50") != code_type(
        "DIAGNOSIS//ICD//10//I50"
    )


def test_recap_is_capped_and_idempotent() -> None:
    events = _events()
    once = add_history_recap(events, max_codes=1)
    assert once.filter(pl.col("code").str.starts_with("HISTORY//")).height == 2
    twice = add_history_recap(add_history_recap(events))
    assert twice.height == add_history_recap(events).height


def test_maybe_history_recap_passthrough() -> None:
    events = _events()
    assert maybe_history_recap(events, enabled=False).height == events.height
    assert maybe_history_recap(events, enabled=True).height > events.height
