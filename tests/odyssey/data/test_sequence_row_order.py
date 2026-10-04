"""ordered_sequence_rows: the raw row behind every sequence position."""

from datetime import datetime, timedelta

import polars as pl

from odyssey.data.sequences import build_patient_sequence, ordered_sequence_rows
from odyssey.data.vocabulary import Vocabulary


T0 = datetime(2024, 1, 1)


def _events() -> pl.DataFrame:
    rows = [
        (1, None, "GENDER//F", None, None),
        (1, T0 - timedelta(days=365 * 60), "MEDS_BIRTH", None, None),
        (1, T0 + timedelta(hours=2), "LAB//B//", 2.0, 11),
        (1, T0, "HOSPITAL_ADMISSION//EW", None, 11),
        # a same-timestamp bundle: shard order must survive
        (1, T0 + timedelta(hours=1), "LAB//Z//", 3.0, 11),
        (1, T0 + timedelta(hours=1), "LAB//A//", 4.0, 11),
        (1, None, "RACE//WHITE", None, None),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def test_rows_are_static_first_then_stable_time_order_without_birth() -> None:
    ordered = ordered_sequence_rows(_events())
    assert ordered.n_static == 2
    assert ordered.birth_time == T0 - timedelta(days=365 * 60)
    assert ordered.rows["code"].to_list() == [
        "GENDER//F",
        "RACE//WHITE",
        "HOSPITAL_ADMISSION//EW",
        "LAB//Z//",
        "LAB//A//",
        "LAB//B//",
    ]
    # static rows take the first timed event's instant and visit
    assert ordered.rows["time"][0] == T0 and ordered.rows["hadm_id"][0] == 11


def test_rows_align_one_to_one_with_sequence_positions() -> None:
    events = _events().with_row_index("row")
    ordered = ordered_sequence_rows(events)
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab)
    assert len(seq) == ordered.rows.height
    assert [vocab.decode(t) for t in seq.concept_ids] == ordered.rows["code"].to_list()
    assert seq.static_mask == [True, True, False, False, False, False]
    # extra columns ride along, so a position maps back to its source row
    assert ordered.rows["row"].to_list() == [0, 6, 3, 4, 5, 2]


def test_static_only_subject_has_no_rows() -> None:
    only_static = _events().filter(pl.col("time").is_null())
    ordered = ordered_sequence_rows(only_static)
    assert ordered.rows.height == 0 and ordered.n_static == 0


def test_empty_frame_gives_empty_rows_and_no_birth() -> None:
    ordered = ordered_sequence_rows(_events().head(0))
    assert ordered.rows.height == 0
    assert ordered.n_static == 0 and ordered.birth_time is None


def test_no_birth_and_no_static_rows() -> None:
    timed = _events().filter(
        pl.col("time").is_not_null() & (pl.col("code") != "MEDS_BIRTH")
    )
    ordered = ordered_sequence_rows(timed)
    assert ordered.birth_time is None and ordered.n_static == 0
    assert ordered.rows["code"][0] == "HOSPITAL_ADMISSION//EW"


def test_frame_without_hadm_id_column_still_orders_static_first() -> None:
    no_visit = _events().drop("hadm_id")
    ordered = ordered_sequence_rows(no_visit)
    assert "hadm_id" not in ordered.rows.columns
    assert ordered.rows["code"].to_list()[:2] == ["GENDER//F", "RACE//WHITE"]
    assert ordered.rows["time"][0] == T0


def test_birth_row_never_becomes_a_position_even_when_it_is_the_earliest() -> None:
    ordered = ordered_sequence_rows(_events())
    assert "MEDS_BIRTH" not in ordered.rows["code"].to_list()
