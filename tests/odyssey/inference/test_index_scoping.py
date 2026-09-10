"""Nothing after the index time may reach the forecast read at the index.

Editing post-index evidence and confirming the index-position forecast does
not move was considered as evidence that the record edits are causal. It is
not: it is null by two independent guarantees, so it can never fail. The
window filter in :func:`apply_value_edits` is ``time <= index_time``, so it
cannot touch a post-index row at all, and :func:`score_record_at` stops
streaming once the index position is scored, so the model never reads that
far even if a row did change.

What is worth pinning is the property those guarantees rest on, because a
refactor could quietly break it. The whole record, post-index rows included,
goes through normalization, history recap, value binning and tokenization
BEFORE the sequence is truncated at the index position. Every one of those
steps is supposed to be row-wise or backward-looking. If any of them grew a
statistic computed over the full frame (a per-subject quantile, a mean, a
vocabulary built from the record in hand), post-index data would reach the
index-position tokens, and every intervention number in the paper would be
scored on a leaked record. The paper claims exactly one whole-record
quantity, the assessability mask, which gates concept supervision only.

These tests assert that claim mechanically, at both ends of the chain: the
tokens up to the index are unchanged, and so is the forecast read from them.
"""

from datetime import datetime, timedelta

import polars as pl
import torch

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import build_patient_sequence
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.counterfactual import score_record_at
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2024, 1, 1)
INDEX = T0 + timedelta(hours=20)
SBP = "LAB//220179//mmHg"
CREAT = "LAB//RESULT//50912//mg/dL"


def _events() -> pl.DataFrame:
    rows: list[tuple[int, str, datetime, float | None, int]] = [
        (1, "HOSPITAL_ADMISSION//EMERGENCY", T0, None, 101)
    ]
    for h in range(1, 40):
        rows.append((1, SBP, T0 + timedelta(hours=h), 120.0, 101))
        if h % 6 == 0:
            rows.append((1, CREAT, T0 + timedelta(hours=h), 1.0, 101))
    rows.append((1, "HOSPITAL_DISCHARGE//HOME", T0 + timedelta(hours=40), None, 101))
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def _wreck_the_future(events: pl.DataFrame) -> pl.DataFrame:
    """Make the post-index record as different as it can be, pre-index intact.

    Extreme values (which would move any quantile or mean computed over the
    frame), extra rows, and a new code the pre-index record never contains.
    """
    after = pl.col("time") > pl.lit(INDEX)
    wrecked = events.with_columns(
        pl.when(after & pl.col("numeric_value").is_not_null())
        .then(pl.lit(9999.0, dtype=pl.Float32))
        .otherwise(pl.col("numeric_value"))
        .alias("numeric_value")
    )
    extra = pl.DataFrame(
        [
            (1, "LAB//NEVER//SEEN//BEFORE", T0 + timedelta(hours=30 + i), 5000.0, 101)
            for i in range(25)
        ],
        schema=events.schema,
        orient="row",
    )
    return pl.concat([wrecked, extra]).sort("time")


def _prepared(events: pl.DataFrame) -> pl.DataFrame:
    """Run the preprocessing chain the counterfactual path applies to a record."""
    out = maybe_normalize(events, enabled=True, source="mimic_iv")
    out = maybe_history_recap(out, enabled=True)
    return add_value_tokens(out, None, source="mimic_iv")


def _positions_up_to_index(
    events: pl.DataFrame, vocab: Vocabulary
) -> list[tuple[int, int, float, float, int, int]]:
    """Every per-position field the model reads, truncated at the index."""
    binned = _prepared(events)
    seq = build_patient_sequence(binned, vocab)
    origin = binned.filter(pl.col("time").is_not_null())["time"].min()
    assert isinstance(origin, datetime)
    index_hours = (INDEX - origin).total_seconds() / 3600.0
    rows = zip(
        seq.concept_ids,
        seq.type_ids,
        seq.time_stamps,
        seq.ages,
        seq.visit_orders,
        seq.visit_segments,
    )
    return [row for row in rows if row[2] <= index_hours + 1e-9]


def _vocab(events: pl.DataFrame) -> Vocabulary:
    return Vocabulary.build(_prepared(events)["code"].to_list(), min_count=1)


def test_post_index_events_do_not_change_the_tokens_at_or_before_the_index() -> None:
    events = _events()
    # One vocabulary for both records: a vocabulary rebuilt per record would
    # itself be a whole-record statistic, which is the leak, not the control.
    vocab = _vocab(_wreck_the_future(events))
    assert _positions_up_to_index(events, vocab) == _positions_up_to_index(
        _wreck_the_future(events), vocab
    )


def test_post_index_events_do_not_change_the_forecast_at_the_index() -> None:
    events = _events()
    wrecked = _wreck_the_future(events)
    vocab = _vocab(wrecked)
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=["vasopressor_start", "death"],
    )
    names = ["hypotension", "aki_stage_3"]
    factual = score_record_at(
        model,
        vocab,
        None,
        _prepared(events),
        index_time=INDEX,
        concept_names=names,
        source="mimic_iv",
    )
    future = score_record_at(
        model,
        vocab,
        None,
        _prepared(wrecked),
        index_time=INDEX,
        concept_names=names,
        source="mimic_iv",
    )
    assert factual.event_risk == future.event_risk
    assert factual.concept_probs == future.concept_probs
    assert factual.top_next == future.top_next
