"""Input-level counterfactual forecasts: edits, re-scoring, cohort summary."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.counterfactual import (
    STANDARD_EDITS,
    ValueEdit,
    apply_value_edits,
    cohort_counterfactual,
    counterfactual_forecast,
    score_record_at,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2024, 1, 1)
SBP = "LAB//220179//mmHg"  # MIMIC non-invasive SBP prefix (76534-7)
CREAT = "LAB//RESULT//50912//mg/dL"


def _events(n_subjects: int = 4) -> pl.DataFrame:
    rows = []
    for sid in range(1, n_subjects + 1):
        hadm = 100 + sid
        rows.append((sid, "HOSPITAL_ADMISSION//EMERGENCY", T0, None, hadm))
        for h in range(1, 40):
            rows.append((sid, SBP, T0 + timedelta(hours=h), 120.0, hadm))
            if h % 6 == 0:
                rows.append((sid, CREAT, T0 + timedelta(hours=h), 1.0, hadm))
        rows.append(
            (sid, "HOSPITAL_DISCHARGE//HOME", T0 + timedelta(hours=40), None, hadm)
        )
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


def _model(vocab: Vocabulary, n_concepts: int) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=n_concepts,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=["vasopressor_start", "death"],
    )


def test_edit_parsing_and_prefix_resolution() -> None:
    e = ValueEdit.parse("sbp_noninvasive:set:80:6")
    assert e.signal == "sbp_noninvasive" and e.mode == "set" and e.value == 80.0
    assert e.window_hours == 6.0 and e.prefixes("mimic_iv") == ["LAB//220179//"]
    assert ValueEdit.parse("creatinine:add:1:all").window_hours is None
    assert ValueEdit.parse("LAB//RESULT//:remove").prefixes("mimic_iv") == [
        "LAB//RESULT//"
    ]
    with pytest.raises(ValueError, match="unknown mode"):
        ValueEdit.parse("creatinine:double:2")
    with pytest.raises(ValueError, match="unknown signal"):
        ValueEdit("nothing", "set", 1.0).prefixes("mimic_iv")
    assert set(STANDARD_EDITS) >= {"hypotension_6h", "creatinine_plus_1"}


def test_apply_value_edits_touches_only_the_window_before_the_index() -> None:
    events = _events(1)
    index = T0 + timedelta(hours=24)
    edited, touched = apply_value_edits(
        events, [ValueEdit("sbp_noninvasive", "set", 80.0, 6.0)], index_time=index
    )
    assert touched == 6  # hours 19..24 inclusive
    sbp = edited.filter(pl.col("code") == SBP).sort("time")
    vals = dict(zip(sbp["time"].to_list(), sbp["numeric_value"].to_list()))
    assert vals[T0 + timedelta(hours=18)] == 120.0
    assert vals[T0 + timedelta(hours=19)] == 80.0
    assert vals[T0 + timedelta(hours=24)] == 80.0
    assert vals[T0 + timedelta(hours=25)] == 120.0  # after the index: untouched
    removed, n = apply_value_edits(
        events, [ValueEdit("creatinine", "remove", 0.0, None)], index_time=index
    )
    assert n == 4 and removed.filter(pl.col("code") == CREAT).height == 6 - 4
    added, _ = apply_value_edits(
        events, [ValueEdit("creatinine", "add", 1.0, 24.0)], index_time=index
    )
    assert added.filter(pl.col("code") == CREAT)[
        "numeric_value"
    ].max() == pytest.approx(2.0)


def test_counterfactual_changes_the_forecast_and_no_op_edit_does_not() -> None:
    events = _events(1)
    binned = add_value_tokens(events)
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    # a hypotensive SBP token must exist in the vocabulary for the edit to
    # produce a known token; build it from an edited copy too
    low, _ = apply_value_edits(
        events,
        [ValueEdit("sbp_noninvasive", "set", 80.0, None)],
        index_time=T0 + timedelta(hours=40),
    )
    vocab = Vocabulary.build(
        binned["code"].to_list() + add_value_tokens(low)["code"].to_list(), min_count=1
    )
    concepts = [c.name for c in concepts_for_source("mimic_iv")]
    model = _model(vocab, len(concepts))
    index = T0 + timedelta(hours=24)
    read = score_record_at(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_names=concepts,
        chunk_size=16,
    )
    assert read.index_time_hours == pytest.approx(24.0)
    # position = last token at or before 24h: admission + 24 SBP + 4 creatinine - 1
    assert read.position == 1 + 24 + 4 - 1
    assert set(read.event_risk) == {"vasopressor_start", "death"}
    assert set(read.event_risk["death"]) == {"8h", "24h", "72h"}
    assert len(read.concept_probs) == len(concepts) and len(read.top_next) == 5

    res = counterfactual_forecast(
        model,
        vocab,
        None,
        events,
        [ValueEdit("sbp_noninvasive", "set", 80.0, 6.0)],
        index_time=index,
        concept_names=concepts,
        chunk_size=16,
    )
    assert res.rows_edited == 6 and res.counterfactual.position == res.factual.position
    assert any(abs(d) > 0 for hs in res.delta_event_risk.values() for d in hs.values())
    noop = counterfactual_forecast(
        model,
        vocab,
        None,
        events,
        [ValueEdit("lactate", "set", 5.0, 6.0)],
        index_time=index,
        concept_names=concepts,
        chunk_size=16,
    )
    assert noop.rows_edited == 0
    assert all(d == 0 for hs in noop.delta_event_risk.values() for d in hs.values())
    assert all(d == 0 for d in noop.delta_concepts.values())


def test_cohort_summary_counts_edited_subjects_and_sign_agreement() -> None:
    events = _events(4)
    binned = add_value_tokens(events)
    low, _ = apply_value_edits(
        events,
        [ValueEdit("sbp_noninvasive", "set", 80.0, None)],
        index_time=T0 + timedelta(hours=40),
    )
    vocab = Vocabulary.build(
        binned["code"].to_list() + add_value_tokens(low)["code"].to_list(), min_count=1
    )
    concepts = [c.name for c in concepts_for_source("mimic_iv")]
    model = _model(vocab, len(concepts))
    edit = ValueEdit("sbp_noninvasive", "set", 80.0, 6.0, {"vasopressor_start": +1})
    summary = cohort_counterfactual(
        model,
        vocab,
        None,
        events,
        [edit],
        concept_names=concepts,
        index_hours=24.0,
        max_subjects=3,
        chunk_size=16,
        keep_per_subject=True,
    )
    assert summary.n_subjects == 3 and summary.n_edited == 3
    assert set(summary.mean_delta_event_risk) == {"vasopressor_start", "death"}
    assert set(summary.sign_agreement) == {"vasopressor_start"}
    assert all(
        0.0 <= v <= 1.0 for v in summary.sign_agreement["vasopressor_start"].values()
    )
    assert len(summary.per_subject) == 3
    assert len(summary.mean_delta_concepts) == len(concepts)
