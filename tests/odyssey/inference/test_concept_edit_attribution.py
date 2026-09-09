"""Occlusion attribution: which codes a concept's probability rests on."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.concept_edit_attribution import (
    WORSEN_EDIT_FOR_SIGNAL,
    CodeAttribution,
    CodeEdit,
    auto_edit_from_attribution,
    occlusion_attribution,
    remove_code_exact,
    score_with_codes_removed,
    score_with_worsen_edits,
    worsen_edits_from_attribution,
)
from odyssey.inference.counterfactual import (
    ValueEdit,
    apply_value_edits,
    score_record_at,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2024, 1, 1)
SBP = "LAB//220179//mmHg"  # MIMIC non-invasive SBP prefix (76534-7)
CREAT = "LAB//RESULT//50912//mg/dL"
LACTATE = "LAB//RESULT//50813//mmol/L"
SCHEMA = {
    "subject_id": pl.Int64,
    "code": pl.Utf8,
    "time": pl.Datetime("us"),
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _events() -> pl.DataFrame:
    rows: list[tuple[int, str, datetime, float | None, int]] = [
        (1, "HOSPITAL_ADMISSION//EMERGENCY", T0, None, 101)
    ]
    for h in range(1, 30):
        rows.append((1, SBP, T0 + timedelta(hours=h), 120.0, 101))
        if h % 6 == 0:
            rows.append((1, CREAT, T0 + timedelta(hours=h), 1.0, 101))
    # lactate only far outside the 24h lookback window, so it's a control:
    # occlusion must not find it, since it never entered the window at all.
    rows.append((1, LACTATE, T0 - timedelta(hours=100), 1.0, 101))
    rows.append((1, "HOSPITAL_DISCHARGE//HOME", T0 + timedelta(hours=30), None, 101))
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


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


def _vocab_and_model(
    events: pl.DataFrame,
) -> tuple[Vocabulary, ConceptBottleneckSequenceModel, list[str]]:
    binned = add_value_tokens(events)
    vocab = Vocabulary.build(binned["code"].to_list(), min_count=1)
    concepts = [c.name for c in concepts_for_source("mimic_iv")]
    model = _model(vocab, len(concepts))
    return vocab, model, concepts


# ---------------------------------------------------------------------------
# remove_code_exact: the window and equality semantics everything else
# depends on
# ---------------------------------------------------------------------------


def test_remove_code_exact_only_touches_the_window_before_the_index() -> None:
    events = _events()
    index = T0 + timedelta(hours=24)
    edited, touched = remove_code_exact(events, SBP, index_time=index, window_hours=6.0)
    assert touched == 6  # hours 19..24 inclusive
    remaining_sbp = edited.filter(pl.col("code") == SBP)
    assert remaining_sbp.height == events.filter(pl.col("code") == SBP).height - 6
    # boundary: the reading exactly at the index time is inside (<=), the
    # one immediately after is not
    kept_times = set(remaining_sbp["time"].to_list())
    assert T0 + timedelta(hours=25) in kept_times  # after index: untouched
    assert T0 + timedelta(hours=24) not in kept_times  # at index: removed
    assert T0 + timedelta(hours=18) in kept_times  # before window: untouched


def test_remove_code_exact_full_lookback_window_boundary() -> None:
    # hourly SBP readings for h=1..29; a 24h window ending at h=24 should
    # remove exactly hours 1..24 (24 rows), an off-by-one-prone boundary.
    events = _events()
    index = T0 + timedelta(hours=24)
    _, touched = remove_code_exact(events, SBP, index_time=index, window_hours=24.0)
    assert touched == 24


def test_remove_code_exact_does_not_touch_other_codes() -> None:
    events = _events()
    index = T0 + timedelta(hours=24)
    edited, touched = remove_code_exact(
        events, SBP, index_time=index, window_hours=24.0
    )
    assert touched > 0
    assert (
        edited.filter(pl.col("code") == CREAT).height
        == events.filter(pl.col("code") == CREAT).height
    )


def test_remove_code_exact_no_occurrences_is_a_no_op() -> None:
    events = _events()
    index = T0 + timedelta(hours=24)
    edited, touched = remove_code_exact(
        events, "NEVER//SEEN//CODE", index_time=index, window_hours=24.0
    )
    assert touched == 0
    assert edited.height == events.height


def test_remove_code_exact_does_not_cross_prefix_boundaries() -> None:
    """Regression test for the whole reason this module avoids prefix match.

    A code that is a literal string-prefix of a different, unrelated code
    must not have that other code's readings removed too.
    """
    short = "ICD//E11"
    long_code = "ICD//E11.9"
    rows = [
        (1, short, T0 + timedelta(hours=1), None, 101),
        (1, long_code, T0 + timedelta(hours=2), None, 101),
    ]
    events = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    index = T0 + timedelta(hours=6)
    edited, touched = remove_code_exact(
        events, short, index_time=index, window_hours=24.0
    )
    assert touched == 1
    assert edited.filter(pl.col("code") == long_code).height == 1
    assert edited.filter(pl.col("code") == short).height == 0


# ---------------------------------------------------------------------------
# occlusion_attribution
# ---------------------------------------------------------------------------


def test_occlusion_attribution_only_considers_the_lookback_window() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=concepts[0],
        concept_names=concepts,
        lookback_hours=24.0,
        chunk_size=16,
    )
    codes = {r.code for r in results}
    assert SBP in codes and CREAT in codes
    # lactate never has a reading inside the 24h window (it's 100h before
    # T0, let alone before the index), so it must be excluded, not
    # reported with n_rows=0.
    assert LACTATE not in codes
    assert all(r.n_rows > 0 for r in results)


def test_occlusion_attribution_exact_row_counts() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=concepts[0],
        concept_names=concepts,
        lookback_hours=24.0,
        chunk_size=16,
    )
    by_code = {r.code: r.n_rows for r in results}
    assert by_code[SBP] == 24  # hourly, hours 1..24
    assert by_code[CREAT] == 4  # every 6h: 6, 12, 18, 24


def test_occlusion_attribution_sorted_by_absolute_delta_descending() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=concepts[0],
        concept_names=concepts,
        lookback_hours=24.0,
        chunk_size=16,
    )
    deltas = [abs(r.delta) for r in results]
    assert deltas == sorted(deltas, reverse=True)
    for r in results:
        assert isinstance(r, CodeAttribution)
        assert r.occluded == pytest.approx(r.baseline + r.delta)


def test_occlusion_attribution_respects_a_candidate_pool() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=concepts[0],
        concept_names=concepts,
        lookback_hours=24.0,
        candidate_codes=[SBP],
        chunk_size=16,
    )
    assert [r.code for r in results] == [SBP]


def test_occlusion_attribution_empty_candidate_pool_returns_empty() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=concepts[0],
        concept_names=concepts,
        candidate_codes=[],
        chunk_size=16,
    )
    assert results == []


def test_occlusion_attribution_candidate_with_no_occurrences_is_skipped() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=concepts[0],
        concept_names=concepts,
        candidate_codes=[SBP, "NEVER//SEEN//CODE"],
        chunk_size=16,
    )
    assert [r.code for r in results] == [SBP]


def test_occlusion_attribution_baseline_matches_score_record_at() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    target = concepts[0]
    direct = score_record_at(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_names=concepts,
        chunk_size=16,
    )
    results = occlusion_attribution(
        model,
        vocab,
        None,
        events,
        index_time=index,
        concept_name=target,
        concept_names=concepts,
        candidate_codes=[SBP],
        chunk_size=16,
    )
    assert results[0].baseline == pytest.approx(direct.concept_probs[target])


def test_occlusion_attribution_unknown_concept_raises() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    with pytest.raises(ValueError, match="not in concept_names"):
        occlusion_attribution(
            model,
            vocab,
            None,
            events,
            index_time=index,
            concept_name="not_a_real_concept",
            concept_names=concepts,
            chunk_size=16,
        )


# ---------------------------------------------------------------------------
# score_with_codes_removed
# ---------------------------------------------------------------------------


def test_score_with_codes_removed_sums_touched_across_codes() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    readout, touched = score_with_codes_removed(
        model,
        vocab,
        None,
        events,
        [SBP, CREAT],
        index_time=index,
        concept_names=concepts,
        window_hours=24.0,
        chunk_size=16,
    )
    assert touched == 24 + 4
    assert len(readout.concept_probs) == len(concepts)


def test_score_with_codes_removed_matches_manual_double_removal() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    manual, t1 = remove_code_exact(events, SBP, index_time=index, window_hours=24.0)
    manual, t2 = remove_code_exact(manual, CREAT, index_time=index, window_hours=24.0)
    expected = score_record_at(
        model,
        vocab,
        None,
        manual,
        index_time=index,
        concept_names=concepts,
        chunk_size=16,
    )
    readout, touched = score_with_codes_removed(
        model,
        vocab,
        None,
        events,
        [SBP, CREAT],
        index_time=index,
        concept_names=concepts,
        window_hours=24.0,
        chunk_size=16,
    )
    assert touched == t1 + t2
    assert readout.concept_probs == pytest.approx(expected.concept_probs)


# ---------------------------------------------------------------------------
# auto_edit_from_attribution
# ---------------------------------------------------------------------------


def test_auto_edit_from_attribution_builds_removal_edits_from_top_k() -> None:
    attributions = [
        CodeAttribution(code="A", n_rows=3, baseline=0.5, occluded=0.9),  # delta 0.4
        CodeAttribution(code="B", n_rows=2, baseline=0.5, occluded=0.4),  # delta -0.1
        CodeAttribution(code="C", n_rows=1, baseline=0.5, occluded=0.51),  # delta 0.01
    ]
    edits = auto_edit_from_attribution(attributions, top_k=2, window_hours=12.0)
    assert edits == [
        CodeEdit(code="A", window_hours=12.0),
        CodeEdit(code="B", window_hours=12.0),
    ]


def test_auto_edit_from_attribution_min_abs_delta_filters_within_top_k() -> None:
    attributions = [
        CodeAttribution(code="A", n_rows=3, baseline=0.5, occluded=0.9),  # delta 0.4
        CodeAttribution(code="B", n_rows=2, baseline=0.5, occluded=0.4),  # delta -0.1
        CodeAttribution(code="C", n_rows=1, baseline=0.5, occluded=0.51),  # delta 0.01
    ]
    filtered = auto_edit_from_attribution(
        attributions, top_k=3, window_hours=12.0, min_abs_delta=0.05
    )
    assert [e.code for e in filtered] == ["A", "B"]


def test_auto_edit_from_attribution_top_k_zero_is_empty() -> None:
    attributions = [CodeAttribution(code="A", n_rows=1, baseline=0.5, occluded=0.9)]
    assert auto_edit_from_attribution(attributions, top_k=0) == []


def test_auto_edit_from_attribution_empty_input_is_empty() -> None:
    assert auto_edit_from_attribution([], top_k=3) == []


def test_auto_edit_from_attribution_min_abs_delta_can_exclude_everything() -> None:
    attributions = [
        CodeAttribution(code="A", n_rows=3, baseline=0.5, occluded=0.51),
    ]
    assert auto_edit_from_attribution(attributions, top_k=3, min_abs_delta=1.0) == []


# ---------------------------------------------------------------------------
# worsen_edits_from_attribution / score_with_worsen_edits
# ---------------------------------------------------------------------------


def test_worsen_edits_from_attribution_resolves_known_signals() -> None:
    attributions = [
        CodeAttribution(code=SBP, n_rows=5, baseline=0.5, occluded=0.9),
        CodeAttribution(code=CREAT, n_rows=1, baseline=0.5, occluded=0.6),
    ]
    edits = worsen_edits_from_attribution(
        attributions, source="mimic_iv", top_k=2, window_hours=12.0
    )
    by_signal = {e.signal: e for e in edits}
    assert set(by_signal) == {"sbp_noninvasive", "creatinine"}
    assert (
        by_signal["sbp_noninvasive"].mode
        == WORSEN_EDIT_FOR_SIGNAL["sbp_noninvasive"].mode
    )
    assert (
        by_signal["sbp_noninvasive"].value
        == WORSEN_EDIT_FOR_SIGNAL["sbp_noninvasive"].value
    )
    assert by_signal["sbp_noninvasive"].window_hours == 12.0


def test_worsen_edits_from_attribution_drops_unresolvable_codes() -> None:
    attributions = [
        CodeAttribution(
            code="NOT//A//REAL//SIGNAL", n_rows=1, baseline=0.5, occluded=0.6
        ),
    ]
    edits = worsen_edits_from_attribution(attributions, source="mimic_iv", top_k=1)
    assert edits == []


def test_worsen_edits_from_attribution_deduplicates_same_signal() -> None:
    # two distinct raw codes that both resolve to sbp_noninvasive (the
    # panel resolves by LOINC-derived prefix, and MIMIC has more than one
    # chartevents itemid family per signal in general -- here we just
    # attribute the same code twice to exercise the collapse path)
    attributions = [
        CodeAttribution(code=SBP, n_rows=5, baseline=0.5, occluded=0.9),
        CodeAttribution(code=SBP, n_rows=5, baseline=0.5, occluded=0.9),
    ]
    edits = worsen_edits_from_attribution(attributions, source="mimic_iv", top_k=2)
    assert len(edits) == 1
    assert edits[0].signal == "sbp_noninvasive"


def test_worsen_edits_from_attribution_respects_top_k_and_min_delta() -> None:
    attributions = [
        CodeAttribution(code=SBP, n_rows=5, baseline=0.5, occluded=0.9),  # delta 0.4
        CodeAttribution(
            code=CREAT, n_rows=1, baseline=0.5, occluded=0.51
        ),  # delta 0.01
    ]
    only_top1 = worsen_edits_from_attribution(attributions, source="mimic_iv", top_k=1)
    assert [e.signal for e in only_top1] == ["sbp_noninvasive"]

    filtered = worsen_edits_from_attribution(
        attributions, source="mimic_iv", top_k=2, min_abs_delta=0.05
    )
    assert [e.signal for e in filtered] == ["sbp_noninvasive"]


def test_score_with_worsen_edits_matches_manual_apply_and_score() -> None:
    events = _events()
    vocab, model, concepts = _vocab_and_model(events)
    index = T0 + timedelta(hours=24)
    edits = [ValueEdit("sbp_noninvasive", "set", 80.0, 24.0)]

    manual_edited, manual_touched = apply_value_edits(
        events, edits, index_time=index, source="mimic_iv"
    )
    expected = score_record_at(
        model,
        vocab,
        None,
        manual_edited,
        index_time=index,
        concept_names=concepts,
        chunk_size=16,
    )
    readout, touched = score_with_worsen_edits(
        model,
        vocab,
        None,
        events,
        edits,
        index_time=index,
        concept_names=concepts,
        source="mimic_iv",
        chunk_size=16,
    )
    assert touched == manual_touched
    assert readout.concept_probs == pytest.approx(expected.concept_probs)


def test_worsen_edit_for_signal_covers_every_sofa_component() -> None:
    # respiration, coagulation, liver, cardiovascular, CNS, renal -- the
    # six components Sepsis-3's SOFA score is built from, plus lactate.
    required = {
        "spo2",
        "fio2",  # respiration
        "platelets",  # coagulation
        "bilirubin_total",  # liver
        "sbp_noninvasive",
        "dbp_noninvasive",  # cardiovascular
        "gcs_eye",
        "gcs_verbal",
        "gcs_motor",  # CNS
        "creatinine",  # renal
        "lactate",
    }
    assert required <= set(WORSEN_EDIT_FOR_SIGNAL)


def test_worsen_edit_for_signal_covers_qsofa_resp_rate() -> None:
    # qSOFA = altered mentation (GCS, already in the SOFA set above) + SBP
    # <=100 (also already covered) + respiratory rate >=22/min.
    edit = WORSEN_EDIT_FOR_SIGNAL["resp_rate"]
    assert edit.mode == "set"
    assert edit.value >= 22.0
