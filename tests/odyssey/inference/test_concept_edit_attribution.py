"""Occlusion attribution: which codes a concept's probability rests on."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# Cohort validation and the random-code control
# ---------------------------------------------------------------------------

import odyssey.inference.legacy_concept_pins as pins_module  # noqa: E402
import odyssey.inference.run_inference as run_inference_module  # noqa: E402
import odyssey.training.data as data_module  # noqa: E402
from odyssey.inference.concept_edit_attribution import (  # noqa: E402
    CohortWorsenResult,
    _main,
    cohort_worsen,
    format_cohort_summary,
    mappable_candidate_codes,
    select_random_codes,
    worsen_edits_for_codes,
)
from odyssey.inference.counterfactual import _index_times_by_subject  # noqa: E402
from scripts.parse_edit_attribution_logs import parse_log  # noqa: E402


PLT = "LAB//RESULT//51265//K/uL"  # platelets: panel signal with a worsen edit
GLUCOSE = "LAB//RESULT//50931//mg/dL"  # panel signal WITHOUT a worsen edit
DIAG = "DIAG//ICD10//E11"  # outside the panel entirely


def _cohort_events(n_subjects: int = 3) -> pl.DataFrame:
    """``n_subjects`` records, each a 30 h visit with three worsenable codes."""
    rows: list[tuple[int, str, datetime, float | None, int]] = []
    for sid in range(1, n_subjects + 1):
        hadm = 100 + sid
        rows.append((sid, "HOSPITAL_ADMISSION//EMERGENCY", T0, None, hadm))
        rows.append((sid, DIAG, T0 + timedelta(hours=1), None, hadm))
        for h in range(1, 30):
            rows.append((sid, SBP, T0 + timedelta(hours=h), 120.0 + sid, hadm))
            if h % 6 == 0:
                rows.append((sid, CREAT, T0 + timedelta(hours=h), 1.0, hadm))
            if h % 8 == 0:
                rows.append((sid, PLT, T0 + timedelta(hours=h), 200.0, hadm))
                rows.append((sid, GLUCOSE, T0 + timedelta(hours=h), 100.0, hadm))
        rows.append(
            (sid, "HOSPITAL_DISCHARGE//HOME", T0 + timedelta(hours=30), None, hadm)
        )
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def test_mappable_candidate_codes_keeps_only_worsenable_window_codes() -> None:
    events = _cohort_events(1)
    pool = mappable_candidate_codes(
        events, index_time=T0 + timedelta(hours=24), lookback_hours=24.0
    )
    # glucose is a panel signal but has no worsen edit; the diagnosis is
    # outside the panel; both are excluded. Sorted, so the pool is stable.
    assert pool == sorted([SBP, CREAT, PLT])


def test_select_random_codes_draws_n_from_pool_and_is_seed_reproducible() -> None:
    pool = [SBP, CREAT, PLT, "LAB//220180//mmHg"]
    a = select_random_codes(pool, n=2, seed=0, subject_id=7)
    b = select_random_codes(pool, n=2, seed=0, subject_id=7)
    assert a == b
    assert len(a) == 2
    assert set(a) <= set(pool)
    assert len(set(a)) == 2  # without replacement
    # a different seed or subject changes the draw somewhere in the pool
    draws = {
        tuple(select_random_codes(pool, n=2, seed=s, subject_id=7)) for s in range(20)
    }
    assert len(draws) > 1
    # fewer candidates than n: everything, same count the attributed arm gets
    assert sorted(select_random_codes([SBP], n=3, seed=0, subject_id=1)) == [SBP]
    with pytest.raises(ValueError, match="non-negative"):
        select_random_codes(pool, n=-1, seed=0, subject_id=1)


def test_worsen_edits_for_codes_matches_attribution_path() -> None:
    attributions = [
        CodeAttribution(code=CREAT, n_rows=4, baseline=0.5, occluded=0.1),
        CodeAttribution(code=SBP, n_rows=24, baseline=0.5, occluded=0.7),
        CodeAttribution(code=DIAG, n_rows=1, baseline=0.5, occluded=0.55),
        CodeAttribution(code=GLUCOSE, n_rows=3, baseline=0.5, occluded=0.52),
    ]
    via_attribution = worsen_edits_from_attribution(attributions, top_k=4)
    via_codes = worsen_edits_for_codes([a.code for a in attributions])
    assert via_codes == via_attribution
    assert [e.signal for e in via_codes] == ["creatinine", "sbp_noninvasive"]


def test_index_frac_places_the_cut_inside_the_visit() -> None:
    events = _cohort_events(1)
    fixed = _index_times_by_subject(events, index_hours=24.0)
    frac = _index_times_by_subject(events, index_hours=24.0, index_frac=0.5)
    assert fixed[1] == T0 + timedelta(hours=24)
    assert frac[1] == T0 + timedelta(hours=15)  # 30 h visit, half way
    with pytest.raises(ValueError, match="index_frac"):
        _index_times_by_subject(events, index_hours=24.0, index_frac=0.0)


def _run_cohort(
    events: pl.DataFrame, **kwargs: object
) -> tuple[list[str], CohortWorsenResult]:
    vocab, model, concepts = _vocab_and_model(events)
    lines: list[str] = []
    result = cohort_worsen(
        model,
        vocab,
        None,
        events,
        concept_name=concepts[0],
        concept_names=concepts,
        top_k=2,
        lookback_hours=24.0,
        index_hours=24.0,
        chunk_size=16,
        log=lines.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return lines, result


def test_cohort_worsen_attributed_default_scores_every_subject() -> None:
    events = _cohort_events(3)
    lines, result = _run_cohort(events)
    assert result.code_selection == "attributed"
    assert result.random_seed is None
    assert [s.subject_id for s in result.subjects] == [1, 2, 3]
    assert result.n_pool_above_top_k == 3  # pool of 3 worsenable codes, top_k=2
    for s in result.subjects:
        assert s.n_candidates == 3
        assert len(s.codes) == 2
        assert set(s.codes) <= {SBP, CREAT, PLT}
        assert s.rows_edited > 0
        assert set(s.delta_event_risk) == {"vasopressor_start", "death"}
        assert set(s.delta_event_risk["death"]) == {"8h", "24h", "72h"}
    assert len(lines) == 3
    assert lines[0].startswith("subject 1 [1/50]: [")


def test_cohort_worsen_random_draws_same_count_from_same_pool_and_subjects() -> None:
    events = _cohort_events(3)
    _, attributed = _run_cohort(events)
    _, random_a = _run_cohort(events, code_selection="random", random_seed=3)
    _, random_b = _run_cohort(events, code_selection="random", random_seed=3)
    assert random_a.code_selection == "random"
    assert random_a.random_seed == 3
    # same subjects, in the same order, as the attributed arm
    assert [s.subject_id for s in random_a.subjects] == [
        s.subject_id for s in attributed.subjects
    ]
    for att, rnd in zip(attributed.subjects, random_a.subjects):
        assert rnd.n_candidates == att.n_candidates
        assert len(rnd.codes) == len(att.codes)
        assert set(rnd.codes) <= {SBP, CREAT, PLT}
        assert rnd.rows_edited > 0
    # seed-reproducible, byte for byte
    assert [s.codes for s in random_a.subjects] == [s.codes for s in random_b.subjects]
    assert [s.delta_event_risk for s in random_a.subjects] == [
        s.delta_event_risk for s in random_b.subjects
    ]
    # and the draw actually varies with the seed somewhere in the cohort
    _, random_c = _run_cohort(events, code_selection="random", random_seed=11)
    assert [s.codes for s in random_a.subjects] != [s.codes for s in random_c.subjects]


def test_cohort_worsen_rejects_unknown_selection_and_pool() -> None:
    events = _cohort_events(1)
    with pytest.raises(ValueError, match="code_selection"):
        _run_cohort(events, code_selection="bogus")
    with pytest.raises(ValueError, match="candidate_pool"):
        _run_cohort(events, candidate_pool="bogus")


def test_cohort_worsen_all_pool_reports_unedited_subjects() -> None:
    # with every window code as a candidate, a random draw can land on
    # codes with no worsen edit; those subjects are kept (same cohort) but
    # not scored, and the header says how many.
    events = _cohort_events(2)
    _, result = _run_cohort(events, candidate_pool="all")
    assert result.candidate_pool == "all"
    assert all(s.n_candidates == 5 for s in result.subjects)  # 3 worsenable + 2 not
    text = format_cohort_summary(result)
    assert "pool=all" in text.splitlines()[0]


def test_format_cohort_summary_round_trips_through_the_parser() -> None:
    events = _cohort_events(3)
    _, attributed = _run_cohort(events)
    _, rnd = _run_cohort(events, code_selection="random", random_seed=5)
    att_text = format_cohort_summary(attributed)
    rnd_text = format_cohort_summary(rnd)
    att_head = att_text.splitlines()[0]
    rnd_head = rnd_text.splitlines()[0]
    assert att_head.endswith("top_k=2, selection=attributed ===")
    assert rnd_head.endswith("top_k=2, selection=random, seed=5 ===")
    parsed_att = parse_log(att_text)
    parsed_rnd = parse_log(rnd_text)
    assert parsed_att["selection"] == "attributed"
    assert parsed_att["random_seed"] is None
    assert parsed_rnd["selection"] == "random"
    assert parsed_rnd["random_seed"] == 5
    for parsed in (parsed_att, parsed_rnd):
        assert parsed["n_subjects"] == 3
        assert parsed["n_baseline_scored"] == 3
        assert {(c["event"], c["horizon"]) for c in parsed["cells"]} == {
            (ev, h)
            for ev in ("death", "vasopressor_start")
            for h in ("8h", "24h", "72h")
        }
        assert all(c["total"] == 3 for c in parsed["cells"])
        assert all(0 <= c["agree"] <= 3 for c in parsed["cells"])
    # the per-subject lines carry the signals, like the banked logs
    assert "edit signal frequency: {" in att_text
    assert "mean concept-probability delta: " in att_text


# ---------------------------------------------------------------------------
# cohort_worsen edge paths and the CLI entry point
# ---------------------------------------------------------------------------


def test_cohort_worsen_unknown_concept_raises() -> None:
    events = _cohort_events(1)
    vocab, model, concepts = _vocab_and_model(events)
    with pytest.raises(ValueError, match="not in concept_names"):
        cohort_worsen(
            model,
            vocab,
            None,
            events,
            concept_name="not_a_concept",
            concept_names=concepts,
            chunk_size=16,
        )


def test_cohort_worsen_stops_at_max_subjects() -> None:
    _, result = _run_cohort(_cohort_events(3), max_subjects=1)
    assert [s.subject_id for s in result.subjects] == [1]


def test_cohort_worsen_skips_subjects_with_an_empty_pool() -> None:
    # subject 3 has a 30 h visit but nothing worsenable in the window, so it
    # is skipped in both arms rather than scored with nothing to edit.
    events = _cohort_events(2)
    extra = [
        (3, "HOSPITAL_ADMISSION//EMERGENCY", T0, None, 103),
        *[(3, DIAG, T0 + timedelta(hours=h), None, 103) for h in range(1, 30)],
        (3, "HOSPITAL_DISCHARGE//HOME", T0 + timedelta(hours=30), None, 103),
    ]
    events = pl.concat([events, pl.DataFrame(extra, schema=SCHEMA, orient="row")])
    _, attributed = _run_cohort(events)
    _, rnd = _run_cohort(events, code_selection="random", random_seed=0)
    assert [s.subject_id for s in attributed.subjects] == [1, 2]
    assert [s.subject_id for s in rnd.subjects] == [1, 2]


def test_cohort_worsen_random_all_pool_can_leave_a_subject_unedited() -> None:
    # seed 6 draws glucose + the diagnosis for subject 1 (neither has a
    # worsen edit) and worsenable codes for subject 2; the unedited subject
    # is kept, scored as a no-op, and counted in the header.
    events = _cohort_events(2)
    _, result = _run_cohort(
        events, code_selection="random", random_seed=6, candidate_pool="all"
    )
    by_sid = {s.subject_id: s for s in result.subjects}
    assert set(by_sid) == {1, 2}
    assert by_sid[1].rows_edited == 0
    assert by_sid[1].signals == []
    assert by_sid[1].concept_after == by_sid[1].concept_before
    assert all(
        v == 0.0 for hs in by_sid[1].delta_event_risk.values() for v in hs.values()
    )
    assert by_sid[2].rows_edited > 0
    assert [s.subject_id for s in result.scored] == [2]
    header = format_cohort_summary(result).splitlines()[0]
    assert "pool=all" in header
    assert header.endswith("no_edit=1 ===")


def test_format_cohort_summary_reports_index_frac() -> None:
    _, result = _run_cohort(_cohort_events(1), index_frac=0.5)
    header = format_cohort_summary(result).splitlines()[0]
    assert "index_frac=0.5000" in header
    assert result.index_frac == 0.5


def test_cli_main_runs_the_cohort_and_writes_the_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = _cohort_events(2)
    vocab, model, concepts = _vocab_and_model(events)
    config = SimpleNamespace(source="mimic_iv", task_set="v1")
    seen: dict[str, object] = {}

    def fake_load_run(run_dir: Path, **kwargs: object) -> tuple[object, ...]:
        seen["run_dir"] = run_dir
        seen["checkpoint"] = kwargs["checkpoint_path"]
        return model, vocab, None, config

    monkeypatch.setattr(run_inference_module, "load_run", fake_load_run)
    monkeypatch.setattr(
        pins_module,
        "resolve_concepts_for_run",
        lambda run_dir, source, task_set: concepts_for_source(source),
    )
    monkeypatch.setattr(
        data_module, "load_meds_shards", lambda path, max_shards: events.clone()
    )
    out = tmp_path / "nested" / "cohort_random.json"
    argv = [
        "prog",
        "--run-dir",
        str(tmp_path / "run"),
        "--held-out-shard-dir",
        str(tmp_path / "held_out"),
        "--concept",
        concepts[0],
        "--output-json",
        str(out),
        "--top-k",
        "2",
        "--max-subjects",
        "2",
        "--code-selection",
        "random",
        "--random-seed",
        "3",
        "--chunk-size",
        "16",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr("sys.argv", argv)
    _main()

    assert seen["run_dir"] == tmp_path / "run"
    assert seen["checkpoint"] == tmp_path / "run" / "checkpoint_best.pt"
    payload = json.loads(out.read_text())
    assert payload["concept"] == concepts[0]
    assert payload["code_selection"] == "random"
    assert payload["random_seed"] == 3
    assert payload["top_k"] == 2
    assert payload["run_dir"] == str(tmp_path / "run")
    assert [s["subject_id"] for s in payload["subjects"]] == [1, 2]
    assert set(payload["sign_agreement"]) == {"vasopressor_start", "death"}
    for horizons in payload["sign_agreement"].values():
        for cell in horizons.values():
            assert set(cell) == {"agree", "total"}
            assert cell["total"] == 2
    parsed = parse_log(payload["summary_text"])
    assert parsed["selection"] == "random"
    assert parsed["random_seed"] == 3
    printed = capsys.readouterr().out
    assert "worsenable signals with a reading in the loaded shards:" in printed
    assert "subject 1 [1/2]:" in printed
    assert payload["summary_text"] in printed
