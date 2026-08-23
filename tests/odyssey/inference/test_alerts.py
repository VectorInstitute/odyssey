"""Tests for the alert (time-to-event) evaluation harness."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pytest
import torch

from odyssey.data.alert_events import (
    ALERT_EVENTS,
    ALERT_EVENTS_V2,
    AlertEvent,
    EventTimes,
    all_event_times,
)
from odyssey.data.concepts import concepts_for_source
from odyssey.data.sequences import BIRTH_CODE
from odyssey.data.streaming import NO_SUBJECT
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference import alerts as alerts_module
from odyssey.inference.alerts import (
    GBM_MIN_OBSERVED,
    LANDMARK_PROTOCOL_VERSION,
    IndexRow,
    _index_rows_from_events,
    _landmark_mask,
    _positive_class_proba,
    _stamp_landmark_protocol_version,
    _tune_gbm,
    _visit_starts,
    baseline_features,
    collect_model_scores,
    features_for_events,
    fit_baselines,
    fit_baselines_streaming,
    index_row_table,
    load_index_row_table,
    outcome_at_horizon,
    score_alerts,
    sparse_columns,
    verify_packed_landmark_rows,
    verify_rows_match_dump,
)
from odyssey.inference.baseline_features import feature_names
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS
from odyssey.training.data import load_meds_shard, load_meds_shards
from odyssey.training.shard_stream import make_preparer, merge_event_times, shard_paths


T0 = datetime(2024, 1, 1)


def _events(n_subjects: int = 24) -> pl.DataFrame:
    """Build hourly heart-rate readings with a planted deterioration signal.

    Every other subject spikes at hour 12 and starts norepinephrine at
    hour 14; every fourth also gets an ICU admission at hour 6.
    """
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in range(1, n_subjects + 1):
        hadm = 1000 + sid
        for h in range(24):
            hr = 80.0
            if sid % 2 == 0 and h >= 12:
                hr = 130.0
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), hr, hadm))
        if sid % 2 == 0:
            rows.append(
                (
                    sid,
                    "MEDICATION//norepinephrine//Administered",
                    T0 + timedelta(hours=14),
                    None,
                    hadm,
                )
            )
        if sid % 4 == 0:
            rows.append(
                (sid, "ICU_ADMISSION//MICU", T0 + timedelta(hours=6), None, hadm)
            )
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


def _vocab(events_binned: pl.DataFrame) -> Vocabulary:
    return Vocabulary.build(events_binned["code"].to_list(), min_count=1)


def _model(vocab_size: int, num_concepts: int) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=num_concepts,
        embedding_dim=4,
        padding_idx=0,
    )


def test_event_times_and_outcomes() -> None:
    events = _events(8)
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")
    vaso = times["vasopressor_start"]
    # subject 2 starts norepinephrine at hour 14; subject 1 never
    assert vaso.onset[(2, 1002)] == 14.0
    assert (1, 1001) not in vaso.onset
    icu = times["icu_admission"]
    assert icu.onset[(4, 1004)] == 6.0
    # outcomes for subject 2 at t=10: within 8h -> yes; at t=2, 8h -> no
    row = IndexRow(2, 1002, 10.0)
    assert outcome_at_horizon(row, vaso, 8.0) == 1
    assert outcome_at_horizon(IndexRow(2, 1002, 2.0), vaso, 8.0) == 0
    # after onset: not at risk
    assert outcome_at_horizon(IndexRow(2, 1002, 15.0), vaso, 8.0) is None
    # subject 1 at t=20 with 8h horizon: follow-up ends at 23h -> censored
    assert outcome_at_horizon(IndexRow(1, 1001, 20.0), vaso, 8.0) is None
    assert outcome_at_horizon(IndexRow(1, 1001, 10.0), vaso, 8.0) == 0


def test_harness_end_to_end_with_planted_signal() -> None:
    events = _events(24)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _model(len(vocab), len(concepts))
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")
    rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
    )
    # landmarks every 4h over 24h -> ~6 per visit
    assert len(rows["vasopressor_start"]) >= 24 * 5
    assert all(
        "concept" in r.scores and "next_mass" in r.scores
        for r in rows["vasopressor_start"]
    )
    assert all("concept" not in r.scores for r in rows["icu_admission"])

    # baseline: fit on the same synthetic data (a separate split in real use)
    train_rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, train_rows, times, horizons=(8.0,), feature_set="basic", tune=False
    )
    feats = {name: baseline_features(binned, rs) for name, rs in rows.items() if rs}
    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
    )
    by = {(r.event, r.scorer): r for r in results}
    # the planted signal (HR HIGH from hour 12) separates the landmark-12
    # positives; the landmark-8 positives (onset at 14, within 8h) still
    # look normal, so the ceiling here is well short of 1 but far above chance
    gbm = by[("vasopressor_start", "baseline_gbm")]
    assert gbm.auroc is not None and gbm.auroc > 0.75
    assert gbm.brier is not None and gbm.calibration
    # untrained model scores exist and are valid AUROCs
    assert 0.0 <= by[("vasopressor_start", "concept")].auroc <= 1.0
    assert 0.0 <= by[("vasopressor_start", "next_mass")].auroc <= 1.0
    # censoring is counted, never silently dropped
    assert gbm.n_censored > 0


def test_landmark_mask_state_prevents_a_spurious_landmark_at_a_chunk_boundary() -> None:
    """Direct unit test of the LandmarkState mechanism, isolated from the sampler.

    Lane 0: chunk 1 ends mid-bucket (hour 2, bucket 0); chunk 2 continues
    the SAME subject/visit at hour 3, still bucket 0 -- must NOT be a new
    landmark now that state carries across the boundary (the bug: it
    always was, before). Chunk 2 then moves to hour 5 (bucket 1) -- must
    be a landmark. Lane 1: a fully-padding chunk 2 must leave state
    unchanged rather than resetting it (checked via a chunk 3 continuing
    lane 1's real subject).
    """
    landmark_hours = 4.0
    starts = torch.zeros(2, 2)

    # Chunk 1: lane 0 subject 1 visit 10 at hours [0, 2] (both bucket 0);
    # lane 1 subject 2 visit 20 at hours [0, 1] (both bucket 0).
    mask1, state1 = _landmark_mask(
        time_hours=torch.tensor([[0.0, 2.0], [0.0, 1.0]]),
        subject_ids=torch.tensor([[1, 1], [2, 2]]),
        visit_ids=torch.tensor([[10, 10], [20, 20]]),
        landmark_hours=landmark_hours,
        visit_start_hours=starts,
    )
    assert mask1.tolist() == [[True, False], [True, False]]

    # Chunk 2: lane 0 continues subject 1/visit 10 at hours [3, 5] (bucket
    # 0 then 1); lane 1 is fully padding (subject ended, nothing yet to
    # replace it).
    mask2, state2 = _landmark_mask(
        time_hours=torch.tensor([[3.0, 5.0], [0.0, 0.0]]),
        subject_ids=torch.tensor([[1, 1], [NO_SUBJECT, NO_SUBJECT]]),
        visit_ids=torch.tensor([[10, 10], [-1, -1]]),
        landmark_hours=landmark_hours,
        visit_start_hours=starts,
        state=state1,
    )
    # The real fix: hour 3 continues bucket 0 from chunk 1 -- NOT a new
    # landmark. Hour 5 (bucket 1) is. Before this fix, position 0 of every
    # chunk was unconditionally a landmark regardless of continuity.
    assert mask2.tolist() == [[False, True], [False, False]]
    # Lane 1's state must be UNCHANGED (still subject 2/visit 20/bucket 0)
    # since chunk 2 had no real position in that lane.
    assert state2.subject_by_lane[1] == 2
    assert state2.last_bucket_by_lane[1] == {20: 0}

    # Chunk 3: lane 1's subject 2 resumes at hour 3 (still bucket 0, same
    # visit) -- must NOT be a new landmark, proving the padding chunk in
    # between didn't reset lane 1's carried state.
    mask3, _ = _landmark_mask(
        time_hours=torch.tensor([[0.0, 0.0], [3.0, 3.0]]),
        subject_ids=torch.tensor([[NO_SUBJECT, NO_SUBJECT], [2, 2]]),
        visit_ids=torch.tensor([[-1, -1], [20, 20]]),
        landmark_hours=landmark_hours,
        visit_start_hours=starts,
        state=state2,
    )
    assert mask3[1, 0].item() is False


def test_landmark_mask_first_call_with_no_state_matches_original_behavior() -> None:
    """Without `state` (first call), position 0 falls back to "nothing before it"."""
    mask, _ = _landmark_mask(
        time_hours=torch.tensor([[0.0, 1.0]]),
        subject_ids=torch.tensor([[1, 1]]),
        visit_ids=torch.tensor([[10, 10]]),
        landmark_hours=4.0,
        visit_start_hours=torch.zeros(1, 2),
    )
    assert mask.tolist() == [[True, False]]


def test_landmark_mask_a_new_subject_at_chunk_boundary_is_still_a_landmark() -> None:
    """Same numeric bucket, but a different (subject, visit) -- must still land."""
    landmark_hours = 4.0
    _, state = _landmark_mask(
        time_hours=torch.tensor([[2.0]]),
        subject_ids=torch.tensor([[1]]),
        visit_ids=torch.tensor([[10]]),
        landmark_hours=landmark_hours,
        visit_start_hours=torch.zeros(1, 1),
    )
    # Next patient in the same lane, same bucket value (0), different subject.
    mask, _ = _landmark_mask(
        time_hours=torch.tensor([[1.0]]),
        subject_ids=torch.tensor([[2]]),
        visit_ids=torch.tensor([[20]]),
        landmark_hours=landmark_hours,
        visit_start_hours=torch.zeros(1, 1),
        state=state,
    )
    assert mask.item() is True


def test_landmark_mask_interleaved_visits_at_a_shared_timestamp_land_once_each() -> (
    None
):
    """The v2->v3 regression: real discharge-instant medication-stop bundles.

    Root cause of a real bug found chasing verify_packed_landmark_rows'
    disagreement with _index_rows_from_events on real eICU data (subject
    454662, held-out shards): at a discharge instant, a subject's own
    medication-STOPPED events for two different admissions (hadm_ids) can
    share the exact same timestamp and appear interleaved in token order
    (ending admission, starting admission, ending, starting, ...) rather
    than grouped by visit. v2's ``_landmark_mask`` compared each position
    only to the one immediately before it (``~same_visit``), so every
    interleave step re-triggered a landmark even though that visit's
    bucket had already been landmarked -- 3 spurious landmarks per visit
    in the real case, from 6 real interleaved positions. v3 tracks the
    last-emitted bucket per visit directly (matching
    _index_rows_from_events' per-(subject, visit, bucket) group-by), so
    order no longer matters: each visit lands exactly once.
    """
    landmark_hours = 4.0
    same_time = 166.183333  # the literal real-data value from the repro
    # One lane, one chunk: 6 positions, visit_id alternating 10/20/10/20/
    # 10/20, all at the exact same timestamp (same bucket for both visits,
    # since visit_start_hours is 0 for both here -- the real case has
    # each visit's own start, this pins the same property more simply).
    mask, state = _landmark_mask(
        time_hours=torch.tensor([[same_time] * 6]),
        subject_ids=torch.tensor([[1, 1, 1, 1, 1, 1]]),
        visit_ids=torch.tensor([[10, 20, 10, 20, 10, 20]]),
        landmark_hours=landmark_hours,
        visit_start_hours=torch.zeros(1, 6),
    )
    # Exactly one landmark per visit: the FIRST occurrence of each, not
    # every interleave step.
    assert mask.tolist() == [[True, True, False, False, False, False]]
    assert state.last_bucket_by_lane[0] == {10: 41, 20: 41}


@pytest.mark.parametrize(
    "chunk_size",
    [
        pytest.param(16, id="multi_chunk"),  # spans >1 chunk per patient
        pytest.param(200, id="single_chunk"),  # whole patient fits in one chunk
    ],
)
def test_collect_model_scores_and_index_rows_from_events_agree_on_landmark_times(
    chunk_size: int,
) -> None:
    """Per-(subject, visit) landmark time-sets must match between the two paths.

    Real, confirmed divergence this pins (review finding 8, repro from 6e
    on eICU: collect_model_scores over-counted landmark rows by ~23%).
    Root cause was _landmark_mask being called fresh per streaming chunk
    with no bucket state carried across chunk boundaries -- a patient
    whose sequence spanned more than one chunk got a spurious extra
    landmark at the boundary even when still inside the same
    landmark_hours bucket as the chunk before it. Fixed (review finding
    19) by threading ``LandmarkState`` across chunks, the same way the
    model's own recurrent state already is. Both regimes are pinned here:
    chunk_size=16 (a subject's ~25-26-event sequence spans more than one
    chunk -- the condition that reproduced the divergence) and
    chunk_size=200 (a whole patient fits in one chunk -- always agreed,
    even before the fix).
    """
    events = _events(24)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _model(len(vocab), len(concepts))

    model_rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=chunk_size,
        device="cpu",
    )
    event_rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    for alert in ALERT_EVENTS:
        model_times = {
            (r.subject_id, r.visit_id, round(r.time_hours, 6))
            for r in model_rows[alert.name]
        }
        event_times = {
            (r.subject_id, r.visit_id, round(r.time_hours, 6))
            for r in event_rows[alert.name]
        }
        assert model_times == event_times, alert.name


# ---------------------------------------------------------------------------
# backbone="transformer": packed-path landmark selection
# ---------------------------------------------------------------------------


def _transformer_model(
    vocab_size: int, num_concepts: int
) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TransformerBackbone(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=2,
            num_heads=4,
            padding_idx=0,
        ),
        vocab_size=vocab_size,
        num_concepts=num_concepts,
        embedding_dim=4,
        padding_idx=0,
    )


@pytest.mark.parametrize(
    "num_lanes,max_context",
    [(4, 200), (2, 30)],
    ids=["one_call_several_patients_per_row", "many_calls_one_patient_per_row"],
)
def test_packed_landmark_rows_match_index_rows_from_events_exactly(
    num_lanes: int, max_context: int
) -> None:
    """The load-bearing proof: two landmark selections must agree exactly.

    (collect_model_scores' packed path, _index_rows_from_events'
    model-free ground truth) only get to coexist once shown to agree, not
    assumed to. Both parametrizations keep max_context above every
    patient's own length (~25-26 events here), so nothing is truncated --
    only truncation is allowed to shrink the landmark set below the
    ground truth, tested separately below. ``num_lanes=2, max_context=30``
    fits only one patient per row, forcing many next_chunk() calls for 24
    subjects: landmark_state is reset every call in the packed path (see
    collect_model_scores' docstring), so this is the case that would
    reveal it if that reset were wrong, the packed-path analogue of the
    lane-path test above's chunk_size=16 case.
    """
    events = _events(24)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _transformer_model(len(vocab), len(concepts))

    model_rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=num_lanes,
        device="cpu",
        backbone="transformer",
        max_context=max_context,
    )
    event_rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    for alert in ALERT_EVENTS:
        model_times = {
            (r.subject_id, r.visit_id, round(r.time_hours, 6))
            for r in model_rows[alert.name]
        }
        event_times = {
            (r.subject_id, r.visit_id, round(r.time_hours, 6))
            for r in event_rows[alert.name]
        }
        assert model_times == event_times, alert.name


def test_verify_packed_landmark_rows_passes_on_agreeing_rows() -> None:
    events = _events(24)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _transformer_model(len(vocab), len(concepts))

    truncation_boundaries: Dict[int, float] = {}
    model_rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        device="cpu",
        backbone="transformer",
        max_context=200,
        truncation_boundaries_out=truncation_boundaries,
    )

    problems = verify_packed_landmark_rows(
        model_rows,
        binned,
        ALERT_EVENTS,
        landmark_hours=4.0,
        truncation_boundaries=truncation_boundaries,
    )

    assert problems == []


def test_verify_packed_landmark_rows_catches_a_real_disagreement() -> None:
    """Not a no-op check: feeding it a genuinely wrong row set must fail."""
    events = _events(8)
    binned = add_value_tokens(events)
    wrong_rows = {alert.name: [IndexRow(9999, 9999, 1.0)] for alert in ALERT_EVENTS}

    problems = verify_packed_landmark_rows(
        wrong_rows,
        binned,
        ALERT_EVENTS,
        landmark_hours=4.0,
        truncation_boundaries={},
    )

    assert len(problems) == len(ALERT_EVENTS)
    assert all("disagree" in p for p in problems)


def test_verify_packed_landmark_rows_tail_aware_all_three_arms() -> None:
    """The review's exact scenario: one truncated subject, one not, all three arms.

    Subject 1: 40 hourly events (hours 0..39) -- truncated to the most
    recent max_context=15 (hours 25..39) by PackedContextSampler.
    Subject 2: 8 hourly events (hours 0..7) -- comfortably whole, never
    truncated. A real run of collect_model_scores/verify_packed_
    landmark_rows over this must find no problems (subject 2 exact,
    subject 1 correctly shrunk); then two hand-corrupted variants of the
    same row set must each be caught, and a third (legitimate
    before-boundary shrinkage) must not be.
    """
    rows_raw: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for h in range(40):
        rows_raw.append((1, "LAB//220045//bpm", T0 + timedelta(hours=h), 80.0, 1001))
    for h in range(8):
        rows_raw.append((2, "LAB//220045//bpm", T0 + timedelta(hours=h), 80.0, 1002))
    events = pl.DataFrame(
        rows_raw,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _transformer_model(len(vocab), len(concepts))

    truncation_boundaries: Dict[int, float] = {}
    model_rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        device="cpu",
        backbone="transformer",
        max_context=15,
        truncation_boundaries_out=truncation_boundaries,
    )

    # Sanity: subject 1 truncated, subject 2 whole.
    is_tail_by_subject = {
        r.subject_id: r.is_tail for rs in model_rows.values() for r in rs
    }
    assert is_tail_by_subject[1] is True
    assert is_tail_by_subject[2] is False

    assert truncation_boundaries == {1: 25.0}  # captured from the sampler,
    # independently of model_rows -- this is what each arm below must use,
    # unchanged, even when model_rows is corrupted: re-deriving it from
    # model_rows (as an earlier version of this fix did) let arm 2's own
    # corruption shift the boundary along with the row it deleted, hiding
    # the very bug the arm exists to catch.
    boundary = truncation_boundaries[1]

    # Arm 0: the real, uncorrupted output has no problems at all -- a
    # correctly shrunk tail is not itself a disagreement.
    assert (
        verify_packed_landmark_rows(
            model_rows,
            binned,
            ALERT_EVENTS,
            landmark_hours=4.0,
            truncation_boundaries=truncation_boundaries,
        )
        == []
    )

    # Arm 1 (tail, invented row): a landmark ground truth has no record of
    # at all, not explained by truncation -- always a bug.
    with_invented = {
        name: [*rs, IndexRow(1, 1001, 999.0, is_tail=True)]
        for name, rs in model_rows.items()
    }
    problems = verify_packed_landmark_rows(
        with_invented,
        binned,
        ALERT_EVENTS,
        landmark_hours=4.0,
        truncation_boundaries=truncation_boundaries,
    )
    assert problems
    assert any("no record" in p for p in problems)

    # Arm 2 (tail, dropped a row at/after the boundary): should have been
    # kept (it's inside the window PackedContextSampler actually kept),
    # missing it is a real bug, not truncation working as intended. The
    # boundary passed in is the one captured above, untouched by this
    # corruption -- proving the check no longer explains the drop away.
    missing_at_boundary = {
        name: [r for r in rs if not (r.subject_id == 1 and r.time_hours == boundary)]
        for name, rs in model_rows.items()
    }
    problems = verify_packed_landmark_rows(
        missing_at_boundary,
        binned,
        ALERT_EVENTS,
        landmark_hours=4.0,
        truncation_boundaries=truncation_boundaries,
    )
    assert problems
    assert any("missing entirely from the packed path" in p for p in problems)

    # Arm 3 (tail, "missing" a row that was never in the kept window at
    # all): legitimate truncation shrinkage, must NOT be flagged. Ground
    # truth has a subject-1 landmark well before the boundary (e.g. hour
    # 0); the real packed output never had it, and that is correct.
    ground_truth = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    assert any(
        r.subject_id == 1 and r.time_hours < boundary
        for r in ground_truth["vasopressor_start"]
    )  # confirms the fixture actually exercises this arm, not vacuously


def test_collect_model_scores_marks_truncated_subjects_is_tail() -> None:
    """A patient longer than max_context is truncated and its rows flagged."""
    events = _events(8)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _transformer_model(len(vocab), len(concepts))

    # Every subject has ~25-26 events; max_context=10 truncates all of them.
    rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        device="cpu",
        backbone="transformer",
        max_context=10,
    )

    assert any(r.is_tail for rs in rows.values() for r in rs)
    # every row belongs to a truncated subject at this max_context, so
    # none should be False.
    assert all(r.is_tail for rs in rows.values() for r in rs)


def test_collect_model_scores_hybrid_backbone_never_marks_is_tail() -> None:
    events = _events(8)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _model(len(vocab), len(concepts))

    rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=4,
        device="cpu",
    )

    assert all(not r.is_tail for rs in rows.values() for r in rs)


def test_baseline_features_shape_and_content() -> None:
    events = _events(4)
    binned = add_value_tokens(events)
    rows = [IndexRow(2, 1002, 13.0), IndexRow(1, 1001, 1.0)]
    x = baseline_features(binned, rows)
    assert x.shape[0] == 2
    # hours since visit start
    assert x[0, 0] == 13.0 and x[1, 0] == 1.0
    # subject 2 at hour 13 has HR HIGH as latest bin (ordinal 1)
    prefixes_start = 2
    assert 1.0 in x[0, prefixes_start:]
    assert not np.isnan(x[0, -6:]).any()  # counts present


def test_baseline_features_ignores_a_code_with_an_unrecognized_bin_suffix() -> None:
    """A code::SUFFIX where SUFFIX isn't LOW/NORMAL/HIGH/CRITICAL must not crash or bin.

    add_value_tokens never produces this today (every clinical-range label
    is one of the four), but a hand-edited or future-format code could --
    _SubjectHistory must skip it rather than silently substituting some
    default ordinal.
    """
    events = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [T0],
            "code": ["LAB//220045//bpm::WEIRD"],
            "numeric_value": [130.0],
            "hadm_id": [1001],
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
    )
    x = baseline_features(events, [IndexRow(1, 1001, 0.0)])
    prefixes_start = 2
    # the malformed bin was never recorded against the heart-rate prefix,
    # so its ordinal feature stays unset (NaN), not some arbitrary default
    assert np.isnan(x[0, prefixes_start:-6]).all()


def test_baseline_features_leaves_a_historyless_subject_all_nan() -> None:
    """An index row for a subject absent from events_binned entirely."""
    events = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [T0],
            "code": ["LAB//220045//bpm"],
            "numeric_value": [80.0],
            "hadm_id": [1001],
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
    )
    x = baseline_features(events, [IndexRow(999, 1, 5.0)])
    assert x.shape == (1, x.shape[1])
    assert np.isnan(x[0]).all()


def test_baseline_features_computes_age_when_birth_and_origin_both_known() -> None:
    """The age column is only ever filled when both birth and origin resolve."""
    events = pl.DataFrame(
        {
            "subject_id": [5, 5],
            "time": [datetime(1970, 1, 1), datetime(2020, 1, 1)],
            "code": [BIRTH_CODE, "LAB//220045//bpm"],
            "numeric_value": [None, 80.0],
            "hadm_id": [None, 2005],
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
    )
    x = baseline_features(events, [IndexRow(5, 2005, 0.0)])
    expected_age_years = (
        (datetime(2020, 1, 1) - datetime(1970, 1, 1)).total_seconds() / 3600.0
    ) / (24 * 365.25)
    assert x[0, 1] == pytest.approx(expected_age_years, abs=1e-6)


def test_subject_scoped_event_ignores_visit() -> None:
    times = EventTimes(
        onset={(7, -1): 30.0}, censor={(7, -1): 40.0}, subject_scoped=True
    )
    assert outcome_at_horizon(IndexRow(7, 999, 25.0), times, 8.0) == 1
    assert outcome_at_horizon(IndexRow(7, 999, 10.0), times, 8.0) == 0
    assert AlertEvent(
        "death", code_prefix="MEDS_DEATH", subject_scoped=True
    ).subject_scoped


def test_hazard_scorer_reports_probability_metrics() -> None:
    events = _events(16)
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=len(concepts),
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=[a.name for a in ALERT_EVENTS],
    )
    times = all_event_times(events, ALERT_EVENTS, "mimic_iv")
    rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        ALERT_EVENTS,
        visit_start=_visit_starts(events),
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
        horizons=(8.0, 24.0),
    )
    assert all(
        "hazard@8h" in r.scores and "hazard@24h" in r.scores for r in rows["death"]
    )
    results = score_alerts(rows, times, horizons=(8.0, 24.0))
    hazard = [r for r in results if r.scorer == "hazard"]
    assert hazard, "hazard scorer should be reported"
    for r in hazard:
        assert r.brier is not None and r.calibration
        assert 0.0 <= r.auroc <= 1.0
    # each hazard row scores its own horizon only: no 8h probability at 24h
    assert {(r.event, r.horizon_hours) for r in hazard} <= {
        (e, h) for e in rows for h in (8.0, 24.0)
    }


def test_icu_admission_prefix_excludes_admission_measurements() -> None:
    icu = next(a for a in ALERT_EVENTS if a.name == "icu_admission")
    assert icu.code_prefix is not None
    assert "ICU_ADMISSION//MICU".startswith(icu.code_prefix)
    assert "ICU_ADMISSION////admit".startswith(icu.code_prefix)
    assert not "ICU_ADMISSION_WEIGHT".startswith(icu.code_prefix)
    assert not "ICU_ADMISSION_HEIGHT".startswith(icu.code_prefix)


def test_strong_baseline_fits_tunes_and_records_metadata() -> None:
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="strong", tune=True
    )
    model = baselines[("vasopressor_start", 8.0)]
    assert model.feature_set == "strong"
    assert model.n_features == len(feature_names())
    assert model.params["n_rounds"] >= 1
    feats = features_for_events(binned, rows, feature_set="strong")
    assert feats["vasopressor_start"].shape[1] == len(feature_names())
    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
    )
    gbm = next(
        r
        for r in results
        if r.scorer == "baseline_gbm" and r.event == "vasopressor_start"
    )
    assert gbm.auroc is not None and gbm.auroc > 0.75
    assert gbm.baseline_feature_set == "strong"
    assert gbm.baseline_n_features == len(feature_names())
    assert gbm.baseline_params and "learning_rate" in gbm.baseline_params


def test_fit_baselines_skips_an_alert_with_no_index_rows() -> None:
    """An alert with an empty row list must be skipped, not fit on nothing.

    _index_rows_from_events never actually produces this (every alert
    shares the same row list), but fit_baselines's own signature accepts
    any per-alert row list, so a caller building train_rows another way
    can legitimately hit this -- the guard is exercised directly rather
    than only through that one caller.
    """
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    rows = {**rows, "vasopressor_start": []}

    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", tune=False
    )

    assert not any(name == "vasopressor_start" for name, _ in baselines)
    # icu_admission (planted for 1/4 of subjects) still fits normally
    assert any(name == "icu_admission" for name, _ in baselines)


def test_unknown_feature_set_is_rejected() -> None:
    events = _events(n_subjects=4)
    binned = add_value_tokens(events, None, source="mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    with pytest.raises(ValueError):
        features_for_events(binned, rows, feature_set="bogus")


def test_features_for_events_returns_empty_dict_when_no_rows_at_all() -> None:
    """No index rows for any event (an empty landmark set) must not crash."""
    events = _events(n_subjects=4)
    binned = add_value_tokens(events, None, source="mimic_iv")
    assert features_for_events(binned, {}, feature_set="basic") == {}
    assert (
        features_for_events(binned, {"vasopressor_start": []}, feature_set="basic")
        == {}
    )


def test_index_row_table_has_scores_outcomes_and_gbm_columns() -> None:
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    for r in rows["vasopressor_start"]:
        r.scores["next_mass"] = 0.5
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="strong", tune=False
    )
    feats = features_for_events(binned, rows, feature_set="strong")
    names = feature_names()
    table = index_row_table(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
        context_columns={
            k: v[:, [names.index("hours_into_visit")]] for k, v in feats.items()
        },
        context_names=["hours_into_visit"],
    )
    assert {"event", "subject_id", "visit_id", "time_hours", "y@8h"} <= set(
        table.columns
    )
    vaso = table.filter(pl.col("event") == "vasopressor_start")
    assert vaso["next_mass"].drop_nulls().len() == vaso.height
    assert (
        "gbm@8h" in table.columns and vaso["gbm@8h"].drop_nulls().len() == vaso.height
    )
    assert "ctx.hours_into_visit" in table.columns
    assert set(vaso["y@8h"].drop_nulls().unique().to_list()) <= {0.0, 1.0}
    assert vaso["y@8h"].null_count() > 0  # censored / not-at-risk rows are null


class _FakeClassifier:
    """Duck-typed classifier: only the ``classes_`` attribute matters here."""

    def __init__(self, classes: List[int]) -> None:
        self.classes_ = np.array(classes)


def test_positive_class_proba_picks_the_column_for_label_1() -> None:
    # Shared by EBMBaselineModel/TabICLBaselineModel (review finding 15,
    # deduplicating what used to be two identical copies of this logic).
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert _positive_class_proba(_FakeClassifier([0, 1]), proba).tolist() == [0.1, 0.8]


def test_positive_class_proba_handles_reordered_classes() -> None:
    # classes_ = [1, 0] (label 1 in column 0) -- must not assume column 1.
    proba = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert _positive_class_proba(_FakeClassifier([1, 0]), proba).tolist() == [0.1, 0.8]


def test_positive_class_proba_falls_back_to_column_1_when_label_1_absent() -> None:
    # Only label 0 was ever observed (a real, if rare, single-class fold) --
    # the documented, if imperfect, fallback: still try column 1.
    proba = np.array([[1.0, 0.0]])
    assert _positive_class_proba(_FakeClassifier([0]), proba).tolist() == [0.0]


def test_load_index_row_table_logs_current_version_when_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    table = _stamp_landmark_protocol_version(pl.DataFrame({"event": ["death"]}))
    assert table["landmark_protocol_version"].to_list() == [LANDMARK_PROTOCOL_VERSION]
    path = tmp_path / "rows.parquet"
    table.write_parquet(path)

    with caplog.at_level("WARNING"):
        loaded = load_index_row_table(path)

    assert loaded["landmark_protocol_version"].to_list() == [LANDMARK_PROTOCOL_VERSION]
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_load_index_row_table_warns_on_a_pre_column_v1_dump(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A dump written before this column existed at all -- must be treated
    # as protocol v1, and must warn (not raise) since it's still a
    # perfectly valid, internally consistent comparison set on its own.
    path = tmp_path / "rows.parquet"
    pl.DataFrame({"event": ["death"]}).write_parquet(path)

    with caplog.at_level("WARNING"):
        loaded = load_index_row_table(path)

    assert "landmark_protocol_version" not in loaded.columns
    assert any(
        "landmark_protocol_version=1" in r.message
        and str(LANDMARK_PROTOCOL_VERSION) in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


def test_load_index_row_table_warns_on_mixed_versions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "rows.parquet"
    pl.DataFrame(
        {"event": ["death", "death"], "landmark_protocol_version": [1, 2]}
    ).write_parquet(path)

    with caplog.at_level("WARNING"):
        load_index_row_table(path)

    assert any(
        "mixed landmark_protocol_version" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


def _write_dump(
    path: Path, rows: Dict[str, List[IndexRow]], times, horizons=(8.0,)
) -> None:
    index_row_table(rows, times, horizons=horizons).write_parquet(path)


def test_verify_rows_match_dump_passes_on_an_exact_match(tmp_path: Path) -> None:
    rows = {
        "death": [
            IndexRow(subject_id=1, visit_id=10, time_hours=0.0),
            IndexRow(subject_id=2, visit_id=20, time_hours=4.0),
        ]
    }
    times = {
        "death": EventTimes(
            onset={(1, 10): 4.0}, censor={(2, 20): 30.0}, subject_scoped=False
        )
    }
    path = tmp_path / "rows.parquet"
    _write_dump(path, rows, times)

    verify_rows_match_dump(rows, times, path, horizons=(8.0,))  # no raise


def test_verify_rows_match_dump_raises_on_a_missing_row(tmp_path: Path) -> None:
    rows = {
        "death": [
            IndexRow(subject_id=1, visit_id=10, time_hours=0.0),
            IndexRow(subject_id=2, visit_id=20, time_hours=4.0),
        ]
    }
    times = {
        "death": EventTimes(
            onset={(1, 10): 4.0}, censor={(2, 20): 30.0}, subject_scoped=False
        )
    }
    path = tmp_path / "rows.parquet"
    _write_dump(path, rows, times)

    extra_rows = {
        "death": rows["death"] + [IndexRow(subject_id=3, visit_id=30, time_hours=1.0)]
    }
    extra_times = {
        "death": EventTimes(
            onset={(1, 10): 4.0},
            censor={(2, 20): 30.0, (3, 30): 30.0},
            subject_scoped=False,
        )
    }
    with pytest.raises(AssertionError, match="reconstructed rows do not match"):
        verify_rows_match_dump(extra_rows, extra_times, path, horizons=(8.0,))


def test_verify_rows_match_dump_raises_on_a_label_disagreement(tmp_path: Path) -> None:
    rows = {"death": [IndexRow(subject_id=1, visit_id=10, time_hours=0.0)]}
    times = {"death": EventTimes(onset={(1, 10): 4.0}, censor={}, subject_scoped=False)}
    path = tmp_path / "rows.parquet"
    _write_dump(path, rows, times)  # dump has y@8h=1 (onset at 4h, within 8h)

    # Same row keys, but reconstructed with different onset -> y@8h=0 instead of 1.
    mismatched_times = {
        "death": EventTimes(onset={}, censor={(1, 10): 30.0}, subject_scoped=False)
    }
    with pytest.raises(AssertionError, match="y@h disagrees"):
        verify_rows_match_dump(rows, mismatched_times, path, horizons=(8.0,))


def test_verify_rows_match_dump_ignores_a_landmark_protocol_version_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Row/label identity is orthogonal to protocol version -- a version
    # mismatch still warns (via load_index_row_table) but must not raise,
    # as long as the rows and labels genuinely agree.
    rows = {"death": [IndexRow(subject_id=1, visit_id=10, time_hours=0.0)]}
    times = {"death": EventTimes(onset={(1, 10): 4.0}, censor={}, subject_scoped=False)}
    path = tmp_path / "rows.parquet"
    table = index_row_table(rows, times, horizons=(8.0,)).with_columns(
        pl.lit(1).alias("landmark_protocol_version")
    )
    table.write_parquet(path)

    with caplog.at_level("WARNING"):
        verify_rows_match_dump(rows, times, path, horizons=(8.0,))  # no raise

    assert any(
        "landmark_protocol_version=1" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


def test_tuning_survives_a_column_missing_only_inside_the_training_fold() -> None:
    rng = np.random.default_rng(0)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    # column 3 is observed only for subjects that land in the validation fold
    # (replicating _tune_gbm's seeded group shuffle), so the training fold
    # sees an all-NaN column that the full fit set does not
    shuffled = np.unique(groups)
    np.random.default_rng(0).shuffle(shuffled)
    val_groups = shuffled[: max(1, int(round(0.1 * len(shuffled))))]
    x[:, 3] = np.nan
    x[np.isin(groups, val_groups), 3] = 1.0
    params, n_rounds = _tune_gbm(x, y, groups, seed=0)
    assert n_rounds >= 1 and "learning_rate" in params


def test_tune_gbm_subsamples_when_over_the_row_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows beyond GBM_TUNE_MAX_ROWS are subsampled before tuning, not fit whole."""
    monkeypatch.setattr(alerts_module, "GBM_TUNE_MAX_ROWS", 200)
    rng = np.random.default_rng(0)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    params, n_rounds = _tune_gbm(x, y, groups, seed=0)
    assert n_rounds >= 1 and "learning_rate" in params


def test_tune_gbm_returns_default_params_when_validation_split_is_degenerate() -> None:
    """One subject group can't be split into disjoint train/val folds.

    n_val = max(1, round(0.1 * n_groups)) with a single group means every
    row's group is the validation group -- is_val.all() must fire and fall
    back to the documented default rather than fitting on an empty or
    single-class fold.
    """
    x = np.ones((20, 4), dtype=np.float32)
    y = np.array([0, 1] * 10)
    groups = np.zeros(20, dtype=int)  # one subject, all 20 rows

    params, n_rounds = _tune_gbm(x, y, groups, seed=0)

    assert params == dict(alerts_module.GBM_GRID[0])
    assert n_rounds == 200


def test_sparse_columns_are_filled_at_fit_and_predict() -> None:
    x = np.full((1000, 3), np.nan, dtype=np.float32)
    x[:, 0] = 1.0
    x[:GBM_MIN_OBSERVED, 1] = 2.0  # exactly the minimum: kept
    x[: GBM_MIN_OBSERVED - 1, 2] = 3.0  # one short: filled
    assert sparse_columns(x).tolist() == [False, False, True]
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="strong", tune=False
    )
    model = baselines[("vasopressor_start", 8.0)]
    # the synthetic record never has most of the panel: those columns are filled
    assert model.fill_columns.sum() > 0
    feats = features_for_events(binned, rows, feature_set="strong")
    p = model.predict_proba(feats["vasopressor_start"])
    assert np.isfinite(p).all()


def _write_event_shards(
    shard_dir: Path, n_shards: int, subjects_per_shard: int
) -> None:
    """Write the planted-deterioration signal from ``_events``, split across shards.

    Subjects are assigned to shards in contiguous blocks (never split
    across files), matching the real pipeline's invariant that a subject's
    full record lives in one shard.
    """
    shard_dir.mkdir(parents=True)
    sid = 0
    for k in range(n_shards):
        rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
        for _ in range(subjects_per_shard):
            sid += 1
            hadm = 1000 + sid
            for h in range(24):
                hr = 80.0
                if sid % 2 == 0 and h >= 12:
                    hr = 130.0
                rows.append(
                    (sid, "LAB//220045//bpm", T0 + timedelta(hours=h), hr, hadm)
                )
            if sid % 2 == 0:
                rows.append(
                    (
                        sid,
                        "MEDICATION//norepinephrine//Administered",
                        T0 + timedelta(hours=14),
                        None,
                        hadm,
                    )
                )
            if sid % 4 == 0:
                rows.append(
                    (sid, "ICU_ADMISSION//MICU", T0 + timedelta(hours=6), None, hadm)
                )
        pl.DataFrame(
            rows,
            schema={
                "subject_id": pl.Int64,
                "code": pl.Utf8,
                "time": pl.Datetime,
                "numeric_value": pl.Float32,
                "hadm_id": pl.Int64,
            },
            orient="row",
        ).write_parquet(shard_dir / f"{k}.parquet")


def test_fit_baselines_streaming_event_times_match_in_memory(tmp_path: Path) -> None:
    shard_dir = tmp_path / "train"
    _write_event_shards(shard_dir, n_shards=3, subjects_per_shard=20)
    whole = load_meds_shards(shard_dir)
    ref_times = all_event_times(whole, ALERT_EVENTS, "mimic_iv")

    paths = shard_paths(shard_dir)
    prepare = make_preparer(
        normalize_medications=False, history_recap=False, source="mimic_iv"
    )
    streamed_times: Dict[str, EventTimes] = {}
    for path in paths:
        raw = prepare(load_meds_shard(path))
        merge_event_times(
            streamed_times, all_event_times(raw, ALERT_EVENTS, "mimic_iv")
        )

    assert streamed_times.keys() == ref_times.keys()
    for name, times in ref_times.items():
        assert streamed_times[name].onset == times.onset
        assert streamed_times[name].censor == times.censor
        assert streamed_times[name].subject_scoped == times.subject_scoped


@pytest.mark.parametrize("feature_set", ["basic", "strong"])
def test_fit_baselines_streaming_matches_in_memory(
    tmp_path: Path, feature_set: str
) -> None:
    """Streaming and in-memory baseline fitting agree on the rows fit.

    Both fit a comparably discriminating model on the same planted
    signal. Row order can differ between the two paths (the in-memory path groups
    over one concatenated frame, the streaming path over each shard
    separately), so this does not require bit-identical models -- only
    that they see the same landmark rows and both learn the planted
    vasopressor signal. See ``test_shard_stream.py`` for the analogous
    training-corpus comparison, where the deterministic parts (counts,
    labels, event times) are checked for exact equality and the
    non-deterministic parts (fit results) are not.
    """
    shard_dir = tmp_path / "train"
    _write_event_shards(shard_dir, n_shards=3, subjects_per_shard=20)
    whole = load_meds_shards(shard_dir)
    binned = add_value_tokens(whole, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    ref_rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    ref_baselines = fit_baselines(
        binned, ref_rows, times, horizons=(8.0,), feature_set=feature_set, tune=False
    )

    paths = shard_paths(shard_dir)
    prepare = make_preparer(
        normalize_medications=False, history_recap=False, source="mimic_iv"
    )
    streamed_baselines = fit_baselines_streaming(
        paths,
        prepare,
        None,
        alerts=ALERT_EVENTS,
        horizons=(8.0,),
        feature_set=feature_set,
        tune=False,
    )

    # same (event, horizon) pairs cleared the fit threshold in both paths
    assert set(streamed_baselines) == set(ref_baselines)

    # same landmark rows, regardless of which order each path visited them in
    ref_key_set = {
        (r.subject_id, r.visit_id, r.time_hours) for r in ref_rows["vasopressor_start"]
    }
    streamed_row_count = 0
    for path in paths:
        raw = prepare(load_meds_shard(path))
        shard_binned = add_value_tokens(raw, None, source="mimic_iv")
        shard_rows = _index_rows_from_events(
            shard_binned, ALERT_EVENTS, landmark_hours=4.0
        )["vasopressor_start"]
        streamed_row_count += len(shard_rows)
        assert {(r.subject_id, r.visit_id, r.time_hours) for r in shard_rows} <= (
            ref_key_set
        )
    assert streamed_row_count == len(ref_key_set)

    # the streamed model learned the same planted signal as the in-memory one
    model = streamed_baselines[("vasopressor_start", 8.0)]
    feats = features_for_events(binned, ref_rows, feature_set=feature_set)
    results = score_alerts(
        ref_rows,
        times,
        horizons=(8.0,),
        baselines={("vasopressor_start", 8.0): model},
        baseline_features_by_event=feats,
    )
    gbm = next(
        r
        for r in results
        if r.scorer == "baseline_gbm" and r.event == "vasopressor_start"
    )
    assert gbm.auroc is not None and gbm.auroc > 0.75


def test_fit_baselines_streaming_empty_shards_returns_no_models(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    # a single subject, no alert events ever trigger
    pl.DataFrame(
        [(1, "LAB//220045//bpm", T0, 80.0, 1001)],
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    ).write_parquet(shard_dir / "0.parquet")
    paths = shard_paths(shard_dir)
    prepare = make_preparer(
        normalize_medications=False, history_recap=False, source="mimic_iv"
    )
    models = fit_baselines_streaming(
        paths,
        prepare,
        None,
        alerts=ALERT_EVENTS,
        horizons=(8.0,),
        feature_set="basic",
        tune=False,
    )
    assert models == {}


def _no_visit_shard(path: Path) -> None:
    """One subject, every event hadm_id-null: zero landmark rows for any alert.

    _index_rows_from_events requires hadm_id.is_not_null(), so a shard
    like this (a subject-scoped-only record, or a static-facts-only
    fragment) contributes nothing to any alert's landmark set.
    """
    pl.DataFrame(
        [(1, "LAB//220045//bpm", T0, 80.0, None)],
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    ).write_parquet(path)


def test_fit_baselines_streaming_skips_a_shard_with_no_landmark_rows(
    tmp_path: Path,
) -> None:
    """A shard contributing zero landmark rows must not break the rest of the run."""
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _no_visit_shard(shard_dir / "0.parquet")  # contributes nothing
    _events(n_subjects=20).write_parquet(shard_dir / "1.parquet")

    paths = shard_paths(shard_dir)
    prepare = make_preparer(
        normalize_medications=False, history_recap=False, source="mimic_iv"
    )
    models = fit_baselines_streaming(
        paths,
        prepare,
        None,
        alerts=ALERT_EVENTS,
        horizons=(8.0,),
        feature_set="basic",
        tune=False,
    )
    # the real shard's planted signal still fits despite the empty one
    assert ("vasopressor_start", 8.0) in models


def test_fit_baselines_streaming_returns_no_models_when_every_shard_is_empty(
    tmp_path: Path,
) -> None:
    """Every shard contributing zero landmark rows: return early, not crash."""
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _no_visit_shard(shard_dir / "0.parquet")

    paths = shard_paths(shard_dir)
    prepare = make_preparer(
        normalize_medications=False, history_recap=False, source="mimic_iv"
    )
    models = fit_baselines_streaming(
        paths,
        prepare,
        None,
        alerts=ALERT_EVENTS,
        horizons=(8.0,),
        feature_set="basic",
        tune=False,
    )
    assert models == {}


def test_fit_baseline_grid_caps_rows_before_the_final_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At full corpus scale GBM_FIT_MAX_ROWS keeps the refit bounded.

    Row-capping is exercised here at a size a unit test can afford by
    lowering the cap itself rather than growing the data to it; the cap
    logic is data-size-agnostic (it only compares a row count to a
    constant), so this exercises the same code path full-scale runs hit.
    """
    monkeypatch.setattr(alerts_module, "GBM_FIT_MAX_ROWS", 20)
    events = _events(n_subjects=40)
    binned = add_value_tokens(events, None, source="mimic_iv")
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    assert len(rows["vasopressor_start"]) > 20  # more rows than the lowered cap
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", tune=False
    )
    model = baselines[("vasopressor_start", 8.0)]
    feats = features_for_events(binned, rows, feature_set="basic")
    p = model.predict_proba(feats["vasopressor_start"])
    assert np.isfinite(p).all()


# ---------------------------------------------------------------------------
# extra_baselines: a second, named baseline family alongside the GBM
# (odyssey.inference.tabicl_baseline is the real use case; a fake stand-in
# here keeps these tests independent of the optional tabicl dependency)
# ---------------------------------------------------------------------------


class _FakeBaselineModel:
    """Duck-typed like BaselineModel/TabICLBaselineModel: a fixed, checkable score."""

    def __init__(self, value: float, n_features: int) -> None:
        self.value = value
        self.feature_set = "fake"
        self.n_features = n_features
        self.params: Dict[str, float] = {"fixed_value": value}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value)


def test_score_alerts_extra_baselines_scores_alongside_the_gbm() -> None:
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    baselines = fit_baselines(
        binned, rows, times, horizons=(8.0,), feature_set="basic", tune=False
    )
    feats = features_for_events(binned, rows, feature_set="basic")

    fake_models = {
        (name, 8.0): _FakeBaselineModel(0.5, feats[name].shape[1])
        for name in rows
        if rows[name]
    }
    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        baselines=baselines,
        baseline_features_by_event=feats,
        extra_baselines={"baseline_fake": (fake_models, feats)},
    )
    by = {(r.event, r.scorer): r for r in results}
    # both the existing GBM path and the new named family are present,
    # neither disturbing the other
    assert ("vasopressor_start", "baseline_gbm") in by
    fake = by[("vasopressor_start", "baseline_fake")]
    # a constant 0.5 prediction is uninformative: AUROC collapses to chance
    assert fake.auroc == pytest.approx(0.5)
    assert fake.baseline_feature_set == "fake"
    assert fake.brier is not None and fake.calibration


def test_score_alerts_extra_baselines_missing_cell_is_skipped_not_erroring() -> None:
    """A scorer_name entry with no model for this (event, horizon) is silently absent.

    Mirrors how the built-in GBM path already behaves when an event/horizon
    combination has no fitted model (too few rows, single outcome class).
    """
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    results = score_alerts(
        rows,
        times,
        horizons=(8.0,),
        extra_baselines={"baseline_fake": ({}, {})},
    )
    assert not any(r.scorer == "baseline_fake" for r in results)


def test_index_row_table_extra_baselines_adds_a_named_column() -> None:
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)
    feats = features_for_events(binned, rows, feature_set="basic")
    fake_models = {
        (name, 8.0): _FakeBaselineModel(0.5, feats[name].shape[1])
        for name in rows
        if rows[name]
    }
    table = index_row_table(
        rows,
        times,
        horizons=(8.0,),
        extra_baselines={"tabicl": (fake_models, feats)},
    )
    assert "tabicl@8h" in table.columns
    vaso = table.filter(pl.col("event") == "vasopressor_start")
    assert (vaso["tabicl@8h"] == 0.5).all()


def test_index_row_table_extra_baselines_missing_cell_is_skipped_not_erroring() -> None:
    """Mirrors score_alerts' identical guard: an (event, horizon) with no model."""
    events = _events(24)
    binned = add_value_tokens(events)
    times = all_event_times(binned, ALERT_EVENTS, "mimic_iv")
    rows = _index_rows_from_events(binned, ALERT_EVENTS, landmark_hours=4.0)

    table = index_row_table(
        rows,
        times,
        horizons=(8.0,),
        extra_baselines={"tabicl": ({}, {})},
    )
    assert "tabicl@8h" not in table.columns


def test_score_alerts_skips_an_alert_with_no_index_rows() -> None:
    results = score_alerts(
        {"vasopressor_start": []},
        {"vasopressor_start": EventTimes(onset={}, censor={}, subject_scoped=False)},
    )
    assert results == []


def test_index_row_table_skips_an_alert_with_no_index_rows() -> None:
    table = index_row_table(
        {"vasopressor_start": []},
        {"vasopressor_start": EventTimes(onset={}, censor={}, subject_scoped=False)},
    )
    assert table.is_empty()


def test_index_row_table_returns_empty_frame_when_every_alert_is_empty() -> None:
    """No alert contributes any rows at all: the concat-nothing path."""
    table = index_row_table(
        {"vasopressor_start": [], "death": []},
        {
            "vasopressor_start": EventTimes(onset={}, censor={}, subject_scoped=False),
            "death": EventTimes(onset={}, censor={}, subject_scoped=True),
        },
    )
    assert table.is_empty()
    assert table.columns == ["event"]


def test_score_alerts_skips_a_scorer_that_is_all_nan_for_the_kept_rows() -> None:
    """A scorer present on some rows but NaN/missing on every at-risk one.

    scorer_names is the union across all of an event's rows, so a scorer
    only ever populated on a not-at-risk (already-happened) row still gets
    a scoring attempt for every horizon -- ok.sum()==0 among the KEPT rows
    must skip it, not divide by zero or crash roc_auc_score on an empty
    array. Outcomes are engineered to still give the two classes score_alerts
    needs to get past its own y.min()==y.max() guard first, so this
    isolates the scorer-level guard specifically.
    """
    times = EventTimes(
        onset={(1, 1): 5.0}, censor={(1, 1): 100.0, (2, 1): 8.0}, subject_scoped=False
    )
    rows = [
        # kept, positive (onset 5.0 falls in [0, 8])
        IndexRow(1, 1, 0.0, scores={"hazard@8h": 0.9}),
        # not at risk (onset already happened by time_hours=20) -- dropped
        # from `keep` entirely, but still contributes "concept" to
        # scorer_names for this event
        IndexRow(1, 1, 20.0, scores={"concept": 0.7}),
        # kept, negative (no onset, follow-up exactly reaches the horizon)
        IndexRow(2, 1, 0.0, scores={"hazard@8h": 0.2}),
    ]
    results = score_alerts(
        {"vasopressor_start": rows}, {"vasopressor_start": times}, horizons=(8.0,)
    )
    # "concept" was never populated on either kept (at-risk) row above
    assert not any(r.scorer == "concept" for r in results)
    assert any(r.scorer == "hazard" for r in results)


# ---------------------------------------------------------------------------
# _main: append-only-by-default guard for alerts.json / alerts_rows.parquet
# ---------------------------------------------------------------------------


def _boom(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("must not evaluate before the overwrite guard fires")


def test_main_refuses_to_overwrite_an_existing_alerts_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guard must fire before evaluate_alerts runs, not just before the write.

    Real incident: an automatic eval chain silently overwrote a finished
    run's own alerts.json/alerts_rows.parquet at their standard output
    paths (2026-08-22) -- the row-level dump was unrecoverable.
    """
    existing = tmp_path / "alerts.json"
    existing.write_text("[]")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            str(existing),
        ],
    )
    monkeypatch.setattr(alerts_module, "evaluate_alerts", _boom)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        alerts_module._main()


def test_main_refuses_to_overwrite_an_existing_dump_rows_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing_rows = tmp_path / "alerts_rows.parquet"
    existing_rows.write_bytes(b"")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            str(tmp_path / "alerts.json"),
            "--dump-rows",
            str(existing_rows),
        ],
    )
    monkeypatch.setattr(alerts_module, "evaluate_alerts", _boom)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        alerts_module._main()


def test_main_allows_a_fresh_output_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative case: no existing files at all must not trip the guard."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            str(tmp_path / "alerts.json"),
        ],
    )
    called = {"evaluate": False}

    def _fake_evaluate(*_args: object, **_kwargs: object) -> list:
        called["evaluate"] = True
        return []

    monkeypatch.setattr(alerts_module, "evaluate_alerts", _fake_evaluate)

    alerts_module._main()

    assert called["evaluate"] is True
    assert (tmp_path / "alerts.json").read_text() == "[]"


# ---------------------------------------------------------------------------
# index_mode="visit_end" (discharge-anchored rows; 30-day readmission)
# ---------------------------------------------------------------------------


def _readmission_events(n_subjects: int = 12) -> pl.DataFrame:
    """Two visits per subject; odd subjects are readmitted within 30 days."""
    rows: List[Tuple[int, str, datetime, Optional[float], Optional[int]]] = []
    for sid in range(1, n_subjects + 1):
        first, second = 2000 + sid, 3000 + sid
        for h in range(10):
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), 80.0, first))
        rows.append(
            (sid, "HOSPITAL_DISCHARGE//HOME", T0 + timedelta(hours=10), None, first)
        )
        gap_days = 10 if sid % 2 else 60
        start = T0 + timedelta(hours=10, days=gap_days)
        rows.append((sid, "HOSPITAL_ADMISSION//EMERGENCY", start, None, second))
        for h in range(1, 6):
            rows.append(
                (sid, "LAB//220045//bpm", start + timedelta(hours=h), 80.0, second)
            )
        # record continues long after the second visit so nothing is censored
        rows.append((sid, "LAB//220045//bpm", start + timedelta(days=90), 80.0, None))
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


@pytest.mark.parametrize("chunk_size", [8, 200])
def test_visit_end_rows_agree_between_model_and_event_paths(chunk_size: int) -> None:
    events = _readmission_events()
    binned = add_value_tokens(events)
    vocab = _vocab(binned)
    concepts = concepts_for_source("mimic_iv")
    model = _model(len(vocab), len(concepts))
    readmit = [a for a in ALERT_EVENTS_V2 if a.next_visit]
    model_rows = collect_model_scores(
        model,
        binned,
        vocab,
        [c.name for c in concepts],
        readmit,
        visit_start=_visit_starts(events),
        num_lanes=2,
        chunk_size=chunk_size,
        device="cpu",
        index_mode="visit_end",
    )
    truth = _index_rows_from_events(
        binned, readmit, landmark_hours=4.0, index_mode="visit_end"
    )
    got = {
        (r.subject_id, r.visit_id, round(r.time_hours, 6))
        for r in model_rows["readmission_30d"]
    }
    want = {
        (r.subject_id, r.visit_id, round(r.time_hours, 6))
        for r in truth["readmission_30d"]
    }
    assert got == want
    assert len(want) == 24  # two visits per subject, one row each, at the last event
    assert (
        verify_packed_landmark_rows(
            model_rows,
            binned,
            readmit,
            landmark_hours=4.0,
            truncation_boundaries={},
            index_mode="visit_end",
        )
        == []
    )
    # outcomes at 30 days: first visits of odd subjects are readmitted (10d),
    # even subjects are not (60d); second visits have no next admission.
    times = all_event_times(events, readmit, "mimic_iv", task_set="v2")[
        "readmission_30d"
    ]
    by_visit = {(r.subject_id, r.visit_id): r for r in truth["readmission_30d"]}
    for sid in range(1, 13):
        first = by_visit[(sid, 2000 + sid)]
        assert outcome_at_horizon(first, times, 720.0) == (1 if sid % 2 else 0)
        second = by_visit[(sid, 3000 + sid)]
        assert outcome_at_horizon(second, times, 720.0) == 0  # record runs 90d on


def test_index_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="index_mode"):
        _index_rows_from_events(
            add_value_tokens(_events(2)),
            ALERT_EVENTS,
            landmark_hours=4.0,
            index_mode="x",
        )
