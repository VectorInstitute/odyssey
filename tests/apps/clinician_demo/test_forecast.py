"""Forecast: the model trace and the pure helpers that shape the replay."""

from datetime import timedelta

import pytest

from apps.clinician_demo.codebook import Codebook
from apps.clinician_demo.forecast import (
    RunContext,
    alert_detail,
    alert_episode_start,
    bundle_ends,
    callout_text,
    displayed_alerts,
    onsets_for,
    trace_patient,
    visit_stats_from_trace,
    visit_view,
)
from apps.clinician_demo.patient_store import PatientStore
from apps.clinician_demo.schemas import OperatingPoint
from odyssey.data.alert_events import alert_events_for
from odyssey.inference.counterfactual import score_record_at
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from tests.apps.clinician_demo.conftest import CHUNK, HEAD_NAMES, ICU, T0


# -- pure helpers --------------------------------------------------------------


def test_callout_text_covers_every_outcome() -> None:
    assert callout_text("ICU admission", cross=4.0, alert_start=6.0, onset=18.0) == (
        "ICU admission began at hour 18. The alert had been on since hour 6: "
        "12 h of warning."
    )
    assert callout_text("ICU admission", cross=17.6, alert_start=17.6, onset=18.0) == (
        "ICU admission began at hour 18. The alert came on just before it."
    )
    assert "but was off again by then" in callout_text(
        "ICU admission", cross=4.0, alert_start=None, onset=18.0
    )
    assert callout_text("Death", cross=None, alert_start=None, onset=5.0).endswith(
        "never reached the alert line: a miss."
    )
    assert "a false alarm" in callout_text(
        "Death", cross=5.0, alert_start=None, onset=None
    )
    assert callout_text("Death", cross=None, alert_start=None, onset=None) == (
        "No Death during this stay, and the risk stayed below the alert line."
    )


def test_alert_detail_states_the_line_and_its_measured_quality() -> None:
    point = OperatingPoint("death", 24.0, 0.013, 0.05, 0.78, 0.08, 0.004, 100, 0.8, 0.1)
    detail = alert_detail("Death", point)
    assert detail.startswith("Alert line: 1.3% risk within 24 h.")
    assert "on for 5% of moments" in detail and "covers 78%" in detail
    assert "8% of the moments it is on are followed by Death" in detail
    assert alert_detail("Death", None) == ""
    undefined = OperatingPoint("e", 24.0, 0.2, 0.05, None, None, 0.0, 10, None, None)
    assert "covers n/a" in alert_detail("X", undefined)


@pytest.mark.parametrize(
    ("values", "end", "expected"),
    [
        ([0.1, 0.5, 0.6, 0.7], None, 1.0),  # on from t=1 to the end
        ([0.5, 0.1, 0.6, 0.7], None, 2.0),  # an earlier episode does not count
        ([0.5, 0.6, 0.1], None, None),  # off at the end: no live alert
        ([0.1, 0.5, 0.6, 0.1], 3.0, 1.0),  # only moments before `end` count
        ([0.5, None, 0.6], None, 0.0),  # masked moments are skipped, not "off"
        ([0.2, 0.2], None, 0.0),  # equality counts as on
        ([], None, None),
        ([0.9, 0.9], 0.0, None),  # nothing before `end`
    ],
)
def test_alert_episode_start(
    values: list[float | None], end: float | None, expected: float | None
) -> None:
    times = [float(i) for i in range(len(values))]
    assert alert_episode_start(times, values, 0.2, end) == expected


def test_displayed_alerts_drop_readmission_and_follow_the_display_order() -> None:
    alerts, index = displayed_alerts(
        alert_events_for("v3", source="mimic_iv"),
        HEAD_NAMES,
        ("death", "icu_admission"),
    )
    names = [a.name for a in alerts]
    assert "readmission_30d" not in names
    assert names[:2] == ["death", "icu_admission"]
    assert [HEAD_NAMES[i] for i in index] == names
    only, _ = displayed_alerts(alert_events_for("v3", source="mimic_iv"), ["death"], ())
    assert [a.name for a in only] == ["death"]


# -- the model trace -----------------------------------------------------------


def test_trace_rows_align_with_the_record_and_hold_every_output(
    ctx: RunContext, store: PatientStore
) -> None:
    trace = trace_patient(ctx, store.raw_events(1))
    n = trace.n_positions
    assert trace.risk.shape == (n, len(ctx.events), 3)
    assert trace.concepts.shape == (n, len(ctx.concept_names))
    assert trace.top_ids.shape == (n, 5) and trace.top_probs.shape == (n, 5)
    assert (
        len(trace.tokens)
        == len(trace.times)
        == len(trace.timestamps)
        == len(trace.visit_ids)
        == n
    )
    assert trace.n_static == 1 and trace.tokens[0] == "GENDER//M"
    assert trace.n_unknown == 0
    assert (trace.risk >= 0).all() and (trace.risk <= 1).all()
    # cumulative risk: 8 h <= 24 h <= 72 h at every position
    assert (trace.risk[..., 0] <= trace.risk[..., 1] + 1e-6).all()
    assert (trace.risk[..., 1] <= trace.risk[..., 2] + 1e-6).all()
    assert (
        trace.values[
            trace.tokens.index(
                next(t for t in trace.tokens if t.startswith("LAB//220179"))
            )
        ]
        == 123.5
    )


def test_trace_at_a_bundle_end_equals_score_record_at(
    ctx: RunContext, store: PatientStore
) -> None:
    """The replay and what-if must read the same forecast for the same moment."""
    raw = store.raw_events(1)
    trace = trace_patient(ctx, raw)
    for pos in [bundle_ends(trace.times)[k] for k in (3, 17, 40)]:
        readout = score_record_at(
            ctx.model,
            ctx.vocab,
            ctx.binner,
            raw,
            index_time=trace.timestamps[pos],
            concept_names=ctx.concept_names,
            chunk_size=CHUNK,
        )
        assert readout.position == pos
        for j, event in enumerate(ctx.events):
            for h, key in enumerate(("8h", "24h", "72h")):
                assert trace.risk[pos, j, h] == pytest.approx(
                    readout.event_risk[event][key], abs=1e-5
                )
        for c, name in enumerate(ctx.concept_names):
            assert trace.concepts[pos, c] == pytest.approx(
                readout.concept_probs[name], abs=1e-5
            )


def test_unknown_tokens_are_counted(ctx: RunContext, store: PatientStore) -> None:
    raw = store.raw_events(2)
    shrunk = ctx.vocab.__class__({"[PAD]": 0, "[UNK]": 1})
    small_ctx = RunContext(**{**ctx.__dict__, "vocab": shrunk})
    trace = trace_patient(small_ctx, raw)
    assert trace.n_unknown == trace.n_positions


def test_trace_refuses_models_and_records_it_cannot_show(
    ctx: RunContext, store: PatientStore
) -> None:
    no_heads = ConceptBottleneckSequenceModel(
        TinyGRUBackbone(vocab_size=len(ctx.vocab), hidden_size=8),
        vocab_size=len(ctx.vocab),
        num_concepts=29,
        embedding_dim=4,
    )
    with pytest.raises(ValueError, match="hazard heads"):
        trace_patient(
            RunContext(**{**ctx.__dict__, "model": no_heads}), store.raw_events(1)
        )
    static_only = store.raw_events(1).filter(store.raw_events(1)["time"].is_null())
    with pytest.raises(ValueError, match="no timed events"):
        trace_patient(ctx, static_only)


# -- onsets and the visit view -------------------------------------------------


def test_onsets_are_visit_scoped_except_death(
    ctx: RunContext, store: PatientStore
) -> None:
    raw = store.raw_events(1)
    first = onsets_for(
        raw, ctx.alerts, source="mimic_iv", task_set="v3", subject_id=1, visit_id=10
    )
    second = onsets_for(
        raw, ctx.alerts, source="mimic_iv", task_set="v3", subject_id=1, visit_id=11
    )
    assert first["icu_admission"] == pytest.approx(20.0)
    assert second["icu_admission"] is None
    death_hours = 30 * 24 + 10.0
    assert first["death"] == pytest.approx(death_hours) and second[
        "death"
    ] == pytest.approx(death_hours)
    assert set(first) == set(ctx.events)


def _view(
    ctx: RunContext,
    store: PatientStore,
    visit_id: int,
    threshold: float,
    **kwargs: object,
):  # type: ignore[no-untyped-def]
    raw = store.raw_events(1)
    trace = trace_patient(ctx, raw)
    visit = next(v for v in store.visits(1) if v.visit_id == visit_id)
    onsets = onsets_for(
        raw,
        ctx.alerts,
        source="mimic_iv",
        task_set="v3",
        subject_id=1,
        visit_id=visit_id,
    )
    points = {
        e: OperatingPoint(e, 24.0, threshold, 0.05, 0.5, 0.2, 0.01, 10, None, None)
        for e in ctx.events
    }
    return visit_view(
        trace,
        visit,
        events=ctx.events,
        horizons=ctx.horizons,
        onsets=onsets,
        points=points,
        codebook=Codebook(),
        decode=ctx.vocab.decode,
        display={e: e for e in ctx.events},
        seen_in_training=True,
        **kwargs,  # type: ignore[arg-type]
    )


def test_visit_view_masks_after_onset_and_reports_the_crossing(
    ctx: RunContext, store: PatientStore
) -> None:
    view = _view(ctx, store, 10, threshold=0.0)
    assert view.times[0] == pytest.approx(0.0) and view.times == sorted(view.times)
    icu = view.risk["icu_admission"]["24h"]
    assert all(v is None for v, t in zip(icu, view.times) if t >= 20.0)
    assert all(v is not None for v, t in zip(icu, view.times) if t < 20.0)
    assert view.onsets["icu_admission"] == pytest.approx(20.0)
    assert view.onsets["death"] is None  # died in the NEXT admission
    alert = next(a for a in view.alerts if a.event == "icu_admission")
    assert alert.first_cross_hours == pytest.approx(0.0)  # threshold 0: crosses at once
    assert (
        alert.lead_hours == pytest.approx(20.0) and "20 h of warning" in alert.callout
    )
    assert [m.label for m in view.markers if m.category == "care"][:2] == [
        "Emergency admission, from emergency room",
        "ICU admission: Medical Intensive Care Unit (MICU)",
    ]
    assert all(e.category != "demographic" for e in view.timeline)
    assert len(view.concepts) == len(view.times) and len(view.concepts[0]) == len(
        ctx.concept_names
    )
    assert all(len(nxt) == 5 for nxt in view.top_next)
    assert view.seen_in_training


def test_a_line_never_reached_is_reported_as_a_miss(
    ctx: RunContext, store: PatientStore
) -> None:
    view = _view(ctx, store, 10, threshold=2.0)
    alert = next(a for a in view.alerts if a.event == "icu_admission")
    assert alert.first_cross_hours is None and alert.lead_hours is None
    assert alert.callout.endswith("never reached the alert line: a miss.")
    assert alert.alert_start_hours is None and "Alert line:" in alert.detail


def test_visit_view_respects_the_point_cap_but_keeps_the_onset(
    ctx: RunContext, store: PatientStore
) -> None:
    view = _view(ctx, store, 10, threshold=2.0, max_points=5)
    assert len(view.times) <= 12  # 5 bins plus kept onset/marker moments
    assert any(19.0 <= t < 20.0 for t in view.times), (
        "the moment just before ICU must survive"
    )


def test_visit_stats_from_trace_marks_positives_and_skips_unthresholded_events(
    ctx: RunContext, store: PatientStore
) -> None:
    raw = store.raw_events(1)
    trace = trace_patient(ctx, raw)
    visits = store.visits(1)
    onsets = {
        v.visit_id: onsets_for(
            raw,
            ctx.alerts,
            source="mimic_iv",
            task_set="v3",
            subject_id=1,
            visit_id=v.visit_id,
        )
        for v in visits
    }
    stats = visit_stats_from_trace(
        trace,
        visits,
        onsets,
        events=ctx.events,
        horizons=ctx.horizons,
        thresholds={"icu_admission": 0.0, "death": 2.0},
    )
    assert set(stats["event"].to_list()) == {"icu_admission", "death"}
    icu10 = stats.filter(
        (stats["visit_id"] == 10) & (stats["event"] == "icu_admission")
    ).row(0, named=True)
    assert icu10["positive"] and icu10["end_hours"] == pytest.approx(20.0)
    assert icu10["first_cross_hours"] == pytest.approx(icu10["start_hours"])
    death11 = stats.filter((stats["visit_id"] == 11) & (stats["event"] == "death")).row(
        0, named=True
    )
    assert (
        death11["positive"] and death11["first_cross_hours"] is None
    )  # threshold 2.0: a miss
    empty = visit_stats_from_trace(
        trace, [], {}, events=ctx.events, horizons=ctx.horizons, thresholds={}
    )
    assert empty.height == 0 and empty.columns == stats.columns


def test_icu_at_the_first_moment_leaves_nothing_at_risk(
    ctx: RunContext, store: PatientStore
) -> None:
    raw = store.raw_events(1)
    trace = trace_patient(ctx, raw)
    visit = next(v for v in store.visits(1) if v.visit_id == 10)
    first_t = trace.times[trace.n_static]
    stats = visit_stats_from_trace(
        trace,
        [visit],
        {10: {"icu_admission": first_t}},
        events=ctx.events,
        horizons=ctx.horizons,
        thresholds={"icu_admission": 0.5},
    )
    assert stats.height == 0
    assert (
        T0 + timedelta(hours=20)
        in store.raw_events(1)
        .filter(store.raw_events(1)["code"] == ICU)["time"]
        .to_list()
    )
