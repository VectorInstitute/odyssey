"""DemoService end to end on the synthetic cohort with a tiny CPU model."""

import time
from pathlib import Path

import polars as pl
import pytest

import apps.clinician_demo.forecast as forecast_module
import odyssey.inference.counterfactual as counterfactual_module
from apps.clinician_demo.service import (
    DISCLAIMERS,
    BadRequestError,
    DemoService,
    NotFoundError,
    concept_label,
    event_infos,
)
from tests.apps.clinician_demo.conftest import CHUNK, ServiceFactory


def _wait_job(service: DemoService, job_id: str) -> object:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = service.job(job_id)
        if job.status in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("evidence job did not finish")


def test_meta_shows_five_events_and_only_the_active_mode_disclaimer(
    make_service: ServiceFactory,
) -> None:
    meta = make_service().meta()
    assert [e.name for e in meta.events] == [
        "icu_admission",
        "vasopressor_start",
        "acute_kidney_injury",
        "sepsis3",
        "death",
    ]
    assert "readmission_30d" not in {e.name for e in meta.events}
    assert meta.data_mode == "open" and not meta.searchable
    assert "open" in meta.disclaimers and "credentialed" not in meta.disclaimers
    assert meta.chunk_size == CHUNK and meta.horizons == [8.0, 24.0, 72.0]
    assert set(DISCLAIMERS) - {"credentialed", "open"} <= set(meta.disclaimers)


def test_open_mode_gallery_is_built_by_tracing_and_lists_every_admission(
    make_service: ServiceFactory,
) -> None:
    gallery = make_service(threshold=0.0).gallery()
    kinds = [s.kind for s in gallery.sections]
    assert kinds == ["early_warning", "quiet", "miss", "false_alarm", "other"]
    everyone = gallery.sections[-1]
    assert {(c.subject_id, c.visit_id) for c in everyone.cases} == {
        (1, 10),
        (1, 11),
        (2, 20),
    }
    warnings = gallery.sections[0].cases
    assert any(
        c.event == "icu_admission" and c.lead_hours == pytest.approx(20.0)
        for c in warnings
    )
    assert all("≈" not in c.headline for c in warnings)  # exact onsets in open mode
    seen = {c.subject_id: c.seen_in_training for c in everyone.cases}
    assert seen == {1: True, 2: False}


def test_patient_and_trace_endpoints(make_service: ServiceFactory) -> None:
    service = make_service()
    patient = service.patient(1)
    assert [v.visit_id for v in patient.visits] == [10, 11] and patient.seen_in_training
    view = service.trace(1, 10)
    assert view.visit.visit_id == 10 and view.onsets["icu_admission"] == pytest.approx(
        20.0
    )
    assert view.banked == []  # open mode never reads banked rows
    with pytest.raises(NotFoundError, match="patient 99"):
        service.patient(99)
    with pytest.raises(NotFoundError, match="no visit 12"):
        service.trace(1, 12)


def test_traces_are_cached_per_patient(
    make_service: ServiceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(prepare=False)
    calls: list[int] = []
    real = forecast_module.trace_patient

    def counting(ctx, raw):  # type: ignore[no-untyped-def]
        calls.append(1)
        return real(ctx, raw)

    monkeypatch.setattr("apps.clinician_demo.service.trace_patient", counting)
    service.trace(1, 10)
    service.trace(1, 11)
    service.whatif(1, 10, {"t_hours": 5, "edits": [{"preset": "sbp"}]})
    assert len(calls) == 1


def test_the_runs_chunk_size_reaches_every_model_call(
    make_service: ServiceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped chunk_size would silently fall back to 256 and change every number."""
    seen: list[int] = []
    real = forecast_module.stream_patient

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs["chunk_size"])
        return real(*args, **kwargs)

    monkeypatch.setattr(forecast_module, "stream_patient", spy)
    monkeypatch.setattr(counterfactual_module, "stream_patient", spy)
    service = make_service(prepare=False)
    service.trace(1, 10)
    service.whatif(1, 10, {"t_hours": 12, "edits": [{"preset": "sbp", "value": 70}]})
    job = service.evidence(
        1,
        10,
        {
            "t_hours": 12,
            "target": {"kind": "event", "name": "death"},
            "lookback_hours": 2,
        },
    )
    assert _wait_job(service, job.job_id).status == "done"  # type: ignore[attr-defined]
    assert len(seen) > 3 and set(seen) == {CHUNK}


def test_whatif_snaps_to_a_shown_moment_and_validates_input(
    make_service: ServiceFactory,
) -> None:
    service = make_service(prepare=False)
    result = service.whatif(
        1, 10, {"t_hours": 12.4, "edits": [{"preset": "sbp", "value": 70}]}
    )
    assert result.t_hours == pytest.approx(12.0) and result.rows_edited == 6
    early = service.whatif(1, 10, {"t_hours": -5, "edits": [{"preset": "sbp"}]})
    assert early.t_hours == pytest.approx(0.0)  # before the visit: its first moment
    for body, message in [
        ({"edits": [{"preset": "sbp"}]}, "t_hours"),
        ({"t_hours": "x", "edits": [{"preset": "sbp"}]}, "t_hours"),
        ({"t_hours": 1, "edits": []}, "non-empty"),
        ({"t_hours": 1, "edits": [{"preset": "zzz"}]}, "unknown preset"),
    ]:
        with pytest.raises(BadRequestError, match=message):
            service.whatif(1, 10, body)


def test_evidence_for_events_and_concepts_and_its_cache(
    make_service: ServiceFactory,
) -> None:
    service = make_service(prepare=False)
    body = {
        "t_hours": 12,
        "target": {"kind": "event", "name": "icu_admission", "horizon_hours": 8},
        "lookback_hours": 3,
    }
    job = service.evidence(1, 10, body)
    done = _wait_job(service, job.job_id)
    assert done.status == "done" and done.result  # type: ignore[attr-defined]
    assert all(
        abs(a.delta) >= abs(b.delta) for a, b in zip(done.result, done.result[1:])
    )  # type: ignore[attr-defined]
    assert service.evidence(1, 10, body).job_id == job.job_id  # cached
    concept = service.evidence(
        1, 10, {"t_hours": 12, "target": {"kind": "concept", "name": "hypotension"}}
    )
    assert _wait_job(service, concept.job_id).status == "done"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"t_hours": 1, "target": "death"}, "target must be an object"),
        (
            {"t_hours": 1, "target": {"kind": "event", "name": "readmission_30d"}},
            "unknown event",
        ),
        (
            {
                "t_hours": 1,
                "target": {"kind": "event", "name": "death", "horizon_hours": 12},
            },
            "horizon",
        ),
        (
            {"t_hours": 1, "target": {"kind": "concept", "name": "nope"}},
            "unknown concept",
        ),
        ({"t_hours": 1, "target": {"kind": "code", "name": "x"}}, "target.kind"),
        (
            {
                "t_hours": 1,
                "target": {"kind": "concept", "name": "fever"},
                "lookback_hours": 0.5,
            },
            "lookback_hours",
        ),
        (
            {
                "t_hours": 1,
                "target": {"kind": "concept", "name": "fever"},
                "lookback_hours": 100,
            },
            "lookback_hours",
        ),
    ],
)
def test_evidence_rejects_bad_targets(
    make_service: ServiceFactory, body: dict[str, object], message: str
) -> None:
    with pytest.raises(BadRequestError, match=message):
        make_service(prepare=False).evidence(1, 10, body)


def test_unknown_job_is_not_found(make_service: ServiceFactory) -> None:
    with pytest.raises(NotFoundError):
        make_service(prepare=False).job("ev404")


def _banked_rows(path: Path) -> Path:
    frame = pl.DataFrame(
        {
            "subject_id": [1.0, 1.0, 1.0, 2.0],
            "visit_id": [10.0, 10.0, 10.0, 20.0],
            "time_hours": [4.0, 8.0, 12.0, 4.0],
            "event": ["icu_admission", "icu_admission", "icu_admission", "death"],
            "hazard@24h": [0.1, 0.3, 0.5, 0.0],
            "gbm@24h": [0.2, 0.2, 0.2, 0.0],
            "y@24h": [0.0, 1.0, 1.0, 0.0],
        }
    )
    frame.write_parquet(path)
    return path


def test_credentialed_mode_builds_the_gallery_from_banked_rows_without_the_model(
    make_service: ServiceFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_model(*_a: object, **_k: object) -> None:
        raise AssertionError("credentialed gallery must not trace anyone")

    monkeypatch.setattr("apps.clinician_demo.service.trace_patient", _no_model)
    service = make_service(
        data_mode="credentialed",
        threshold=0.25,
        banked_rows_path=_banked_rows(tmp_path / "rows.parquet"),
    )
    assert service.meta().searchable
    kinds = [s.kind for s in service.gallery().sections]
    assert kinds == ["early_warning", "quiet", "miss", "false_alarm"]
    warnings = service.gallery().sections[0].cases
    assert warnings == [] or all("≈" in c.headline for c in warnings)


def test_credentialed_trace_carries_the_banked_overlay(
    make_service: ServiceFactory, tmp_path: Path
) -> None:
    service = make_service(
        data_mode="credentialed",
        banked_rows_path=_banked_rows(tmp_path / "rows.parquet"),
    )
    view = service.trace(1, 10)
    assert [(b.t, b.hazard_24h, b.gbm_24h) for b in view.banked] == [
        (4.0, 0.1, 0.2),
        (8.0, 0.3, 0.2),
        (12.0, 0.5, 0.2),
    ]
    assert service.trace(2, 20).banked[0].event == "death"


def test_self_check_reports_the_invariants(
    make_service: ServiceFactory, tmp_path: Path
) -> None:
    service = make_service(
        data_mode="credentialed",
        banked_rows_path=_banked_rows(tmp_path / "rows.parquet"),
        threshold=0.0,
    )
    service._gallery = service.prepare_gallery()  # noqa: SLF001
    report = service.self_check()
    assert report["chunk_size"] == CHUNK and report["subjects"] == 2
    assert "readmission_30d" not in report["events"]
    if "trace_seconds" in report:
        assert report["unknown_token_share"] == 0.0
        assert report["whatif_rows_edited"] >= 0


def test_self_check_on_an_empty_gallery_says_so(make_service: ServiceFactory) -> None:
    service = make_service(prepare=False)
    assert "gallery is empty" in service.self_check()["warning"]


def test_warm_up_traces_gallery_patients_in_the_background(
    make_service: ServiceFactory,
) -> None:
    service = make_service(threshold=0.0)
    service.warm_up().join(timeout=60)
    assert len(service._traces) == 2  # noqa: SLF001


def test_event_infos_fall_back_for_unknown_events() -> None:
    (info,) = event_infos(("something_new",))
    assert (info.display, info.short, info.definition) == (
        "something new",
        "something_new",
        "",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("qsofa", "qSOFA"),
        ("sirs", "SIRS"),
        ("sepsis3", "Sepsis-3"),
        ("acute_kidney_injury", "AKI (any stage)"),
        ("aki_stage_3", "AKI stage 3"),
        ("shock", "Sustained hypotension (MAP)"),  # legacy slot name
        ("anemia", "Severe anemia"),
        ("hypoxia", "Hypoxia"),
    ],
)
def test_concept_labels_read_like_clinical_writing(name: str, expected: str) -> None:
    assert concept_label(name) == expected
