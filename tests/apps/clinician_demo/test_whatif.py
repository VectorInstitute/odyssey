"""What-if: request validation, preset sanity, and the no-GPU short cut."""

from datetime import timedelta

import pytest

import apps.clinician_demo.whatif as whatif_module
from apps.clinician_demo.forecast import RunContext
from apps.clinician_demo.patient_store import PatientStore
from apps.clinician_demo.whatif import (
    MAX_EDITS,
    PRESETS,
    PRESETS_BY_ID,
    EditRequest,
    parse_edit_requests,
    run_whatif,
    to_value_edits,
    untouched_warnings,
)
from tests.apps.clinician_demo.conftest import T0


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_every_preset_resolves_and_is_unit_safe_on_mimic(preset) -> None:  # type: ignore[no-untyped-def]
    assert preset.min <= preset.value <= preset.max and preset.step > 0
    assert 1.0 <= preset.window_hours <= 72.0
    for value in (preset.min, preset.value, preset.max):
        for edit in to_value_edits(EditRequest(preset.id, value)):
            prefixes = edit.prefixes("mimic_iv")
            assert prefixes, f"{edit.signal} resolves to no MIMIC code"
            for prefix in prefixes:
                edit.value_for(
                    prefix, "mimic_iv"
                )  # must not raise on a unit-tagged prefix


def test_preset_ids_are_unique() -> None:
    assert len(PRESETS_BY_ID) == len(PRESETS)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "non-empty list"),
        ([], "non-empty list"),
        ({"preset": "sbp"}, "non-empty list"),
        ([{"preset": "sbp"}] * (MAX_EDITS + 1), "at most"),
        (["sbp"], "must be an object"),
        ([{"preset": "nope"}], "unknown preset"),
        ([{}], "unknown preset"),
        ([{"preset": "sbp", "value": "80"}], "must be a number"),
        ([{"preset": "sbp", "value": True}], "must be a number"),
        ([{"preset": "sbp", "value": None}], "must be a number"),
        ([{"preset": "sbp", "value": 49.9}], "outside"),
        ([{"preset": "sbp", "value": 180.1}], "outside"),
        ([{"preset": "sbp", "value": float("nan")}], "outside"),
        ([{"preset": "sbp"}, {"preset": "sbp", "value": 90}], "listed twice"),
    ],
)
def test_bad_requests_are_refused(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_edit_requests(payload)


def test_defaults_and_bounds_are_accepted() -> None:
    parsed = parse_edit_requests(
        [
            {"preset": "sbp"},
            {"preset": "lactate", "value": 0.25},
            {"preset": "creatinine", "value": 4},
        ]
    )
    assert parsed == [
        EditRequest("sbp", PRESETS_BY_ID["sbp"].value),
        EditRequest("lactate", 0.25),
        EditRequest("creatinine", 4.0),
    ]


def test_value_edit_carries_the_preset_mode_and_window() -> None:
    (edit,) = to_value_edits(EditRequest("lactate", 2.0))
    assert (edit.signal, edit.mode, edit.value, edit.window_hours) == (
        "lactate",
        "scale",
        2.0,
        12.0,
    )


def test_blood_pressure_presets_cover_cuff_and_arterial_line() -> None:
    signals = [e.signal for e in to_value_edits(EditRequest("sbp", 80.0))]
    assert signals == ["sbp_noninvasive", "sbp_arterial"]
    assert [e.signal for e in to_value_edits(EditRequest("map", 60.0))] == [
        "map_noninvasive",
        "map_arterial",
    ]


def test_untouched_edits_warn_without_touching_the_gpu(
    ctx: RunContext, store: PatientStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("no rows changed: the model must not run")

    monkeypatch.setattr(whatif_module, "counterfactual_forecast", _boom)
    result = run_whatif(
        ctx,
        store.raw_events(1),
        [
            EditRequest("lactate", 3.0),
            EditRequest("spo2", 85.0),
        ],  # never charted in the cohort
        index_time=T0 + timedelta(hours=10),
        t_hours=10.0,
    )
    assert result.rows_edited == 0 and len(result.warnings) == 2
    assert result.factual.risk == {} and result.delta.concepts == {}


def test_a_real_edit_moves_the_forecast_and_deltas_are_consistent(
    ctx: RunContext, store: PatientStore
) -> None:
    result = run_whatif(
        ctx,
        store.raw_events(1),
        [EditRequest("sbp", 60.0), EditRequest("lactate", 3.0)],
        index_time=T0 + timedelta(hours=12),
        t_hours=12.0,
    )
    assert result.rows_edited == 6  # SBP at hours 7..12
    assert len(result.warnings) == 1 and "Lactate" in result.warnings[0]
    assert set(result.factual.risk) == set(ctx.events)
    assert "readmission_30d" not in result.counterfactual.risk
    for event, by_h in result.delta.risk.items():
        for h, d in by_h.items():
            assert d == pytest.approx(
                result.counterfactual.risk[event][h] - result.factual.risk[event][h]
            )
    assert any(abs(d) > 0 for hs in result.delta.risk.values() for d in hs.values())


def test_untouched_warnings_count_rows_per_edit(store: PatientStore) -> None:
    total, warnings = untouched_warnings(
        store.raw_events(1),
        [EditRequest("heart_rate", 100.0)],
        index_time=T0 + timedelta(hours=3),
        source="mimic_iv",
    )
    assert total == 3 and warnings == []
