"""What-if: change recent readings in the record and re-score the forecast.

Built on :mod:`odyssey.inference.counterfactual`: each preset rewrites one
panel signal's numeric readings inside a window before the chosen moment
(set to a value, add, or scale), re-tokenizes with the run's own binner,
and streams the factual and edited records through the frozen model.

What it measures is how the model's CURRENT forecast responds to a
different record -- model sensitivity, not the effect of a treatment. An
edit can only change readings that exist: if no reading of the signal
falls in the window, nothing changes and the result says so instead of
spending GPU time.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from apps.clinician_demo.forecast import RunContext
from apps.clinician_demo.schemas import Readout, WhatIfPreset, WhatIfResult
from odyssey.inference.counterfactual import (
    ForecastReadout,
    ValueEdit,
    apply_value_edits,
    counterfactual_forecast,
)


MAX_EDITS = 3

PRESETS: tuple[WhatIfPreset, ...] = (
    WhatIfPreset(
        "sbp",
        "Systolic BP",
        "sbp_noninvasive",
        "set",
        80.0,
        50.0,
        180.0,
        5.0,
        "mmHg",
        6.0,
        "Set every systolic reading (cuff or arterial line) in the last 6 h to this value.",
    ),
    WhatIfPreset(
        "map",
        "Mean arterial pressure",
        "map_noninvasive",
        "set",
        60.0,
        40.0,
        120.0,
        5.0,
        "mmHg",
        6.0,
        "Set every MAP reading (cuff or arterial line) in the last 6 h to this value.",
    ),
    WhatIfPreset(
        "heart_rate",
        "Heart rate",
        "heart_rate",
        "set",
        130.0,
        30.0,
        190.0,
        5.0,
        "bpm",
        6.0,
        "Set every heart-rate reading in the last 6 h to this value.",
    ),
    WhatIfPreset(
        "resp_rate",
        "Respiratory rate",
        "resp_rate",
        "set",
        28.0,
        6.0,
        45.0,
        1.0,
        "/min",
        6.0,
        "Set every respiratory-rate reading in the last 6 h to this value.",
    ),
    WhatIfPreset(
        "spo2",
        "SpO2",
        "spo2",
        "set",
        86.0,
        70.0,
        100.0,
        1.0,
        "%",
        6.0,
        "Set every SpO2 reading in the last 6 h to this value.",
    ),
    WhatIfPreset(
        "lactate",
        "Lactate (multiply)",
        "lactate",
        "scale",
        3.0,
        0.25,
        5.0,
        0.25,
        "x",
        12.0,
        "Multiply every lactate result in the last 12 h by this factor.",
    ),
    WhatIfPreset(
        "creatinine",
        "Creatinine (add)",
        "creatinine",
        "add",
        1.0,
        -1.5,
        4.0,
        0.1,
        "mg/dL",
        24.0,
        "Add this amount to every creatinine result in the last 24 h.",
    ),
    WhatIfPreset(
        "potassium",
        "Potassium",
        "potassium",
        "set",
        6.5,
        2.0,
        8.0,
        0.1,
        "mEq/L",
        24.0,
        "Set every potassium result in the last 24 h to this value.",
    ),
    WhatIfPreset(
        "platelets",
        "Platelets",
        "platelets",
        "set",
        40.0,
        5.0,
        500.0,
        5.0,
        "K/uL",
        24.0,
        "Set every platelet count in the last 24 h to this value.",
    ),
)
PRESETS_BY_ID: dict[str, WhatIfPreset] = {p.id: p for p in PRESETS}
#: Presets that edit more than one panel signal: blood pressure is charted
#: by cuff on the ward and by arterial line in the ICU, and a clinician
#: asking "what if the BP were 80" means both.
EXTRA_SIGNALS: dict[str, tuple[str, ...]] = {
    "sbp": ("sbp_arterial",),
    "map": ("map_arterial",),
}


@dataclass(frozen=True)
class EditRequest:
    """One requested edit: a preset and the value chosen on its control."""

    preset_id: str
    value: float


def parse_edit_requests(payload: object) -> list[EditRequest]:
    """Validate the ``edits`` list of a what-if request body.

    Each item is ``{"preset": id, "value": number}`` (``value`` defaults to
    the preset's own). At most :data:`MAX_EDITS`, each preset at most once,
    every value inside its preset's bounds.

    Raises
    ------
    ValueError
        On any malformed, unknown, duplicate or out-of-range edit.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError("edits must be a non-empty list")
    if len(payload) > MAX_EDITS:
        raise ValueError(f"at most {MAX_EDITS} edits at once")
    out: list[EditRequest] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each edit must be an object")
        preset = PRESETS_BY_ID.get(str(item.get("preset")))
        if preset is None:
            raise ValueError(f"unknown preset {item.get('preset')!r}")
        raw = item.get("value", preset.value)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{preset.id}: value must be a number")
        value = float(raw)
        if not preset.min <= value <= preset.max:
            raise ValueError(
                f"{preset.id}: value {value:g} outside [{preset.min:g}, {preset.max:g}]"
            )
        if any(r.preset_id == preset.id for r in out):
            raise ValueError(f"{preset.id}: listed twice")
        out.append(EditRequest(preset.id, value))
    return out


def to_value_edits(request: EditRequest) -> list[ValueEdit]:
    """Return the counterfactual-module edits a request stands for."""
    preset = PRESETS_BY_ID[request.preset_id]
    return [
        ValueEdit(
            signal=signal,
            mode=preset.mode,  # type: ignore[arg-type]
            value=request.value,
            window_hours=preset.window_hours,
        )
        for signal in (preset.signal, *EXTRA_SIGNALS.get(preset.id, ()))
    ]


def readout(forecast: ForecastReadout, events: Sequence[str]) -> Readout:
    """Restrict a forecast to the displayed events."""
    return Readout(
        risk={
            e: dict(forecast.event_risk[e]) for e in events if e in forecast.event_risk
        },
        concepts=dict(forecast.concept_probs),
    )


def untouched_warnings(
    raw_events: pl.DataFrame,
    requests: Sequence[EditRequest],
    *,
    index_time: object,
    source: str,
) -> tuple[int, list[str]]:
    """Count touched rows and warn about each edit that touches none."""
    total = 0
    warnings: list[str] = []
    for request in requests:
        _, touched = apply_value_edits(
            raw_events, to_value_edits(request), index_time=index_time, source=source
        )
        total += touched
        if touched == 0:
            preset = PRESETS_BY_ID[request.preset_id]
            warnings.append(
                f"No {preset.label} readings in the {preset.window_hours:g} h before this "
                "moment, so that edit changed nothing."
            )
    return total, warnings


def run_whatif(
    ctx: RunContext,
    raw_events: pl.DataFrame,
    requests: Sequence[EditRequest],
    *,
    index_time: object,
    t_hours: float,
) -> WhatIfResult:
    """Compare the factual and edited forecast at ``index_time`` (a record time)."""
    touched, warnings = untouched_warnings(
        raw_events, requests, index_time=index_time, source=ctx.source
    )
    if touched == 0:
        empty = Readout(risk={}, concepts={})
        return WhatIfResult(t_hours, 0, warnings, empty, empty, empty)
    result = counterfactual_forecast(
        ctx.model,
        ctx.vocab,
        ctx.binner,
        raw_events,
        [edit for r in requests for edit in to_value_edits(r)],
        index_time=index_time,
        concept_names=ctx.concept_names,
        source=ctx.source,
        device=ctx.device,
        chunk_size=ctx.chunk_size,
    )
    factual = readout(result.factual, ctx.events)
    counterfactual = readout(result.counterfactual, ctx.events)
    delta = Readout(
        risk={
            e: {h: counterfactual.risk[e][h] - p for h, p in hs.items()}
            for e, hs in factual.risk.items()
        },
        concepts={
            c: counterfactual.concepts[c] - p for c, p in factual.concepts.items()
        },
    )
    return WhatIfResult(
        t_hours, result.rows_edited, warnings, factual, counterfactual, delta
    )


__all__ = [
    "MAX_EDITS",
    "PRESETS",
    "PRESETS_BY_ID",
    "EditRequest",
    "parse_edit_requests",
    "readout",
    "run_whatif",
    "EXTRA_SIGNALS",
    "to_value_edits",
    "untouched_warnings",
]
