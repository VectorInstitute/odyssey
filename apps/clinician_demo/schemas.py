"""JSON contracts between the demo's backend and its browser UI.

Every API response is one of these frozen dataclasses passed through
:func:`to_jsonable`, so the wire format is defined in exactly one place and
the static JS never has to guess at shapes. Times are hours since the
visit's first event unless a field says otherwise; probabilities are
floats in [0, 1]; ``None`` means "not available" (never NaN, which JSON
cannot carry).
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any


FLOAT_DIGITS = 5


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventInfo:
    """One forecast event the demo displays."""

    name: str
    display: str
    short: str
    definition: str


@dataclass(frozen=True)
class ConceptInfo:
    """One named concept of the bottleneck."""

    name: str
    display: str
    description: str
    readout_auroc: float | None


@dataclass(frozen=True)
class OperatingPoint:
    """The alert line for one (event, horizon), with its measured quality.

    Computed over the run's own at-risk landmark rows: flagging every
    moment whose risk is at or above ``threshold`` flags ``alert_rate`` of
    them, catching ``sensitivity`` of the moments followed by the event
    within the horizon, with ``ppv`` of flags being right. The tuned GBM
    flagging the same share of moments is reported alongside.
    """

    event: str
    horizon_hours: float
    threshold: float
    alert_rate: float
    """Measured share of at-risk moments flagged (ties can push it above target)."""
    sensitivity: float | None
    """``None`` when no at-risk moment was followed by the event."""
    ppv: float | None
    base_rate: float
    n_rows: int
    gbm_sensitivity: float | None
    gbm_ppv: float | None


@dataclass(frozen=True)
class Meta:
    """Static facts the UI needs once at start-up."""

    run_name: str
    checkpoint: str
    data_mode: str
    chunk_size: int
    horizons: list[float]
    events: list[EventInfo]
    concepts: list[ConceptInfo]
    operating_points: list[OperatingPoint]
    disclaimers: dict[str, str]
    searchable: bool
    """Whether patients can be looked up by id (off in open mode, where
    the gallery lists every patient anyway)."""


# ---------------------------------------------------------------------------
# Gallery and patient lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GalleryCase:
    """One curated visit in the gallery."""

    subject_id: int
    visit_id: int
    kind: str
    """``early_warning``, ``quiet``, ``miss``, ``false_alarm`` or ``other``."""
    event: str | None
    headline: str
    lead_hours: float | None
    los_hours: float
    seen_in_training: bool
    lead_approximate: bool = False
    """The lead is a lower bound from 4-hourly landmarks, not an exact onset."""


@dataclass(frozen=True)
class GallerySection:
    """A titled group of gallery cases with a one-line honest summary."""

    kind: str
    title: str
    summary: str
    cases: list[GalleryCase]


@dataclass(frozen=True)
class Gallery:
    """The gallery page."""

    sections: list[GallerySection]


@dataclass(frozen=True)
class VisitSummary:
    """One admission of a patient."""

    visit_id: int
    start_hours: float
    """Hours since the patient's first recorded event."""
    end_hours: float
    admission: str
    n_events: int


@dataclass(frozen=True)
class PatientSummary:
    """A patient's header facts and their admissions."""

    subject_id: int
    split: str | None
    seen_in_training: bool
    sex: str | None
    age_years: float | None
    visits: list[VisitSummary]


# ---------------------------------------------------------------------------
# Chart replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineEntry:
    """One recorded event, in plain language."""

    t: float
    category: str
    label: str
    value: str | None = None
    flag: str | None = None
    """``LOW``, ``HIGH``, ``CRITICAL`` or ``None`` (normal / not assessed)."""


@dataclass(frozen=True)
class NextEvent:
    """One of the model's most likely next events."""

    label: str
    probability: float


@dataclass(frozen=True)
class AlertCrossing:
    """When the risk first crossed its alert line before the event."""

    event: str
    horizon_hours: float
    threshold: float
    first_cross_hours: float | None
    onset_hours: float | None
    lead_hours: float | None
    callout: str
    alert_start_hours: float | None = None
    """Start of the alert episode still on when the event began (visit hours)."""
    detail: str = ""
    """How this alert line performs across held-out patients."""


@dataclass(frozen=True)
class BankedPoint:
    """One landmark row from the run's own evaluation (credentialed mode only)."""

    t: float
    event: str
    hazard_24h: float | None
    gbm_24h: float | None
    outcome_24h: float | None


@dataclass(frozen=True)
class VisitTrace:
    """Everything the replay view draws for one visit.

    ``times`` indexes every per-point series: ``risk[event][horizon_key][i]``,
    ``concepts[i][c]`` and ``top_next[i]`` all describe the moment
    ``times[i]`` (the end of a same-time bundle of events). Risk is ``None``
    at and after the event's onset, where the forecast has no meaning.
    """

    subject_id: int
    visit: VisitSummary
    seen_in_training: bool
    times: list[float]
    risk: dict[str, dict[str, list[float | None]]]
    concepts: list[list[float]]
    top_next: list[list[NextEvent]]
    timeline: list[TimelineEntry]
    markers: list[TimelineEntry]
    onsets: dict[str, float | None]
    alerts: list[AlertCrossing]
    banked: list[BankedPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# What-if, evidence, scorecard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhatIfPreset:
    """One what-if control the UI offers."""

    id: str
    label: str
    signal: str
    mode: str
    value: float
    min: float
    max: float
    step: float
    unit: str
    window_hours: float
    description: str


@dataclass(frozen=True)
class Readout:
    """The forecast at one moment: risk per event and horizon, concept beliefs."""

    risk: dict[str, dict[str, float]]
    concepts: dict[str, float]


@dataclass(frozen=True)
class WhatIfResult:
    """Factual vs edited forecast at one moment."""

    t_hours: float
    rows_edited: int
    warnings: list[str]
    factual: Readout
    counterfactual: Readout
    delta: Readout


@dataclass(frozen=True)
class EvidenceItem:
    """One recorded code and how much removing it moves the target."""

    code: str
    label: str
    n_rows: int
    baseline: float
    occluded: float
    delta: float


@dataclass(frozen=True)
class EvidenceJob:
    """A long-running evidence search, polled by the UI."""

    job_id: str
    status: str
    """``pending``, ``running``, ``done`` or ``error``."""
    done: int
    total: int
    target: str
    result: list[EvidenceItem]
    error: str | None
    note: str


@dataclass(frozen=True)
class CalibrationBin:
    """One decile of the calibration curve."""

    predicted: float
    observed: float
    n: int


@dataclass(frozen=True)
class ScoreCell:
    """How well one (event, horizon) is predicted, against the tuned GBM."""

    event: str
    horizon_hours: float
    n_at_risk: int
    n_positive: int
    base_rate: float
    hazard_auroc: float | None
    hazard_ci: list[float] | None
    gbm_auroc: float | None
    gbm_ci: list[float] | None
    delta: float | None
    delta_ci: list[float] | None
    separated: bool | None
    calibration: list[CalibrationBin]


@dataclass(frozen=True)
class Scorecard:
    """The model's report card, from the run's banked held-out evaluation."""

    headline: str
    cells: list[ScoreCell]
    concepts: list[ConceptInfo]
    notes: list[str]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _clean(value: Any) -> Any:  # noqa: ANN401 -- recursive JSON walk
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, FLOAT_DIGITS)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, type):
        raise TypeError(f"cannot serialize the class {value.__name__} to JSON")
    if dataclasses.is_dataclass(value):
        return {
            f.name: _clean(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    # numpy / torch scalars: unwrap to the Python number they hold
    item = getattr(value, "item", None)
    if callable(item):
        return _clean(item())
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def to_jsonable(value: Any) -> Any:  # noqa: ANN401 -- returns plain JSON types
    """Convert a schema object (or nested plain data) to JSON-safe builtins.

    Floats are rounded to :data:`FLOAT_DIGITS` places (the UI never shows
    more, and it shrinks traces several-fold); NaN and infinities become
    ``None``; dataclasses become dicts; tuples become lists.

    Raises
    ------
    TypeError
        For a value with no JSON form, rather than silently stringifying it.
    """
    return _clean(value)


__all__ = [
    "FLOAT_DIGITS",
    "AlertCrossing",
    "BankedPoint",
    "CalibrationBin",
    "ConceptInfo",
    "EventInfo",
    "EvidenceItem",
    "EvidenceJob",
    "Gallery",
    "GalleryCase",
    "GallerySection",
    "Meta",
    "NextEvent",
    "OperatingPoint",
    "PatientSummary",
    "Readout",
    "ScoreCell",
    "Scorecard",
    "TimelineEntry",
    "VisitSummary",
    "VisitTrace",
    "WhatIfPreset",
    "WhatIfResult",
    "to_jsonable",
]
