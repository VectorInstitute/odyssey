"""Clinical concept labels derived from MEDS events.

A registry of clinically-recognizable vital-sign/lab abnormalities,
derived from a MEDS event table to supervise
:class:`odyssey.models.concept_bottleneck.ConceptBottleneck`.
Itemids are MIMIC-IV's, matching the itemids used by the widely-cited
`MIT-LCP/mimic-code <https://github.com/MIT-LCP/mimic-code>`_ concept
queries; the MEDS ETL keys chartevents on ``LAB//{itemid}//`` and
labevents on ``LAB//RESULT//{itemid}//`` -- see
:mod:`odyssey.data.code_mapping` for the LOINC mapping that makes these
rules portable to other institutions.

Labels are per-subject (did this ever happen across the subject's full
extracted window), not per-timepoint -- extend once the patient sequence
windowing scheme calls for finer granularity. Four rule types now exist,
per research_journal/04_concept_pipeline.html decision (b):

- :class:`ConceptRule` -- a single instantaneous threshold crossing (v1's
  only rule type; still appropriate for concepts with no known
  over-triggering problem, e.g. a single very-high heart-rate reading is
  already clinically meaningful on its own).
- :class:`SustainedRule` -- a threshold crossing that must recur at least
  ``min_gap_hours`` apart, not just once. Fixes the exact failure mode v1
  had: ``tachypnea`` (RR > 20) triggered for 96.5% of subjects with
  respiratory-rate data in the real extraction, because one transient
  spike (e.g. during a single stressful procedure) was indistinguishable
  from true sustained tachypnea.
- :class:`BaselineRelativeRule` -- KDIGO-style: a value rose (or fell) by
  at least an absolute ``delta`` or a proportional ``ratio`` from an
  earlier reading of the same subject/code within ``window_hours``.
  Baseline is the subject's own earlier value, not a fixed population
  reference range -- AKI is specifically about deviation from a
  patient's own normal.
- :class:`DerivedGcsTotalRule` -- qSOFA/NEWS2-style: total Glasgow Coma
  Scale (eye + verbal + motor) below a threshold. MIMIC-IV charts GCS as
  three separately-timed components with no single "GCS total" itemid
  (unlike MIMIC-III's itemid 198 -- confirmed by reading
  ``meas_chartevents_main.csv`` directly, not assumed); this derives a
  total by pairing each component with its nearest other-component
  readings within ``max_component_gap_minutes`` (GCS components are
  typically charted together in practice) and summing.

:class:`CompositeConceptDefinition` combines several of the above into an
"N of M criteria" concept (SIRS/qSOFA-style), optionally nesting
:class:`AnyOf` where one criterion is itself satisfied by any of several
rules (e.g. SIRS's "abnormal temperature" criterion is satisfied by
either a too-high or too-low reading, but must only count once even if a
subject had both at different points in their stay).

Full SOFA and NEWS2 are deliberately NOT implemented here: both are
ordinal point-scale scores (each component contributes 0-4 or 0-3
points, summed to a total), not "N of M criteria met" like SIRS/qSOFA --
a materially different aggregation this module's rule framework doesn't
yet express. See research_journal/04_concept_pipeline.html, Section 08,
"still open".
"""

import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import polars as pl


# Which side of a threshold triggers a rule. The `at_or_*` variants make
# the threshold value itself qualify -- clinical criteria are precise
# about this (qSOFA is RR >= 22, KDIGO Stage 3 is creatinine >= 4.0),
# and vitals/labs are often charted as exact integers, so a strict
# inequality would silently miss real boundary readings.
Direction = Literal["above", "below", "at_or_above", "at_or_below"]

# Direction of a *change* relative to a subject's own baseline
# (BaselineRelativeRule): a rise or a fall. Inclusive variants don't
# apply -- the delta/ratio comparison is already >=.
TrendDirection = Literal["above", "below"]


def _threshold_expr(value: pl.Expr, threshold: float, direction: Direction) -> pl.Expr:
    """One polars comparison expression for a threshold rule's direction."""
    if direction == "above":
        return value > threshold
    if direction == "below":
        return value < threshold
    if direction == "at_or_above":
        return value >= threshold
    if direction == "at_or_below":
        return value <= threshold
    raise ValueError(f"unknown direction: {direction!r}")


@dataclass(frozen=True)
class ConceptRule:
    """One instantaneous (MEDS code prefix, threshold) crossing."""

    code_prefix: str
    """Matched via ``code.str.starts_with(code_prefix)``, e.g. ``"LAB//220045//"``."""

    threshold: float
    direction: Direction


@dataclass(frozen=True)
class CodeOccurrenceRule:
    """The event's *occurrence* is the whole signal: no numeric threshold.

    Triggers for a subject if any event's code matches
    ``code_pattern`` (a case-insensitive regular expression over the
    full code string, not a prefix -- medication codes embed free-text
    drug names mid-string, e.g.
    ``MEDICATION//Norepinephrine 8 mg/250 mL``, which a prefix can't
    reach). Unlike every threshold rule, "observed" and "triggered" are
    the same set here: there is no separate "the lab was drawn but the
    value was normal" state for an order/administration event. The
    observed-mask semantics that
    :func:`~odyssey.models.concept_bottleneck.concept_loss` relies on
    are preserved by ``observed_pattern``: subjects with at least one
    event matching it (default: any event in ``observed_families``'s
    code families) count as observed, so "no vasopressor was given"
    is a real negative wherever medication data exists at all, rather
    than every non-triggered subject being masked out of supervision.
    """

    code_pattern: str
    """Case-insensitive regex matched against the whole code string."""

    observed_families: Tuple[str, ...] = (
        "MEDICATION",
        "INFUSION_DRUG",
        "INFUSION_START",
        "INFUSION_END",
    )
    """Code families (first ``//`` segment) whose presence marks the
    subject as observed -- i.e. this kind of data was recorded at all,
    so an absent match is a genuine negative. Defaults cover both the
    MIMIC-IV families (MEDICATION, INFUSION_START/END) and eICU's
    (MEDICATION, INFUSION_DRUG)."""

    match_text_value: bool = False
    """Also match ``code_pattern`` against the ``text_value`` column
    when present (eICU charts infusion drug names there, under a bare
    ``INFUSION_DRUG`` code)."""


@dataclass(frozen=True)
class SustainedRule:
    """A threshold crossing that must recur at least ``min_gap_hours`` apart.

    Operationalized as: among this subject's qualifying (threshold-crossing)
    observations for this code, the earliest and latest are at least
    ``min_gap_hours`` apart. Weaker than "stayed above threshold
    continuously the whole time" (which would need every intervening
    reading to also qualify), but a real, cheaply-implementable
    improvement over a single instantaneous crossing: a lone transient
    spike has zero span and correctly does not trigger, while genuine
    recurring/sustained abnormality spread across the stay does.
    """

    code_prefix: str
    threshold: float
    direction: Direction
    min_gap_hours: float = 1.0


@dataclass(frozen=True)
class BaselineRelativeRule:
    """KDIGO-style: value rose/fell by an absolute delta or a ratio.

    Triggers if any two readings of the same subject and code, with the
    later one within ``window_hours`` of the earlier one, differ enough
    in the direction given by ``direction`` (``"above"``: a rise
    triggers; ``"below"``: a fall triggers). Exactly one of ``delta``
    (an absolute difference, e.g. KDIGO AKI Stage 1's "+0.3 mg/dL within
    48h") or ``ratio`` (a proportional difference, e.g. Stage 1's other
    trigger "1.5x baseline within 7 days", or Stage 2/3's 2x/3x) must be
    given. Baseline is the subject's own earlier value, not a fixed
    population reference range.
    """

    code_prefix: str
    direction: TrendDirection
    window_hours: float
    delta: Optional[float] = None
    ratio: Optional[float] = None

    def __post_init__(self) -> None:
        """Require exactly one of ``delta``/``ratio``."""
        if (self.delta is None) == (self.ratio is None):
            raise ValueError(
                "BaselineRelativeRule needs exactly one of delta or ratio, got "
                f"delta={self.delta!r} ratio={self.ratio!r}"
            )


@dataclass(frozen=True)
class DerivedGcsTotalRule:
    """qSOFA/NEWS2-style: total GCS (eye + verbal + motor) crosses a threshold.

    See the module docstring for why this exists (MIMIC-IV has no single
    "GCS total" itemid) and how components are paired.
    """

    eye_prefix: str
    verbal_prefix: str
    motor_prefix: str
    threshold: float
    direction: Direction = "below"
    max_component_gap_minutes: float = 15.0


ComponentRule = Union[
    ConceptRule,
    SustainedRule,
    BaselineRelativeRule,
    DerivedGcsTotalRule,
    CodeOccurrenceRule,
]


@dataclass(frozen=True)
class AnyOf:
    """One criterion satisfied if any of several component rules fire.

    For nesting an OR inside a :class:`CompositeConceptDefinition`'s N-of-M
    count, e.g. SIRS's "abnormal temperature" criterion is satisfied by
    either a too-high or a too-low reading, but must count as exactly one
    criterion toward ``min_criteria``, not (potentially) two.
    """

    rules: List[ComponentRule]


CompositeComponent = Union[ComponentRule, AnyOf]


@dataclass(frozen=True)
class ConceptDefinition:
    """A clinical concept, derived by OR-ing one or more component rules."""

    name: str
    rules: List[ComponentRule]
    description: str


@dataclass(frozen=True)
class CompositeConceptDefinition:
    """A concept triggered when >= ``min_criteria`` of several criteria fire.

    SIRS/qSOFA-style. Each entry in ``components`` is one criterion;
    :class:`AnyOf` nests an OR of rules that still only counts once.
    "Met" is evaluated per-subject across their whole extracted window
    (did this criterion ever fire), not required to hold simultaneously
    at one instant -- a per-subject, not per-timepoint, simplification
    consistent with every other concept in this module (see the module
    docstring).
    """

    name: str
    components: List[CompositeComponent]
    min_criteria: int
    description: str


AnyConceptDefinition = Union[ConceptDefinition, CompositeConceptDefinition]


# ---------------------------------------------------------------------------
# Simple instantaneous vital-sign/lab concepts (v1, kept): no known
# over-triggering problem, so an instantaneous threshold remains
# appropriate -- see the module docstring for which v1 concepts were
# upgraded instead (tachypnea, acute_kidney_injury).
# ---------------------------------------------------------------------------

_TACHYCARDIA = ConceptDefinition(
    "tachycardia",
    [ConceptRule("LAB//220045//", 100.0, "above")],
    "Heart rate > 100 bpm.",
)
_BRADYCARDIA = ConceptDefinition(
    "bradycardia",
    [ConceptRule("LAB//220045//", 60.0, "below")],
    "Heart rate < 60 bpm.",
)
_HYPOTENSION = ConceptDefinition(
    "hypotension",
    [
        ConceptRule("LAB//220179//", 90.0, "below"),  # non-invasive systolic BP
        ConceptRule("LAB//220050//", 90.0, "below"),  # arterial systolic BP
    ],
    "Systolic blood pressure < 90 mmHg.",
)
_HYPERTENSION = ConceptDefinition(
    "hypertension",
    [
        ConceptRule("LAB//220179//", 140.0, "above"),
        ConceptRule("LAB//220050//", 140.0, "above"),
    ],
    "Systolic blood pressure > 140 mmHg.",
)
_HYPOXIA = ConceptDefinition(
    "hypoxia",
    [ConceptRule("LAB//220277//", 92.0, "below")],
    "SpO2 < 92%.",
)
_FEVER = ConceptDefinition(
    "fever",
    [
        ConceptRule("LAB//223761//", 100.4, "above"),  # Fahrenheit
        ConceptRule("LAB//223762//", 38.0, "above"),  # Celsius
    ],
    "Temperature > 100.4F / 38.0C.",
)
_HYPOTHERMIA = ConceptDefinition(
    "hypothermia",
    [
        ConceptRule("LAB//223761//", 96.8, "below"),
        ConceptRule("LAB//223762//", 36.0, "below"),
    ],
    "Temperature < 96.8F / 36.0C.",
)
_ELEVATED_LACTATE = ConceptDefinition(
    "elevated_lactate",
    [ConceptRule("LAB//RESULT//50813//", 2.0, "above")],
    "Serum lactate > 2.0 mmol/L.",
)

# ---------------------------------------------------------------------------
# v2: upgraded from v1's single-instantaneous-threshold proxies.
# ---------------------------------------------------------------------------

_SUSTAINED_TACHYPNEA = ConceptDefinition(
    "sustained_tachypnea",
    [SustainedRule("LAB//220210//", 20.0, "above", min_gap_hours=1.0)],
    "Respiratory rate > 20 breaths/min, recurring at least 1 hour apart -- "
    "replaces v1's single-instantaneous-reading 'tachypnea', which "
    "triggered for 96.5% of subjects with respiratory-rate data in the "
    "real MIMIC-IV extraction (too loose to be a useful signal; a single "
    "transient spike, e.g. during one stressful procedure, was "
    "indistinguishable from true sustained tachypnea).",
)

_CREATININE = "LAB//RESULT//50912//"

_ACUTE_KIDNEY_INJURY = ConceptDefinition(
    "acute_kidney_injury",
    [
        BaselineRelativeRule(
            _CREATININE, delta=0.3, direction="above", window_hours=48.0
        ),
        BaselineRelativeRule(
            _CREATININE, ratio=1.5, direction="above", window_hours=168.0
        ),
    ],
    "KDIGO AKI Stage 1 (either trigger): serum creatinine rose by >= 0.3 "
    "mg/dL within 48 hours, OR rose to >= 1.5x an earlier reading within "
    "7 days (168h). Replaces v1's 'creatinine > 1.5 mg/dL' single-value "
    "proxy, which ignored a patient's own baseline. See aki_stage_2 and "
    "aki_stage_3 for higher severity; urine-output-based staging is not "
    "implemented -- see 'still open'.",
)

_AKI_STAGE_2 = ConceptDefinition(
    "aki_stage_2",
    [
        BaselineRelativeRule(
            _CREATININE, ratio=2.0, direction="above", window_hours=168.0
        )
    ],
    "KDIGO AKI Stage 2: serum creatinine rose to >= 2.0x an earlier "
    "reading within 7 days. Urine-output-based staging (<0.5 mL/kg/h for "
    ">= 12h) is not implemented -- see 'still open'.",
)

_AKI_STAGE_3 = ConceptDefinition(
    "aki_stage_3",
    [
        BaselineRelativeRule(
            _CREATININE, ratio=3.0, direction="above", window_hours=168.0
        ),
        ConceptRule(_CREATININE, 4.0, "at_or_above"),  # KDIGO: >= 4.0, inclusive
    ],
    "KDIGO AKI Stage 3 (either trigger): serum creatinine rose to "
    ">= 3.0x an earlier reading within 7 days, OR any reading >= 4.0 "
    "mg/dL. Renal-replacement-therapy initiation and urine-output-based "
    "staging (<0.3 mL/kg/h for >= 24h, or anuria for >= 12h) are not "
    "implemented -- see 'still open'.",
)

_SIRS = CompositeConceptDefinition(
    "sirs",
    components=[
        AnyOf(
            [
                ConceptRule("LAB//223761//", 100.4, "above"),
                ConceptRule("LAB//223762//", 38.0, "above"),
                ConceptRule("LAB//223761//", 96.8, "below"),
                ConceptRule("LAB//223762//", 36.0, "below"),
            ]
        ),  # criterion 1: temp > 38C/100.4F or < 36C/96.8F
        ConceptRule("LAB//220045//", 90.0, "above"),  # criterion 2: HR > 90
        ConceptRule("LAB//220210//", 20.0, "above"),  # criterion 3: RR > 20
        AnyOf(
            [
                ConceptRule("LAB//RESULT//51301//", 12.0, "above"),
                ConceptRule("LAB//RESULT//51301//", 4.0, "below"),
            ]
        ),  # criterion 4: WBC > 12k or < 4k per uL
    ],
    min_criteria=2,
    description=(
        "SIRS (Systemic Inflammatory Response Syndrome): >= 2 of "
        "{abnormal temperature, HR > 90, RR > 20, abnormal WBC}. The "
        "bands-percentage alternative for the WBC criterion (>10% bands, "
        "an option even with a normal WBC count) is not included -- no "
        "verified MIMIC-IV itemid for band percentage has been found yet."
    ),
)

_QSOFA = CompositeConceptDefinition(
    "qsofa",
    components=[
        ConceptRule("LAB//220210//", 22.0, "at_or_above"),  # RR >= 22
        AnyOf(
            [
                ConceptRule("LAB//220179//", 100.0, "at_or_below"),
                ConceptRule("LAB//220050//", 100.0, "at_or_below"),
            ]
        ),  # SBP <= 100
        DerivedGcsTotalRule(
            eye_prefix="LAB//220739//",
            verbal_prefix="LAB//223900//",
            motor_prefix="LAB//223901//",
            threshold=15.0,
            direction="below",
        ),  # GCS < 15 (any drop from fully alert)
    ],
    min_criteria=2,
    description=(
        "qSOFA (quick Sequential Organ Failure Assessment), a bedside "
        "sepsis-screening score: >= 2 of {RR >= 22, SBP <= 100 mmHg, "
        "GCS < 15}."
    ),
)


# entry 06/07 decision (f): the first occurrence-keyed concept. The drug-name
# regex reaches both MIMIC-IV medication codes (drug name embedded in the code
# string) and, via match_text_value, eICU infusion events (bare INFUSION_DRUG
# code, drug name in text_value). "epinephrine" also matching "norepinephrine"
# is deliberate; both are vasopressors and Rust regex has no lookbehind.
# Dobutamine is excluded on purpose: an inotrope, not a vasopressor.
_ON_VASOPRESSORS = ConceptDefinition(
    "on_vasopressors",
    [
        CodeOccurrenceRule(
            r"norepinephrine|levophed|epinephrine|vasopressin|phenylephrine"
            r"|neo-?synephrine|dopamine|angiotensin",
            match_text_value=True,
        )
    ],
    "Received at least one vasopressor (norepinephrine, epinephrine, "
    "vasopressin, phenylephrine, dopamine, or angiotensin II) -- the "
    "canonical shock/deterioration marker, derived from medication and "
    "infusion events rather than a numeric threshold. 'Observed' means "
    "the subject has any medication/infusion data at all, so an absent "
    "match is a genuine negative, not missingness.",
)


CONCEPTS: List[AnyConceptDefinition] = [
    _TACHYCARDIA,
    _BRADYCARDIA,
    _HYPOTENSION,
    _HYPERTENSION,
    _HYPOXIA,
    _FEVER,
    _HYPOTHERMIA,
    _ELEVATED_LACTATE,
    _SUSTAINED_TACHYPNEA,
    _ACUTE_KIDNEY_INJURY,
    _AKI_STAGE_2,
    _AKI_STAGE_3,
    _SIRS,
    _QSOFA,
    _ON_VASOPRESSORS,
]


def _instantaneous_ids(
    events: pl.DataFrame,
    rule: ConceptRule,
    subject_id_col: str,
    code_col: str,
    value_col: str,
) -> Tuple[Set[int], Set[int]]:
    matched = events.filter(
        pl.col(code_col).str.starts_with(rule.code_prefix)
        & pl.col(value_col).is_not_null()
    )
    observed = set(matched[subject_id_col].to_list())
    comparison = _threshold_expr(pl.col(value_col), rule.threshold, rule.direction)
    triggered = set(matched.filter(comparison)[subject_id_col].to_list())
    return observed, triggered


def _sustained_ids(
    events: pl.DataFrame,
    rule: SustainedRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> Tuple[Set[int], Set[int]]:
    matched = events.filter(
        pl.col(code_col).str.starts_with(rule.code_prefix)
        & pl.col(value_col).is_not_null()
    )
    observed = set(matched[subject_id_col].to_list())
    comparison = _threshold_expr(pl.col(value_col), rule.threshold, rule.direction)
    qualifying = matched.filter(comparison)
    if qualifying.height == 0:
        return observed, set()

    span = qualifying.group_by(subject_id_col).agg(
        (pl.col(time_col).max() - pl.col(time_col).min()).alias("_span")
    )
    min_gap = timedelta(hours=rule.min_gap_hours)
    triggered = set(span.filter(pl.col("_span") >= min_gap)[subject_id_col].to_list())
    return observed, triggered


def _baseline_relative_ids(
    events: pl.DataFrame,
    rule: BaselineRelativeRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> Tuple[Set[int], Set[int]]:
    """Check whether any reading exceeds an earlier one (within window) by delta/ratio.

    Uses a per-subject rolling extreme (min for direction="above", max for
    "below") as the baseline, rather than an explicit self-join over every
    pair of a subject's readings: a self-join keyed only on subject_id
    produces a full k^2 cartesian product of that subject's own readings
    (k = readings for that subject), which blew up badly for a frequently
    -repeated lab value (creatinine, for AKI staging) on real data -- a
    100-shard MIMIC-IV run OOM-killed an 83GB VM inside this exact
    function. The rewrite is mathematically equivalent: for direction=
    "above", "exists an earlier reading v1 in the window with v2 - v1 >=
    delta (or v2 >= ratio*v1)" is true iff it's true for the *smallest*
    v1 in that window (subtracting/dividing by a smaller positive number
    only makes the comparison easier to satisfy) -- symmetric with the
    largest v1 for direction="below". ``closed="left"`` matches the
    original's strict ``t1 < t2`` (baseline excludes the current reading
    itself).
    """
    matched = (
        events.filter(
            pl.col(code_col).str.starts_with(rule.code_prefix)
            & pl.col(value_col).is_not_null()
        )
        .select(subject_id_col, time_col, value_col)
        .sort(subject_id_col, time_col)
    )
    observed = set(matched[subject_id_col].to_list())
    if matched.height == 0:
        return observed, set()

    window_size = timedelta(hours=rule.window_hours)
    baseline_expr = (
        pl.col(value_col).rolling_min_by(
            time_col, window_size=window_size, closed="left"
        )
        if rule.direction == "above"
        else pl.col(value_col).rolling_max_by(
            time_col, window_size=window_size, closed="left"
        )
    ).over(subject_id_col)
    matched = matched.with_columns(baseline_expr.alias("_baseline"))

    if rule.ratio is not None:
        comparison = (
            pl.col(value_col) >= rule.ratio * pl.col("_baseline")
            if rule.direction == "above"
            else pl.col(value_col) <= pl.col("_baseline") / rule.ratio
        )
    else:
        delta_expr = (
            pl.col(value_col) - pl.col("_baseline")
            if rule.direction == "above"
            else pl.col("_baseline") - pl.col(value_col)
        )
        comparison = delta_expr >= rule.delta

    triggered = set(
        matched.filter(pl.col("_baseline").is_not_null() & comparison)[
            subject_id_col
        ].to_list()
    )
    return observed, triggered


def _derived_gcs_total_ids(
    events: pl.DataFrame,
    rule: DerivedGcsTotalRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> Tuple[Set[int], Set[int]]:
    def _component(prefix: str) -> pl.DataFrame:
        return (
            events.filter(
                pl.col(code_col).str.starts_with(prefix)
                & pl.col(value_col).is_not_null()
            )
            .select(subject_id_col, time_col, value_col)
            .sort([subject_id_col, time_col])
        )

    eye = _component(rule.eye_prefix)
    verbal = _component(rule.verbal_prefix)
    motor = _component(rule.motor_prefix)
    observed = (
        set(eye[subject_id_col].to_list())
        | set(verbal[subject_id_col].to_list())
        | set(motor[subject_id_col].to_list())
    )
    if eye.height == 0 or verbal.height == 0 or motor.height == 0:
        return observed, set()

    tolerance = timedelta(minutes=rule.max_component_gap_minutes)
    # eye/verbal/motor are each already sorted by [subject_id_col, time_col]
    # (see _component above); polars' join_asof(by=...) still can't verify
    # per-group sortedness cheaply and warns regardless -- a known library
    # limitation (https://github.com/pola-rs/polars/issues), not a sign the
    # data is actually unsorted here, so it's suppressed rather than left
    # as noise on every call.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sortedness of columns cannot be checked"
        )
        paired = eye.join_asof(
            verbal.rename({value_col: f"{value_col}_verbal"}),
            on=time_col,
            by=subject_id_col,
            strategy="nearest",
            tolerance=tolerance,
        )
        paired = paired.filter(pl.col(f"{value_col}_verbal").is_not_null()).sort(
            [subject_id_col, time_col]
        )
        paired = paired.join_asof(
            motor.rename({value_col: f"{value_col}_motor"}),
            on=time_col,
            by=subject_id_col,
            strategy="nearest",
            tolerance=tolerance,
        )
    paired = paired.filter(pl.col(f"{value_col}_motor").is_not_null())
    if paired.height == 0:
        return observed, set()

    total = (
        pl.col(value_col) + pl.col(f"{value_col}_verbal") + pl.col(f"{value_col}_motor")
    )
    comparison = _threshold_expr(total, rule.threshold, rule.direction)
    triggered = set(paired.filter(comparison)[subject_id_col].to_list())
    return observed, triggered


def _component_ids(
    events: pl.DataFrame,
    rule: ComponentRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> Tuple[Set[int], Set[int]]:
    if isinstance(rule, ConceptRule):
        return _instantaneous_ids(events, rule, subject_id_col, code_col, value_col)
    if isinstance(rule, SustainedRule):
        return _sustained_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    if isinstance(rule, BaselineRelativeRule):
        return _baseline_relative_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    if isinstance(rule, DerivedGcsTotalRule):
        return _derived_gcs_total_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    if isinstance(rule, CodeOccurrenceRule):
        return _occurrence_ids(
            events, rule, subject_id_col=subject_id_col, code_col=code_col
        )
    raise TypeError(f"unknown component rule type: {type(rule)!r}")


def _occurrence_ids(
    events: pl.DataFrame,
    rule: CodeOccurrenceRule,
    *,
    subject_id_col: str,
    code_col: str,
) -> Tuple[Set[int], Set[int]]:
    """Observed/triggered sets for an occurrence-keyed rule.

    Observed = subjects with any event in ``rule.observed_families``
    (this kind of data exists at all for them); triggered = subjects
    with an event whose code (or, opted in, ``text_value``) matches
    ``rule.code_pattern`` case-insensitively. Triggered subjects are
    always counted observed, even if the matching code sits outside the
    declared families -- a match is the strongest possible evidence the
    data exists.
    """
    pattern = f"(?i){rule.code_pattern}"
    family = pl.col(code_col).str.split("//").list.first()
    observed = set(
        events.filter(family.is_in(list(rule.observed_families)))[
            subject_id_col
        ].to_list()
    )
    match = pl.col(code_col).str.contains(pattern)
    if rule.match_text_value and "text_value" in events.columns:
        match = match | pl.col("text_value").str.contains(pattern).fill_null(False)
    triggered = set(events.filter(match)[subject_id_col].to_list())
    return observed | triggered, triggered


def _composite_component_ids(
    events: pl.DataFrame,
    component: CompositeComponent,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> Tuple[Set[int], Set[int]]:
    if isinstance(component, AnyOf):
        observed: Set[int] = set()
        triggered: Set[int] = set()
        for rule in component.rules:
            obs, trig = _component_ids(
                events,
                rule,
                subject_id_col=subject_id_col,
                code_col=code_col,
                value_col=value_col,
                time_col=time_col,
            )
            observed |= obs
            triggered |= trig
        return observed, triggered
    return _component_ids(
        events,
        component,
        subject_id_col=subject_id_col,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )


def label_concepts(
    events: pl.DataFrame,
    concepts: Sequence[AnyConceptDefinition] = CONCEPTS,
    *,
    subject_id_col: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
) -> pl.DataFrame:
    """Derive per-subject concept labels and an observed-mask from MEDS events.

    Every subject in ``events`` gets a ``{name}`` binary label and a
    ``{name}_observed`` mask (1 if at least one matching measurement
    exists at all, vs. that lab/vital simply never having been drawn for
    that subject -- the distinction
    :func:`odyssey.models.concept_bottleneck.concept_loss`'s masking is
    meant to exclude from supervision). For a
    :class:`CompositeConceptDefinition`, ``{name}`` is 1 if at least
    ``min_criteria`` of its components fired (anywhere in the subject's
    history, not necessarily simultaneously -- see that class's
    docstring), and ``{name}_observed`` is 1 if at least one component
    had at least one matching measurement.
    """
    subject_ids = events.select(subject_id_col).unique()
    out = subject_ids

    for concept in concepts:
        if isinstance(concept, CompositeConceptDefinition):
            if not concept.components:
                raise ValueError(f"concept {concept.name!r} has no components defined")
            observed_ids: Set[int] = set()
            criteria_met_count: Dict[int, int] = {}
            for component in concept.components:
                obs, trig = _composite_component_ids(
                    events,
                    component,
                    subject_id_col=subject_id_col,
                    code_col=code_col,
                    value_col=value_col,
                    time_col=time_col,
                )
                observed_ids |= obs
                for sid in trig:
                    criteria_met_count[sid] = criteria_met_count.get(sid, 0) + 1
            triggered_ids = {
                sid
                for sid, count in criteria_met_count.items()
                if count >= concept.min_criteria
            }
        else:
            if not concept.rules:
                raise ValueError(f"concept {concept.name!r} has no rules defined")
            observed_ids = set()
            triggered_ids = set()
            for rule in concept.rules:
                obs, trig = _component_ids(
                    events,
                    rule,
                    subject_id_col=subject_id_col,
                    code_col=code_col,
                    value_col=value_col,
                    time_col=time_col,
                )
                observed_ids |= obs
                triggered_ids |= trig

        out = out.with_columns(
            pl.col(subject_id_col)
            .is_in(triggered_ids)
            .cast(pl.Int8)
            .alias(concept.name),
            pl.col(subject_id_col)
            .is_in(observed_ids)
            .cast(pl.Int8)
            .alias(f"{concept.name}_observed"),
        )

    return out


def label_concepts_by_visit(
    events: pl.DataFrame,
    concepts: Sequence[AnyConceptDefinition] = CONCEPTS,
    *,
    subject_id_col: str = "subject_id",
    visit_col: str = "hadm_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
) -> pl.DataFrame:
    """:func:`label_concepts`, scoped to each visit instead of the whole stay.

    Returns one row per ``(subject_id, visit_col)`` pair present in
    ``events`` (rows with a null ``visit_col`` -- solo/outpatient events
    -- are excluded), with the same ``{name}`` / ``{name}_observed``
    columns, evaluated over only that visit's events.

    This is the label side of visit-scoped concept supervision: a
    whole-stay "did this ever happen" label asks the model to recall a
    possibly single event from arbitrarily far back, which a compressed
    recurrent state cannot guarantee -- the exact failure mode the
    subset-run evaluation showed for instantaneous vital-sign concepts
    (weak AUROCs, entangled traces) while windowed concepts thrived.
    "Did this happen during this visit", supervised at each visit's last
    event, aligns the label's memory demands with what the architecture
    actually retains, and grounds many positions per subject instead of
    one.

    One deliberate semantic narrowing: :class:`BaselineRelativeRule`
    baselines cannot reach across visits (KDIGO's 48h/7-day creatinine
    windows evaluate within one admission), matching how the criteria
    are used clinically during a stay.
    """
    scoped = events.filter(pl.col(visit_col).is_not_null())
    keys = (
        scoped.select(subject_id_col, visit_col)
        .unique(maintain_order=True)
        .with_row_index("_visit_key")
    )
    scoped = scoped.join(keys, on=[subject_id_col, visit_col], how="left")
    labeled = label_concepts(
        scoped,
        concepts,
        subject_id_col="_visit_key",
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    return (
        labeled.join(keys, on="_visit_key", how="left")
        .drop("_visit_key")
        .select(
            [subject_id_col, visit_col]
            + [c for c in labeled.columns if c != "_visit_key"]
        )
    )
