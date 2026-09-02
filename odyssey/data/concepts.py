"""Clinical concept labels derived from MEDS events.

A registry of clinically-recognizable vital-sign/lab abnormalities,
derived from a MEDS event table to supervise
:class:`odyssey.models.concept_bottleneck.ConceptBottleneck`.

The clinical knowledge lives in :data:`CANONICAL_CONCEPTS`, written once
against LOINC codes -- the portable, institution-agnostic vocabulary --
and :func:`concepts_for_source` expands it to the concrete MEDS code
prefixes one source's extraction uses (MIMIC-IV's
``LAB//{itemid}//``/``LAB//RESULT//{itemid}//``, eICU's ``VITALS//...``
and ``LAB//{labname}//``), via the per-source mapping tables in
:mod:`odyssey.data.code_mapping`. :data:`CONCEPTS` is the MIMIC-IV
expansion, kept as the module-level default. Thresholds match the
widely-cited `MIT-LCP/mimic-code
<https://github.com/MIT-LCP/mimic-code>`_ concept queries.

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

Rule types added since, each alongside the concept that needed it rather
than to that original list: :class:`CodeOccurrenceRule` (the event's
occurrence is the whole signal, no numeric value -- vasopressor
administration, renal-replacement-therapy initiation),
:class:`DerivedSofaSignalRule` and :class:`Sepsis3Rule` (signals derived
by :mod:`odyssey.data.sofa`), and :class:`DerivedUrineRateRule` (KDIGO's
urine-output leg: mL/kg/h over a 6/12/24 h trailing window, or absolute
mL for the anuria branch).

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

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from odyssey.data.code_mapping import prefixes_for_loinc, unit_for
from odyssey.data.sidecars import ANTIBIOTIC_ORDERS, MICROBIOLOGY, active_sidecar
from odyssey.data.sofa import (
    assessable_keys,
    pf_ratio_readings,
    sofa_supported,
    sofa_timeseries,
    urine_output_24h,
    urine_output_rate,
)


logger = logging.getLogger(__name__)


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

    observed_families: tuple[str, ...] = (
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
    when present (eICU charts infusion drug names there; the v1
    extraction emitted a bare ``INFUSION_DRUG`` code and only v2 puts
    the name in the code as well)."""


@dataclass(frozen=True)
class SustainedRule:
    """A threshold crossing that must recur at least ``min_gap_hours`` apart.

    Operationalized as: among this subject's qualifying (threshold-crossing)
    observations for this signal, the earliest and latest are at least
    ``min_gap_hours`` apart. Weaker than "stayed above threshold
    continuously the whole time" (which would need every intervening
    reading to also qualify), but a real, cheaply-implementable
    improvement over a single instantaneous crossing: a lone transient
    spike has zero span and correctly does not trigger, while genuine
    recurring/sustained abnormality spread across the stay does.

    ``extra_prefixes`` widens the match to clinically-interchangeable
    measurements of the same signal (a cuff MAP and an arterial-line MAP),
    POOLED before the recurrence check. Recurrence is a property of the
    patient's physiology, not of one charting route: a MAP < 65 on the
    cuff at t and on the arterial line at t+2h is sustained hypotension,
    and evaluating each prefix separately (the pre-2026-08-30 expansion
    of a multi-prefix ``LoincSustained``) silently missed exactly that
    cross-modality recurrence.
    """

    code_prefix: str
    threshold: float
    direction: Direction
    min_gap_hours: float = 1.0
    extra_prefixes: tuple[str, ...] = ()


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
    delta: float | None = None
    ratio: float | None = None

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


# Antibacterials (stems + common MIMIC brand names), after mimic-code's
# ``antibiotic.sql`` list; antifungals/antivirals deliberately excluded,
# as there. Matched case-insensitively against medication START codes.
ANTIBIOTIC_PATTERN = (
    r"cillin|cef|ceph|penem|vancomycin|vancocin|linezolid|zyvox|daptomycin|cubicin"
    r"|azithromycin|zithromax|clarithromycin|biaxin|erythromycin|clindamycin|cleocin"
    r"|gentamicin|tobramycin|amikacin|streptomycin|kanamycin|neomycin"
    r"|floxacin|cipro|levaquin|avelox|noroxin|factive|proquin"
    r"|cycline|doxy|minocin|solodyn|vibramycin|tetra"
    r"|metronidazole|flagyl|sulfamethoxazole|trimethoprim|bactrim|septra|smz"
    r"|aztreonam|azactam|cayston|rifampin|rifadin|nitrofurantoin|macrobid|macrodantin"
    r"|chloramphenicol|synercid|vibativ|mupirocin|bactroban|monurol|fosfomycin"
    r"|augmentin|unasyn|zosyn|timentin|tazobactam|sulbactam|bicillin|pfizerpen"
    r"|keflex|kefzol|rocephin|claforan|fortaz|tazicef|maxipime|cefotan|mefoxin"
    r"|ceftin|zinacef|omnicef|suprax|vantin|spectracef|cedax|duricef|raniclor"
)
# Routes that do not count as systemic antibiotic therapy (topical, eye,
# ear, nasal, inhaled, irrigation, vaginal, rectal, intravitreal, locks):
# mimic-code's word exclusions plus MIMIC-IV's own route ABBREVIATIONS as
# actually charted on prescriptions/eMAR (TP, NU, NS/ND/NAS, OU/OS/OD, AU/AS/AD,
# IH/IN, IRR/IR, VG, PR, IVT, LOCK, DWELL -- audited on the real orders table,
# research journal entry 44). Matched case-insensitively, whole field.
ANTIBIOTIC_ROUTE_EXCLUDE = (
    r"^(OU|OS|OD|AU|AS|AD|TP|NU|NS|ND|NAS|IH|IN|IRR|IR|VG|PR|IVT|LOCK|DWELL|EX)$"
    r"|ear|eye|ophth|otic|nasal|inhal|neb|topical|cream|gel|ointment|intravitreal"
    r"|desensitization|irrig|vaginal|rectal"
)


@dataclass(frozen=True)
class DerivedSofaSignalRule:
    """A threshold on a signal derived by :mod:`odyssey.data.sofa`.

    Two clinical states need a derivation the plain rule types cannot
    express, and both already exist (and are tested) inside the SOFA
    scorer, so they are called rather than re-derived here:

    - ``pf_ratio``: PaO2/FiO2, pairing each arterial gas with the most
      recent FiO2 (:func:`~odyssey.data.sofa.pf_ratio_readings`);
    - ``urine_24h``: millilitres voided in the trailing 24 hours, only
      once a key has 24 hours of record (:func:`~odyssey.data.sofa.urine_output_24h`).

    ``source`` is fixed at expansion time, like :class:`Sepsis3Rule`: the
    derivation needs that source's non-LOINC ingredients (ventilation
    codes) and only sources with a :data:`~odyssey.data.sofa.SOFA_SOURCE_CONFIG`
    entry can expand it.
    """

    signal: Literal["pf_ratio", "urine_24h"]
    threshold: float
    direction: Direction
    source: str


@dataclass(frozen=True)
class DerivedUrineRateRule:
    """KDIGO's urine-output leg: trailing urine output crosses a threshold.

    :class:`DerivedSofaSignalRule`'s ``urine_24h`` signal is fixed at
    SOFA's own shape (absolute mL, 24 h). KDIGO stages AKI on a
    *weight-normalized rate* (mL/kg/h) over three different windows
    (6 h, 12 h, 24 h), so the window and the normalization are rule
    parameters here, resolved by
    :func:`~odyssey.data.sofa.urine_output_rate`:

    - ``weight_normalized=True``: ``value`` is mL/kg/h, and a window is
      scored only where a body weight was charted at or before it (daily
      weight preferred, admission weight as the early-stay fallback).
      Windows with no weight at all are not scored -- see that function
      for why defaulting a weight would be worse than abstaining.
    - ``weight_normalized=False``: ``value`` is absolute mL over the
      window, which is what Stage 3's anuria branch (0 mL over 12 h)
      needs -- 0 mL is 0 mL at any body weight, so that branch stays
      assessable for the majority of keys that have no weight reading.

    ``source`` is fixed at expansion time for the same reason as
    :class:`DerivedSofaSignalRule`: the derivation needs that source's
    non-LOINC weight item ids.
    """

    threshold: float
    direction: Direction
    window_hours: float
    source: str
    weight_normalized: bool = True


# Renal replacement therapy: KDIGO makes RRT initiation an automatic AKI
# Stage 3, whatever creatinine and urine output say. The item ids are
# mimic-code's own ``mimic-iv/concepts/treatment/rrt.sql``
# ``dialysis_active = 1`` procedureevents set, and its two deliberate
# exclusions are excluded here too: 224270 (Dialysis Catheter -- line
# placement, not therapy) and 225436 (CRRT Filter Change -- maintenance on
# an already-active line, which would double-count an episode already
# caught by its own START row). The codes take the same
# ``PROCEDURE//START//{itemid}`` shape as the ventilation item ids in
# :data:`odyssey.data.sofa.SOFA_SOURCE_CONFIG`; the alternation is anchored
# so a longer item id that merely begins with one of these cannot match.
RRT_ITEMIDS: tuple[str, ...] = (
    "225441",  # Hemodialysis (intermittent, IHD)
    "225802",  # Dialysis - CRRT
    "225803",  # Dialysis - CVVHD
    "225805",  # Peritoneal Dialysis
    "225809",  # Dialysis - CVVHDF
    "225955",  # Dialysis - SCUF
)
RRT_CODE_PATTERN = r"^PROCEDURE//START//(" + "|".join(RRT_ITEMIDS) + r")(//|$)"


@dataclass(frozen=True)
class Sepsis3Rule:
    """Sepsis-3 onset (Singer 2016) as operationalized by mimic-code's ``sepsis3``.

    Suspected infection = a culture specimen (from the ``microbiology``
    sidecar, :mod:`odyssey.data.sidecars`) and a systemic antibiotic start
    within the standard window (antibiotic within ``antibiotic_after_hours``
    after the culture, or culture within ``culture_after_hours`` after the
    antibiotic); the suspicion time is the earlier of the two. Sepsis =
    a SOFA total >= ``sofa_threshold`` (baseline assumed 0, as mimic-code
    does) at any instant from ``sofa_before_hours`` before to
    ``sofa_after_hours`` after suspicion. Diagnosis codes are deliberately
    not used (no onset time).

    The reported first-trigger time departs from mimic-code's onset in one
    deliberate way (2026-08-30): it is the first instant ALL the label's
    ingredients exist in the record -- ``max(SOFA crossing, suspicion,
    completion of the confirming culture/antibiotic pair)`` -- rather than
    ``max(SOFA crossing, suspicion)``, which can precede the pair's second
    element by up to ``antibiotic_after_hours``. The triggered set is
    identical; only the stamp moves. This makes the time safe as a
    running-label / hazard-onset anchor (nothing is labeled true before it
    was determinable). Residual, documented limitation: the culture (and
    the preferred antibiotic-order source) live in label-only sidecars the
    models never see as inputs, so sepsis3 remains harder than concepts
    whose trigger is an input-visible event -- that is a modeling
    challenge, not leakage.
    """

    source: str
    antibiotic_pattern: str = ANTIBIOTIC_PATTERN
    route_exclude: str = ANTIBIOTIC_ROUTE_EXCLUDE
    antibiotic_after_hours: float = 72.0
    culture_after_hours: float = 24.0
    sofa_before_hours: float = 48.0
    sofa_after_hours: float = 24.0
    sofa_threshold: int = 2
    subject_col: str = "subject_id"
    visit_col: str = "hadm_id"
    route_col: str = "route"
    antibiotic_event_pattern: str = r"//START//|//Administered//"
    """Which medication rows count as the drug being given (not stopped)."""


ComponentRule = (
    ConceptRule
    | SustainedRule
    | BaselineRelativeRule
    | DerivedGcsTotalRule
    | CodeOccurrenceRule
    | Sepsis3Rule
    | DerivedSofaSignalRule
    | DerivedUrineRateRule
)


@dataclass(frozen=True)
class AnyOf:
    """One criterion satisfied if any of several component rules fire.

    For nesting an OR inside a :class:`CompositeConceptDefinition`'s N-of-M
    count, e.g. SIRS's "abnormal temperature" criterion is satisfied by
    either a too-high or a too-low reading, but must count as exactly one
    criterion toward ``min_criteria``, not (potentially) two.
    """

    rules: list[ComponentRule]


CompositeComponent = ComponentRule | AnyOf


@dataclass(frozen=True)
class ConceptDefinition:
    """A clinical concept, derived by OR-ing one or more component rules."""

    name: str
    rules: list[ComponentRule]
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
    components: list[CompositeComponent]
    min_criteria: int
    description: str


AnyConceptDefinition = ConceptDefinition | CompositeConceptDefinition


# ---------------------------------------------------------------------------
# Canonical (source-agnostic) concept registry.
#
# Clinical knowledge is written ONCE here, keyed by LOINC codes -- the
# portable vocabulary -- never by any single institution's item
# identifiers. :func:`concepts_for_source` resolves each LOINC to the
# concrete MEDS code prefixes a source's extraction uses (via
# :mod:`odyssey.data.code_mapping`) and produces the prefix-keyed
# concrete definitions everything downstream consumes. A criterion whose
# LOINC has no mapping in a source (e.g. GCS in eICU, whose spec does
# not extract nurseCharting yet) is dropped for that source with a
# logged warning; a composite survives as long as it retains at least
# ``min_criteria`` criteria, matching how the entry-06 eICU analysis
# translated the registry by hand.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoincThreshold:
    """Canonical form of :class:`ConceptRule`: threshold keyed by LOINC.

    ``loincs`` may list several codes when distinct measurements are
    clinically interchangeable for the rule (non-invasive 76534-7 and
    arterial 8480-6 systolic BP); inside a composite, a multi-prefix
    expansion is wrapped in :class:`AnyOf` so it still counts as one
    criterion. ``threshold`` applies to every prefix whose unit is
    unambiguous (no tag from :func:`odyssey.data.code_mapping.unit_for`);
    ``unit_thresholds`` gives the same cutoff in other units for prefixes
    that carry a tag, the way :class:`LoincBaselineRelative.unit_deltas`
    does for deltas. Temperature is charted in Fahrenheit or Celsius
    depending on source and itemid under one LOINC 8310-5, so it lists
    both units and no default; glucose and hemoglobin are mg/dL and g/dL
    on the US sources and mmol/L and g/L on GEMINI, so they keep the
    conventional default and add the SI cutoff. At least one of the two
    must be given, and a tagged prefix whose unit has no entry is an
    error rather than a silent fall-through to the wrong unit.
    """

    loincs: tuple[str, ...]
    direction: Direction
    threshold: float | None = None
    unit_thresholds: tuple[tuple[str, float], ...] | None = None

    def __post_init__(self) -> None:
        """Require a default threshold, per-unit thresholds, or both."""
        if self.threshold is None and self.unit_thresholds is None:
            raise ValueError(
                "LoincThreshold needs a threshold, unit_thresholds, or both"
            )


@dataclass(frozen=True)
class LoincSustained:
    """Canonical form of :class:`SustainedRule`."""

    loincs: tuple[str, ...]
    threshold: float
    direction: Direction
    min_gap_hours: float = 1.0


@dataclass(frozen=True)
class LoincBaselineRelative:
    """Canonical form of :class:`BaselineRelativeRule`.

    ``unit_deltas`` overrides ``delta`` per unit tag the way
    :class:`LoincThreshold.unit_thresholds` overrides ``threshold``: an
    absolute delta is unit-sensitive (KDIGO's "rose by 0.3" means mg/dL;
    the same clinical rule is "rose by 26.5" in umol/L), while ``ratio``
    is unit-free and never needs this. A prefix whose unit tag has no
    entry falls back to ``delta`` (the canonical mg/dL number), matching
    the untagged-prefix case.
    """

    loincs: tuple[str, ...]
    direction: TrendDirection
    window_hours: float
    delta: float | None = None
    ratio: float | None = None
    unit_deltas: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True)
class LoincGcsTotal:
    """Canonical form of :class:`DerivedGcsTotalRule`.

    Each component LOINC must resolve to exactly one prefix in a source;
    zero for any component makes the whole rule unresolvable there.
    """

    threshold: float
    direction: Direction = "below"
    max_component_gap_minutes: float = 15.0
    eye_loinc: str = "9267-6"
    verbal_loinc: str = "9270-0"
    motor_loinc: str = "9268-4"


# CodeOccurrenceRule is already source-agnostic (it matches drug names in
# code strings / text_value, not institution item ids), so it doubles as
# its own canonical form.
@dataclass(frozen=True)
class CanonicalSofaSignal:
    """Canonical form of :class:`DerivedSofaSignalRule` (source-agnostic)."""

    signal: Literal["pf_ratio", "urine_24h"]
    threshold: float
    direction: Direction = "below"


@dataclass(frozen=True)
class CanonicalUrineRate:
    """Canonical form of :class:`DerivedUrineRateRule` (source-agnostic).

    The urine LOINC itself resolves through the mapping layer inside
    :func:`~odyssey.data.sofa.urine_output_rate`; what is source-specific
    is only the weight item ids, so this rule -- like
    :class:`CanonicalSofaSignal` -- expands wherever
    :func:`~odyssey.data.sofa.sofa_supported` holds and is dropped
    elsewhere.
    """

    threshold: float
    window_hours: float
    direction: Direction = "below"
    weight_normalized: bool = True


@dataclass(frozen=True)
class CanonicalSepsis3:
    """Canonical Sepsis-3: expands to :class:`Sepsis3Rule` where SOFA is scorable."""

    sofa_threshold: int = 2


CanonicalRule = (
    LoincThreshold
    | LoincSustained
    | LoincBaselineRelative
    | LoincGcsTotal
    | CodeOccurrenceRule
    | CanonicalSepsis3
    | CanonicalSofaSignal
    | CanonicalUrineRate
)


@dataclass(frozen=True)
class CanonicalAnyOf:
    """Canonical form of :class:`AnyOf`."""

    rules: tuple[CanonicalRule, ...]


CanonicalComponent = CanonicalRule | CanonicalAnyOf


@dataclass(frozen=True)
class CanonicalConcept:
    """Canonical form of :class:`ConceptDefinition`."""

    name: str
    rules: tuple[CanonicalRule, ...]
    description: str


@dataclass(frozen=True)
class CanonicalComposite:
    """Canonical form of :class:`CompositeConceptDefinition`."""

    name: str
    components: tuple[CanonicalComponent, ...]
    min_criteria: int
    description: str


AnyCanonicalConcept = CanonicalConcept | CanonicalComposite


def _loinc_prefixes(loincs: tuple[str, ...], source: str) -> list[str]:
    """Every concrete prefix for ``loincs`` in ``source``, deterministic order."""
    out: list[str] = []
    for loinc in loincs:
        out.extend(sorted(prefixes_for_loinc(loinc, source=source)))
    return out


def _prefix_threshold(rule: LoincThreshold, prefix: str, source: str) -> float:
    """Pick the threshold that applies to one concrete prefix.

    A tagged unit takes its own entry from ``unit_thresholds``; an
    untagged prefix takes the default ``threshold``. A tagged prefix with
    no entry, or an untagged prefix on a rule with no default, is a
    configuration error: silently using the wrong unit's number would
    make a concept fire on assay noise or never at all.
    """
    unit = unit_for(prefix, source=source)
    if unit is not None and rule.unit_thresholds is not None:
        for tagged_unit, threshold in rule.unit_thresholds:
            if tagged_unit == unit:
                return threshold
    if unit is None and rule.threshold is not None:
        return rule.threshold
    listed = [u for u, _ in rule.unit_thresholds or ()]
    raise ValueError(
        f"prefix {prefix!r} in source {source!r} has unit tag {unit!r}, but "
        f"the rule defines a default threshold of {rule.threshold!r} and "
        f"per-unit thresholds for {listed!r} -- add the unit tag to "
        f"code_mapping._PREFIX_UNITS or a threshold for that unit."
    )


def _expand_non_loinc(  # noqa: PLR0911
    rule: CanonicalRule, source: str
) -> list[ComponentRule] | None:
    """Expand the non-LOINC-keyed rules; ``None`` when ``rule`` is LOINC-keyed."""
    if isinstance(rule, CodeOccurrenceRule):
        return [rule]
    if isinstance(rule, CanonicalSofaSignal):
        if not sofa_supported(source):
            return []
        return [
            DerivedSofaSignalRule(
                signal=rule.signal,
                threshold=rule.threshold,
                direction=rule.direction,
                source=source,
            )
        ]
    if isinstance(rule, CanonicalUrineRate):
        if not sofa_supported(source):
            return []
        return [
            DerivedUrineRateRule(
                threshold=rule.threshold,
                direction=rule.direction,
                window_hours=rule.window_hours,
                weight_normalized=rule.weight_normalized,
                source=source,
            )
        ]
    if isinstance(rule, CanonicalSepsis3):
        # Needs vasopressor rates and ventilation intervals (SOFA's
        # non-LOINC ingredients) and the microbiology sidecar: MIMIC-IV
        # today; other sources drop the concept with the usual warning.
        if not sofa_supported(source):
            return []
        return [Sepsis3Rule(source=source, sofa_threshold=rule.sofa_threshold)]
    return None


def _expand_rule(rule: CanonicalRule, source: str) -> list[ComponentRule]:
    """Resolve one canonical rule to concrete rules; [] if unresolvable."""
    non_loinc = _expand_non_loinc(rule, source)
    if non_loinc is not None:
        return non_loinc
    if isinstance(rule, LoincGcsTotal):
        components = [
            _loinc_prefixes((loinc,), source)
            for loinc in (rule.eye_loinc, rule.verbal_loinc, rule.motor_loinc)
        ]
        if any(len(prefixes) == 0 for prefixes in components):
            return []
        if any(len(prefixes) > 1 for prefixes in components):
            raise ValueError(
                f"a GCS component LOINC maps to multiple prefixes in "
                f"{source!r}: {components!r} -- ambiguous, refine the mapping."
            )
        return [
            DerivedGcsTotalRule(
                eye_prefix=components[0][0],
                verbal_prefix=components[1][0],
                motor_prefix=components[2][0],
                threshold=rule.threshold,
                direction=rule.direction,
                max_component_gap_minutes=rule.max_component_gap_minutes,
            )
        ]
    assert not isinstance(  # for mypy
        rule,
        (
            CodeOccurrenceRule,
            CanonicalSepsis3,
            CanonicalSofaSignal,
            CanonicalUrineRate,
        ),
    )
    prefixes = _loinc_prefixes(rule.loincs, source)
    if isinstance(rule, LoincThreshold):
        return [
            ConceptRule(prefix, _prefix_threshold(rule, prefix, source), rule.direction)
            for prefix in prefixes
        ]
    if isinstance(rule, LoincSustained):
        # ONE pooled rule over every prefix, never one rule per prefix:
        # per-prefix expansion would apply the recurrence check within
        # each charting route separately (see SustainedRule.extra_prefixes).
        return (
            [
                SustainedRule(
                    prefixes[0],
                    rule.threshold,
                    rule.direction,
                    rule.min_gap_hours,
                    extra_prefixes=tuple(prefixes[1:]),
                )
            ]
            if prefixes
            else []
        )
    if isinstance(rule, LoincBaselineRelative):
        expanded: list[ComponentRule] = []
        for prefix in prefixes:
            delta = rule.delta
            if rule.unit_deltas is not None:
                unit = unit_for(prefix, source=source)
                for tagged_unit, tagged_delta in rule.unit_deltas:
                    if tagged_unit == unit:
                        delta = tagged_delta
                        break
            expanded.append(
                BaselineRelativeRule(
                    prefix,
                    direction=rule.direction,
                    window_hours=rule.window_hours,
                    delta=delta,
                    ratio=rule.ratio,
                )
            )
        return expanded
    raise TypeError(f"unknown canonical rule type: {type(rule)!r}")


def _expand_component(
    component: CanonicalComponent, source: str
) -> CompositeComponent | None:
    """Resolve one composite criterion; None if nothing resolves in ``source``."""
    if isinstance(component, CanonicalAnyOf):
        rules = [r for rule in component.rules for r in _expand_rule(rule, source)]
    else:
        rules = _expand_rule(component, source)
    if not rules:
        return None
    # A criterion is one criterion no matter how many prefixes implement
    # it -- multiple concrete rules must be OR-ed inside AnyOf so they
    # cannot each count toward min_criteria.
    return rules[0] if len(rules) == 1 else AnyOf(rules)


# Task-set versions: which canonical concepts a run supervises. "v1" is the
# 15-concept registry every run before Aug 23 2026 trained with (its
# checkpoints hard-code that count); "v2" adds sepsis3. A run records its
# task_set in config.json so evaluation rebuilds exactly its concept list.
TASK_SETS: dict[str, tuple[str, ...]] = {
    "v1": (
        "tachycardia",
        "bradycardia",
        "hypotension",
        "hypertension",
        "hypoxia",
        "fever",
        "hypothermia",
        "elevated_lactate",
        "sustained_tachypnea",
        "acute_kidney_injury",
        "aki_stage_2",
        "aki_stage_3",
        "sirs",
        "qsofa",
        "on_vasopressors",
    ),
}
TASK_SETS["v2"] = TASK_SETS["v1"] + ("sepsis3",)
# "v3" adds structurally-derived electrolyte/metabolic/hematologic concepts
# (Track B item 11) -- see their CANONICAL_CONCEPTS entries for thresholds
# and sources. v1/v2 are untouched by this addition (concepts_for_source
# only ever reads the names listed for the task_set actually requested).
TASK_SETS["v3"] = TASK_SETS["v2"] + (
    "hyperkalemia",
    "hypokalemia",
    "hyponatremia",
    "hypernatremia",
    "hypoglycemia",
    "hyperglycemia",
    "anemia",
    "thrombocytopenia",
    "coagulopathy",
    "metabolic_acidosis",
    "shock",
    # Derived-signal concepts: only on sources with SOFA ingredients (the
    # LOINC layer drops them elsewhere, like sepsis3).
    "hypoxemic_respiratory_failure",
    "oliguria",
)
DEFAULT_TASK_SET = "v1"


def concepts_for_source(
    source: str = "mimic_iv", *, task_set: str = DEFAULT_TASK_SET
) -> list[AnyConceptDefinition]:
    """Expand the canonical registry to one source's concrete definitions.

    ``task_set`` selects which canonical concepts are included (see
    :data:`TASK_SETS`); unknown names raise.

    Criteria whose LOINCs have no mapping in ``source`` are dropped with
    a warning; a composite that retains fewer criteria than its
    ``min_criteria``, or a simple concept that retains no rules at all,
    is dropped entirely. The result is therefore per-source both in its
    prefixes and, potentially, in its length -- always take concept
    names/count from the same expansion the model was trained with.
    """
    out: list[AnyConceptDefinition] = []
    if task_set not in TASK_SETS:
        raise ValueError(f"unknown task_set {task_set!r}; known: {sorted(TASK_SETS)}")
    wanted = set(TASK_SETS[task_set])
    for canonical in CANONICAL_CONCEPTS:
        if canonical.name not in wanted:
            continue
        if isinstance(canonical, CanonicalComposite):
            components = []
            for component in canonical.components:
                expanded = _expand_component(component, source)
                if expanded is None:
                    logger.warning(
                        "[concepts] source %r: dropping a criterion of %r -- "
                        "no code mapping",
                        source,
                        canonical.name,
                    )
                    continue
                components.append(expanded)
            if len(components) < canonical.min_criteria:
                logger.warning(
                    "[concepts] source %r: dropping concept %r -- only %d of "
                    "its criteria resolve (min_criteria=%d)",
                    source,
                    canonical.name,
                    len(components),
                    canonical.min_criteria,
                )
                continue
            out.append(
                CompositeConceptDefinition(
                    canonical.name,
                    components,
                    canonical.min_criteria,
                    canonical.description,
                )
            )
        else:
            rules = [r for rule in canonical.rules for r in _expand_rule(rule, source)]
            if not rules:
                logger.warning(
                    "[concepts] source %r: dropping concept %r -- no code mapping",
                    source,
                    canonical.name,
                )
                continue
            out.append(ConceptDefinition(canonical.name, rules, canonical.description))
    return out


# LOINC shorthands, named for readability of the registry below. See
# odyssey/data/code_mapping.py for what each maps to per source.
_HR = ("8867-4",)  # heart rate
_RR = ("9279-1",)  # respiratory rate
_SPO2 = ("59408-5",)  # O2 saturation, pulse oximetry
_SBP = ("76534-7", "8480-6")  # systolic BP: non-invasive cuff OR arterial line
_TEMP = ("8310-5",)  # body temperature (unit-split: F or C by source/itemid)
_LACTATE = ("32693-4",)
_CREATININE = ("2160-0",)
_WBC = ("6690-2",)

# v3 additions (Track B item 11): every LOINC below is already mapped for
# mimic_iv and eicu in odyssey/data/code_mapping.py (none needed a new
# mapping-table entry) and none of them is unit-split there (no entry in
# code_mapping._PREFIX_UNITS), so plain single-``threshold`` LoincThreshold
# rules apply directly, same as _LACTATE/_CREATININE/_WBC above.
_POTASSIUM = ("2823-3",)
_SODIUM = ("2951-2",)
_GLUCOSE = ("2345-7",)
_BICARBONATE = ("1963-8",)
_HEMOGLOBIN = ("718-7",)
_PLATELETS = ("777-3",)
_INR = ("6301-6",)
_PH = ("11558-4",)
_MAP = ("76536-2", "8478-0")  # mean arterial pressure: non-invasive OR arterial

_TEMP_HIGH = (("F", 100.4), ("C", 38.0))
_TEMP_LOW = (("F", 96.8), ("C", 36.0))
# SI cutoffs for the sources that chart glucose in mmol/L (1 mmol/L =
# 18.016 mg/dL) and hemoglobin in g/L; the untagged US prefixes keep the
# conventional-unit defaults on the rules themselves.
_GLUCOSE_LOW = (("mmol/L", 3.9),)
_GLUCOSE_HIGH = (("mmol/L", 13.9),)
_HEMOGLOBIN_LOW = (("g/L", 70.0),)
# KDIGO stage-3 absolute creatinine, 4.0 mg/dL, in the umol/L GEMINI charts
# (x 88.42). Before per-unit thresholds this rule compared umol/L values
# against 4.0 and fired on essentially every creatinine result there.
_CREATININE_STAGE3 = (("umol/L", 353.7),)


CANONICAL_CONCEPTS: list[AnyCanonicalConcept] = [
    # -- Simple instantaneous vital-sign/lab concepts (v1, kept): no known
    # over-triggering problem, so an instantaneous threshold remains
    # appropriate -- see the module docstring for which v1 concepts were
    # upgraded instead (tachypnea, acute_kidney_injury).
    CanonicalConcept(
        "tachycardia",
        (LoincThreshold(_HR, "above", 100.0),),
        "Heart rate > 100 bpm.",
    ),
    CanonicalConcept(
        "bradycardia",
        (LoincThreshold(_HR, "below", 60.0),),
        "Heart rate < 60 bpm.",
    ),
    CanonicalConcept(
        "hypotension",
        (LoincThreshold(_SBP, "below", 90.0),),
        "Systolic blood pressure < 90 mmHg.",
    ),
    CanonicalConcept(
        "hypertension",
        (LoincThreshold(_SBP, "above", 140.0),),
        "Systolic blood pressure > 140 mmHg.",
    ),
    CanonicalConcept(
        "hypoxia",
        (LoincThreshold(_SPO2, "below", 92.0),),
        "SpO2 < 92%.",
    ),
    CanonicalConcept(
        "fever",
        (LoincThreshold(_TEMP, "above", unit_thresholds=_TEMP_HIGH),),
        "Temperature > 100.4F / 38.0C.",
    ),
    CanonicalConcept(
        "hypothermia",
        (LoincThreshold(_TEMP, "below", unit_thresholds=_TEMP_LOW),),
        "Temperature < 96.8F / 36.0C.",
    ),
    CanonicalConcept(
        "elevated_lactate",
        (LoincThreshold(_LACTATE, "above", 2.0),),
        "Serum lactate > 2.0 mmol/L.",
    ),
    # -- v2: upgraded from v1's single-instantaneous-threshold proxies.
    CanonicalConcept(
        "sustained_tachypnea",
        (LoincSustained(_RR, 20.0, "above", min_gap_hours=1.0),),
        "Respiratory rate > 20 breaths/min, recurring at least 1 hour apart -- "
        "replaces v1's single-instantaneous-reading 'tachypnea', which "
        "triggered for 96.5% of subjects with respiratory-rate data in the "
        "real MIMIC-IV extraction (too loose to be a useful signal; a single "
        "transient spike, e.g. during one stressful procedure, was "
        "indistinguishable from true sustained tachypnea).",
    ),
    CanonicalConcept(
        "acute_kidney_injury",
        (
            LoincBaselineRelative(
                _CREATININE,
                delta=0.3,
                unit_deltas=(("umol/L", 26.5),),
                direction="above",
                window_hours=48.0,
            ),
            LoincBaselineRelative(
                _CREATININE, ratio=1.5, direction="above", window_hours=168.0
            ),
            # KDIGO's urine leg: < 0.5 mL/kg/h for 6-12h. Evaluated at the
            # 6h lower bound, since every rule here is a single-instant
            # check rather than a multi-criteria interval test: a window
            # that stays under the rate past 12h has already triggered at
            # 6h, and Stage 2 is the >= 12h escalation of the same rate.
            CanonicalUrineRate(threshold=0.5, window_hours=6.0),
        ),
        "KDIGO AKI Stage 1 (any trigger): serum creatinine rose by >= 0.3 "
        "mg/dL within 48 hours, OR rose to >= 1.5x an earlier reading within "
        "7 days (168h), OR urine output under 0.5 mL/kg/h over a trailing 6 "
        "hours. Replaces v1's 'creatinine > 1.5 mg/dL' single-value proxy, "
        "which ignored a patient's own baseline. See aki_stage_2 and "
        "aki_stage_3 for higher severity. The urine leg needs a charted body "
        "weight at or before the window (daily weight preferred, admission "
        "weight as the early-stay fallback) and is simply not scored without "
        "one, so it adds triggers without ever adding a silent negative.",
    ),
    CanonicalConcept(
        "aki_stage_2",
        (
            LoincBaselineRelative(
                _CREATININE, ratio=2.0, direction="above", window_hours=168.0
            ),
            CanonicalUrineRate(threshold=0.5, window_hours=12.0),
        ),
        "KDIGO AKI Stage 2 (either trigger): serum creatinine rose to >= 2.0x "
        "an earlier reading within 7 days, OR urine output under 0.5 mL/kg/h "
        "over a trailing 12 hours (the same rate as Stage 1, sustained twice "
        "as long).",
    ),
    CanonicalConcept(
        "aki_stage_3",
        (
            LoincBaselineRelative(
                _CREATININE, ratio=3.0, direction="above", window_hours=168.0
            ),
            # KDIGO: >= 4.0, inclusive
            LoincThreshold(
                _CREATININE, "at_or_above", 4.0, unit_thresholds=_CREATININE_STAGE3
            ),
            # RRT initiation is an automatic Stage 3 in KDIGO, whatever
            # creatinine and urine output say (mimic-code's rrt.sql item ids;
            # see RRT_CODE_PATTERN). First occurrence only, which is what
            # CodeOccurrenceRule already reports.
            CodeOccurrenceRule(RRT_CODE_PATTERN, observed_families=("PROCEDURE",)),
            CanonicalUrineRate(threshold=0.3, window_hours=24.0),
            # Anuria: 0 mL over 12h. Absolute volume, not a rate, so it needs
            # no body weight (0 mL is 0 mL at any weight) and stays
            # assessable for the many keys with no weight charted.
            CanonicalUrineRate(
                threshold=0.0,
                window_hours=12.0,
                direction="at_or_below",
                weight_normalized=False,
            ),
        ),
        "KDIGO AKI Stage 3 (any trigger): serum creatinine rose to >= 3.0x an "
        "earlier reading within 7 days, OR any reading >= 4.0 mg/dL, OR "
        "renal-replacement therapy was initiated (hemodialysis, CRRT, CVVHD, "
        "CVVHDF, SCUF or peritoneal dialysis -- an automatic Stage 3 in "
        "KDIGO, independent of creatinine and urine output), OR urine output "
        "under 0.3 mL/kg/h over a trailing 24 hours, OR anuria (0 mL) over a "
        "trailing 12 hours. The rate leg needs a charted body weight and is "
        "not scored without one; the anuria and RRT legs need no weight.",
    ),
    CanonicalComposite(
        "sirs",
        components=(
            # criterion 1: temp > 38C/100.4F or < 36C/96.8F
            CanonicalAnyOf(
                (
                    LoincThreshold(_TEMP, "above", unit_thresholds=_TEMP_HIGH),
                    LoincThreshold(_TEMP, "below", unit_thresholds=_TEMP_LOW),
                )
            ),
            LoincThreshold(_HR, "above", 90.0),  # criterion 2: HR > 90
            LoincThreshold(_RR, "above", 20.0),  # criterion 3: RR > 20
            # criterion 4: WBC > 12k or < 4k per uL
            CanonicalAnyOf(
                (
                    LoincThreshold(_WBC, "above", 12.0),
                    LoincThreshold(_WBC, "below", 4.0),
                )
            ),
        ),
        min_criteria=2,
        description=(
            "SIRS (Systemic Inflammatory Response Syndrome): >= 2 of "
            "{abnormal temperature, HR > 90, RR > 20, abnormal WBC}. The "
            "bands-percentage alternative for the WBC criterion (>10% bands, "
            "an option even with a normal WBC count) is not included -- no "
            "verified MIMIC-IV itemid for band percentage has been found yet."
        ),
    ),
    CanonicalComposite(
        "qsofa",
        components=(
            LoincThreshold(_RR, "at_or_above", 22.0),  # RR >= 22
            LoincThreshold(_SBP, "at_or_below", 100.0),  # SBP <= 100
            # GCS < 15 (any drop from fully alert)
            LoincGcsTotal(threshold=15.0, direction="below"),
        ),
        min_criteria=2,
        description=(
            "qSOFA (quick Sequential Organ Failure Assessment), a bedside "
            "sepsis-screening score: >= 2 of {RR >= 22, SBP <= 100 mmHg, "
            "GCS < 15}."
        ),
    ),
    # entry 06/07 decision (f): the first occurrence-keyed concept. The
    # drug-name regex reaches both MIMIC-IV medication codes (drug name
    # embedded in the code string) and eICU infusion events (drug name in
    # the code since spec v2, and in text_value via match_text_value for
    # v1 extractions with a bare INFUSION_DRUG code).
    # "epinephrine" also matching "norepinephrine" is deliberate; both are
    # vasopressors and Rust regex has no lookbehind. Dobutamine is excluded
    # on purpose: an inotrope, not a vasopressor.
    CanonicalConcept(
        "on_vasopressors",
        (
            CodeOccurrenceRule(
                r"norepinephrine|levophed|epinephrine|vasopressin|phenylephrine"
                r"|neo-?synephrine|dopamine|angiotensin",
                match_text_value=True,
            ),
        ),
        "Received at least one vasopressor (norepinephrine, epinephrine, "
        "vasopressin, phenylephrine, dopamine, or angiotensin II) -- the "
        "canonical shock/deterioration marker, derived from medication and "
        "infusion events rather than a numeric threshold. 'Observed' means "
        "the subject has any medication/infusion data at all, so an absent "
        "match is a genuine negative, not missingness.",
    ),
    CanonicalConcept(
        "hypoxemic_respiratory_failure",
        (CanonicalSofaSignal(signal="pf_ratio", threshold=300.0, direction="below"),),
        "PaO2/FiO2 < 300 mmHg (the Berlin definition's mild-ARDS oxygenation "
        "threshold, and SOFA's respiration level 2), each arterial gas paired "
        "with the most recent FiO2 within 4 hours. Ventilation status is not "
        "required here (unlike the Berlin definition's PEEP criterion), so this "
        "is an oxygenation-failure marker, not an ARDS diagnosis. 'Observed' "
        "means a gas with a pairable FiO2 exists.",
    ),
    CanonicalConcept(
        "oliguria",
        (CanonicalSofaSignal(signal="urine_24h", threshold=500.0, direction="below"),),
        "Under 500 mL of urine over the trailing 24 hours (SOFA's renal "
        "level 3), scored only once a key has 24 hours of record so a partial "
        "window cannot read as oliguria. Absolute volume, not KDIGO's "
        "weight-normalized mL/kg/h: per-key weight is not reliably extracted.",
    ),
    CanonicalConcept(
        "sepsis3",
        (CanonicalSepsis3(sofa_threshold=2),),
        "Sepsis-3 onset (Singer et al. 2016; mimic-code operationalization): "
        "suspected infection (culture specimen and systemic antibiotic start "
        "within 72h after / 24h before each other; suspicion time = the "
        "earlier) with a SOFA total >= 2 between 48h before and 24h after "
        "suspicion; onset = first instant both hold. Cultures come from the "
        "microbiology sidecar (label-only; no model sees them); SOFA from "
        "odyssey.data.sofa. 'Observed' = the visit has at least one SOFA "
        "component reading, so the score could have been assessed. MIMIC-IV "
        "only until another source has SOFA ingredients and a sidecar.",
    ),
    # -- v3 additions (Track B item 11, concept-set widening): structurally-
    # derived electrolyte/metabolic/hematologic abnormalities, all plain
    # instantaneous thresholds (no known over-triggering problem reported
    # for any of these in ICU practice, unlike v1's tachypnea) -- no new
    # rule machinery. None of these is a mimic-code concept-library query
    # (mimic-code's queries are scores/staging systems -- SOFA, SIRS,
    # sepsis3, KDIGO -- not raw lab-abnormality flags); each threshold
    # below is a standard, independently-citable clinical cutoff, sourced
    # in its own description rather than attributed to mimic-code.
    CanonicalConcept(
        "hyperkalemia",
        (LoincThreshold(_POTASSIUM, "above", 5.5),),
        "Serum/plasma potassium > 5.5 mEq/L -- a standard severe-"
        "hyperkalemia threshold (cardiac-risk cutoff used across ICU "
        "electrolyte-repletion protocols).",
    ),
    CanonicalConcept(
        "hypokalemia",
        (LoincThreshold(_POTASSIUM, "below", 3.0),),
        "Serum/plasma potassium < 3.0 mEq/L -- a standard severe-"
        "hypokalemia threshold.",
    ),
    CanonicalConcept(
        "hyponatremia",
        (LoincThreshold(_SODIUM, "below", 130.0),),
        "Serum/plasma sodium < 130 mEq/L -- a standard moderate/severe-"
        "hyponatremia threshold.",
    ),
    CanonicalConcept(
        "hypernatremia",
        (LoincThreshold(_SODIUM, "above", 150.0),),
        "Serum/plasma sodium > 150 mEq/L -- a standard hypernatremia threshold.",
    ),
    CanonicalConcept(
        "hypoglycemia",
        (LoincThreshold(_GLUCOSE, "below", 70.0, unit_thresholds=_GLUCOSE_LOW),),
        "Serum/plasma glucose < 70 mg/dL (3.9 mmol/L) -- the ADA (American "
        "Diabetes Association) hypoglycemia threshold.",
    ),
    CanonicalConcept(
        "hyperglycemia",
        (LoincThreshold(_GLUCOSE, "above", 250.0, unit_thresholds=_GLUCOSE_HIGH),),
        "Serum/plasma glucose > 250 mg/dL (13.9 mmol/L) -- a standard marked-"
        "hyperglycemia threshold, well above routine stress-hyperglycemia "
        "noise and below DKA-range values.",
    ),
    CanonicalConcept(
        "anemia",
        (LoincThreshold(_HEMOGLOBIN, "below", 7.0, unit_thresholds=_HEMOGLOBIN_LOW),),
        "Hemoglobin < 7 g/dL (70 g/L) -- the restrictive-transfusion-strategy "
        "trigger (TRICC/TRISS-style ICU threshold), the more specific of "
        "the two commonly-cited cutoffs (7 vs. 8 g/dL).",
    ),
    CanonicalConcept(
        "thrombocytopenia",
        (LoincThreshold(_PLATELETS, "below", 100.0),),
        "Platelet count < 100 x10^9/L -- the SOFA coagulation-component "
        "Stage-2 threshold (odyssey.data.sofa), a standard clinically-"
        "significant cutoff independent of the SOFA score itself.",
    ),
    CanonicalConcept(
        "coagulopathy",
        (LoincThreshold(_INR, "above", 1.5),),
        "INR > 1.5 -- a standard clinically-significant coagulopathy "
        "threshold (e.g. the King's College hepatic-coagulopathy cutoff).",
    ),
    CanonicalConcept(
        "metabolic_acidosis",
        (
            LoincThreshold(_BICARBONATE, "below", 18.0),
            LoincThreshold(_PH, "below", 7.3),
        ),
        "Serum bicarbonate < 18 mEq/L OR arterial/venous pH < 7.3 -- "
        "either criterion alone triggers (a plain concept's rules are "
        "OR'd, see the module docstring); both are standard metabolic-"
        "acidosis thresholds, and using both catches partially-"
        "compensated cases a single measure would miss.",
    ),
    CanonicalConcept(
        "shock",
        (LoincSustained(_MAP, 65.0, "below", min_gap_hours=1.0),),
        "Mean arterial pressure < 65 mmHg, recurring (not a single "
        "transient reading, same SustainedRule operationalization as "
        "sustained_tachypnea) -- the standard MAP-based shock/hemodynamic-"
        "instability threshold. Deliberately NOT also OR'd with "
        "vasopressor administration (the concept's other common trigger "
        "in clinical use): that would make it near-redundant with the "
        "existing on_vasopressors concept (almost every vasopressor start "
        "co-occurs with it) while adding little 'was this patient "
        "actually hypotensive' signal of its own -- see the v3 report for "
        "the overlap check this dropped.",
    ),
]


# The MIMIC-IV expansion, kept as the module-level default registry:
# every existing entry point (training config default source, tests,
# report tooling) reads this exactly as before the canonical layer
# existed. Other sources call concepts_for_source(...) directly.
CONCEPTS: list[AnyConceptDefinition] = concepts_for_source("mimic_iv")


# subject (or visit key) -> the earliest time a rule/concept was satisfied.
FirstTimes = dict[int, datetime]


def _first_times(frame: pl.DataFrame, subject_id_col: str, time_col: str) -> FirstTimes:
    """Earliest ``time_col`` per subject in ``frame`` (a set of qualifying rows)."""
    if frame.height == 0:
        return {}
    firsts = frame.group_by(subject_id_col).agg(pl.col(time_col).min().alias("_t"))
    return dict(zip(firsts[subject_id_col].to_list(), firsts["_t"].to_list()))


def _merge_min(into: FirstTimes, other: FirstTimes) -> None:
    """Fold ``other`` into ``into``, keeping the earlier time per key."""
    for sid, t in other.items():
        prev = into.get(sid)
        if prev is None or t < prev:
            into[sid] = t


def _instantaneous_ids(
    events: pl.DataFrame,
    rule: ConceptRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    matched = events.filter(
        pl.col(code_col).str.starts_with(rule.code_prefix)
        & pl.col(value_col).is_not_null()
    )
    observed = set(matched[subject_id_col].to_list())
    comparison = _threshold_expr(pl.col(value_col), rule.threshold, rule.direction)
    triggered = _first_times(matched.filter(comparison), subject_id_col, time_col)
    return observed, triggered


def _sustained_ids(
    events: pl.DataFrame,
    rule: SustainedRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    prefix_match = pl.col(code_col).str.starts_with(rule.code_prefix)
    for extra in rule.extra_prefixes:
        prefix_match = prefix_match | pl.col(code_col).str.starts_with(extra)
    matched = events.filter(prefix_match & pl.col(value_col).is_not_null())
    observed = set(matched[subject_id_col].to_list())
    comparison = _threshold_expr(pl.col(value_col), rule.threshold, rule.direction)
    qualifying = matched.filter(comparison)
    if qualifying.height == 0:
        return observed, {}

    # The rule is satisfied at the earliest qualifying reading that is at
    # least min_gap after the subject's first qualifying reading; the
    # earliest-vs-latest span check is the same criterion evaluated at
    # the last reading.
    min_gap = timedelta(hours=rule.min_gap_hours)
    first_qualifying = pl.col(time_col).min().over(subject_id_col)
    satisfied = qualifying.filter(pl.col(time_col) - first_qualifying >= min_gap)
    triggered = _first_times(satisfied, subject_id_col, time_col)
    return observed, triggered


def _baseline_relative_ids(
    events: pl.DataFrame,
    rule: BaselineRelativeRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
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
        return observed, {}

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

    triggered = _first_times(
        matched.filter(pl.col("_baseline").is_not_null() & comparison),
        subject_id_col,
        time_col,
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
) -> tuple[set[int], FirstTimes]:
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
        return observed, {}

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
            verbal.rename({value_col: f"{value_col}_verbal"}).with_columns(
                pl.col(time_col).alias("_t_verbal")
            ),
            on=time_col,
            by=subject_id_col,
            strategy="nearest",
            tolerance=tolerance,
        )
        paired = paired.filter(pl.col(f"{value_col}_verbal").is_not_null()).sort(
            [subject_id_col, time_col]
        )
        paired = paired.join_asof(
            motor.rename({value_col: f"{value_col}_motor"}).with_columns(
                pl.col(time_col).alias("_t_motor")
            ),
            on=time_col,
            by=subject_id_col,
            strategy="nearest",
            tolerance=tolerance,
        )
    paired = paired.filter(pl.col(f"{value_col}_motor").is_not_null())
    if paired.height == 0:
        return observed, {}

    total = (
        pl.col(value_col) + pl.col(f"{value_col}_verbal") + pl.col(f"{value_col}_motor")
    )
    comparison = _threshold_expr(total, rule.threshold, rule.direction)
    # Stamp the trigger at the LAST of the three paired component times,
    # not the eye time: "nearest" pairing may pull a verbal/motor reading
    # charted up to max_component_gap_minutes AFTER the eye reading, and a
    # total is only computable once all three components exist. Stamping
    # at the eye time put up to that many minutes of future information
    # into first_times (a small but real running-label leak).
    paired = paired.with_columns(
        pl.max_horizontal(time_col, "_t_verbal", "_t_motor").alias("_t_known")
    )
    triggered = _first_times(paired.filter(comparison), subject_id_col, "_t_known")
    return observed, triggered


def _component_ids(  # noqa: PLR0911
    events: pl.DataFrame,
    rule: ComponentRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    if isinstance(rule, ConceptRule):
        return _instantaneous_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
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
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            time_col=time_col,
        )
    if isinstance(rule, DerivedSofaSignalRule):
        return _sofa_signal_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    if isinstance(rule, DerivedUrineRateRule):
        return _urine_rate_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    if isinstance(rule, Sepsis3Rule):
        return _sepsis3_ids(
            events,
            rule,
            subject_id_col=subject_id_col,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    raise TypeError(f"unknown component rule type: {type(rule)!r}")


def _derived_reading_ids(
    readings: pl.DataFrame,
    threshold: float,
    direction: Direction,
    *,
    subject_id_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    """Observed keys and first-trigger times of a derived (key, time, value) frame.

    Observed = the keys the derivation could actually be evaluated for
    (it emits no row where a needed ingredient is missing: no pairable
    FiO2, no full trailing window, no body weight), so an absent key is
    "not assessable", never a silent negative.
    """
    observed = set(readings[subject_id_col].to_list())
    if readings.height == 0:
        return observed, {}
    fired = readings.filter(_threshold_expr(pl.col("value"), threshold, direction))
    return observed, _first_times(fired, subject_id_col, time_col)


def _sofa_signal_ids(
    events: pl.DataFrame,
    rule: DerivedSofaSignalRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    """Observed keys and first-trigger times for a SOFA-derived signal."""
    readings = (pf_ratio_readings if rule.signal == "pf_ratio" else urine_output_24h)(
        events,
        source=rule.source,
        key=subject_id_col,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    return _derived_reading_ids(
        readings,
        rule.threshold,
        rule.direction,
        subject_id_col=subject_id_col,
        time_col=time_col,
    )


def _urine_rate_ids(
    events: pl.DataFrame,
    rule: DerivedUrineRateRule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    """Observed keys and first-trigger times for KDIGO's urine-output leg."""
    readings = urine_output_rate(
        events,
        source=rule.source,
        key=subject_id_col,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
        window_hours=rule.window_hours,
        weight_normalized=rule.weight_normalized,
    )
    return _derived_reading_ids(
        readings,
        rule.threshold,
        rule.direction,
        subject_id_col=subject_id_col,
        time_col=time_col,
    )


_SIDECAR_WARNED: set[str] = set()


def _attribute_sidecar_rows(
    rows: pl.DataFrame, spans: pl.DataFrame, *, key: str, time_alias: str
) -> pl.DataFrame:
    """Attach sidecar rows (subject_id, hadm_id, time, ...) to label keys.

    A row belongs to a key when its ``hadm_id`` equals the key's visit, or,
    when the row has no ``hadm_id``, when its time falls inside the key's
    event span. ``spans`` has ``key, _subject, _visit, _t0, _t1``. Extra
    columns of ``rows`` are carried through; ``time`` is renamed to
    ``time_alias``.
    """
    joined = rows.rename(
        {"subject_id": "_subject", "hadm_id": "_rhadm", "time": time_alias}
    ).join(
        spans.select(key, "_subject", "_visit", "_t0", "_t1"),
        on="_subject",
        how="inner",
    )
    return joined.filter(
        (
            pl.col("_rhadm").is_not_null()
            & (pl.col("_rhadm") == pl.col("_visit").cast(pl.Int64))
        )
        | (
            pl.col("_rhadm").is_null()
            & (pl.col(time_alias) >= pl.col("_t0"))
            & (pl.col(time_alias) <= pl.col("_t1"))
        )
    ).drop("_subject", "_rhadm", "_visit", "_t0", "_t1")


def _sepsis3_ids(  # noqa: PLR0915
    events: pl.DataFrame,
    rule: Sepsis3Rule,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    """Sepsis-3 first-onset per key; see :class:`Sepsis3Rule`."""
    key = subject_id_col
    cultures = active_sidecar(MICROBIOLOGY)
    if cultures is None:
        if MICROBIOLOGY not in _SIDECAR_WARNED:
            _SIDECAR_WARNED.add(MICROBIOLOGY)
            logger.warning(
                "[concepts] sepsis3: no active %r sidecar (see "
                "odyssey.data.sidecars.activate_sidecars) -- concept is "
                "unobserved everywhere in this call",
                MICROBIOLOGY,
            )
        return set(), {}

    timed = events.filter(pl.col(time_col).is_not_null())
    # --- per-key span and identity (subject + visit), for culture attribution
    has_subject = rule.subject_col in timed.columns and rule.subject_col != key
    has_visit = rule.visit_col in timed.columns
    aggs = [pl.col(time_col).min().alias("_t0"), pl.col(time_col).max().alias("_t1")]
    if has_subject:
        aggs.append(pl.col(rule.subject_col).first().alias("_subject"))
    if has_visit:
        aggs.append(pl.col(rule.visit_col).first().alias("_visit"))
    spans = timed.group_by(key).agg(aggs)
    if not has_subject:
        spans = spans.with_columns(pl.col(key).alias("_subject"))
    if not has_visit:
        spans = spans.with_columns(pl.lit(None, dtype=pl.Int64).alias("_visit"))

    # --- cultures attributed to keys: same hadm_id, or (no hadm_id) inside the span
    cult = _attribute_sidecar_rows(
        cultures.select(
            "subject_id",
            pl.col("hadm_id").cast(pl.Int64),
            pl.col("time").cast(timed.schema[time_col]),
        ),
        spans,
        key=key,
        time_alias="_ctime",
    ).select(key, "_ctime")

    # --- systemic antibiotic starts. Preferred source: the antibiotic_orders
    # sidecar (prescription ORDERS, mimic-code's anchor; hadm_id on every
    # row). Fallback: the tokenized record's pharmacy START / eMAR
    # Administered rows -- which under-call suspicion on MIMIC-IV (pharmacy
    # START rows carry no hadm_id in the standard extraction and eMAR covers
    # only part of the years; research journal entry 43).
    orders = active_sidecar(ANTIBIOTIC_ORDERS)
    if orders is not None:
        abx = _attribute_sidecar_rows(
            orders.select(
                "subject_id",
                pl.col("hadm_id").cast(pl.Int64),
                pl.col("time").cast(timed.schema[time_col]),
                pl.col("route"),
            ),
            spans,
            key=key,
            time_alias="_atime",
        )
        abx = abx.filter(
            ~pl.col("route").fill_null("").str.contains("(?i)" + rule.route_exclude)
        ).select(key, "_atime")
    else:
        starts = timed.filter(
            pl.col(code_col).str.starts_with("MEDICATION//")
            & pl.col(code_col).str.contains(rule.antibiotic_event_pattern)
            & pl.col(code_col).str.contains("(?i)" + rule.antibiotic_pattern)
        )
        if rule.route_col in starts.columns:
            starts = starts.filter(
                ~pl.col(rule.route_col)
                .fill_null("")
                .str.contains("(?i)" + rule.route_exclude)
            )
        abx = starts.select(key, pl.col(time_col).alias("_atime"))

    # observed = the SOFA score could be assessed at all for this key,
    # whether or not infection was ever suspected.
    sofa_obs_keys: set[int] = set(
        assessable_keys(  # type: ignore[arg-type]
            timed,
            source=rule.source,
            key=key,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )
    )
    if cult.height == 0 or abx.height == 0:
        return sofa_obs_keys, {}

    # --- suspicion time per key: earliest min(culture, antibiotic) over valid pairs
    pairs = cult.join(abx, on=key, how="inner").filter(
        (
            pl.col("_atime")
            >= pl.col("_ctime") - pl.duration(hours=rule.culture_after_hours)
        )
        & (
            pl.col("_atime")
            <= pl.col("_ctime") + pl.duration(hours=rule.antibiotic_after_hours)
        )
    )
    if pairs.height == 0:
        return sofa_obs_keys, {}
    # _si: mimic-code's suspicion time, the EARLIER element of the earliest
    # valid pair -- it anchors the SOFA window below, exactly as published.
    # _confirm: the earliest instant a valid pair is COMPLETE (the later
    # element, minimized over pairs). Suspicion is only knowable once both
    # elements exist: a culture on day 1 plus an antibiotic on day 3 is a
    # valid pair, but on day 1 nothing distinguishes that culture from any
    # never-confirmed one. Folding _confirm into the onset (below) keeps
    # the first-trigger time at an instant the label was determinable,
    # which is what running_labels.py's "true from first-trigger onward"
    # contract needs -- without it, RandInt/CEM interventions were feeding
    # up-to-72h-of-future sepsis3 "ground truth" into training.
    suspicion = (
        pairs.with_columns(
            pl.min_horizontal("_ctime", "_atime").alias("_si"),
            pl.max_horizontal("_ctime", "_atime").alias("_pair_done"),
        )
        .group_by(key)
        .agg(pl.col("_si").min(), pl.col("_pair_done").min().alias("_confirm"))
    )

    # --- SOFA on the suspected keys only, suspicion instants added to the grid
    sus_events = timed.join(suspicion.select(key), on=key, how="semi")
    sofa = sofa_timeseries(
        sus_events,
        source=rule.source,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
        grid_times=suspicion.select(key, pl.col("_si").alias(time_col)),
    )
    crossed = (
        sofa.join(suspicion, on=key, how="inner")
        .filter(
            (pl.col("sofa") >= rule.sofa_threshold)
            & (
                pl.col(time_col)
                >= pl.col("_si") - pl.duration(hours=rule.sofa_before_hours)
            )
            & (
                pl.col(time_col)
                <= pl.col("_si") + pl.duration(hours=rule.sofa_after_hours)
            )
        )
        .group_by(key)
        .agg(
            pl.col(time_col).min().alias("_tsofa"),
            pl.col("_si").first(),
            pl.col("_confirm").first(),
        )
        # Onset = the first instant every ingredient of the label is in
        # the record: the SOFA crossing, the suspicion instant, AND the
        # pair's completion (_confirm, see above). mimic-code's own onset
        # is max(_tsofa, _si), which can precede _confirm by up to
        # antibiotic_after_hours -- correct as a retrospective epidemiology
        # timestamp, but a leak when used as a running-label anchor. The
        # triggered SET is unchanged by this; only the stamp moves later.
        .with_columns(pl.max_horizontal("_tsofa", "_si", "_confirm").alias("_onset"))
    )
    onsets: FirstTimes = dict(zip(crossed[key].to_list(), crossed["_onset"].to_list()))
    return sofa_obs_keys, onsets


def _occurrence_ids(
    events: pl.DataFrame,
    rule: CodeOccurrenceRule,
    *,
    subject_id_col: str,
    code_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
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
    triggered = _first_times(events.filter(match), subject_id_col, time_col)
    return observed | set(triggered), triggered


def _composite_component_ids(
    events: pl.DataFrame,
    component: CompositeComponent,
    *,
    subject_id_col: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> tuple[set[int], FirstTimes]:
    if isinstance(component, AnyOf):
        observed: set[int] = set()
        triggered: FirstTimes = {}
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
            _merge_min(triggered, trig)
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
    include_first_time: bool = False,
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
    docstring), and ``{name}_observed`` is 1 only if at least
    ``min_criteria`` components each had at least one matching
    measurement -- with fewer, the composite could never have reached
    ``min_criteria`` no matter what the readings said, so the subject is
    structurally unassessable, not a genuine negative.

    With ``include_first_time``, a ``{name}_first_time`` Datetime column
    is added: the earliest time at which the concept became satisfied
    (null where the label is 0). For a simple concept that is the
    earliest rule firing; for a composite it is the moment the
    ``min_criteria``-th criterion first fired -- the first instant a
    clinician tallying the score would have called it met. This is what
    lets a whole-visit label be turned into a valid *running* label at
    any position: "true from ``first_time`` on, false before".
    """
    subject_ids = events.select(subject_id_col).unique()
    out = subject_ids

    for concept in concepts:
        first_times: FirstTimes = {}
        if isinstance(concept, CompositeConceptDefinition):
            if not concept.components:
                raise ValueError(f"concept {concept.name!r} has no components defined")
            observed_counts: dict[int, int] = {}
            criteria_times: dict[int, list[datetime]] = {}
            for component in concept.components:
                obs, trig = _composite_component_ids(
                    events,
                    component,
                    subject_id_col=subject_id_col,
                    code_col=code_col,
                    value_col=value_col,
                    time_col=time_col,
                )
                for sid in obs:
                    observed_counts[sid] = observed_counts.get(sid, 0) + 1
                for sid, t in trig.items():
                    criteria_times.setdefault(sid, []).append(t)
            # Observed = at least min_criteria components were measurable.
            # A subject observed for only one SIRS criterion can never
            # reach min_criteria=2, so "any one component observed" (the
            # pre-2026-08-30 mask) supervised structurally-unassessable
            # subjects as negatives -- label noise, not a real negative.
            observed_ids: set[int] = {
                sid
                for sid, count in observed_counts.items()
                if count >= concept.min_criteria
            }
            for sid, times in criteria_times.items():
                if len(times) >= concept.min_criteria:
                    first_times[sid] = sorted(times)[concept.min_criteria - 1]
        else:
            if not concept.rules:
                raise ValueError(f"concept {concept.name!r} has no rules defined")
            observed_ids = set()
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
                _merge_min(first_times, trig)
        triggered_ids = set(first_times)

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
        if include_first_time:
            first_frame = pl.DataFrame(
                {
                    subject_id_col: list(first_times.keys()),
                    f"{concept.name}_first_time": list(first_times.values()),
                },
                schema={
                    subject_id_col: out.schema[subject_id_col],
                    f"{concept.name}_first_time": pl.Datetime("us"),
                },
            )
            out = out.join(first_frame, on=subject_id_col, how="left")

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
    include_first_time: bool = False,
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
        include_first_time=include_first_time,
    )
    return (
        labeled.join(keys, on="_visit_key", how="left")
        .drop("_visit_key")
        .select(
            [subject_id_col, visit_col]
            + [c for c in labeled.columns if c != "_visit_key"]
        )
    )
