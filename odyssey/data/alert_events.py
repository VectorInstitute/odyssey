"""Alert events: onset and censoring times of clinically meaningful events.

Shared by training (per-event hazard heads, :mod:`odyssey.training.event_targets`)
and evaluation (:mod:`odyssey.inference.alerts`): the events a general
sequence forecaster is asked to alert on, defined once. An event's onset
is either the first trigger of a concept from the same rule registry the
bottleneck is supervised with (vasopressor start, acute kidney injury)
or the first occurrence of a code family (ICU admission, death). Death is
subject-scoped -- it is not tied to a visit -- the others visit-scoped.
Censoring is the end of observed follow-up (last event of the visit, or
of the subject's record for subject-scoped events). All times are hours
on the sequence time origin (the subject's first timed non-birth event),
so they line up with chunk time stamps position for position.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from odyssey.data.concepts import concepts_for_source, label_concepts_by_visit
from odyssey.data.sequences import BIRTH_CODE


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertEvent:
    """One clinically meaningful event whose onset is forecast."""

    name: str
    concept: str | None = None
    """Concept whose first-trigger time defines onset (visit-scoped)."""

    code_prefix: str | None = None
    """Code family whose first occurrence defines onset."""

    code_regex: str | None = None
    """Regex over the raw ``code`` string (case-insensitive) whose first
    match defines onset -- for events that aren't a clean prefix (e.g. a
    drug class matched by ingredient name anywhere in the code), the way
    :data:`~odyssey.inference.baseline_features.DRUG_CLASSES` already
    matches for the GBM baseline. Mutually exclusive with ``code_prefix``
    in practice, not enforced. Distinct from ``token_regex``, which
    matches already-tokenized VOCABULARY tokens (post-binning) for the
    next-event-mass score, not raw codes for onset computation."""

    subject_scoped: bool = False
    """Onset and censoring are taken over the subject's whole record
    rather than the visit (death is not tied to a hadm_id)."""

    token_regex: str | None = None
    """Regex over vocabulary tokens naming this event's own next-event
    tokens, for the next-event-mass score. Defaults to ``^code_prefix``."""

    next_visit: bool = False
    """Onset is the first occurrence of ``code_prefix`` strictly AFTER the
    visit's last event (the next admission); follow-up runs to the end of
    the subject's record, not the visit's (30-day readmission)."""

    def __post_init__(self) -> None:
        """``next_visit`` is only implemented for ``code_prefix``, not ``code_regex``.

        Silently falling through to plain first-occurrence semantics for
        that combination would be a wrong-answer bug, not a clean error --
        ``_next_visit_onsets`` only knows ``str.starts_with``.
        """
        if self.code_regex is not None and self.next_visit:
            raise ValueError(
                f"alert {self.name!r}: next_visit is only implemented for "
                "code_prefix, not code_regex"
            )


ALERT_EVENTS_V1: tuple[AlertEvent, ...] = (
    AlertEvent(
        "vasopressor_start",
        concept="on_vasopressors",
        token_regex=(
            r"norepinephrine|levophed|epinephrine|vasopressin|phenylephrine"
            r"|neo-?synephrine|dopamine|angiotensin"
        ),
    ),
    # "ICU_ADMISSION//" with the separator: eICU also emits
    # ICU_ADMISSION_WEIGHT / ICU_ADMISSION_HEIGHT measurement codes at
    # unit admission, which are not admission events. Note that on eICU
    # the unit stay IS the visit, so a visit-scoped "first ICU admission"
    # is degenerate there (few at-risk index times); it is a MIMIC-IV
    # alert first and foremost.
    AlertEvent("icu_admission", code_prefix="ICU_ADMISSION//"),
    AlertEvent("acute_kidney_injury", concept="acute_kidney_injury"),
    AlertEvent("death", code_prefix="MEDS_DEATH", subject_scoped=True),
)

ALERT_EVENTS_V2: tuple[AlertEvent, ...] = ALERT_EVENTS_V1 + (
    # Sepsis-3 onset from the sepsis3 concept (suspected infection + SOFA
    # >= 2; see odyssey.data.concepts.Sepsis3Rule). MIMIC-IV only today.
    AlertEvent("sepsis3", concept="sepsis3"),
    # 30-day readmission: the next hospital admission after this visit's
    # last event; scored at discharge-anchored index rows with a 720h
    # horizon (the hazard head trains on every position's gap to it).
    AlertEvent("readmission_30d", code_prefix="HOSPITAL_ADMISSION//", next_visit=True),
)

ALERT_TASK_SETS: dict[str, tuple[AlertEvent, ...]] = {
    "v1": ALERT_EVENTS_V1,
    "v2": ALERT_EVENTS_V2,
    # v3 widens the CONCEPT set only (odyssey.data.concepts.TASK_SETS); its
    # alert events are v2's.
    "v3": ALERT_EVENTS_V2,
}

# The v1 set, kept as the module-level default every pre-task-set caller
# reads (run configs without a task_set are v1 by definition).
ALERT_EVENTS: tuple[AlertEvent, ...] = ALERT_EVENTS_V1


def alert_events_for(
    task_set: str = "v1", *, source: str | None = None
) -> tuple[AlertEvent, ...]:
    """Return the alert events a run with ``task_set`` trains heads for / scores.

    With ``source``, concept-backed alerts whose concept does not resolve
    for that source are dropped (sepsis3 on eICU): head construction, the
    event-time computation, and scoring must all see the identical
    source-resolved list, so pass the run's source wherever it is known.
    Without ``source`` the unresolved list is returned unchanged (the
    MIMIC-only scripts and pre-source callers).
    """
    if task_set not in ALERT_TASK_SETS:
        raise ValueError(
            f"unknown task_set {task_set!r}; known: {sorted(ALERT_TASK_SETS)}"
        )
    events = ALERT_TASK_SETS[task_set]
    if source is None:
        return events
    resolved = {c.name for c in concepts_for_source(source, task_set=task_set)}
    kept = tuple(a for a in events if a.concept is None or a.concept in resolved)
    for dropped in (a for a in events if a not in kept):
        logger.warning(
            "[alerts] source %r: dropping alert event %r -- its concept %r "
            "does not resolve for this source",
            source,
            dropped.name,
            dropped.concept,
        )
    return kept


# Widened counting-hazard AUXILIARY events (2026-08-28): the GBM
# feature-group ablation found explicit drug-class occurrence counts
# explain 70-99% of the GBM's margin on vasopressor_start/icu_admission;
# a follow-up probe (scripts/probe_counting_signal.py) measured that the
# backbone's own hidden state cannot linearly recover these specific
# counts (R^2 0.01-0.33) the way it already recovers recency (R^2 0.92)
# or general activity counts like lab/visit volume (R^2 0.6-0.9). The
# mechanism under test: training the model to be accurate at predicting
# TIME TO NEXT OCCURRENCE of a code is, for a roughly rate-stationary
# process, an estimator of the same underlying quantity a window COUNT
# estimates (count ~ rate x window; inter-arrival time ~
# Exponential(rate)) -- so widening the existing per-event hazard-head
# machinery (the same mechanism vasopressor_start/death/etc. already use)
# to include these specific codes as an AUXILIARY training signal should
# force that rate-like representation, the same way the curated events
# already do for the concepts they cover. These are trained but never
# scored as alerts (see TrainingConfig.auxiliary_event_names,
# hazard_events_for below) -- deliberately NOT added to any
# ALERT_TASK_SETS entry.
#
# Regex strings are copied from odyssey.inference.baseline_features.
# DRUG_CLASSES (the GBM's own drug-class matching) rather than imported --
# baseline_features.py already imports FROM this module (origin_hours),
# so the reverse import would be circular. Keep these in sync with
# DRUG_CLASSES by hand if either changes. Limited to the 6 classes the
# counting probe found worst-recovered (R^2 < 0.1, plus vasopressor at
# 0.15-0.33 as the clinically central one) rather than all 13 in
# DRUG_CLASSES, to keep this a small, targeted experiment.
COUNTING_AUXILIARY_EVENTS: tuple[AlertEvent, ...] = (
    AlertEvent(
        "vasopressor",
        code_regex=(
            r"norepinephrine|levophed|epinephrine|vasopressin|phenylephrine"
            r"|neo-?synephrine|dopamine|angiotensin"
        ),
    ),
    AlertEvent("inotrope", code_regex=r"dobutamine|milrinone"),
    AlertEvent(
        "antiarrhythmic", code_regex=r"amiodarone|diltiazem|esmolol|adenosine|lidocaine"
    ),
    AlertEvent(
        "neuromuscular_blocker",
        code_regex=r"cisatracurium|vecuronium|rocuronium|succinylcholine",
    ),
    AlertEvent(
        "corticosteroid",
        code_regex=(
            r"hydrocortisone|methylprednisolone|dexamethasone|prednisone"
            r"|prednisolone"
        ),
    ),
    AlertEvent("bicarbonate", code_regex=r"sodium bicarbonate|bicarb"),
)

COUNTING_AUXILIARY_EVENTS_BY_NAME: dict[str, AlertEvent] = {
    a.name: a for a in COUNTING_AUXILIARY_EVENTS
}


def hazard_events_for(
    task_set: str,
    auxiliary_event_names: Sequence[str] = (),
    *,
    source: str | None = None,
) -> tuple[AlertEvent, ...]:
    """Return the full event list an ``EventHazardHeads`` trains.

    Single shared source for what was three independently-duplicated
    ``alert_events_for(task_set)`` call sites in
    :mod:`odyssey.training.train`/:mod:`odyssey.training.shard_stream`.
    ``auxiliary_event_names`` names entries from
    :data:`COUNTING_AUXILIARY_EVENTS_BY_NAME`; empty (the default)
    reproduces ``alert_events_for(task_set)`` exactly, so every existing
    config/checkpoint is unaffected. Auxiliary events are deliberately
    NOT included in :func:`alert_events_for`'s own output -- they are
    trained but never read by :mod:`odyssey.inference.alerts`' scoring,
    which iterates ``alert_events_for(task_set)`` directly.
    """
    extra = tuple(
        COUNTING_AUXILIARY_EVENTS_BY_NAME[name] for name in auxiliary_event_names
    )
    events = alert_events_for(task_set, source=source) + extra
    names = [a.name for a in events]
    if len(names) != len(set(names)):
        # An auxiliary name listed twice, or colliding with a curated alert
        # name, would silently collapse in all_event_times' name-keyed dict
        # while EventHazardHeads still built one head per entry -- refuse
        # loudly instead.
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"hazard_events_for: duplicate event name(s) {dupes} -- check "
            "auxiliary_event_names for repeats or collisions with "
            f"alert_events_for({task_set!r})"
        )
    return events


# EHRSHOT-style PROBE tasks (Wornow et al., NeurIPS D&B 2023): frozen-probe
# evaluation targets, deliberately NOT trained alert heads. Amrit's 2026-08-28
# directive: don't keep adding heads that compete for shared-backbone
# gradient (documented cost, e.g. eicu_subset_v9's recency channel: -2 to
# -9pp on other families) -- instead measure whether the GENERAL
# representation already supports these tasks under a frozen probe (see
# odyssey.inference.probe_baseline). These 5 are the "Anticipating Lab Test
# Values" category, adapted from EHRSHOT's 5-way severity multiclass to a
# binary "does the existing task_set=v3 concept trigger within horizon h"
# (the concepts already exist; EHRSHOT's literal severity buckets do not).
# "Long length of stay" and the "Assignment of New Diagnoses"/"Chest X-ray
# Findings" categories are NOT expressible as first-trigger-of-a-concept-or-
# code-family AlertEvents (LOS is a static per-visit label; new-diagnosis
# needs vetted ICD/SNOMED code lists not yet in this repo; chest x-ray needs
# structured labels this project doesn't have) -- LOS gets its own path in
# odyssey.inference.probe_baseline (visit_envelope + a fixed early snapshot,
# not a landmark sweep); the rest are left for when their inputs exist.
PROBE_EVENTS: tuple[AlertEvent, ...] = tuple(
    AlertEvent(name=c, concept=c)
    for c in (
        "anemia",
        "hyperkalemia",
        "hypoglycemia",
        "hyponatremia",
        "thrombocytopenia",
    )
)


# ---------------------------------------------------------------------------
# Onset and censoring times
# ---------------------------------------------------------------------------


@dataclass
class EventTimes:
    """Onset / censoring per key, all in hours on the sequence time origin."""

    onset: dict[tuple[int, int], float]
    """(subject_id, visit_id) -> onset hours; missing = never observed.
    For subject-scoped events the visit_id is ignored (all visits of the
    subject share the subject's onset)."""

    censor: dict[tuple[int, int], float]
    """(subject_id, visit_id) -> last observed time (end of follow-up)."""

    subject_scoped: bool


def origin_hours(events: pl.DataFrame) -> pl.DataFrame:
    """subject_id -> first timed non-birth event (the sequence time origin)."""
    return (
        events.filter(pl.col("time").is_not_null() & (pl.col("code") != BIRTH_CODE))
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
    )


def hours_since_origin(
    frame: pl.DataFrame, col: str, origins: pl.DataFrame
) -> pl.DataFrame:
    """Rewrite Datetime column ``col`` as hours since each subject's origin."""
    return frame.join(origins, on="subject_id", how="left").with_columns(
        ((pl.col(col) - pl.col("_origin")).dt.total_seconds() / 3600.0).alias(col)
    )


def visit_envelope(events: pl.DataFrame) -> dict[tuple[int, int], tuple[float, float]]:
    """(subject_id, hadm_id) -> (hours of first event, hours of last event).

    Shared building block for tasks defined over a visit's TOTAL span
    (e.g. length of stay) rather than a first-trigger onset -- distinct
    from :func:`event_times`, whose ``last`` groupby only ever becomes the
    right-censoring time, never a label input itself.
    """
    origins = origin_hours(events)
    timed = events.filter(
        pl.col("time").is_not_null() & pl.col("hadm_id").is_not_null()
    )
    span = (
        timed.group_by("subject_id", "hadm_id")
        .agg(pl.col("time").min().alias("_start"), pl.col("time").max().alias("_end"))
        .join(origins, on="subject_id", how="left")
        .with_columns(
            ((pl.col("_start") - pl.col("_origin")).dt.total_seconds() / 3600.0).alias(
                "_start"
            ),
            ((pl.col("_end") - pl.col("_origin")).dt.total_seconds() / 3600.0).alias(
                "_end"
            ),
        )
    )
    return {
        (int(s), int(v)): (float(a), float(b))
        for s, v, a, b in zip(
            span["subject_id"], span["hadm_id"], span["_start"], span["_end"]
        )
    }


def _next_visit_onsets(
    timed: pl.DataFrame, code_prefix: str, origins: pl.DataFrame
) -> dict[tuple[int, int], float]:
    """(subject, visit) -> hours of the first ``code_prefix`` event of a LATER visit.

    "Later" is by admission order: the earliest ``code_prefix`` row of a
    different ``hadm_id`` whose time is after this visit's first event. A
    visit's own rows can carry timestamps past its real discharge (results
    finalized later, stray late-attributed rows), so anchoring on the
    visit's last row skipped genuine back-to-back readmissions in ~1% of
    visits (research journal entry 44); admissions do not overlap, so the
    next admission by start time is the next visit.
    """
    visits = (
        timed.filter(pl.col("hadm_id").is_not_null())
        .group_by("subject_id", "hadm_id")
        .agg(pl.col("time").min().alias("_start"))
    )
    hits = timed.filter(pl.col("code").str.starts_with(code_prefix)).select(
        "subject_id",
        pl.col("hadm_id").alias("_next_hadm"),
        pl.col("time").alias("_next"),
    )
    joined = (
        visits.join(hits, on="subject_id", how="inner")
        .filter(
            (pl.col("_next") > pl.col("_start"))
            & (
                pl.col("_next_hadm").is_null()
                | (pl.col("_next_hadm") != pl.col("hadm_id"))
            )
        )
        .group_by("subject_id", "hadm_id")
        .agg(pl.col("_next").min())
    )
    joined = hours_since_origin(joined, "_next", origins)
    return {
        (int(s), int(v)): float(t)
        for s, v, t in zip(joined["subject_id"], joined["hadm_id"], joined["_next"])
    }


def event_times(
    events: pl.DataFrame,
    alert: AlertEvent,
    *,
    concept_first_times: pl.DataFrame | None = None,
) -> EventTimes:
    """Onset and censoring times for ``alert`` over ``events``.

    ``concept_first_times`` is the output of
    :func:`~odyssey.data.concepts.label_concepts_by_visit` with
    ``include_first_time=True``, shared across concept-defined events so
    it is computed once.
    """
    origins = origin_hours(events)
    timed = events.filter(pl.col("time").is_not_null() & (pl.col("code") != BIRTH_CODE))
    if alert.subject_scoped:
        last = timed.group_by("subject_id").agg(pl.col("time").max().alias("_last"))
        last = hours_since_origin(last, "_last", origins)
        censor = {
            (int(s), -1): float(t) for s, t in zip(last["subject_id"], last["_last"])
        }
    else:
        visits = timed.filter(pl.col("hadm_id").is_not_null())
        last = visits.group_by("subject_id", "hadm_id").agg(
            pl.col("time").max().alias("_last")
        )
        if alert.next_visit:
            # Follow-up for "the next admission" is the subject's whole
            # record: a visit with nothing after it is censored at the
            # subject's last event, not at its own discharge.
            subject_last = timed.group_by("subject_id").agg(
                pl.col("time").max().alias("_last")
            )
            last = last.drop("_last").join(subject_last, on="subject_id", how="left")
        last = hours_since_origin(last, "_last", origins)
        censor = {
            (int(s), int(v)): float(t)
            for s, v, t in zip(last["subject_id"], last["hadm_id"], last["_last"])
        }

    onset: dict[tuple[int, int], float] = {}
    if alert.concept is not None:
        if concept_first_times is None:
            raise ValueError(f"alert {alert.name!r} needs concept_first_times")
        col = f"{alert.concept}_first_time"
        frame = concept_first_times.filter(pl.col(col).is_not_null()).select(
            "subject_id", "hadm_id", col
        )
        frame = hours_since_origin(frame, col, origins)
        onset = {
            (int(s), int(v)): float(t)
            for s, v, t in zip(frame["subject_id"], frame["hadm_id"], frame[col])
        }
    elif alert.code_prefix is not None and alert.next_visit:
        onset = _next_visit_onsets(timed, alert.code_prefix, origins)
    elif alert.code_prefix is not None or alert.code_regex is not None:
        hits = (
            timed.filter(pl.col("code").str.starts_with(alert.code_prefix))
            if alert.code_prefix is not None
            else timed.filter(pl.col("code").str.contains(f"(?i){alert.code_regex}"))
        )
        if alert.subject_scoped:
            first = hits.group_by("subject_id").agg(pl.col("time").min().alias("_t"))
            first = hours_since_origin(first, "_t", origins)
            onset = {
                (int(s), -1): float(t) for s, t in zip(first["subject_id"], first["_t"])
            }
        else:
            hits = hits.filter(pl.col("hadm_id").is_not_null())
            first = hits.group_by("subject_id", "hadm_id").agg(
                pl.col("time").min().alias("_t")
            )
            first = hours_since_origin(first, "_t", origins)
            onset = {
                (int(s), int(v)): float(t)
                for s, v, t in zip(first["subject_id"], first["hadm_id"], first["_t"])
            }
    else:
        raise ValueError(
            f"alert {alert.name!r} defines neither concept, code_prefix, nor code_regex"
        )
    return EventTimes(onset=onset, censor=censor, subject_scoped=alert.subject_scoped)


def all_event_times(
    raw_events: pl.DataFrame,
    alerts: Sequence[AlertEvent],
    source: str,
    *,
    task_set: str = "v1",
) -> dict[str, EventTimes]:
    """Onset/censoring times for every alert, labeling concepts once."""
    concepts = concepts_for_source(source, task_set=task_set)
    needed = [c for c in concepts if any(a.concept == c.name for a in alerts)]
    first = (
        label_concepts_by_visit(raw_events, needed, include_first_time=True)
        if needed
        else None
    )
    return {
        a.name: event_times(raw_events, a, concept_first_times=first) for a in alerts
    }


__all__ = [
    "ALERT_EVENTS_V1",
    "ALERT_EVENTS_V2",
    "ALERT_TASK_SETS",
    "alert_events_for",
    "ALERT_EVENTS",
    "AlertEvent",
    "COUNTING_AUXILIARY_EVENTS",
    "COUNTING_AUXILIARY_EVENTS_BY_NAME",
    "EventTimes",
    "PROBE_EVENTS",
    "all_event_times",
    "event_times",
    "hazard_events_for",
    "hours_since_origin",
    "origin_hours",
    "visit_envelope",
]
