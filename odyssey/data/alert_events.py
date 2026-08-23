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

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import polars as pl

from odyssey.data.concepts import concepts_for_source, label_concepts_by_visit
from odyssey.data.sequences import BIRTH_CODE


@dataclass(frozen=True)
class AlertEvent:
    """One clinically meaningful event whose onset is forecast."""

    name: str
    concept: Optional[str] = None
    """Concept whose first-trigger time defines onset (visit-scoped)."""

    code_prefix: Optional[str] = None
    """Code family whose first occurrence defines onset."""

    subject_scoped: bool = False
    """Onset and censoring are taken over the subject's whole record
    rather than the visit (death is not tied to a hadm_id)."""

    token_regex: Optional[str] = None
    """Regex over vocabulary tokens naming this event's own next-event
    tokens, for the next-event-mass score. Defaults to ``^code_prefix``."""

    next_visit: bool = False
    """Onset is the first occurrence of ``code_prefix`` strictly AFTER the
    visit's last event (the next admission); follow-up runs to the end of
    the subject's record, not the visit's (30-day readmission)."""


ALERT_EVENTS_V1: Tuple[AlertEvent, ...] = (
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

ALERT_EVENTS_V2: Tuple[AlertEvent, ...] = ALERT_EVENTS_V1 + (
    # Sepsis-3 onset from the sepsis3 concept (suspected infection + SOFA
    # >= 2; see odyssey.data.concepts.Sepsis3Rule). MIMIC-IV only today.
    AlertEvent("sepsis3", concept="sepsis3"),
    # 30-day readmission: the next hospital admission after this visit's
    # last event; scored at discharge-anchored index rows with a 720h
    # horizon (the hazard head trains on every position's gap to it).
    AlertEvent("readmission_30d", code_prefix="HOSPITAL_ADMISSION//", next_visit=True),
)

ALERT_TASK_SETS: Dict[str, Tuple[AlertEvent, ...]] = {
    "v1": ALERT_EVENTS_V1,
    "v2": ALERT_EVENTS_V2,
    # v3 widens the CONCEPT set only (odyssey.data.concepts.TASK_SETS); its
    # alert events are v2's.
    "v3": ALERT_EVENTS_V2,
}

# The v1 set, kept as the module-level default every pre-task-set caller
# reads (run configs without a task_set are v1 by definition).
ALERT_EVENTS: Tuple[AlertEvent, ...] = ALERT_EVENTS_V1


def alert_events_for(task_set: str = "v1") -> Tuple[AlertEvent, ...]:
    """Return the alert events a run with ``task_set`` trains heads for / scores."""
    if task_set not in ALERT_TASK_SETS:
        raise ValueError(
            f"unknown task_set {task_set!r}; known: {sorted(ALERT_TASK_SETS)}"
        )
    return ALERT_TASK_SETS[task_set]


# ---------------------------------------------------------------------------
# Onset and censoring times
# ---------------------------------------------------------------------------


@dataclass
class EventTimes:
    """Onset / censoring per key, all in hours on the sequence time origin."""

    onset: Dict[Tuple[int, int], float]
    """(subject_id, visit_id) -> onset hours; missing = never observed.
    For subject-scoped events the visit_id is ignored (all visits of the
    subject share the subject's onset)."""

    censor: Dict[Tuple[int, int], float]
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


def _next_visit_onsets(
    timed: pl.DataFrame, code_prefix: str, origins: pl.DataFrame
) -> Dict[Tuple[int, int], float]:
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
    concept_first_times: Optional[pl.DataFrame] = None,
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

    onset: Dict[Tuple[int, int], float] = {}
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
    elif alert.code_prefix is not None:
        hits = timed.filter(pl.col("code").str.starts_with(alert.code_prefix))
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
            f"alert {alert.name!r} defines neither concept nor code_prefix"
        )
    return EventTimes(onset=onset, censor=censor, subject_scoped=alert.subject_scoped)


def all_event_times(
    raw_events: pl.DataFrame,
    alerts: Sequence[AlertEvent],
    source: str,
    *,
    task_set: str = "v1",
) -> Dict[str, EventTimes]:
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
    "EventTimes",
    "all_event_times",
    "event_times",
    "hours_since_origin",
    "origin_hours",
]
