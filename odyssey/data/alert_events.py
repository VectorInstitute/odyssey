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


ALERT_EVENTS: Tuple[AlertEvent, ...] = (
    AlertEvent(
        "vasopressor_start",
        concept="on_vasopressors",
        token_regex=(
            r"norepinephrine|levophed|epinephrine|vasopressin|phenylephrine"
            r"|neo-?synephrine|dopamine|angiotensin"
        ),
    ),
    AlertEvent("icu_admission", code_prefix="ICU_ADMISSION"),
    AlertEvent("acute_kidney_injury", concept="acute_kidney_injury"),
    AlertEvent("death", code_prefix="MEDS_DEATH", subject_scoped=True),
)


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
    raw_events: pl.DataFrame, alerts: Sequence[AlertEvent], source: str
) -> Dict[str, EventTimes]:
    """Onset/censoring times for every alert, labeling concepts once."""
    concepts = concepts_for_source(source)
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
    "ALERT_EVENTS",
    "AlertEvent",
    "EventTimes",
    "all_event_times",
    "event_times",
    "hours_since_origin",
    "origin_hours",
]
