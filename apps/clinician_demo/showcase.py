"""Pick the gallery: visits that show what the model does well AND badly.

A gallery of only early warnings would overclaim. Every gallery therefore
has four sections, chosen by fixed deterministic rules from a visit-level
stats table:

- **early warnings**: the event happened, and the 24 h alert had been on
  continuously for at least ``MIN_LEAD_HOURS`` when it did;
- **quiet stays**: no event, and the risk stayed far below every line
  (the model does not cry wolf on everyone);
- **misses**: the event happened with no alert on;
- **false alarms**: the alert came on and the event never followed.

Lead time is measured from the start of the alert episode that was still
on when the event began, not from the first time the risk ever touched
the line: a line brushed on day 1 of a two-week stay is not a warning for
an event on day 12. Featured warnings are further limited to
``MIN_LEAD_HOURS``..``MAX_FEATURED_LEAD_HOURS`` (clinically actionable),
and every section states its rate over ALL eligible stays, so a
hand-picked case is never mistaken for the norm.

The stats table has one row per (subject, visit, event); times are hours
since the subject's first event. ``end_hours`` is the event's onset for
positives and the last at-risk moment otherwise; ``first_cross_hours`` is
the first time the risk reached the line; ``alert_start_hours`` is the
start of the alert episode live at ``end_hours`` (positives only).
Credentialed mode builds the table from the run's banked landmark rows
(:func:`visit_stats_from_alert_rows`, no model run); open mode builds it
from traced patients.
"""

from collections.abc import Callable, Mapping, Sequence

import polars as pl

from apps.clinician_demo.schemas import Gallery, GalleryCase, GallerySection
from apps.clinician_demo.thresholds import horizon_key


VISIT_STATS_SCHEMA: dict[str, pl.DataType] = {
    "subject_id": pl.Int64(),
    "visit_id": pl.Int64(),
    "event": pl.Utf8(),
    "positive": pl.Boolean(),
    "first_cross_hours": pl.Float64(),
    "alert_start_hours": pl.Float64(),
    "end_hours": pl.Float64(),
    "start_hours": pl.Float64(),
    "max_risk": pl.Float64(),
    "threshold": pl.Float64(),
}
MIN_LEAD_HOURS = 6.0
MAX_FEATURED_LEAD_HOURS = 72.0
MAX_STAY_HOURS = 14 * 24.0
MIN_QUIET_STAY_HOURS = 48.0
QUIET_FRACTION = 0.25
LANDMARK_HOURS = 4.0
_NEVER = -1e18


def empty_visit_stats() -> pl.DataFrame:
    """Return an empty stats table with the right schema."""
    return pl.DataFrame(schema=VISIT_STATS_SCHEMA)


def visit_stats_from_alert_rows(
    rows: pl.DataFrame,
    thresholds: Mapping[str, float],
    horizon_hours: float = 24.0,
) -> pl.DataFrame:
    """Build visit-level stats from banked landmark rows, one per (visit, event).

    ``rows`` are ``alerts_rows.parquet`` rows (``subject_id``/``visit_id``
    stored as floats, ``time_hours``, ``event``, ``hazard@{h}h``,
    ``y@{h}h``). Landmark rows exist only while the patient is at risk, so
    for a positive visit the onset lies within one landmark interval after
    the last row; ``end_hours`` is set to that last row, which makes the
    derived lead time a LOWER bound (the UI marks it approximate until the
    patient is traced and the exact onset is known).
    """
    key = horizon_key(horizon_hours)
    hazard, outcome = f"hazard@{key}", f"y@{key}"
    frame = rows.filter(pl.col("event").is_in(list(thresholds))).with_columns(
        pl.col("subject_id").cast(pl.Int64),
        pl.col("visit_id").cast(pl.Int64),
        pl.col("event")
        .replace_strict(thresholds, return_dtype=pl.Float64)
        .alias("_thr"),
    )
    if frame.height == 0:
        return empty_visit_stats()
    on = pl.col("_on")
    time = pl.col("time_hours")
    last_off = time.filter(~on).max().fill_null(_NEVER)
    return (
        frame.with_columns((pl.col(hazard) >= pl.col("_thr")).alias("_on"))
        .sort("time_hours")
        .group_by("subject_id", "visit_id", "event", maintain_order=True)
        .agg(
            (pl.col(outcome) == 1).any().alias("positive"),
            time.filter(on).min().alias("first_cross_hours"),
            pl.when(on.last())
            .then(time.filter(on & (time > last_off)).min())
            .otherwise(None)
            .alias("alert_start_hours"),
            time.max().alias("end_hours"),
            time.min().alias("start_hours"),
            pl.col(hazard).max().alias("max_risk"),
            pl.col("_thr").first().alias("threshold"),
        )
        .with_columns(
            pl.col("positive").fill_null(value=False),
            pl.when(pl.col("positive").fill_null(value=False))
            .then(pl.col("alert_start_hours"))
            .otherwise(None)
            .alias("alert_start_hours"),
        )
        .select(list(VISIT_STATS_SCHEMA))
        .cast(VISIT_STATS_SCHEMA)  # type: ignore[arg-type]
    )


def _with_derived(stats: pl.DataFrame) -> pl.DataFrame:
    return stats.with_columns(
        (pl.col("end_hours") - pl.col("alert_start_hours")).alias("lead_hours"),
        (pl.col("end_hours") - pl.col("start_hours")).alias("stay_hours"),
    )


def _pick(
    frame: pl.DataFrame, n: int, sort_by: Sequence[str], descending: Sequence[bool]
) -> pl.DataFrame:
    """Top ``n`` rows by ``sort_by``, one per subject, ties broken by ids."""
    ordered = frame.sort(
        [*sort_by, "subject_id", "visit_id"], descending=[*descending, False, False]
    )
    return ordered.unique("subject_id", keep="first", maintain_order=True).head(n)


def _pct(k: int, n: int) -> str:
    return f"{round(100 * k / n)}%" if n else "n/a"


def _lead(kind: str, lead: object) -> float | None:
    """Return an early warning's lead time (``None`` for other kinds)."""
    if kind != "early_warning" or not isinstance(lead, (int, float)):
        return None
    return float(lead)


def build_gallery(  # noqa: PLR0913 -- every knob is a documented selection rule
    stats: pl.DataFrame,
    *,
    display: Mapping[str, str],
    seen_in_training: Callable[[int], bool] = lambda _sid: False,
    per_event: int = 3,
    n_quiet: int = 3,
    n_misses: int = 2,
    n_false_alarms: int = 2,
    approximate_leads: bool = True,
) -> Gallery:
    """Build the four-section gallery from a visit-stats table.

    ``display`` maps event names to readable names and also fixes which
    events are shown and in what order. ``approximate_leads`` marks lead
    times as lower bounds (banked landmark rows) rather than exact.
    """
    frame = _with_derived(stats.filter(pl.col("event").is_in(list(display))))
    approx = "≈" if approximate_leads else ""

    def _case(row: Mapping[str, object], kind: str, headline: str) -> GalleryCase:
        sid = int(row["subject_id"])  # type: ignore[call-overload]
        return GalleryCase(
            subject_id=sid,
            visit_id=int(row["visit_id"]),  # type: ignore[call-overload]
            kind=kind,
            event=str(row["event"]) if row.get("event") is not None else None,
            headline=headline,
            lead_hours=_lead(kind, row.get("lead_hours")),
            los_hours=float(row["stay_hours"]),  # type: ignore[arg-type]
            seen_in_training=seen_in_training(sid),
            lead_approximate=approximate_leads and kind == "early_warning",
        )

    warning_cases: list[GalleryCase] = []
    rates: list[str] = []
    for event, name in display.items():
        positives = frame.filter((pl.col("event") == event) & pl.col("positive"))
        if positives.height == 0:
            continue
        warned = positives.filter(pl.col("lead_hours") >= MIN_LEAD_HOURS)
        rates.append(
            f"{name} {warned.height} of {positives.height} "
            f"({_pct(warned.height, positives.height)})"
        )
        featured = warned.filter(
            (pl.col("lead_hours") <= MAX_FEATURED_LEAD_HOURS)
            & (pl.col("stay_hours") <= MAX_STAY_HOURS)
        )
        warning_cases += [
            _case(
                r,
                "early_warning",
                f"{name}: alert on {approx}{r['lead_hours']:.0f} h before it began",
            )
            for r in _pick(featured, per_event, ["lead_hours"], [True]).iter_rows(
                named=True
            )
        ]
    sections = [
        GallerySection(
            kind="early_warning",
            title="Early warnings",
            summary=(
                "Stays where the alert was already on at least "
                f"{MIN_LEAD_HOURS:g} h when the event began: " + "; ".join(rates)
                if rates
                else "No stay in this data had one of the events."
            ),
            cases=warning_cases,
        )
    ]

    by_visit = frame.group_by("subject_id", "visit_id").agg(
        pl.col("positive").any().alias("any_positive"),
        (pl.col("max_risk") < QUIET_FRACTION * pl.col("threshold"))
        .all()
        .alias("all_quiet"),
        pl.col("stay_hours").max(),
        pl.col("event").n_unique().alias("n_events"),
    )
    quiet = by_visit.filter(
        ~pl.col("any_positive")
        & pl.col("all_quiet")
        & (pl.col("n_events") == len(display))
        & (pl.col("stay_hours") >= MIN_QUIET_STAY_HOURS)
    ).with_columns(pl.lit(None, dtype=pl.Utf8).alias("event"))
    sections.append(
        GallerySection(
            kind="quiet",
            title="Quiet stays",
            summary=(
                f"{quiet.height} stays had none of the events and stayed far below "
                "every alert line"
            ),
            cases=[
                _case(
                    r,
                    "quiet",
                    f"No event; risk stayed low for {r['stay_hours'] / 24:.1f} days",
                )
                for r in _pick(quiet, n_quiet, ["stay_hours"], [True]).iter_rows(
                    named=True
                )
            ],
        )
    )

    positives_all = frame.filter(pl.col("positive"))
    misses = positives_all.filter(pl.col("alert_start_hours").is_null())
    sections.append(
        GallerySection(
            kind="miss",
            title="Misses",
            summary=(
                f"{misses.height} of {positives_all.height} events "
                f"({_pct(misses.height, positives_all.height)}) began with no alert on"
            ),
            cases=[
                _case(
                    r,
                    "miss",
                    f"{display[str(r['event'])]} began with no alert on",
                )
                for r in _pick(misses, n_misses, ["max_risk"], [False]).iter_rows(
                    named=True
                )
            ],
        )
    )

    negatives = frame.filter(~pl.col("positive"))
    alarms = negatives.filter(pl.col("first_cross_hours").is_not_null())
    sections.append(
        GallerySection(
            kind="false_alarm",
            title="False alarms",
            summary=(
                f"In {alarms.height} of {negatives.height} cases "
                f"({_pct(alarms.height, negatives.height)}) an event's alert came "
                "on at least once and that event never happened during the stay"
            ),
            cases=[
                _case(
                    r,
                    "false_alarm",
                    f"{display[str(r['event'])]} alert came on; no event followed",
                )
                for r in _pick(alarms, n_false_alarms, ["max_risk"], [True]).iter_rows(
                    named=True
                )
            ],
        )
    )
    return Gallery(sections=sections)


__all__ = [
    "LANDMARK_HOURS",
    "MAX_FEATURED_LEAD_HOURS",
    "MIN_LEAD_HOURS",
    "VISIT_STATS_SCHEMA",
    "build_gallery",
    "empty_visit_stats",
    "visit_stats_from_alert_rows",
]
