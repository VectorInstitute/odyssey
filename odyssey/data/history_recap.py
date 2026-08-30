"""Inject a recap of prior diagnoses at each hospital admission.

Motivation (research journal, bundle analysis on the v3 subset run): at
the moment discharge coding begins, the model recovers only ~11% of the
diagnosis bundle in its top-|B|, no better than code frequency, while
simply copying the patient's *previous* admission's diagnosis bundle
recovers ~35%. Chronic conditions recur across admissions, but the
previous discharge codes sit thousands of tokens back -- outside the
attention window and compressed away in the recurrent state.

This transform makes that history local. At each ``HOSPITAL_ADMISSION``
event it appends, at the admission's own timestamp, one
``HISTORY//DIAGNOSIS//ICD//{version}//{category}`` token per distinct
3-character ICD category the patient was coded with in *earlier*
admissions (most recent first, capped). Category level keeps the
vocabulary small (~1.5K categories vs. ~15K codes) and matches how
chronic conditions are actually referred to. The recap is what a
clinician sees on the problem list at admission; it uses only the past.

Off by default; a training-config switch (``history_recap``) applies it
identically to train, tuning and held-out data, before value binning and
vocabulary building. HISTORY tokens fall in the ``other`` code family, so
diagnosis-family metrics are unaffected by their presence as targets.
"""

import polars as pl


HISTORY_PREFIX = "HISTORY//"
ADMISSION_PREFIX = "HOSPITAL_ADMISSION//"
DIAGNOSIS_ICD_PREFIX = "DIAGNOSIS//ICD//"


def _category_expr(code_col: str) -> pl.Expr:
    """``DIAGNOSIS//ICD//v//CODE`` -> ``HISTORY//DIAGNOSIS//ICD//v//CAT`` (3 chars)."""
    parts = pl.col(code_col).str.split("//")
    return pl.concat_str(
        [
            pl.lit("HISTORY//DIAGNOSIS//ICD//"),
            parts.list.get(2),
            pl.lit("//"),
            parts.list.get(3).str.slice(0, 3),
        ]
    )


def add_history_recap(
    events: pl.DataFrame,
    *,
    max_codes: int = 30,
    code_col: str = "code",
    time_col: str = "time",
    subject_col: str = "subject_id",
    visit_col: str = "hadm_id",
) -> pl.DataFrame:
    """Return ``events`` plus recap rows at every admission (see module docstring).

    Recap rows carry the admission's ``hadm_id`` and time, null
    ``numeric_value``, and every other column null. Categories are
    ordered by their most recent prior occurrence and capped at
    ``max_codes`` per admission. Admissions with no prior diagnoses get no
    rows. Idempotent in effect: existing HISTORY rows are dropped first.
    """
    events = events.filter(~pl.col(code_col).str.starts_with(HISTORY_PREFIX))
    if visit_col not in events.columns:
        return events
    admissions = events.filter(
        pl.col(code_col).str.starts_with(ADMISSION_PREFIX)
        & pl.col(time_col).is_not_null()
        & pl.col(visit_col).is_not_null()
    ).select(subject_col, pl.col(time_col).alias("_adm_time"), visit_col)
    if admissions.height == 0:
        return events
    diagnoses = (
        events.filter(
            pl.col(code_col).str.starts_with(DIAGNOSIS_ICD_PREFIX)
            & pl.col(time_col).is_not_null()
        )
        .select(subject_col, pl.col(time_col).alias("_dx_time"), code_col)
        .with_columns(_category_expr(code_col).alias("_recap"))
        .select(subject_col, "_dx_time", "_recap")
        .unique()
    )
    if diagnoses.height == 0:
        return events
    # every (admission, prior category) pair, keeping the category's most
    # recent prior occurrence for ordering
    pairs = (
        admissions.join(diagnoses, on=subject_col, how="inner")
        .filter(pl.col("_dx_time") < pl.col("_adm_time"))
        .group_by(subject_col, visit_col, "_adm_time", "_recap")
        .agg(pl.col("_dx_time").max().alias("_last"))
        .sort([subject_col, visit_col, "_last"], descending=[False, False, True])
        .with_columns(
            pl.int_range(pl.len()).over(subject_col, visit_col).alias("_rank")
        )
        .filter(pl.col("_rank") < max_codes)
    )
    if pairs.height == 0:
        return events
    recap = pairs.select(
        pl.col(subject_col),
        pl.col("_adm_time").alias(time_col),
        pl.col("_recap").alias(code_col),
        pl.col(visit_col),
    )
    # match the events schema: every other column null
    for name, dtype in events.schema.items():
        if name not in recap.columns:
            recap = recap.with_columns(pl.lit(None, dtype=dtype).alias(name))
    recap = recap.select(events.columns).cast(events.schema)
    # Stable ordering: recap rows sort after the admission row at the same
    # timestamp (the sequence builder keeps input order for ties).
    return pl.concat([events, recap], how="vertical").sort(
        [subject_col, time_col], maintain_order=True
    )


def maybe_history_recap(
    events: pl.DataFrame, *, enabled: bool, max_codes: int | None = None
) -> pl.DataFrame:
    """Apply :func:`add_history_recap` when ``enabled``, else pass through."""
    if not enabled:
        return events
    return add_history_recap(events, max_codes=max_codes or 30)


__all__ = ["HISTORY_PREFIX", "add_history_recap", "maybe_history_recap"]
