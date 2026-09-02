"""Running SOFA score from MEDS events (the organ-dysfunction half of Sepsis-3).

Sequential Organ Failure Assessment (Vincent et al. 1996) as operationalized
for MIMIC-IV by mimic-code's ``sofa.sql`` / ``sepsis3.sql`` (Johnson et al.
2018): six components, each scored 0-4 from the *worst* value in the
trailing 24 hours, summed. Evaluated on a per-key (visit) time grid made of
every component observation time, so the first instant the total crosses a
threshold is exact rather than hourly-rounded.

Component rules (mimic-code's thresholds; deviations stated inline):

- **Respiration**: PaO2/FiO2 from an arterial PaO2 paired with the most
  recent FiO2 charted within the previous 4 h (no FiO2 -> not assessed);
  ventilated (invasive or non-invasive) and PF < 100 -> 4, < 200 -> 3; any
  PF < 300 -> 2, < 400 -> 1.
- **Coagulation**: platelets < 150 -> 1, < 100 -> 2, < 50 -> 3, < 20 -> 4.
- **Liver**: bilirubin >= 1.2 -> 1, >= 2.0 -> 2, >= 6.0 -> 3, >= 12.0 -> 4.
- **Cardiovascular**: MAP < 70 -> 1; dopamine <= 5 or any dobutamine -> 2;
  dopamine > 5 or epinephrine/norepinephrine <= 0.1 -> 3; dopamine > 15 or
  epinephrine/norepinephrine > 0.1 -> 4 (rates in mcg/kg/min from the
  infusion-start rows; an infusion counts from its START row until its
  END row).
- **CNS**: GCS total (eye + verbal + motor paired within 15 min, the same
  pairing the qSOFA concept uses) 13-14 -> 1, 10-12 -> 2, 6-9 -> 3, < 6 -> 4.
- **Renal**: creatinine >= 1.2 -> 1, >= 2.0 -> 2, >= 3.5 -> 3, >= 5.0 -> 4;
  urine output over the trailing 24 h < 500 mL -> 3, < 200 mL -> 4, only
  once the key has at least 24 h of record behind it (a partial window
  would read as oliguria).

Vasopressor item ids, ventilation procedure ids and body-weight item ids
are not LOINC-keyed and live in a per-source table here
(:data:`SOFA_SOURCE_CONFIG`); the numeric-signal codes resolve through the
shared LOINC tables. Only sources with an entry can be scored (MIMIC-IV
today).

Beyond SOFA itself, :func:`urine_output_rate` generalizes the renal
component's trailing urine sum to an arbitrary window and, optionally, to
KDIGO's own weight-normalized mL/kg/h form -- the shape
:mod:`odyssey.data.concepts`' AKI staging needs.
:func:`urine_output_24h` is now the (unchanged) 24 h absolute-volume
special case of it.
"""

import functools
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    TypeVar,
    cast,
)

import polars as pl

from odyssey.data.code_mapping import prefixes_for_loinc


@dataclass(frozen=True)
class SofaSourceConfig:
    """Source-specific, non-LOINC ingredients of the SOFA score."""

    norepinephrine: tuple[str, ...]
    """Infusion code prefixes (START rows carry the rate, mcg/kg/min)."""
    epinephrine: tuple[str, ...]
    dopamine: tuple[str, ...]
    dobutamine: tuple[str, ...]
    infusion_start_prefix: str
    infusion_end_prefix: str
    ventilation_start: tuple[str, ...]
    ventilation_end: tuple[str, ...]
    cardiovascular_fixed_tier: dict[str, int] = field(default_factory=dict)
    """Score a drug's whole infusion at one fixed tier instead of reading
    its START row's ``value`` as a mcg/kg/min rate. For a source whose
    numeric field is not confirmed to be in that unit (free-text dosages
    mixing boluses and continuous rates, or a rate charted per-minute
    rather than per-kg-per-minute with no verified conversion) --
    guessing a conversion would silently produce a wrong severity tier
    rather than fail loudly. Keyed by drug name (``"dopamine"``,
    ``"epinephrine"``, ``"norepinephrine"``, ``"dobutamine"``); a drug not
    present here still uses the rate-based tiers, so this is opt-in per
    drug, not a source-wide switch. Set the *lower* of a drug's two
    rate-dependent tiers so the approximation never over-scores (an
    infusion always exists at some real rate, so the true tier is always
    >= this one); this does not affect *whether* Sepsis-3's SOFA-rise
    criterion fires, since starting any pressor already lifts the
    cardiovascular component by at least 2 from a baseline of 0 or 1 --
    only the exact score is approximate. Same spirit as ``oliguria``
    using absolute urine volume rather than KDIGO's weight-normalized
    rate: a stated, documented simplification rather than an unverifiable
    guess.
    """
    daily_weight: tuple[str, ...] = ()
    """Code prefixes of the recurring charted body weight, in kg.

    Body weight is only needed by :func:`urine_output_rate` (KDIGO's
    mL/kg/h form), never by SOFA itself, so a source may leave both
    weight fields empty: the rate criterion is then simply unassessable
    there, exactly as it is for a key with no weight charted.
    """
    admission_weight: tuple[str, ...] = ()
    """Code prefixes of the once-per-stay admission weight, in kg (the
    fallback :func:`urine_output_rate` uses before any recurring weight
    has been charted)."""


SOFA_SOURCE_CONFIG: dict[str, SofaSourceConfig] = {
    "mimic_iv": SofaSourceConfig(
        norepinephrine=("221906",),
        epinephrine=("221289",),
        dopamine=("221662",),
        dobutamine=("221653",),
        infusion_start_prefix="INFUSION_START//",
        infusion_end_prefix="INFUSION_END//",
        ventilation_start=("PROCEDURE//START//225792", "PROCEDURE//START//225794"),
        ventilation_end=("PROCEDURE//END//225792", "PROCEDURE//END//225794"),
        # Dobutamine has no rate-dependent SOFA tier (any dose scores 2);
        # MIMIC-IV's other three pressors ARE rate-based here, chartevents'
        # rate items being natively charted in mcg/kg/min already.
        cardiovascular_fixed_tier={"dobutamine": 2},
        # chartevents 224639 "Daily Weight" / 226512 "Admission Weight (Kg)",
        # both charted in kg. Deliberately NOT routed through the LOINC layer:
        # both carry the same body-weight LOINC, so a LOINC lookup could not
        # keep them apart, and urine_output_rate's fallback order (daily
        # first, admission only until a daily weight exists) needs exactly
        # that distinction.
        daily_weight=("LAB//224639//",),
        admission_weight=("LAB//226512//",),
    ),
    "eicu": SofaSourceConfig(
        # Vasopressor identity comes from odyssey.data.code_normalization's
        # HICL-first normalization (the same source-of-truth
        # odyssey.data.concepts.on_vasopressors and
        # odyssey.data.alert_events's vasopressor_start already build on),
        # NOT a hand-rolled name/regex match here: every real HICL entry
        # for these four drugs (checked against
        # odyssey/data/resources/eicu_hicl_ingredients.csv directly --
        # e.g. HICL 2051 "norepinephrine bitartrate", 36346
        # "norepinephrine", 37410 "norepinephrine bitar-0.9% nacl") reduces
        # to an ingredient string starting with the drug's generic name,
        # so a plain prefix on the post-normalization code catches every
        # named AND HICL-resolved-but-originally-unnamed row (36% of
        # eICU's medication rows carry no drugname at all; 94% of those
        # carry a resolving HICL). This assumes normalize_medications=True
        # has already run (true of every training/inference path by
        # default, verified against maybe_normalize's call sites).
        norepinephrine=("norepinephrine", "levophed"),
        epinephrine=("epinephrine",),
        dopamine=("dopamine",),
        dobutamine=("dobutamine",),
        infusion_start_prefix="MEDICATION//STARTED//",
        infusion_end_prefix="MEDICATION//STOPPED//",
        # No rate confirmed in mcg/kg/min for any of the four -- eICU's
        # medication.dosage is free text mixing bolus doses and continuous
        # rates in unverified units, and infusionDrug.infusionrate's own
        # extraction comment notes some drugs are charted per-minute, not
        # per-kg-per-minute ("Norepinephrine (mcg/min)"), with no shipped
        # weight-based conversion. Guessing one would silently produce a
        # wrong severity tier, so every drug here is fixed rather than
        # rate-scored -- set to the LOWER of each drug's two
        # rate-dependent SOFA tiers so this never over-scores (see
        # SofaSourceConfig.cardiovascular_fixed_tier's own docstring for
        # why this does not affect whether/when sepsis3 fires, only the
        # exact score). Same documented-limitation spirit as oliguria's
        # absolute-volume choice below.
        cardiovascular_fixed_tier={
            "dopamine": 3,
            "epinephrine": 3,
            "norepinephrine": 3,
            "dobutamine": 2,
        },
        # eICU has no respiratoryCare/respiratoryCharting table; ventilation
        # status comes from carePlanGeneral's periodically recharted
        # nursing assessment (CAREPLAN_GENERAL//Ventilation//{value}), not
        # a discrete start/end procedure event. "Ventilated"/"Spontaneous"
        # cover the two states _intervals_to_points needs (repeated
        # "Ventilated" readings before the next "Spontaneous" are handled
        # there, not here -- see that function's docstring). NOT verified
        # against a real cplitemvalue distinct-value dump: the exact
        # capitalization here is inferred from eICU-CRD's public
        # documentation, and a tracheostomy-ventilated patient (a
        # "Trach - ..." value, if the real data uses one) would not match
        # either prefix and reads as never-ventilated. Confirm both before
        # this lands in a training run.
        ventilation_start=("CAREPLAN_GENERAL//Ventilation//Ventilated",),
        ventilation_end=("CAREPLAN_GENERAL//Ventilation//Spontaneous",),
        # No recurring daily weight in eICU (the patient table has only
        # once-per-stay admission/discharge weight); left empty, same
        # graceful-degradation this dataclass already supports for any
        # source without one.
        admission_weight=("ICU_ADMISSION_WEIGHT",),
    ),
}

# LOINC ingredients (resolved per source through code_mapping).
_PAO2 = "11556-8"
_FIO2 = "3150-0"
_PLATELETS = "777-3"
_BILIRUBIN = "1975-2"
_MAP = ("76536-2", "8478-0")  # non-invasive and arterial MAP
_GCS_EYE, _GCS_VERBAL, _GCS_MOTOR = "9267-6", "9270-0", "9268-4"
_CREATININE = "2160-0"
_URINE = "9187-6"

COMPONENTS: tuple[str, ...] = (
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
)
WINDOW_HOURS = 24.0
MAX_INFUSION_HOURS = 24.0 * 14  # an END-less infusion is capped, not eternal


_F = TypeVar("_F", bound=Callable[..., Any])


def _quiet_asof(fn: _F) -> _F:
    """Silence polars' per-call "Sortedness ... cannot be checked" asof warning.

    Every frame handed to ``join_asof(by=...)`` here is sorted by
    ``[key, time]`` first; polars cannot verify per-group sortedness
    cheaply and warns regardless (the same library limitation
    :mod:`odyssey.data.concepts` documents).
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Sortedness of columns cannot be checked"
            )
            return fn(*args, **kwargs)

    return cast(_F, wrapper)


def sofa_supported(source: str) -> bool:
    """Whether ``source`` has the non-LOINC ingredients SOFA needs."""
    return source in SOFA_SOURCE_CONFIG


def _starts_with_any(col: str, prefixes: Sequence[str]) -> pl.Expr:
    expr = pl.lit(False)
    for p in prefixes:
        expr = expr | pl.col(col).str.starts_with(p)
    return expr


def _loinc_prefixes(loincs: Sequence[str], source: str) -> list[str]:
    out: list[str] = []
    for loinc in loincs:
        out.extend(sorted(prefixes_for_loinc(loinc, source=source)))
    return out


def _numeric(
    events: pl.DataFrame,
    prefixes: Sequence[str],
    *,
    key: str,
    code_col: str,
    value_col: str,
    time_col: str,
) -> pl.DataFrame:
    """(key, time, value) rows of the codes under ``prefixes`` with a value."""
    if not prefixes:
        return pl.DataFrame(
            schema={
                key: events.schema[key],
                time_col: events.schema[time_col],
                "value": pl.Float64,
            }
        )
    return (
        events.filter(
            _starts_with_any(code_col, prefixes) & pl.col(value_col).is_not_null()
        )
        .select(key, time_col, pl.col(value_col).cast(pl.Float64).alias("value"))
        .sort([key, time_col])
    )


def _scored(
    frame: pl.DataFrame, score: pl.Expr, component: str, key: str, time_col: str
) -> pl.DataFrame:
    return frame.select(
        key,
        time_col,
        pl.lit(component).alias("component"),
        score.cast(pl.Int8).alias("score"),
    ).filter(pl.col("score") > 0)


def _band(
    value: pl.Expr, bands: Sequence[tuple[float, int]], *, ascending: bool
) -> pl.Expr:
    """Piecewise score: ``bands`` as (threshold, score) from mildest to worst.

    ``ascending=True`` means worse = higher value (``>= threshold``);
    ``False`` means worse = lower value (``< threshold``).
    """
    expr = pl.lit(0)
    for threshold, pts in bands:
        cond = value >= threshold if ascending else value < threshold
        expr = pl.when(cond).then(pl.lit(pts)).otherwise(expr)
    return expr


@_quiet_asof
def _intervals_to_points(
    starts: pl.DataFrame,
    ends: pl.DataFrame,
    *,
    key: str,
    time_col: str,
    step_hours: float = 1.0,
) -> pl.DataFrame:
    """Expand (key, start[, value]) / (key, end) rows into hourly points.

    Each START is paired with the first END after it for the same key
    (missing END -> capped at :data:`MAX_INFUSION_HOURS`), then sampled
    every ``step_hours`` so a 24 h worst-value window sees an active
    infusion / ventilation continuously, not only at its first row.

    A source that recharts status repeatedly rather than emitting a single
    discrete START (eICU's care-plan ventilation readings: nursing
    re-enters "Ventilated" at every assessment, not just at intubation)
    produces one paired-and-exploded range per repeated START, all ending
    at the same next END -- the same points several times over, not a
    correctness problem (a worst-in-window score is insensitive to
    duplicates) but an avoidable memory cost at nursing-assessment
    frequency. Deduplicated below rather than left to the caller.

    A trailing START with no later END is still capped at
    :data:`MAX_INFUSION_HOURS` -- correct for a patient discharged still
    ventilated, but a source without a reliable end signal (a missed
    extubation note) reads as ventilated for the full cap, not truly
    unbounded.
    """
    if starts.height == 0:
        return starts.select(
            key, time_col, *[c for c in starts.columns if c == "value"]
        )
    ends = ends.select(key, pl.col(time_col).alias("_end")).sort([key, "_end"])
    paired = (
        starts.sort([key, time_col])
        .join_asof(ends, left_on=time_col, right_on="_end", by=key, strategy="forward")
        .with_columns(
            pl.coalesce(
                [
                    pl.col("_end"),
                    pl.col(time_col) + pl.duration(hours=MAX_INFUSION_HOURS),
                ]
            ).alias("_end")
        )
        .with_columns(
            pl.min_horizontal(
                pl.col("_end"), pl.col(time_col) + pl.duration(hours=MAX_INFUSION_HOURS)
            ).alias("_end")
        )
    )
    step = f"{int(step_hours * 60)}m"
    expanded = paired.with_columns(
        pl.datetime_ranges(pl.col(time_col), pl.col("_end"), interval=step).alias("_ts")
    ).explode("_ts")
    keep = [key, pl.col("_ts").alias(time_col)] + (
        [pl.col("value")] if "value" in starts.columns else []
    )
    return expanded.select(keep).unique()


@_quiet_asof
def component_observations(  # noqa: PLR0912, PLR0915
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    key: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
) -> pl.DataFrame:
    """Instantaneous component scores: one row per (key, time, component, score > 0).

    The worst-in-24h rolling is applied by :func:`sofa_timeseries`; this
    function only turns raw readings into per-reading component scores.
    """
    cfg = SOFA_SOURCE_CONFIG[source]
    parts: list[pl.DataFrame] = []

    def num(loincs: Sequence[str]) -> pl.DataFrame:
        return _numeric(
            events,
            _loinc_prefixes(loincs, source),
            key=key,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )

    # -- coagulation / liver / renal (creatinine): plain bands
    parts.append(
        _scored(
            num([_PLATELETS]),
            _band(
                pl.col("value"), [(150, 1), (100, 2), (50, 3), (20, 4)], ascending=False
            ),
            "coagulation",
            key,
            time_col,
        )
    )
    parts.append(
        _scored(
            num([_BILIRUBIN]),
            _band(
                pl.col("value"),
                [(1.2, 1), (2.0, 2), (6.0, 3), (12.0, 4)],
                ascending=True,
            ),
            "liver",
            key,
            time_col,
        )
    )
    parts.append(
        _scored(
            num([_CREATININE]),
            _band(
                pl.col("value"),
                [(1.2, 1), (2.0, 2), (3.5, 3), (5.0, 4)],
                ascending=True,
            ),
            "renal",
            key,
            time_col,
        )
    )

    # -- cardiovascular: MAP < 70 -> 1; vasopressor rates (hourly points)
    parts.append(
        _scored(
            num(list(_MAP)),
            pl.when(pl.col("value") < 70).then(1).otherwise(0),
            "cardiovascular",
            key,
            time_col,
        )
    )
    ends = events.filter(
        pl.col(code_col).str.starts_with(cfg.infusion_end_prefix)
    ).select(
        key,
        time_col,
        pl.col(code_col).str.strip_prefix(cfg.infusion_end_prefix).alias("_item"),
    )

    def infusion_points(items: tuple[str, ...]) -> pl.DataFrame:
        starts = events.filter(
            pl.col(code_col).str.starts_with(cfg.infusion_start_prefix)
            & _starts_with_any(code_col, [cfg.infusion_start_prefix + i for i in items])
        ).select(key, time_col, pl.col(value_col).cast(pl.Float64).alias("value"))
        item_ends = ends.filter(pl.col("_item").is_in(list(items))).select(
            key, time_col
        )
        pts: pl.DataFrame = _intervals_to_points(
            starts, item_ends, key=key, time_col=time_col
        )
        return pts

    dopamine = infusion_points(cfg.dopamine)
    if dopamine.height:
        fixed = cfg.cardiovascular_fixed_tier.get("dopamine")
        parts.append(
            _scored(
                dopamine,
                pl.lit(fixed)
                if fixed is not None
                else pl.when(pl.col("value") > 15)
                .then(4)
                .when(pl.col("value") > 5)
                .then(3)
                .otherwise(2),
                "cardiovascular",
                key,
                time_col,
            )
        )
    dobutamine = infusion_points(cfg.dobutamine)
    if dobutamine.height:
        # Dobutamine has no rate-dependent tier in SOFA (any dose scores 2),
        # so it is always fixed -- every SOFA_SOURCE_CONFIG entry with a
        # non-empty dobutamine tuple must set this key.
        parts.append(
            _scored(
                dobutamine,
                pl.lit(cfg.cardiovascular_fixed_tier["dobutamine"]),
                "cardiovascular",
                key,
                time_col,
            )
        )
    for drug, items in (
        ("epinephrine", cfg.epinephrine),
        ("norepinephrine", cfg.norepinephrine),
    ):
        pts = infusion_points(items)
        if pts.height:
            fixed = cfg.cardiovascular_fixed_tier.get(drug)
            parts.append(
                _scored(
                    pts,
                    pl.lit(fixed)
                    if fixed is not None
                    else pl.when(pl.col("value") > 0.1).then(4).otherwise(3),
                    "cardiovascular",
                    key,
                    time_col,
                )
            )

    # -- respiration: PaO2 / FiO2 (FiO2 within the previous 4h), ventilation
    pf = pf_ratio_readings(
        events,
        source=source,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    if pf.height:
        vent_flag = pl.col("ventilated")
        score = (
            pl.when(vent_flag & (pl.col("value") < 100))
            .then(4)
            .when(vent_flag & (pl.col("value") < 200))
            .then(3)
            .when(pl.col("value") < 300)
            .then(2)
            .when(pl.col("value") < 400)
            .then(1)
            .otherwise(0)
        )
        parts.append(_scored(pf, score, "respiration", key, time_col))

    # -- cns: GCS total from the paired components
    gcs = gcs_total_readings(
        events,
        source=source,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    if gcs.height:
        parts.append(
            _scored(
                gcs,
                _band(
                    pl.col("value"),
                    [(15, 1), (13, 2), (10, 3), (6, 4)],
                    ascending=False,
                ),
                "cns",
                key,
                time_col,
            )
        )

    # -- renal: urine output < 500 / < 200 mL over the trailing 24h
    rolled = urine_output_24h(
        events,
        source=source,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    ).rename({"value": "_uo24"})
    if rolled.height:
        parts.append(
            _scored(
                rolled,
                pl.when(pl.col("_uo24") < 200)
                .then(4)
                .when(pl.col("_uo24") < 500)
                .then(3)
                .otherwise(0),
                "renal",
                key,
                time_col,
            )
        )

    schema = {
        key: events.schema[key],
        time_col: events.schema[time_col],
        "component": pl.Utf8,
        "score": pl.Int8,
    }
    frames = [p.select(list(schema)).cast(schema) for p in parts if p.height > 0]  # type: ignore[arg-type]
    if not frames:
        return pl.DataFrame(schema=schema)  # type: ignore[arg-type]
    return pl.concat(frames).sort([key, time_col])


def assessable_keys(
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    key: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
) -> set[object]:
    """Keys with at least one numeric reading of any SOFA ingredient.

    The "observed" mask for SOFA-derived concepts: a normal platelet count
    is a reading (the score could be assessed and was 0), whereas a visit
    with none of the ingredients charted cannot be scored at all.
    """
    prefixes = _loinc_prefixes(
        [
            _PAO2,
            _PLATELETS,
            _BILIRUBIN,
            *_MAP,
            _GCS_EYE,
            _GCS_VERBAL,
            _GCS_MOTOR,
            _CREATININE,
            _URINE,
        ],
        source,
    )
    readings = _numeric(
        events,
        prefixes,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    return set(readings[key].to_list())


@_quiet_asof
def pf_ratio_readings(
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    key: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
    fio2_tolerance: str = "4h",
) -> pl.DataFrame:
    """(key, time, value=PaO2/FiO2, ventilated) per arterial blood gas.

    Each PaO2 is paired with the most recent FiO2 charted within
    ``fio2_tolerance`` before it (no FiO2 in range: the gas is not
    assessable and is dropped); FiO2 is accepted as a percentage (21-100)
    or a fraction (0.21-1.0). ``ventilated`` is True when an invasive or
    non-invasive ventilation episode
    (:class:`SofaSourceConfig`) is active at that time, which the SOFA
    respiration bands need for the two most severe levels.
    """
    cfg = SOFA_SOURCE_CONFIG[source]

    def num(loincs: Sequence[str]) -> pl.DataFrame:
        return _numeric(
            events,
            _loinc_prefixes(loincs, source),
            key=key,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        )

    pao2 = num([_PAO2])
    fio2 = num([_FIO2]).rename({"value": "_fio2"})
    if pao2.height == 0 or fio2.height == 0:
        return pao2.head(0).with_columns(pl.lit(False).alias("ventilated"))
    pf = pao2.join_asof(
        fio2, on=time_col, by=key, strategy="backward", tolerance=fio2_tolerance
    ).filter(pl.col("_fio2").is_not_null() & (pl.col("_fio2") > 0))
    pf = pf.with_columns(
        pl.when(pl.col("_fio2") > 1.0)
        .then(pl.col("_fio2") / 100.0)
        .otherwise(pl.col("_fio2"))
        .alias("_fio2")
    ).with_columns((pl.col("value") / pl.col("_fio2")).alias("value"))
    vent = _intervals_to_points(
        events.filter(_starts_with_any(code_col, cfg.ventilation_start)).select(
            key, time_col
        ),
        events.filter(_starts_with_any(code_col, cfg.ventilation_end)).select(
            key, time_col
        ),
        key=key,
        time_col=time_col,
    )
    if vent.height:
        vent = vent.with_columns(pl.lit(True).alias("ventilated")).sort([key, time_col])
        pf = pf.sort([key, time_col]).join_asof(
            vent, on=time_col, by=key, strategy="backward", tolerance="1h"
        )
        pf = pf.with_columns(pl.col("ventilated").fill_null(False))
    else:
        pf = pf.with_columns(pl.lit(False).alias("ventilated"))
    return pf.select(key, time_col, "value", "ventilated")


@_quiet_asof
def urine_output_rate(
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    key: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
    window_hours: float = WINDOW_HOURS,
    weight_normalized: bool = True,
) -> pl.DataFrame:
    """Trailing urine output per reading: (key, time, value).

    ``value`` is millilitres per kilogram per hour (KDIGO's own form)
    when ``weight_normalized``, and plain millilitres over the window
    otherwise (SOFA's renal component and the ``oliguria`` concept, and
    KDIGO Stage 3's anuria branch, which is "0 mL" regardless of weight
    and so must not require one).

    Rows are emitted only at times with at least ``window_hours`` of
    record behind them: a partial window sums less urine simply because
    less time has passed, which would read as oliguria. Multiple
    collection routes (Foley, void, condom cath, suprapubic,
    nephrostomy, ureteral stents) are summed, resolved through the LOINC
    layer, not hardcoded.

    Weight (``weight_normalized=True`` only) is attached by a backward
    ``join_asof``, so each window uses the most recent weight
    charted at or before its end instant, never a later one:

    - the most recent **daily** weight if the key has one by then (it is
      the current weight, which is what a mL/kg/h rate means clinically);
    - otherwise the **admission** weight, the best available estimate
      early in a stay before any daily weight has been charted;
    - otherwise the window is **dropped**, not scored against a default
      or population-average weight. Weight coverage in the real MIMIC-IV
      extraction is poor (~10-17% of subjects have any reading at all),
      so this criterion is genuinely unassessable for most keys --
      inventing a weight would silently turn "unknown" into a gold-
      standard label. A non-positive charted weight (bad data) is
      dropped the same way rather than dividing by it.

    Callers therefore must treat the *absence* of a key from this frame
    as "not assessable", not as "not oliguric" -- the same observability
    convention :func:`assessable_keys` gives the SOFA-derived concepts.

    Known limitation, inherited from the 24 h form and deliberately not
    changed here: a window is summed over whatever urine rows it
    contains, so *sparse charting* is indistinguishable from low output.
    A key charted once a day reads as oliguric (and, if that one row is
    0 mL, as anuric) on the strength of a single row. Guarding it would
    need a minimum-readings-per-window rule, which is a modelling
    decision affecting the existing ``oliguria`` concept too, not
    something to slip in with this change.
    """
    urine = _numeric(
        events,
        _loinc_prefixes([_URINE], source),
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    if urine.height == 0:
        return urine
    window_minutes = int(round(window_hours * 60))
    first_time = (
        events.filter(pl.col(time_col).is_not_null())
        .group_by(key)
        .agg(pl.col(time_col).min().alias("_first"))
    )
    rolled = (
        urine.sort([key, time_col])
        .rolling(index_column=time_col, period=f"{window_minutes}m", group_by=key)
        .agg(pl.col("value").sum().alias("value"))
        .join(first_time, on=key, how="left")
        .filter(
            (pl.col(time_col) - pl.col("_first")) >= pl.duration(minutes=window_minutes)
        )
        .select(key, time_col, "value")
    )
    if not weight_normalized:
        return rolled

    cfg = SOFA_SOURCE_CONFIG[source]
    out = rolled.sort([key, time_col])
    for weight_codes, alias in (
        (cfg.daily_weight, "_w_daily"),
        (cfg.admission_weight, "_w_admission"),
    ):
        weights = _numeric(
            events,
            list(weight_codes),
            key=key,
            code_col=code_col,
            value_col=value_col,
            time_col=time_col,
        ).rename({"value": alias})
        if weights.height == 0:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(alias))
            continue
        out = out.join_asof(
            weights.sort([key, time_col]),
            on=time_col,
            by=key,
            strategy="backward",
        )
    return (
        out.with_columns(pl.coalesce("_w_daily", "_w_admission").alias("_weight"))
        .filter(pl.col("_weight").is_not_null() & (pl.col("_weight") > 0))
        .with_columns(
            (pl.col("value") / pl.col("_weight") / window_hours).alias("value")
        )
        .select(key, time_col, "value")
    )


def urine_output_24h(
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    key: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
) -> pl.DataFrame:
    """(key, time, value=mL voided in the trailing 24h) per urine-output reading.

    The absolute-volume, 24 h special case of
    :func:`urine_output_rate` -- what SOFA's renal component and the
    ``oliguria`` concept score. Weight-normalized rates (mL/kg/h,
    KDIGO's own form) are deliberately *not* used here: weight is not
    reliably present per key in the extractions, and SOFA's own renal
    bands are defined on absolute volume anyway.
    """
    return urine_output_rate(
        events,
        source=source,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
        window_hours=WINDOW_HOURS,
        weight_normalized=False,
    )


@_quiet_asof
def gcs_total_readings(
    events: pl.DataFrame,
    *,
    source: str,
    key: str,
    code_col: str,
    value_col: str,
    time_col: str,
    max_component_gap_minutes: float = 15.0,
) -> pl.DataFrame:
    """(key, time, value=GCS total) from eye/verbal/motor readings paired in time.

    Same pairing as the qSOFA concept's derived-GCS rule (nearest readings
    within ``max_component_gap_minutes``); MIMIC-IV charts no single GCS
    total itemid.
    """
    eye = _numeric(
        events,
        _loinc_prefixes([_GCS_EYE], source),
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    verbal = _numeric(
        events,
        _loinc_prefixes([_GCS_VERBAL], source),
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    motor = _numeric(
        events,
        _loinc_prefixes([_GCS_MOTOR], source),
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    if eye.height == 0 or verbal.height == 0 or motor.height == 0:
        return eye.head(0)
    tol = f"{int(max_component_gap_minutes)}m"
    paired = (
        eye.join_asof(
            verbal.rename({"value": "_v"}),
            on=time_col,
            by=key,
            strategy="nearest",
            tolerance=tol,
        )
        .filter(pl.col("_v").is_not_null())
        .sort([key, time_col])
        .join_asof(
            motor.rename({"value": "_m"}),
            on=time_col,
            by=key,
            strategy="nearest",
            tolerance=tol,
        )
        .filter(pl.col("_m").is_not_null())
    )
    return paired.select(
        key, time_col, (pl.col("value") + pl.col("_v") + pl.col("_m")).alias("value")
    )


def sofa_timeseries(
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    key: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
    grid_times: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return the running SOFA total per key at every component observation time.

    Returns ``(key, time, <component columns>, sofa)`` where each component
    column is the worst score of that component in the trailing 24 h
    (0 if unobserved) and ``sofa`` is their sum. ``grid_times`` (``key``,
    ``time``) adds extra evaluation instants (e.g. suspected-infection
    times) to the grid; by default the grid is the union of the instants at
    which some component scored > 0 per key. That is exactly the set of
    instants at which the running total can first reach any threshold (a
    trailing-window maximum only rises at a new abnormal reading), so
    first-crossing queries are exact; a value *between* abnormal readings
    (window decay) needs an explicit ``grid_times`` entry.
    """
    obs = component_observations(
        events,
        source=source,
        key=key,
        code_col=code_col,
        value_col=value_col,
        time_col=time_col,
    )
    grid = obs.select(key, time_col)
    if grid_times is not None:
        grid = pl.concat([grid, grid_times.select(key, time_col).cast(grid.schema)])
    grid = grid.unique().sort([key, time_col])
    if grid.height == 0:
        return pl.DataFrame(
            schema={
                key: obs.schema[key],
                time_col: obs.schema[time_col],
                **dict.fromkeys(COMPONENTS, pl.Int8),
                "sofa": pl.Int8,
            }
        )
    # Every grid instant needs a value for every component: cross the grid
    # with the component names (score null), union with the observations,
    # rolling-max per (key, component) over 24h, then read back the grid
    # rows. Null scores sit in the window as "no reading" (max ignores them).
    comps = pl.DataFrame({"component": list(COMPONENTS)})
    grid_rows = grid.join(comps, how="cross").with_columns(
        pl.lit(None, dtype=pl.Int8).alias("score"), pl.lit(True).alias("_grid")
    )
    obs_rows = obs.with_columns(pl.lit(False).alias("_grid"))
    stacked = pl.concat([obs_rows.select(grid_rows.columns), grid_rows]).sort(
        [key, "component", time_col, "_grid"]
    )
    rolled = (
        stacked.rolling(
            index_column=time_col, period="24h", group_by=[key, "component"]
        )
        .agg(
            pl.col("score").max().alias("_worst"),
            pl.col("_grid").last().alias("_is_grid"),
        )
        .with_columns(pl.col("_worst").fill_null(0).cast(pl.Int8))
    )
    # rolling().agg() emits one row per input row; keep the grid rows.
    # Two rows at the same (key, component, time) (an observation and a
    # grid instant) share the same window, so either copy is fine.
    at_grid = (
        rolled.join(
            stacked.filter(pl.col("_grid")).select(key, "component", time_col).unique(),
            on=[key, "component", time_col],
            how="semi",
        )
        .unique(subset=[key, "component", time_col])
        .pivot(on="component", index=[key, time_col], values="_worst")
    )
    for c in COMPONENTS:
        if c not in at_grid.columns:
            at_grid = at_grid.with_columns(pl.lit(0, dtype=pl.Int8).alias(c))
    at_grid = at_grid.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int8) for c in COMPONENTS]
    )
    result: pl.DataFrame = (
        at_grid.with_columns(
            pl.sum_horizontal([pl.col(c) for c in COMPONENTS])
            .cast(pl.Int8)
            .alias("sofa")
        )
        .select(key, time_col, *COMPONENTS, "sofa")
        .sort([key, time_col])
    )
    return result


__all__ = [
    "COMPONENTS",
    "pf_ratio_readings",
    "urine_output_24h",
    "urine_output_rate",
    "assessable_keys",
    "SOFA_SOURCE_CONFIG",
    "SofaSourceConfig",
    "component_observations",
    "gcs_total_readings",
    "sofa_supported",
    "sofa_timeseries",
]
