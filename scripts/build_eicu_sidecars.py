"""Build label-only sidecar tables for the eICU-CRD MEDS extraction.

Same role as :mod:`scripts.build_mimic_sidecars`: the Sepsis-3 label anchors
"suspected infection" on a culture being drawn near a systemic antibiotic
start, and neither ingredient is tokenized. Both are carried as small
parquets next to the MEDS ``data/`` directory, read only by the label
pipeline (:mod:`odyssey.data.sidecars`), never by any model or baseline.

eICU stores minute offsets, not timestamps, so every sidecar time is the
same pseudotime the extraction (``specs/eICU.yaml``) assigns: hospital
discharge anchored at Dec 31 of ``hospitaldischargeyear`` at
``hospitaldischargetime24``, unit admission = discharge minus
``hospitaldischargeoffset``, and each event = unit admission plus its own
offset, with the same +/- 1 year guard on garbage offsets. Subjects are
``patienthealthsystemstayid`` and ``hadm_id`` is ``patientunitstayid``,
matching the extraction.

Usage::

    uv run python scripts/build_eicu_sidecars.py \
        --eicu-root /Volumes/clinical-data/physionet.org/files/eicu-crd/2.0 \
        --meds-root /Volumes/clinical-data/meds/eicu_2.0_v2

writes ``<meds-root>/sidecars/microbiology.parquet`` (one row per
specimen: ``subject_id``, ``hadm_id``, ``time``, ``spec_type_desc``,
``positive_culture``, ``micro_specimen_id``) and
``antibiotic_orders.parquet`` (``subject_id, hadm_id, time, stoptime,
drug, route``). A specimen is one (unit stay, offset, site); it is
positive when any organism is named (eICU's "no growth" is the negative).
Orders with no drug name are resolved through the HICL ingredient
dictionary the code normalizer uses, so the 36% null-name rows are not
lost; cancelled orders are dropped.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from odyssey.data.concepts import ANTIBIOTIC_PATTERN


logger = logging.getLogger("build_eicu_sidecars")

# Offsets at or beyond +/- one year null the derived time (specs/eICU.yaml).
OFFSET_GUARD_MINUTES = 525_600
HICL_DICTIONARY = (
    Path(__file__).resolve().parent.parent
    / "odyssey"
    / "data"
    / "resources"
    / "eicu_hicl_ingredients.csv"
)
NEGATIVE_ORGANISMS = ("", "no growth")


def _offset_time(anchor: str, offset: str) -> pl.Expr:
    """``anchor + offset`` minutes, null when the offset is outside the guard."""
    off = pl.col(offset).cast(pl.Int64)
    return (
        pl.when(off.abs() >= OFFSET_GUARD_MINUTES)
        .then(None)
        .otherwise(pl.col(anchor) + pl.duration(minutes=off))
    )


def unit_admissions(eicu_root: Path) -> pl.DataFrame:
    """Per unit stay: subject id and the pseudotime of unit admission."""
    patient = pl.read_csv(
        eicu_root / "patient.csv.gz",
        columns=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "hospitaldischargeyear",
            "hospitaldischargetime24",
            "hospitaldischargeoffset",
        ],
        schema_overrides={
            "patientunitstayid": pl.Int64,
            "patienthealthsystemstayid": pl.Int64,
            "hospitaldischargeyear": pl.Int64,
            "hospitaldischargetime24": pl.Utf8,
            "hospitaldischargeoffset": pl.Int64,
        },
    )
    discharge = (
        pl.col("hospitaldischargeyear").cast(pl.Utf8)
        + pl.lit("-12-31 ")
        + pl.col("hospitaldischargetime24")
    ).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False)
    return (
        patient.with_columns(discharge.alias("_discharge"))
        .with_columns(
            pl.when(pl.col("hospitaldischargeoffset").abs() >= OFFSET_GUARD_MINUTES)
            .then(None)
            .otherwise(
                pl.col("_discharge")
                - pl.duration(minutes=pl.col("hospitaldischargeoffset"))
            )
            .alias("unitadmit")
        )
        .select(
            pl.col("patientunitstayid").alias("hadm_id"),
            pl.col("patienthealthsystemstayid").alias("subject_id"),
            "unitadmit",
        )
        .filter(pl.col("unitadmit").is_not_null())
    )


def build_microbiology_sidecar(eicu_root: Path, stays: pl.DataFrame) -> pl.DataFrame:
    """One row per culture specimen: who, which unit stay, when, site, positive."""
    micro = pl.read_csv(
        eicu_root / "microLab.csv.gz",
        columns=[
            "microlabid",
            "patientunitstayid",
            "culturetakenoffset",
            "culturesite",
            "organism",
        ],
        schema_overrides={
            "microlabid": pl.Int64,
            "patientunitstayid": pl.Int64,
            "culturetakenoffset": pl.Int64,
            "culturesite": pl.Utf8,
            "organism": pl.Utf8,
        },
    )
    positive = ~pl.col("organism").fill_null("").str.to_lowercase().is_in(
        list(NEGATIVE_ORGANISMS)
    )
    return (
        micro.with_columns(positive.alias("_pos"))
        .group_by(["patientunitstayid", "culturetakenoffset", "culturesite"])
        .agg(
            pl.col("microlabid").min().alias("micro_specimen_id"),
            pl.col("_pos").max().alias("positive_culture"),
        )
        .join(stays, left_on="patientunitstayid", right_on="hadm_id", how="inner")
        .with_columns(_offset_time("unitadmit", "culturetakenoffset").alias("time"))
        .filter(pl.col("time").is_not_null())
        .select(
            "subject_id",
            pl.col("patientunitstayid").alias("hadm_id"),
            "time",
            pl.col("culturesite").alias("spec_type_desc"),
            "positive_culture",
            "micro_specimen_id",
        )
        .sort(["subject_id", "time"])
    )


def build_antibiotic_orders_sidecar(
    eicu_root: Path, stays: pl.DataFrame, hicl_dictionary: Path = HICL_DICTIONARY
) -> pl.DataFrame:
    """Antibacterial medication ORDERS, one row per order, by name or HICL."""
    meds = pl.read_csv(
        eicu_root / "medication.csv.gz",
        columns=[
            "patientunitstayid",
            "drugstartoffset",
            "drugstopoffset",
            "drugordercancelled",
            "drugname",
            "drughiclseqno",
            "routeadmin",
        ],
        schema_overrides={
            "patientunitstayid": pl.Int64,
            "drugstartoffset": pl.Int64,
            "drugstopoffset": pl.Int64,
            "drugordercancelled": pl.Utf8,
            "drugname": pl.Utf8,
            "drughiclseqno": pl.Int64,
            "routeadmin": pl.Utf8,
        },
    )
    hicl = pl.read_csv(
        hicl_dictionary, schema_overrides={"hicl": pl.Int64, "ingredient": pl.Utf8}
    ).select("hicl", "ingredient")
    return (
        meds.filter(pl.col("drugordercancelled").fill_null("No") != "Yes")
        .join(hicl, left_on="drughiclseqno", right_on="hicl", how="left")
        .with_columns(
            pl.coalesce([pl.col("drugname"), pl.col("ingredient")]).alias("drug")
        )
        .filter(pl.col("drug").str.contains("(?i)" + ANTIBIOTIC_PATTERN))
        .join(stays, left_on="patientunitstayid", right_on="hadm_id", how="inner")
        .with_columns(
            _offset_time("unitadmit", "drugstartoffset").alias("time"),
            _offset_time("unitadmit", "drugstopoffset").alias("stoptime"),
        )
        .filter(pl.col("time").is_not_null())
        .select(
            "subject_id",
            pl.col("patientunitstayid").alias("hadm_id"),
            "time",
            "stoptime",
            "drug",
            pl.col("routeadmin").alias("route"),
        )
        .sort(["subject_id", "time"])
    )


def main() -> None:
    """Write both sidecars next to the MEDS data directory."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eicu-root", type=Path, required=True)
    parser.add_argument("--meds-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    stays = unit_admissions(args.eicu_root)
    logger.info("%d unit stays with a derivable admission time", stays.height)
    out_dir = args.meds_root / "sidecars"
    out_dir.mkdir(parents=True, exist_ok=True)

    micro = build_microbiology_sidecar(args.eicu_root, stays)
    micro.write_parquet(out_dir / "microbiology.parquet")
    logger.info(
        "wrote %s: %d specimens, %.1f%% positive, %d subjects",
        out_dir / "microbiology.parquet",
        micro.height,
        100 * micro.filter(pl.col("positive_culture")).height / max(micro.height, 1),
        micro["subject_id"].n_unique(),
    )
    orders = build_antibiotic_orders_sidecar(args.eicu_root, stays)
    orders.write_parquet(out_dir / "antibiotic_orders.parquet")
    logger.info(
        "wrote %s: %d antibacterial orders, %d subjects",
        out_dir / "antibiotic_orders.parquet",
        orders.height,
        orders["subject_id"].n_unique(),
    )


if __name__ == "__main__":
    main()
