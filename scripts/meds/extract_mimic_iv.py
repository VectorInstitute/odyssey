"""Extract MIMIC-IV 3.1 tables to MEDS format using polars.

MEDS schema per subject-shard parquet:
    subject_id  int64
    time        timestamp[us, UTC]  (null for static events)
    code        str
    numeric_value float32           (nullable)
    text_value  str                 (nullable)

Usage
-----
python scripts/meds/extract_mimic_iv.py \\
    --mimic_dir  data/physionet.org/files/mimiciv/3.1 \\
    --output_dir data/meds \\
    --n_subjects 0          # 0 = all subjects

The output directory will contain:
    data/meds/
        metadata/
            dataset.json        (MEDS dataset metadata)
            codes.parquet       (per-code statistics)
        data/
            train/0000000.parquet
            train/0000001.parquet
            ...
            tuning/0000000.parquet
            held_out/0000000.parquet
"""

import argparse
import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── code prefix conventions ──────────────────────────────────────────────────
ADMISSION_CODE = "ADMISSION//HOSPITAL"
DISCHARGE_CODE = "DISCHARGE//HOSPITAL"
ED_ADMISSION_CODE = "ADMISSION//ED"
ICU_ADMISSION_CODE = "ADMISSION//ICU"
ICU_DISCHARGE_CODE = "DISCHARGE//ICU"
BIRTH_CODE = "MEDS_BIRTH"
DEATH_CODE = "MEDS_DEATH"
GENDER_PREFIX = "GENDER//"
ETHNICITY_PREFIX = "ETHNICITY//"
INSURANCE_PREFIX = "INSURANCE//"
ICD9DX_PREFIX = "ICD9CM//"
ICD10DX_PREFIX = "ICD10CM//"
ICD9PX_PREFIX = "ICD9PROC//"
ICD10PX_PREFIX = "ICD10PCS//"
LAB_PREFIX = "LAB//"
MED_PREFIX = "MEDICATION//"
DRG_PREFIX = "DRG//"


def _read_gz(path: Path, **kwargs: object) -> pl.LazyFrame:
    return pl.scan_csv(str(path), **kwargs)  # type: ignore[arg-type]


def extract_patients(hosp_dir: Path, n_subjects: int = 0) -> pl.LazyFrame:
    """Map patients.csv.gz → MEDS rows (birth, death, gender, ethnicity)."""
    df = _read_gz(
        hosp_dir / "patients.csv.gz",
        schema_overrides={"dod": pl.Utf8},
    )

    # DOB approximated as Jan 1 of (anchor_year - anchor_age)
    birth = df.with_columns(
        [
            (pl.lit(None).cast(pl.Utf8)).alias("text_value"),
            pl.lit(None).cast(pl.Float32).alias("numeric_value"),
            pl.col("subject_id").cast(pl.Int64),
            # DOB approximated as Jan 1 of (anchor_year - anchor_age)
            (pl.col("anchor_year") - pl.col("anchor_age"))
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime("us", "UTC"), format="%Y", strict=False)
            .alias("time"),
            pl.lit(BIRTH_CODE).alias("code"),
        ]
    ).select(["subject_id", "time", "code", "numeric_value", "text_value"])

    gender = df.with_columns(
        [
            pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("time"),  # static
            (pl.lit(GENDER_PREFIX) + pl.col("gender").str.to_uppercase()).alias("code"),
            pl.lit(None).cast(pl.Float32).alias("numeric_value"),
            pl.lit(None).cast(pl.Utf8).alias("text_value"),
            pl.col("subject_id").cast(pl.Int64),
        ]
    ).select(["subject_id", "time", "code", "numeric_value", "text_value"])

    death = (
        df.filter(pl.col("dod").is_not_null())
        .with_columns(
            [
                pl.col("subject_id").cast(pl.Int64),
                pl.col("dod")
                .str.strptime(pl.Datetime("us", "UTC"), format="%Y-%m-%d", strict=False)
                .alias("time"),
                pl.lit(DEATH_CODE).alias("code"),
                pl.lit(None).cast(pl.Float32).alias("numeric_value"),
                pl.lit(None).cast(pl.Utf8).alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )

    out = pl.concat([birth, gender, death])
    if n_subjects > 0:
        ids = df.select("subject_id").limit(n_subjects).collect()["subject_id"]
        out = out.filter(pl.col("subject_id").is_in(ids))
    return out


def extract_admissions(hosp_dir: Path) -> pl.LazyFrame:
    """Map admissions.csv.gz → MEDS rows (admission, discharge, ethnicity, insurance).

    Produces admission, discharge, ethnicity, and insurance events.
    """
    df = _read_gz(
        hosp_dir / "admissions.csv.gz",
        schema_overrides={
            "admittime": pl.Utf8,
            "dischtime": pl.Utf8,
            "deathtime": pl.Utf8,
        },
    ).with_columns(pl.col("subject_id").cast(pl.Int64))

    admit = df.with_columns(
        [
            pl.col("admittime")
            .str.strptime(
                pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
            )
            .alias("time"),
            pl.lit(ADMISSION_CODE).alias("code"),
            pl.lit(None).cast(pl.Float32).alias("numeric_value"),
            pl.col("admission_location").alias("text_value"),
        ]
    ).select(["subject_id", "time", "code", "numeric_value", "text_value"])

    discharge = (
        df.filter(pl.col("dischtime").is_not_null())
        .with_columns(
            [
                pl.col("dischtime")
                .str.strptime(
                    pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
                )
                .alias("time"),
                pl.lit(DISCHARGE_CODE).alias("code"),
                pl.lit(None).cast(pl.Float32).alias("numeric_value"),
                pl.col("discharge_location").alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )

    ethnicity = df.with_columns(
        [
            pl.col("admittime")
            .str.strptime(
                pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
            )
            .alias("time"),
            (pl.lit(ETHNICITY_PREFIX) + pl.col("race").str.to_uppercase()).alias(
                "code"
            ),
            pl.lit(None).cast(pl.Float32).alias("numeric_value"),
            pl.lit(None).cast(pl.Utf8).alias("text_value"),
        ]
    ).select(["subject_id", "time", "code", "numeric_value", "text_value"])

    insurance = df.with_columns(
        [
            pl.col("admittime")
            .str.strptime(
                pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
            )
            .alias("time"),
            (pl.lit(INSURANCE_PREFIX) + pl.col("insurance").str.to_uppercase()).alias(
                "code"
            ),
            pl.lit(None).cast(pl.Float32).alias("numeric_value"),
            pl.lit(None).cast(pl.Utf8).alias("text_value"),
        ]
    ).select(["subject_id", "time", "code", "numeric_value", "text_value"])

    return pl.concat([admit, discharge, ethnicity, insurance])


def extract_diagnoses(hosp_dir: Path) -> pl.LazyFrame:
    """Map diagnoses_icd.csv.gz → MEDS rows (ICD codes at discharge time)."""
    diag = _read_gz(hosp_dir / "diagnoses_icd.csv.gz").with_columns(
        pl.col("subject_id").cast(pl.Int64)
    )
    admissions = _read_gz(
        hosp_dir / "admissions.csv.gz",
        schema_overrides={"dischtime": pl.Utf8},
    ).with_columns(pl.col("hadm_id").cast(pl.Int64))

    adm_times = admissions.select(
        [
            "hadm_id",
            pl.col("dischtime")
            .str.strptime(
                pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
            )
            .alias("time"),
        ]
    )

    return (
        diag.join(adm_times, on="hadm_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("icd_version") == 9)
                .then(pl.lit(ICD9DX_PREFIX) + pl.col("icd_code"))
                .otherwise(pl.lit(ICD10DX_PREFIX) + pl.col("icd_code"))
                .alias("code"),
                pl.lit(None).cast(pl.Float32).alias("numeric_value"),
                pl.lit(None).cast(pl.Utf8).alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )


def extract_procedures(hosp_dir: Path) -> pl.LazyFrame:
    """Map procedures_icd.csv.gz → MEDS rows (ICD procedure codes at discharge)."""
    proc = _read_gz(hosp_dir / "procedures_icd.csv.gz").with_columns(
        pl.col("subject_id").cast(pl.Int64)
    )
    admissions = _read_gz(
        hosp_dir / "admissions.csv.gz",
        schema_overrides={"dischtime": pl.Utf8},
    )
    adm_times = admissions.select(
        [
            pl.col("hadm_id").cast(pl.Int64),
            pl.col("dischtime")
            .str.strptime(
                pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
            )
            .alias("time"),
        ]
    )

    return (
        proc.join(adm_times, on="hadm_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("icd_version") == 9)
                .then(pl.lit(ICD9PX_PREFIX) + pl.col("icd_code"))
                .otherwise(pl.lit(ICD10PX_PREFIX) + pl.col("icd_code"))
                .alias("code"),
                pl.lit(None).cast(pl.Float32).alias("numeric_value"),
                pl.lit(None).cast(pl.Utf8).alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )


def extract_labevents(hosp_dir: Path) -> Optional[pl.LazyFrame]:
    """Map labevents.csv.gz → MEDS rows (lab results)."""
    lab_path = hosp_dir / "labevents.csv.gz"
    if not lab_path.exists():
        log.warning("labevents.csv.gz not found — skipping")
        return None

    d_items = _read_gz(hosp_dir / "d_labitems.csv.gz").select(
        [pl.col("itemid").cast(pl.Int64), "label"]
    )

    labs = _read_gz(
        lab_path,
        schema_overrides={"charttime": pl.Utf8, "valuenum": pl.Utf8},
    ).with_columns(
        [pl.col("subject_id").cast(pl.Int64), pl.col("itemid").cast(pl.Int64)]
    )

    return (
        labs.join(d_items, on="itemid", how="left")
        .with_columns(
            [
                pl.col("charttime")
                .str.strptime(
                    pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
                )
                .alias("time"),
                (pl.lit(LAB_PREFIX) + pl.col("label").fill_null("UNKNOWN")).alias(
                    "code"
                ),
                pl.col("valuenum")
                .cast(pl.Float32, strict=False)
                .alias("numeric_value"),
                pl.col("value").alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )


def extract_prescriptions(hosp_dir: Path) -> Optional[pl.LazyFrame]:
    """Map prescriptions.csv.gz → MEDS rows (medications)."""
    rx_path = hosp_dir / "prescriptions.csv.gz"
    if not rx_path.exists():
        log.warning("prescriptions.csv.gz not found — skipping")
        return None

    return (
        _read_gz(rx_path, schema_overrides={"starttime": pl.Utf8})
        .with_columns(
            [
                pl.col("subject_id").cast(pl.Int64),
                pl.col("starttime")
                .str.strptime(
                    pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
                )
                .alias("time"),
                (
                    pl.lit(MED_PREFIX)
                    + pl.col("drug").fill_null("UNKNOWN").str.to_uppercase()
                ).alias("code"),
                pl.col("dose_val_rx")
                .cast(pl.Float32, strict=False)
                .alias("numeric_value"),
                pl.col("route").alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )


def extract_drgcodes(hosp_dir: Path) -> pl.LazyFrame:
    """Map drgcodes.csv.gz → MEDS rows (DRG codes)."""
    admissions = _read_gz(
        hosp_dir / "admissions.csv.gz",
        schema_overrides={"dischtime": pl.Utf8},
    )
    adm_times = admissions.select(
        [
            pl.col("hadm_id").cast(pl.Int64),
            pl.col("dischtime")
            .str.strptime(
                pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
            )
            .alias("time"),
        ]
    )

    return (
        _read_gz(hosp_dir / "drgcodes.csv.gz")
        .with_columns(
            [
                pl.col("subject_id").cast(pl.Int64),
                pl.col("hadm_id").cast(pl.Int64),
            ]
        )
        .join(adm_times, on="hadm_id", how="left")
        .with_columns(
            [
                (
                    pl.lit(DRG_PREFIX)
                    + pl.col("drg_type").fill_null("")
                    + pl.lit("//")
                    + pl.col("drg_code").cast(pl.Utf8)
                ).alias("code"),
                pl.lit(None).cast(pl.Float32).alias("numeric_value"),
                pl.col("description").alias("text_value"),
            ]
        )
        .select(["subject_id", "time", "code", "numeric_value", "text_value"])
    )


# ── sharding ─────────────────────────────────────────────────────────────────

MEDS_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us", "UTC"),
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "text_value": pl.Utf8,
}

SPLITS = {"train": 0.8, "tuning": 0.1, "held_out": 0.1}
SUBJECTS_PER_SHARD = 10_000


def write_shards(
    df: pl.DataFrame, output_dir: Path, split: str, n_per_shard: int
) -> None:
    """Write subject-sorted MEDS data as numbered shards.

    Output path: output_dir/data/{split}/{shard_idx:07d}.parquet
    """
    shard_dir = output_dir / "data" / split
    shard_dir.mkdir(parents=True, exist_ok=True)

    subjects = df["subject_id"].unique().sort()
    n_shards = max(1, math.ceil(len(subjects) / n_per_shard))

    for shard_idx in range(n_shards):
        shard_subjects = subjects[
            shard_idx * n_per_shard : (shard_idx + 1) * n_per_shard
        ]
        shard = df.filter(pl.col("subject_id").is_in(shard_subjects)).sort(
            ["subject_id", "time"], nulls_last=True
        )
        out_path = shard_dir / f"{shard_idx:07d}.parquet"
        shard.write_parquet(out_path)
        log.info(
            "  wrote %s (%d rows, %d subjects)",
            out_path.name,
            len(shard),
            len(shard_subjects),
        )


def write_metadata(df: pl.DataFrame, output_dir: Path) -> None:
    """Write MEDS dataset metadata JSON and code statistics parquet."""
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = df["subject_id"].n_unique()
    n_events = len(df)

    metadata = {
        "dataset_name": "MIMIC-IV",
        "dataset_version": "3.1",
        "etl_name": "odyssey/scripts/meds/extract_mimic_iv.py",
        "etl_version": "0.1.0",
        "meds_version": "0.3.x",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_subjects": n_subjects,
        "n_events": n_events,
    }
    with open(meta_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)

    code_stats = (
        df.group_by("code")
        .agg(
            [
                pl.len().alias("n_occurrences"),
                pl.col("subject_id").n_unique().alias("n_subjects"),
                pl.col("numeric_value").mean().alias("mean_numeric_value"),
            ]
        )
        .sort("n_occurrences", descending=True)
    )
    code_stats.write_parquet(meta_dir / "codes.parquet")
    log.info("Metadata written to %s", meta_dir)


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Extract MIMIC-IV to MEDS format")
    p.add_argument(
        "--mimic_dir",
        required=True,
        help="Root MIMIC-IV 3.1 directory (containing hosp/ and icu/)",
    )
    p.add_argument("--output_dir", required=True, help="Output directory for MEDS data")
    p.add_argument(
        "--n_subjects",
        type=int,
        default=0,
        help="Limit to first N subjects (0 = all, useful for dev runs)",
    )
    p.add_argument(
        "--subjects_per_shard",
        type=int,
        default=SUBJECTS_PER_SHARD,
        help="Max subjects per shard parquet file",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/tune/held_out split",
    )
    return p.parse_args()


def main() -> None:
    """Run the MIMIC-IV to MEDS extraction pipeline."""
    args = parse_args()
    mimic_dir = Path(args.mimic_dir)
    hosp_dir = mimic_dir / "hosp"
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        log.warning("Output dir %s exists — clearing it", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ── extract each table ────────────────────────────────────────────────────
    frames = []

    log.info("Extracting patients …")
    frames.append(extract_patients(hosp_dir, args.n_subjects))

    log.info("Extracting admissions …")
    frames.append(extract_admissions(hosp_dir))

    if (hosp_dir / "diagnoses_icd.csv.gz").exists():
        log.info("Extracting diagnoses …")
        frames.append(extract_diagnoses(hosp_dir))

    if (hosp_dir / "procedures_icd.csv.gz").exists():
        log.info("Extracting procedures …")
        frames.append(extract_procedures(hosp_dir))

    if (hosp_dir / "drgcodes.csv.gz").exists():
        log.info("Extracting DRG codes …")
        frames.append(extract_drgcodes(hosp_dir))

    lab_frame = extract_labevents(hosp_dir)
    if lab_frame is not None:
        log.info("Extracting lab events …")
        frames.append(lab_frame)

    rx_frame = extract_prescriptions(hosp_dir)
    if rx_frame is not None:
        log.info("Extracting prescriptions …")
        frames.append(rx_frame)

    # ── collect & cast ────────────────────────────────────────────────────────
    log.info("Collecting all events …")
    all_events = pl.concat(frames).collect()

    # Cast to canonical MEDS schema
    all_events = all_events.cast(
        {
            "subject_id": pl.Int64,
            "time": pl.Datetime("us", "UTC"),
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
            "text_value": pl.Utf8,
        }
    )

    if args.n_subjects > 0:
        kept_ids = all_events["subject_id"].unique().sort()[: args.n_subjects]
        all_events = all_events.filter(pl.col("subject_id").is_in(kept_ids))

    log.info(
        "Total events: %d across %d subjects",
        len(all_events),
        all_events["subject_id"].n_unique(),
    )

    # ── split subjects ────────────────────────────────────────────────────────
    all_subjects = all_events["subject_id"].unique().shuffle(seed=args.seed).to_list()
    n = len(all_subjects)
    n_train = int(n * SPLITS["train"])
    n_tune = int(n * SPLITS["tuning"])

    train_ids = set(all_subjects[:n_train])
    tune_ids = set(all_subjects[n_train : n_train + n_tune])
    held_ids = set(all_subjects[n_train + n_tune :])

    # ── write shards ──────────────────────────────────────────────────────────
    for split_name, id_set in [
        ("train", train_ids),
        ("tuning", tune_ids),
        ("held_out", held_ids),
    ]:
        log.info("Writing %s split (%d subjects) …", split_name, len(id_set))
        split_df = all_events.filter(pl.col("subject_id").is_in(id_set))
        write_shards(split_df, output_dir, split_name, args.subjects_per_shard)

    # ── metadata ──────────────────────────────────────────────────────────────
    write_metadata(all_events, output_dir)
    log.info("Done. MEDS data written to %s", output_dir)


if __name__ == "__main__":
    main()
