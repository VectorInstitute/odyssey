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
import random
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
    """Map labevents.csv.gz → MEDS rows (lab results).

    d_labitems is collected eagerly (tiny: ~900 rows) so that the main
    labevents plan contains no join.  Joins prevent polars sink_parquet
    from using its streaming engine, causing the full 168 M-row table to
    be materialised in RAM.  Using Expr.replace() instead keeps the plan
    fully streaming-compatible.
    """
    lab_path = hosp_dir / "labevents.csv.gz"
    if not lab_path.exists():
        log.warning("labevents.csv.gz not found — skipping")
        return None

    # Collect d_labitems eagerly — it is ~900 rows, negligible memory.
    d_items_df = (
        _read_gz(hosp_dir / "d_labitems.csv.gz")
        .select([pl.col("itemid").cast(pl.Int64), "label"])
        .collect()
    )
    itemids: list[int] = d_items_df["itemid"].to_list()
    labels: list[str] = d_items_df["label"].fill_null("UNKNOWN").to_list()

    labs = _read_gz(
        lab_path,
        schema_overrides={"charttime": pl.Utf8, "valuenum": pl.Utf8},
        low_memory=True,
    ).with_columns(
        [pl.col("subject_id").cast(pl.Int64), pl.col("itemid").cast(pl.Int64)]
    )

    return (
        labs
        # replace() is streaming-compatible; join() is not
        .with_columns(
            pl.col("itemid")
            .replace(old=itemids, new=labels, default="UNKNOWN")
            .alias("label")
        )
        .with_columns(
            [
                pl.col("charttime")
                .str.strptime(
                    pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S", strict=False
                )
                .alias("time"),
                (pl.lit(LAB_PREFIX) + pl.col("label")).alias("code"),
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
        _read_gz(rx_path, schema_overrides={"starttime": pl.Utf8}, low_memory=True)
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

MEDS_SCHEMA: dict[str, pl.PolarsDataType] = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us", "UTC"),
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "text_value": pl.Utf8,
}

SPLITS = {"train": 0.8, "tuning": 0.1, "held_out": 0.1}
SUBJECTS_PER_SHARD = 10_000


def write_metadata(output_dir: Path, n_subjects: int, n_events: int) -> None:
    """Write MEDS dataset metadata JSON and code statistics parquet.

    Scans final shard parquets rather than loading the full dataset.
    """
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

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

    all_parquets = sorted((output_dir / "data").glob("**/*.parquet"))
    code_stats = (
        pl.scan_parquet([str(p) for p in all_parquets])
        .group_by("code")
        .agg(
            [
                pl.len().alias("n_occurrences"),
                pl.col("subject_id").n_unique().alias("n_subjects"),
                pl.col("numeric_value").mean().alias("mean_numeric_value"),
            ]
        )
        .sort("n_occurrences", descending=True)
        .collect()
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


def _subject_to_shard_map(
    all_subjects: list[int],
    n_per_shard: int,
    seed: int,
) -> tuple[dict[int, str], dict[int, int], dict[str, list[int]]]:
    """Return (subject→split, subject→shard_idx, split→sorted_subjects)."""
    rng = random.Random(seed)
    shuffled = all_subjects[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLITS["train"])
    n_tune = int(n * SPLITS["tuning"])

    split_of: dict[int, str] = {}
    for sid in shuffled[:n_train]:
        split_of[sid] = "train"
    for sid in shuffled[n_train : n_train + n_tune]:
        split_of[sid] = "tuning"
    for sid in shuffled[n_train + n_tune :]:
        split_of[sid] = "held_out"

    # Within each split assign shard index by sorted position
    split_subjects: dict[str, list[int]] = {"train": [], "tuning": [], "held_out": []}
    for sid, sp in split_of.items():
        split_subjects[sp].append(sid)
    shard_of: dict[int, int] = {}
    sorted_split_subjects: dict[str, list[int]] = {}
    for sp, sids in split_subjects.items():
        sorted_sids = sorted(sids)
        sorted_split_subjects[sp] = sorted_sids
        for i, sid in enumerate(sorted_sids):
            shard_of[sid] = i // n_per_shard

    return split_of, shard_of, sorted_split_subjects


def _sink_large_table(
    lf: pl.LazyFrame,
    table_name: str,
    temp_dir: Path,
    sorted_split_subjects: dict[str, list[int]],
    n_per_shard: int,
) -> None:
    """Stream a large LazyFrame to a temp parquet, then partition by shard.

    Uses polars sink_parquet (streaming engine) to avoid materialising the
    full table in memory — essential for labevents (~168 M rows).
    """
    full_temp = temp_dir / f"_large_{table_name}.parquet"
    log.info("Streaming %s → temp parquet (no collect) …", table_name)
    lf.sink_parquet(str(full_temp))
    size_mb = full_temp.stat().st_size / 1024**2
    log.info("  %.0f MB written — partitioning into shards …", size_mb)

    for split_name, sids in sorted_split_subjects.items():
        n_shards = max(1, (len(sids) + n_per_shard - 1) // n_per_shard)
        for shard_idx in range(n_shards):
            shard_sids = set(
                sids[shard_idx * n_per_shard : (shard_idx + 1) * n_per_shard]
            )
            shard_df = (
                pl.scan_parquet(str(full_temp))
                .filter(pl.col("subject_id").is_in(shard_sids))
                .collect()
            )
            if len(shard_df) == 0:
                continue
            dest = temp_dir / split_name / f"{shard_idx:07d}"
            dest.mkdir(parents=True, exist_ok=True)
            shard_df.write_parquet(dest / f"{table_name}.parquet")
            del shard_df

    full_temp.unlink()


def _write_table_to_temp(
    df: pl.DataFrame,
    table_name: str,
    temp_dir: Path,
    split_of: dict[int, str],
    shard_of: dict[int, int],
) -> None:
    """Partition df by (split, shard) and write to temp parquet files."""
    sids = df["subject_id"].to_list()
    splits_col = [split_of.get(s) for s in sids]
    shards_col = [shard_of.get(s, 0) for s in sids]

    df = df.with_columns(
        [
            pl.Series("_split", splits_col, dtype=pl.Utf8),
            pl.Series("_shard", shards_col, dtype=pl.Int32),
        ]
    ).filter(pl.col("_split").is_not_null())

    for keys, group in df.group_by(["_split", "_shard"]):
        split_name, shard_idx = str(keys[0]), int(keys[1])  # type: ignore[call-overload]
        dest = temp_dir / split_name / f"{shard_idx:07d}"
        dest.mkdir(parents=True, exist_ok=True)
        group.drop(["_split", "_shard"]).write_parquet(dest / f"{table_name}.parquet")


def _merge_temp_shards(temp_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Merge per-table temp parquets into final sorted MEDS shard files.

    Returns (total_events, total_subjects).
    """
    total_events = 0
    total_subjects = 0

    for split_name in ("train", "tuning", "held_out"):
        split_temp = temp_dir / split_name
        if not split_temp.exists():
            continue
        shard_dirs = sorted(split_temp.iterdir())
        log.info("Writing %s split (%d shards) …", split_name, len(shard_dirs))
        out_dir = output_dir / "data" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for shard_dir in shard_dirs:
            parts = sorted(shard_dir.glob("*.parquet"))
            shard_df = pl.concat([pl.read_parquet(p) for p in parts]).sort(
                ["subject_id", "time"], nulls_last=True
            )
            out_path = out_dir / f"{shard_dir.name}.parquet"
            shard_df.write_parquet(out_path)
            n_subj = shard_df["subject_id"].n_unique()
            log.info(
                "  %s → %d rows, %d subjects", out_path.name, len(shard_df), n_subj
            )
            total_events += len(shard_df)
            total_subjects += n_subj

    return total_events, total_subjects


def main() -> None:
    """Run the MIMIC-IV to MEDS extraction pipeline.

    Processes each source table independently to avoid loading the full
    dataset into memory at once (labevents alone is ~158 M rows).
    """
    args = parse_args()
    mimic_dir = Path(args.mimic_dir)
    hosp_dir = mimic_dir / "hosp"
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        log.warning("Output dir %s exists — clearing it", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    temp_dir = output_dir / "_tmp"
    temp_dir.mkdir()

    # ── Step 1: collect patients (small) to determine splits ─────────────────
    log.info("Extracting patients …")
    patients_df = (
        extract_patients(hosp_dir, args.n_subjects).collect().cast(MEDS_SCHEMA)  # type: ignore[arg-type]
    )
    all_subjects = patients_df["subject_id"].unique().to_list()
    log.info("  %d subjects", len(all_subjects))

    split_of, shard_of, sorted_split_subjects = _subject_to_shard_map(
        all_subjects, args.subjects_per_shard, args.seed
    )
    _write_table_to_temp(patients_df, "patients", temp_dir, split_of, shard_of)
    del patients_df

    # ── Step 2: extract each table one at a time ──────────────────────────────
    subject_filter: Optional[set[int]] = (
        set(all_subjects) if args.n_subjects > 0 else None
    )

    def _collect_and_write(name: str, lf: Optional[pl.LazyFrame]) -> None:
        """Collect small tables (~tens of millions of rows) into memory."""
        if lf is None:
            return
        if subject_filter is not None:
            lf = lf.filter(pl.col("subject_id").is_in(subject_filter))
        log.info("Collecting %s …", name)
        df = lf.collect().cast(MEDS_SCHEMA)  # type: ignore[arg-type]
        log.info("  %d rows", len(df))
        _write_table_to_temp(df, name, temp_dir, split_of, shard_of)
        del df

    def _stream_and_write(name: str, lf: Optional[pl.LazyFrame]) -> None:
        """Stream large tables via sink_parquet to avoid OOM (labevents ~168 M rows)."""
        if lf is None:
            return
        if subject_filter is not None:
            lf = lf.filter(pl.col("subject_id").is_in(subject_filter))
        lf = lf.cast(MEDS_SCHEMA)  # type: ignore[arg-type]
        _sink_large_table(
            lf, name, temp_dir, sorted_split_subjects, args.subjects_per_shard
        )

    _collect_and_write("admissions", extract_admissions(hosp_dir))

    if (hosp_dir / "diagnoses_icd.csv.gz").exists():
        _collect_and_write("diagnoses", extract_diagnoses(hosp_dir))

    if (hosp_dir / "procedures_icd.csv.gz").exists():
        _collect_and_write("procedures", extract_procedures(hosp_dir))

    if (hosp_dir / "drgcodes.csv.gz").exists():
        _collect_and_write("drgcodes", extract_drgcodes(hosp_dir))

    # labevents (~168 M rows) and prescriptions (~20 M rows): use streaming
    _stream_and_write("labevents", extract_labevents(hosp_dir))
    _stream_and_write("prescriptions", extract_prescriptions(hosp_dir))

    # ── Step 3: merge temp shards into final MEDS parquets ───────────────────
    total_events, total_subjects = _merge_temp_shards(temp_dir, output_dir)

    shutil.rmtree(temp_dir)

    # ── Step 4: metadata (scans final parquets, no full collect) ─────────────
    write_metadata(output_dir, n_subjects=total_subjects, n_events=total_events)
    log.info(
        "Done. %d events across %d subjects written to %s",
        total_events,
        total_subjects,
        output_dir,
    )


if __name__ == "__main__":
    main()
