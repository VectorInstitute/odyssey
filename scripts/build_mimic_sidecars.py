"""Build label-only sidecar tables for the MIMIC-IV MEDS extraction.

The standard ``meds-extract`` MIMIC-IV spec (hosp + icu) carries no
``microbiologyevents``, yet the Sepsis-3 label (Seymour et al. 2016 as
operationalized by mimic-code's ``suspicion_of_infection``) anchors
"suspected infection" on a culture being drawn near an antibiotic start.
Rather than re-extract and re-tokenize every dataset version, culture
specimen times are carried as a **sidecar**: a small parquet next to the
MEDS ``data/`` directory, read only by the label pipeline
(:mod:`odyssey.data.sidecars`), never by tokenization or any baseline's
features. Every model family is therefore equally blind to culture draws
as inputs; only the outcome definition uses them.

Usage::

    uv run python scripts/build_mimic_sidecars.py \
        --mimic-root /Volumes/clinical-data/physionet.org/files/mimiciv/3.1 \
        --meds-root /Volumes/clinical-data/meds/mimiciv_3.1_v1

writes ``<meds-root>/sidecars/microbiology.parquet`` with one row per
specimen (``micro_specimen_id``): ``subject_id``, ``hadm_id`` (nullable),
``time`` (``charttime``, falling back to ``chartdate`` at midnight when the
time is missing, as mimic-code does), ``spec_type_desc``, and
``positive_culture`` (any organism named on the specimen). No patient data
leaves the machine; the parquet stays next to the gitignored extraction.
"""

import argparse
import logging
from pathlib import Path

import polars as pl


logger = logging.getLogger("build_mimic_sidecars")


def build_microbiology_sidecar(mimic_root: Path) -> pl.DataFrame:
    """One row per microbiology specimen: who, which admission, when, what."""
    path = mimic_root / "hosp" / "microbiologyevents.csv.gz"
    lf = pl.scan_csv(
        path,
        schema_overrides={
            "subject_id": pl.Int64,
            "hadm_id": pl.Int64,
            "micro_specimen_id": pl.Int64,
            "chartdate": pl.Utf8,
            "charttime": pl.Utf8,
            "spec_type_desc": pl.Utf8,
            "org_name": pl.Utf8,
        },
    )
    per_specimen = (
        lf.select(
            "micro_specimen_id",
            "subject_id",
            "hadm_id",
            pl.col("charttime").str.strptime(
                pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False
            ),
            pl.col("chartdate").str.strptime(
                pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False
            ),
            "spec_type_desc",
            (pl.col("org_name").is_not_null() & (pl.col("org_name") != "")).alias(
                "_pos"
            ),
        )
        .group_by("micro_specimen_id")
        .agg(
            pl.col("subject_id").min(),
            pl.col("hadm_id").min(),
            pl.col("charttime").min(),
            pl.col("chartdate").min(),
            pl.col("spec_type_desc").min(),
            pl.col("_pos").max().alias("positive_culture"),
        )
        .with_columns(
            pl.coalesce([pl.col("charttime"), pl.col("chartdate")]).alias("time")
        )
        .filter(pl.col("time").is_not_null())
        .select(
            "subject_id",
            "hadm_id",
            "time",
            "spec_type_desc",
            "positive_culture",
            "micro_specimen_id",
        )
        .sort(["subject_id", "time"])
    )
    return per_specimen.collect()


def _midnight_fraction(micro: pl.DataFrame) -> float:
    """Share of specimens stamped at exactly 00:00 (chartdate fallback)."""
    at_midnight = micro["time"].dt.hour().eq(0) & micro["time"].dt.minute().eq(0)
    return float(at_midnight.sum()) / max(micro.height, 1)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--meds-root", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.meds_root / "sidecars"
    out_dir.mkdir(parents=True, exist_ok=True)
    micro = build_microbiology_sidecar(args.mimic_root)
    out = out_dir / "microbiology.parquet"
    micro.write_parquet(out)
    logger.info(
        "[sidecars] microbiology: %d specimens, %d subjects, %.1f%% with charttime-less "
        "dates -> %s",
        micro.height,
        micro["subject_id"].n_unique(),
        100.0 * _midnight_fraction(micro),
        out,
    )


if __name__ == "__main__":
    main()
