"""Build the eICU HICL -> ingredient dictionary shipped as a package resource.

eICU-CRD's ``medication`` table leaves ``drugname`` empty on 36% of rows, but
94% of those rows carry a First Databank HICL sequence number
(``drughiclseqno``), and only 2% of medication rows have neither. HICL is a
proprietary vocabulary with no public name table, so this script derives one
empirically: over every ``medication`` and ``admissionDrug`` row that has BOTH
a name and a HICL, normalize the name to its ingredient (the same rule the
training pipeline applies, :func:`odyssey.data.code_normalization
.normalize_medication_code`) and keep, per HICL, the majority ingredient with
its support. The result is a drug dictionary (HICL number, ingredient string,
counts) and contains no patient-level information.

Run against the raw eICU-CRD 2.0 directory (the ``*.csv.gz`` files):

    uv run python scripts/build_eicu_hicl_lookup.py \\
        --eicu-dir /path/to/eicu-crd/2.0 \\
        --out odyssey/data/resources/eicu_hicl_ingredients.csv

Regenerate whenever the name normalization rule changes; the resource is read
by :func:`odyssey.data.code_normalization.load_eicu_hicl_ingredients`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from odyssey.data.code_normalization import normalize_medication_code


TABLES = ("medication.csv.gz", "admissionDrug.csv.gz")


def _ingredient(name: str) -> str:
    """Ingredient of one raw drug string, via the pipeline's normalizer."""
    return normalize_medication_code(f"MEDICATION//STARTED//{name}").split("//")[-1]


def build_lookup(eicu_dir: Path) -> pl.DataFrame:
    """Majority ingredient per HICL over rows carrying both name and HICL.

    Columns: ``hicl`` (int), ``ingredient`` (str), ``support`` (rows of the
    HICL that normalized to that ingredient), ``total`` (rows of the HICL
    with any name); ``support / total`` is the dictionary's purity per code.
    """
    frames = []
    for table in TABLES:
        frames.append(
            pl.read_csv(
                eicu_dir / table,
                columns=["drugname", "drughiclseqno"],
                null_values=[""],
                infer_schema_length=10000,
            )
        )
    both = pl.concat(frames).filter(
        pl.col("drugname").is_not_null() & pl.col("drughiclseqno").is_not_null()
    )
    # Normalize distinct names once (thousands), then join back (millions).
    names = both.select(pl.col("drugname").unique()).with_columns(
        pl.col("drugname")
        .map_elements(_ingredient, return_dtype=pl.Utf8)
        .alias("ingredient")
    )
    both = both.join(names, on="drugname", how="left")
    per_pair = both.group_by(["drughiclseqno", "ingredient"]).len("support")
    per_hicl = both.group_by("drughiclseqno").len("total")
    return (
        per_pair.sort(
            ["drughiclseqno", "support", "ingredient"], descending=[False, True, False]
        )
        .unique(subset=["drughiclseqno"], keep="first", maintain_order=True)
        .join(per_hicl, on="drughiclseqno")
        .rename({"drughiclseqno": "hicl"})
        .select("hicl", "ingredient", "support", "total")
        .sort("hicl")
    )


def main() -> None:
    """Build the dictionary from ``--eicu-dir`` and write it to ``--out``."""
    parser = argparse.ArgumentParser(
        description="Build the eICU HICL -> ingredient dictionary resource."
    )
    parser.add_argument("--eicu-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("odyssey/data/resources/eicu_hicl_ingredients.csv"),
    )
    args = parser.parse_args()
    lookup = build_lookup(args.eicu_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lookup.write_csv(args.out)
    purity = lookup.select((pl.col("support") / pl.col("total")).median()).item()
    print(
        f"wrote {lookup.height} HICL codes to {args.out} (median purity {float(purity):.2f})"
    )


if __name__ == "__main__":
    main()
