"""The HICL dictionary builder: majority ingredient per HICL over both tables."""

import gzip
import importlib.util
from pathlib import Path

import polars as pl


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_eicu_hicl_lookup.py"
    spec = importlib.util.spec_from_file_location("build_eicu_hicl_lookup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        handle.write("patientunitstayid,drugname,drughiclseqno\n")
        for name, hicl in rows:
            handle.write(f"1,{name},{hicl}\n")


def test_ingredient_survives_literal_double_slash_in_name() -> None:
    # A raw name containing "//" must not split into extra code segments;
    # unsanitized, the last-segment read returned the name's raw tail
    # (" KCL 20 MEQ SOLN") instead of the normalized drug segment.
    mod = _load_module()
    assert mod._ingredient("DEXTROSE 5% // KCL 20 MEQ SOLN") == "dextrose"
    assert mod._ingredient("DEXTROSE 5% / KCL 20 MEQ SOLN") == "dextrose"


def test_build_lookup_majority_and_support(tmp_path: Path) -> None:
    mod = _load_module()
    _write(
        tmp_path / "medication.csv.gz",
        [
            ("ZOFRAN 4 MG IV SOLN", 33598),
            ("ONDANSETRON 4 MG PO TABS", 33598),
            ("ONDANSETRON 8 MG PO TABS", 33598),
            ("", 33598),  # unnamed rows contribute nothing
            ("SODIUM CHLORIDE 0.9 % IV SOLN", ""),  # no HICL: nothing
        ],
    )
    _write(tmp_path / "admissionDrug.csv.gz", [("VASOPRESSIN 20 UNITS", 2839)])
    lookup = mod.build_lookup(tmp_path)
    assert lookup.columns == ["hicl", "ingredient", "support", "total"]
    rows = {r["hicl"]: r for r in lookup.to_dicts()}
    assert rows[33598]["ingredient"] == "ondansetron"
    assert rows[33598]["support"] == 2 and rows[33598]["total"] == 3
    assert rows[2839]["ingredient"] == "vasopressin"
    assert isinstance(lookup, pl.DataFrame) and lookup.height == 2
