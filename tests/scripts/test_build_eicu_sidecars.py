"""The eICU sidecars reproduce the extraction's pseudotimes and the MIMIC schema."""

import gzip
from datetime import datetime
from pathlib import Path

from scripts.build_eicu_sidecars import (
    build_antibiotic_orders_sidecar,
    build_microbiology_sidecar,
    unit_admissions,
)


def _gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def _raw(tmp_path: Path) -> Path:
    root = tmp_path / "eicu"
    root.mkdir()
    _gz(
        root / "patient.csv.gz",
        "patientunitstayid,patienthealthsystemstayid,hospitaldischargeyear,hospitaldischargetime24,hospitaldischargeoffset\n"
        "10,1,2015,03:50:00,3596\n"  # discharge 2015-12-31 03:50, unit admit 3596 min earlier
        "11,1,2015,03:50:00,600000\n",  # garbage offset -> no admission time -> dropped
    )
    _gz(
        root / "microLab.csv.gz",
        "microlabid,patientunitstayid,culturetakenoffset,culturesite,organism,antibiotic,sensitivitylevel\n"
        '5,10,-60,"Blood, Venipuncture",no growth,"",""\n'
        '6,10,120,"Urine, Voided Specimen",Escherichia coli,"",""\n'
        '7,10,120,"Urine, Voided Specimen",Escherichia coli,"cefazolin","Sensitive"\n'  # same specimen
        '8,11,10,"Blood, Venipuncture",Staphylococcus aureus,"",""\n',  # stay without a time
    )
    _gz(
        root / "medication.csv.gz",
        "patientunitstayid,drugstartoffset,drugstopoffset,drugordercancelled,drugname,drughiclseqno,routeadmin\n"
        "10,30,1470,No,VANCOMYCIN 1 G IV SOLN,,IV\n"
        "10,45,,No,,9999,PO\n"  # null name, HICL says ceftriaxone
        "10,50,100,No,METOPROLOL TARTRATE 25 MG PO TABS,,PO\n"
        "10,60,100,Yes,PIPERACILLIN-TAZOBACTAM,,IV\n",  # cancelled
    )
    (tmp_path / "hicl.csv").write_text(
        "hicl,ingredient,support,total\n9999,ceftriaxone,5,5\n"
    )
    return root


def test_unit_admission_is_the_spec_pseudotime_and_garbage_offsets_drop(
    tmp_path: Path,
) -> None:
    stays = unit_admissions(_raw(tmp_path))
    assert stays.height == 1
    row = stays.row(0, named=True)
    assert row["hadm_id"] == 10 and row["subject_id"] == 1
    assert row["unitadmit"] == datetime(
        2015, 12, 28, 15, 54
    )  # 03:50 on Dec 31 minus 3596 min


def test_microbiology_specimens_are_grouped_timed_and_signed(tmp_path: Path) -> None:
    root = _raw(tmp_path)
    micro = build_microbiology_sidecar(root, unit_admissions(root))
    assert micro.columns == [
        "subject_id",
        "hadm_id",
        "time",
        "spec_type_desc",
        "positive_culture",
        "micro_specimen_id",
    ]
    assert (
        micro.height == 2
    )  # the two E. coli rows are one specimen; stay 11 has no time
    blood, urine = micro.sort("time").iter_rows(named=True)
    assert blood["positive_culture"] is False and blood["time"] == datetime(
        2015, 12, 28, 14, 54
    )
    assert urine["positive_culture"] is True and urine["micro_specimen_id"] == 6
    assert urine["time"] == datetime(2015, 12, 28, 17, 54)


def test_antibiotic_orders_use_names_or_hicl_and_drop_cancelled(tmp_path: Path) -> None:
    root = _raw(tmp_path)
    orders = build_antibiotic_orders_sidecar(
        root, unit_admissions(root), hicl_dictionary=tmp_path / "hicl.csv"
    )
    assert orders.columns == [
        "subject_id",
        "hadm_id",
        "time",
        "stoptime",
        "drug",
        "route",
    ]
    assert orders["drug"].to_list() == ["VANCOMYCIN 1 G IV SOLN", "ceftriaxone"]
    assert orders["route"].to_list() == ["IV", "PO"]
    assert orders["stoptime"].to_list()[1] is None
    assert orders["time"].to_list()[0] == datetime(2015, 12, 28, 16, 24)
