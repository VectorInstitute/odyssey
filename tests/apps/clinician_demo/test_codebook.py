"""Codebook: MEDS codes and model tokens in plain clinical language."""

from pathlib import Path

import polars as pl
import pytest

from apps.clinician_demo.codebook import (
    Codebook,
    admission_label,
    bin_flag,
    bin_words,
    category,
    split_bin,
)


DESCRIPTIONS = {
    "LAB//220045//bpm": "Heart Rate",
    "LAB//RESULT//50912//mg/dL": "Creatinine [Mass/volume] in Serum or Plasma",
    "INFUSION_START//221906": "Norepinephrine",
    "DIAGNOSIS//ICD//10//I5021": "Acute systolic (congestive) heart failure",
}


@pytest.fixture
def book() -> Codebook:
    return Codebook(DESCRIPTIONS)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("LAB//220045//bpm::HIGH", ("LAB//220045//bpm", "HIGH")),
        ("LAB//220045//bpm", ("LAB//220045//bpm", None)),
        ("LAB//x::", ("LAB//x", None)),
        ("MEDS_DEATH", ("MEDS_DEATH", None)),
    ],
)
def test_split_bin(token: str, expected: tuple[str, str | None]) -> None:
    assert split_bin(token) == expected


def test_flags_only_for_abnormal_clinical_bins() -> None:
    assert [bin_flag(b) for b in ("LOW", "HIGH", "CRITICAL", "NORMAL", "Q5", None)] == [
        "LOW",
        "HIGH",
        "CRITICAL",
        None,
        None,
        None,
    ]
    assert bin_words("Q1") == "bottom fifth" and bin_words("CRITICAL") == "critical"
    assert bin_words(None) is None and bin_words("Z9") is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("LAB//RESULT//50912//mg/dL", "lab"),
        ("LAB//SPECIMEN_COLLECTED//50912//mg/dL", "order"),
        ("LAB//220045//bpm", "vital"),
        ("MEDICATION//norepinephrine//Administered", "medication"),
        ("INFUSION_START//221906", "infusion"),
        ("SUBJECT_FLUID_OUTPUT//226559//mL", "output"),
        ("DIAGNOSIS//ICD//10//I5021", "diagnosis"),
        ("PROCEDURE//START//225792", "procedure"),
        ("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", "care"),
        ("TRANSFER_TO//transfer//Medical Intensive Care Unit (MICU)", "care"),
        ("MEDS_DEATH", "death"),
        ("GENDER//F", "demographic"),
        ("DRG//HCFA//123", "billing"),
        ("BMI (kg/m2)", "other"),
    ],
)
def test_category(code: str, expected: str) -> None:
    assert category(code) == expected


def test_dictionary_labels_win_and_long_lab_names_are_trimmed(book: Codebook) -> None:
    assert book.label("LAB//220045//bpm::HIGH") == "Heart Rate"
    assert book.label("LAB//RESULT//50912//mg/dL") == "Creatinine"
    assert (
        book.label("DIAGNOSIS//ICD//10//I5021")
        == "Acute systolic (congestive) heart failure"
    )
    assert len(book) == 4


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("MEDICATION//norepinephrine//Administered", "Norepinephrine (administered)"),
        ("MEDICATION//START//vancomycin", "Started vancomycin"),
        ("MEDICATION//STOP//vancomycin", "Stopped vancomycin"),
        (
            "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM",
            "Emergency admission, from emergency room",
        ),
        ("HOSPITAL_DISCHARGE//HOME", "Discharge to home"),
        ("TRANSFER_TO//transfer//MICU", "Transfer to MICU"),
        ("ICU_ADMISSION//MICU", "ICU admission: MICU"),
        ("ED_REGISTRATION", "ED registration"),
        ("MEDS_DEATH", "Death"),
        ("GENDER//F", "Sex: F"),
        ("DIAGNOSIS//ICD//10//I50", "Diagnosis ICD-10 I50"),
        ("INFUSION_START//999", "Infusion started (item 999)"),
        ("LAB//RESULT//12345//mg/dL", "Lab item 12345"),
        ("SOMETHING_ELSE//a//b", "Something else · a · b"),
        ("BMI (kg/m2)", "BMI (kg/m2)"),
    ],
)
def test_structural_labels_without_a_dictionary(code: str, expected: str) -> None:
    assert Codebook().label(code) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("LAB//220045//bpm", "bpm"),
        ("LAB//RESULT//50912//mg/dL::HIGH", "mg/dL"),
        ("LAB//220739//UNK", None),
        ("SUBJECT_FLUID_OUTPUT//226559//mL", "mL"),
        ("MEDICATION//x//Administered", None),
        ("LAB//12345", None),
    ],
)
def test_units(code: str, expected: str | None) -> None:
    assert Codebook().unit(code) == expected


def test_token_label_puts_the_bin_in_words(book: Codebook) -> None:
    assert book.token_label("LAB//220045//bpm::HIGH") == "Heart Rate (high)"
    assert book.token_label("LAB//RESULT//50912//mg/dL::Q4") == "Creatinine (4th fifth)"
    assert book.token_label("MEDS_DEATH") == "Death"


def test_timeline_entry_formats_value_unit_and_flag(book: Codebook) -> None:
    entry = book.entry("LAB//RESULT//50912//mg/dL::CRITICAL", 3.5, 4.2)
    assert (entry.t, entry.category, entry.label) == (3.5, "lab", "Creatinine")
    assert entry.value == "4.2 mg/dL" and entry.flag == "CRITICAL"
    assert book.entry("LAB//220045//bpm::NORMAL", 0.0, 72.0).flag is None
    assert book.entry("MEDS_DEATH", 9.0, None).value is None
    assert book.entry("LAB//220045//bpm", 0.0, float("nan")).value is None
    assert book.entry("LAB//220045//bpm", 0.0, 1234567.0).value == "1.235e+06 bpm"


def test_from_metadata_dir_reads_codes_parquet(tmp_path: Path) -> None:
    pl.DataFrame(
        {"code": ["LAB//220045//bpm"], "description": ["Heart Rate"]}
    ).write_parquet(tmp_path / "codes.parquet")
    assert (
        Codebook.from_metadata_dir(tmp_path).label("LAB//220045//bpm") == "Heart Rate"
    )
    assert len(Codebook.from_metadata_dir(None)) == 0


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM",
            "Emergency admission, from emergency room",
        ),
        (
            "HOSPITAL_ADMISSION//SURGICAL SAME DAY ADMISSION//PHYSICIAN REFERRAL",
            "Same-day surgery admission, from physician referral",
        ),
        ("HOSPITAL_ADMISSION//EU OBSERVATION", "Emergency observation admission"),
        (
            "HOSPITAL_ADMISSION//SOMETHING NEW//CLINIC",
            "Something new admission, from clinic",
        ),
        ("HOSPITAL_ADMISSION", "Admission"),
        (
            "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM HOSPITAL",
            "Urgent admission, transferred from hospital",
        ),
        ("HOSPITAL_ADMISSION//ELECTIVE//PACU", "Elective admission, from PACU"),
        (
            "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM SKILLED NURSING FACILITY",
            "Urgent admission, transferred from skilled nursing facility",
        ),
    ],
)
def test_admission_labels_are_plain(code: str, expected: str) -> None:
    assert admission_label(code) == expected


def test_loinc_names_are_cut_to_the_analyte_and_samples_are_marked() -> None:
    book = Codebook(
        {
            "LAB//RESULT//51221//%": "Hematocrit [Volume Fraction] of Blood by Automated count",
            "LAB//RESULT//51146//%": "Basophils/100 leukocytes in Blood by Automated count",
            "LAB//SPECIMEN_COLLECTED//50912//mg/dL": "Creatinine [Mass/volume] in Serum or Plasma",
            "LAB//220045//bpm": "Heart Rate in Bed",  # vitals are never cut
        }
    )
    assert book.label("LAB//RESULT//51221//%") == "Hematocrit"
    assert book.label("LAB//RESULT//51146//%") == "Basophils/100 leukocytes"
    assert (
        book.label("LAB//SPECIMEN_COLLECTED//50912//mg/dL")
        == "Creatinine (sample sent)"
    )
    assert book.label("LAB//220045//bpm") == "Heart Rate in Bed"


def test_unknown_and_weight_tokens_are_named_for_a_clinician() -> None:
    book = Codebook()
    assert (
        book.token_label("[UNK]")
        == "An uncommon event (outside the model's vocabulary)"
    )
    assert (
        book.token_label("SUBJECT_WEIGHT_AT_INFUSION//KG::Q2")
        == "Weight at infusion (2nd fifth)"
    )
