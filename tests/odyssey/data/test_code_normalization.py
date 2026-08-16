"""Tests for medication ingredient normalization and ICD category backoff."""

import polars as pl

from odyssey.data.code_normalization import (
    icd_category_code,
    normalize_medication_code,
    normalize_medication_codes,
)


# ---------------------------------------------------------------------------
# normalize_medication_code (scalar)
# ---------------------------------------------------------------------------


def test_dose_form_route_are_stripped() -> None:
    assert (
        normalize_medication_code("MEDICATION//Acetaminophen 325 mg PO TABS")
        == "MEDICATION//acetaminophen"
    )
    assert (
        normalize_medication_code("MEDICATION//Acetaminophen 650 mg RE SUPP")
        == "MEDICATION//acetaminophen"
    )


def test_variants_of_one_ingredient_collapse_to_one_token() -> None:
    variants = [
        "MEDICATION//Metoprolol Tartrate 25 mg PO TABS",
        "MEDICATION//Metoprolol Tartrate 50 mg PO TABS",
        "MEDICATION//METOPROLOL TARTRATE 5 mg IV SOLN",
    ]
    normalized = {normalize_medication_code(v) for v in variants}
    assert normalized == {"MEDICATION//metoprolol tartrate"}


def test_eicu_container_prefix_and_action_segment() -> None:
    assert (
        normalize_medication_code(
            "MEDICATION//STARTED//1000 ML FLEX CONT : SODIUM CHLORIDE 0.9 % IV SOLN"
        )
        == "MEDICATION//STARTED//sodium chloride"
    )


def test_digits_inside_a_name_token_survive() -> None:
    # The cut point is a space-preceded digit; digits embedded in a name
    # token (B12, D5W) are not dose text.
    assert (
        normalize_medication_code("MEDICATION//Vitamin B12 1000 mcg PO TABS")
        == "MEDICATION//vitamin b12"
    )


def test_never_produces_an_empty_ingredient() -> None:
    out = normalize_medication_code("MEDICATION//5 ML VIAL")
    assert out.startswith("MEDICATION//")
    assert len(out.split("//")[-1]) > 0


def test_non_medication_codes_pass_through() -> None:
    for code in ("LAB//220045//bpm::HIGH", "DIAGNOSIS//ICD//10//I5023", "MEDS_DEATH"):
        assert normalize_medication_code(code) == code


def test_vasopressor_names_survive_for_concept_matching() -> None:
    out = normalize_medication_code("MEDICATION//Norepinephrine 8 mg/250 mL NS")
    assert "norepinephrine" in out


# ---------------------------------------------------------------------------
# normalize_medication_codes (vectorized) matches the scalar form
# ---------------------------------------------------------------------------


def test_vectorized_matches_scalar_on_a_mixed_frame() -> None:
    codes = [
        "MEDICATION//Acetaminophen 325 mg PO TABS",
        "MEDICATION//STARTED//1000 ML FLEX CONT : SODIUM CHLORIDE 0.9 % IV SOLN",
        "MEDICATION//Vitamin B12 1000 mcg PO TABS",
        "MEDICATION//Norepinephrine 8 mg/250 mL NS",
        "LAB//220045//bpm::HIGH",
        "DIAGNOSIS//ICD//10//I5023",
        "MEDICATION//Metoprolol Tartrate 50 mg PO TABS",
        "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM",
    ]
    frame = pl.DataFrame({"code": codes})
    out = normalize_medication_codes(frame)["code"].to_list()
    assert out == [normalize_medication_code(c) for c in codes]


# ---------------------------------------------------------------------------
# icd_category_code
# ---------------------------------------------------------------------------


def test_icd_backoff_truncates_to_three_characters() -> None:
    assert icd_category_code("DIAGNOSIS//ICD//10//I5023") == "DIAGNOSIS//ICD//10//I50"
    assert icd_category_code("DIAGNOSIS//ICD//9//41401") == "DIAGNOSIS//ICD//9//414"
    assert icd_category_code("PROCEDURE//ICD//10//0BH17EZ") == "PROCEDURE//ICD//10//0BH"


def test_icd_backoff_leaves_category_level_and_non_icd_alone() -> None:
    assert icd_category_code("DIAGNOSIS//ICD//10//I50") is None
    assert icd_category_code("LAB//220045//bpm") is None
    assert icd_category_code("MEDICATION//acetaminophen") is None


# ---------------------------------------------------------------------------
# real MIMIC-IV shapes (verified against the actual extraction)
# ---------------------------------------------------------------------------


def test_mimic_emar_shape_drops_ndc_and_keeps_event_text() -> None:
    # MEDICATION//{drug}//{event_txt}//{ndc}: the trailing NDC packaging
    # code is pure fragmentation; the event text is clinical signal.
    assert (
        normalize_medication_code("MEDICATION//Heparin//Administered//63323026201")
        == "MEDICATION//heparin//Administered"
    )
    assert (
        normalize_medication_code(
            "MEDICATION//Insulin//Not Given per Sliding Scale//unk"
        )
        == "MEDICATION//insulin//Not Given per Sliding Scale"
    )


def test_mimic_emar_ndc_variants_of_one_drug_collapse() -> None:
    variants = [
        "MEDICATION//Heparin//Administered//63323026201",
        "MEDICATION//Heparin//Administered//00409779362",
        "MEDICATION//Heparin//Administered//unk",
    ]
    assert len({normalize_medication_code(v) for v in variants}) == 1


def test_mimic_pharmacy_start_stop_shape() -> None:
    assert (
        normalize_medication_code("MEDICATION//START//Sodium Chloride 0.9%  Flush//unk")
        == "MEDICATION//START//sodium chloride"
    )
    assert (
        normalize_medication_code("MEDICATION//STOP//UNK//unk")
        == "MEDICATION//STOP//unk"
    )


def test_dose_embedded_in_drug_segment_is_stripped() -> None:
    assert (
        normalize_medication_code(
            "MEDICATION//Sodium Chloride 0.9%  Flush//Flushed//unk"
        )
        == "MEDICATION//sodium chloride//Flushed"
    )
