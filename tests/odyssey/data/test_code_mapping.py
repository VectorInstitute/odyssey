"""Tests for the MEDS code -> LOINC mapping layer."""

import ast
from pathlib import Path

import pytest

from odyssey.data import code_mapping
from odyssey.data.code_mapping import (
    EICU_TO_LOINC,
    GEMINI_TO_LOINC,
    MIMIC_IV_TO_LOINC,
    assert_all_mapped,
    loinc_for,
    prefixes_for_loinc,
    unit_for,
)


def test_loinc_for_known_prefix() -> None:
    assert loinc_for("LAB//220045//") == "8867-4"  # heart rate


def test_loinc_for_unknown_prefix_returns_none() -> None:
    assert loinc_for("LAB//999999//") is None


def test_loinc_for_unregistered_source_raises() -> None:
    with pytest.raises(KeyError):
        loinc_for("LAB//220045//", source="not_a_real_source")


def test_prefixes_for_loinc_finds_both_temperature_units() -> None:
    # Fahrenheit (223761) and Celsius (223762) both represent LOINC
    # 8310-5 -- a concept rule keyed on the LOINC needs both prefixes.
    prefixes = prefixes_for_loinc("8310-5")
    assert prefixes == frozenset({"LAB//223761//", "LAB//223762//"})


def test_prefixes_for_loinc_unmapped_code_is_empty() -> None:
    assert prefixes_for_loinc("0000-0") == frozenset()


def test_outputevents_prefix_is_a_distinct_family_from_labevents() -> None:
    # Confirmed against the real extraction's own event-conversion config
    # (convert_to_MEDS_events/messy.yaml): outputevents does not share the
    # LAB//... prefix family chartevents/labevents use.
    assert loinc_for("SUBJECT_FLUID_OUTPUT//226559//") == "9187-6"


def test_assert_all_mapped_passes_for_every_currently_mapped_prefix() -> None:
    assert_all_mapped(MIMIC_IV_TO_LOINC.keys())


def test_assert_all_mapped_raises_with_the_missing_prefix_named() -> None:
    with pytest.raises(ValueError, match="LAB//999999//"):
        assert_all_mapped(["LAB//220045//", "LAB//999999//"])


def test_eicu_table_is_populated_and_mimic_prefixes_do_not_leak_into_it() -> None:
    # The eICU extraction exists now (specs/eICU.yaml), so its table is
    # real; MIMIC itemid-keyed prefixes must not resolve under it.
    assert loinc_for("VITALS//PERIODIC//HEARTRATE", source="eicu") == "8867-4"
    assert loinc_for("LAB//creatinine//", source="eicu") == "2160-0"
    assert loinc_for("LAB//220045//", source="eicu") is None


def test_same_loinc_resolves_across_sources() -> None:
    # The whole point of the LOINC layer: one concept rule, grounded in a
    # LOINC code, finds each institution's own prefixes.
    assert loinc_for("LAB//220045//", source="mimic_iv") == loinc_for(
        "VITALS//PERIODIC//HEARTRATE", source="eicu"
    )
    assert prefixes_for_loinc("2160-0", source="eicu") == frozenset(
        {"LAB//creatinine//"}
    )


def test_assert_all_mapped_covers_the_eicu_table() -> None:
    assert_all_mapped(EICU_TO_LOINC.keys(), source="eicu")


def test_gemini_table_is_still_an_empty_placeholder() -> None:
    # GEMINI extraction doesn't exist yet; its table must stay empty
    # rather than someone guessing entries ahead of a real extraction to
    # verify against. If this test starts failing because entries were
    # added, that's good news -- update the test, don't just delete it.
    assert loinc_for("LAB//220045//", source="gemini") is None


# The alert feature panel: every LOINC a cross-source alert rule may key
# on, with the sources whose extraction actually charts it. Each source
# listed must resolve the LOINC to at least one prefix.
_PANEL: dict[str, tuple[str, ...]] = {
    # vitals
    "8867-4": ("mimic_iv", "eicu"),  # heart rate
    "9279-1": ("mimic_iv", "eicu"),  # respiratory rate
    "59408-5": ("mimic_iv", "eicu"),  # SpO2
    "8310-5": ("mimic_iv", "eicu"),  # temperature
    "76534-7": ("mimic_iv", "eicu"),  # NIBP systolic
    "76535-4": ("mimic_iv", "eicu"),  # NIBP diastolic
    "76536-2": ("mimic_iv", "eicu"),  # NIBP mean
    "8480-6": ("mimic_iv", "eicu"),  # arterial systolic
    "8462-4": ("mimic_iv", "eicu"),  # arterial diastolic
    "8478-0": ("mimic_iv", "eicu"),  # arterial mean
    "3150-0": ("mimic_iv", "eicu"),  # FiO2
    "9267-6": ("mimic_iv",),  # GCS eye (eICU spec does not extract GCS)
    "9270-0": ("mimic_iv",),  # GCS verbal
    "9268-4": ("mimic_iv",),  # GCS motor
    "9187-6": ("mimic_iv",),  # urine output (eICU spec emits no intakeOutput)
    # labs
    "2160-0": ("mimic_iv", "eicu"),  # creatinine
    "3094-0": ("mimic_iv", "eicu"),  # BUN
    "32693-4": ("mimic_iv", "eicu"),  # lactate
    "6690-2": ("mimic_iv", "eicu"),  # WBC
    "718-7": ("mimic_iv", "eicu"),  # hemoglobin
    "4544-3": ("mimic_iv", "eicu"),  # hematocrit
    "777-3": ("mimic_iv", "eicu"),  # platelets
    "2951-2": ("mimic_iv", "eicu"),  # sodium
    "2823-3": ("mimic_iv", "eicu"),  # potassium
    "2075-0": ("mimic_iv", "eicu"),  # chloride
    "1963-8": ("mimic_iv", "eicu"),  # bicarbonate, serum/plasma
    "1959-6": ("mimic_iv", "eicu"),  # bicarbonate, whole blood
    "2345-7": ("mimic_iv", "eicu"),  # glucose, serum/plasma
    "2339-0": ("mimic_iv", "eicu"),  # glucose, whole blood
    "1863-0": ("mimic_iv", "eicu"),  # anion gap
    "17861-6": ("mimic_iv", "eicu"),  # calcium, total
    "19123-9": ("mimic_iv", "eicu"),  # magnesium
    "2777-1": ("mimic_iv", "eicu"),  # phosphate
    "1751-7": ("mimic_iv", "eicu"),  # albumin
    "1975-2": ("mimic_iv", "eicu"),  # bilirubin, total
    "1742-6": ("mimic_iv", "eicu"),  # ALT
    "1920-8": ("mimic_iv", "eicu"),  # AST
    "6768-6": ("mimic_iv", "eicu"),  # alkaline phosphatase
    "6301-6": ("mimic_iv", "eicu"),  # INR
    "14979-9": ("mimic_iv", "eicu"),  # PTT
    "11558-4": ("mimic_iv", "eicu"),  # pH of blood
    "11557-6": ("mimic_iv", "eicu"),  # pCO2
    "11556-8": ("mimic_iv", "eicu"),  # pO2
    "11555-0": ("mimic_iv", "eicu"),  # base excess
    "6598-7": ("mimic_iv", "eicu"),  # troponin T
    "10839-9": ("mimic_iv", "eicu"),  # troponin I
    "33762-6": ("mimic_iv",),  # NT-proBNP (eICU charts BNP, a different analyte)
    "1988-5": ("mimic_iv", "eicu"),  # CRP
}


@pytest.mark.parametrize("loinc,sources", sorted(_PANEL.items()))
def test_alert_panel_loinc_resolves_in_every_source_that_charts_it(
    loinc: str, sources: tuple[str, ...]
) -> None:
    for source in sources:
        assert prefixes_for_loinc(loinc, source=source), (
            f"LOINC {loinc} has no prefix in source {source!r}"
        )


def test_alert_panel_labs_map_to_the_same_loinc_on_both_sources() -> None:
    # Spot-check the cross-source contract on the newly added labs: the
    # eICU labname and the MIMIC itemid must land on one LOINC.
    pairs = [
        ("LAB//RESULT//51006//", "LAB//BUN//"),
        ("LAB//RESULT//51222//", "LAB//Hgb//"),
        ("LAB//RESULT//50983//", "LAB//sodium//"),
        ("LAB//RESULT//50882//", "LAB//bicarbonate//"),
        ("LAB//RESULT//50803//", "LAB//HCO3//"),
        ("LAB//RESULT//50931//", "LAB//glucose//"),
        ("LAB//RESULT//51237//", "LAB//PT - INR//"),
        ("LAB//RESULT//50820//", "LAB//pH//"),
        ("LAB//RESULT//51003//", "LAB//troponin - T//"),
        ("LAB//RESULT//50889//", "LAB//CRP//"),
    ]
    for mimic_prefix, eicu_prefix in pairs:
        assert loinc_for(mimic_prefix, source="mimic_iv") == loinc_for(
            eicu_prefix, source="eicu"
        ), (mimic_prefix, eicu_prefix)


def test_no_duplicate_prefix_keys_in_the_source_tables() -> None:
    # A Python dict literal silently keeps only the last of two identical
    # keys, so duplicates can't be seen at runtime -- parse the module
    # source and check every dict literal's string keys instead.
    source_path = Path(code_mapping.__file__)
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        assert not dupes, f"duplicate prefix keys in code_mapping.py: {dupes}"


def test_unit_tags_for_temperature_and_crp() -> None:
    assert unit_for("LAB//223761//") == "F"
    assert unit_for("LAB//223762//") == "C"
    assert unit_for("VITALS//PERIODIC//TEMPERATURE", source="eicu") == "C"
    # CRP is charted in mg/L in MIMIC (50889) and mg/dL in eICU: a 10x gap
    # any shared threshold must account for.
    assert unit_for("LAB//RESULT//50889//") == "mg/L"
    assert unit_for("LAB//CRP//", source="eicu") == "mg/dL"
    # Unit-unambiguous prefixes carry no tag.
    assert unit_for("LAB//220045//") is None
    assert unit_for("LAB//sodium//", source="eicu") is None


def test_every_unit_tagged_prefix_is_also_mapped() -> None:
    # A unit tag on a prefix that has no LOINC mapping would be dead
    # config; every tag must sit on a mapped prefix of its source.
    for source, tags in code_mapping._PREFIX_UNITS.items():
        for prefix in tags:
            assert loinc_for(prefix, source=source) is not None, (source, prefix)


# ---------------------------------------------------------------------------
# GEMINI source table
# ---------------------------------------------------------------------------


def test_gemini_prefixes_resolve_to_registry_loincs() -> None:
    """Every registry LOINC the GEMINI table claims is reachable both ways."""
    for prefix, loinc in GEMINI_TO_LOINC.items():
        assert loinc_for(prefix, source="gemini") == loinc
        assert prefix in prefixes_for_loinc(loinc, source="gemini")
    # the three lactate ids collapse onto one canonical LOINC
    assert prefixes_for_loinc("32693-4", source="gemini") == frozenset(
        {"LAB//3018405//", "LAB//3008037//", "LAB//3020138//"}
    )
    # unit-split signals carry their tags; unambiguous ones do not
    assert unit_for("LAB//3020564//", source="gemini") == "umol/L"
    assert unit_for("VITALS//3020891//", source="gemini") == "C"
    assert unit_for("VITALS//3027018//", source="gemini") is None
    # the lab panel: SI-tagged where the cutoff differs, untagged where
    # mmol/L equals mEq/L and x10^9/L equals K/uL
    assert loinc_for("LAB//3019550//", source="gemini") == "2951-2"  # sodium
    assert loinc_for("LAB//3023103//", source="gemini") == "2823-3"  # potassium
    assert loinc_for("LAB//3016293//", source="gemini") == "1963-8"  # bicarbonate
    assert loinc_for("LAB//3007461//", source="gemini") == "777-3"  # platelets
    assert loinc_for("LAB//3032080//", source="gemini") == "6301-6"  # INR
    assert loinc_for("LAB//3000963//", source="gemini") == "718-7"  # hemoglobin
    assert loinc_for("LAB//3013826//", source="gemini") == "2345-7"  # glucose
    assert unit_for("LAB//3000963//", source="gemini") == "g/L"
    assert unit_for("LAB//3013826//", source="gemini") == "mmol/L"
    assert unit_for("LAB//3019550//", source="gemini") is None
    assert (
        loinc_for("LAB//3040151//", source="gemini") is None
    )  # capillary glucose, as on MIMIC
