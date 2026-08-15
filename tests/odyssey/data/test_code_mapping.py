"""Tests for the MEDS code -> LOINC mapping layer."""

import pytest

from odyssey.data.code_mapping import (
    EICU_TO_LOINC,
    MIMIC_IV_TO_LOINC,
    assert_all_mapped,
    loinc_for,
    prefixes_for_loinc,
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
