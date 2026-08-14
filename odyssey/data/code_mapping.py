"""Per-institution MEDS code -> LOINC standard code mapping.

Clinical concept rules (:mod:`odyssey.data.concepts`) are written once
against LOINC codes, the portable, institution-agnostic vocabulary --
not against one hospital's internal item identifiers. This module is the
thin translation layer that resolves a LOINC code to the actual MEDS
code prefixes a given source institution's extraction uses, so the same
concept rule runs unchanged against MIMIC-IV today and against eICU or
GEMINI once their extractions exist (research_journal/04_concept_pipeline.html,
decision (c)).

This is deliberately lightweight: a maintained lookup table per source,
not full OMOP CDM adoption. Standing up real OMOP CDM infrastructure
needs a genuine ETL project per institution with uneven existing
coverage; a LOINC-code mapping table gets most of the cross-institution
portability benefit without that cost -- see entry 04, Section 05.

The MIMIC-IV table below is transcribed directly from two files fetched
by the standard extraction pipeline (see README's ``do_download=false``
section: ``meas_chartevents_main.csv`` for chartevents/vitals/GCS,
``d_labitems_to_loinc.csv`` for labevents), both published by
`MIT-LCP/mimic-code <https://github.com/MIT-LCP/mimic-code>`_ -- not
guessed from memory. Two things worth noting from reading those files
directly: (1) labevents itemid 51301 ("White Blood Cells") and 51300
("WBC Count") map to the same LOINC code and are explicitly documented
as duplicates in ``d_labitems_to_loinc.csv``'s notes column; 51301 is
used here since its real row count in that file (3.3M) vastly exceeds
51300's (27K), meaning 51301 is what real chartevents/labevents data
actually uses at scale. (2) chartevents/labevents share the ``LAB//...``
code-prefix family in this project's MEDS extraction, but outputevents
(needed for KDIGO's urine-output criterion) does not -- confirmed by
reading the extraction's own event-conversion config
(``convert_to_MEDS_events/messy.yaml``) directly rather than assuming
it follows the same pattern: outputevents codes are
``SUBJECT_FLUID_OUTPUT//{itemid}//{unit}``, a separate prefix family.
"""

from typing import Dict, FrozenSet, Iterable, Optional


# code_prefix -> LOINC code. Verified directly against MIT-LCP/mimic-code's
# meas_chartevents_main.csv (chartevents: vitals, GCS components, FiO2) and
# d_labitems_to_loinc.csv (labevents), plus outputevents_to_loinc.csv for
# urine output -- see the module docstring.
MIMIC_IV_TO_LOINC: Dict[str, str] = {
    # Vitals (icu/chartevents -> "LAB//{itemid}//{unit}")
    "LAB//220045//": "8867-4",  # Heart Rate
    "LAB//220210//": "9279-1",  # Respiratory Rate
    "LAB//220277//": "59408-5",  # O2 saturation pulse oximetry
    "LAB//220179//": "76534-7",  # Non-invasive BP systolic
    "LAB//220181//": "76536-2",  # Non-invasive BP mean
    "LAB//220052//": "8478-0",  # Arterial BP mean
    "LAB//220050//": "8480-6",  # Arterial BP systolic
    "LAB//223761//": "8310-5",  # Temperature, Fahrenheit
    "LAB//223762//": "8310-5",  # Temperature, Celsius (same LOINC, different unit)
    "LAB//223835//": "3150-0",  # Inspired O2 Fraction (FiO2)
    # GCS components (icu/chartevents): MIMIC-IV has no single "GCS total"
    # itemid -- unlike MIMIC-III's itemid 198, IV splits it into these
    # three; a total must be derived by summing all three (see
    # odyssey.data.concepts's qSOFA/NEWS2 definitions).
    "LAB//220739//": "9267-6",  # GCS - Eye Opening
    "LAB//223900//": "9270-0",  # GCS - Verbal Response
    "LAB//223901//": "9268-4",  # GCS - Motor Response
    # Urine output (icu/outputevents -> different prefix family, see docstring)
    "SUBJECT_FLUID_OUTPUT//226559//": "9187-6",  # Foley catheter urine output
    # Labs (hosp/labevents -> "LAB//RESULT//{itemid}//{unit}")
    "LAB//RESULT//50912//": "2160-0",  # Creatinine
    "LAB//RESULT//50813//": "32693-4",  # Lactate
    "LAB//RESULT//50821//": "11556-8",  # pO2
    "LAB//RESULT//50885//": "1975-2",  # Bilirubin, Total
    "LAB//RESULT//51265//": "777-3",  # Platelet Count
    "LAB//RESULT//51301//": "6690-2",  # White Blood Cells
}

# Per-source tables. eICU and GEMINI are Phase 2 placeholders (README
# roadmap item 10): empty until those extractions exist to build a real,
# verified mapping against, deliberately not guessed ahead of time.
_SOURCE_TABLES: Dict[str, Dict[str, str]] = {
    "mimic_iv": MIMIC_IV_TO_LOINC,
    "eicu": {},
    "gemini": {},
}


def loinc_for(code_prefix: str, *, source: str = "mimic_iv") -> Optional[str]:
    """Return the LOINC code for ``code_prefix`` in ``source``, or ``None``.

    ``source`` must be one of the registered sources (currently
    ``"mimic_iv"``, ``"eicu"``, ``"gemini"``); an unregistered source
    raises ``KeyError`` rather than silently returning ``None``, since
    that's almost always a typo, not a real "no mapping exists" case.
    """
    return _SOURCE_TABLES[source].get(code_prefix)


def prefixes_for_loinc(loinc_code: str, *, source: str = "mimic_iv") -> FrozenSet[str]:
    """Return every ``code_prefix`` in ``source`` mapped to ``loinc_code``.

    More than one prefix can map to the same LOINC code (e.g. Fahrenheit
    and Celsius temperature readings both represent LOINC 8310-5); a
    concept rule keyed on a LOINC code needs every matching prefix, not
    just one.
    """
    table = _SOURCE_TABLES[source]
    return frozenset(prefix for prefix, loinc in table.items() if loinc == loinc_code)


def assert_all_mapped(code_prefixes: Iterable[str], *, source: str = "mimic_iv") -> None:
    """Raise ``ValueError`` if any of ``code_prefixes`` has no LOINC mapping.

    A cheap, direct guardrail for decision (c): every concept rule's
    source code should have a known LOINC mapping, so a new rule added
    without updating this table's coverage is caught immediately (by a
    test, see ``test_code_mapping.py``) rather than silently losing
    portability to future institutions.
    """
    table = _SOURCE_TABLES[source]
    unmapped = sorted({prefix for prefix in code_prefixes if prefix not in table})
    if unmapped:
        raise ValueError(
            f"no LOINC mapping for source {source!r}: {unmapped}. Add it to "
            f"the corresponding table in odyssey/data/code_mapping.py."
        )
