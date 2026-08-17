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
    "LAB//220180//": "76535-4",  # Non-invasive BP diastolic (meas_chartevents_main.csv L7)
    "LAB//220181//": "76536-2",  # Non-invasive BP mean
    "LAB//220052//": "8478-0",  # Arterial BP mean
    "LAB//220050//": "8480-6",  # Arterial BP systolic
    "LAB//220051//": "8462-4",  # Arterial BP diastolic (meas_chartevents_main.csv L11)
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
    # Urine output (icu/outputevents -> different prefix family, see docstring).
    # Every itemid below carries LOINC 9187-6 "Urine output" in
    # outputevents_to_loinc.csv (category "Output"); together they are the
    # urine routes mimic-code's own urine_output concept sums. Drain-category
    # itemids that file also tags 9187-6 (pigtails, drainage bag) and the
    # GU-irrigant/OR/PACU rows are deliberately left out.
    "SUBJECT_FLUID_OUTPUT//226559//": "9187-6",  # Foley catheter urine output
    "SUBJECT_FLUID_OUTPUT//226560//": "9187-6",  # Void (outputevents_to_loinc.csv L36)
    "SUBJECT_FLUID_OUTPUT//226561//": "9187-6",  # Condom Cath (L37)
    "SUBJECT_FLUID_OUTPUT//226563//": "9187-6",  # Suprapubic (L38)
    "SUBJECT_FLUID_OUTPUT//226567//": "9187-6",  # Straight Cath (L42)
    "SUBJECT_FLUID_OUTPUT//226557//": "9187-6",  # R Ureteral Stent (L33)
    "SUBJECT_FLUID_OUTPUT//226558//": "9187-6",  # L Ureteral Stent (L34)
    "SUBJECT_FLUID_OUTPUT//226564//": "9187-6",  # R Nephrostomy (L39)
    "SUBJECT_FLUID_OUTPUT//226565//": "9187-6",  # L Nephrostomy (L40)
    # Labs (hosp/labevents -> "LAB//RESULT//{itemid}//{unit}"). Line numbers
    # refer to d_labitems_to_loinc.csv (header = line 1); where that file
    # lists both a serum/plasma and a whole-blood (blood-gas) itemid under
    # *different* LOINCs (sodium, potassium, chloride, hematocrit,
    # creatinine), only the serum/plasma one -- by far the larger row
    # count -- is mapped.
    "LAB//RESULT//50912//": "2160-0",  # Creatinine
    "LAB//RESULT//51006//": "3094-0",  # Urea Nitrogen (BUN) (L206)
    "LAB//RESULT//50813//": "32693-4",  # Lactate
    "LAB//RESULT//50821//": "11556-8",  # pO2
    "LAB//RESULT//50818//": "11557-6",  # pCO2 (L19)
    "LAB//RESULT//50820//": "11558-4",  # pH of Blood (L21; the arterial-only itemid 52041 is not in the extraction)
    "LAB//RESULT//50802//": "11555-0",  # Base Excess (L3)
    "LAB//RESULT//50885//": "1975-2",  # Bilirubin, Total
    "LAB//RESULT//51265//": "777-3",  # Platelet Count
    "LAB//RESULT//51301//": "6690-2",  # White Blood Cells
    # Hemoglobin: the file gives the same LOINC 718-7 to the CBC itemid
    # (51222, 3.3M rows) and the blood-gas itemid (50811, 117K rows).
    "LAB//RESULT//51222//": "718-7",  # Hemoglobin, Hematology (L411)
    "LAB//RESULT//50811//": "718-7",  # Hemoglobin, Blood Gas (L12)
    "LAB//RESULT//51221//": "4544-3",  # Hematocrit (L410)
    "LAB//RESULT//50983//": "2951-2",  # Sodium (L184)
    "LAB//RESULT//50971//": "2823-3",  # Potassium (L172)
    "LAB//RESULT//50902//": "2075-0",  # Chloride (L103)
    # Bicarbonate: serum/plasma (chemistry) and calculated whole-blood
    # (blood gas) carry distinct LOINCs in the file; both are mapped so a
    # concept can choose either.
    "LAB//RESULT//50882//": "1963-8",  # Bicarbonate, serum/plasma (L83)
    "LAB//RESULT//50803//": "1959-6",  # Calculated Bicarbonate, Whole Blood (L4)
    # Glucose: serum/plasma chemistry vs whole-blood (blood gas), distinct LOINCs.
    "LAB//RESULT//50931//": "2345-7",  # Glucose, serum/plasma (L132)
    "LAB//RESULT//50809//": "2339-0",  # Glucose, whole blood / blood gas (L10)
    "LAB//RESULT//50868//": "1863-0",  # Anion Gap (L69)
    "LAB//RESULT//50893//": "17861-6",  # Calcium, Total (L94)
    "LAB//RESULT//50960//": "19123-9",  # Magnesium (L161)
    "LAB//RESULT//50970//": "2777-1",  # Phosphate (L171)
    "LAB//RESULT//50862//": "1751-7",  # Albumin (L63)
    "LAB//RESULT//50861//": "1742-6",  # Alanine Aminotransferase (ALT) (L62)
    "LAB//RESULT//50878//": "1920-8",  # Asparate Aminotransferase (AST) (L79)
    "LAB//RESULT//50863//": "6768-6",  # Alkaline Phosphatase (L64)
    "LAB//RESULT//51237//": "6301-6",  # INR(PT) (L426)
    "LAB//RESULT//51275//": "14979-9",  # PTT (L464)
    # Troponin: MIMIC-IV's high-volume assay is Troponin T (51003, 374K
    # rows); Troponin I exists only as 52642 (153 rows; 51002 has zero rows
    # and is absent from the extraction).
    "LAB//RESULT//51003//": "6598-7",  # Troponin T (L203)
    "LAB//RESULT//52642//": "10839-9",  # Troponin I (L1569)
    "LAB//RESULT//50963//": "33762-6",  # NTproBNP (L164)
    "LAB//RESULT//50889//": "1988-5",  # C-Reactive Protein, mg/L (L90)
}

# code_prefix -> LOINC code for the eICU extraction (specs/eICU.yaml).
# Vitals codes are the fixed VITALS//... family that spec emits from
# vitalPeriodic/vitalAperiodic (systemic BP is arterial-line, noninvasive
# BP is cuff -- mirroring the MIMIC 220050/220179 split above); lab
# prefixes are `LAB//{labname}//`, where each labname was verified
# against the actual `lab.csv.gz` of the public eICU demo (2.0.1), not
# guessed from documentation. GCS is charted in eICU's nurseCharting
# table, which the spec does not extract yet -- no GCS mapping until it
# does. eICU temperature is Celsius (same LOINC as MIMIC's two
# temperature itemids, unit-differentiated there).
EICU_TO_LOINC: Dict[str, str] = {
    # vitalPeriodic
    "VITALS//PERIODIC//HEARTRATE": "8867-4",  # Heart Rate
    "VITALS//PERIODIC//RESPIRATION": "9279-1",  # Respiratory Rate
    "VITALS//PERIODIC//SAO2": "59408-5",  # O2 saturation, pulse oximetry
    "VITALS//PERIODIC//TEMPERATURE": "8310-5",  # Temperature (Celsius)
    "VITALS//PERIODIC//BP//SYSTEMIC_SYSTOLIC": "8480-6",  # Arterial BP systolic
    "VITALS//PERIODIC//BP//SYSTEMIC_DIASTOLIC": "8462-4",  # Arterial BP diastolic (specs/eICU.yaml L514)
    "VITALS//PERIODIC//BP//SYSTEMIC_MEAN": "8478-0",  # Arterial BP mean
    # vitalAperiodic
    "VITALS//APERIODIC//BP//NONINVASIVE_SYSTOLIC": "76534-7",
    "VITALS//APERIODIC//BP//NONINVASIVE_DIASTOLIC": "76535-4",  # specs/eICU.yaml L418
    "VITALS//APERIODIC//BP//NONINVASIVE_MEAN": "76536-2",
    # lab (prefix matches any unit suffix). Every labname below is an exact
    # `LAB//{labname}//` prefix present in the eICU 2.0 v1 extraction's
    # metadata/codes.parquet; the LOINC is the one the matched MIMIC-IV
    # itemid carries above (that is the whole point of the layer).
    "LAB//creatinine//": "2160-0",  # Creatinine
    "LAB//BUN//": "3094-0",  # Urea Nitrogen (matches MIMIC 51006)
    "LAB//lactate//": "32693-4",  # Lactate
    "LAB//WBC x 1000//": "6690-2",  # White Blood Cells
    "LAB//paO2//": "11556-8",  # pO2
    "LAB//paCO2//": "11557-6",  # pCO2 (matches MIMIC 50818)
    # eICU's `pH` labname is not specimen-qualified; like MIMIC 50820 it is
    # "pH of Blood" and may mix arterial with venous/capillary draws.
    "LAB//pH//": "11558-4",  # pH of Blood (matches MIMIC 50820)
    "LAB//Base Excess//": "11555-0",  # Base Excess (matches MIMIC 50802; eICU also has `Base Deficit`, unmapped)
    "LAB//total bilirubin//": "1975-2",  # Bilirubin, Total
    "LAB//platelets x 1000//": "777-3",  # Platelet Count
    "LAB//FiO2//": "3150-0",  # Inspired O2 Fraction
    "LAB//Hgb//": "718-7",  # Hemoglobin (matches MIMIC 51222/50811)
    "LAB//Hct//": "4544-3",  # Hematocrit (matches MIMIC 51221)
    "LAB//sodium//": "2951-2",  # Sodium (matches MIMIC 50983)
    "LAB//potassium//": "2823-3",  # Potassium (matches MIMIC 50971)
    "LAB//chloride//": "2075-0",  # Chloride (matches MIMIC 50902)
    # eICU charts chemistry-panel `bicarbonate` and blood-gas `HCO3` as
    # separate labnames -- the same serum-vs-whole-blood split MIMIC makes
    # with 50882/50803.
    "LAB//bicarbonate//": "1963-8",  # Bicarbonate, serum/plasma (matches MIMIC 50882)
    "LAB//HCO3//": "1959-6",  # HCO3, blood gas (matches MIMIC 50803)
    # `glucose` is the chemistry-panel value; `bedside glucose` is the
    # point-of-care whole-blood reading, matched to the whole-blood glucose
    # LOINC MIMIC's blood-gas itemid 50809 carries (not to serum 2345-7).
    "LAB//glucose//": "2345-7",  # Glucose, serum/plasma (matches MIMIC 50931)
    "LAB//bedside glucose//": "2339-0",  # Glucose, whole blood (matches MIMIC 50809)
    "LAB//anion gap//": "1863-0",  # Anion Gap (matches MIMIC 50868)
    "LAB//calcium//": "17861-6",  # Calcium, Total (matches MIMIC 50893; `ionized calcium` is separate, unmapped)
    "LAB//magnesium//": "19123-9",  # Magnesium (matches MIMIC 50960)
    "LAB//phosphate//": "2777-1",  # Phosphate (matches MIMIC 50970)
    "LAB//albumin//": "1751-7",  # Albumin (matches MIMIC 50862)
    "LAB//ALT (SGPT)//": "1742-6",  # ALT (matches MIMIC 50861)
    "LAB//AST (SGOT)//": "1920-8",  # AST (matches MIMIC 50878)
    "LAB//alkaline phos.//": "6768-6",  # Alkaline Phosphatase (matches MIMIC 50863)
    "LAB//PT - INR//": "6301-6",  # INR (matches MIMIC 51237)
    "LAB//PTT//": "14979-9",  # PTT (matches MIMIC 51275; `PTT ratio` is separate, unmapped)
    "LAB//troponin - T//": "6598-7",  # Troponin T (matches MIMIC 51003)
    "LAB//troponin - I//": "10839-9",  # Troponin I (matches MIMIC 52642)
    # eICU's `CRP` is charted in mg/dL where MIMIC 50889 is mg/L -- see
    # _PREFIX_UNITS. `CRP-hs` (high-sensitivity) is a different assay and
    # has no MIMIC counterpart in the mapping files, so it is unmapped; the
    # same goes for eICU `BNP` (MIMIC only has NTproBNP, a different analyte).
    "LAB//CRP//": "1988-5",  # C-Reactive Protein, mg/dL (matches MIMIC 50889)
}

# Per-source tables. GEMINI remains a Phase 2 placeholder (README roadmap
# item 10): empty until that extraction exists to build a real, verified
# mapping against, deliberately not guessed ahead of time.
_SOURCE_TABLES: Dict[str, Dict[str, str]] = {
    "mimic_iv": MIMIC_IV_TO_LOINC,
    "eicu": EICU_TO_LOINC,
    "gemini": {},
}


# Unit tags for prefixes whose LOINC alone is ambiguous about units.
# Two signals split by unit across our sources: temperature (MIMIC charts it
# under separate Fahrenheit/Celsius itemids, same LOINC 8310-5; eICU charts
# Celsius only) and CRP (MIMIC 50889 is mg/L per d_labitems_to_loinc.csv
# L90 and the extraction's `LAB//RESULT//50889//mg/L` code; eICU's code is
# `LAB//CRP//mg/dL` -- a 10x difference). Canonical concept thresholds and
# clinical bin ranges use these tags to pick the right per-unit numbers.
_PREFIX_UNITS: Dict[str, Dict[str, str]] = {
    "mimic_iv": {
        "LAB//223761//": "F",
        "LAB//223762//": "C",
        "LAB//RESULT//50889//": "mg/L",
    },
    "eicu": {
        "VITALS//PERIODIC//TEMPERATURE": "C",
        "LAB//CRP//": "mg/dL",
    },
    "gemini": {},
}


def unit_for(code_prefix: str, *, source: str = "mimic_iv") -> Optional[str]:
    """Return the unit tag for ``code_prefix`` in ``source``, or ``None``.

    ``None`` means the prefix's unit is unambiguous for its LOINC across
    our sources (the common case); only unit-split signals carry tags.
    """
    return _PREFIX_UNITS[source].get(code_prefix)


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


def assert_all_mapped(
    code_prefixes: Iterable[str], *, source: str = "mimic_iv"
) -> None:
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
