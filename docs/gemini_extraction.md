# GEMINI to MEDS

Draft mapping from GEMINI's real schema and its first `extract-dry` run
(`scripts/gemini/out/{schema,extract_dry}.{json,md}`, datacut
`subdural_hematoma_v1_0_0` -- a diagnosis-filtered cohort useful for schema
exploration, **not** the general internal-medicine population odyssey's own
training cut will use, see [Open questions](#open-questions)) to the MEDS
event-stream shape the rest of the pipeline already consumes for MIMIC-IV and
eICU. Everything below is checked against real query results now, not a
guess from column names/types alone -- but still not a spec to implement
blind: extraction code has not started (see [Status](#status)).

**The single most important finding from `extract-dry`: GEMINI's labs are
SI units, not the US-conventional units this project's canonical clinical
ranges assume.** See [Units and canonical clinical ranges](#units-and-canonical-clinical-ranges)
before writing or trusting a single binned lab value from GEMINI.

## Open questions

Resolve these before or during extraction-spec implementation, roughly in
priority order:

1. **Which datacut is odyssey's actual training cut?** `subdural_hematoma_v1_0_0`
   (the current `GEMINI_DATACUT` default, chosen for schema exploration
   convenience per its own commit message) is a narrow, diagnosis-filtered
   cohort -- not representative of the general GIM population the README's
   generalization claim is about. A real training-cut decision, and
   confirmation it has the same table/column shape as this cut, comes
   before any extraction is run for real.
2. ~~**Mortality signal**~~ **Resolved**: `derived_variables_subset.in_hospital_mortality_derived`
   is essentially fully populated (fewer than 6 nulls out of 2,268,000 rows,
   per `extract-dry`'s null-fraction check) -- use it directly as the
   primary mortality signal rather than decoding `discharge_disposition`
   ourselves. `discharge_disposition` remains available as a cross-check
   (via `lookup_cihi_codes` filtered to `column_name = 'discharge_disposition'`,
   still unconfirmed) but isn't on the critical path anymore.
3. ~~**`lookup_vitals_concept` is genuinely empty**~~ **Resolved, and the
   opposite of what the schema report's rounded count suggested**:
   `extract-dry`'s real `EXISTS` check found all four "suspect-empty"
   lookups (`lookup_hospital`, `lookup_pharmacy_route`,
   `lookup_transfusion_concept`, `lookup_vitals_concept`) are **not**
   genuinely empty -- each has a real but small (under ~500, hence
   rounding to `"0"`) row count. Consistent with this: the vitals concept
   frequency query found real, non-null `concept_desc` values for every one
   of the 13 top vitals codes (heart rate, SpO2, temperature, blood
   pressure, ...) -- the lookup is small but covers exactly the
   high-frequency vitals that matter most. No fallback to raw
   `measurement_name` needed for the common case.
4. **A `lookup_lab_concept` data-quality wrinkle**: nearly every code in the
   lab concept-frequency query appears **twice** -- once joined to a real
   description, once joined to a row with `concept_desc = NULL` -- with
   the same row count both times (e.g. concept 3019550/Sodium: 17,224,000
   both times). This means `lookup_lab_concept` has more than one row per
   `concept_id` for at least the high-frequency codes, at least one of them
   with a null description. A straight `JOIN ... ON concept_id = code`
   used naively in the real extraction would double-count or need
   deduplicating (`SELECT DISTINCT ON (concept_id) ... WHERE concept_desc
   IS NOT NULL ORDER BY concept_id`, or equivalent) -- not yet confirmed
   *why* the duplicate rows exist (multiple vocabularies mapped to the same
   id? a data-loading artifact?), just that they're really there.
5. **Two parallel pharmacy-identity mapping paths**: `rxnorm_cache`
   (`rxcui`, `raw_input`, `search_type`) and `lookup_pharmacy_mapping`
   (`raw_input`, `rxnorm_match`, `drug_group`) both look like
   raw-drug-name-to-RxNorm bridges. Whether they're redundant, one
   supersedes the other, or they cover different input types
   (`search_type`/`project_name` suggest possibly different callers) isn't
   confirmed -- check for overlapping `raw_input` values and whether their
   `rxcui`/`rxnorm_match` outputs ever disagree before picking one.
6. **Patient-level linkage across encounters**: `patient_id_hashed` appears
   *only* in `admdad_subset`, not in any event table -- every event table
   carries `genc_id` (encounter) only. Linking two encounters to the same
   real patient always requires a join through `admdad_subset`, and
   whether `patient_id_hashed` is stable/reusable across a patient's
   multiple encounters in this data cut isn't yet confirmed. This is the
   same "subject: patient, or a hospital encounter if patient-level
   linkage isn't straightforward" question this doc's predecessor section
   in `docs/gemini.md` already flagged -- now grounded in a concrete schema
   fact rather than a guess.
7. ~~**Text datetime format**~~ **Partly resolved -- and it's worse than
   just "format unconfirmed."** `extract-dry`'s year-range query (leading
   `YYYY` extracted via regex) found real, physically impossible outliers:

   | table.column | min year | max year |
   | --- | --- | --- |
   | `pharmacy_subset.med_start_date_time` | 1930 | **9022** |
   | `pharmacy_subset.med_end_date_time` | 1840 | **8186** |
   | `radiology_subset.performed_date_time` | 1915 | **9999** |
   | `lab_subset.collection_date_time` | 1945 | 2025 |

   (`admdad_subset`, `er_subset`, and `ipscu_subset` timestamps all land in
   the plausible 2010-2024 range -- this is specific to pharmacy and
   radiology, not universal.) These are data-entry artifacts (transposed
   digits, sentinel/placeholder dates), not real events centuries away.
   **The fix, same pattern already used for eICU's own timestamp outliers**:
   guard every parsed pharmacy/radiology timestamp against the owning
   encounter's `admdad_subset.admission_date_time` -- accept it only if it
   falls within roughly plus-or-minus one year of admission, else drop the
   event (or the specific timestamp field) and log it rather than silently
   extracting a nonsense date decades or centuries out. Exact text format
   for the *valid* rows is still unconfirmed (only year-level extraction
   has been checked so far).
8. **`diagnosis_prefix`** on `ipdiagnosis_subset` may already be the
   3-character ICD-10-CA backoff code this project's other concept rules
   use when a fully-specified code isn't in the vocabulary -- worth
   confirming its exact derivation (first 3 chars of `diagnosis_code`?
   something else?) before assuming it can be used directly instead of
   truncating `diagnosis_code` ourselves.
9. **Locality/StatCan tables** (`locality_variables_subset`,
   `lookup_statcan_v2016`, `lookup_statcan_v2021`) are area-level
   socioeconomic covariates keyed by dissemination area (`da16uid`/`da21uid`),
   not clinical events -- almost certainly out of scope for the MEDS event
   stream itself (they're static, not time-stamped clinical facts), but
   worth an explicit decision rather than silently dropping them, since
   they could matter for a fairness/equity axis of the eventual GEMINI
   generalization assessment.

## MEDS mapping

| MEDS concept | GEMINI source | Notes |
| --- | --- | --- |
| **Subject** | `admdad_subset.patient_id_hashed` | Only table carrying it -- see open question 6. |
| **Visit** | `genc_id` (GEMINI's encounter id; treated as MEDS' `hadm_id`-equivalent) | Present on nearly every table; the actual join key throughout. |
| **Admission / discharge / death** | `admdad_subset`: `admission_date_time`, `discharge_date_time`; death via `derived_variables_subset.in_hospital_mortality_derived` | Mortality signal resolved, see open question 2 -- use the derived flag, not a `discharge_disposition` decode. |
| **ED triage / disposition** | `er_subset`: `triage_date_time`, `disposition_date_time`, plus `registration_date_time`, `physician_initial_assessment_date_time`, `ambulance_arrival_date_time`, `left_er_date_time` | Richer timestamp set than admission/discharge alone -- worth deciding which ED timestamps become MEDS events vs. stay as event attributes. |
| **ICU admission / discharge** | `ipscu_subset`: `scu_admit_date_time`, `scu_discharge_date_time`, `icu_flag` | `icu_flag` gates whether a `scu_*` stay counts as ICU specifically (`scu_unit_number` suggests non-ICU special-care units exist too, e.g. step-down). |
| **Labs** | `lab_subset`: code via `test_type_mapped_omop` (join `lookup_lab_concept`, deduplicated -- see open question 4), numeric parse of `result_value`, unit from `result_unit`, time from `collection_date_time` | `result_value` is `text` -- needs the same numeric-parse-with-fallback-to-categorical pattern already used for MIMIC/eICU labs. **SI units, not MIMIC/eICU's US-conventional units -- see [Units and canonical clinical ranges](#units-and-canonical-clinical-ranges).** |
| **Vitals** | `vitals_subset`: code via `measurement_mapped_omop` (join `lookup_vitals_concept` -- confirmed real and covers the common vitals, open question 3), numeric parse of `measurement_value`, unit from `measurement_unit`, time from `measure_date_time` | |
| **Medications** | `pharmacy_subset`: ingredient via `rxnorm_cache`/`lookup_pharmacy_mapping` (see open question 5) or `med_id_generic_name_raw` as a fallback identity; `med_start_date_time`/`med_end_date_time` for course timing; `route`, `dose_amount`/`dose_unit`, `frequency`, `PRN_IND` as attributes | Real datetime columns exist, but their values include physically impossible year outliers (1930-9022) -- needs the admission-date guard, see open question 7. |
| **Diagnoses** | `ipdiagnosis_subset`: `diagnosis_code` (ICD-10-CA, via `lookup_icd10_ca_description`), `diagnosis_type` (e.g. most-responsible vs. secondary), `icd3` backoff via `diagnosis_prefix` (see open question 8) | |
| **Procedures** | `ipintervention_subset`: `intervention_code` (CCI, via `lookup_cci`), `intervention_type`, `procedure_location` as an attribute | |
| **Imaging** | `radiology_subset`: code via `modality_mapped` + `body_part_mapped`, `ordered_date_time`/`performed_date_time` as two candidate event times (also has outlier years, same guard as pharmacy), `imaging_result` (free text -- likely excluded from the event stream itself, same treatment as free-text report fields elsewhere in the project) | |

**OMOP -> LOINC bridge**: `test_type_mapped_omop`/`measurement_mapped_omop`
are OMOP concept ids, not LOINC directly -- the intended bridge is a new
`gemini` table inside `odyssey/data/code_mapping.py`, the same module that
already keys MIMIC-IV/eICU concept rules by LOINC (see the README's
[Data pipeline](../README.md#data-pipeline)). That file is outside this
doc's ownership (lead session reviews it directly); this doc only records
the intended shape (OMOP concept id -> LOINC, via a standard OMOP-to-LOINC
crosswalk, not hand-mapped) **plus the unit each bridged concept actually
carries**, since the bridge is where a wrong unit would silently corrupt
every downstream clinical-range decision -- see the next section.

## Units and canonical clinical ranges

`odyssey/data/value_binning.py`'s `CANONICAL_CLINICAL_RANGES` already
supports per-unit variants (temperature's `"F"`/`"C"` split via
`odyssey.data.code_mapping.unit_for`) -- GEMINI needs the same treatment for
several labs, because **its `result_value`s are SI units** (confirmed from
`lab_subset.test_type_mapped_omop`'s OMOP concept descriptions: `[Moles/
volume]` = molar/SI concentration, e.g. creatinine in µmol/L, not the
`[Mass/volume]` mg/dL-style convention MIMIC-IV/eICU use and the existing
canonical thresholds are written against). **Concretely: creatinine's
existing range (LOINC `2160-0`: NORMAL below 1.5, HIGH 1.5-4.0, CRITICAL
above 4.0) is calibrated for mg/dL. A real SI creatinine value (~60-110
µmol/L normal range) is always below 1.5 on that scale, so every GEMINI
creatinine reading would silently bin as NORMAL regardless of true severity
-- a silent correctness bug, not a crash, if the existing range is reused
unchanged.**

The table below cross-references `extract-dry`'s top-200 lab/vitals concept
frequencies against every LOINC already in `CANONICAL_CLINICAL_RANGES`, plus
the unit *family* each GEMINI concept's OMOP description implies. **The
exact unit string (e.g. `umol/L` vs `mmol/L`, or the literal value of
`result_unit`/`measurement_unit`) is not yet confirmed** -- concept
descriptions distinguish Mass/volume from Moles/volume but not the specific
unit, and `extract-dry` doesn't currently sample `result_unit` values per
code. A follow-on `extract-dry` query (distinct `result_unit` per code, for
just these LOINCs) would close that gap before the bridge table is written;
flagging rather than guessing at exact unit strings here.

| LOINC | Concept | Existing range (assumes) | GEMINI OMOP concept(s) | Unit family found | Action needed |
| --- | --- | --- | --- | --- | --- |
| `8867-4` | Heart rate | count/min (unitless) | 3027018 Heart rate | count, no unit ambiguity | None -- safe as-is. |
| `8480-6` / `76534-7` | Systolic BP | mmHg | 3004249 Systolic blood pressure | mmHg (standard) | Confirm `measurement_unit` = mmHg, likely safe. |
| `9279-1` | Respiratory rate | count/min | 3024171 Respiratory rate | count, no unit ambiguity | None -- safe as-is. |
| `59408-5` | SpO2 | % | 3013502 Oxygen saturation in Blood | % (standard) | Confirm `measurement_unit` = %, likely safe. |
| `8310-5` | Temperature | already unit-split F/C | 3020891 Body temperature | unconfirmed which of F/C GEMINI uses | Sample `measurement_unit` -- the existing F/C split may already cover it once known. |
| `2160-0` | Creatinine | **mg/dL** | 3020564 Creatinine [Moles/volume] in Serum or Plasma | **Moles/volume (SI, likely umol/L)** | **New unit-split entry needed, same pattern as temperature -- do not reuse the mg/dL thresholds unchanged.** |
| `32693-4` | Lactate | **mmol/L already (matches MIMIC convention)** | 3018405 / 3008037 Lactate [Moles/volume] in Arterial/Venous blood; 3020138 also joins to a [Mass/volume] Serum/Plasma description at the same code | mixed -- see note | `3020138`'s dual Mass/Moles join may be the same `lookup_lab_concept` duplicate-row issue (open question 4) rather than a real second unit convention -- verify before assuming two conventions coexist under one code. |

Not yet cross-referenced (no existing canonical range, so no silent-bug risk,
but candidates for new ranges given how frequently they appear): sodium
(3019550), potassium (3023103), chloride (3014576), hemoglobin (3000963),
hematocrit (3009542), platelets (3007461), glucose (3040151/3013826),
bicarbonate (3016293), INR (3032080), bilirubin.total (3006140), ALT/AST
(3006923/3013721) -- all in the top-30 most frequent lab codes.

## Sharding and output

Same pattern as the MIMIC-IV/eICU extractions: shard by patient hash, not
by encounter or arrival order, so shards are reproducible and balanced.
**MEDS parquet output is written to the enclave's own NFS storage only,
never to git** -- consistent with `docs/gemini.md`'s governance rules
(only small, aggregate, cell-suppressed reports leave GEMINI; the actual
per-patient MEDS shards are exactly the kind of patient-level data that
must never cross that boundary). The extraction script itself (schema and
logic) is what gets pushed/pulled through the git-only channel, same as
every other GEMINI-facing script; its *output* stays on the node.

## Status

`extract-dry` has run once for real (`scripts/gemini/out/extract_dry.{json,md}`,
datacut `subdural_hematoma_v1_0_0`), resolving open questions 2 and 3 and
sharpening 4 and 7 from guesses into confirmed findings; a follow-on
`extract-dry` run now also samples raw `result_unit`/`measurement_unit`
values for every LOINC-mapped concept in the units table above plus all 13
vitals concepts, to close that section's remaining gap.

`scripts/gemini/extract_meds.py` implements the resolved parts of the
mapping above: admission-anchored `+-1y` timestamp guards (pharmacy,
radiology), the deduplicated `lookup_lab_concept` lookup, and subject-hash
sharding, streaming throughout (never loads a whole source table into
memory). `code` values are GEMINI's own raw identifiers, namespaced
(`LAB//<omop_id>`, `DIAGNOSIS//<icd10ca>`, ...) -- the OMOP -> LOINC bridge
and value-binning unit-split entries are a separate, later stage (owned by
the lead session, see the OMOP -> LOINC bridge note above), not yet run
against real data. Real, aggregate-only run output (rounded row/subject/
shard counts) lands in `scripts/gemini/out/extraction_summary.json` once
run via `scripts/gemini/run.sh extract` -- the actual MEDS parquet shards
never leave the enclave.

Still open before a full real run: the actual training datacut (question 1,
still `subdural_hematoma_v1_0_0` at time of writing), the pharmacy
dual-mapping relationship (question 5, `rxnorm_cache` vs
`lookup_pharmacy_mapping` -- `extract_meds.py` currently sidesteps this by
using `med_id_generic_name_raw` as a raw fallback identity rather than
picking one), `diagnosis_prefix`'s exact derivation (question 8, not yet
used by the extractor), and the locality/StatCan scope decision (question
9, not extracted at all currently -- treated as out of scope for the MEDS
event stream per that question's reasoning).
