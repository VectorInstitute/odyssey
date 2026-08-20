# GEMINI to MEDS

Draft mapping from GEMINI's real schema (`scripts/gemini/out/schema.json`,
`schema.md`, datacut `subdural_hematoma_v1_0_0` -- a diagnosis-filtered cohort
useful for schema exploration, **not** the general internal-medicine
population odyssey's own training cut will use, see
[Open questions](#open-questions)) to the MEDS event-stream shape the rest of
the pipeline already consumes for MIMIC-IV and eICU. Everything below is a
target to validate against `extract-dry`'s design-critical queries
(`scripts/gemini/extract_dry.py`) before writing real extraction code, not a
spec to implement blind -- see `docs/gemini.md` for why (git-only channel,
one real run per round trip).

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
2. **Mortality signal**: `admdad_subset.discharge_disposition` is an
   uncoded integer with no lookup table found in schema.json (unlike
   `discharge_disposition`-style codes elsewhere, which usually have a
   lookup) -- the mapping from code to "died" isn't yet confirmed.
   `derived_variables_subset.in_hospital_mortality_derived` is a
   precomputed boolean that may be more reliable, but its exact derivation
   logic isn't ours to verify from the schema alone. Check
   `lookup_cihi_codes` filtered to `column_name = 'discharge_disposition'`
   first (a generic coded-value lookup covering multiple columns); fall
   back to the derived flag if that doesn't resolve it.
3. **`lookup_vitals_concept` is genuinely empty** (confirmed via a real
   `EXISTS` check, not just the schema report's rounded "0" -- see
   `extract_dry.lookup_emptiness`), so `vitals_subset.measurement_mapped_omop`
   has no local name lookup at all. `measurement_name` (raw text) is the
   only source of a human-readable vital label until/unless that lookup is
   populated -- concept-rule mapping for vitals may need to key off raw
   text rather than the OMOP id, unlike labs.
4. **Two parallel pharmacy-identity mapping paths**: `rxnorm_cache`
   (`rxcui`, `raw_input`, `search_type`) and `lookup_pharmacy_mapping`
   (`raw_input`, `rxnorm_match`, `drug_group`) both look like
   raw-drug-name-to-RxNorm bridges. Whether they're redundant, one
   supersedes the other, or they cover different input types
   (`search_type`/`project_name` suggest possibly different callers) isn't
   confirmed -- check for overlapping `raw_input` values and whether their
   `rxcui`/`rxnorm_match` outputs ever disagree before picking one.
5. **Patient-level linkage across encounters**: `patient_id_hashed` appears
   *only* in `admdad_subset`, not in any event table -- every event table
   carries `genc_id` (encounter) only. Linking two encounters to the same
   real patient always requires a join through `admdad_subset`, and
   whether `patient_id_hashed` is stable/reusable across a patient's
   multiple encounters in this data cut isn't yet confirmed. This is the
   same "subject: patient, or a hospital encounter if patient-level
   linkage isn't straightforward" question this doc's predecessor section
   in `docs/gemini.md` already flagged -- now grounded in a concrete schema
   fact rather than a guess.
6. **Text datetime format**: every date/time column in the schema is typed
   `text`, not a native `date`/`timestamp` (except `lookup_data_coverage`'s
   `min_date`/`max_date`, which are real `date`). `extract_dry.py`'s
   `table_date_ranges` query assumes a leading `YYYY` and extracts it via
   regex, ignoring non-matching rows -- real format(s) not yet confirmed
   from actual values, only from the column *type*.
7. **`diagnosis_prefix`** on `ipdiagnosis_subset` may already be the
   3-character ICD-10-CA backoff code this project's other concept rules
   use when a fully-specified code isn't in the vocabulary -- worth
   confirming its exact derivation (first 3 chars of `diagnosis_code`?
   something else?) before assuming it can be used directly instead of
   truncating `diagnosis_code` ourselves.
8. **Locality/StatCan tables** (`locality_variables_subset`,
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
| **Subject** | `admdad_subset.patient_id_hashed` | Only table carrying it -- see open question 5. |
| **Visit** | `genc_id` (GEMINI's encounter id; treated as MEDS' `hadm_id`-equivalent) | Present on nearly every table; the actual join key throughout. |
| **Admission / discharge / death** | `admdad_subset`: `admission_date_time`, `discharge_date_time`, `discharge_disposition` | Death via `discharge_disposition` -- see open question 2. |
| **ED triage / disposition** | `er_subset`: `triage_date_time`, `disposition_date_time`, plus `registration_date_time`, `physician_initial_assessment_date_time`, `ambulance_arrival_date_time`, `left_er_date_time` | Richer timestamp set than admission/discharge alone -- worth deciding which ED timestamps become MEDS events vs. stay as event attributes. |
| **ICU admission / discharge** | `ipscu_subset`: `scu_admit_date_time`, `scu_discharge_date_time`, `icu_flag` | `icu_flag` gates whether a `scu_*` stay counts as ICU specifically (`scu_unit_number` suggests non-ICU special-care units exist too, e.g. step-down). |
| **Labs** | `lab_subset`: code via `test_type_mapped_omop` (join `lookup_lab_concept`, ~1000 concepts -- a small, controlled vocabulary), numeric parse of `result_value`, unit from `result_unit`, time from `collection_date_time` | `result_value` is `text` -- needs the same numeric-parse-with-fallback-to-categorical pattern already used for MIMIC/eICU labs. |
| **Vitals** | `vitals_subset`: code via `measurement_mapped_omop` (lookup empty, see open question 3 -- fall back to `measurement_name`), numeric parse of `measurement_value`, unit from `measurement_unit`, time from `measure_date_time` | |
| **Medications** | `pharmacy_subset`: ingredient via `rxnorm_cache`/`lookup_pharmacy_mapping` (see open question 4) or `med_id_generic_name_raw` as a fallback identity; `med_start_date_time`/`med_end_date_time` for course timing; `route`, `dose_amount`/`dose_unit`, `frequency`, `PRN_IND` as attributes | Confirms `docs/gemini.md`'s framing question directly: yes, usable datetime columns exist (`med_start_date_time`/`med_end_date_time`), text-typed like everywhere else. |
| **Diagnoses** | `ipdiagnosis_subset`: `diagnosis_code` (ICD-10-CA, via `lookup_icd10_ca_description`), `diagnosis_type` (e.g. most-responsible vs. secondary), `icd3` backoff via `diagnosis_prefix` (see open question 7) | |
| **Procedures** | `ipintervention_subset`: `intervention_code` (CCI, via `lookup_cci`), `intervention_type`, `procedure_location` as an attribute | |
| **Imaging** | `radiology_subset`: code via `modality_mapped` + `body_part_mapped`, `ordered_date_time`/`performed_date_time` as two candidate event times, `imaging_result` (free text -- likely excluded from the event stream itself, same treatment as free-text report fields elsewhere in the project) | |

**OMOP -> LOINC bridge**: `test_type_mapped_omop`/`measurement_mapped_omop`
are OMOP concept ids, not LOINC directly -- the intended bridge is a new
`gemini` table inside `odyssey/data/code_mapping.py`, the same module that
already keys MIMIC-IV/eICU concept rules by LOINC (see the README's
[Data pipeline](../README.md#data-pipeline)). That file is outside this
doc's ownership (lead session reviews it directly); this doc only records
the intended shape (OMOP concept id -> LOINC, via a standard OMOP-to-LOINC
crosswalk, not hand-mapped) so the extraction spec and the concept-mapping
code stay consistent once both are written.

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

No extraction code exists yet. Next step is resolving the open questions
above via `extract-dry`'s design-critical queries (already run once,
`scripts/gemini/out/extract_dry.{json,md}` once Amrit's next run lands),
then a real extraction spec once GEMINI_DATACUT points at the actual
training cut (open question 1) rather than the schema-exploration cohort.
