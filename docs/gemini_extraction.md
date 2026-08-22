# GEMINI to MEDS

Draft mapping from GEMINI's real schema and its first `extract-dry` run
(`scripts/gemini/out/{schema,extract_dry}.{json,md}`, datacut
`subdural_hematoma_v1_0_0` -- a diagnosis-filtered cohort useful for schema
exploration, **not** the general internal-medicine population odyssey's own
training cut will use, see [Open questions](#open-questions)) to the MEDS
event-stream shape the rest of the pipeline already consumes for MIMIC-IV and
eICU. Everything below is checked against real query results, not a guess
from column names/types alone. `scripts/gemini/extract_meds.py` implements
this mapping and is the actual extraction spec in force -- see
[Status](#status).

## Why no MESSY spec

MIMIC-IV and eICU are both extracted through a declarative MESSY (MEDS-Extract
Specification Syntax) YAML spec (`specs/eICU.yaml`, run via `meds-extract-run`)
against **files**: a downloaded or already-local directory of CSVs/parquet.
GEMINI has no equivalent file distribution at all -- the only access is a live
SQL connection to the GEMINI node, and `meds-extract-run`'s whole design assumes
a file-based `input_dir`, not a remote database. Making GEMINI fit that model
would mean dumping its tables to files first, which this project deliberately
doesn't do: it would roughly **double** the enclave's storage footprint (the
same data living twice -- once in Postgres, once as flat-file dumps) for no
functional benefit, and it would require validating the entire
`meds-extract-run`/MESSY toolchain actually runs correctly *inside* the
closed enclave (nobody has done this, and it's a real, untested risk, not a
given). SQL-based extraction avoids both costs: `extract_meds.py` queries
GEMINI directly and writes MEDS parquet shards straight out, no intermediate
file dump ever exists. **The output is standard MEDS parquet either way** --
downstream (tokenization, concept labeling, training) is source-agnostic and
doesn't know or care whether a shard came from a MESSY YAML spec or a
hand-written SQL extractor. See `specs/GEMINI.md` for the one-line pointer
anyone browsing `specs/` will find instead of a `GEMINI.yaml`.

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
2. ~~**Mortality signal**~~ **Resolved, and implemented (2026-08-22)**:
   `derived_variables_subset.in_hospital_mortality_derived`
   is essentially fully populated (fewer than 6 nulls out of 2,268,000 rows,
   per `extract-dry`'s null-fraction check) -- used directly as the
   primary (and only) mortality signal rather than decoding
   `discharge_disposition` ourselves. `extract_meds.extract_death` emits a
   single bare `MEDS_DEATH` event (matching MIMIC-IV's own convention;
   `vocabulary.py`'s `"MEDS_DEATH": DEMOGRAPHIC_TYPE` needed no change)
   timed at `admdad_subset.discharge_date_time` where the derived flag is
   true, guarded against discharge-before-admission.
   `discharge_disposition` (candidate death codes `{7, 72, 73, 74}`) is
   still read alongside it, but purely as a cross-check tally
   (both-agree / derived-only / disposition-only counts, logged once per
   extraction run) -- it never gates emission. **Real run numbers
   (2026-08-22, post-durability-fix rerun)**: 170,148 both agree, 0
   derived-only, 11 disposition-only, out of 2,268,279 admissions --
   near-perfect agreement, validating both the derived-flag-primary
   decision and the candidate code set; the 11 disposition-only rows
   (0.006%) read as coding-edge noise, not a systematic gap. Cohort
   in-hospital mortality is ~7.5% (170,148 / 2,268,279). The death
   extractor itself ran clean end-to-end (2.27M rows, ~23s, all five
   batches) on this rerun.
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
| **Admission / discharge / death** | `admdad_subset`: `admission_date_time`, `discharge_date_time`; death via `derived_variables_subset.in_hospital_mortality_derived` -> bare `MEDS_DEATH` | Implemented, see open question 2 -- the derived flag is the sole primary signal; `discharge_disposition` is a logged cross-check only, never a `discharge_disposition` decode of its own. |
| **ED registration / triage / out** | `er_subset`: `registration_date_time` -> `ED_REGISTRATION`, `triage_date_time` -> `ED_TRIAGE` (new prefix), `left_er_date_time` -> `ED_OUT` | Resolved: three of `er_subset`'s six timestamps become events, matching MIMIC's own `ED_REGISTRATION`/`ED_OUT` prefixes (already in `odyssey/data/vocabulary.py`); `disposition_date_time`/`physician_initial_assessment_date_time`/`ambulance_arrival_date_time` stay unextracted (describe the visit, don't bound a stage transition). Same admission-window guard as pharmacy/radiology. A `genc_id` here can be missing from `admdad_subset` entirely (ED visit, no admission) -- dropped via the existing subject-lookup-miss path, not a new failure mode. |
| **ED diagnoses** | `erdiagnosis_subset.er_diagnosis_code` -> `ED_DIAGNOSIS//<code>` (new prefix, kept distinct from `ipdiagnosis_subset`'s `DIAGNOSIS//`) | No event-level timestamp -- attributed to the encounter's admission time, same convention as `physicians_subset`. |
| **ED procedures** | `erintervention_subset.intervention_code` -> `PROCEDURE//<code>` (reuses `ipintervention_subset`'s own prefix -- same CCI coding system, not a new ER-specific vocabulary) | Two passes: timed (own timestamp, admission-window guard) and `_untimed` (the exact complement, attributed to admission time). `intervention_episode_start_date_time` is blank (not `NULL`) on ~96.7% of coded rows -- the timed pass alone kept only ~90k of ~2.94M coded interventions; the untimed pass rescues the rest. Real incident, see "Real run, real incident, real fix" below. |
| **ED consults** | `erconsults_subset.consult_service_code` -> `ER_CONSULT//<code>` (new prefix, `OTHER_TYPE`) | Timed at `consult_request_date_time`; `consult_arrival_date_time` not extracted as a second event. |
| **Transfers** | `lookup_transfer_subset.institution_to_mns` -> `TRANSFER_TO//<institution>` (already in `odyssey/data/vocabulary.py`, matching MIMIC's own convention) | Despite the table name, real per-encounter rows, not a static lookup. No event-level timestamp -- attributed to admission time. |
| **Billing (CMG)** | `ipcmg_subset.cmg` -> `BILLING_CMG//<cmg>` (new prefix, `BILLING_TYPE`) | Canada's CIHI casemix-group system -- kept distinct from MIMIC's own `DRG` prefix (different vocabulary, same type bucket). No event-level timestamp -- attributed to discharge time (grouper codes finalize at stay close, same reasoning as diagnoses below). |
| **Billing (HIG)** | `iphig_subset.hig_code` -> `BILLING_HIG//<code>` (new prefix, `BILLING_TYPE`) | CIHI's Health-based Inpatient Group system, distinct from both CMG and DRG. Same discharge-time attribution as CMG. |
| **ICU admission / discharge** | `ipscu_subset`: `scu_admit_date_time`, `scu_discharge_date_time`, `icu_flag` | `icu_flag` gates whether a `scu_*` stay counts as ICU specifically (`scu_unit_number` suggests non-ICU special-care units exist too, e.g. step-down). **Incident, resolved (2026-08-22)**: the real finalized dataset's code inventory had zero `ICU_ADMISSION`/`ICU_DISCHARGE` codes despite `ipscu_subset` having ~631k rows. Root cause, forensically confirmed (not the `icu_flag` boolean-coercion bug, which had already landed by then, confirmed via the operator's direct query -- 541,688 real `t` rows correctly parsed): `MedsShardWriter` buffered rows in memory and only wrote a valid Parquet footer at `close()`, called exactly once at the very end of the whole run -- but each table's manifest entry was marked `"complete"` immediately after its own generator drained, long before that single end-of-run `close()`. Attempt 5: `ipscu_subset`'s ~1.08M events buffered under the flush threshold, manifest marked complete, then `lab_subset`'s crash (`103@POST`, see the ER incident above) killed the process before `close()` ever ran -- silently losing every `ipscu_subset` row while the manifest kept claiming completion, so every later resumed run skipped re-extracting it. Confirmed by accounting: `finalize`'s shard totals exceeded the per-run table sums by almost exactly `admdad_subset`'s events alone, ~zero from `ipscu_subset`. Fixed in `extract_meds.py` (`MedsShardWriter.flush_all`, called per table before its manifest mark, plus a generator-emitted-vs-durably-written row assert) -- see the Status entry below. The already-planned re-extract heals the data hole for free (fresh manifest re-extracts `ipscu_subset` under the fixed code). |
| **Labs** | `lab_subset`: code via `test_type_mapped_omop` (join `lookup_lab_concept`, deduplicated -- see open question 4), numeric parse of `result_value`, unit from `result_unit`, time from `collection_date_time` | `result_value` is `text` -- needs the same numeric-parse-with-fallback-to-categorical pattern already used for MIMIC/eICU labs. **SI units, not MIMIC/eICU's US-conventional units -- see [Units and canonical clinical ranges](#units-and-canonical-clinical-ranges).** |
| **Vitals** | `vitals_subset`: code via `measurement_mapped_omop` (join `lookup_vitals_concept` -- confirmed real and covers the common vitals, open question 3), numeric parse of `measurement_value`, unit from `measurement_unit`, time from `measure_date_time` | Two passes: mapped (above) and `_unmapped` (the exact complement, `measurement_mapped_omop` null). ~119M of ~412M rows have no concept mapping at all -- the entire 71%-retention story, confirmed real (FiO2/oxygen-delivery/pain-score/pupillary-response/LOC names among the top unmapped) -- rescued via `measurement_name` as a fallback identity, the eICU convention. See "Real run, real incident, real fix" below. |
| **Medications** | `pharmacy_subset`: ingredient via `rxnorm_cache`/`lookup_pharmacy_mapping` (see open question 5) or `med_id_generic_name_raw` as a fallback identity; `med_start_date_time`/`med_end_date_time` for course timing; `route`, `dose_amount`/`dose_unit`, `frequency`, `PRN_IND` as attributes | Real datetime columns exist, but their values include physically impossible year outliers (1930-9022) -- needs the admission-date guard, see open question 7. |
| **Diagnoses** | `ipdiagnosis_subset`: `diagnosis_code` (ICD-10-CA, via `lookup_icd10_ca_description`), `diagnosis_type` (e.g. most-responsible vs. secondary), `icd3` backoff via `diagnosis_prefix` (see open question 8) | |
| **Procedures** | `ipintervention_subset`: `intervention_code` (CCI, via `lookup_cci`), `intervention_type`, `procedure_location` as an attribute | |
| **Imaging** | `radiology_subset`: code via `modality_mapped` + `body_part_mapped`, `ordered_date_time`/`performed_date_time` as two candidate event times (also has outlier years, same guard as pharmacy), `imaging_result` (free text -- likely excluded from the event stream itself, same treatment as free-text report fields elsewhere in the project) | |
| **Provider** | `physicians_subset`: `mrp_cpso_hashed`/`adm_phy_cpso_hashed`/`dis_phy_cpso_hashed` (already-hashed CPSO ids), one event each at the encounter's admission time | Not used by any current MEDS-consuming stage -- preserved for a *tabled*, not abandoned, physician-preference IV study (see the note below the table). |

**Why provider ids are preserved**: a physician-preference instrumental-variable study (provider assignment as the instrument, comparing IV-ICL vs. 2SLS) was proposed and debated for the comparative-methods paper; the debate tabled it as a separate paper -- real instrument-validity risk (is admitting-provider assignment actually quasi-random within a GEMINI hospital, or driven by call schedule/acuity-matched triage?) and real dilution risk for a paper already spanning four model families across three datasets, see the debate decision commit. Tabling it isn't the same as ruling it out: extracting `physicians_subset` now, while `extract_meds.py` is already being touched for other reasons, is cheap (one more unordered table scan, ~2.27M rows, comparable to `admdad_subset`) and keeps the option open for a genuinely separate follow-on study without needing another full extraction pass through GEMINI later.

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

**Text/varchar columns are truncated to 128 characters and newline/carriage-return
stripped server-side, in the `COPY`/`SELECT` itself** (`extract_meds.py`'s
`_select_expr_sql`). Real incident this closes: `_CopyChunkSink` finds row
boundaries by counting raw `\n` bytes, not by CSV-aware parsing, so a
literal newline embedded in a quoted CSV field (Postgres quotes any field
containing one) would silently corrupt row alignment -- confirmed live via
`lab_subset.result_value`, which is genuine free text (`'103@POST'` and
similar), not the numeric-only reading the column name suggests. Every
column confirmed `integer`/`boolean` in `scripts/gemini/out/schema.md`
(`genc_id`, `test_type_mapped_omop`, `measurement_mapped_omop`,
`icu_flag`) is exempt -- those can never contain a newline by
construction. Every other selected column, across all 9 source tables
(datetime columns, medical codes, hashed physician/patient identifiers,
drug names, lab/vitals values and units), is `text` or `character
varying` and gets the wrap; 128 characters is generous for all of them in
practice.

## Status

`extract-dry` has run once for real (`scripts/gemini/out/extract_dry.{json,md}`,
datacut `subdural_hematoma_v1_0_0`), resolving open questions 2 and 3 and
sharpening 4 and 7 from guesses into confirmed findings; a follow-on
`extract-dry` run now also samples raw `result_unit`/`measurement_unit`
values for every LOINC-mapped concept in the units table above plus all 13
vitals concepts, to close that section's remaining gap.

`scripts/gemini/extract_meds.py` implements the resolved parts of the
mapping above: admission-anchored `+-1y` timestamp guards (pharmacy,
radiology), the deduplicated `lookup_lab_concept` lookup, subject-hash
sharding, and a preflight check (one `COUNT DISTINCT` on `admdad_subset` to
size the shard count, then raising the process's open-file-descriptor limit
if `MedsShardWriter`'s one-writer-per-shard design needs more than the
default allows) -- all streaming throughout (never loads a whole source
table into memory). `code` values are GEMINI's own raw identifiers,
namespaced (`DIAGNOSIS//<icd10ca>`, `PROCEDURE//<cci>`, ...); labs and
vitals additionally carry their literal, normalized unit as a third segment
-- `LAB//<omop_id>//<unit>` / `VITALS//<omop_id>//<unit>`, the same shape
MIMIC-IV's own `LAB//<itemid>//<unit>` codes use -- since GEMINI is
multi-hospital and the same OMOP concept can carry a different unit at a
different site (see [Units and canonical clinical ranges](#units-and-canonical-clinical-ranges)):
the unit has to be part of the token identity so mixed-unit values never
share a quantile bin. The OMOP -> LOINC bridge and value-binning
per-unit-family clinical ranges are a separate, later stage (owned by the
lead session, see the OMOP -> LOINC bridge note above), not yet run against
real data.

**Schema-completeness gap, found and closed (2026-08-21)**: the tables
above covered admission/discharge, ICU, labs, vitals, medications,
inpatient diagnoses/procedures, imaging, and providers -- but left the ED
family (`er_subset`, `erdiagnosis_subset`, `erintervention_subset`,
`erconsults_subset`), unit transfers (`lookup_transfer_subset`), and
billing/casemix (`ipcmg_subset`, `iphig_subset`) unextracted entirely.
Added as purely additive `table_generators` entries (see the MEDS mapping
table above for the per-table event shape and code prefixes) -- existing
table entries, resumability, and guards untouched.
`derived_variables_subset`/`locality_variables_subset` (per-admission
sidecars) and `cohort` (duplicates `admdad_subset`) remain deliberately
out of scope; `transfusion` isn't in this datacut at all
(`lookup_transfusion` is empty).

Real, aggregate-only run output (rounded row/subject/shard
counts) lands in `scripts/gemini/out/extraction_summary.json` once run via
`scripts/gemini/run.sh extract` -- the actual MEDS parquet shards never
leave the enclave.

`scripts/gemini/finalize_meds.py` (`run.sh finalize`, run only once
`extract`'s manifest shows every table complete) rewrites `extract`'s flat
output into the MEDS-conformant layout
`odyssey/data/meds_validation.py`'s `validate_meds_dataset` checks --
`subject_id` remapped from the raw hashed patient string to a stable,
deterministic `Int64` (`subject_id_mapping.parquet`, kept server-side only
like everything else here, never under `metadata/`), a seeded
subject-random 80/10/10 train/tuning/held_out split (`MEDS_extract`'s own
default `split_fracs_dict`, not a GEMINI-specific choice) baked into
`metadata/subject_splits.parquet`, and shards resharded under
`data/<split>/` -- compacting the multi-part resumability layout away in
the same pass. Two stronger, GEMINI-specific evaluation protocols are
deliberately left derivable at eval time rather than baked into the split:
hospital-held-out, via a small `metadata/hadm_id_hospital.parquet` sidecar
(`admdad_subset.hospital_num`, joined on the `hadm_id` every MEDS row
already carries), and temporal validation, directly against the real event
timestamps already in the output. Ends by running
`validate_meds_dataset(root, deep=True)` itself and refusing to declare
success on anything but zero errors -- see that module's own docstring for
the full design, memory shape, and crash semantics.

Real run, real incident, real fix (2026-08-21): the operator's first live run of
`extract_meds.py` measured ~400 rows/s on the small tables and, on
`lab_subset`, zero output growth over 5 minutes -- these matviews carry no
index on `row_num`, so the original keyset-paginated fetch
(`WHERE row_num > :cursor ORDER BY row_num LIMIT ...`) forced a full
scan-and-sort of the table before returning even the first row of any page,
unusable at `lab_subset`'s real ~659M rows. He stopped the job.
`extract_meds.py`'s fetch layer now reads each source table exactly once,
unordered (`COPY ... TO STDOUT`, falling back to an unordered server-side
cursor -- nothing downstream needs source order: `MedsShardWriter` hashes by
subject, and any per-subject ordering comes from `build_patient_sequence`'s
own sort later), with resumability at table granularity (a killed table
restarts from scratch; a completed table is skipped on resume) rather than
the row-level checkpointing the sorted approach would have needed. Every
per-table transform is vectorized (polars, whole-chunk operations) rather
than a Python per-row loop, the other half of the original performance
request -- see `extract_meds.py`'s module docstring for the full
before/after measurements and reasoning.

Still open before a full real run: the actual training datacut (question 1,
still `subdural_hematoma_v1_0_0` at time of writing), the pharmacy
dual-mapping relationship (question 5, `rxnorm_cache` vs
`lookup_pharmacy_mapping` -- `extract_meds.py` currently sidesteps this by
using `med_id_generic_name_raw` as a raw fallback identity rather than
picking one), `diagnosis_prefix`'s exact derivation (question 8, not yet
used by the extractor), and the locality/StatCan scope decision (question
9, not extracted at all currently -- treated as out of scope for the MEDS
event stream per that question's reasoning).

Real run, real incident, real fix (2026-08-21, second wave): the ER
extraction landed 90k of ~2.94M coded `erintervention_subset` rows --
`intervention_episode_start_date_time` is blank (empty string, not SQL
`NULL`) on 2,838,024 of 2,936,194 rows (96.7%), invisible to
`extract-dry`'s `null_fraction` at the time (a plain `COUNT(*) -
COUNT(column)`, which counts a blank string as "present"), so the report
showed ~768k nulls where the real blank-or-null count was ~2.8M. Root
cause confirmed server-side, not an extractor bug: within-guard-window
real timestamps = 98,168, matching the ~90k observed after the code/id
filters. `null_fraction` now counts blank/whitespace-only strings as null
for `text`/`character varying` columns (`NULLIF(TRIM(column::text), '')`)
-- see `extract_dry.py`'s `_is_blank_string_type`. Rescued by
`extract_er_procedures_untimed`, the exact complement of
`extract_er_procedures`'s own admission-guard predicate, attributed to
admission time instead -- see that function's docstring for why this is a
second pass rather than one function silently preferring the coarser
attribution.

`vitals_subset`'s own 71% retention turned out to be a different, larger
mechanism, not the blank-timestamp one -- measured blank timestamps there
are only ~44k. The real cause: `extract_vitals` drops every row whose
`measurement_mapped_omop` doesn't resolve to a concept id at all, ~119M of
~412M rows. Real unmapped names by frequency: FiO2 variants, oxygen
delivery method/flow/therapy, pain score, `NEURO.PIR`/`PIA` (pupillary
response -- core signal for a subdural-hematoma cohort), `VS.NWS.LOC1`
(level of consciousness), plus assorted qualifier fields. Rescued by
`extract_vitals_unmapped`, the exact complement of `extract_vitals`'s own
concept-id filter, via `measurement_name` as a fallback identity -- the
same convention eICU's own extraction already uses for an unmapped
vital/lab. Name normalization is casefold + whitespace-collapse only
(`_normalize_name_series`), deliberately without a cross-name
canonicalization map -- unlike units, there is no known-variant-cluster
map for vitals *names* yet; collapsing spelling variants of the same
measurement onto one code is future `odyssey/data/code_mapping.py` work.
This closes every retention anomaly found in the real-run accounting so
far.

**MEDS_DEATH landed, and the sequencing plan to get it into the trained
dataset (2026-08-22)**: `extract_death` (open question 2, MEDS mapping
table above) is implemented and merged. Landing it in the actual dataset
requires a full re-extract, not an incremental one -- `finalize_meds.py`
deletes `extract`'s flat per-run layout once it rewrites it into the
MEDS-conformant `data/` layout, so a schema addition discovered
*after* a `finalize` has already run has no partial-update path back into
`data/` today; the only way in is a full `extract` (all tables, from a
wiped manifest) followed by a full `finalize`. **Sequencing (the operator's
order)**: (1) the operator's direct `icu_flag` query against `ipscu_subset`
first (see the ICU mapping row above), so any recognized-values fix lands
in the same re-extract rather than needing a second one; (2) full
re-extract, run concurrently with the in-flight `train-full` (`extract`'s
flat writes never touch `data/`, so this is safe to overlap); (3)
`finalize` strictly after both `train-full` and `eval-forecast` complete,
since `finalize` deletes the flat layout those don't depend on but a
concurrent `finalize` run competing for enclave I/O during eval would.
**Determinism expectation**: subject-id remapping and the seeded
80/10/10 split are both deterministic (stable hash / seeded RNG over the
same subject population), so the re-finalized dataset should be
identical to the current one *plus* death events layered into existing
subjects' shards -- no subject should move splits, gain, or lose a shard.
Verify, don't just assert: once the re-finalize completes, compare
per-split subject counts against the previous run's
`finalize_summary.json` (exact match expected) before treating the new
`data/` layout as ready to train on.

If a third schema addition after a `finalize` happens, a `finalize
--amend` mode (partitioned-sink merge of only the new event rows into the
existing `data/` layout, skipping the full compacting rewrite) becomes
worth building -- not yet, this is only the second occurrence.

**Real incident, real fix (2026-08-22): a completed table's manifest mark
could outlive its own data.** Root cause of the ICU zero-codes incident
above, forensically confirmed: `MedsShardWriter` buffered rows in memory
and only wrote a valid Parquet footer at `close()`, called exactly once
at the very end of the whole run, across every table -- but each table's
manifest entry was marked `"complete"` immediately after its own
generator drained, well before that single end-of-run `close()`.
Attempt 5: `ipscu_subset` buffered under threshold, manifest marked
complete, then `lab_subset`'s crash killed the process before `close()`
ever ran -- silently losing every `ipscu_subset` row while the manifest
kept claiming completion. Fixed: `MedsShardWriter.flush_all()`
force-flushes every buffered shard and closes its writer (a real footer,
not just buffered `write_table()` calls) without ending the writer's
lifetime -- the next table's writes to the same shard reopen a fresh
`_partN` file via the existing multi-part-shard resumability design.
`run_extraction` calls it immediately after each table's generator
drains, *before* that table's manifest mark -- durability precedes the
claim of durability. A new per-table assert (generator-emitted rows ==
writer-durable rows written + dropped) raises loudly on any future
mismatch instead of letting one through silently. Deliberately does
*not* also close writers on an arbitrary exception mid-table: since
`flush_all()` already durably closes every *completed* table before the
next one starts, the only writer state still open at exception time
belongs to the table currently mid-flight, and closing it there would
finalize a valid-but-partial file that a retry (which redoes the whole
table from scratch) would then double-count alongside its own fresh part
files -- left unclosed, that case still reads as a loud, footer-missing
parquet error on next read rather than a silent duplicate.

**Real incident, real fix (2026-08-22, third): two OOM kills on
`lab_subset` (318M then 340M buffered rows, 64 GB then 96 GB jobs).**
`SHARD_FLUSH_ROW_THRESHOLD` only bounds *each* shard's own buffer, but
subject-hash sharding is uniform, so with ~1,118 shards and a table's
rows arriving evenly across subjects, every shard tends to cross that
threshold within the same processing window -- a synchronized aggregate
high-water of up to `SHARD_FLUSH_ROW_THRESHOLD * n_shards` (a real,
measured ~279M rows) resident at once, at ~250-400 bytes/row in a pandas
object-dtype frame with string codes (not an earlier ~130-byte
estimate), plus each flush's own `pd.concat` copy -- enough to exceed
even a 96 GB job. Fixed: a new global cap,
`WRITER_MAX_BUFFERED_ROWS` (env-tunable via
`GEMINI_WRITER_MAX_BUFFERED_ROWS`, default 40M rows), tracks rows
buffered across every shard and force-flushes the fullest shards first
once exceeded -- both bounding peak memory and desynchronizing future
waves, since a flushed shard restarts from 0 while others keep
accumulating independently. `SHARD_FLUSH_ROW_THRESHOLD` itself and the
`flush_all`/durability semantics from the previous incident are
untouched. A reference-retention leak in the per-table writer lifecycle
(frames or closed `pq.ParquetWriter` objects never actually released)
was also suspected as a possible regression from that same commit --
audited the exact diff line-by-line and wrote an empirical
weakref-based regression test (feed many batches through many
`flush_all()` cycles, assert every internal buffer structure is
genuinely empty afterward and closed writers are actually
garbage-collected, not just zeroed) -- the test passes, finding no such
leak. The global cap is the fix; the leak hypothesis is not confirmed
(kept open pending real-run RSS diagnostics, but the cap is correct
regardless of the exact mechanism). Total buffered rows are now logged
in the per-batch timing line so this class of question is answerable
from logs directly next time.
