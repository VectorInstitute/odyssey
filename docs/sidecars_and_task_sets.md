# Sidecars and task sets

How label-only auxiliary data and versioned concept/alert-event registries
work together, and what a checkpoint's `task_set` actually pins.

## Sidecars

A sidecar is a small parquet under `<meds root>/sidecars/<name>.parquet`,
sibling to `data/` and `metadata/` in a MEDS extraction. It carries
information an outcome definition needs but the extraction does not
tokenize -- e.g. culture specimen times for the Sepsis-3 label. Sidecars
are read only by the label pipeline (`odyssey.data.concepts`,
`odyssey.data.alert_events`); tokenization and every baseline's
featurization never see them. **No model family sees a sidecar as
input** -- this is the mechanism's whole point: adding a new outcome
definition never gives one family information the others lack.

### Mechanism (`odyssey/data/sidecars.py`)

- `sidecar_root_for(shard_dir)`: given a split directory
  (`<root>/data/train`) or `<root>/data` or `<root>` itself, returns the
  first ancestor's `sidecars/` that exists, falling back to the
  conventional `<root>/sidecars`.
- `activate_sidecars(shard_dir)`: loads every `*.parquet` under that root
  into a module-level active set and returns their names; `None` clears
  it. A dozen entry points (training, the alerts harness, inference,
  interventions, case studies, baseline prep) each activate sidecars for
  the shard directory they're about to read, so a concept's definition
  doesn't depend on which entry point asked.
- `active_sidecar(name)` / `active_sidecar_names()`: what a rule reads.
- `sidecar_context(tables)`: a context manager that temporarily swaps in
  hand-built tables, for tests.

### The tables

- **`microbiology`** (`scripts/build_mimic_sidecars.py`): one row per
  culture specimen -- `subject_id`, `hadm_id` (nullable), `time`
  (`charttime`, falling back to `chartdate` at midnight when time is
  missing, as mimic-code does), `spec_type_desc`, `positive_culture`.
  MIMIC-IV's standard `meds-extract` spec carries no
  `microbiologyevents`, so this is a sidecar rather than a re-extraction.
- **`antibiotic_orders`** (same script): antibacterial prescription
  orders -- `subject_id`, `hadm_id`, `time`, `stoptime`, `drug`, `route`.
  Together with `microbiology`, these anchor mimic-code's
  `suspicion_of_infection` definition (see Sepsis-3 below).
- **`notes`** (`scripts/build_mimic_note_sidecar.py`): raw MIMIC-IV-Note
  text -- `note_id`, `subject_id`, `hadm_id`, `note_type`, `charttime`,
  `text`. Radiology reports carry in-visit `charttime` (usable as an
  alert-time signal); discharge summaries are stamped at discharge (only
  visible to the *next* visit or a readmission task). Can be restricted
  to a subject subset via `--shard-dir` (e.g. the probe's train/held-out
  scope) or built for every subject.
- **`note_embeddings`** (`odyssey/text/embed_notes.py`, reads `notes`):
  pooled frozen-encoder embeddings, one row per note -- mean of the last
  hidden layer over non-padding tokens, long notes split into windows and
  token-count-weighted. Optionally PCA-reduced (`--pca`) to a size small
  enough to hand the tuned GBM as extra alert features
  (`odyssey.text.note_features`), the headroom probe that decides whether
  text-modality fusion work is warranted. Needs the optional `text` extra
  (`uv sync --extra text`; see the README) -- the `transformers` import is
  deferred to `load_encoder` so nothing else in the package depends on it.

### What must be copied to a VM

`sidecar_root_for` only ever looks at ancestors of the shard directory a
run is pointed at, so a VM needs the extraction's `sidecars/` directory
copied alongside `data/` and `metadata/` -- not just the split(s) being
trained/evaluated on. Concretely: any run with `task_set` >= `"v2"`
(needs `microbiology` for Sepsis-3) or scoring the `strong_text` baseline
feature set (needs `note_embeddings`) will raise a clear error at
activation/scoring time if the sidecar is missing rather than silently
degrade -- so a missing copy fails fast, but only when that code path is
actually exercised.

## Task sets

Two versioned registries, both keyed by the same `task_set` string on
`TrainingConfig` and saved in `config.json` so evaluation rebuilds
exactly what a run trained:

- **Concepts** (`odyssey.data.concepts.TASK_SETS`): which canonical
  concept-bottleneck concepts a run supervises.
- **Alert events** (`odyssey.data.alert_events.ALERT_TASK_SETS`, read via
  `alert_events_for(task_set)`): which clinically meaningful events a
  run's event-hazard heads train on / `evaluate_alerts` scores.

### Versions

| task_set | concepts (`TASK_SETS`) | alert events (`ALERT_TASK_SETS`) |
|---|---|---|
| `v1` | 15: tachycardia, bradycardia, hypotension, hypertension, hypoxia, fever, hypothermia, elevated_lactate, sustained_tachypnea, acute_kidney_injury, aki_stage_2, aki_stage_3, sirs, qsofa, on_vasopressors | 4: vasopressor_start, icu_admission, acute_kidney_injury, death |
| `v2` | v1 + sepsis3 (16) | v1 + sepsis3, readmission_30d (6) |
| `v3` | v2 + 11 structurally-derived electrolyte/metabolic/hematologic concepts (hyperkalemia, hypokalemia, hyponatremia, hypernatremia, hypoglycemia, hyperglycemia, anemia, thrombocytopenia, coagulopathy, metabolic_acidosis, shock) -- 27 total | same as v2 (v3 widens concepts only; `alert_events_for("v3")` returns the v2 events) |

`v1` is the default (`DEFAULT_TASK_SET` / `TrainingConfig.task_set`
default) and is what every run before 2026-08-23 trained with; its
checkpoints hard-code that concept count, which is why `task_set` is
saved in `config.json` rather than inferred. `v2` needs the
`microbiology` sidecar next to the data (Sepsis-3's suspected-infection
anchor); training raises a clear error if it's missing
(`odyssey.training.train.activate_sidecars`'s guard).

### Backward-compat rule

`concepts_for_source`/`alert_events_for` both raise on an unknown
`task_set` name rather than guessing -- an old checkpoint's `config.json`
always names the exact task_set it was trained with (defaulting to
`"v1"` if the field predates this mechanism entirely), so evaluation
always rebuilds the same concept/alert-event list a run trained with,
never a newer one by accident.

## Sepsis-3 definition summary

`odyssey.data.concepts.Sepsis3Rule`, operationalizing Singer et al. 2016
the way mimic-code's `sepsis3` view does:

- **Suspected infection**: a culture specimen (`microbiology` sidecar)
  and a systemic antibiotic start within the standard window (antibiotic
  within `antibiotic_after_hours` (72h) after the culture, or culture
  within `culture_after_hours` (24h) after the antibiotic); suspicion
  time is the earlier of the two.
- **Sepsis**: SOFA total >= `sofa_threshold` (2, baseline assumed 0, as
  mimic-code does) at any instant from `sofa_before_hours` (48h) before to
  `sofa_after_hours` (24h) after suspicion.
- **Onset**: the first instant both hold. Diagnosis codes are
  deliberately not used (no onset time).

Validated against mimic-code's own reference `sepsis3` view (research
journal entry 43): post-fix agreement 89.3%, Cohen's kappa 0.79.

## Readmission index mode

`readmission_30d` (`task_set="v2"+`) is an `AlertEvent` with
`next_visit=True`: onset is the first `HOSPITAL_ADMISSION//` strictly
after the current visit's last event, with follow-up running to the end
of the subject's record (not the visit's) -- a 720h (30-day) horizon on
the hazard head. Because the outcome is about the *next* admission, it is
scored differently from the within-visit alerts: `evaluate_alerts
--index-mode visit_end` gives one index row per visit at discharge
(`INDEX_MODES = ("landmark", "visit_end")`; the discharge position is the
tokenizer's own `PatientSequence.visit_ends` marker) instead of the
default 4-hourly landmark grid. When `--alerts` is not given explicitly,
`evaluate_alerts` picks the task set's `next_visit`-matching events for
the requested `index_mode` automatically (`a.next_visit == (index_mode ==
"visit_end")`), so `--index-mode visit_end` without `--alerts` scores
exactly `readmission_30d` and nothing else.
