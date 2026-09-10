# Cohort-sensitivity producers

These two scripts write the `cohort_check.json` that
`scripts/make_cohort_table.py` renders as the paper's cohort-sensitivity
table. Their bodies are committed **verbatim as they ran**, not tidied (only a
`# ruff: noqa` header was added): their
marker regexes are the definition of the published groups, and rewriting
them would silently redefine what the table measures.

They lived only on the VMs until 2026-09-10, which is how the paper came
to ship a decomposed-arm table in a mixture-only manuscript -- there was
no producer in the repo to re-run against the right checkpoints, so the
mismatch could only be relabelled, not fixed. Keep them here.

## Running

Each takes the held-out MEDS shard directory, the run's own
`alerts_rows.parquet`, and an output path:

    python scripts/cohort/cohort_check_mimic.py \
        ~/data/mimiciv_3.1_v1/data/held_out \
        ~/runs/full_run_v10/alerts_rows.parquet \
        ~/runs/full_run_v10/cohort_check.json

    python scripts/cohort/cohort_check_eicu.py \
        ~/data/eicu_2.0_v1/data/held_out \
        ~/runs/eicu_full_v10/alerts_rows.parquet \
        ~/runs/eicu_full_v10/cohort_check.json

No GPU and no re-scoring: everything comes from the banked row dump plus
a marker scan of the shards. Minutes, not hours.

## The gate

`auroc_24h_all` in the output must reproduce the run's own comparator
table (`tab:mimic`, `tab:eicu`) on every event. If it does not, the row
dump and the table describe different rows and nothing else in the file
should be believed. On 2026-09-10 all nine cells matched to three
decimals.

## Markers

Source-specific by necessity, which is why there are two scripts.
MIMIC-IV reads ICD diagnosis and procedure codes plus hospice discharge;
eICU-CRD has no ICD-coded palliative entries, so its palliative group is
care-plan comfort measures and palliative consults. The eICU script's
module docstring still says "MIMIC" -- it was copied from its sibling.
Left as-is rather than corrected, so the file matches what produced the
published numbers line for line below the header.
