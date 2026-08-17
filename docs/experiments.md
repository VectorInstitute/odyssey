# Experiment registry

One row per training run. Kept in git because it holds no patient data,
only configs, commits and aggregate numbers. The lead session maintains it
from the operating sessions' reports; report artifacts (JSON, HTML) live in
the gitignored `research_journal/` and on the VMs.

Recipe shorthand: **v6 recipe** = bundle-invariant family-restricted loss,
family balancing alpha 0.5, visit-scoped concepts, icd3 backoff, medication
normalization, time-to-next-event head, per-event hazard heads, randint 0,
64 lanes x 512, 2 epochs, checkpoint_every 2000.

Data versions: **MIMIC** = `mimiciv_3.1_v1` (292/?/37 shards). **eICU v1** =
`eicu_2.0_v1` (spec v1: 36% of medication rows `UNK`, infusions a bare
`INFUSION_DRUG` token; 134/17/17 shards). **eICU v2** = `eicu_2.0_v2` (spec
v2 from bc7ac4c + 60310b0: HICL segment, named infusions, nurseCharting GCS,
intakeOutput urine output; extraction pending the raw tables).

| Run | VM | Data | Commit | Purpose | Status | Key results |
|---|---|---|---|---|---|---|
| subset_run_v3 | cbm | MIMIC 30 shards | 4a1c0a8 | visit-scoped concepts, pos_weight, on_vasopressors | done | exact top-1 60.7; same-family set 69.9 (med 35.9, diag 31.0, lab 75.2); concepts 0.62-0.99; lever inert |
| subset_run_v4 | cbm | MIMIC 30 shards | 6ac235a+ | RandInt 0.25 A/B vs v3 | done | exact 51.9 (-8.8); set 65.3; lever correctly signed but +0.1 pt only; retired |
| subset_run_v5 | cbm | MIMIC 30 shards | 68ba07c | bundle objective + time head + hazard heads, randint 0 | done | set 74.9 (lab 80.6, med 41.9, proc 37.5, diag 23.4 [loss bug], billing 20.8); time calibrated; alerts hazard 8/12 vs GBM (mean 0.831 vs 0.798) |
| subset_baseline_v5 | cbm | MIMIC 30 shards | 68ba07c | no-bottleneck baseline for v5 | done | set 76.6 (-1.7 vs v5; diag +4.9 for v5); time head equal; alerts hazard 7/12 |
| subset_run_v6 | cbm | MIMIC 30 shards | a883441 | v5 + family-restricted bundle loss (diagnosis fix) | training | pending |
| subset_baseline_v6 | cbm | MIMIC 30 shards | >= a883441 | baseline on the fixed loss | queued | pending |
| eicu_subset_v6 | eicu | eICU v1 30 shards | 4d8a5b2 | pipeline replication on eICU with the v6 recipe | done | 10,289 steps, 1.02h; set 85.3 (lab 86.8, diag 58.4, med 45.8, proc 46.2, visit 28.6, other 54.2); concepts 0.64-0.96; time calibration within 0.2 pt; alerts hazard vs GBM: vasopressor 8h 0.856/0.838, 24h 0.805/0.800, 72h 0.737/0.763; AKI 0.70/0.74, 0.69/0.71, 0.65/0.68; death 0.878/0.837, 0.825/0.836, 0.756/0.775; ICU admission n/a on eICU (unit stay = visit) |
