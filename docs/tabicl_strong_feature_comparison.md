# GBM (strong) vs TabICL (strong, full capability) on MIMIC

**Date:** 2026-08-27

## Why this run exists

Every prior TabICL row in this project's registry (`docs/experiments.md`) scored TabICL on the
**basic** 17-feature panel, while the GBM it was compared against used the **strong** 609-feature
panel. TabICL never saw the wide panel at all: a naive fit at 50,000 context rows, 609 features,
and the library's default of 8 estimators costs an estimated ~70 GB per `predict_proba` call, and
reliably OOM-killed the host (three confirmed kills on 2026-08-23).

This report is the first TabICL run on the strong panel at its actual full capability
(`n_estimators=8`, 50,000-row context, no reduction). Getting there took three attempts:

1. A reduced config (`n_estimators=1`, 20,000-row context) fit and scored fine on the original
   82 GB-RAM host, but it isn't TabICL at its documented best, and Amrit asked for the real thing.
2. Disk offload (`offload_mode="disk"`) was tried next, to fit the full config in less RAM. It
   works for fitting, but TabICL rereads its entire context from disk on every `predict_proba`
   call with no caching. Scoring 200 rows took 12 minutes; scoring a normal batch (8,192 rows)
   did not finish in 15 minutes. Extrapolated, the full comparison would take a day or more. Not
   a memory problem, a disk I/O throughput problem.
3. The fix was hardware, not software: the host was migrated to `a2-ultragpu-1g` (170 GB RAM, one
   A100-80GB, `odyssey-cbm-a100-ultra`, `us-central1-a`). RAM-resident full capability
   (`offload_mode="cpu"`, no disk) fits comfortably (peak RSS ~102 GB) and scores at normal
   compute speed, just slower than the reduced config because there is more to compute: roughly
   1,800 seconds (30 minutes) to score one horizon, versus ~120 seconds for the reduced config.
   All 12 cells took about 6 hours of sequential GPU compute.

## Setup

- Checkpoint: `subset_run_v8_taskset_v3` (MIMIC, task_set v3, `model_kind=cbm`)
- GPU host: `odyssey-cbm-a100-ultra` (`a2-ultragpu-1g`, 170 GB RAM, one A100-80GB)
- GBM (strong) scores: the existing registered alerts run (`alerts_rows_v3.parquet`), fit on 30
  train shards, scored on 4 held-out shards
- TabICL (strong, full): `n_estimators=8`, 50,000-row context (this project's real defaults, not
  reduced), fit on 8 of the 30 train shards, `offload_mode="cpu"`, scored on the same 4 held-out
  shards. One model fit, scored, and dropped at a time; no more than one model resident at once.
- Both models are scored on exactly the same rows and labels
- Script: `scripts/tabicl_strong_compare.py` (committed, reusable; reuses the existing GBM scores
  instead of refitting)

**Label caveat, found after this sweep completed, not yet corrected:** the acute kidney injury
label used here, for both the GBM and TabICL scores, comes from `alerts_rows_v3.parquet`,
generated 2026-08-25 05:07 UTC. The KDIGO AKI staging completeness fix (commit `3d7ecbb`, adding
the renal-replacement-therapy and urine-output legs that were previously missing, creatinine-only
staging before that) landed later the same day, at 14:04 UTC. The model checkpoint itself was
also trained before the fix (03:35 UTC). So every AKI number in this report, GBM and TabICL both,
uses the old, incomplete label, which under-counts true AKI-3 cases that only qualify through RRT
or oliguria, not creatinine. This does not explain away the AKI gap below, since the same label
is used for both models, but it means the true AKI numbers (positive or negative for either model)
are not yet known. Worth a rerun with a fresh alerts dump once that's a priority.

## The caveat that matters most

Matching the feature set controls how much information each model gets per row. It does not
control the modeling approach itself. The GBM is gradient-boosted and hyperparameter-tuned per
task on the full training set (400 rounds, grid search over 4 configs). TabICL is zero-shot
in-context learning: no gradient descent on this data at all, just one forward pass conditioned
on a subsampled context. At full capability, that context is closer to the library's own
validated regime (its authors report strong results up to 50,000 rows), so this asymmetry is
smaller here than it was in the reduced-config run, but it has not gone away.

## Results

95% subject-clustered bootstrap confidence intervals, 1000 resamples
(`odyssey.inference.uncertainty.bootstrap_auroc`).

| Event | Horizon | n | GBM (strong) AUROC | TabICL (strong, full) AUROC | Gap | Verdict |
|---|---|---|---|---|---|---|
| Acute kidney injury | 8h | 95,471 | 0.894 [0.881, 0.906] | 0.766 [0.746, 0.786] | 0.128 | real gap |
| Acute kidney injury | 24h | 84,147 | 0.845 [0.827, 0.861] | 0.739 [0.714, 0.761] | 0.106 | real gap |
| Acute kidney injury | 72h | 57,442 | 0.782 [0.758, 0.805] | 0.685 [0.657, 0.714] | 0.097 | real gap |
| Death | 8h | 136,850 | 0.953 [0.934, 0.969] | 0.961 [0.941, 0.975] | -0.007 | within noise |
| Death | 24h | 135,061 | 0.959 [0.947, 0.969] | 0.940 [0.917, 0.958] | 0.020 | within noise |
| Death | 72h | 130,818 | 0.940 [0.923, 0.954] | 0.925 [0.905, 0.944] | 0.015 | within noise |
| ICU admission | 8h | 85,838 | 0.968 [0.962, 0.974] | 0.963 [0.956, 0.969] | 0.005 | within noise |
| ICU admission | 24h | 74,839 | 0.954 [0.945, 0.963] | 0.940 [0.929, 0.951] | 0.014 | within noise |
| ICU admission | 72h | 48,971 | 0.931 [0.912, 0.946] | 0.914 [0.894, 0.933] | 0.017 | within noise |
| Vasopressor start | 8h | 111,450 | 0.934 [0.916, 0.950] | 0.916 [0.896, 0.936] | 0.018 | within noise |
| Vasopressor start | 24h | 98,722 | 0.914 [0.895, 0.933] | 0.893 [0.871, 0.915] | 0.021 | within noise |
| Vasopressor start | 72h | 67,385 | 0.883 [0.850, 0.913] | 0.868 [0.840, 0.893] | 0.014 | within noise |

"Real gap" means the two confidence intervals do not overlap. "Within noise" means they do.

## What this means

Full capability changes the answer. At the reduced config, TabICL lost on all 12 cells, 7 of them
a real (CI-separated) loss. At full capability, TabICL is statistically indistinguishable from
the tuned GBM on 9 of 12 cells: all of death, all of ICU admission, all of vasopressor start.
Death@8h even slightly favors TabICL, though within noise.

The reduced config was a real handicap, not a formality. Cutting the ensemble from 8 members to 1
and the context from 50,000 rows to 20,000 cost TabICL real, measurable performance across the
board, not just on the cells that happened to look weak.

Acute kidney injury is the one place a real gap survives at full capability (0.097 to 0.128,
separated on all three horizons). This matches an earlier finding in this project
(`docs/experiments.md`, journal entry 52): the GBM's edge on AKI comes from window aggregates and
trend statistics it computes explicitly from the raw values, not from having more context to work
with. TabICL sees the same raw features but has no equivalent way to aggregate them across time
in a single forward pass, at any context size. But see the label caveat above: this comparison
used an AKI label known to be incomplete, so the true size of this gap (bigger, smaller, or
possibly gone) is not yet established.

## Where things live

- Comparison script (committed): `scripts/tabicl_strong_compare.py`
- Code changes that made this possible: `odyssey/inference/tabicl_baseline.py` (adds
  `offload_mode`, `batch_size`, and `disk_offload_dir` passthrough to `TabICLClassifier`)
- Raw per-cell results, including full bootstrap output (mean, std, resample counts): pulled to
  `/tmp/tabicl_full_sweep.json` this session. Ask if you want it moved somewhere durable.
- GBM (strong) scores: `~/runs/subset_run_v8_taskset_v3/alerts_rows_v3.parquet`, generated
  2026-08-25 05:07 UTC (see the label caveat above)
- GPU host used for this run: `odyssey-cbm-a100-ultra` (`a2-ultragpu-1g`, `us-central1-a`),
  stopped after this sweep completed. The original host, `odyssey-cbm-a100` (`a2-highgpu-1g`,
  `us-central1-f`), is stopped and untouched, kept as a fallback.
