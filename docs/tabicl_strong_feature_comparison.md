# GBM (strong) vs TabICL (strong, reduced) on MIMIC

**Date:** 2026-08-25

## Why this run exists

Every prior TabICL row in this project's registry (`docs/experiments.md`) scored TabICL on the
**basic** 17-feature panel, while the GBM it was compared against used the **strong** 609-feature
panel. TabICL never saw the wide panel at all: a naive fit at 50,000 context rows, 609 features,
and the library's default of 8 estimators costs an estimated ~70 GB per `predict_proba` call, and
reliably OOM-killed the host (three confirmed kills on 2026-08-23).

This is the first TabICL run on the strong panel. To make it fit in memory, three settings were
changed from this project's usual TabICL defaults:

- `offload_mode="cpu"` (moves the library's large column-wise embedding tensor off GPU VRAM)
- `n_estimators` reduced from the library default of 8 to **1**
- the in-context row cap (`TABICL_MAX_ROWS`) cut from this project's usual 50,000 to **20,000**

Two earlier attempts at the full-default configuration failed before this one worked:

1. `offload_mode="disk"` with `batch_size=1` stalled in what looked like a disk I/O deadlock at
   the standard 50,000-row, 8-estimator scale. Never resolved, just abandoned.
2. `offload_mode="cpu"` with all 12 models fit up front and held in memory for the whole predict
   loop was kernel OOM-killed partway through (9 of 12 cells done, `shmem-rss` around 75 GB). This
   was a repeat of a memory-accumulation problem this project had already hit and fixed once
   before, not a new failure mode.

The run reported here fits, scores, and drops one model at a time, which is what let all 12 cells
finish cleanly.

## Setup

- Checkpoint: `subset_run_v8_taskset_v3` (MIMIC, task_set v3, `model_kind=cbm`)
- GPU host: `odyssey-cbm-a100`
- GBM (strong) scores: the existing registered alerts run (`alerts_rows_v3.parquet`), fit on 30
  train shards, scored on 4 held-out shards
- TabICL (strong, reduced) scores: fit on 8 of those same 30 train shards (not all 30; TabICL's
  in-context window caps out well below 30 shards' worth of rows regardless), scored on the same
  4 held-out shards
- Both models are scored on exactly the same rows and labels

## The caveat that matters most

Matching the feature set controls how much information each model gets per row. It does not
control the modeling approach itself. The GBM is gradient-boosted and hyperparameter-tuned per
task on the full training set (400 rounds, grid search over 4 configs). TabICL is zero-shot
in-context learning: no gradient descent on this data at all, just one forward pass conditioned
on a capped, subsampled 20,000-row context, with only 1 ensemble member instead of the library's
usual 8 (so the random feature/class permutation averaging TabICL normally relies on for
robustness is largely absent here).

If TabICL still loses after matching features, that is not evidence TabICL "can't use" these
features. It is evidence that zero-shot, single-member, context-capped in-context learning is not
competitive with a fully-supervised, tuned GBM on this task family. That is a narrower and
different claim.

## Can TabICL run at full capability instead of reduced?

We tried. Here is what happened.

The `offload_mode`/`disk_offload_dir` params in this project's code only cover one stage of
TabICL, the column-wise embedding step. That stage's disk offload works: a test with synthetic
data at full scale (50,000 rows, 609 features, 8 estimators, `disk_dtype` set to float16) wrote a
real file to disk, about 7.4 GB per estimator, and fit in 27 seconds with no memory error.

Fitting was never the problem. Scoring is. TabICL rereads its entire context from disk on every
`predict_proba` call, with no caching by default. Scoring 200 rows against the full context took
728.9 seconds (about 12 minutes). Scoring 8,192 rows (this project's normal batch size) did not
finish inside a 15-minute timeout.

At that rate, one MIMIC alert cell (about 48,000 held-out rows, 6+ batches) would take hours, and
the full 12-cell comparison would take a day or more of sequential compute. Not memory. Disk I/O
repeated on every call.

One thing we did not try: `kv_cache="repr"`, which caches embeddings across calls and might avoid
the repeated disk reads. Flagged as a real next step, not assumed to work.

Bottom line: full capability (8 estimators, 50,000-row context) fits and scores without crashing,
just not in a practical amount of time on this hardware. The reduced-config result below is what
was actually achievable, not a shortcut taken for convenience.

## Update, 2026-08-26: full capability tested on new hardware

The host was migrated to `a2-ultragpu-1g` (170 GB RAM, one A100-80GB, `odyssey-cbm-a100-ultra`,
`us-central1-a`), specifically to test whether more RAM, not disk, fixes this. It does, partly.

RAM-resident full capability (`offload_mode="cpu"`, `n_estimators=8`, 50,000-row context, no
disk) works. It does not hang and does not crash. Peak RSS during a single (event, horizon)
fit and predict reached about 98 GB, comfortably inside the new 165 GB free RAM, with zero swap
use. This confirms the diagnosis: disk I/O was the wrong lever, not memory capacity.

One full cell was run end to end: vasopressor_start@8h. Fit took 109 seconds for all 3 horizons
of this event together. Scoring the held-out rows for just the 8h horizon took 1,854 seconds
(31 minutes). Result:

| Event | Horizon | n | GBM (strong) | TabICL (strong, reduced) | TabICL (strong, full) |
|---|---|---|---|---|---|
| Vasopressor start | 8h | 111,450 | 0.934 [0.916, 0.950] | 0.859 [0.832, 0.887] | 0.915 [0.896, 0.936] |

Full capability closes most of the gap. The reduced config's 0.075 AUROC deficit against the
GBM drops to 0.019, and the confidence intervals now overlap (0.916-0.950 vs 0.896-0.936): at
full capability, this cell is statistically indistinguishable from the GBM, not a real loss.
The reduced config was a real handicap, not just a formality.

But 31 minutes for one horizon of one event is real, and it does not fit in one session for all
12 cells. At the measured rate (about 0.0166 seconds per held-out row), scoring all 12 core
(event, horizon) cells in the table below would take on the order of 5 hours of sequential
compute, not counting fit time (negligible by comparison). This was not run to completion. The
process was stopped deliberately after this one cell, and the VM was stopped, rather than run
the full sweep unattended on an hourly-billed A100-80GB host without asking first.

Where this leaves the table below: it is a real result, honestly obtained, and still the only
complete 12-cell comparison that exists. But the vasopressor@8h data point above shows it likely
understates TabICL's true performance across the board, not just for that one cell. Whether to
spend the roughly 5 hours of GPU time to get the real, full 12-cell table at full capability is
a decision for Amrit, not something to default into.

## Results

95% subject-clustered bootstrap confidence intervals, 1000 resamples
(`odyssey.inference.uncertainty.bootstrap_auroc`).

| Event | Horizon | n | GBM (strong) AUROC | TabICL (strong, reduced) AUROC | Gap | Verdict |
|---|---|---|---|---|---|---|
| Acute kidney injury | 8h | 95,471 | 0.894 [0.881, 0.906] | 0.723 [0.702, 0.745] | 0.171 | real gap |
| Acute kidney injury | 24h | 84,147 | 0.845 [0.827, 0.861] | 0.714 [0.690, 0.736] | 0.131 | real gap |
| Acute kidney injury | 72h | 57,442 | 0.782 [0.758, 0.805] | 0.674 [0.646, 0.702] | 0.108 | real gap |
| Death | 8h | 136,850 | 0.953 [0.934, 0.969] | 0.941 [0.921, 0.958] | 0.012 | within noise |
| Death | 24h | 135,061 | 0.959 [0.947, 0.969] | 0.924 [0.900, 0.945] | 0.035 | real gap |
| Death | 72h | 130,818 | 0.940 [0.923, 0.954] | 0.903 [0.879, 0.926] | 0.036 | within noise (barely) |
| ICU admission | 8h | 85,838 | 0.968 [0.962, 0.974] | 0.957 [0.950, 0.964] | 0.011 | within noise |
| ICU admission | 24h | 74,839 | 0.954 [0.945, 0.963] | 0.940 [0.929, 0.950] | 0.014 | within noise |
| ICU admission | 72h | 48,971 | 0.931 [0.912, 0.946] | 0.907 [0.886, 0.926] | 0.024 | within noise |
| Vasopressor start | 8h | 111,450 | 0.934 [0.916, 0.950] | 0.859 [0.832, 0.887] | 0.075 | real gap |
| Vasopressor start | 24h | 98,722 | 0.914 [0.895, 0.933] | 0.848 [0.820, 0.877] | 0.066 | real gap |
| Vasopressor start | 72h | 67,385 | 0.883 [0.850, 0.913] | 0.812 [0.778, 0.845] | 0.070 | real gap |

"Real gap" means the two confidence intervals do not overlap. "Within noise" means they do.

## What this means

Matching features does not close the gap. TabICL (strong, reduced) loses to the tuned GBM on all
12 cells, and the loss is statistically real on 7 of them.

AKI has the largest and most consistent gap (0.108 to 0.171, real on all three horizons). This
matches an earlier finding in this project (`docs/experiments.md`, journal entry 52): the GBM's
edge on AKI comes from window aggregates and trend statistics it computes explicitly from the raw
values. TabICL sees the same raw features but has no equivalent way to aggregate them across time
in a single zero-shot forward pass.

ICU admission is where TabICL comes closest to the GBM. That also matches the same earlier
finding: the GBM's edge on ICU admission concentrates in a small set of count features, which may
be easier for in-context learning to pick up directly from raw values than a window trend is.

The remaining gap is probably a mix of two things this run cannot separate: the real zero-shot vs.
tuned-and-supervised difference, and the memory-driven reductions (1 estimator instead of 8,
20,000-row context instead of 50,000). A full-ensemble, full-context run was not achievable on
this host in a reasonable time. Treat this result as a lower bound on TabICL (strong)'s real
ceiling, not as its true performance.

## Where things live

- Comparison script (not committed, diagnostic only): `scripts/tabicl_strong_compare.py`, run on
  the GPU host from a disposable git worktree (since removed)
- Code changes needed to make this run possible: `odyssey/inference/tabicl_baseline.py` (adds
  `offload_mode`, `batch_size`, and `disk_offload_dir` passthrough to `TabICLClassifier`)
- Raw per-cell results, including full bootstrap output (mean, std, resample counts):
  `compare_result.json`, pulled to `/tmp/compare_result.json` this session. Ask if you want it
  moved somewhere durable.
- GBM (strong) scores: `~/runs/subset_run_v8_taskset_v3/alerts_rows_v3.parquet` on
  `odyssey-cbm-a100` (already-registered run, unchanged by this work)
- Full-capability timing test (synthetic data, not committed, since removed from the host):
  `~/tabicl_disk_test/probe.py`, a standalone script isolating just the fit/predict cost at
  50,000 rows x 609 features x 8 estimators with disk offload, independent of this project's
  data-loading pipeline
- Full-capability real-data script (not committed, diagnostic only): `scripts/tabicl_strong_compare.py`
  on `odyssey-cbm-a100-ultra` (`us-central1-a`), fits/scores one (event, horizon) cell at a time using
  the existing `alerts_rows_v3.parquet` for GBM reference scores. The one completed cell's full
  bootstrap output is in `~/tabicl_full_validate.json` on that host.
