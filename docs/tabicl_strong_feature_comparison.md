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

## Can TabICL run at its actual full capability instead of reduced?

Amrit asked directly for this, since a reduced config (1 estimator instead of 8, a 20,000-row
context instead of this project's usual 50,000) is not TabICL at its documented best, and a
comparison against it is not the fairest one to draw conclusions from. This section reports what
was tried to get the real thing running, and why it still is not the number reported above.

**First, a real bug was found and worked around.** The `offload_mode`/`disk_offload_dir`
parameters used in this project's code only affect one stage of TabICL's pipeline, the
column-wise embedding step (`COL_CONFIG`), per the library's own docstring. That stage's disk
offload does work: a controlled test with synthetic data (50,000 rows, 609 features, 8
estimators, `disk_dtype` forced to float16 through the library's `inference_config` option)
produced a real memory-mapped file in the configured directory, sized close to the predicted
7.4 GB per estimator, and completed a fit in 27 seconds without touching more than that at once
(each estimator's file is written, used, and deleted before the next one starts). This is the
config that should be used if a full-capability fit is ever attempted again: `offload_mode="disk"`
plus `inference_config={"COL_CONFIG": {"disk_dtype": torch.float16}}`. It fits without any memory
error on this host, at either 82 GB free disk or the 82 GB of RAM this project already measured
as its ceiling.

**But it is not fast enough to use.** Fitting the full-capability context (27 seconds) was never
the problem. Scoring it is. A single `predict_proba` call for just 200 query rows against that
same 50,000-row, 8-estimator context took 728.9 seconds (about 12 minutes), because each
estimator's column-embedding file has to be written to disk and read back for every call, and
there is no caching between calls by default. A second test, scoring 8,192 query rows (this
project's actual per-batch size in `TabICLBaselineModel.predict_proba`) did not finish inside a
15-minute timeout at all, confirming the cost grows meaningfully with query size and is not just
a fixed per-call overhead.

Extrapolated conservatively: this project's smallest MIMIC alert cell has about 48,000 held-out
rows, meaning at least 6 batches of 8,192 rows each. If even one such batch takes 15 minutes or
more, a single (event, horizon) cell would take multiple hours, and the 12 cells in the table
below would take, at minimum, a day or more of sequential compute, not something achievable in a
single working session on this hardware.

**The actual constraint is not memory. It is that TabICL re-reads its entire context on every
single scoring call, and at 609 features that context is large enough that reading it from disk
repeatedly is prohibitively slow, even though it comfortably fits on disk.** Reducing context
rows and ensemble size (what the reported config below does) reduces the size of that context so
disk round-trips are fast enough to be practical. There is one untested option that might close
part of this gap without reducing capacity: TabICL's `kv_cache="repr"` mode, which caches part of
the column and row embeddings across calls at about 24x less memory than full key-value caching,
which could avoid re-reading the same context from disk on every batch. This was not tried here
and should not be assumed to work; it is a real next step if this is worth revisiting.

**Bottom line: full-capability TabICL (8 estimators, 50,000-row context) on the 609-feature
panel can be fit and scored without crashing, but not in practical time on this hardware with
the caching this library ships with by default.** The reduced-config result below is not a
stand-in chosen for convenience; it is what is actually achievable in a reasonable amount of time
right now, and should be read with that in mind, not as a claim that TabICL cannot do better in
principle.

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
