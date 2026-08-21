# Re-evaluation wave: landmark protocol v2

**Status: APPROVED, execution still gated on odyssey-db's explicit go.**
Covers the coordinated
re-evaluation of the alerts harness under `LANDMARK_PROTOCOL_VERSION=2`
(`odyssey/inference/alerts.py`, commit `85dde80`), which fixed a real bug:
`_landmark_mask` carried no state across streaming-chunk boundaries, so any
patient whose sequence spanned more than one chunk got a spurious extra
landmark row at the boundary. Confirmed ~23% row over-count at eICU scale
(6e's repro); already-visible fingerprint in a stored dump: `eicu_survivalpfn`'s
registry entry notes 9,108 duplicate landmark keys (up to 5x multiplicity)
in `alerts_rows.parquet` -- almost certainly this bug's real-world footprint,
not independent noise. v1 dumps/results remain internally consistent and
valid comparisons among themselves; they are just not comparable row-for-row
against a fresh v2 run. Nothing about this wave touches training or model
weights -- it only re-runs `evaluate_alerts` and rescales/rescore baselines
against the corrected row set.

## 1. Preconditions (all must hold before step 2 starts)

- **A100 back on VM2**, mamba-ssm rebuilt against it (`docs/gemini.md`'s
  documented recipe), confirmed importable
  (`from mamba_ssm.modules.mamba2 import Mamba2`).
- **Reproducibility canary (pinned-commit method)**: `git checkout 85dde80^`
  (the parent of the landmark fix, i.e. the last commit still on the old
  per-chunk-reset behavior), re-run `evaluate_alerts` against
  `eicu_subset_v8`'s `checkpoint_best.pt`, and confirm `death@8h` hazard
  AUROC reproduces at **exactly 0.908** -- not a tolerance band. Deterministic
  eval means the same code, checkpoint, data, and environment must reproduce
  the same number bit-for-bit-equivalent-in-effect; this value has held flat
  across `eicu_subset_v7`, `v8`, and `v9` in the registry (three different
  recipes), so any deviation here means the environment itself has drifted,
  independent of the landmark fix, and the wave should not proceed until
  that's resolved separately. `git checkout main` immediately after the run
  regardless of outcome. If 6e's same-method run (already executing as of
  this draft) lands 0.908 first, this precondition is satisfied by that run
  -- mark it satisfied here with the date rather than repeating it.
  **SATISFIED 2026-08-21**: 6e's pinned-commit run at `85dde80^` (`47c7925`)
  on `eicu_subset_v8`'s `checkpoint_best.pt` reproduced `death@8h = 0.9078726583`
  (0.908 exact) plus to-the-digit matches on `vasopressor_start@8h` 0.862,
  `icu_admission@8h` 0.802, `AKI@8h` 0.681, on the rebuilt mamba-ssm +
  restored transformers env. No drift.
- **MEDS-Tab v1 run landed** (6e's pipeline glue, `47c7925`, is in; the
  actual run + registry entry is not yet done as of this draft) -- needed
  so the v2 wave scores against a MEDS-Tab baseline that already exists,
  rather than blocking this wave on a second in-flight piece of work.
- `env_fingerprint.py`'s numeric canary for the checkpoint(s) being
  re-evaluated should show no mismatch (`verify_run_provenance`) --
  confirms the environment driving re-eval is the one the checkpoint was
  actually trained in, independent of the landmark fix.

## 2. Regeneration: v2-protocol `alerts_rows.parquet` dumps

One dump per (dataset, checkpoint) pair currently backing a published
comparison. Candidate list (confirm against the registry at execution
time, since new runs may land before this wave starts):

| Dataset | Checkpoint | VM | Notes |
|---|---|---|---|
| MIMIC | `full_run_v8` `checkpoint_best.pt` | cbm | flagship full-scale MIMIC run |
| eICU | `eicu_subset_v9` `checkpoint_best.pt` | eicu | flagship eICU run (recency features) |
| eICU | `eicu_subset_v8` `checkpoint_best.pt` | eicu | only if a published table still cites it directly |

For each: run `scripts/eval_run.sh` (or `evaluate_alerts` directly) with
`--dump-rows` pointed at a **new path** (e.g. `alerts_rows_v2.parquet`,
never overwriting the existing `alerts_rows.parquet`), same
`held_out_shard_dir`/`max_shards`/`landmark_hours`/`num_lanes`/
`chunk_size` as the original run so the only thing that changes is the
landmark-selection code.

**Sanity check per dump**, before it's trusted for anything downstream:
compare v2 row counts against the existing v1 dump, per alert event.
Expected direction: v2 count `<=` v1 count always (v2 only *removes*
spurious extras, never adds rows) -- eICU's confirmed real-world figure is
~23% fewer rows; MIMIC's delta is currently unmeasured and may differ
(different chunk_size/sequence-length distribution) -- **do not assume the
eICU percentage transfers; measure it.** A dump where v2 count equals v1
count exactly is not necessarily wrong (a run with `chunk_size` larger than
every patient's sequence would never have hit the bug), but is worth a
second look given none of these runs used an unusually large chunk_size.
`load_index_row_table()` logs the protocol version on every read, so this
check is one line against both dumps' `.height`.

## 3. Rescoring matrix

| Baseline | Train rows source | Affected? | Action |
|---|---|---|---|
| GBM | `_index_rows_from_events` (no chunking) | No | **Rescore-only.** Trains exclusively via `_index_rows_from_events`, which has no chunking and was never touched by the landmark bug -- only the held-out scoring rows change under v2, so rescoring against the v2 dump is sufficient; no refit. |
| EBM | `_index_rows_from_events` (no chunking) | No | **Rescore-only**, same reasoning as GBM. |
| TabICL | zero-shot, no training | No | Rescore-only against v2 held-out rows/features. |
| SurvivalPFN | fit per-event on `(T, delta)` from the scored row set | Yes | Refit on v2 rows (matches your plan). |
| MEDS-Tab | own pipeline, label exports off the row set | Yes | Definitively affected -- 6e confirmed the label export is built from `alerts_rows.parquet`'s own rows. Full pipeline rerun on v2 label exports required. |
| Hazard/concept/next-token (model-native) | `collect_model_scores` directly | Yes -- this is the fix itself | Automatic once dumps regenerate (step 2). |

## 4. Registry plan

- v2 rows are **added alongside** existing v1 rows in `docs/experiments.md`,
  never overwriting them -- run names stay unchanged (no `_v2eval` or
  similar suffixes); a v2 result is a new row for the same run name, not a
  renamed run.
- Every registry row touched or added by this wave states its landmark
  protocol version explicitly via a `[protocol v1]` / `[protocol v2]`
  prefix in the `Key results` cell -- no schema change needed for a
  markdown table.
- One-line migration note at the top of `docs/experiments.md`, near the
  existing "Data versions" paragraph: what `LANDMARK_PROTOCOL_VERSION`
  means, and that rows predating a given date are v1 unless marked
  otherwise.

## 5. Failure / rollback

- Every v2 artifact (dumps, JSON, registry rows) is written to a **new**
  path or a **new** registry row -- v1 `alerts_rows.parquet`/`alerts.json`
  files are never opened for writing during this wave, only read (for the
  step-2 sanity comparison).
- The wave is a sequence of independent (dataset, checkpoint) regenerations
  plus independent baseline rescoring runs -- abortable after any single
  one completes, with no cross-step state that needs unwinding. A failed
  or paused step leaves only its own new-path artifacts (or none) behind;
  nothing already-published is ever in a partially-updated state.
- If the reproducibility canary (section 1) fails at any point *during*
  the wave (not just before it), stop immediately -- an environment drift
  discovered mid-wave invalidates every v2 number produced so far, not
  just the one run being evaluated when it was caught.
