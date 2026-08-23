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
  **Re-running 2026-08-22**: a `uv sync` landed on VM2 since the above run --
  the strict same-env rule this canary exists to enforce means a dependency
  sync counts as a possible environment change same as anything else, so
  the pinned-commit re-cert is being redone post-sync before being trusted
  again, not assumed still valid from yesterday's run.
  **VM1 (MIMIC leg), SATISFIED 2026-08-22 by different means, documented so
  the asymmetry with VM2 isn't silent**: no separate pinned-commit historical
  reproduction was run. VM2's canary exists to size an *actual incident*
  (the mamba-ssm ABI break); VM1 has had no documented environment change
  this session -- no `uv sync`, nothing analogous. The M1 auto eval chain
  (`eval`/`interventions`/`alerts`/`cases`/`report`, all exit 0) completed
  cleanly on this exact environment minutes before the M2 v2 dump started,
  which is real, direct operational evidence the environment is healthy
  right now, not a proxy for it. `env_fingerprint.py`'s stored-checkpoint
  canary was checked and came back a silent no-op (`full_run_v8` predates
  the provenance-file feature, so `verify_run_provenance` has nothing
  stored to compare against) -- inconclusive, not a confirmed pass, noted
  for completeness rather than treated as the basis for this call.
- **MEDS-Tab v1 run landed** -- **scope amended 2026-08-22**: gates *only*
  the MEDS-Tab rescoring row (section 2's MEDS-Tab table entry and any
  MEDS-Tab-baseline comparison in section 3+); everything else in this wave
  gates on the reproducibility canary above alone. The eICU leg
  (re-certification + v2 dump regeneration + every non-MEDS-Tab rescore) is
  starting today on VM2 without waiting for MEDS-Tab v1 -- the shared-grid
  tabularization MEDS-Tab needs costs ~16h of CPU time that has no reason to
  idle the A100 while it waits. (6e's pipeline glue, `47c7925`, is in; the
  actual run + registry entry is not yet done as of this amendment.)
- `env_fingerprint.py`'s numeric canary for the checkpoint(s) being
  re-evaluated should show no mismatch (`verify_run_provenance`) --
  confirms the environment driving re-eval is the one the checkpoint was
  actually trained in, independent of the landmark fix.
- **Fetch-reset-before-launch, hard precondition (added 2026-08-22, after
  the second stale-checkout incident in one night -- VM2's E6 gate, then
  VM1's M2 leg)**: before launching ANY wave-relevant run on either VM,
  `git fetch origin main && git log HEAD..origin/main --oneline` and
  confirm the checkout is actually current, not just "on a branch called
  main". VM1's `main` sat 40 commits behind origin (predating the landmark
  fix itself, `85dde80`) for the entire M1/M2 leg without anyone noticing
  -- two "v2" dumps were quietly run under v1-protocol code as a result
  (see the stamp-check ritual below, which is how it was finally caught).
  A branch name is not a promise the code on it is current; check the
  commit, every time, before trusting a run.
- **Stamp-check ritual, hard precondition on every reported row count
  (added 2026-08-22)**: before a dump's row counts get reported to
  anyone, confirm `alerts_rows.parquet`'s `landmark_protocol_version`
  column is present and equals `2` (`pl.read_parquet(path)
  ["landmark_protocol_version"].unique()`) -- its *absence* was the tell
  in tonight's incident, noticed and then wrongly dismissed as
  not-crucial. State the stamp value in the report line itself (e.g. "1.12M
  rows, protocol v2 confirmed") so the check can't be silently skipped by
  whoever reads the report later, including the person who ran it.

## 2. Regeneration: v2-protocol `alerts_rows.parquet` dumps

One dump per (dataset, checkpoint) pair currently backing a published
comparison. Candidate list (confirm against the registry at execution
time, since new runs may land before this wave starts):

| Dataset | Checkpoint | VM | Notes |
|---|---|---|---|
| MIMIC | `full_run_v8` `checkpoint_best.pt` | cbm | flagship full-scale MIMIC run (epoch-2 continuation; new flagship, not directly comparable to the registry's v1 row -- different checkpoint, see note below) |
| eICU | `eicu_subset_v9` `checkpoint_best.pt` | eicu | flagship eICU run (recency features) |
| eICU | `eicu_subset_v8` `checkpoint_best.pt` | eicu | only if a published table still cites it directly |

**MIMIC checkpoint-provenance note (2026-08-22)**: the registry's `full_run_v8`
row (137,605 steps, 1 epoch, research_journal 20) was backed by that run's
own `checkpoint_best.pt`, step ~134,500 -- not `checkpoint_final.pt`. The
epoch-2 continuation overwrote `checkpoint_best.pt` (it now holds epoch-2's
own best) and no periodic checkpoint near step 134,500 survived the
continuation's rolling retention window. The only artifact preserved from
that era is `checkpoint_final_epoch1.pt` (that run's renamed
`checkpoint_final.pt`) -- a genuinely different checkpoint from the one
the registry numbers came from, not a substitute for it. Consequences:
(1) a same-model v1-vs-v2 **protocol delta** for MIMIC is still obtainable
by dumping both protocols against `checkpoint_final_epoch1.pt` -- valid on
any fixed model -- but its *absolute* alert numbers will not match the
registry row, expected from checkpoint provenance, not a sign of drift.
(2) MIMIC's extra-baseline v1 rows (TabICL etc., anything needing the
original `checkpoint_best.pt`-era row set) can never be produced now --
MIMIC extra-baseline comparisons are v2-only going forward. The registry's
existing v1 row keeps its historical numbers unchanged, with a provenance
note added ("backing checkpoint no longer on disk; see wave notes").

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
- **Incident, 2026-08-22 (lesson filed)**: the never-overwrite rule above
  was violated once, but not by wave code -- `full_run_v8`'s pre-wave-era
  automatic epoch-2-completion chain (`vm_epoch2_wait_and_eval.sh` ->
  `scripts/eval_run.sh`) ran its own `alerts` stage against the standard
  `alerts.json`/`alerts_rows.parquet` paths as part of ordinary post-training
  eval, unaware it was about to become the wave's own protected v1 baseline.
  Since `LANDMARK_PROTOCOL_VERSION` is a hardcoded constant on `main`
  rather than an opt-in flag, that ordinary chain silently wrote *v2*-
  protocol rows over what had been the only surviving v1 dump, for a
  different checkpoint besides (see the MIMIC checkpoint-provenance note
  in section 2) -- destroying MIMIC's ability to produce extra-baseline v1
  rows against the registry's original checkpoint. Eval tooling is growing
  an anti-clobber guard (refuse to write `alerts.json`/`alerts_rows.parquet`
  if a wave-in-progress marker or an existing dump under active comparison
  is present) so an ordinary eval run can't silently destroy a wave
  precondition again.
- **Precision note, 2026-08-23**: `time_stamps` is now stacked as
  `torch.double` rather than `torch.float` in both
  `odyssey/data/packed_context.py` and `odyssey/data/streaming.py`
  (root-caused chasing `verify_packed_landmark_rows`' real-data
  disagreement -- float32 loses enough precision at real-data time
  magnitudes, hundreds of hours since a subject's origin, to disagree
  with `_index_rows_from_events`' float64 ground truth at the 6th
  decimal place). This can shift a dump's `time_hours` values by up to
  ~1e-6 hours (a few milliseconds) relative to dumps produced before this
  fix landed -- below every decision threshold this wave cares about, but
  stamped here so a future exact-key join across pre-fix and post-fix
  dumps isn't surprised by a landmark that "moved." The model's own
  forward pass is unaffected (`TimeEmbeddingLayer.forward` already casts
  to float32 internally before use); this only touches the
  landmark-bookkeeping and `IndexRow.time_hours` path.
- **v2->v3, 2026-08-23**: `LANDMARK_PROTOCOL_VERSION` bumped to 3.
  `_landmark_mask` tracked only the immediately-preceding token position,
  so a patient's own token order interleaving two visits at one shared
  timestamp (e.g. a discharge instant stopping medication orders under
  both an ending and a starting admission id -- a real, observed pattern)
  re-triggered a landmark on every interleave step even though that
  visit's bucket had already been landmarked. v3 tracks the last-emitted
  bucket per visit directly (matching `_index_rows_from_events`' own
  per-(subject, visit, bucket) group-by semantics), so token order no
  longer matters. Confirmed at ~1.4% of rows on a real eICU repro;
  affected every backbone, not just `backbone="transformer"` --
  `verify_packed_landmark_rows` now runs unconditionally for both, not
  gated to transformer only.
- **Known residual (transformer/packed only), tracked as a P2 follow-up,
  not a blocker**: on real eICU-scale data, `backbone="transformer"` runs
  still show approximately 2 invented / 22 dropped-at-boundary
  `verify_packed_landmark_rows` warnings for TRUNCATED subjects
  specifically (non-truncated subjects are exact: 0 missing, 0 extra).
  Mechanism: `PackedContextSampler`'s truncation rebases a truncated
  subject's kept-window time_stamps to start at 0, then
  `_unrebase_truncated_times` restores them by adding the boundary back
  -- `(a - b) + b` is not bit-exact in float64, and the ~1e-13 residual
  is occasionally enough to flip a `floor()` right at a bucket boundary.
  Real fix (not implemented here -- touches five layers of plumbing:
  `PatientSequence` -> `_Row` -> `PackedContextSampler` ->
  `StreamingChunk` -> `collect_model_scores`): compute each event's true,
  never-rebased `time_hours` once at `build_patient_sequence` time and
  thread it through truncation unchanged as a landmark-bookkeeping-only
  field; the model's own input tensor keeps the rebased-for-locality
  convention untouched. **Operator note: this residual only ever affects
  `backbone="transformer"` (`PackedLaneSampler`/`backbone="hybrid"` never
  truncates, so it never triggers this path) -- treat ~2/~22-scale
  warnings on a transformer run as this known, already-diagnosed issue,
  but INVESTIGATE if the counts grow beyond that scale, since that would
  mean a different or additional bug.** Gates: this follow-up is a
  precondition for any paper-grade transformer-backbone dump; the tf1
  alerts rerun for the provisional control table may proceed under the
  known residual. No further protocol-version bump when the follow-up
  lands -- it is a fix toward the v3 spec, not a spec change.
