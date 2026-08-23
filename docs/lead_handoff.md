# Lead-session handoff (written 2026-08-23, morning)

You are the lead session for odyssey. You plan, review, and merge; three
peer sessions execute and report to you. Your persistent memory directory
carries the standing rules (session-protocol, gpu-hosts, code-quality,
journal-style) -- read MEMORY.md first. This page is only the live state.

## The team (unchanged, all running)

- **odyssey-e6** -- VM1 (MIMIC, A100). Running the M2 rescores
  (TabICL/EBM/SurvivalPFN vs the flagship v3 dump; EBM refit ~4.6h).
- **odyssey-6e** -- VM2 (eICU, A100) + research journal. Running the
  ~15h MEDS-Tab re-tabularization (gates on completion), E5 legs pending
  a script update to main's new streaming loader.
- **odyssey-20** -- GEMINI mirror + CI + scripts/gemini + pyproject.
  Mirroring main to GEMINI; small run.sh follow-up (96G salloc hint).
- **Amrit** operates GEMINI by hand; send him exact commands.

Rules that bite if forgotten: every ruling goes by SendMessage to the
executing session (never only narrated to Amrit); terse orders, sessions
report back; never switch branches in the shared checkout (worktrees);
never `git add -A`; peers branch, you review+merge, then delete branches.

## State of play

- **Protocol v3 closed** (merge dbfd447): all four v3 dumps done clean
  (MIMIC -50.5% v1->v3 rows, eICU 629,068/dump, verifier silent,
  exact-match verify passed in production). Old dumps quarantined.
  Known residual: ~24-row transformer-only truncation artifact, tracked
  P2, gated before any paper-grade transformer dump.
- **MEDS-Tab**: join_asof silent-corruption root-caused (unsorted
  label_df); grid builder fixed + guarded (3b611c7); re-tab running on
  VM2 (~15h); then 3 gates vs the rebuilt one-shard reference; then the
  12-task sweep. E4/E7 are one path now.
- **OOM class root-fixed** (7a9b9ec): odyssey/inference/baseline_prep
  streams shards one at a time; 6e must port their fit_score scripts to
  it before relaunching E5 legs; e6 verifying a times-provenance detail
  post-run (raw vs prepared events -- expected identical on MIMIC).
- **GEMINI**: G3 done (set top-1 76.0, near-perfect time calibration,
  registry has it). Finalize of the 18-table extraction rerunning in a
  96G salloc, chained to export-codes; its end-of-chain git push may be
  rejected (mirror moved mid-flight) -- data is safe, re-push after
  fetch-reset. Next GPU job: `run.sh train-rung2` (G4, 60M, data pinned
  to the quarantined 12-table copy). After export-codes: lead's G6
  concept wiring (GEMINI_TO_LOINC exists in code_mapping.py; extend for
  new death/ICU/ER codes).

## What lands next, in order

1. Reports arriving: M2 rescore numbers (closes MIMIC wave table),
   re-tab + gate verdict (unlocks sweep), finalize/export-codes done.
2. Wave table assembly + protocol-delta write-up (yours; wave doc +
   registry have every number).
3. G4 launch (Amrit, one command) -> eval-forecast -> ladder decision.
4. Then the master sequence in docs/experiment_plan.md, phases 2-5
   (GEMINI depth/R5, missingness runs, task expansion, Track B designs
   N1/N2 in docs/track_b_designs.md, paper assembly).

## Key files

docs/experiment_plan.md (queues, gates, master sequence, defect log) ·
docs/experiments.md (registry) · docs/reeval_wave_v2.md ·
docs/missingness_protocol.md · docs/track_b_designs.md ·
paper/outline.md + paper/writing_guide.md · research_journal/ (6e owns;
entries 38-41 are the verification arc).
