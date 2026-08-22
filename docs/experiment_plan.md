# Experiment plan (living document)

Owner: lead session. Updated whenever the queue or a gate changes; sessions
execute against this, the registry records what actually happened. Every run
listed here states the question it answers -- compute with no question
attached does not launch.

## Principles

1. **No new v1-protocol numbers.** Everything scientific lands under
   `LANDMARK_PROTOCOL_VERSION=2`. The only sanctioned v1 run remaining is the
   v9 clean-v1 pass (its question is the protocol delta + contamination
   sizing, which requires v1).
2. **One GPU job per card; RSS arithmetic before concurrent CPU jobs** (the
   Aug 22 OOM lesson: alerts-class jobs ~33GB, tabularize ~17GB, on 83GB
   boxes).
3. **The critical path is the lead's GEMINI code/concept/event mapping** --
   it blocks every alert-side and concept-side GEMINI result (paper R5). It
   is CPU/laptop work and runs in parallel with everything below.
4. **H200 and A100s never wait on each other.** GEMINI runs the
   forecasting-side ladder (needs no concepts) while the A100s run the wave.
5. **Standardized eval:** every ladder rung and checkpoint gets the same
   `eval-forecast` protocol (same held-out shards, same lanes) so points are
   comparable; alert evals follow wave conventions exactly.
6. **Decision gates before expensive rungs** (listed per item below).

## H200 / GEMINI queue (forecasting side; operator-mediated)

| # | Run | Question | Est. | Gate |
|---|-----|----------|------|------|
| G1 | `eval-forecast gemini_smoke_2` | First GEMINI held-out numbers; anchors the data-scaling curve | ~30m | mirror landed |
| G2 | `train-full` (14M, 64x512, 2 epochs) | First real GEMINI checkpoint; 3rd data-scaling point; measures whether CPU feed holds at 16x smoke throughput | ~2-3h | none |
| G3 | `eval-forecast gemini_full_14m_v1` | Held-out at full data; data-scaling figure point | ~30m | G2 done |
| G4 | Ladder rung 2 (~60M: hidden 512, spec from lead) | Model-scale vs data-scale disentangled (Track A item 9) | ~4-8h | G2 throughput report; feed parallelization first if steps/s collapsed |
| G5 | Ladder rung 3 (~150M) | Upper rung of the curve | ~8-16h | G4 shows a real gain over 14M |
| G6 | GEMINI alert/concept subset run | First GEMINI concepts + alert heads | -- | lead's code_mapping lands; spec to follow |

## VM2 / eICU queue (wave leg; serialized after the OOM lesson)

| # | Run | Question | Gate |
|---|-----|----------|------|
| E1 | v8 v2 dump regen (running) | v8's corrected row set + clean hazard/GBM v2 scores | -- |
| E2 | v9 clean-v1 alerts (pinned worktree 85dde80^) | **Was the recency effect ever real?** + sizes ABI contamination + v1-vs-v2 protocol delta | E1 done |
| E3 | v9 inference/interventions/case regen (current main) | Clean replacements for every contaminated v9 artifact | E2 done |
| E4 | MEDS-Tab shared-grid xgboost stage (12 tasks, n_trials=200/seed 0) | Field-standard baseline row of the v2 comparator table | tabularize done (overnight) + task-1 exact-reproduction trust gate |
| E5 | v2 rescores: TabICL / EBM / SurvivalPFN against E1+v9-v2 rows | The eICU v2 comparator table | E1 done (CPU, can interleave when GPU busy) |
| E6 | Transformer-control subset run (eICU 30 shards, matched budget) | Backbone priced (Track A item 5); born under v2 | GPU free after E3 |
| E7 | MEDS-Tab v2 label export + rerun | MEDS-Tab's v2 row | E4 done + v2 rows exist |

## VM1 / MIMIC queue (post-epoch-2, in order, on lead's go)

| # | Run | Question | Gate |
|---|-----|----------|------|
| M1 | Epoch-2 final eval chain (auto) | Does epoch 2 hold at final eval; checkpoint-selection read | running |
| M2 | Wave MIMIC leg: full_run_v8 v2 dump + rescores | MIMIC v2 comparator table; measures MIMIC's own v1 inflation (do NOT assume eICU's ~19-23%) | M1 done |
| M3 | v9-MIMIC + full-run case-study regen (fixed trace code) | Restores qualitative traces for reports | M2 done (short) |
| M4 | L2-L4 six-mode intervention reruns | Completes the lever figure across the L-series (Track B opener) | M3 done (short) |
| M5 | GPU coverage measure (test_hybrid_gpu with --cov, once) | Closes the CUDA-gated coverage blind spot | any idle moment |
| M6 | v9-MIMIC seed replicate | Recency non-replication: real or convergence variance? (also waits on E2/E3 -- the eICU side of that comparison must be clean first) | last |

## Eval-only / CPU work (post-wave)

- **Missingness stress protocol** (Track A item 6): all families on identical
  degraded records, v2 dumps. Design doc first; runs mostly CPU. Gate: wave
  tables closed.
- **eICU subject-to-hospital sidecar** (Track C item 17): built locally from
  raw tables on /Volumes/clinical-data; site-holdout eval machinery
  developed on eICU, then inherited by GEMINI. Gate: none (parallel).
- **Registry/README/paper updates**: v2 tables + cleaned recency claims land
  together when the wave closes; paper R2/R3/R4 sections become writable.

## Lead's own critical-path work (parallel to all of the above)

1. GEMINI code_mapping table (OMOP -> LOINC + unit tags incl. SI variants)
   + per-unit clinical ranges (the umol/L creatinine trap) + gemini alert
   event definitions -> unlocks G6 and paper R5.
2. Ladder rung specs (G4/G5) sized from G2's measured throughput.
3. Wave table assembly + protocol-delta write-up as legs complete.

## Standing gates recap

- Nothing launches without a row here or an explicit lead go.
- Contaminated v9 numbers stay quarantined until E2/E3 replace them.
- Framing-gate experiments (temporal cutoff, hospital-holdout, frozen
  cross-system transfer) get their own specs once G6 exists -- they are the
  paper's R5 and get designed deliberately, not squeezed in.
