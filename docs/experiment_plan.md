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
| E4 | MEDS-Tab shared-grid xgboost stage (12 tasks, n_trials=200/seed 0) | Field-standard baseline row of the comparator table | ROOT-CAUSED Aug 23: MEDS-Tab's join_asof silently corrupts window boundaries when label_df is unsorted (polars precondition, no validation); build_shared_landmark_label_df lacked the sort + OR-aggregation export_task_labels got. Existing shared .npz matrices are wrong on disk. Path: fix builder (branch+review) -> re-tabularize ~15h against corrected grid (RSS-gated: launch only after >=1 v3 alerts leg frees its ~33GB) -> rerun all 3 gates vs sorted standalone reference -> sweep. Grid is model-free ground truth, v3-compatible by construction -- no separate v3 label export step; E4/E7 collapse into this path. Doctrine: ANY label_df handed to MEDS-Tab must be sorted; runtime guard added to the slicer path. |
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
| M5 | GPU coverage measure (test_hybrid_gpu with --cov, once) | Closes the CUDA-gated coverage blind spot | DONE Aug 23 (9c1c75a, VM1): 12/12 pass, hybrid.py 89% (151 stmts, 17 missed); misses are the incremental-decode step() branch, the varlen batch=1 conv path, and the mamba-ssm import guard -- none called anywhere in odyssey/ outside hybrid.py, so this is full coverage of the live paths |
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
2. Ladder rung specs (G4/G5) -- SPECIFIED (Aug 23, sized from train-full's
   measured behavior):

   **Evidence base:** gemini_full_14m_v1 runs ~1.8-1.9 steps/s deep into
   epoch 1 at 64x512 (feed-bound; smoke runs showed ~3GB VRAM / ~20% util
   at 14M, so GPU compute is nowhere near the H200's ceiling). Val task
   loss still improving well into epoch 2 (2.137 @ step 27.2k -> 1.555 @
   71.2k), i.e. 14M is not saturated on 894 shards -- the curve argument
   needs bigger rungs, and the data supports them.

   **G4 (rung 2, ~60M):** same config as gemini_full_14m_v1 EXCEPT the
   width/depth bump (target ~4x params; hidden 256->512 first, extra
   layers only if the printed param count lands far from ~60M -- the
   launch report must state the actual count printed at init). Same data,
   same 64x512 geometry, same 2-epoch budget, same seed, same
   eval-forecast protocol on the same held-out shards. Because the run is
   feed-bound at 14M, wall-clock should grow far less than 4x; the launch
   report must state measured steps/s in the first 2k steps -- if it
   collapses below ~1 step/s, stop and report (feed parallelization
   becomes the prerequisite, per the G4 gate).

   **G5 (rung 3, ~150M):** only after G4's eval-forecast shows a real
   gain over 14M (gate unchanged). Same recipe, next width/depth step
   (~hidden 768 or 512 + deeper); same everything else. If G4's gain is
   marginal, G5's budget goes to the GEMINI alert/concept subset run (G6)
   instead -- curve-with-plateau is still a publishable curve.

   **Both rungs:** checkpoint_every 2000 + prune to best/final (the
   MIMIC best-vs-final lesson); eval-forecast immediately after each run
   under the standard protocol; registry row per rung with the stamp
   ritual once alert-side evals exist.
3. Wave table assembly + protocol-delta write-up as legs complete.

## Active defects (affect gates above)

- **Interleaved-visit landmark duplication** (found Aug 23 by
  `verify_packed_landmark_rows` after the float64 fix c147b9f cleared the
  precision classes): same-timestamp event bundles spanning two hadm_ids with
  alternating token order make the adjacency-based `_landmark_mask` emit a
  landmark per visit-boundary *crossing* instead of per (subject, visit,
  bucket) -- ~0.35% phantom extras on the eICU repro, and the likely cause of
  the surviving truncated-subject classes. Fix owner: e6. Approved design:
  per-lane `{visit_id: last_bucket_emitted}` state cleared at subject
  boundaries, persisted across chunks; acceptance = all verifier classes zero
  on the real repro + literal interleaved-pattern regression test.
  **CLOSED (Aug 23): v3 merged as dbfd447** -- interleaved-visit fix +
  unconditional both-backbone verification + sequences.py time-derivation
  alignment. Known residual: ~2 invented / ~22 dropped verifier warnings on
  transformer/packed runs only (float64 (a-b)+b truncation-rebase
  round-trip, ~1e-13); true-time-threading follow-up tracked P2, gated
  before any paper-grade transformer dump; no version bump when it lands.
  Regen queue is ACTIVE: E1 (6e, after current legs finish), M2 (e6, under
  v3 from the start), E5 rerun + E4/E7 v3 labels after E1.
  Earlier note for the record: lane path exposed -- -- collect_model_scores builds rows
  through one unconditional code path for both backbones (e6, confirmed by
  reading), so every v2 dump carries duplicate landmarks (measured 1.4% on the transformer repro subset; 5.2% on the full eICU v9 64x512 dump -- population-dependent). **Ruling: the
  fix bumps LANDMARK_PROTOCOL_VERSION to 3** (same doctrine as v1->v2), and
  the lane path gains a production row-set assertion against the group-by
  ground truth. Regen queue once v3 merges: E1 dumps + rescores, E5 mechanical
  rerun, E4/E7 label export before the sweep (tabularization salvage
  unaffected), M2 born under v3 (never run as v2). v1 legs stay as-is: the
  bug is symmetric across v1/v2, so E2's protocol-delta attribution survives.
  Interim E5 rows are marked "v2, pre-interleaving-fix" in the registry.

## Master sequence (consolidated Aug 23; nothing ships half-baked)

Phase 1 -- **close the wave** (in flight, days): v3 dumps both datasets ->
E5/M2 rescores -> re-tab -> gates -> E4 sweep -> wave tables + protocol
delta write-up -> R3/R4 writable. Includes M3 (case studies), M4 (L-series
six-mode completion), M5 (GPU coverage), M6 (seed replicate) on VM1's tail.

Phase 2 -- **GEMINI depth** (overlaps 1, H200 + lead CPU): G4 60M rung ->
G5 gate decision; G6 concepts+alerts on the 18-table dataset (lead's
mapping is ready; wiring after export-codes lands) -> frozen cross-system
transfer + temporal-cutoff + 30-site hospital-holdout (R5, the title
gate). eICU subject-to-hospital sidecar built in parallel (CPU, feeds the
same holdout machinery).

Phase 3 -- **stress + tasks** (after wave tables close): missingness
protocol runs (designed, docs/missingness_protocol.md); task-suite
expansion decision executed (Sepsis-3 vs 30-day readmission -- pick on
clinical-impact grounds, spec then run); transformer-control v3 alerts
rerun (paper-grade gate: true-time-threading P2 fix first).

Phase 4 -- **interpretability push** (Track B, once modeling clears):
next lever design from the M-series frontier; distributional time head
probe (item 13); GEMINI concept-transfer readout (R7's strongest
potential claim, falls out of G6). Causal framing stays honest either way.

Phase 5 -- **paper assembly**: entry-19 related-work consolidation; v3
tables into R2-R7; title decision (waits on R5); TRIPOD+AI checklist;
Amrit rewrites for voice. Deployment/translator work (MEDS-to-FHIR) stays
explicitly post-paper.

## Standing gates recap

- Nothing launches without a row here or an explicit lead go.
- Contaminated v9 numbers stay quarantined until E2/E3 replace them.
- Framing-gate experiments (temporal cutoff, hospital-holdout, frozen
  cross-system transfer) get their own specs once G6 exists -- they are the
  paper's R5 and get designed deliberately, not squeezed in.
