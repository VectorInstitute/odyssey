# Experiment plan: the ML4H paper (nothing else)

Owner: lead session. Rewritten 2026-08-30 per Amrit: paper experiments
only, complete vs remaining. Registry (docs/experiments.md) records what
actually happened; this file is the queue.

Paper contract (Amrit, 2026-08-30): results on MIMIC-IV, eICU, and
GEMINI; full scale only, no subset numbers anywhere; core is the CBM
trust audit (readout / completeness / lever / audit-of-the-eval) plus a
strong-model comparison against many baselines. Everything under
landmark protocol v4, task_set v3, bin-edge v2, fixed labels,
reset_prob 0.0 (R1 alone pins 0.1 as the cross-generation bridge).
Within-run claims: paired subject-clustered bootstraps. Cross-run
claims: seed replicates or they are labeled hypotheses.

## COMPLETE (what the paper can already stand on)

| Item | Paper use |
|------|-----------|
| Label + protocol + tokenization fixes, audited (sepsis3 evidence anchor, shock pooling, composite masks, GCS stamp, KDIGO completion 3d7ecbb, landmark v4, bin-edge v2 stamped on binner, note storetime, subject-disjointness deep scan) | Sec 5.5 audit findings; the reason for the retrains |
| Statistics tooling: fast subject-clustered AUROC bootstrap, generic AUPRC path, paired deltas, `scripts/alerts_cis.py`, `scripts/intervention_cis.py`, `--dump-per-subject` | every CI and bold in the tables |
| Prior-generation full-scale runs harvested (e20 MIMIC joint, e21 eICU joint) incl. the per-mode top-1 + loss decomposition and 2x2 completeness cells | [PRIOR-GEN] placeholders; tab:decomp joint rows; the joint-eICU metric-disagreement finding |
| Prior-generation independent arms (e25/e26) harvested | tab:decomp independent rows, SUBSET-tagged, to be replaced by R3/R7 |
| v4 re-eval, flagship prior-gen dumps: landmark + readmission legs (EXIT 0) | v3-to-v4 per-cell delta appendix (a paper commitment) |
| R1 `subset_run_v10_taskset_v3` training (binner stamped v2, reset_prob pinned 0.1) | gate only, not paper content; final steps as of Aug 30 ~21:15 UTC |
| eICU spec v2 (HICL drug rescue, bc7ac4c) | prerequisite for R6 |
| GEMINI extraction tooling refreshed (scripts/gemini, Aug 30); 14M forecasting-only checkpoint from the old ladder | starting point for G1/G3; no concept/alert GEMINI results exist |
| ML4H draft with both review passes applied; figures pipeline (make_figures.py); citation audit | swap-ready skeleton |
| **R2** `full_run_v10` (VM1) full-scale MIMIC joint training: 40,750 steps, early-stopped, best_val_loss=2.0556, ~7.2h, zero errors, `clinical_edge_version: v2` stamped, checkpoint_final.pt written | flagship MIMIC training arm; eval chain in flight |
| **R6** `eicu_full_v10` (VM2) full-scale eICU joint training: 35,500 steps, early-stopped, best_val_loss=2.0148, ~4.9h, zero errors, `clinical_edge_version: v2` stamped, checkpoint_final.pt written | flagship eICU training arm; eval chain in flight |

## IN FLIGHT (2026-08-31 morning)

- **R2 eval chain** (VM1, `scripts/eval_run.sh`, commit cdbd4e7): launched
  10:42 UTC against `checkpoint_best.pt`. LANES=64, ALERT_SHARDS=37 (full
  held-out), BASELINE_SHARDS=292 (full train, STREAM_BASELINE=1 per the
  OOM lesson). Log `~/r2_eval_launch.log` on VM1.
- **R6 eval chain** (VM2, same script/commit): launched 10:42 UTC.
  ALERT_SHARDS=17, BASELINE_SHARDS=134 (both full-scale, streamed). Log
  `~/r6_eval_launch.log` on VM2.
- GEMINI deferred (Amrit): MIMIC-IV + eICU full runs with thorough evals
  complete first.

## REMAINING

### Code work items (block eval chains, not training)

| # | Item | Source |
|---|------|--------|
| W1 | Concept overrides scored on hazard-direction sign agreement at the input-counterfactual positions (one-variable contrast) | reviewer M5 |
| W2 | Rule-replay bound (concept recomputed from tokens visible at scoring time), per concept | reviewer m8 |
| W3 | Per-concept band coverage / truth-flip deltas / band-width sensitivity (0.10/0.15/0.20) / CI on the base-rate correlation | reviewer M6 |
| W4 | Decomposition harvest script (per-mode top-1 + loss, retained percents) for every new arm | reviewer N1-N5 |
| W5 | Per-concept assessable fraction beside readout AUROC | reviewer m9 |
| W6 | Calibration column through alerts_cis (AUPRC already there) | paper Sec 4 |

### MIMIC-IV (VM1, ~4h/full run at reset_prob 0.0)

| # | Run | Question | Gate |
|---|-----|----------|------|
| R1e | R1 validation eval | do the fixes reproduce known subset behavior (audit-support, no longer gating -- R2 launched and finished before this ran; backfill once a GPU is free) | R1 training done |
| A1 | counting-run v4 legs + probe pass (prior-gen, audit-support) | completes the v3-to-v4 delta appendix | peer's GBM done-ping |
| A2 | `alerts_cis.py` over flagship v4 dumps | CIs on the delta appendix | RSS headroom |
| R2 | `full_run_v10` seed 0 + full eval chain | flagship joint: readout trajectory (29 concepts), completeness cell, lever, alert cells vs all five baseline families | training done 05:22 UTC Aug 31 (40,750 steps, best_val_loss=2.0556); eval chain launched 10:42 UTC |
| R2b/c | seeds 1, 2 | CIs for every cross-run claim; icu_admission variance watch | R2 sane |
| R3 | independent stage A+B full scale, seed 0 | does the conventional asymmetry survive scale; completeness cell | R2 |
| R3b/c | stage-B replicates | CI on independent-vs-joint cost | R3 |
| R4 | intervention sweep over R2's intermediate checkpoints | "scale fixes the readout, not the lever" as a within-run claim | R2 + W3 |
| R5 | M-series full scale (M1, M3; M2 optional) | gradient-dial hypothesis or its demotion | DECISION D2 |
| B1 | MIMIC v4 baseline completion: MEDS-Tab v4 export+sweep, TabICL/EBM/SurvivalPFN rescores on R2 rows | the comparator table | R2 dumps |

### eICU (VM2: TERMINATED; restart + code sync + v2 re-extraction first)

| # | Run | Question | Gate |
|---|-----|----------|------|
| E0 | VM2 restart, git sync, eICU v2 re-extraction + deep validation | prerequisite | DONE (VM2 up, eicu_2.0_v2 in place, R6 trained against it) |
| R6 | `eicu_full_v10` seed 0 + eval chain | cross-dataset trust replication; joint-eICU metric disagreement retested post-fix | training done 03:14 UTC Aug 31 (35,500 steps, best_val_loss=2.0148); eval chain launched 10:42 UTC |
| R6b | seed replicate | eICU cross-run variance (never measured) | R6 |
| R7 | eICU independent stage A+B full scale | the 2x2's anomaly cell at full scale | R6 |
| B2 | eICU v4 baselines: GBM refit, TabICL/EBM/SurvivalPFN, MEDS-Tab (~15h tabularize, sorted-label doctrine) | eICU comparator table | R6 dumps |

### GEMINI (H200, in-environment, aggregate exports only)

| # | Item | Question | Gate |
|---|------|----------|------|
| G1 | full extraction + `meds_validation --deep` | clean full-scale GEMINI MEDS, subject-disjoint splits | scripts landed |
| G2 | concept resolution audit: which of the 29 resolve (reported as a portability RESULT, paper Sec 3); per-unit ranges (SI creatinine); **binning portability: fraction of GEMINI lab tokens in curated shared-threshold bins vs source-fit quantile bins** (reviewer E4; curated bins are the semantically portable subset) | the GEMINI concept count + the transfer denominator | code_mapping (lead, CPU) |
| G3 | `gemini_full_v10` joint training + eval chain | GEMINI-native readout / completeness / lever cells | G1 + G2 |
| G4 | GEMINI baselines: tuned GBM + MEDS-Tab minimum; TabICL/EBM if the feature path ports | strong-model story on the generalization dataset | G3 dumps |
| G5 | MIMIC-trained R2 zero-shot on GEMINI rows through the shared LOINC token space; **readout reported separately on the curated-bin subset** (E4: separates "concepts do not port" from "quantile edges do not port") | the generalization headline | G1 + G2 + R2 |
| G6 | site-holdout within GEMINI | within-network external validity | STRETCH, D3 |

### Standard eval chain (every trained arm)

Held-out inference; concept readout + CI + assessable fraction (W5) +
rule-replay bound (W2), intermediate checkpoints for flagships;
completeness all modes, top-1 AND loss (W4); interventions with W3 dumps
+ per-subject + intervention_cis; hazard-sign concept overrides (W1)
beside the input-counterfactual suite; alerts v4 vs baselines with
alerts_cis + calibration (W6); registry row; append-only artifacts.

### Decisions owed to Amrit

- D1: confirm reset_prob 0.0 for the whole paper generation (recommended).
- D2: M-series (R5) in or out (~12-18 GPU-h).
- D3: GEMINI scope if time forces a cut: G5-only transfer is the floor;
  note the IRB/REB reference must be confirmed either way (paper has a
  BLOCKING verify tag on the GEMINI ethics statement).
- D4: strong_text stays out of this paper (sidecars need re-embed).

Cut order if Sept 7 arrives with legs unfinished: G6, then R5, then
R6b/R3c. The paper does not survive without R2 (+replicates), R3, R6,
and G2+G5.
