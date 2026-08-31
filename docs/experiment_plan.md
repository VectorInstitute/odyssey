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

- **R2 eval chain**: COMPLETE 15:56 UTC Aug 31, all five stages EXIT 0
  (eval 22m, interventions 102m, alerts 190m). Outputs in
  `~/runs/full_run_v10/` on VM1 incl. alerts_rows.parquet (B1's gate).
  Headlines: set top-1 81.60 (prior-gen 80.4), 29-concept readout mean
  0.9307; interventions none 37.25 / truth -0.19 / flip -0.05 (wrong
  sign survives on top-1; prior-gen loss-ordering corroboration did
  NOT survive, substitution losses within 0.0013 nats); zero_known
  retains 6.6%, zero_unknown 98.4% and improves loss. Paper MIMIC cells
  swapped 2026-08-31 (see main.tex header log).
- **R8 training** (VM1): LAUNCHED 15:58 UTC Aug 31 by odyssey-4a,
  checkout b98ff7d. R2 config + concept_global_pairs=true, output
  `~/runs/full_run_L_v10`, log `~/r8_train.log`. Eval chain + R2
  supplemental scoring queued behind it.
- **R6 eval chain**: COMPLETE 15:07 UTC Aug 31, all five stages EXIT 0
  (eval 16m, interventions 75m, alerts 173m, cases+report <1m). Outputs in
  `~/runs/eicu_full_v10/` on VM2: inference_results.json,
  interventions_band15.json, alerts.json + alerts_rows.parquet (B2's
  gate), case_studies.json, report.html.
- **R9 training** (VM2): COMPLETE 18:04:57 UTC Aug 31. 18,250 steps in
  ~2.5h, early_stop=True (15/15 evals without improvement),
  best_val_loss=2.2677, zero errors, checkpoint_final.pt written.
  Against R6 this arm converges EARLIER and WORSE (18,250 vs 35,500
  steps; 2.2677 vs 2.0148), so the global-pairs capability cost is
  visible in the optimisation itself at full scale, larger than subset
  L1 suggested. Registered in docs/experiments.md.
- **R9 eval chain** (VM2): LAUNCHED 18:07:44 UTC Aug 31 by odyssey-7d,
  checkout b99be32 (floor >=8161e19 verified, imports smoke-tested, no
  uv sync). STREAM_BASELINE=1, LANES=64. Produces Guide Labs modes,
  attribution, per-subject dumps. THEN, in order: scripts/
  intervention_cis.py on interventions_band15_per_subject.json (NOT part
  of the chain), scripts/vm_oneoff/readmission_alerts.sh, and
  scripts/vm_oneoff/supplemental_r6_vm2.sh for the R6 checkpoint.
- GEMINI deferred (Amrit): MIMIC-IV + eICU full runs with thorough evals
  complete first.

## NEXT CYCLE (post Sept 10, pre-registered 2026-08-31)

Prognosis-conditioned generation (the "patient simulator" paper, Amrit +
odyssey-4a discussion 2026-08-31). Motivation from this paper's lever
result: current-state concepts cannot steer because they are rule-derived
from the history the model already read (L2 teacher forcing confirmed:
conditioning on a redundant signal is ignored), and CB-pLM's working lever
is output-conditioning, not a learned causal response. The constructive
successor: condition next-event generation on FUTURE concept outcomes
(e.g. sepsis within 24h), which are uncertain given history and therefore
informative; classifier-free-guidance-style label dropout so conditioned
and unconditioned modes coexist; evaluate as a SIMULATOR, never a
forecaster (conditioned-mode accuracy is leakage by construction and must
never appear beside forecasting baselines). Eval design: (a) fidelity =
replay our own concept rules on generated rollouts, do flagged rollouts
exhibit the concept; (b) calibration = unconditioned rollout concept
incidence vs real incidence; (c) semantics stated as conditional
simulation, explicitly not do()-counterfactuals (the Foresight line this
paper's Discussion draws). Builds on existing pieces: running labels,
rollouts.py, global-pairs bottleneck (R8/R9), landmark protocol v4.

## REMAINING

### Code work items (block eval chains, not training)

| # | Item | Source |
|---|------|--------|
| W1 | Concept overrides scored on hazard-direction sign agreement at the input-counterfactual positions (one-variable contrast) | reviewer M5 |
| W2 | Rule-replay bound (concept recomputed from tokens visible at scoring time), per concept | reviewer m8 |
| W3 | Per-concept band coverage / truth-flip deltas / band-width sensitivity at 0.02/0.05/0.10/0.15/0.20, re-scoped 2026-08-31 as the IDENTIFYING TEST for the truth-flip displacement-asymmetry confound (truth injects L, flip injects 1-L at the same p, so their displacements are complementary and sum to 1, equal only at p=0.5 -- the band narrows, not removes, the asymmetry; found in both odyssey/models/concept_bottleneck.py's docstring and paper Sec 4, both fixed). Pass condition: asymmetry -> ~0 as band narrows toward 0.02-0.05 while the truth-flip contrast survives. Optional stronger fix if cheap: reweight applied positions to equal-mean displacement, or report truth-flip delta stratified by displacement bin. / CI on the base-rate correlation PARTIAL CODE 2026-08-31 (odyssey-4a): per-concept band coverage (n_replaced_by_concept) and per-concept mean displacement now reported by the interventions scorer for every replacing mode; the 0.02/0.05/0.10/0.20 band sweep (modes none/truth/flip, band15 from the chain) is queued in the R2/R6 supplemental scoring scripts to run when a GPU frees. Still open: per-concept truth-flip deltas (needs one-concept-at-a-time passes) and the CI on the base-rate correlation. | reviewer M6 + odyssey-cf audit 2026-08-31 |
| W4 | Decomposition harvest script (per-mode top-1 + loss, retained percents) for every new arm | reviewer N1-N5 |
| W5 | Per-concept assessable fraction beside readout AUROC | reviewer m9 |
| W6 | Calibration column through alerts_cis (AUPRC already there) | paper Sec 4 |
| W7 | Output-calibrated intervention magnitude (Guide Labs eq: gamma = tau / peak(e_c), peak(e_c) = max_y e_c'W_y, one global tau gives every concept the same largest achievable logit shift) as the PRIMARY intervention protocol, superseding the |p-0.5| band rather than adding to it; keep the band as a secondary analysis; report per concept. Motivation: our 0.97 sensitivity-vs-base-rate correlation is plausibly a band artifact (rare concepts rarely populate the band), and output-calibration is a better-targeted fix than band-width alone. Needs eval-chain plumbing (new intervention-strength computation over the LM head weights); ride with the retrain chain, not blocking. CODE LANDED 2026-08-31 (odyssey-4a): truth_calibrated/flip_calibrated modes in odyssey/inference/interventions.py (per-concept gamma_i = tau/peak_i from calibrated_gammas over the LM head weights; step is relative to the model's own p, clipped to [0,1]; NO uncertain band -- calibration replaces it as the equalizer; per-concept gammas recorded in the result JSON) + --calibrated-tau CLI + eval_run.sh runs them at tau=1.0 in the same interventions stage as the banded modes. Directions: exact parameters for global-pairs runs, mean held-out (w+ - w-) for context-pair runs (mean_concept_directions). Band kept as secondary analysis per this row; tau sweep still open if 1.0 proves the wrong scale. | Guide Labs (arXiv:2608.07594) comparison, odyssey-cf 2026-08-31 |
| W8 | ReLU-gated-logit-mask control for `flip` (their Fig 19: naive negative steering promotes anti-aligned/unrelated vocabulary, not just suppressing the aligned direction). Cheap, no retrain: rerun the intervention scorer with a ReLU-gated variant of `flip` and compare truth-flip/flip-none to the naive version, especially on the two independent arms where flip-none is nearly the entire observed effect (-0.78 eICU, -0.45 MIMIC). If the gated variant shrinks flip-none substantially, "independent training makes the model vulnerable to false overrides" is partly a naive-steering artifact, not a finding about the model, and needs re-stating. CODE LANDED 2026-08-31 (odyssey-4a): `flip_gated` mode in odyssey/inference/interventions.py (suppression-only gate, logits_none + min(0, logits_flip - logits_none); two forwards from the same backbone state) and in eval_run.sh's default mode list, so R8/R9 chains score it automatically; still owed on R2/R6 checkpoints (supplemental run, GPU or CPU, after their chains finish). | Guide Labs comparison, odyssey-cf 2026-08-31 |
| W9 | Concept Contribution metric (Guide Labs Eq 22 adapted): sum over known slots of \|slot_i' W_y,i\| divided by that plus \|slot_unknown' W_y,unk\|, averaged over held-out predictions. Exact (our concat+linear head is additively decomposable, same property as their additive bottleneck), ablation-free, no OOD edit -- replaces the zero_known/zero_unknown OOD-edit concern with an exact attribution. Computable on existing checkpoints, no retrain. Report beside the zeroing 2x2 for one paper cycle (comparison on record), not instead of it. CODE LANDED 2026-08-31 (odyssey-4a): odyssey/inference/concept_attribution.py (exact per-slot decomposition of the predicted token's logit, bias excluded; partition-to-one tested against the real forward) + `attribution` stage in eval_run.sh -> attribution.json; still owed on R2/R6 checkpoints. | Guide Labs metric spec, odyssey-cf 2026-08-31 |
| W10 | Known Concept Alignment (Guide Labs T_k(c) = TopK(W K_c) adapted): for our architecture the intervention direction is (w+ - w-) per concept, so compute TopK over W_i(w+ - w-) -- the event tokens each concept override actually promotes. Sharpest available test of "surprisal rather than meaning" (Sec 5.3): if overriding e.g. on_vasopressors does not promote norepinephrine-family tokens, the lever cannot work regardless of training. Cheap, no retrain, highest-value new experiment in the Guide Labs comparison per reviewer. CODE LANDED 2026-08-31 (odyssey-4a): same module/stage as W9; ConceptBottleneck.concept_pair_directions gives (w+ - w-) exactly (identity-tested against the intervention embedding shift); global-pairs runs (R8/R9, L-series) get the exact input-independent TopK, context-pair runs (R2/R6) get TopK of the mean held-out direction; reports activate_promotes and deactivate_promotes per concept. | Guide Labs metric spec, odyssey-cf 2026-08-31 |
| W11 | Concept Independence cross-covariance (Guide Labs' HSIC-style normalized cross-covariance between k-hat and u-hat) as a cheap, citable second measure alongside our capacity-controlled linear probes (kept as the primary, arguably stronger, measure). Optional / lower priority. CODE LANDED 2026-08-31 (odyssey-4a): mean/per-concept absolute Pearson correlation between concept probs and unknown-embedding coordinates, accumulated in the same attribution pass, reported in attribution.json (mean_abs_concept_unknown_correlation). | Guide Labs metric spec, odyssey-cf 2026-08-31 (optional) |

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
| R8 | L-series full scale, seed 0: `concept_global_pairs=True` (context-free per-concept embeddings replace the (w+,w-) pair) + full eval chain | Amrit-authorized 2026-08-31 following the Guide Labs (Steerling) comparison: subset L1 (a14c386, 30 shards) corrected the truth-flip sign (+0.28) at a capability cost, every mode still below none -- fixes the paper's [SUBSET-PLACEHOLDER] tag on this claim (Sec 5.3) and tests whether the sign correction survives AND becomes useful at full scale. No new code (existing `TrainingConfig.concept_global_pairs` flag, already load-bearing in run_inference's checkpoint auto-detect). ~4h training, same eval chain as R2. | VM1 free (after R2's eval chain finishes) |
| B1 | MIMIC v4 baseline completion: MEDS-Tab v4 export+sweep, TabICL/EBM/SurvivalPFN rescores on R2 rows. **TabICL MUST be the strong 609-feature panel at full capability (`n_estimators=8`, 50,000-row context), NOT the basic/reduced config** -- Amrit ruled 2026-08-31 that the basic-feature row is dropped and the full-capability one reported. This needs the 170GB ultra host (`odyssey-cbm-a100-ultra`, a2-ultragpu-1g), ~6h sequential; **BLOCKING-ASK, Amrit must approve the spin-up**. Rerunning on v4 rows also resolves three things the Aug 27 sweep (docs/tabicl_strong_feature_comparison.md, a02c1b3) cannot: its GBM came from a different dump than the paper's (same rows, but refits disagree, death 8h 0.953 vs 0.949), it is protocol v3, and its AKI cells predate the KDIGO fix -- AKI being the one cell where TabICL still loses, so the sole surviving gap currently rests on the known-bad label. Consequence to carry into the paper: at full capability TabICL ties the tuned GBM on 9/12 and BEATS our hazard heads on all three ICU-admission cells, so Sec 5.4's "TabICL loses 12 of 12" is known-false and is marked in main.tex as do-not-submit. | R2 dumps + ultra host approval |

### eICU (VM2: TERMINATED; restart + code sync + v2 re-extraction first)

| # | Run | Question | Gate |
|---|-----|----------|------|
| E0 | VM2 restart, git sync, eICU v2 re-extraction + deep validation | prerequisite | DONE (VM2 up, eicu_2.0_v2 in place, R6 trained against it) |
| R6 | `eicu_full_v10` seed 0 + eval chain | cross-dataset trust replication; joint-eICU metric disagreement retested post-fix | training done 03:14 UTC Aug 31 (35,500 steps, best_val_loss=2.0148); eval chain launched 10:42 UTC |
| R6b | seed replicate | eICU cross-run variance (never measured) | R6 |
| R7 | eICU independent stage A+B full scale | the 2x2's anomaly cell at full scale | R6 |
| R9 | L-series full scale, seed 0: `concept_global_pairs=True`, eICU counterpart to R8 | does the sign-correction + capability-cost tradeoff replicate cross-dataset | VM2 free (after R6's eval chain finishes) |
| B2 | eICU v4 baselines: GBM refit, TabICL/EBM/SurvivalPFN, MEDS-Tab (~15h tabularize, sorted-label doctrine) | eICU comparator table | R6 dumps |

### GEMINI (H200, in-environment, aggregate exports only)

STARTED 2026-08-31 per Amrit (supersedes the "defer until MIMIC+eICU solid"
note): Amrit operates the node (runs `scripts/gemini/run.sh <step>` only);
the paper session (odyssey-6b at time of writing) owns the leg end-to-end
(GitHub->GEMINI mirroring, step sequencing with Amrit, git copy-back of
results, paper integration); code changes and G2 stay with the lead
session. First action: mirror (gemini/main was at 9e83e5f, pre-Aug-30
tooling; step-0 out/ diff verified empty 2026-08-31 ~12:20 UTC).

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
- D5: abstract framing, DEFERRED BY AMRIT 2026-08-31 ("let CI land, then
  discuss with odyssey-cf"; that lane is odyssey-6b this generation).
  Blocked on `intervention_cis.json` for R2/R6 from the supplemental
  scoring pass. Three coupled questions, to be settled in ONE pass, not
  separately: (a) does the eval-audit sentence earn abstract space, or
  should its surviving clause fold into the close; (b) the closing
  "tested by intervention, not by readout accuracy" is NOT a novel
  claim and Related Work concedes as much, so does it keep the last
  line of the abstract; (c) if the paired CI on eICU truth$-$none
  (+0.19 top-1) clears significance, "the lever never helps" needs a
  decision, not a wording tweak. Full assessment and the caution that
  the current close was a deliberate prior W-item choice are logged in
  the paper/ml4h/main.tex header block. No abstract edits before then.

Cut order if Sept 7 arrives with legs unfinished: G6, then R5, then
R6b/R3c. The paper does not survive without R2 (+replicates), R3, R6,
and G2+G5.
