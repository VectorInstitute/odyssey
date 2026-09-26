# ML4H 2026 submission 234: rebuttal and camera-ready plan

Written 2026-09-25 from the internal review in `reviews/234_review.md` and the repo state on branch `summary-head`.

## Dates that fix the plan

| Date | Event |
|---|---|
| Oct 5 | Reviews released; author response opens |
| Oct 12 | Author response closes (7 days) |
| Oct 22 | Decisions |
| Nov 7 (tentative) | Camera-ready |

ML4H policy for the response window: authors may "add specific types of new experimental results as requested by the reviewers", but "no conceptual changes to the original formulation are allowed beyond clarifications". So every experiment below must be banked BEFORE Oct 5. During Oct 5 to 12 we only pick which banked result answers which reviewer. Nothing new gets trained in that week.

That leaves 10 days of compute, Sep 25 to Oct 4.

## What the internal review says reviewers will most likely ask for

Ranked by how likely a real reviewer raises it, times how much it moves the decision.

1. **Score the label override on the hazard heads**, not only on next-event top-1/loss (review W1). Nearly certain to be asked. Inference only.
2. **A no-bottleneck arm** of the same backbone, to price the bottleneck (W4). Very likely. Needs full-scale training.
3. **Show the forecast runs through the concept probabilities, not just the concept embeddings** (W2): a Yeh-style completeness score and a leakage probe. Likely. Inference only.
4. **A RandInt or intervention-aware arm** (W5). Likely from any CBM-literate reviewer. Needs training.
5. **A GEMINI paired bootstrap against the current refit, plus how many panel signals resolve on GEMINI** (W3). Likely. Needs Amrit inside the GEMINI environment.
6. **Model size and training settings** (W10). Certain, and free.
7. **Cohort table, per-concept prevalence, mid-visit readout** (W7, W11). Possible. Cheap for MIMIC/eICU, GEMINI needs Amrit.

Everything else in the review (W6 edit test is a whole-model test, W7 upper-bound wording, W8 subset disclosure, W9 bolding, minor issues) is text, not experiments. It goes in the rebuttal as clarifications and in the camera-ready as edits.

## Work packages

Each package says what exists, what to build, who runs it, and where the result lands. Owner "lead" = the Claude lead session under Amrit's instruction; "VM1/VM2" = the two A100 operator sessions; "Amrit" = needs a human, usually for GEMINI.

### WP1. Override on the hazard heads (W1). Inference only. Days 1 to 3.

- Exists: `odyssey/inference/interventions.py` runs truth/flip/none/calibrated/zeroing but scores only `top1_accuracy` and `mean_task_loss`. `odyssey/inference/steering.py` already scores hazard heads and knows the landmark rows.
- Build: add hazard-head outputs to `interventions.py` (per event, per horizon, at landmark positions, paired subject-clustered bootstrap of truth minus none and truth minus flip on AUROC and on mean hazard). Reuse the landmark row set and the bootstrap from `scripts/alerts_cis.py` so the rows match Tables 12 to 14.
- Run: on the banked flagship checkpoints, MIMIC `full_run_v10` on VM1 and the eICU joint mixture on VM2, band 0.15, all held-out shards (full-data rule).
- Built 2026-09-25 (branch `rebuttal/hazard-override`): `--hazard-heads` adds a `hazard` block per mode ({event: {horizon: {auroc, mean_risk, n_at_risk, n_positive, n_censored}}}) read at the alert protocol's landmark rows (same mask and outcome rule as `alerts.py`, so n_at_risk matches Tables 12 to 14 on the hybrid flagships), and a `hazard_paired` block on the left-hand mode's entry (truth carries truth_minus_none and truth_minus_flip, flip carries flip_minus_none) with 95% subject-clustered paired bootstrap intervals on the AUROC difference and on the mean P(event within h) difference. Off by default; the existing top-1/loss JSON is unchanged. Command to run on each VM (one pass per mode, the hazard readout is free; the bootstrap runs on the landmark table only):

  ```
  python -m odyssey.inference.interventions --run-dir R --held-out-shard-dir D/held_out \
      --output-json R/interventions_band15_hazard.json --max-shards 37 --num-lanes 64 \
      --chunk-size 512 --uncertain-band 0.15 --modes none truth flip random \
      --hazard-heads --hazard-boot 1000 --checkpoint checkpoint_best.pt
  ```

  Set `--max-shards` to the split's full shard count on each VM (37 is MIMIC's). The transformer arm keeps the lever test's TBTT view (whole history, no context truncation), so its landmark rows are the model-free set, not the packed-context set `alerts.py` scores for that backbone.
- Bank: `research_journal/figure_data/{vm1,vm2}/<run>/interventions_hazard.json`. New table generator `scripts/make_lever_hazard_table.py`.
- Rebuttal use: if truth moves the 24 h death or vasopressor hazard the right way, the lever verdict changes and the paper's Q3 gets a real endpoint. If it does not, the negative result becomes like-for-like with the edit test, which is what the review asked for. Either way it answers W1.
- GEMINI: same script, run by Amrit, only if time.

### WP2. No-bottleneck arm, full scale (W4). Training. Days 1 to 7.

- Exists: `TrainingConfig.model_kind = "baseline"` (same backbone and heads, no concept module). Only `subset_baseline_v5/v6` (30 shards) were run; v6 showed baseline top-1 +2.2 over the bottleneck on the subset.
- Build: nothing. Copy the flagship `config.json` and set `model_kind: baseline`.
- Run: MIMIC on VM1 and eICU on VM2, every training shard, 2 epochs like the flagship, then the full alert protocol (`alerts_cis.py --scorers hazard gbm`) on all held-out shards. Start this on day 1 because it is the longest job.
- Bank: `research_journal/figure_data/{vm1,vm2}/full_baseline_v13/`. Add rows to `docs/experiments.md` and a column to `make_comparator_tables.py`.
- Rebuttal use: reports the bottleneck's cost as a paired AUROC difference per cell, which is the number Q4 promises. Also settles whether the TabICLv2 result means "we lack the panel" or "the sequence model is undertrained".
- Rule: full-data numbers only. No subset numbers go in the response.
- Paired scoring (branch `rebuttal/integration`): `scripts/compare_runs_paired.py` takes the two runs' `alerts_rows.parquet` dumps, inner-joins them per event on the row key `(subject_id, visit_id, time_hours)` that `index_row_table` writes, and refuses (exit 2, both row counts and the unmatched count printed) when more than 0.1% of either dump's rows are unmatched, so a row-set mismatch cannot pass silently. On the joined rows it reports, per (event, horizon), both hazard-head AUROCs and the paired subject-clustered bootstrap of `baseline minus bottleneck` (`bootstrap_auroc_delta`, 1000 draws), the two GBM refits side by side without a bootstrap (their difference is refit variance, for scale), and with `--inference-a/--inference-b` the next-event set top-1, exact top-1 and cross-entropy from each `inference_results.json`. VM1 command, from `~/odyssey` after the baseline alert chain has written its dump: `uv run python scripts/compare_runs_paired.py --dump-a ~/runs/full_run_v10/alerts_rows.parquet --dump-b ~/runs/full_run_baseline_v10/alerts_rows.parquet --label-a bottleneck --label-b baseline --inference-a ~/runs/full_run_v10/inference_results.json --inference-b ~/runs/full_run_baseline_v10/inference_results.json --output-json ~/runs/full_run_baseline_v10/paired_vs_v10.json`. VM2: the same with `~/runs/eicu_full_v10` and `~/runs/eicu_full_baseline_v10`. The dump file names must be the ones the chain actually wrote (the flagships' dumps may carry a `_v4` suffix; `ls ~/runs/<run>/alerts_rows*.parquet` first). Copy the JSON to `research_journal/figure_data/{vm1,vm2}/<baseline run>/paired_vs_v10.json`; the printed markdown table is the rebuttal table for Q4.

### WP3. Completeness and leakage probes (W2). Inference only. Days 2 to 4.

- Exists: `compute_completeness` in `odyssey/training/metrics.py` (Yeh et al.: logistic readout on concept probabilities), only called from a test. `odyssey/inference/leakage.py` defines the CTL probes (probs-only, known-embeddings, residual, random-projected probs) with no banked output.
- Build: a script `scripts/probe_channel.py` that loads a checkpoint, dumps at landmark positions the concept probabilities k, the named embeddings z, and the poles w+/w-, and fits (a) a readout on k alone, (b) on z alone, (c) on the poles with k held fixed, for next-event top-1 and for each hazard event. Report retained accuracy relative to the full model.
- Run: flagship MIMIC and eICU checkpoints, all held-out shards or a stated subsample if the dump is too large (say so in the table).
- Bank: `research_journal/figure_data/<run>/channel_probes.json`.
- Rebuttal use: the review's strongest technical point is that the poles, not k, carry the forecast. This measures it directly. If k alone recovers most of the accuracy, W2 is answered. If not, the Q2 sentence in the abstract must be reworded to "runs through the named embeddings", and we say so in the response rather than defend it.
- Script (branch `rebuttal/channel-probes`): `scripts/probe_channel.py` dumps, at every scored held-out position (a seeded, shard-stratified subsample capped by `--max-positions`, default 2,000,000), the concept probabilities k, the unnamed slot's probability u, the poles w+/w- and the unnamed embedding z_u, then fits fresh linear readouts on a split BY SUBJECT (70/10/20 train/tune/test) and scores exact next-event top-1 on the held-out subjects: `k_only`, `k_plus_u`, `z_named` (the named embeddings), `poles_mean_k` (z rebuilt with every k held at its training mean), `poles_both` (raw [w+, w-]), `h_bar` (the full bottleneck, the ceiling) and `model_head` (the model's own head, no refit). Each carries a subject-clustered bootstrap CI (1000 resamples) and its ratio to the model's own accuracy on the same rows ("retained", with a paired CI); the CTL leakage probes of `odyssey/inference/leakage.py` run on the same split from the same dump. Hazard heads at landmarks are not dumped: the landmark protocol lives in `alerts.py`, and a second implementation here would risk a number that disagrees with the alert tables. VM1 command, from `~/odyssey` after `git fetch && git reset --hard origin/main` (never `uv sync` on a VM): `setsid nohup uv run python scripts/probe_channel.py --run-dir ~/runs/full_run_v10 --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out --output-json ~/runs/full_run_v10/channel_probes.json --num-lanes 16 --chunk-size 512 --max-positions 2000000 --seed 0 > ~/runs/full_run_v10/channel_probes.log 2>&1 & disown`. VM2: the same with `~/runs/eicu_full_v10` and `~/data/eicu_2.0_v1/data/held_out`. Runtime guess: the dump is one streaming pass at 16 lanes, about 20 to 30 minutes on MIMIC's 37 shards (the Sep 1 intervention rerun took about 20 minutes per mode) and less on eICU's 17; then six readout fits of 5 epochs over about 1.4M rows against the full vocabulary (a few minutes each on the A100) and four CTL probes on a 500k-row cap. Budget one hour per flagship. Memory: the bank keeps both poles per concept, 2M x 29 x 32 x 2 x 2 bytes = 7.4 GB in fp16 on the GPU; pass `--bank-on-cpu` if training shares the card. Copy the JSON to `research_journal/figure_data/<run>/channel_probes.json`. Reading it: if `k_only` recovers most of `h_bar` (retained near 1), the forecast runs through the named STATES and W2 is answered; if `poles_mean_k` recovers most of it while `k_only` does not, it runs through the poles and the Q2 sentence must say "through the named embeddings". `k_plus_u` against `k_only` prices the unnamed slot's probability; `z_named` against `h_bar` prices the unnamed embedding.

### WP4. RandInt arm (W5). Training. Days 2 to 8.

- Exists: `randint_prob` in `TrainingConfig` (default 0.25; flagships set 0.0). Steerling-style steering epochs `eicu_full_DEC_v13_steer2` and a control epoch `_ctrl` are finished, banked under `research_journal/figure_data/vm2/`, and recorded in `docs/experiments.md` (eval rows): neither epoch makes the override help (truth minus none, top-1 points: v13 -0.16, steer2 -0.18, ctrl -0.12), the extra epoch rather than the steering losses carries the accuracy gains, and no epoch strengthens the lever. The paper body says only that the retrofit "does not reliably strengthen the dials".
- Also exists, at subset scale only (docs/experiments.md rows 36, 60, 65; journal entries 08 and 25): `subset_run_v4` (MIMIC 30 shards, RandInt 0.25, early architecture without hazard heads: lever correctly signed but +0.1 top-1 point, exact top-1 -8.8 vs no RandInt, retired), `subset_run_L2` (MIMIC 30 shards, RandInt 1.0: none 25.9, truth 25.9, flip 25.8, no separation) and `eicu_subset_indep_b` (eICU 30 shards, stage-B independent training with RandInt 1.0: truth minus flip +0.78 points, truth ties none, set top-1 collapsed to 47.1). So the "standard remedy" was applied three times and never produced a usable lever. Under the paper's rules the lever numbers are admissible (interventions are exempt from the full-data rule), the accuracy costs are not.
- Build: nothing for training. The steer2/ctrl result is already a banked intervention-aware arm and goes straight into the rebuttal block for W5, together with the three subset RandInt arms as development-scale history. The full-scale eICU RandInt run below turns that history into one full-data number; the subset runs tell us what to expect (correct sign, tiny magnitude, an accuracy cost).
- Run: eICU joint mixture with `randint_prob: 0.25`, full scale, on VM2 after WP2 finishes there. MIMIC only if VM1 is free by day 5. Score with `interventions.py` (top-1/loss and the new hazard output from WP1) and the readout table.
- Bank: `research_journal/figure_data/vm2/eicu_full_RI_v13/`.
- Rebuttal use: the reviewer will say "CEM's RandInt fixes this and you turned it off". We answer with the RandInt arm's truth-vs-none numbers and with the already-banked steering epochs. If RandInt makes truth beat none, that is a positive finding and goes in the camera-ready as the predicted remedy confirmed. If not, the negative result is stronger.

### WP5. GEMINI items (W3, W11). Needs Amrit. Days 3 to 9, one session inside the secure environment.

- (a) Paired bootstrap against the current refit: `scripts/alerts_cis.py --dump <alerts_rows.parquet> --scorers hazard gbm` on the current GEMINI GBM refit. Built as the `alerts-cis` step of `scripts/gemini/run.sh` (branch `rebuttal/gemini-stages`): it reads `~/runs/gemini_full_DEC_v12/alerts_rows_allshards.parquet`, the dump the `_allshards` alerts pass left on the node (GBM on all 894 train shards at 10% row thinning, all 112 held-out shards), prints its row and subject counts, runs 1000 paired subject-clustered draws, and exports `scripts/gemini/out/evals/gemini_full_DEC_v12_allshards_alerts_cis.json` (per event x horizon: hazard AUROC, GBM AUROC, hazard minus GBM with a 95% interval, n_at_risk, n_positive, n_subjects). The full dump is about 25M rows and the bootstrap takes 15 to 20 h; set `GEMINI_CIS_MAX_SUBJECTS` if the session cannot afford that, and the output records the subsample size.
- (b) Panel coverage: `scripts/panel_coverage.py` resolves the 48 panel signals against a source's LOINC table in `odyssey/data/code_mapping.py` and, given `metadata/codes.parquet`, says which resolved prefixes were actually charted. The in-repo table already answers the review's guess: GEMINI resolves 15 of 48, and the non-invasive blood-pressure panel is among the unresolved. The `panel-coverage` step confirms it on the node's code inventory and exports the names. For MIMIC-IV and eICU run `uv run python scripts/panel_coverage.py --source mimic_iv` (or `eicu`) anywhere.
- (c) Cohort counts: `scripts/cohort_counts.py` over the MEDS shards a run used (train and tuning from its `config.json`, held-out from the eval dir): subjects, admissions, hospitals from `metadata/hadm_id_hospital.parquet`, admission year range, length of stay, sex and age where the source charts them (GEMINI extracts neither, so those read "not available"), and the share of subjects with each hazard event under the alerts leg's own onset definitions (sepsis3 does not resolve on GEMINI and is listed under `events_dropped`). Every count under 10 leaves as `"<10"`. The `cohort-counts` step exports `scripts/gemini/out/evals/gemini_full_DEC_v12_cohort_counts.json`. The same script runs on the VMs for MIMIC-IV and eICU with `--run-dir <run> --split held_out=<dir>`.
- Commands for Amrit's session, in this order, each inside tmux, each self-syncing and idempotent (re-running with the output present only re-exports it):

  ```
  tmux new -s wp5 'scripts/gemini/run.sh panel-coverage gemini_full_DEC_v12'
  tmux new -s wp5 'scripts/gemini/run.sh cohort-counts gemini_full_DEC_v12'
  tmux new -s wp5 'scripts/gemini/run.sh alerts-cis gemini_full_DEC_v12'
  ```

  `alerts-cis` goes last because it is the long one. If the printed dump size makes the full bootstrap unaffordable, use `GEMINI_CIS_MAX_SUBJECTS=200000 scripts/gemini/run.sh alerts-cis gemini_full_DEC_v12` instead, and the rebuttal cites the subsample size the JSON records. If the `_allshards` dump is missing, the step prints the exact `alerts` command that regenerates it.
- (d) GEMINI GBM count-feature ablation (Table 15 on GEMINI) if the session has time. This is the one that would let us keep "the deficit belongs to those sites" or force us to drop it.
- Rebuttal use: the abstract's "the deficit belongs to those sites, not the design" is the sentence most at risk. If (a) separates the 9 wins against the current refit and (b) shows the panel is not handicapped, the sentence stays. If (b) shows the panel lost blood pressure on GEMINI, we soften the sentence in the response and in the camera-ready to "reverses by point estimate on GEMINI; the panel there lacks N of 48 signals".

### WP6. Free items (W10, W11, W7). Text and small scripts. Days 1 to 4.

- Hyperparameter table for the appendix from the banked `config.json` files: hybrid backbone, hidden 256, 8 layers, 8 heads, mamba state 128, concept embedding 32, 64 lanes x 512 chunk, context 4096, AdamW, lr 3e-4, weight decay 0.01, clip 1.0, 2 epochs, early-stop patience 15, randint 0. Add the parameter count (write a 10-line script that instantiates the flagship config and counts). Add the GBM's four configurations and estimator class from the panel code.
- Cohort table for MIMIC and eICU (subjects, admissions, age, sex, length of stay, per-event prevalence per subject and per row) from the MEDS shards on the VMs. Extend `make_cohort_table.py` or add `make_demographics_table.py`.
- Per-concept prevalence column in the readout table (`make_readout_table.py`).
- Mid-visit readout: concept AUROC at each landmark instead of visit end. Add a `--at landmarks` option to the readout scorer. Run on MIMIC and eICU flagships. Optional; do it if WP1 to WP3 finish early.
- AUPRC for the rare cells: the paper says they are in the released files. Put the death and Sepsis-3 AUPRCs in an appendix table so nobody has to open the files.

### WP7. Rebuttal text, prepared before Oct 5

Write `paper/ml4h/rebuttal_draft.md` with one block per expected criticism, each with: one-sentence concession or disagreement, the banked number to cite, and the camera-ready edit we commit to. Blocks:

1. Lever scored on next-event only. Concede; cite WP1.
2. Slot versus state. Concede the mechanism; cite WP3; commit to rewording Q2 if k alone does not carry it.
3. GEMINI reversal without an interval. Cite WP5(a) and (b); commit to softening if needed.
4. Bottleneck cost not isolated. Cite WP2.
5. RandInt not run. Cite WP4 and the banked steering epochs; update the related-work paragraph to say Koh, CEM, IntCEM predicted it.
6. Edit test is a whole-model test. Agree; reword "first to probe a concept bottleneck's lever by editing the input events" to "first to test record edits on an EHR forecaster"; keep the occlusion-discovered edits as the one place the bottleneck contributes.
7. Readout is an upper bound. Agree; add the qualifier to the abstract; add the mid-visit readout if run.
8. Subset and stale-checkpoint numbers in the abstract. Agree; mark the development-scale, five-subject and 30-shard origins in the abstract or drop those sentences.
9. Bolded cells below the 0.01 floor; no multiplicity correction. Unbold the two cells; add a sentence on family size; keep paired inference as primary.
10. Reproducibility. Cite WP6 and the anonymous repo.
11. Missing citations. Add Laguna et al. 2024, Sun et al. 2025, Vandenhirtz et al. 2024, Sawada and Nakamura 2022, Delphi-2M, Zeiler and Fergus, and the EHR FM list.
12. Anonymity. If a reviewer notes the REB number, say it will be replaced by a placeholder; nothing else to do now.

Keep each block under 150 words. The response box on OpenReview is short.

### WP8. Camera-ready edits, prepared now, applied after Oct 22

- Abstract: cut from 463 words to about 250, at most 8 numbers, with the upper-bound and subset qualifiers.
- Move Table 3 and a compact Tables 12 to 14 into the body; move Figure 1 to the appendix if space is short. Check the ML4H camera-ready page allowance first.
- Define the decomposed arm or drop Tables 4, 5 and 19 to the released files.
- Fix the 26-versus-29 sentence, the "same result a third time" sentence, "10 of 15", "0.030 to 0.054", the Table 2 "median" label, "Two heads", "Three forecasting terms", the 24% versus 4.6% sentence, the naming drift, the stray copyright line, and the acronym expansions. Full list in `reviews/234_review.md`, Minor issues.
- Move the AKI stage 3 and metabolic acidosis rule caveats and the palliative-code note from captions into the body.
- Run `figures/pagecheck.py` and the awk comment check after every edit (the build has lost prose to `%` lines before).
- The submitted source is `paper/ml4h/main_mixture.tex`; the old `main.tex` and its aux files were retired to `paper/ml4h/retired/` on 2026-09-25. `make_steering_table.py` and `make_specificity_table.py` now feed no table in the paper; keep them for the steering follow-up.

## Results so far (updated 2026-09-26, 01:30 UTC)

Every number below is banked under `research_journal/figure_data/` in the run directory named, and was computed on all held-out shards unless stated. Runs launched 2026-09-25 from the `rebuttal/integration` branch; VM recipe and logs are in the session memory and the chain scripts under the VM home directories.

### WP1, override scored on the hazard heads: done, negative on both databases

`interventions_band15_hazard.json` under `vm1/full_run_v10/` and `vm2/eicu_full_v10/`. Band 0.15, modes none / truth / flip / random, hazards read at the landmark rows of Tables 12 and 13 (the no-intervention hazard AUROCs reproduce those tables to three decimals), paired subject-clustered bootstrap with 1,000 resamples.

- eICU-CRD (655,415 landmark rows): truth minus none on hazard AUROC is negative and separated in all 12 cells (vasopressor -0.006 to -0.008, AKI -0.003 to -0.006, death -0.002 to -0.006, ICU -0.001 to -0.003). Mean predicted risk moves by at most 0.004 absolute.
- MIMIC-IV (1,214,849 landmark rows): truth minus none between -0.0002 and -0.006 across 15 cells, negative and separated in 9, never positive. Mean risk moves by at most 0.0015.
- Reading: the label override is inert on the clinical hazards as it is on next-event accuracy. The two halves of Q3 are now scored on the same endpoint.

### WP2, the cost of the bottleneck: done pending the code-drift retrains

No-bottleneck arms (`model_kind=baseline`, flagship recipe, RandInt 0) trained at full scale: `vm1/full_run_baseline_v10` (49,375 steps, best val 1.7985) and `vm2/eicu_full_baseline_v10` (30,250 steps, early stop, best val 1.5362). Scored with the standard chain (GBM refit on every train shard) and compared with the flagship on identical landmark rows by `scripts/compare_runs_paired.py` (`paired_vs_v10.json`, `paired_vs_v10_rescored.json`).

- MIMIC-IV: the no-bottleneck arm is ahead in 15 of 15 cells, every interval clear of zero: AKI +0.011 / +0.010 / +0.008 (8 / 24 / 72 h), death +0.005 / +0.007 / +0.007, ICU admission +0.010 / +0.013 / +0.018, Sepsis-3 +0.013 / +0.012 / +0.013, vasopressor start +0.006 / +0.006 / +0.011. Next-event set top-1 81.6 to 83.3, exact top-1 37.25 to 37.79, cross-entropy 3.556 to 3.525.
- eICU-CRD: ahead in 11 of 12 cells (ICU admission at 72 h ties): death +0.029 / +0.034 / +0.039, vasopressor +0.015 / +0.014 / +0.009, ICU +0.012 / +0.007 / +0.005, AKI +0.107 / +0.097 / +0.103 under the current AKI label. Exact top-1 54.0 to 55.9, cross-entropy 1.975 to 1.798.
- Against the panel, the no-bottleneck arm still loses every original cell on both databases (`alerts_cis.json`), by about half the bottleneck's margin on death and a third on vasopressor start, and it wins the new eICU Sepsis-3 cells at 8 and 24 h where the panel has 469 positives.
- Caveat still open: the flagships were trained on 2026-08-31 code and the baselines on current code. Like-for-like bottleneck retrains on current code are queued: `eicu_full_v10_re` (VM2, after the RandInt chain) and `full_run_v10_re` (VM1, after the MIMIC baseline CIs). If they match the flagships, the numbers above are the bottleneck's cost; if they close part of the gap, the difference was code drift and the retrain pair replaces the flagship pair.

Two facts found on the way. First, the banked eICU Table 13 AKI cells are under an older AKI label: rescoring the same checkpoint with current code (`vm2/eicu_full_v10/alerts_rescore.json`) leaves death, vasopressor and ICU AUROCs identical and drops AKI from 0.741 to 0.649 at 8 h (at-risk rows 451,747 to 308,700; the GBM drops from 0.891 to 0.820). The paper must state which label Table 13 uses. Second, the panel's death-at-8 h AUROC on MIMIC-IV moved from 0.944 to 0.896 between two refits of the same recipe (1,987 positives), larger than the 0.029 refit variance the paper reports; the other cells agree within 0.005.

### WP3, which channel carries the forecast: done, the poles carry it

`channel_probes.json` under both flagship run directories (banks `channel_probes.bank.pt`, about 8 GB each, stay on the VMs). Subject-split held-out sample of 2,000,000 positions; fresh linear readouts scored on strict next-event top-1 over about 390,000 (MIMIC) and 400,000 (eICU) test positions; 95% subject-clustered intervals within 0.005.

| readout | MIMIC-IV | eICU-CRD |
|---|---|---|
| model's own head | 0.372 | 0.539 |
| concept probabilities k only | 0.261 | 0.565 |
| k plus the unnamed slot | 0.264 | 0.585 |
| named embeddings z | 0.584 | 0.790 |
| poles with k fixed at its mean | 0.583 | 0.790 |
| poles alone (raw w+, w-) | 0.569 | 0.786 |
| full bottleneck | 0.584 | 0.788 |

- The poles with the probabilities held constant recover everything the bottleneck carries. The probabilities alone recover 45% of it on MIMIC-IV and 72% on eICU-CRD. The forecast runs through the named embeddings; the reviewer's W2 point stands and Q2 must be reworded.
- Fresh readouts beat the model's own head because they optimise strict top-1 while the model trains bundle-invariant; compare readouts with the full-bottleneck readout, not with the model head.
- CTL leakage probes on the next-token code family (9 classes): probabilities only 0.935 / 0.967, embeddings only 0.956 / 0.973, unnamed slot only 0.878 / 0.953, random-projected probabilities 0.936 / 0.968 (MIMIC / eICU).

### WP4, RandInt: three subset arms banked, the full-scale arm training

`eicu_full_RI_v10` (flagship recipe with `randint_prob 0.25`) started 2026-09-25 20:24 UTC on VM2, then the full eval chain and CIs. The three subset-scale arms and the steering / control epochs are listed above under WP4.

### WP5, GEMINI: code ready, the node session is Amrit's

`scripts/panel_coverage.py` on the in-repo mapping table: GEMINI resolves 15 of the 48 panel signals; the non-invasive systolic, diastolic and mean pressures are among the unresolved (only arterial systolic maps). The three run.sh steps (`panel-coverage`, `cohort-counts`, `alerts-cis`) are built and tested; commands are listed under WP5.

### WP6, free items: done

- Parameters: 32,233,101 (MIMIC-IV flagship) and 20,384,390 (eICU-CRD), the difference being the per-source next-event vocabulary head; `research_journal/figure_data/param_counts.json`. `paper/ml4h/tables/hparams.tex` generated by `scripts/make_hparams_table.py` with the GBM grid, estimator and panel sizes read from the code.
- Cohort counts (`cohort_counts.json` under both flagship run directories): MIMIC-IV 291,702 / 36,463 / 36,462 subjects (train / tuning / held-out), 435,803 / 54,898 / 55,328 admissions, median stay 2.8 days, 53% female; per-subject prevalence vasopressor 5.0%, ICU admission 17.9%, AKI 18.2%, death 10.5%, Sepsis-3 12.6%, 30-day readmission 13.8%. eICU-CRD 133,084 / 16,636 / 16,635 subjects, 160,643 / 20,094 / 20,122 stays, years 2014 to 2016, median stay 5.5 days; prevalence vasopressor 27.8%, AKI 55.9%, death 8.8%, Sepsis-3 0.28%, readmission 10.5%.

### Code landed on `rebuttal/integration`

Event pinning for old checkpoints (eICU checkpoints trained before PR #222 have five hazard heads; the registry now builds six because Sepsis-3 resolves; `run_pins.json` plus a legacy table), hazard-head scoring in `interventions.py` (`--hazard-heads`), `scripts/probe_channel.py` (with bank save and resume), the three GEMINI run.sh steps, `scripts/make_hparams_table.py`, `scripts/cohort_counts.py`, `scripts/panel_coverage.py`, `scripts/compare_runs_paired.py`. 1,503 tests pass. Not merged to main yet.

## Schedule

| Day | VM1 (MIMIC) | VM2 (eICU) | Lead / Amrit |
|---|---|---|---|
| Sep 25 to 26 | Start WP2 baseline training | Start WP2 baseline training | WP1 code; WP6 hyperparameter table; registry rows for steer2/ctrl |
| Sep 27 to 28 | WP1 hazard override on v10 | WP1 hazard override on eICU flagship | WP3 probe script; WP5 alerts_cis stage for GEMINI run.sh; signal-panel coverage report |
| Sep 29 to 30 | WP3 probes on v10 | WP3 probes; then start WP4 RandInt | WP6 cohort and prevalence tables |
| Oct 1 to 2 | WP2 alert scoring; WP4 MIMIC if free | WP2 alert scoring | Amrit: WP5 GEMINI session |
| Oct 3 to 4 | Mid-visit readout (optional) | WP4 scoring | Bank everything; write WP7 rebuttal blocks with numbers filled in |
| Oct 5 to 12 | | | Read reviews; pick blocks; submit response by Oct 11 |
| Oct 22 to Nov 7 | | | WP8 camera-ready |

Both VMs were stopped after the last queues. Day 1 starts with bringing them up and checking disk. Kill remote jobs by PID, never `pkill -f` over SSH.

## What we do not do

- No hand-engineered features in the foundation model. The count-feature gap is explained, not closed. The summary-head work on this branch is a next-cycle experiment and stays out of the response.
- No new architecture and no new datasets. The policy forbids conceptual changes, and a reviewer would read it as a different paper.
- No subset numbers in the response. Every number we cite is full-data on all held-out shards, or labelled as an intervention cohort with its n.

## Ownership and hand-off

- Lead session: WP1, WP3, WP6 code; WP7 and WP8 drafts; registry updates.
- VM1 and VM2 operator sessions: WP2, WP4 training and scoring, under exact instructions from the lead.
- Amrit: WP5 GEMINI session (one sitting, about half a day), final read of the response, and the OpenReview submission.
