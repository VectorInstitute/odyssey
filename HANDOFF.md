# HANDOFF — odyssey ML4H paper push

From session `odyssey-7d`, 2026-08-31 ~18:30 UTC. Successor to `odyssey-4a`.
Repo `~/src/odyssey`, branch `main`, pushed clean at **8aa24e5**.

Target: ML4H 2026 Proceedings, deadline **Sept 10 AoE**. A CBM trust audit of an
EHR foundation model on MIMIC-IV + eICU + GEMINI.

**Read these first, in this order:** the header comment block of
`paper/ml4h/main.tex` (logs every pass with full provenance, and currently
carries three do-not-submit / frozen markers), then `docs/experiment_plan.md`
(the queue), then `docs/experiments.md` (the registry of what happened).

---

## 1. Running right now — do not disturb, one GPU job per card

| What | Host | Started | Expect | Watch |
|---|---|---|---|---|
| **R8 training** (MIMIC L-series, `full_run_L_v10`) | `odyssey-cbm-a100` (VM1, us-central1-f) | 16:32 UTC | ~23:30, maybe earlier | `~/r8_train.log`; done = `checkpoint_final.pt`. At 20:27 it was step 22,225, best_val 2.3235, early-stop 2/15 |
| **R6 supplemental** (Guide Labs modes, W3 band sweep, per-subject, CIs) | `odyssey-eicu-a100` (VM2, us-central1-f) | 20:41:36 UTC | hours (9 modes + a 4-band sweep + attribution) | `~/r6_guidelabs.log`; done = `GUIDELABS_DONE`. **Produces R6's `intervention_cis.json`, the eICU CI that D5 waits on** |
| **B1 TabICL(strong, full capability)** | `odyssey-cbm-a100-ultra` (us-central1-a) | 18:32 UTC | ~7.5h, so ~02:00 UTC | `~/b1_tabicl.log`; done = final `INFO wrote <path>` line, NOT file existence. At 20:30, 3 of 15 cells |

SSH: `gcloud compute ssh --zone <zone> <vm> --project agentic-ai-evaluation-bootcamp --tunnel-through-iap`
(note the ultra host is **us-central1-a**, the other two are **us-central1-f**).

My watchers die with my session — **re-arm your own**. Done/dead checks are in
the table above. Three watchers were live: R8 training, R9 eval chain, B1.

**Watcher trap, learned the hard way:** `tabicl_strong_compare.py` rewrites its
`--output-json` after EVERY cell, so the file existing means "at least one cell
done", not "finished". A watcher keyed on file existence fired on cell 1 of 15 and
reported B1 complete when it had ~7 hours left. The real signal is the script's
final `INFO wrote <path>` log line. Each cell costs ~30 minutes of scoring
(measured 1,826s), matching the Aug 27 run, so 15 cells is ~7.5h.

**The ultra host is billing.** Stop it (`gcloud compute instances stop
odyssey-cbm-a100-ultra --zone us-central1-a`) once B1's JSON is pulled. Amrit
approved the spin-up explicitly for B1 and nothing else.

---

## 1b. READ THIS FIRST — the calibrated-mode bug (FIXED AND VERIFIED IN PRODUCTION)

**Every calibrated intervention mode had never once completed on a GPU.**
`calibration_gammas` is built from the model's LM-head weights so it lives on the
model's device; the running labels are assembled on CPU; the offsets multiply cast
dtype but **not** device. Every GPU run of `truth_calibrated`/`flip_calibrated`
died with "Expected all tensors to be on the same device". The CPU-only test suite
could not see it, so the calibrated-mode tests passed the whole time.

Fixed in **414cc0d** (plus **b12a138**, a type annotation for CI's torch stubs),
regression tests in **8aa24e5** (one CUDA-gated, which will run on the VMs; one
dtype test that runs anywhere). **The CUDA test still has not been executed on a
GPU** — all three cards have been busy — so run
`pytest tests/odyssey/inference/test_interventions.py` on a VM once one frees.

**VERIFIED IN PRODUCTION 20:31 UTC.** R9's re-run completed both calibrated modes
with zero device errors, so this is closed, not merely believed fixed. It also
reproduced `none`/`truth`/`flip`/`flip_gated` digit-for-digit against the crashed
attempt, which is a clean determinism check on the re-run.

Consequences you must handle:

- **R9's interventions stage EXIT 1** (18:38:54). It got through `none`, `truth`,
  `flip`, `flip_gated`, then crashed on the first calibrated mode, so
  `interventions_band15.json` and the per-subject dump were never written.
  **R9's interventions stage must be re-run** once VM2's GPU is free (its chain
  carried on to attribution, which passed EXIT 0, and then alerts). Checked
  19:39 UTC: the run dir holds only `interventions.log`, no JSON and no
  per-subject dump, so the append-only guard will NOT block the re-run and no
  `--overwrite` is needed. Re-run with VM2 reset to >=414cc0d.
- **Before launching R8's eval chain, make sure VM1 is reset to ≥414cc0d**, or it
  will fail in exactly the same place.
- Both `supplemental_r2_vm1.sh` and `supplemental_r6_vm2.sh` score the calibrated
  modes, so **both would have died too** — and they are the only source of
  `intervention_cis.json`, which gates Figure 3's CIs and the frozen abstract
  decision (§6). This was on the critical path and nobody had noticed.

No published number changes: nothing had ever been produced by these modes.

**One real result did survive R9's partial run**, from the log before the crash:
`none` 0.5551, `truth` 0.5113 (−0.0438), `flip` 0.4985 (−0.0566). So truth−flip =
**+0.0128, correctly signed** — the L-series sign correction reproduces at full
scale on eICU, while every mode still sits well below `none`. That is exactly the
subset-L1 pattern (sign fixed, capability cost) and it is the intervention half of
the L-series question. Treat as provisional until the stage is re-run cleanly.

---

## 1bb. THE PAPER'S INTERVENTION NUMBERS ARE PRE-REFACTOR — RE-SWAP OWED

Found 20:51 UTC. **Both flagship eval chains ran at `cdbd4e7`**, confirmed from
each chain's own logged `commit=` line, not just the registry. The W8-W10 refactor
(`e7cbbe7`, 07:42) and W7 (`8161e19`, 07:56) landed after, and that refactor
restructured the streaming intervention loop. **Current code does not reproduce
the chain's numbers.** Re-scoring R6's own `checkpoint_best.pt`, same 4 shards,
same band 0.15, same lanes/chunk, same data: `none` gives top-1 **0.5450** / loss
**1.9562** where the chain recorded **0.54008** / **1.97496**.

That is code, not noise. These modes are deterministic, and the control is R9,
whose re-run reproduced its own numbers digit-for-digit *within* one code version.

**Affected:** every intervention cell swapped into the paper today — `tab:decomp`'s
joint MIMIC and joint eICU rows, the Sec 5.3 prose quoting them, and Figure 3
panel (a), which reads `interventions_band15.json`.
**Not affected — VERIFIED, not assumed:** `git log cdbd4e7..HEAD` over
`alerts.py`, `run_inference.py`, `baseline_features.py` and `baseline_prep.py`
returns **nothing**, so the comparator tables, the concept readout and the alert
numbers are code-stable since the chains ran. The only concept/label-path commit
is `36e1055` (GEMINI per-source prefixes), separately confirmed a no-op for
mimic_iv/eicu. R9's L-series numbers are also unaffected (its chain and re-run are
both post-refactor). The blast radius is interventions only.

**Resolution, already in motion and cheap.** The supplemental scripts re-score each
flagship checkpoint at current code with a superset of modes *and* dump per-subject
counts, so one run gives the point estimates and their paired CIs together. Take
the eICU cells from `eicu_full_v10/interventions_guidelabs.json` (running since
20:41) and the MIMIC cells from `full_run_v10`'s twin when VM1 frees, then re-swap
both rows from those. **Do not mix**: a CI from the supplemental next to a point
estimate from the chain is exactly the cross-provenance error the TabICL block
refuses.

**Severity, now measured (R6's first three modes): the refactor moved the ABSOLUTE
level and left the DELTAS alone** — and every claim in this paper is about deltas.

| | chain (in the paper) | current code |
|---|---|---|
| none top-1 / loss | 0.54008 / 1.97496 | 0.5450 / 1.9562 |
| truth Δtop-1 / Δloss | +0.19 pts / +0.0235 | +0.16 pts / +0.0235 |
| flip Δtop-1 / Δloss | −0.38 pts / +0.0269 | −0.37 pts / +0.0267 |
| truth − flip | +0.57 pts | +0.53 pts |

All three eICU claims hold under current code: truth helps on top-1, truth hurts
on loss (the metric disagreement), truth beats flip on loss. The truth loss delta
is +0.0235 either way, i.e. **+0.023** — which independently re-confirms this
morning's `tab:decomp` correction from +0.024.

So treat the re-swap as a **consistency refresh**, so point estimates and CIs come
from one code version — not as a claim risk. MIMIC still to check.

---

## 1c. NEW RESULTS from this session (register/verify, don't re-derive)

**R9's intervention numbers — the L-series question is answered on eICU.** With
`concept_global_pairs=True`, the lever is CORRECTLY SIGNED under both protocols
and against the random control, while every mode stays below no-intervention:

| mode | top-1 | Δ vs none | loss |
|---|---|---|---|
| none | 0.5551 | — | 2.0828 |
| truth | 0.5113 | −0.0438 | 2.3928 |
| random | 0.4996 | −0.0555 | 2.4364 |
| flip | 0.4985 | −0.0566 | 2.4176 |
| flip_gated | 0.5450 | −0.0101 | 2.1788 |
| truth_calibrated | 0.5327 | −0.0223 | 2.2172 |
| flip_calibrated | 0.5260 | −0.0291 | 2.2232 |

Reading: truth > random > flip is the ordering a working lever should give, and it
holds on the banded protocol (+0.0128 truth−flip) AND on Guide Labs'
output-calibrated protocol with no band at all (+0.0067, and truth also better on
loss). That the sign survives a change of protocol is much stronger than the
banded result alone — W7 built the calibrated protocol precisely because the band
was suspected of doing the work. It isn't. Compare the flagship MIMIC arm (R2),
where the same ordering is INVERTED with truth worst: that contrast is the paper's
headline. Also note `flip_gated` costs only −0.0101 against flip's −0.0566, i.e.
gating recovers most of flip's damage (the Guide Labs Fig 19 artifact check),
suggesting much of an override's damage is logit-magnitude disruption.
zero_known/zero_unknown were still running at handoff; get them from
`interventions_band15.json`.

**B1 has already confirmed the TabICL claim reversal on v4 rows, same-dump.** Its
4th cell, `icu_admission@8h`: GBM 0.9743, **TabICL 0.9642, our hazard head 0.9353**.
So TabICL beats our own model by +0.029 there and trails the tuned GBM by only
0.010. This matters beyond the number: the Aug 27 evidence for the reversal came
from a DIFFERENT dump than the paper's, which is why I refused to splice it in
(see the TABICL block in the main.tex header). B1 is scoring against R2's own v4
dump, so when it finishes there is no cross-dump caveat left and Sec 5.4's "TabICL
loses 12 of 12 cells" can be rewritten from a single consistent source.

**R9's finish chain COMPLETED 20:41 UTC, all three stages EXIT 0**, producing the
**first paired intervention CIs in this project** (`intervention_cis.json`, 2000
boots, 3,916 subjects). Aggregates are already local in
`research_journal/figure_data/vm2/eicu_full_L_v10/`. **All ten pairs SEPARATE:**

| pair | delta (pts) | 95% CI |
|---|---|---|
| truth − flip | **+1.279** | [+0.863, +1.672] |
| truth_calibrated − flip_calibrated | **+0.677** | [+0.293, +1.041] |
| truth − none | −4.379 | [−4.604, −4.172] |
| flip − none | −5.658 | [−5.932, −5.361] |
| flip_gated − none | −1.010 | [−1.117, −0.904] |
| random − none | −5.549 | [−5.686, −5.415] |
| zero_known − none | −15.353 | [−15.683, −15.017] |
| zero_unknown − none | −26.077 | [−26.556, −25.566] |

So the sign correction is significant under the banded protocol AND under the
output-calibrated one that removes the band entirely — the band was not doing the
work, which is exactly what W7 was built to test — while every mode stays
significantly below none.

**COMPLETENESS REVERSAL, the finding most worth a second pair of eyes.** The 2x2
is now complete: zero_known retains 72.3% (R6: 48.5%), zero_unknown retains 53.0%
(R6: 100.7%). In the global-pairs arm the **unnamed residual is MORE load-bearing
than the named channel**, the opposite of the flagship pattern, which is awkward
for an ablation whose purpose was to make concepts carry meaning. Natural reading:
it fixes the lever partly by making the model depend on named concepts less
overall. **One seed, cross-run — a hypothesis, not a finding.** Needs R8 and
replicates. Do not let this into the paper as a claim without them.

**The capability cost of global pairs is broad, not just forecasting** (registered
in docs/experiments.md, R9 row). R9 vs R6 on eICU: set top-1 −1.46, top-5 −2.99,
xent +0.126, **concept readout mean AUROC −0.023**, and hazard alerts worse on
**9 of 12 cells** (mean −0.016, worst on AKI and icu_admission; the three death
cells are the only gains). Top-1 alone improves (+1.19) — the same metric
disagreement this arm shows on interventions. R8 looks like it will repeat the
pattern on MIMIC: best_val 2.3235 vs R2's 2.0556 as of 20:27.
**This is a cross-run comparison with one seed per arm, so it is a labeled
hypothesis under the stats policy, NOT a CI-backed claim.** Do not upgrade it
without seed replicates.

---

## 2. Queue, in order

1. **R9 finish chain completes (~20:55)** → pull `interventions_band15.json` and
   `intervention_cis.json` into `research_journal/figure_data/vm2/eicu_full_L_v10/`
   (aggregates only), register the zero_known/zero_unknown cells, then on VM2:
   - `scripts/intervention_cis.py --per-subject ~/runs/eicu_full_L_v10/interventions_band15_per_subject.json --output-json ~/runs/eicu_full_L_v10/intervention_cis.json`
     **The eval chain does NOT do this.** The previous handoff said the chain
     "auto-produces everything"; that is wrong for the CI step.
   - `scripts/vm_oneoff/readmission_alerts.sh ~/runs/eicu_full_L_v10 ~/data/eicu_2.0_v2/data`
   - `scripts/vm_oneoff/supplemental_r6_vm2.sh` (R6 checkpoint's Guide Labs
     modes, W3 band sweep, per-subject dumps, `intervention_cis.json`)
   - `scripts/alerts_cis.py` over the v4 dumps (pure CPU, no GPU contention)
2. **R8 training finishes** → update VM1 checkout (`git fetch origin` +
   `merge-base --is-ancestor` floor check + `git reset --hard origin/main`;
   **fetch/reset only, NEVER `uv sync` on a VM**), then
   `STREAM_BASELINE=1 LANES=64 setsid nohup scripts/eval_run.sh ~/runs/full_run_L_v10 ~/data/mimiciv_3.1_v1/data ... & disown`,
   then the same four follow-ups with `supplemental_r2_vm1.sh`.
3. **Comparator table swaps** (see §4 — this is the biggest outstanding paper debt).
4. **Figure 3 CIs**: copy `intervention_cis.json` into
   `research_journal/figure_data/vm1/full_run_v10/` and rerun
   `paper/ml4h/figures/make_figures.py`. Error bars appear automatically.
5. **B2** eICU baselines; **R3/R7** independent arms and **R4** (R2's
   intermediate-checkpoint sweep) wait on Amrit's D2/D3.

---

## 3. What I found and changed (the part that matters most)

Amrit spent the session re-reading the paper and asking four questions. All four
had the same root cause: **the comparator tables and Figure 4 are prior-generation
and understate or misstate baselines, while corrected data already existed.**

- **The AKI numbers in the paper predate the KDIGO staging fix.** R2's post-fix v4
  values are much higher for *both* scorers: at 8h hazard 0.831 → **0.870**, GBM
  0.893 → **0.913**; 24h 0.773 → 0.821 / 0.845 → 0.878; 72h 0.725 → 0.771 /
  0.783 → 0.828. The GBM still leads AKI and the gap narrows slightly
  (0.062 → 0.043 at 8h), so the qualitative claim survives but every printed
  number is wrong.
- **TabICL is reported at a config we know handicaps it.** The paper shows
  `TabICL(basic)`, 17 features, reduced capacity, beside a 609-feature GBM. At
  full capability it ties the tuned GBM on **9 of 12 cells** and on ICU admission
  **beats our own hazard heads at every horizon** (0.963/0.940/0.914 vs
  0.934/0.893/0.837). So **Sec 5.4's "TabICL loses 12 of 12 cells" is
  known-false** and is marked do-not-submit inline and in the header. Amrit ruled:
  drop the basic row, report full capability. That is what B1 is computing.
- **Both comparator tables were never swapped to v4.** `tab:eicu` still holds
  subset-scale cells (which also violates the contract's "no subset numbers
  anywhere"); `tab:mimic` is still `[PRIOR-GEN, protocol v3]`. R2 and R6's v4
  replacements have been in `research_journal/figure_data/` since this morning.
- **Missing tasks.** sepsis3 is already computed for MIMIC at full scale under v4
  and going unused (hazard 0.912/0.891/0.851 vs GBM 0.943/0.928/0.906, GBM wins
  all three) — free to add. sepsis3 is structurally unavailable on eICU (dropped
  by source). `readmission_30d` was scored by **nothing, on any dataset**, because
  it is a `next_visit` event needing `--index-mode visit_end` at 168/720h and the
  eval chain only ever runs the landmark grid at 8/24/72h. Amrit asked for it on
  all three datasets; runner committed (§5).

Also fixed, unprompted, because they were latent traps:

- **Figure 3 would have crashed at the last step before submission** (`9a6afcf`).
  `intervention_cis.py` emits a pair only when *both* modes are in the per-subject
  dump; the supplemental scripts scored no `random`; `fig_lever` indexes
  `random_minus_none` unconditionally. Both scripts now also score
  `random`/`zero_known`/`zero_unknown`, and `fig_lever` degrades loudly.
- **A wrong number in `tab:decomp`**: joint-eICU loss Δtruth read +0.024, the raw
  value is 0.02348885 → **+0.023**. Fixed in the table and the Sec 5.3 sentence;
  odyssey-6b independently re-derived and confirmed.
- **Both flagship arms' aggregates had never been pulled off the VMs.**
  `vm2/eicu_full_v10/` did not exist locally at all. Now local with manifest rows,
  so numbers can be re-derived from source without SSH.
- **229 GB freed on VM1** (85% → 26%) by deleting the MEDS-Tab tabularization
  tree, at Amrit's instruction. The four result JSONs behind that registry row
  existed **only on the VM** and were copied off and verified first.

---

## 4. The comparator table swap — the main outstanding paper debt

Both tables in `app:tables` need replacing with v4 data. Inputs are ready except
TabICL (B1, in flight):

- MIMIC: `research_journal/figure_data/vm1/full_run_v10/alerts.json` (15 cells:
  the four core events **plus sepsis3**, all protocol v4, post-KDIGO).
- eICU: `research_journal/figure_data/vm2/eicu_full_v10/alerts.json` (12 cells;
  no sepsis3 by source).
- TabICL column: B1's `tabicl_strong_v4.json`.
- Bold rule: both captions promise bold requires the **paired subject-clustered
  bootstrap** to separate the cell, plus AUPRC and calibration columns. That needs
  `scripts/alerts_cis.py` over each arm's `alerts_rows.parquet`. Pure CPU.

**The v4 claim check is already done** (full detail in the main.tex header). Every
Sec 5.4 comparator claim was re-derived from the post-KDIGO v4 data before the
swap, so this is a numbers edit and not a rewrite: the "win 2 of 12 cells" claim
survives on the *same two cells* and becomes 2 of 15 (sepsis3 adds three, all GBM
wins); "the gap widening in horizon" survives and is now uniform across all five
events, AKI included, where at v3 AKI was the flat one; and eICU's "lose all 12"
survives exactly. One caution recorded there: the older "AKI stays flat" reading,
used as evidence that AKI is a different failure mode from the counting failures,
no longer holds at v4 -- do not re-assert it from the old rows.

**Use `scripts/make_comparator_tables.py` (new, tested) -- do not hand-transcribe.**
It reads `alerts.json` (+ optional `--tabicl` and `--cis`) and emits the LaTeX row
body. Hand transcription is what produced the `+0.024` error, so the tables are now
a build product. Missing inputs become `% WARNING` comments inside the generated
`.tex` rather than a silently short table, and bold is withheld entirely when CIs
are present but the paired delta does not separate. Verified on both arms today:
MIMIC 15 cells / 5 events (sepsis3 included), eICU 12 cells / 4 events.

**RESOLVED by Amrit 2026-08-31: "skip ebm and survivalpfn, ship
hazard/gbm/tabicl".** The v4 tables get three scorer columns. EBM comes out of
tab:mimic; SurvivalPFN was already omitted by its own caption. Coupled consequence
for the D5 pass: "five tuned baseline families" appears in four places including
the abstract, and only GBM and TabICL now appear in a v4 table. Details below.

**Context -- the v4 alerts contain only hazard and GBM.** Both `alerts.json` files
carry scorers `hazard`, `baseline_gbm`, `concept`, `next_mass` and **no EBM, no
TabICL, no SurvivalPFN, no MEDS-Tab**. B1's registry scope says "TabICL/EBM/
SurvivalPFN rescores on R2 rows" but **only TabICL is running**. So a table swap
done today would silently DROP the EBM column the paper currently shows. EBM and
SurvivalPFN need `scripts/rescore_extra_baselines.py` against each arm's
`alerts_rows.parquet` before either table is complete. Do that before swapping,
or the swap trades one stale table for one missing a baseline.

**Do not splice the Aug 27 TabICL numbers in** (`docs/tabicl_strong_feature_comparison.md`).
Three reasons, all in the main.tex header: its GBM came from a different dump than
the paper's (identical rows, but the refits disagree — death 8h 0.953 vs 0.949, so
combining them presents a cross-dump comparison as within-dump); it is protocol v3;
and its AKI cells predate the KDIGO fix, which matters because AKI is the *only*
cell where TabICL still loses. B1 on v4 rows fixes all three at once.

---

## 5. Code I changed this session

| Commit | What |
|---|---|
| `9a6afcf` | supplemental scripts score `random`/`zero_*`; `fig_lever` no longer crashes on a missing CI pair |
| `a67642b` | D5: abstract frozen (see §6) |
| `35d1de5` | MEDS-Tab deletion recorded, results preserved first |
| `d77c08d` | reworded a scale-related compound word that the typos hook flags as a typo (`docs/experiment_plan.md` is not in the hook's exclude list) |
| `66247f6` | B1 extended: TabICL must be strong-panel full capability, needs ultra host |
| `656062b` | `tabicl_strong_compare.py` scores the run's own task-set events (adds sepsis3; v1 runs unchanged; `next_visit` events explicitly excluded) |
| `b99be32` | **new** `scripts/vm_oneoff/readmission_alerts.sh` |
| `f79e8e3` | R9 registered |
| `635888f` | `tabicl_strong_compare`: thread `task_set` into `prepare_baseline_data` |
| `6a50ecc` | `tabicl_strong_compare`: call `activate_sidecars` |
| `414cc0d` | **device fix**: calibrated intervention modes had never run on a GPU |
| `8aa24e5` | CUDA-gated + dtype regression tests for the above |
| `6984d39` | **new** `scripts/make_comparator_tables.py` + 12 tests: generate table bodies from run JSONs |
| `3c0569c` | generator warns on PARTIAL inputs, not just absent ones |
| `33e018a` | R9 registered incl. the measured capability cost of global pairs |
| `d282e4f` | **new** `scripts/vm_oneoff/post_chain.sh`: the two steps `eval_run.sh` omits |
| `4aa6662` | readmission unblocked on GEMINI; verified 36e1055 is a no-op for MIMIC/eICU |

**B1 needs `ODYSSEY_TABICL_MEMORY_BUDGET_GB=120` in its environment.** The cost
guard in `tabicl_baseline.py` defaults to a 16 GB budget and refuses the
strong-panel fit at fit time (it estimates ~70 GB per `predict_proba` call:
50,000 context rows x 609 features x 8 estimators). That guard is right on an
82 GB host and wrong on this one, which has 165 GB free and measured ~102 GB peak
RSS on the Aug 27 run. 120 permits the real fit while still refusing anything that
would not fit. If you relaunch B1, **you must set this var or it dies in seconds**.
The failed attempt is kept at `~/b1_tabicl_budgetfail.log`.

The last two commits are worth understanding, because they are the same class of
bug and B1 hit both within 25 seconds of launch. `prepare_baseline_data` has its **own**
`task_set` parameter defaulting to `"v1"`, so resolving sepsis3 into the alert list
without threading `task_set` through produced a missing-column crash. Then, with
that fixed, sepsis3 depends on the **microbiology sidecar**, and the script never
called `activate_sidecars` (its original four events never needed one) — which
does *not* crash: the concept goes unobserved everywhere, its rows collapse to
zero, and the empty-rows guard silently skips the event. That would have been a
7-hour run that quietly dropped the task it was extended to add. **If you extend
any scoring script to a new event, check both.**

Gates before every push: `ruff format`, `ruff check`, `mypy odyssey scripts`,
`pytest`. All green at `d282e4f` (1319 passed, 7 skipped).

**Other sessions share this working tree.** odyssey-4a edited
`odyssey/data/alert_events.py` in it while I was working, which showed up as an
uncommitted modification that was not mine. Stage specific paths, never
`git add -A`, or you will sweep up someone else's work in progress. Also never
touch the pre-existing uncommitted `scripts/gemini/out/*.json`.

---

## 6. Frozen / blocked — do not act unilaterally

- **The abstract is FROZEN** until `intervention_cis.json` exists (D5, Amrit:
  "let CI land, then discuss with odyssey-cf"; that lane is the paper-reviewer
  session, `odyssey-6b` this generation). Three coupled questions to settle in
  **one** pass: whether the eval-audit sentence earns its space; whether the
  closing "tested by intervention, not by readout accuracy" keeps the last line
  (it is **not** a novel claim, and our own Related Work concedes it); and — the
  reason they are coupled — if the paired CI on eICU truth−none (+0.19 on top-1)
  clears significance, "the lever never helps" needs a decision, not a wording
  tweak. Full assessment in the main.tex header. Caution: the current close was a
  *deliberate* prior choice by that same lane.
- **GEMINI: stay out.** That channel is Amrit + odyssey-6b only. Never fetch,
  poll or push the gemini remote. Your only touchpoints are code changes 6b
  requests, and running `scripts/gemini_concept_audit.py` when 6b pings that
  `codes_inventory.json` is back (currently 15/29 concepts resolve). Amrit asked
  for readmission on all three datasets — MIMIC and eICU are yours, GEMINI is
  **not**; hand the runner to that lane.
  **UPDATE 2026-08-31 ~20:00: readmission is no longer GEMINI-blocked.** odyssey-4a
  landed per-source alert prefixes (36e1055) that remap `readmission_30d` to
  GEMINI's bare `ADMISSION` token while keeping `next_visit` semantics, so the
  event is scoreable there once a GEMINI bottleneck checkpoint exists (that commit
  also adds the train-*-cbm steps). `scripts/vm_oneoff/readmission_alerts.sh` is
  parameterised by run dir and data root and should work unchanged. Still 6b's and
  Amrit's lane to run, not this one.
  That commit touches `odyssey/data/alert_events.py`, which R8/R9 depend on. It is
  a **verified no-op** for mimic_iv/eicu/source=None: I re-derived the event lists
  across v1/v2/v3 after it landed and they are identical to before, and the full
  suite passes at 1,316. So R8 and R9 stay comparable with R2/R6 and no
  label-definition registry note is needed.
- Open decisions owed to Amrit: **D2** (M-series in/out), **D3** (GEMINI scope),
  the **GEMINI REB reference** (BLOCKING-VERIFY in the IRB paragraph), and whether
  to add concept-probability calibration (ECE) to the eval.

---

## 7. Standing rules (learned the hard way, several this session)

- One GPU job per card. `setsid nohup ... & disown` for long jobs.
- **Never `uv sync` on a VM.** Update checkouts with fetch + `reset --hard` only,
  after a `git merge-base --is-ancestor <floor> origin/main` check — **floors, not
  recency**. Smoke-test imports after a reset.
- **Patient-level dumps never leave the VMs.** `alerts_rows.parquet`,
  `case_studies.json`, `*_per_subject.json`. Aggregates travel and get a
  `research_journal/figure_data/manifest.md` row. Moving R2's dump to the ultra
  host went VM→VM over the internal network (a scoped key, `~/.ssh/vm2vm_tabicl`
  on VM1, authorized on the ultra host — **remove it when B1 is done**), verified
  by md5 on both ends. Never route such a file through the laptop or a bucket.
- Science outputs are append-only. Never `--overwrite` without cause; a real
  irreplaceable row dump was lost to a silent overwrite on 2026-08-22.
- **Never touch the pre-existing uncommitted `scripts/gemini/out/*.json`.**
- `pgrep`/`pkill -f` over SSH will match your own command line. I killed my own
  session that way. Use the `[t]ool` bracket trick, and make sure the *whole*
  remote command string cannot self-match.
- Journal/paper prose style: no em dashes, no fluff, every claim carries its
  source, hypotheses labeled and kept separate from findings.
- Verify numbers against the raw JSON, not against another prose summary. Every
  error found this session was found that way.

---

## 8. Peer sessions

- **`odyssey-6b`** — paper reviewer and GEMINI-channel owner. Verifies every paper
  change independently and does the abstract-vs-body consistency check after each
  swap; it has caught real contradictions. It re-derives from raw source by
  default now that the JSONs are local. Coordinate via `SendMessage`; ask before
  editing `main.tex` if unsure whether it holds the file. Name changes on
  reconnect — find it with `ListAgents`.
- **`Odyssey ML4H paper push`** (odyssey-4a, background) — wrote the previous
  handoff, still alive. I asked it to stand down its pollers to avoid duplication.
