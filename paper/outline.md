# Paper outline — npj Digital Medicine (Article)

Drafted against `writing_guide.md` (section order, Results conventions, claim
calibration, TRIPOD+AI anchors). Status tags: **[HAVE]** result in hand (see
`research_journal/figure_data/manifest.md`), **[WAVE]** lands with the v2
re-evaluation wave, **[GEMINI]** lands with GEMINI training/external validation,
**[QUEUED]** experiment approved and scheduled. Amrit rewrites all prose for
voice; this is structure and content allocation.

Working title candidates: see the comment block above `\title` in `main.tex`
(decision deferred to GEMINI results, per framing gates).

---

## Abstract (~150 words, one paragraph)

Background (one sentence: bespoke single-task alerting doesn't compose) → gap
(sequence models judged on next-token accuracy, not as calibrated alert systems)
→ what we did (one general timeline model + four comparator families, N tasks ×
3 horizons, three health systems incl. external validation on a ~30-hospital
network; 1.2B + 0.7B + 0.4B events) → two results sentences with real numbers
(comparator trade-off headline per Hegselmann precedent; external-validation
number per framing gate 1) → hedged implication ("supports the potential of…",
never "validated/deployment-ready").

## Introduction (no subheadings, 4–6 paragraphs)

1. Bespoke clinical prediction models: one model per task per site; alert
   portfolios don't compose; cite representative sepsis/deterioration/AKI work.
2. Sequence models over EHR event streams. **Open lineage with Doctor AI**
   (choi2016doctorai; see drafting notes in main.tex): posed next-event +
   time-to-next-visit a decade ago; its own reported failure (time R²=0.25) is
   the direct motivation for our survival-native formulation. Through
   BEHRT/ETHOS-class models to today: mostly judged on token accuracy or
   fine-tuned per task, not run as calibrated alerting systems.
3. What a clinician needs from an alert: event-specific risk over a horizon,
   calibrated, updating as the record evolves, with something inspectable.
4. This work: one MTPP-over-bundles model of the whole timeline (hazard heads →
   survival curves; concept bottleneck for named states), evaluated as
   forecaster and as alert system against the strongest specialized
   alternatives (tuned GBM, tabular ICL-FM, additive models, survival-native
   FM, MEDS-Tab) on identical landmark rows; pipeline unchanged across
   MIMIC-IV, eICU-CRD, GEMINI; generalization assessed ONLY on GEMINI
   (state the MIMIC↔eICU non-transfer decision here or in Methods, citing the
   Tranchellini overlap-audit precedent).
5. Contributions paragraph (numbered in prose): the comparative evidence, the
   protocol (MEDS narrow waist + landmark protocol + reproducibility
   machinery), the external validation, the honest interpretability account.

## Results (one subsection per question; declarative headings for headlines)

Roadmap sentence first (Tranchellini device), mirroring the contributions.

### R1. Cohorts and the forecasting task [HAVE]
Three systems table (subjects/events/sites/span); MTPP-over-bundles task
definition; landmark-alert task suite (events × horizons); Figure 1 pointer.
Methods-forward content: what was measured and why; parameters live in Methods.

### R2. One model forecasts every event family and improves monotonically with scale [HAVE]
Scaling curve (set top-1 74.6 → 81.0 across epochs 1–2; per-family curves;
alert-gap trend under v2 when wave lands). The "curve vs point" argument
against static tabular baselines. Data: lc-eval series (manifest, Forecasting
theme).

### R3. Tuned GBMs remain the bar on fixed-horizon alerts; the gap closes with scale and inverts where timelines matter [WAVE]
The comparator grid: 5 families × 12 event-horizon pairs × 2 datasets under
protocol v2 (numbers in a table/summary figure; prose carries the comparative
story only, per Guo convention). Honest scoreboard incl. where we lose;
TabICL's targeted wins as probes; EBM/SurvivalPFN not competitive (their
results reported plainly); MEDS-Tab as field-standard external baseline.
Conservative-comparison note (Guo device): comparators get their native tuning
protocols; search-depth parity documented.

### R4. Alerts are survival curves: calibration and time-resolved risk [HAVE, v2 refresh in WAVE]
Calibration as first-class (per-horizon reliability; Brier/AUPRC to
supplementary); survival-curve panel for a worked patient; time-before-event
detection vs false-alarm panel (Xie precedent) if the analysis lands in time.

### R5. External validation on a provincial hospital network [GEMINI]
The framing-gate-1 result: frozen models from MIMIC/eICU evaluated on GEMINI
(zero-shot transfer + few-shot/refit protocol); GEMINI-trained reference;
hospital-holdout variation figure (30-site forest/distribution — generalization
shown, not buried, per Lee precedent); temporal-cutoff validation. THE section
the title decision waits on.

### R6. Robustness under missingness and staleness [QUEUED — stress protocol]
Degradation curves (event-level dropout, family blackout, lab-turnaround lag)
for all families on identical degraded records; recency-channel story told
honestly across datasets (eICU gain [HAVE], MIMIC non-replication as candidate
finding pending seed replicate [QUEUED]; dataset-conditional framing if
confirmed).

### R7. Named physiological states: a faithful readout, and what interventions actually test [HAVE; L-series completion QUEUED]
Concept AUROC table (readout faithful); the lever story with full honesty:
correlational-not-causal finding (surprise-proportional interventions,
base-rate correlation 0.97), independent-training frontier (M-series,
six-mode banded protocol; M3b working point: 77% of task gap recovered with
correctly-signed lever), cost quantified. Xie causal-bounding sentence as the
closing register. This section feeds the Track B phase; scope grows if the
causality push lands new results before submission.

## Discussion (no subheadings; ends "In conclusion,")

- Headline as trade-off (Hegselmann device), not victory.
- What the probes taught (staleness→recency; degenerate sub-tasks; where
  timeline scope matters vs where tabular suffices) — the honest possibility
  of a hybrid end-state.
- Interpretability: readout ≠ lever ≠ causality; what the framing gates would
  require before landmark claims.
- Limitations paragraph (enumerated, each flaw + its effect on the claim):
  family-confound sentence (Hegselmann verbatim device); MIMIC↔eICU
  non-transfer + overlap-audit precedent; retrospective only; single-seed
  hedges where applicable (recency non-replication); GCS/site coverage gaps;
  charted-time vs available-time optimism (all families share it); label-rule
  imperfection (rule-derived concepts).
- "Still needed before deployment" statement (TRIPOD 27c): prospective
  validation, calibration monitoring, per-site checks — explicitly not claimed.
- In conclusion: one paragraph, scoped claim + the curve argument + the open
  causal question.

## Methods (bold subheadings; everything reproducible; TRIPOD+AI cited)

1. Data sources and ethics — per-source IRB/REB sentences (MIMIC BIDMC/MIT;
   eICU exemption; GEMINI REB + DSA); TRIPOD+AI conformance sentence.
2. MEDS convergence layer — extractors per source (incl. GEMINI SQL extractor,
   guards, unit canonicalization, provider preservation), conformance
   validation, split construction (seeded 80/10/10; GEMINI split rule +
   derivable hospital-holdout/temporal protocols).
3. Tokenization and inputs — value binning, static facts, time encoding,
   recency channel (opt-in, per-source).
4. Model families (TRIPOD 12a–c per family): hybrid Mamba-2+attention (+
   transformer control), concept bottleneck (CBGM basis, leakage controls,
   independent training); GBM panel + tuning; TabICL; EBM; SurvivalPFN;
   MEDS-Tab native pipeline (windows/aggs, 200-trial budget rationale).
5. Training — streaming TBPTT / packed-context regimes, schedules,
   checkpointing/selection (best-vs-final finding), reproducibility machinery
   (env fingerprints, per-checkpoint canaries, landmark protocol versioning —
   the v1→v2 story stated plainly).
6. Alert evaluation protocol — landmark grid, at-risk filtering, per-horizon
   masks, identical-rows discipline, v2 protocol, statistical procedures
   (CIs/tests; DeLong or bootstrap — decide once, Yoon formatting convention).
7. Interpretability evaluation — concept scoring, six-mode banded intervention
   suite, displacement matching.
8. Missingness stress protocol.
9. Use of AI tools — per revised guide §2.2: Amber-tier disclosure by default
   (agent proposed analyses/approaches; human direction, selection,
   verification, accountability), audit-trail sentence, Red-tier disclaimed.
10. (Compute statement; software versions.)

## Data availability
Per-source, real mechanisms: PhysioNet credentialing (MIMIC-IV, eICU with
URLs); GEMINI governed access via [actual mechanism]. No "reasonable request"
platitudes.

## Code availability
GitHub repo URL; states what reproduces what (incl. extraction + analysis
code); versions pinned via lockfile.

## Figure plan (main; rest → single Supplementary PDF)

- **F1** Study design: three systems → MEDS narrow waist → one pipeline →
  landmark evaluation; cohort table inset. [HAVE materials]
- **F2** Architecture: timeline → backbone (hybrid; transformer control noted)
  → concept bottleneck → heads (survival curves out). [HAVE]
- **F3** Scaling: learning curves + alert-gap-vs-scale. [HAVE, v2 refresh]
- **F4** Comparator grid summary (families × tasks × datasets, significance
  marks; Tranchellini ranking-figure device). [WAVE]
- **F5** Calibration + survival-curve panels. [HAVE/WAVE]
- **F6** GEMINI external validation + 30-site variation. [GEMINI]
- **F7** Interpretability: concept readout + intervention frontier (M-series
  cost-vs-lever; six-mode when L-series completes). [HAVE + QUEUED]
- (F8 if R6 lands strongly: missingness degradation curves.) [QUEUED]
- Supplementary: per-task tables, AUPRC/Brier, per-family forecasting detail,
  ablations (bottleneck price, no-bottleneck baseline), protocol v1/v2 deltas,
  TRIPOD+AI checklist, tail-slice analyses.

## Reference budget
55–75. Anchors: Doctor AI lineage + test-of-time note; CBGM; TRIPOD+AI;
comparator-family sources; MEDS ecosystem; the seven style-precedent papers
where topically warranted (Tranchellini for overlap/transfer, Guo for
conservative-comparison, Hegselmann for family-confound).
