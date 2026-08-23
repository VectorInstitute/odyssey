# Missingness stress protocol (design; runs gated on wave close)

Paper section R6. Question: **when the record degrades the way real
deployments degrade, which model family loses accuracy and calibration
slowest?** If the timeline model is more robust and better calibrated than
the GBMs under degradation, that is a deployment-relevant differentiator
the comparator grid alone cannot show.

## Principles

1. **Frozen models, degraded inputs.** No retraining. We measure how each
   already-trained family behaves when observation quality drops --- the
   real-time deployment concern (sensors unplugged, labs delayed, feeds
   down), not the train-on-degraded-data question.
2. **Identical degraded records for every family.** Degradation is applied
   once, at the MEDS row level, before any family-specific feature
   construction or tokenization; every family scores the same degraded
   copies. Same discipline as the identical-landmark-rows rule.
3. **Labels and landmark rows stay clean.** Outcomes and the landmark grid
   are defined on the un-degraded record (ground truth happened regardless
   of what was observed). Degradation touches only what the model sees as
   input at scoring time. This keeps the row set identical across every
   degradation cell, so AUROC/calibration deltas are attributable to input
   degradation alone. Protocol v3 rows, stamp ritual applies.
4. **Seeded and reproducible.** Each degraded shard copy is derived by a
   deterministic seeded transform of the held-out shards; the transform,
   seed, and cell parameters land in the artifact's metadata.

## Degradation axes and cells

Three axes matching the three real failure modes; kept to a small grid
(8 cells + clean baseline) so the full family sweep stays affordable.

- **A. Random event dropout (MCAR):** drop each non-anchor event row
  independently with p in {0.1, 0.3, 0.5}. (Anchor rows -- admission /
  discharge / demographic statics -- are never dropped; a record with no
  visit envelope is not a degraded record, it is a different task.)
- **B. Family blackout (structured):** remove ALL events of one family:
  {labs}, {vitals/charting}, {medications}. One cell each. Models that
  lean on a single family reveal it here; the concept bottleneck's
  behavior under lab blackout is a specific interpretability probe (which
  named states go dark).
- **C. Lab staleness (availability lag):** shift every lab event's
  *visible* time by +{4h, 8h}: at a landmark, labs newer than the lag are
  invisible (they have not "returned from the lab" yet). Implemented as
  filtering at feature/tokenization time against landmark_time - lag, not
  by editing event times (event times are used for labels and must stay
  true). This is the cell where v9's recency channel gets stress-tested:
  hypothesis (label it as such in the journal) -- recency features help
  under staleness because the model can condition on "how old is my
  newest lab" instead of silently trusting stale values.

## Families, datasets, metrics

- Families: hazard (flagship checkpoints), tuned GBM, TabICL, EBM,
  SurvivalPFN. MEDS-Tab is optional/appendix: each cell requires a full
  re-tabularization, so it runs on a subset of cells (clean, A@0.3,
  B-labs) only if the wave leaves budget.
- Datasets: eICU + MIMIC held-out (the two with closed wave tables);
  GEMINI inherits the protocol later with the same machinery.
- Metrics per cell, per event-horizon pair: AUROC, AUPRC, ECE +
  reliability curves. Headline figure (F8 candidate): degradation curves
  (x = severity, y = AUROC delta from clean) with one line per family,
  plus a calibration companion panel. Aggregate numbers only in the
  journal, per the usual rule.

## Compute shape and sequencing

Eval-only: score dumps per (dataset x family x cell). Model forwards for
the hazard family are the only GPU cost (~8 cells x one alerts pass ~=
overnight on one A100 per dataset); baseline families are CPU rescores of
the degraded features. Sequencing: after the v3 wave tables close, on the
VM that frees up first; the degraded-shard generator + staleness filter
are lead-owned code (odyssey/, tested) built before then, so the sessions
only launch runs.

## Non-goals

No imputation comparisons (different paper), no train-time robustness, no
MNAR modeling claims -- the framing is operational stress testing, and the
limitations paragraph says so.
