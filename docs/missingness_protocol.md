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

## Implementation

What actually landed, matching the design above.

### Degraded-shard generator (`odyssey/data/degrade.py`)

All three axes are MEDS-row-level transforms on the raw shard, applied
before any family-specific feature construction or tokenization
(Principle 2), so no family needs axis-specific glue:

- **A. MCAR** (`apply_mcar`): drops each non-anchor, non-origin row
  independently with probability `p` in `{0.1, 0.3, 0.5}` (cells
  `mcar_0_1`, `mcar_0_3`, `mcar_0_5`).
- **B. Family blackout** (`apply_family_blackout`): removes every
  non-origin row of one family -- `labs`, `vitals`, `medications`
  (`odyssey.data.vocabulary.ROW_FAMILIES`) -- one cell each
  (`blackout_labs`, `blackout_vitals`, `blackout_medications`).
- **C. Lab availability lag** (`apply_lab_lag`): shifts every lab-family
  event's `time` forward by `lag_hours` in `{4.0, 8.0}` (cells `lag_4h`,
  `lag_8h`). Event times drive both visibility and labels, so shifting
  them forward is enough -- no separate feature-time filter is needed.

That is the fixed 8-cell grid (`all_cells`), plus the clean baseline.
Every transform protects two things by construction, not just by chance:
anchor rows (admission/discharge/demographic statics,
`odyssey.data.vocabulary.is_anchor`) and each subject's time origin
(first timed non-birth event) -- a subject's very first charted event can
legitimately be a lab, so origin protection is separate from anchor
protection. `_assert_origin_preserved` checks this at generation time for
every transform, in addition to the scoring-time
`baseline_prep._verify_matching_origins` check below (belt-and-suspenders,
not the only line of defense). `generate_cell` writes one shard directory
per cell plus a `metadata.json` (transform, seed, params, source, sha256
of every source shard) -- Principle 4's reproducibility record.

### Scoring path (`odyssey.inference.alerts.evaluate_alerts`)

Labels and the visit envelope always come from the clean
`held_out_shard_dir` (Principle 3); when `degraded_shard_dir` is given,
only the tokenized/featurized record (`binned`) is loaded from the
degraded copy. Before scoring, `baseline_prep._verify_matching_origins`
checks every subject's time origin in the degraded record still matches
the clean one (degrade.py's own guarantee, checked, not trusted).

The model is scored at the clean dump's own rows, not the degraded
record's landmark grid (lab lag shifts it; a dropped visit start would
re-bucket it): `collect_model_scores_at_rows` walks the degraded record
and, for each clean `(subject, visit, time)` row, scores at the first
token charted AT that time when one exists, else the last visible token
strictly BEFORE it -- what the model would actually know at that instant,
given this degraded record. A clean row with no visible token at or
before its time (a landmark earlier than the degraded record's own kept
window, e.g. under a lab lag or a transformer's context truncation) is
**unscoreable**: excluded from this cell's metrics rather than forced
onto a token it couldn't see, and returned as a set of
`(subject_id, visit_id, time_hours)` keys via `unscoreable_out`.

`verify_against_dump` (pointed at the clean run's own `--dump-rows`
parquet) is the acceptance check for a degraded cell: `verify_rows_match_dump`
asserts the row set actually scored matches the dump exactly, **minus**
the unscoreable keys (which are excluded from the comparison, not treated
as a mismatch) -- both row identity and, per horizon, label agreement.
This runs in addition to the existing `verify_packed_landmark_rows`
self-consistency check that always runs on a clean (non-degraded) pass.

The GBM baseline is fit once, on the clean split, and reused frozen
against every cell (`prefit_baselines`/`fitted_baselines_out`, the same
mutable-out-param convention as `unscoreable_out`) -- Principle 1 for the
baseline family too: refitting per cell would confound the degradation
signal with fit-to-fit hyperparameter-search variance.

### One-command sweep (`scripts/missingness_sweep.py`)

Sequences the above for a whole run: generate the 8 cells, score the
clean baseline once (fits the GBM, dumps `clean_alerts_rows.parquet` as
the `verify_against_dump` target), score each cell against the same
held-out split reading from its degraded shard directory, then aggregate.
Idempotent and resumable (`--output-root`, `--overwrite`); nothing it
writes ever lands in git. Outputs under `--output-root`:

- `degraded_shards/<cell>/` -- degrade.py's own output, one per cell.
- `results/<cell>_alerts.json` -- that cell's `AlertMetrics` records plus
  `n_unscoreable` (the `unscoreable_out` count for that cell; `None` for
  the clean baseline, which never scores against a degraded record) and
  the cell's degrade.py metadata.
- `results/<cell>_alerts_rows.parquet` -- the per-index-row dump.
- `degradation_table.json` / `.md` (`odyssey.reporting.missingness_report`)
  -- one row per (cell, scorer, event, horizon): AUROC/AUPRC/ECE and their
  delta from the matching clean row, plus `n_unscoreable`. The markdown
  table also gets a "Reduced row sets" note listing any cell whose
  `n_unscoreable` is nonzero, since that cell's metrics (and deltas) are
  over a smaller cohort than clean, not the exact-match cohort Principle 3
  otherwise guarantees.

### Transformer-truncation caveat

`backbone="transformer"` uses `PackedContextSampler`, which truncates a
subject's sequence to the last `max_context` tokens rather than chunking
it (unlike the hybrid backbone's lane sampler). Under a degradation that
shifts or drops early tokens (lab lag most visibly), a subject's kept
window can start later than it would on the clean record -- this is
exactly the mechanism that produces unscoreable rows above, not a bug:
the clean landmark simply predates what the degraded record's truncated
window still has visible. It is real signal about the transformer
backbone specifically: the hybrid backbone's chunked lane sampler walks a
subject's whole sequence with no truncation, so the protected origin
token is always available as the earliest fallback and this mechanism
does not apply to it -- not an artifact of the harness.
