# Track B designs (phase 4; launch when a VM frees post-wave)

Two pre-registered designs. Both are cheap relative to the wave; neither
launches before the wave tables close (master sequence, phase 4).

## N1: combine the M-series' two partial wins

The independent-training series ended with a split result: M2
(stage B + unfreeze_top_backbone_layers=2) gave the best overall balance
and a step up on every alert row; M3b (task_weight 0.1 in stage A) gave
the best task recovery (73.5 set top-1, ~77% of the gap to fully-joint
v8) but the weakest lever of the series (truth-flip +0.30). Nobody has
run the combination.

- **Config:** stage A with task_weight 0.1 (M3a recipe) -> stage B with
  freeze_bottleneck + unfreeze_top_backbone_layers=2 (M2 recipe),
  randint 1.0, otherwise the indep_b recipe unchanged. MIMIC 30 shards,
  same commit discipline, same six-mode banded intervention suite +
  displacement matching.
- **Question:** do task recovery and lever sign survive TOGETHER, or do
  they trade off intrinsically? This is the single most informative
  remaining point on the cost-vs-lever frontier.
- **Pre-registered read:** success = truth-flip >= +0.5 pt AND truth >=
  none AND set top-1 >= 70 AND zero_known retention < 30%. Anything
  less means the frontier bends: we report the frontier itself in R7
  and stop adding lever runs (the honest-account framing carries the
  section; see outline R7).
- **Cost:** ~4-5h A100 (stage A ~3.5h + stage B ~1h), eval ~1h.

## N2: distributional time-head probe (roadmap item 13, Doctor AI vein)

The current head is a point-process hazard; its horizon calibration is
excellent (within 0.2 pt everywhere on GEMINI), so the probe must beat a
strong baseline, not rescue a weak one. The hardest cell is
after-bundle 1h (base rate ~55%): distributional structure (bundle-size
counts, multimodal gaps) plausibly lives there.

- **Design:** freeze a flagship checkpoint's backbone; train ALTERNATIVE
  time heads on the frozen features (post-training, per the project lead's framing
  -- no backbone retraining): (a) current hazard head refit (control for
  head-capacity), (b) discretized-grid Bernoulli head (Doctor AI's
  framing, modernized), (c) log-normal mixture density (k=3), (d)
  Poisson bundle-size head for the same-instant channel specifically.
- **Metrics:** time NLL, per-horizon calibration incl. after-bundle 1h,
  same-instant accuracy. One dataset first (eICU, cheapest), one seed.
- **Decision rule:** a head family only graduates to a flagship-run
  A/B if it beats (a)-refit on after-bundle-1h calibration AND overall
  NLL. Otherwise the probe result is a supplementary table and the point
  head stands (a defensible, evidence-backed choice for the paper).
- **Cost:** frozen-feature head training is minutes-to-an-hour per head
  on one GPU; the whole probe fits in half a day, interleavable.

## Explicitly not in scope

Sepsis-3 / readmission task expansion (phase 3, separate spec);
GEMINI concept transfer (falls out of G6, not a Track B run); any new
lever mechanism beyond N1 until N1's read is in.
