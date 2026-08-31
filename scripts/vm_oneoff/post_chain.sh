#!/usr/bin/env bash
# Everything that must happen AFTER scripts/eval_run.sh finishes, in order.
#
#   scripts/vm_oneoff/post_chain.sh RUN_DIR DATA_ROOT
#
# eval_run.sh is not the whole story, and the gap has bitten twice:
#
#   1. It does NOT run scripts/intervention_cis.py. A previous handoff
#      described the chain as producing "everything", but the paired
#      subject-clustered CIs are a separate step over the per-subject dump.
#      Those CIs gate the paper's Figure 3 error bars.
#   2. It does NOT score readmission_30d. That is a next_visit event needing
#      --index-mode visit_end at 168/720h, while the chain only ever runs the
#      landmark grid at 8/24/72h -- which is why no run in this project had
#      readmission numbers on any dataset.
#
# Both are GPU/CPU work on a card that is free once the chain is done, so
# they belong together. Each step is skipped (not failed) when its input is
# missing or its output already exists, so this is safe to re-run.
#
# Patient-level outputs (the per-subject dump, the readmission row parquet)
# stay on the VM. Pull only the JSONs.
set -u
RUN="${1:?RUN_DIR required}"
DATA="${2:?DATA_ROOT required}"
PY="${PYTHON:-$HOME/odyssey/.venv/bin/python}"
cd "$HOME/odyssey"

echo "post-chain for $RUN at commit $(git rev-parse --short HEAD)"

step() {
  local name="$1"; shift
  echo "=== POST $name START $(date -u +%FT%TZ) ==="
  "$@"
  echo "=== POST $name EXIT $? $(date -u +%FT%TZ) ==="
}

# 1. Paired subject-clustered CIs on the intervention deltas.
PS="$RUN/interventions_band15_per_subject.json"
OUT_CIS="$RUN/intervention_cis.json"
if [ ! -f "$PS" ]; then
  echo "=== POST intervention_cis SKIPPED: no $PS (did the interventions stage run with --dump-per-subject?) ==="
elif [ -e "$OUT_CIS" ]; then
  echo "=== POST intervention_cis SKIPPED: $OUT_CIS exists (append-only) ==="
else
  step intervention_cis "$PY" scripts/intervention_cis.py \
    --per-subject "$PS" --output-json "$OUT_CIS"
fi

# 2. 30-day readmission, discharge-anchored. Its own script so it can also be
#    run standalone; it self-checks its outputs and refuses to clobber them.
if [ -e "$RUN/alerts_readmission.json" ]; then
  echo "=== POST readmission SKIPPED: alerts_readmission.json exists ==="
else
  step readmission bash scripts/vm_oneoff/readmission_alerts.sh "$RUN" "$DATA"
  echo "    (readmission runs detached; poll ~/readmission_alerts.log)"
fi

echo "=== POST-CHAIN SEQUENCE DONE $(date -u +%FT%TZ) ==="
echo "Still owed separately, per arm: the supplemental Guide Labs/W3 scoring"
echo "(scripts/vm_oneoff/supplemental_r{2,6}_*.sh) against the FLAGSHIP"
echo "checkpoint, and alerts_cis.py over the row dumps for the table bolds."
