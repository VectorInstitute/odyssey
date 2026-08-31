#!/usr/bin/env bash
# 30-day readmission alert scoring for a finished run, on any VM.
#
#   scripts/vm_oneoff/readmission_alerts.sh RUN_DIR DATA_ROOT [TAG]
#
# Why this is a separate pass and not part of scripts/eval_run.sh:
# readmission_30d is a next_visit event (odyssey/data/alert_events.py). It is
# anchored at DISCHARGE, one index row per visit, at 168h/720h horizons. The
# eval chain's alerts stage runs the default landmark grid (every 4h within a
# visit, 8/24/72h), which structurally cannot score it -- so no run in this
# project has readmission numbers, on any dataset, until this is run.
#
# --index-mode visit_end already selects the next-visit events and the
# 168/720 horizons by itself, so --alerts and --horizons are deliberately NOT
# passed: let the run's own task set decide what is in scope.
#
# Append-only: writes NEW files (alerts_readmission.json,
# alerts_rows_readmission.parquet) and never --overwrite, so this cannot
# clobber the chain's own alerts.json / alerts_rows.parquet.
#
# GPU: this scores the hazard heads, so it needs the card. One job per card --
# run it after the run's training AND eval chain are done.
#
# The row dump is patient-level and stays on the VM. Pull only the JSON.
set -euo pipefail
RUN_DIR="${1:?RUN_DIR required}"
DATA_ROOT="${2:?DATA_ROOT required}"
TAG="${3:-readmission}"

PY="${PYTHON:-$HOME/odyssey/.venv/bin/python}"
LANES="${LANES:-64}"
CHUNK="${CHUNK:-512}"
SHARDS="${SHARDS:-4}"
BASELINE_SHARDS="${BASELINE_SHARDS:-30}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pt}"

OUT_JSON="$RUN_DIR/alerts_${TAG}.json"
OUT_ROWS="$RUN_DIR/alerts_rows_${TAG}.parquet"
LOG="$HOME/${TAG}_alerts.log"

for f in "$OUT_JSON" "$OUT_ROWS"; do
  [ -e "$f" ] && { echo "REFUSING: $f already exists (science outputs are append-only)"; exit 1; }
done

cd "$HOME/odyssey"
echo "run=$RUN_DIR data=$DATA_ROOT commit=$(git rev-parse --short HEAD)"

setsid nohup "$PY" -m odyssey.inference.alerts \
  --run-dir "$RUN_DIR" \
  --held-out-shard-dir "$DATA_ROOT/held_out" \
  --baseline-shard-dir "$DATA_ROOT/train" \
  --index-mode visit_end \
  --max-shards "$SHARDS" \
  --max-baseline-shards "$BASELINE_SHARDS" \
  --stream-baseline-shards \
  --num-lanes "$LANES" \
  --chunk-size "$CHUNK" \
  --checkpoint "$CHECKPOINT" \
  --output-json "$OUT_JSON" \
  --dump-rows "$OUT_ROWS" \
  > "$LOG" 2>&1 < /dev/null &
disown
echo "readmission alerts launched (log: $LOG)"
echo "done when: $OUT_JSON exists; then run scripts/alerts_cis.py with --dump $OUT_ROWS"
