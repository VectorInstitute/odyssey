#!/usr/bin/env bash
# Canonical evaluation sequence for a finished training run, on either VM.
#
# Usage:
#   scripts/eval_run.sh RUN_DIR DATA_ROOT [OUT_HTML]
#
#   RUN_DIR    a training output dir (config.json, vocabulary.json, checkpoint_best.pt)
#   DATA_ROOT  the MEDS data root with train/ tuning/ held_out/ (e.g. ~/data/mimiciv_3.1_v1/data)
#   OUT_HTML   report path (default: RUN_DIR/report.html)
#
# Environment knobs (all optional):
#   LANES=64               lanes for eval passes (use 16 when training runs alongside)
#   CHUNK=512              chunk size (must match what the run trained with)
#   ALERT_SHARDS=4         held-out shards for interventions/alerts
#   BASELINE_SHARDS=30     train shards the GBM baseline is fitted on (= the run's max_train_shards)
#   STREAM_BASELINE=0      1 = fit the GBM baseline shard by shard instead of loading
#                          BASELINE_SHARDS whole into memory; set for full-scale runs
#                          (hundreds of shards), where the whole-frame path OOMs
#   CHECKPOINT=checkpoint_best.pt
#   PYTHON=<path>          interpreter (default: RUN_DIR/../../odyssey/.venv/bin/python, else `python`)
#   DRY_RUN=1              print the commands instead of running them
#
# Stages (each logs "=== STAGE <name> EXIT <code> <utc time> ==="):
#   eval          full held-out standard evaluation -> inference_results.json
#   interventions banded (equal-displacement) lever test plus zero_known / zero_unknown completeness probes on ALERT_SHARDS -> interventions_band15.json
#                 (skipped for model_kind=baseline: no bottleneck)
#   alerts        hazard heads vs the strong tuned per-event GBM on ALERT_SHARDS -> alerts.json,
#                 plus the per-index-row table alerts_rows.parquet (patient-level: stays in RUN_DIR)
#   cases         case traces -> case_studies.json (skipped for baseline; an empty list is written)
#   report        the HTML report -> OUT_HTML
# A failed stage does not stop the later ones; read the EXIT codes.
set -u
RUN_DIR="${1:?RUN_DIR required}"; DATA_ROOT="${2:?DATA_ROOT required}"; OUT_HTML="${3:-$RUN_DIR/report.html}"
LANES="${LANES:-64}"; CHUNK="${CHUNK:-512}"; ALERT_SHARDS="${ALERT_SHARDS:-4}"; BASELINE_SHARDS="${BASELINE_SHARDS:-30}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pt}"; DRY_RUN="${DRY_RUN:-0}"; STREAM_BASELINE="${STREAM_BASELINE:-0}"
ALERTS_ARGS=(--run-dir "$RUN_DIR" --held-out-shard-dir "$DATA_ROOT/held_out" --baseline-shard-dir "$DATA_ROOT/train" --max-shards "$ALERT_SHARDS" --max-baseline-shards "$BASELINE_SHARDS" --num-lanes "$LANES" --chunk-size "$CHUNK" --output-json "$RUN_DIR/alerts.json" --dump-rows "$RUN_DIR/alerts_rows.parquet" --checkpoint "$CHECKPOINT")
[ "$STREAM_BASELINE" = "1" ] && ALERTS_ARGS+=(--stream-baseline-shards)
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HOME/odyssey/.venv/bin/python" ]; then PYTHON="$HOME/odyssey/.venv/bin/python"; else PYTHON="python"; fi
fi
MODEL_KIND=$("$PYTHON" - "$RUN_DIR" <<'PY'
import json, sys
print(json.load(open(sys.argv[1] + "/config.json")).get("model_kind", "bottleneck"))
PY
)
stage() {  # stage NAME CMD...
  local name="$1"; shift
  echo "=== STAGE $name START $(date -u +%FT%TZ) ==="
  if [ "$DRY_RUN" = "1" ]; then printf '  %q' "$@"; echo; echo "=== STAGE $name EXIT 0 (dry) ==="; return 0; fi
  "$@" > "$RUN_DIR/${name}.log" 2>&1
  local code=$?
  echo "=== STAGE $name EXIT $code $(date -u +%FT%TZ) ==="
  return $code
}
echo "run=$RUN_DIR data=$DATA_ROOT model_kind=$MODEL_KIND lanes=$LANES chunk=$CHUNK checkpoint=$CHECKPOINT commit=$(cd "$HOME/odyssey" 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

stage eval "$PYTHON" -m odyssey.inference.run_inference --run-dir "$RUN_DIR" --held-out-shard-dir "$DATA_ROOT/held_out" --output-json "$RUN_DIR/inference_results.json" --num-lanes "$LANES" --chunk-size "$CHUNK" --checkpoint "$CHECKPOINT"

if [ "$MODEL_KIND" = "bottleneck" ]; then
  stage interventions "$PYTHON" -m odyssey.inference.interventions --run-dir "$RUN_DIR" --held-out-shard-dir "$DATA_ROOT/held_out" --output-json "$RUN_DIR/interventions_band15.json" --max-shards "$ALERT_SHARDS" --num-lanes "$LANES" --chunk-size "$CHUNK" --uncertain-band 0.15 --modes none truth flip random zero_known zero_unknown --checkpoint "$CHECKPOINT"
else
  echo "=== STAGE interventions SKIPPED (baseline model) ==="
fi

stage alerts "$PYTHON" -m odyssey.inference.alerts "${ALERTS_ARGS[@]}"

if [ "$MODEL_KIND" = "bottleneck" ]; then
  stage cases "$PYTHON" -m odyssey.inference.case_study --run-dir "$RUN_DIR" --held-out-shard-dir "$DATA_ROOT/held_out" --output-json "$RUN_DIR/case_studies.json" --n-cases 15 --max-shards 2 --checkpoint "$CHECKPOINT"
else
  [ "$DRY_RUN" = "1" ] || echo "[]" > "$RUN_DIR/case_studies.json"
  echo "=== STAGE cases SKIPPED (baseline model; empty case_studies.json written) ==="
fi

REPORT_ARGS=(--run-dir "$RUN_DIR" --inference-results "$RUN_DIR/inference_results.json" --case-studies "$RUN_DIR/case_studies.json" --alerts "$RUN_DIR/alerts.json" --output-html "$OUT_HTML")
[ "$MODEL_KIND" = "bottleneck" ] && REPORT_ARGS+=(--interventions "$RUN_DIR/interventions_band15.json")
stage report "$PYTHON" -m odyssey.reporting.concept_bottleneck_report "${REPORT_ARGS[@]}"
echo "=== EVAL SEQUENCE DONE $(date -u +%FT%TZ) ==="
