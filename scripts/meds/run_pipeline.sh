#!/usr/bin/env bash
# Run the full MIMIC-IV → MEDS → odyssey tokenized pipeline.
#
# Usage:
#   bash scripts/meds/run_pipeline.sh \
#       --mimic_dir  data/physionet.org/files/mimiciv/3.1 \
#       --output_dir data/pipeline_output \
#       [--n_subjects 1000]   # omit or 0 for all subjects
#
# Steps:
#   1. Extract MIMIC-IV CSVs → MEDS parquet (scripts/meds/extract_mimic_iv.py)
#   2. Apply meds-transforms 0.6.1 pipeline (filter + normalize)
#   3. Convert MEDS → odyssey tokenized parquet (scripts/meds/meds_to_odyssey.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── argument parsing ──────────────────────────────────────────────────────────
MIMIC_DIR=""
OUTPUT_DIR=""
N_SUBJECTS=0
MAX_SEQ_LEN=2048

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mimic_dir)   MIMIC_DIR="$2";   shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --n_subjects)  N_SUBJECTS="$2";  shift 2 ;;
        --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MIMIC_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --mimic_dir <path> --output_dir <path> [--n_subjects N] [--max_seq_len L]"
    exit 1
fi

MEDS_DIR="${OUTPUT_DIR}/meds"
MEDS_PROCESSED_DIR="${OUTPUT_DIR}/meds_processed"
ODYSSEY_DIR="${OUTPUT_DIR}/odyssey_tokenized"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtualenv not found at ${REPO_ROOT}/.venv. Run 'uv sync --dev' first."
    exit 1
fi

# ── step 1: MIMIC-IV → MEDS ──────────────────────────────────────────────────
echo "=== Step 1: Extract MIMIC-IV → MEDS ==="
"$PYTHON" "${SCRIPT_DIR}/extract_mimic_iv.py" \
    --mimic_dir  "$MIMIC_DIR" \
    --output_dir "$MEDS_DIR" \
    --n_subjects "$N_SUBJECTS"

echo "MEDS data written to: $MEDS_DIR"

# ── step 2: meds-transforms 0.6.1 pipeline ───────────────────────────────────
echo ""
echo "=== Step 2: meds-transforms post-processing ==="
PIPELINE_YAML="${SCRIPT_DIR}/pipeline.yaml"

if "$PYTHON" -c "import meds_transforms" 2>/dev/null; then
    MEDS_TRANSFORM_CMD="${REPO_ROOT}/.venv/bin/MEDS_transform-pipeline"
    if [[ -x "$MEDS_TRANSFORM_CMD" ]]; then
        "$MEDS_TRANSFORM_CMD" \
            "$PIPELINE_YAML" \
            input_dir="$MEDS_DIR" \
            output_dir="$MEDS_PROCESSED_DIR"
        echo "meds-transforms output written to: $MEDS_PROCESSED_DIR"
    else
        echo "WARNING: MEDS_transform-pipeline not found in venv — skipping step 2."
        MEDS_PROCESSED_DIR="$MEDS_DIR"
    fi
else
    echo "WARNING: meds_transforms not installed — skipping step 2."
    MEDS_PROCESSED_DIR="$MEDS_DIR"
fi

# ── step 3: MEDS → odyssey tokenized ─────────────────────────────────────────
echo ""
echo "=== Step 3: Convert MEDS → odyssey tokenized format ==="
"$PYTHON" "${SCRIPT_DIR}/meds_to_odyssey.py" \
    --meds_dir   "$MEDS_PROCESSED_DIR" \
    --output_dir "$ODYSSEY_DIR" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --split all

echo ""
echo "=== Pipeline complete ==="
echo "  MEDS raw:        $MEDS_DIR"
echo "  MEDS processed:  $MEDS_PROCESSED_DIR"
echo "  Odyssey format:  $ODYSSEY_DIR"
echo ""
echo "To pretrain, point --data_dir to $ODYSSEY_DIR"
