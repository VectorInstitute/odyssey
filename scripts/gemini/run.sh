#!/usr/bin/env bash
# Single entry point for everything that runs inside GEMINI. See
# docs/gemini.md's "scripts/gemini/run.sh" section for the full picture:
# nobody but Amrit can log into the node, so this is what he actually runs.
#
# Usage (on the GEMINI node, from the repo root):
#   scripts/gemini/run.sh [probe|schema|env-gpu|extract-dry|extract|train|eval|all]
#
# Steps:
#   probe        scripts/gemini/probe_env.sh -> scripts/gemini/out/env_probe.txt
#   schema       scripts/gemini/explore_schema.py -> scripts/gemini/out/schema.{json,md}
#   env-gpu      builds the H200 training venv (pinned torch cu124 + the
#                two-step mamba-ssm CUDA rebuild, docs/gemini.md's recipe)
#                and writes scripts/gemini/out/env_fingerprint.json. Separate
#                from the lightweight venv the other steps use -- this one
#                pulls in torch/mamba-ssm, which none of them need.
#   extract-dry  scripts/gemini/extract_dry.py -> scripts/gemini/out/
#                extract_dry.{json,md}. Needs schema.json (run `schema`
#                first); prints "pending schema report" and does nothing
#                otherwise.
#   extract      scripts/gemini/extract_meds.py -> streams real MEDS parquet
#                shards to GEMINI_MEDS_OUTPUT_DIR (default ~/gemini_meds_v1,
#                outside the repo -- never committed, see docs/gemini.md's
#                governance rules), and commits only a small, suppressed
#                scripts/gemini/out/extraction_summary.json. A real,
#                long-running, patient-data-writing operation, not a quick
#                check -- see docs/gemini_extraction.md for the design.
#   train        not built yet.
#   eval         not built yet.
#   all          probe, schema, extract-dry, in order (default; deliberately
#                excludes env-gpu and extract -- see below)
#
# Self-syncing: every invocation starts with `git fetch origin && git reset
# --hard origin/main` (never `git pull` -- every mirror rewrites history, so
# pull's merge sees the same content as two unrelated commits and refuses).
# Refuses to reset, and stops instead, if the tree is dirty or there are
# local commits origin/main doesn't have -- see docs/gemini.md's
# "fetch+reset, never pull" note for what to do if that happens.
#
# Activates (creating if missing) a venv under $HOME, installs only what the
# selected step needs, runs it, then commits and pushes ONLY the output it
# produced. Safe to re-run: venv/install steps are idempotent, each step
# overwrites its own output deterministically, and if nothing changed there
# is nothing to commit. Refuses to commit anything outside
# scripts/gemini/out/ or docs/gemini*, anything over 900 KB, or any path
# that looks like data.
set -euo pipefail

REPO_DIR=$(git rev-parse --show-toplevel)
cd "$REPO_DIR"

STEP="${1:-all}"
VENV="${GEMINI_VENV:-$HOME/.venvs/odyssey-gemini}"

# --- sync with the mirror (fetch + reset, never pull) ---------------------
#
# Every mirror rewrites gemini's history (see docs/gemini.md's mirroring
# section): a commit made on this node and its mirrored equivalent from
# GitHub can be the exact same content under two different hashes, so
# `git pull` (fetch + merge) sees them as unrelated and refuses -- the
# predictable divergence Amrit hit. Fetch + hard reset to origin/main is
# the fix, but only when it's safe: if the working tree is dirty or this
# node has real local commits origin/main doesn't have, resetting would
# silently throw them away. Runs before every step, not just once, so a
# plain `scripts/gemini/run.sh <step>` is always self-syncing.
sync_with_mirror() {
    git fetch origin --quiet

    local dirty
    dirty=$(git status --porcelain)
    if [[ -n "$dirty" ]]; then
        echo "REFUSING to sync: working tree is dirty:" >&2
        echo "$dirty" >&2
        echo "Commit or stash these changes first, then re-run." >&2
        exit 1
    fi

    local unpushed
    unpushed=$(git log --oneline origin/main..HEAD)
    if [[ -n "$unpushed" ]]; then
        echo "REFUSING to sync: local commits not on origin/main:" >&2
        echo "$unpushed" >&2
        echo "These outputs must be pushed before resetting over them -- if" >&2
        echo "run.sh's own commit-and-push step made them, re-run this step" >&2
        echo "to retry the push; if you committed some other way, push it" >&2
        echo "first." >&2
        exit 1
    fi

    git reset --hard origin/main
}

sync_with_mirror

# --- environment -------------------------------------------------------

# GLIBCXX mismatch workaround, same as gemini-variation-study's run.sh --
# harmless if the system libstdc++ already matches what Python needs.
LIBSTDCPP=$(find /usr/lib/x86_64-linux-gnu /usr/local/lib -name "libstdc++.so.6" 2>/dev/null | head -1)
if [[ -n "$LIBSTDCPP" ]]; then
    export LD_PRELOAD="$LIBSTDCPP"
fi

# Check for bin/activate specifically, not just the directory: a venv
# creation attempt that failed partway through (e.g. system python3 with
# no ensurepip) can leave a directory behind with no working activate
# script, which a bare directory-existence check would mistake for done.
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Creating venv at $VENV"
    rm -rf "$VENV"
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV" --python 3.12
    else
        python3 -m venv "$VENV"
    fi
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# odyssey itself, without its heavy required deps (torch, MIMIC_IV_MEDS,
# ...) -- none of the steps below need them yet. `--no-deps` also means the
# `gemini` extra's packages have to be installed explicitly.
echo "Installing odyssey (editable, no deps) + gemini extras..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -q -e . --no-deps
    uv pip install -q "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0" "pandas>=2.2.0" "pyarrow>=15.0.0"
else
    pip install -q -e . --no-deps
    pip install -q "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0" "pandas>=2.2.0" "pyarrow>=15.0.0"
fi

# --- steps ---------------------------------------------------------------

run_probe() {
    echo "=== probe ==="
    bash scripts/gemini/probe_env.sh
}

run_schema() {
    echo "=== schema ==="
    python scripts/gemini/explore_schema.py
}

run_extract_dry() {
    echo "=== extract-dry ==="
    python scripts/gemini/extract_dry.py
}

run_pending_stub() {
    # train / eval: real work belongs here once extract has actually run
    # and there's a real MEDS shard directory to train/eval against. Until
    # then this is intentionally a no-op that always succeeds, so wiring
    # these into an operator's routine `all` run (once they're added
    # there) never fails on a step that isn't built yet.
    echo "=== $1 ==="
    echo "pending real extraction -- $1 is not built yet, see docs/gemini.md"
}

run_extract() {
    echo "=== extract ==="
    echo "Writing MEDS parquet shards to \${GEMINI_MEDS_OUTPUT_DIR:-\$HOME/gemini_meds_v1} (not committed) ..."
    python scripts/gemini/extract_meds.py
}

run_env_gpu() {
    echo "=== env-gpu ==="
    GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
    if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
        echo "Creating GPU venv at $GPU_VENV"
        rm -rf "$GPU_VENV"
        python3 -m venv "$GPU_VENV"
    fi
    # shellcheck source=/dev/null
    source "$GPU_VENV/bin/activate"

    # Idempotent: only (re)installs torch/rebuilds mamba-ssm if they aren't
    # already importable. The mamba-ssm rebuild specifically is a real ~30
    # minute CUDA compile (docs/gemini.md), not something to redo blind on
    # every run.
    if python -c "import torch; assert torch.__version__.startswith('2.6.0')" 2>/dev/null; then
        echo "torch==2.6.0 already installed, skipping."
    else
        echo "Installing torch==2.6.0+cu124..."
        pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    fi

    echo "Installing odyssey (editable, no deps)..."
    pip install -q -e . --no-deps

    if python -c "from mamba_ssm.modules.mamba2 import Mamba2" 2>/dev/null; then
        echo "mamba_ssm already importable, skipping rebuild."
    else
        echo "Force-rebuilding mamba-ssm==2.3.0 against CUDA 12.8 (real compile, ~30 min)..."
        PATH="/usr/local/cuda-12.8/bin:$PATH" \
            CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=12 \
            MAMBA_FORCE_BUILD=TRUE \
            pip install --no-build-isolation --no-binary mamba-ssm \
            --no-deps --no-cache-dir --force-reinstall 'mamba-ssm==2.3.0'
    fi

    echo "Import-checking mamba_ssm..."
    python -c "from mamba_ssm.modules.mamba2 import Mamba2; print('mamba_ssm import OK')"

    echo "Writing env fingerprint..."
    mkdir -p scripts/gemini/out
    python -c "
import json
from odyssey.utils.env_fingerprint import environment_fingerprint
with open('scripts/gemini/out/env_fingerprint.json', 'w') as f:
    json.dump(environment_fingerprint(), f, indent=2)
"
    cat scripts/gemini/out/env_fingerprint.json

    deactivate
    # Back to the lightweight venv, in case a later step in the same
    # invocation (e.g. `all`) doesn't want torch/mamba-ssm on its path.
    source "$VENV/bin/activate"
}

case "$STEP" in
    probe) run_probe ;;
    schema) run_schema ;;
    env-gpu) run_env_gpu ;;
    extract-dry) run_extract_dry ;;
    extract) run_extract ;;
    train) run_pending_stub train ;;
    eval) run_pending_stub eval ;;
    all) run_probe; run_schema; run_extract_dry ;;
    *)
        echo "unknown step: $STEP (expected probe, schema, env-gpu, extract-dry, extract, train, eval, or all)" >&2
        exit 1
        ;;
esac

# --- commit and push only the small, allowed outputs ----------------------

ALLOWED_PREFIX_RE='^(scripts/gemini/out/|docs/gemini)'
DATA_LIKE_RE='\.(parquet|csv|pt|ckpt|pth|npz|npy)$'
MAX_BYTES=900000

echo "Checking for changes to commit..."
TO_ADD=()
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if [[ ! "$f" =~ $ALLOWED_PREFIX_RE ]]; then
        echo "skipping (outside allowed paths, not committing): $f"
        continue
    fi
    if [[ "$f" =~ $DATA_LIKE_RE ]]; then
        echo "REFUSING to commit data-like file: $f" >&2
        exit 1
    fi
    if [[ -f "$f" ]]; then
        SIZE=$(wc -c <"$f" | tr -d ' ')
        if ((SIZE > MAX_BYTES)); then
            echo "REFUSING to commit oversized file ($SIZE bytes > $MAX_BYTES): $f" >&2
            exit 1
        fi
    fi
    TO_ADD+=("$f")
done < <(git status --porcelain --untracked-files=all | awk '{print $2}')

if [[ ${#TO_ADD[@]} -eq 0 ]]; then
    echo "Nothing to commit."
    exit 0
fi

git add "${TO_ADD[@]}"
git commit -m "scripts/gemini/run.sh $STEP: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin main
echo "Pushed."
