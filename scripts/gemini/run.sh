#!/usr/bin/env bash
# Single entry point for everything that runs inside GEMINI. See
# docs/gemini.md's "scripts/gemini/run.sh" section for the full picture:
# nobody but Amrit can log into the node, so this is what he actually runs.
#
# Usage (on the GEMINI node, from the repo root):
#   scripts/gemini/run.sh [probe|schema|all]
#
# Steps:
#   probe   scripts/gemini/probe_env.sh  -> scripts/gemini/out/env_probe.txt
#   schema  scripts/gemini/explore_schema.py -> scripts/gemini/out/schema.{json,md}
#   all     every step above, in order (default)
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
    uv pip install -q "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0" "pandas>=2.2.0"
else
    pip install -q -e . --no-deps
    pip install -q "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0" "pandas>=2.2.0"
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

case "$STEP" in
    probe) run_probe ;;
    schema) run_schema ;;
    all) run_probe; run_schema ;;
    *)
        echo "unknown step: $STEP (expected probe, schema, or all)" >&2
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
