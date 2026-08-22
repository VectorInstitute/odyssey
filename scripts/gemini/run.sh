#!/usr/bin/env bash
# Single entry point for everything that runs inside GEMINI. See
# docs/gemini.md's "scripts/gemini/run.sh" section for the full picture:
# nobody but Amrit can log into the node, so this is what he actually runs.
#
# Usage (on the GEMINI node, from the repo root):
#   scripts/gemini/run.sh [probe|schema|env-gpu|extract-dry|extract|finalize|train|eval|all]
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
#                shards to GEMINI_MEDS_OUTPUT_DIR (default
#                /mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1,
#                outside the repo -- never committed, see docs/gemini.md's
#                governance rules), and commits only a small, suppressed
#                scripts/gemini/out/extraction_summary.json. A real,
#                long-running, patient-data-writing operation, not a quick
#                check -- see docs/gemini_extraction.md for the design.
#   finalize     scripts/gemini/finalize_meds.py -> rewrites `extract`'s flat
#                output into the MEDS-conformant data/<split>/ + metadata/
#                layout odyssey/data/meds_validation.py checks, in place
#                under the same GEMINI_MEDS_OUTPUT_DIR. Only runs once
#                `extract`'s manifest shows every table complete (checked,
#                not assumed) -- do not run this until told to; see
#                scripts/gemini/finalize_meds.py's own docstring for the
#                design and crash semantics.
#   train        not built yet.
#   eval         not built yet.
#   all          probe, schema, extract-dry, in order (default; deliberately
#                excludes env-gpu, extract, and finalize -- see below)
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
        # This can be a false positive, not real divergence: every mirror
        # rewrites history (see the mirroring section above), so a commit
        # this node made can be byte-identical to what already reached
        # origin/main via the fetch-and-copy-files mirror-back direction,
        # just under a completely unrelated hash -- ancestry alone can't
        # tell "content-preserved" from "genuinely local-only". Diff the
        # actual paths the unpushed commits touched (not the whole tree,
        # which differs constantly for unrelated reasons as origin/main
        # keeps moving) against origin/main's current content for those
        # exact paths.
        local touched_paths
        touched_paths=$(
            git rev-list origin/main..HEAD |
                xargs -I{} git diff-tree --no-commit-id --name-only -r {} |
                sort -u
        )

        local lost_paths=""
        local path
        while IFS= read -r path; do
            [[ -z "$path" ]] && continue
            if ! git diff --quiet origin/main HEAD -- "$path"; then
                lost_paths+="$path"$'\n'
            fi
        done <<<"$touched_paths"

        if [[ -n "$lost_paths" ]]; then
            echo "REFUSING to sync: local commits not on origin/main, and" >&2
            echo "these paths differ from origin/main's current content" >&2
            echo "(real content that would be lost by resetting):" >&2
            echo "$lost_paths" >&2
            echo "$unpushed" >&2
            echo "These must be pushed before resetting over them -- if" >&2
            echo "run.sh's own commit-and-push step made them, re-run this step" >&2
            echo "to retry the push; if you committed some other way, push it" >&2
            echo "first." >&2
            exit 1
        fi

        echo "Local commits not on origin/main by hash, but every path they"
        echo "touched matches origin/main's current content exactly --"
        echo "outputs preserved upstream, resetting."
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
# `gemini` extra's packages have to be installed explicitly, since
# `--no-deps` skips extras resolution too, not just the base dependency
# list -- there's no single pip/uv flag for "install only this extra's own
# packages, skip everything else". The list below MUST be kept in sync
# with pyproject.toml's `gemini` extra by hand -- real incident: polars
# was added there for extract_meds.py's vectorized transforms but not
# here, and Amrit's extract crashed on ModuleNotFoundError before
# touching the database, because this list still didn't have it.
echo "Installing odyssey (editable, no deps) + gemini extras..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -q -e . --no-deps
    uv pip install -q "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0" "pandas>=2.2.0" "pyarrow>=15.0.0" "polars>=1.30.0"
else
    pip install -q -e . --no-deps
    pip install -q "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0" "pandas>=2.2.0" "pyarrow>=15.0.0" "polars>=1.30.0"
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
    echo "This can take HOURS -- lab_subset/vitals_subset alone are hundreds"
    echo "of millions of rows. Writes real patient data to"
    echo "\${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1} (never committed)."
    # run.sh does not, and will not, background itself: it commits and
    # pushes the summary synchronously after the step finishes (see the
    # bottom of this script), which a self-daemonizing step would break.
    # If this doesn't look like a detached session, warn loudly rather
    # than silently losing hours of work to a dropped SSH connection.
    if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
        echo "WARNING: this doesn't look like a tmux or screen session --" >&2
        echo "if the SSH connection drops, the extraction dies with it." >&2
        echo "Run it detached instead, e.g.:" >&2
        echo "  tmux new -s extract 'scripts/gemini/run.sh extract'" >&2
        echo "  # or: nohup scripts/gemini/run.sh extract > extract.log 2>&1 &" >&2
    fi
    python scripts/gemini/extract_meds.py
}

run_finalize() {
    echo "=== finalize ==="
    echo "Rewrites GEMINI_MEDS_OUTPUT_DIR's flat extract output into the"
    echo "MEDS-conformant data/<split>/ + metadata/ layout, in place. Refuses"
    echo "to run unless extract's manifest shows every table complete."
    if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
        echo "WARNING: this doesn't look like a tmux or screen session --" >&2
        echo "if the SSH connection drops, finalize dies with it." >&2
        echo "Run it detached instead, e.g.:" >&2
        echo "  tmux new -s finalize 'scripts/gemini/run.sh finalize'" >&2
        echo "  # or: nohup scripts/gemini/run.sh finalize > finalize.log 2>&1 &" >&2
    fi
    python scripts/gemini/finalize_meds.py
}

run_env_gpu() {
    echo "=== env-gpu ==="

    # --- probe first: fail loudly here, in seconds, not 25+ minutes into
    # a mamba-ssm compile that was always going to fail. The CUDA toolkit
    # version on whatever node this runs on is not assumed -- discovered
    # fresh every run rather than trusting docs/gemini.md's last probe
    # (2026-08-18, possibly a different node or a since-changed one).
    echo "--- probing GPU/CUDA ---"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found -- no NVIDIA driver visible on this node." >&2
        echo "env-gpu needs a real GPU node; refusing to proceed." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

    CUDA_TOOLKIT_DIR=""
    for candidate in /usr/local/cuda-*; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            CUDA_TOOLKIT_DIR="$candidate"
            break
        fi
    done
    if [[ -z "$CUDA_TOOLKIT_DIR" ]]; then
        echo "No CUDA toolkit with a working nvcc found under /usr/local/cuda-*." >&2
        echo "mamba-ssm's CUDA extension cannot be built without one -- the" >&2
        echo "driver alone (see nvidia-smi output above) is not enough." >&2
        echo "See docs/gemini.md's Environment section and pyproject.toml's" >&2
        echo "cuda extra comment for what this needs and why." >&2
        exit 1
    fi
    echo "Using CUDA toolkit: $CUDA_TOOLKIT_DIR"
    "$CUDA_TOOLKIT_DIR/bin/nvcc" --version

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

    # This venv is a curated no-deps world: torch and mamba-ssm are
    # installed separately (above/below) for exact-CUDA-pin reasons, and
    # `--no-deps` on odyssey itself means every OTHER real runtime import
    # odyssey.training/models/inference actually makes has to be listed
    # here explicitly, same discipline as the lightweight venv's
    # gemini-extras mirror above. Audited against every top-level AND
    # deferred (`# noqa: PLC0415`) third-party import across odyssey/ --
    # polars and scikit-learn (odyssey.training.metrics) were already
    # missing from any install path for this venv; einops
    # (odyssey.models.backbones.hybrid, needed by TrainingConfig's own
    # backbone="hybrid" default) was missing from pyproject.toml entirely,
    # fixed there too (see the cuda extra). Real incident this discipline
    # exists for: polars was once missing from the lightweight venv's own
    # mirror list the same way.
    echo "Installing odyssey (editable, no deps) + training runtime deps..."
    pip install -q -e . --no-deps
    pip install -q "polars>=1.30.0" "scikit-learn>=1.7.0" "einops>=0.7.0"

    if python -c "from mamba_ssm.modules.mamba2 import Mamba2" 2>/dev/null; then
        echo "mamba_ssm already importable, skipping rebuild."
    else
        echo "Force-rebuilding mamba-ssm==2.3.0 against $CUDA_TOOLKIT_DIR (real compile, ~30 min)..."
        PATH="$CUDA_TOOLKIT_DIR/bin:$PATH" \
            CUDA_HOME="$CUDA_TOOLKIT_DIR" MAX_JOBS=12 \
            MAMBA_FORCE_BUILD=TRUE \
            pip install --no-build-isolation --no-binary mamba-ssm \
            --no-deps --no-cache-dir --force-reinstall 'mamba-ssm==2.3.0'
    fi

    echo "Import-checking mamba_ssm and einops..."
    python -c "from mamba_ssm.modules.mamba2 import Mamba2; print('mamba_ssm import OK')"
    python -c "import einops; print('einops import OK')"

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
    finalize) run_finalize ;;
    train) run_pending_stub train ;;
    eval) run_pending_stub eval ;;
    all) run_probe; run_schema; run_extract_dry ;;
    *)
        echo "unknown step: $STEP (expected probe, schema, env-gpu, extract-dry, extract, finalize, train, eval, or all)" >&2
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

# origin/main can move between sync_with_mirror() at the top of this run
# and this push at the end -- a mirror landing mid-run, or (someday)
# another operator's own output commit. A plain push then gets rejected
# non-fast-forward. Retry with rebase rather than failing outright: our
# own commits only ever touch scripts/gemini/out/ (enforced above), so a
# rebase onto whatever's new on origin/main is always a clean, disjoint
# apply in the ordinary case. If it's ever not clean (a real conflict),
# stop and show the state rather than guessing how to resolve it.
PUSH_ATTEMPTS=3
attempt=1
while true; do
    if git push origin main; then
        echo "Pushed."
        break
    fi
    if ((attempt >= PUSH_ATTEMPTS)); then
        echo "REFUSING: push rejected after $PUSH_ATTEMPTS attempts. Current state:" >&2
        git status >&2
        git log --oneline -5 >&2
        exit 1
    fi
    echo "Push rejected (attempt $attempt/$PUSH_ATTEMPTS) -- origin/main moved;" \
        "pulling and rebasing our output commit onto it..."
    if ! git pull --rebase origin main; then
        echo "REFUSING: rebase did not apply cleanly -- a real conflict, not" >&2
        echo "just the ordinary disjoint scripts/gemini/out/ case. Current state" >&2
        echo "(resolve manually, then 'git rebase --continue' and re-run):" >&2
        git status >&2
        exit 1
    fi
    attempt=$((attempt + 1))
done
