#!/usr/bin/env bash
# Single entry point for everything that runs inside GEMINI. See
# docs/gemini.md's "scripts/gemini/run.sh" section for the full picture:
# nobody can log into the node directly, so this is what actually runs there.
#
# Usage (on the GEMINI node, from the repo root):
#   scripts/gemini/run.sh [probe|schema|env-gpu|extract-dry|extract|finalize|export-codes|train-smoke|train-smoke-2|train-full|eval-forecast <run-name>|train|eval|all]
#
# Steps:
#   probe        scripts/gemini/probe_env.sh -> scripts/gemini/out/env_probe.txt
#   schema       scripts/gemini/explore_schema.py -> scripts/gemini/out/schema.{json,md}
#   env-gpu      builds the H200 training venv (pinned torch==2.6.0, plain
#                PyPI -- see the install step's own comment for why not the
#                +cu124 channel -- plus the two-step mamba-ssm CUDA
#                rebuild, docs/gemini.md's recipe)
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
#   export-codes scripts/gemini/export_codes.py -> reads finalize's
#                metadata/codes.parquet and writes scripts/gemini/out/
#                codes_inventory.json, every distinct code with a suppressed
#                count (rounded to the nearest 1000, or "<1000") -- needs
#                `finalize` done. Unblocks the exhaustive OMOP->LOINC mapping
#                work; code strings aren't patient data, only small counts
#                are suppressed. Falls back to dropping the "<1000" entries
#                if the full inventory would exceed the commit-size cap
#                below (real risk at GEMINI's ~13-50k-code scale, not
#                hypothetical -- see the script's own docstring).
#   train-smoke  proves data+env+backbone on a small slice before committing
#                to a full run: model_kind=baseline, source=gemini,
#                concepts/alerts off (event_hazards=False), backbone stays
#                the default "hybrid". max_train_shards=5, max_tuning_shards=2
#                -- output under ~/runs/gemini_smoke_1. Needs env-gpu and
#                finalize already done. Real training, not a check -- can
#                still take a while at full hidden_size/layers, so this too
#                warns about tmux/screen.
#   train-smoke-2 same as train-smoke but max_train_shards=30, no
#                max_tuning_shards cap -- output under ~/runs/gemini_smoke_2.
#                Run only after train-smoke completes clean end to end
#                (train + checkpoint + provenance written); a 30-shard epoch
#                on an untested node is a long time to wait for a crash the
#                5-shard run would have surfaced identically.
#   train-full   the real run: all train shards (no max_train_shards cap),
#                num_lanes=64/chunk_size=512 (exact full_run_v8 geometry, for
#                cross-dataset comparability -- the smokes ran at the
#                TrainingConfig default 8x256, 1/16th this throughput, which
#                is why they showed ~3GB/20% GPU util), model_kind=baseline,
#                source=gemini, concepts/alerts off, num_epochs=2,
#                checkpoint_every=2000 -- output under
#                ~/runs/gemini_full_14m_v1. Echoes an estimated duration
#                before launching (arithmetic, not measured -- the run's own
#                heartbeat is the real signal, especially whether steps/s
#                holds up at 16x the smokes' consumption rate or the CPU feed
#                becomes the bottleneck).
#   eval-forecast <run-name>
#                run_inference against data/held_out for the named run under
#                ~/runs/ (e.g. `eval-forecast gemini_full_14m_v1`) --
#                forecast/concept/orthogonality metrics, written to
#                ~/runs/<run-name>/eval_forecast.json. max_shards/num_lanes
#                default to 20/16 (a first read, not the full held-out set),
#                overridable via GEMINI_EVAL_MAX_SHARDS/GEMINI_EVAL_NUM_LANES
#                so later ladder rungs don't need another code change. Takes
#                the run name as a second positional argument, not a step of
#                its own config, so it serves every ladder rung the same way.
#   train        not built yet (a general, non-GEMINI-specific full run).
#   eval         not built yet.
#   all          probe, schema, extract-dry, in order (default; deliberately
#                excludes env-gpu, extract, finalize, train-smoke*,
#                train-full, and eval-forecast -- see below)
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
#
# Wrapped in a single main(): bash parses an entire function body as one
# syntactic unit before executing any of it, so once `main "$@"` runs, the
# whole script is already safely parsed into memory -- immune to
# sync_with_mirror (called from within main, near the top) replacing this
# very file on disk mid-run. Real incident, 2026-08-22: the first live sync
# that actually changed run.sh mid-invocation (the torch-install fix) left
# bash reading stale top-level byte offsets into the NEW file, executing a
# hybrid of old and new code -- the log showed the OLD torch==2.6.0+cu124
# install line running even though HEAD had already moved to the fixed
# commit. Without this wrapper, top-level statements are read from disk one
# at a time as execution reaches them; wrapping everything in one function
# forces bash to fully parse it before `main "$@"` starts executing any of it.
main() {
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
    # predictable divergence hit in practice. Fetch + hard reset to origin/main is
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
    # here, and a real extract crashed on ModuleNotFoundError before
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

    run_export_codes() {
        echo "=== export-codes ==="
        echo "Reads metadata/codes.parquet (needs finalize done) and writes"
        echo "the suppressed code inventory to scripts/gemini/out/codes_inventory.json."
        python scripts/gemini/export_codes.py
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

        # Real incident, 2026-08-22: the GPU venv's system python3.12 had no
        # dev headers at all (/usr/include/python3.12 empty/absent), so
        # mamba-ssm's CUDA extension build got most of the way through (the
        # sm_90 kernels compiled fine) and then died on `Python.h: No such
        # file or directory` -- a ~30 minute compile in, not at venv-creation
        # time. Same probe-first philosophy as the CUDA discovery above:
        # check the interpreter that's actually going to build the venv has
        # headers BEFORE creating anything. PYTHON_FOR_GPU_VENV is the
        # escape hatch for whatever this node's real fix turns out to be
        # (an environment-modules python, an already-on-disk uv-managed
        # python, a python3.12-dev install) without needing another code
        # change to point at it.
        PYTHON_FOR_GPU_VENV="${PYTHON_FOR_GPU_VENV:-python3}"
        PYTHON_INCLUDE_DIR=$(
            "$PYTHON_FOR_GPU_VENV" -c \
                "import sysconfig; print(sysconfig.get_path('include'))" \
                2>/dev/null || true
        )
        if [[ -z "$PYTHON_INCLUDE_DIR" || ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
            echo "No Python.h found for '$PYTHON_FOR_GPU_VENV' (checked" >&2
            echo "${PYTHON_INCLUDE_DIR:-<python -c sysconfig lookup failed>}/Python.h)." >&2
            echo "mamba-ssm's CUDA extension build needs Python dev headers --" >&2
            echo "the GPU venv can't be built from a header-less interpreter." >&2
            echo "Point PYTHON_FOR_GPU_VENV at a header-bearing python3.12 and" >&2
            echo "re-run, e.g.:" >&2
            echo "  PYTHON_FOR_GPU_VENV=/path/to/python3.12 scripts/gemini/run.sh env-gpu" >&2
            echo "On the GEMINI H200 node specifically: conda at /opt/Miniconda" >&2
            echo "is configured against an internal mirror (packages.gemini-hpc.ca," >&2
            echo "not blocked by the proxy that blocks pytorch.org/uv's python" >&2
            echo "downloads) -- this is the confirmed working route there:" >&2
            echo "  conda create -y -p ~/py312 python=3.12" >&2
            echo "  PYTHON_FOR_GPU_VENV=~/py312/bin/python3.12 scripts/gemini/run.sh env-gpu" >&2
            echo "On a different node, other candidates: an environment-modules" >&2
            echo "python ('module avail python', then 'module load <name>' before" >&2
            echo "re-running this with that module's python3 on PATH), an" >&2
            echo "already-on-disk uv-managed python ('uv python list' --" >&2
            echo "uv's own python *downloads* are likely proxy-blocked, same as" >&2
            echo "pytorch.org, so only a python uv already has helps), or a" >&2
            echo "python3.12-dev package from a node admin." >&2
            exit 1
        fi
        echo "Using $PYTHON_FOR_GPU_VENV for the GPU venv (Python.h found at $PYTHON_INCLUDE_DIR)."

        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        # venv *creation* doesn't need headers -- only the mamba-ssm compile
        # later does -- so a venv already sitting at $GPU_VENV from a prior,
        # header-less attempt looks perfectly valid to the plain
        # bin/activate check below and would otherwise get silently reused,
        # sending a re-run with a now-fixed PYTHON_FOR_GPU_VENV straight
        # back into the same Python.h failure ~30 minutes in. Check the
        # existing venv's own python for headers first and wipe it if it's
        # the stale, header-less kind.
        if [[ -f "$GPU_VENV/bin/activate" ]]; then
            existing_include=$(
                "$GPU_VENV/bin/python3" -c \
                    "import sysconfig; print(sysconfig.get_path('include'))" \
                    2>/dev/null || true
            )
            if [[ -z "$existing_include" || ! -f "$existing_include/Python.h" ]]; then
                echo "Existing GPU venv at $GPU_VENV has no Python.h (built from" >&2
                echo "a header-less interpreter) -- wiping and recreating from" >&2
                echo "$PYTHON_FOR_GPU_VENV." >&2
                rm -rf "$GPU_VENV"
            fi
        fi
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "Creating GPU venv at $GPU_VENV"
            rm -rf "$GPU_VENV"
            "$PYTHON_FOR_GPU_VENV" -m venv "$GPU_VENV"
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
            # Real incident (first live H200 run): the node's proxy 403s
            # download.pytorch.org ("Tunnel connection failed") while pypi.org
            # is demonstrably reachable -- every other install tonight went
            # through it. Plain PyPI's linux wheel for torch==2.6.0 IS the
            # CUDA 12.4 build (pulls its own nvidia-* runtime packages from
            # PyPI), functionally equivalent to the +cu124 channel for the
            # mamba-ssm compile below (which needs the system nvcc the probe
            # above already validated, plus torch's headers) -- so drop the
            # index entirely rather than routing around a proxy block.
            # Version string differs (bare "2.6.0" here vs "2.6.0+cu124" from
            # the pytorch.org channel): the startswith('2.6.0') check above
            # matches either, and env_fingerprint's canary comparison is
            # numeric, so nothing downstream string-matches the suffix.
            echo "Installing torch==2.6.0 (plain PyPI -- see comment above)..."
            pip install -q torch==2.6.0
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
        #
        # huggingface_hub/packaging/transformers below are NOT this project's
        # own imports -- they're mamba-ssm 2.3.0's own eager runtime imports,
        # found by downloading its sdist and grepping every top-level import
        # in the package: mamba_ssm/__init__.py unconditionally imports
        # models.mixer_seq_simple, which imports utils.hf (transformers.utils,
        # transformers.utils.hub) and utils.generation (transformers.generation)
        # eagerly; modules.mamba2 imports huggingface_hub.PyTorchModelHubMixin
        # and (via ops.triton.ssd_combined) packaging.version, also eagerly.
        # Real incident, 2026-08-22: the first live env-gpu run's mamba-ssm
        # compile succeeded, but the post-build import check then failed on
        # a missing huggingface_hub -- invisible to the odyssey/-only import
        # audit above since none of odyssey's own code imports it; only
        # auditing mamba-ssm's own source surfaces it. triton is NOT listed
        # here despite heavy use throughout mamba_ssm/ops/triton/ -- it's a
        # real dependency of torch's own Linux+CUDA wheel, confirmed present
        # transitively by the fact that the real compile above only failed on
        # huggingface_hub, never triton.
        echo "Installing odyssey (editable, no deps) + training runtime deps..."
        pip install -q -e . --no-deps
        pip install -q "polars>=1.30.0" "scikit-learn>=1.7.0" "einops>=0.7.0" \
            "huggingface_hub>=0.20.0" "packaging>=23.0" "transformers>=4.40.0"

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

    run_train_smoke() {
        # Shared by train-smoke (5 shards) and train-smoke-2 (30 shards) --
        # same config otherwise, per the two-command "fail fast, then commit"
        # smoke sequence: a 5-shard run surfaces any data/env/backbone crash
        # in minutes; only relaunch at 30 once that's clean end to end.
        local max_train_shards="$1"
        local max_tuning_shards="$2" # empty = uncapped (TrainingConfig default)
        local run_name="$3"

        echo "=== $STEP ($run_name, max_train_shards=$max_train_shards) ==="
        echo "Proving data+env+backbone on GEMINI: model_kind=baseline,"
        echo "source=gemini, concepts/alerts off, backbone=hybrid (default)."
        echo "This is a real training run, not a quick check."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: this doesn't look like a tmux or screen session --" >&2
            echo "if the SSH connection drops, training dies with it." >&2
            echo "Run it detached instead, e.g.:" >&2
            echo "  tmux new -s $run_name 'scripts/gemini/run.sh $STEP'" >&2
            echo "  # or: nohup scripts/gemini/run.sh $STEP > $run_name.log 2>&1 &" >&2
        fi

        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "GPU venv not found at $GPU_VENV -- run 'scripts/gemini/run.sh env-gpu' first." >&2
            exit 1
        fi

        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        TRAIN_SHARD_DIR="$MEDS_DIR/data/train"
        TUNING_SHARD_DIR="$MEDS_DIR/data/tuning"
        if [[ ! -d "$TRAIN_SHARD_DIR" || ! -d "$TUNING_SHARD_DIR" ]]; then
            echo "Expected finalized MEDS data at $TRAIN_SHARD_DIR and" >&2
            echo "$TUNING_SHARD_DIR -- run 'scripts/gemini/run.sh finalize' first." >&2
            exit 1
        fi

        OUTPUT_DIR="$HOME/runs/$run_name"
        mkdir -p "$OUTPUT_DIR"
        CONFIG_JSON="$OUTPUT_DIR/smoke_config.json"

        # Overrides only -- everything not listed here (backbone, hidden_size,
        # concept_supervision, ...) stays whatever TrainingConfig's own default
        # is, same as the reviewed draft. Passed as argv, not interpolated into
        # the python source, so shard counts never touch string formatting.
        python3 - "$CONFIG_JSON" "$max_train_shards" "$max_tuning_shards" <<'PY'
import json
import sys

config_path, max_train_shards, max_tuning_shards = sys.argv[1:4]
overrides = {
    "model_kind": "baseline",
    "source": "gemini",
    "stream_shards": True,
    "event_hazards": False,
    "max_train_shards": int(max_train_shards),
}
if max_tuning_shards:
    overrides["max_tuning_shards"] = int(max_tuning_shards)
with open(config_path, "w") as f:
    json.dump(overrides, f, indent=2)
PY

        echo "Run dir: $OUTPUT_DIR"
        echo "Config ($CONFIG_JSON):"
        cat "$CONFIG_JSON"
        echo "Once running, tail progress with:"
        echo "  tail -f $OUTPUT_DIR/loss_log.jsonl"

        source "$GPU_VENV/bin/activate"
        python -m odyssey.training.train \
            --train-shard-dir "$TRAIN_SHARD_DIR" \
            --tuning-shard-dir "$TUNING_SHARD_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --config-json "$CONFIG_JSON"
        deactivate
        # Back to the lightweight venv, same reasoning as env-gpu: a later
        # step in the same invocation (e.g. `all`) shouldn't inherit torch/
        # mamba-ssm on its path.
        source "$VENV/bin/activate"

        echo "$STEP complete. Checkpoints and loss_log.jsonl under $OUTPUT_DIR"
    }

    run_train_full() {
        local run_name="gemini_full_14m_v1"

        echo "=== train-full ($run_name) ==="
        echo "Real training run: all train shards, num_lanes=64/chunk_size=512"
        echo "(exact full_run_v8 geometry, for cross-dataset comparability),"
        echo "model_kind=baseline, source=gemini, concepts/alerts off."
        echo "Estimated (arithmetic, not measured): ~32.7k steps/epoch at"
        echo "64 lanes x 512 chunk over ~1.07B train events; 1-1.5h/epoch if"
        echo "throughput holds -- the heartbeat below is the real signal,"
        echo "especially whether steps/s holds up at 16x the smokes' rate or"
        echo "the CPU feed becomes the bottleneck."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: this doesn't look like a tmux or screen session --" >&2
            echo "if the SSH connection drops, training dies with it." >&2
            echo "Run it detached instead, e.g.:" >&2
            echo "  tmux new -s $run_name 'scripts/gemini/run.sh $STEP'" >&2
            echo "  # or: nohup scripts/gemini/run.sh $STEP > $run_name.log 2>&1 &" >&2
        fi

        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "GPU venv not found at $GPU_VENV -- run 'scripts/gemini/run.sh env-gpu' first." >&2
            exit 1
        fi

        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        TRAIN_SHARD_DIR="$MEDS_DIR/data/train"
        TUNING_SHARD_DIR="$MEDS_DIR/data/tuning"
        if [[ ! -d "$TRAIN_SHARD_DIR" || ! -d "$TUNING_SHARD_DIR" ]]; then
            echo "Expected finalized MEDS data at $TRAIN_SHARD_DIR and" >&2
            echo "$TUNING_SHARD_DIR -- run 'scripts/gemini/run.sh finalize' first." >&2
            exit 1
        fi

        OUTPUT_DIR="$HOME/runs/$run_name"
        mkdir -p "$OUTPUT_DIR"
        CONFIG_JSON="$OUTPUT_DIR/train_full_config.json"

        # No max_train_shards key at all -- TrainingConfig's own default
        # (None = every shard) is what "all train shards" means here.
        cat > "$CONFIG_JSON" <<'JSON'
{
  "model_kind": "baseline",
  "source": "gemini",
  "stream_shards": true,
  "event_hazards": false,
  "num_lanes": 64,
  "chunk_size": 512,
  "num_epochs": 2,
  "checkpoint_every": 2000
}
JSON

        echo "Run dir: $OUTPUT_DIR"
        echo "Config ($CONFIG_JSON):"
        cat "$CONFIG_JSON"
        echo "Once running, tail progress with:"
        echo "  tail -f $OUTPUT_DIR/loss_log.jsonl"

        source "$GPU_VENV/bin/activate"
        python -m odyssey.training.train \
            --train-shard-dir "$TRAIN_SHARD_DIR" \
            --tuning-shard-dir "$TUNING_SHARD_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --config-json "$CONFIG_JSON"
        deactivate
        source "$VENV/bin/activate"

        echo "$STEP complete. Checkpoints and loss_log.jsonl under $OUTPUT_DIR"
    }

    run_eval_forecast() {
        local run_name="$1"
        if [[ -z "$run_name" ]]; then
            echo "eval-forecast needs a run name: scripts/gemini/run.sh eval-forecast <run-name>" >&2
            echo "e.g.: scripts/gemini/run.sh eval-forecast gemini_full_14m_v1" >&2
            exit 1
        fi
        local max_shards="${GEMINI_EVAL_MAX_SHARDS:-20}"
        local num_lanes="${GEMINI_EVAL_NUM_LANES:-16}"

        echo "=== eval-forecast ($run_name) ==="
        echo "Forecast/concept/orthogonality metrics against data/held_out."
        echo "max_shards=$max_shards, num_lanes=$num_lanes (override via"
        echo "GEMINI_EVAL_MAX_SHARDS/GEMINI_EVAL_NUM_LANES for later ladder rungs)."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: this doesn't look like a tmux or screen session --" >&2
            echo "if the SSH connection drops, eval dies with it." >&2
            echo "Run it detached instead, e.g.:" >&2
            echo "  tmux new -s eval-$run_name 'scripts/gemini/run.sh $STEP $run_name'" >&2
            echo "  # or: nohup scripts/gemini/run.sh $STEP $run_name > eval-$run_name.log 2>&1 &" >&2
        fi

        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "GPU venv not found at $GPU_VENV -- run 'scripts/gemini/run.sh env-gpu' first." >&2
            exit 1
        fi

        RUN_DIR="$HOME/runs/$run_name"
        if [[ ! -d "$RUN_DIR" ]]; then
            echo "No run directory at $RUN_DIR -- run the training step that" >&2
            echo "produces it first (e.g. train-full)." >&2
            exit 1
        fi

        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        HELD_OUT_SHARD_DIR="$MEDS_DIR/data/held_out"
        if [[ ! -d "$HELD_OUT_SHARD_DIR" ]]; then
            echo "Expected finalized MEDS data at $HELD_OUT_SHARD_DIR -- run" >&2
            echo "'scripts/gemini/run.sh finalize' first." >&2
            exit 1
        fi

        OUTPUT_JSON="$RUN_DIR/eval_forecast.json"
        echo "Run dir: $RUN_DIR"
        echo "Output: $OUTPUT_JSON"

        source "$GPU_VENV/bin/activate"
        python -m odyssey.inference.run_inference \
            --run-dir "$RUN_DIR" \
            --held-out-shard-dir "$HELD_OUT_SHARD_DIR" \
            --output-json "$OUTPUT_JSON" \
            --max-shards "$max_shards" \
            --num-lanes "$num_lanes"
        deactivate
        source "$VENV/bin/activate"

        echo "$STEP complete. Results at $OUTPUT_JSON"
    }

    case "$STEP" in
        probe) run_probe ;;
        schema) run_schema ;;
        env-gpu) run_env_gpu ;;
        extract-dry) run_extract_dry ;;
        extract) run_extract ;;
        finalize) run_finalize ;;
        export-codes) run_export_codes ;;
        train-smoke) run_train_smoke 5 2 gemini_smoke_1 ;;
        train-smoke-2) run_train_smoke 30 "" gemini_smoke_2 ;;
        train-full) run_train_full ;;
        eval-forecast) run_eval_forecast "${2:-}" ;;
        train) run_pending_stub train ;;
        eval) run_pending_stub eval ;;
        all) run_probe; run_schema; run_extract_dry ;;
        *)
            echo "unknown step: $STEP (expected probe, schema, env-gpu, extract-dry, extract, finalize, export-codes, train-smoke, train-smoke-2, train-full, eval-forecast, train, eval, or all)" >&2
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
}

main "$@"
