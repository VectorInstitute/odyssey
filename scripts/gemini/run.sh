#!/usr/bin/env bash
# Single entry point for everything that runs inside GEMINI. See
# docs/gemini.md's "scripts/gemini/run.sh" section for the full picture:
# nobody can log into the node directly, so this is what actually runs there.
#
# Usage (on the GEMINI node, from the repo root):
#   scripts/gemini/run.sh [probe|schema|env-gpu|extract-dry|extract|finalize|export-codes|pipeline|train-smoke|train-smoke-2|train-full|train-smoke-cbm|train-full-cbm|train-smoke-dec|train-full-dec|train-rung2|eval-forecast <run-name>|alerts <run-name>|tabicl <run-name>|steering <run-name>|atlas <run-name>|train|eval|all]
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
#   pipeline     chains extract -> finalize -> export-codes end to end, each
#                stage's own guards unchanged. Adds two chain-level guards
#                on top, both from a real incident (2026-08-22): finalize
#                was OOM-killed by the kernel 9 minutes into its 1.2B-row
#                repartition scan, run outside any Slurm allocation. (1) a
#                memory preflight upfront (checks SLURM_MEM_PER_NODE, or
#                /proc/meminfo directly if unset) that refuses in seconds,
#                not 9 minutes, with the exact salloc command, if under
#                ~64G. (2) a partial-output guard: if the OOM kill (or any
#                other) left GEMINI_MEDS_OUTPUT_DIR/data/ behind with no
#                metadata/.finalize_complete sentinel, refuses with the
#                exact rm commands rather than letting finalize's own
#                auto-wipe (see finalize_meds.py's _wipe_partial_output)
#                silently redo the work unattended -- this chain is meant
#                to run unwatched, so an ambiguous partial state should
#                stop and ask, not guess.
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
#   train-smoke-cbm  G3 SMOKE: the concept-bottleneck path on 5 train
#                shards. Run this BEFORE train-full-cbm -- concept labels,
#                visit scoping and hazard targets are a new failure
#                surface on GEMINI data, and a crash surfaces in minutes
#                here rather than hours into the full run.
#   train-full-cbm   G3: the full-scale concept-bottleneck run
#                (~/runs/gemini_full_v10), mirroring R2/R6's recipe --
#                model_kind=bottleneck, task_set v3 (15 of 29 concepts
#                resolve on GEMINI, see scripts/gemini/out/
#                concept_audit.json), concept_supervision=visit, hazard
#                heads on with event_head_hidden=256, value_embeddings,
#                num_lanes=64/chunk_size=512, reset_prob=0.0, 2 epochs.
#                normalize_medications stays FALSE here: that normalizer
#                targets MIMIC/eICU code shapes, and GEMINI's medication
#                codes are already clean. This is the run every GEMINI
#                trust-audit cell in the paper comes from; the baseline
#                train-full/train-rung2 steps produce no readout,
#                completeness or lever result at all.
#   train-smoke-dec  smoke of the DECOMPOSED bottleneck (5 train shards,
#                ~/runs/gemini_smoke_dec). Run before train-full-dec.
#   train-full-dec   the decomposed concept bottleneck (Steerling
#                h = k + u + eps) at full scale, ~/runs/gemini_full_DEC_v12:
#                train-full-cbm's recipe plus the v12 decomposition block
#                (bottleneck_kind=decomposed, unknown_ratio 3, residual
#                dropout 0.3, reconstruction/independence losses,
#                teacher forcing 1.0 -> 0.5 over 4,500 steps), verbatim
#                from the MIMIC/eICU v12 runs so the three datasets'
#                decomposition cells are comparable. Needs odyssey at
#                69c8cb0 or later. The steering and atlas steps, and the
#                alerts step's hazard column, are meant to read THIS run.
#   train-rung2  ladder rung 2 (G4, docs/experiment_plan.md): same geometry
#                as train-full, one deliberate delta -- hidden_size 256->512,
#                an estimated ~60M parameters (the real, measured count logs
#                at training init, same as every step). CRITICAL: trains on
#                the quarantined 12-table dataset G2/G3 already used
#                (GEMINI_MEDS_OUTPUT_DIR/data_12table_trainfull by default,
#                overridable via GEMINI_LADDER_DATA_DIR), never the newer
#                18-table finalize output -- the scaling curve holds data
#                fixed while varying model scale, so a schema change here
#                would confound the comparison. Output under
#                ~/runs/gemini_rung2_60m_v1. Periodic (non-best/final/epoch)
#                checkpoints beyond the most recent
#                GEMINI_RUNG2_KEEP_CHECKPOINTS (default 3) are pruned in the
#                background as training runs, since a ~60M model's
#                checkpoints are large enough at checkpoint_every=2000 to
#                threaten disk over a full 2-epoch run.
#   steering <run-name>
#                the clinical dial benchmark on a decomposed run: pushes
#                each concept's direction into the residual stream
#                (Steerling Sec. 6.2, gamma = tau / max alignment) and
#                reports, per dial, the change in every hazard head's risk
#                over at-risk patients with a paired subject bootstrap,
#                against the clinical expectation table in
#                odyssey/inference/steering.py. Every held-out shard by
#                default (GEMINI_STEERING_MAX_SHARDS caps it), 16 lanes.
#                Writes ~/runs/<run-name>/steering_full.json and exports it
#                (aggregate-only) to scripts/gemini/out/evals/.
#   atlas <run-name>
#                the concept atlas for the paper appendix: per known and
#                unknown concept, the tokens its direction promotes and
#                suppresses through the next-event head, and the share of
#                the next-event logit carried by named / unknown / residual
#                parts. Writes ~/runs/<run-name>/concept_atlas.json and
#                exports it to scripts/gemini/out/evals/.
#   eval-forecast <run-name>
#                run_inference against data/held_out for the named run under
#                ~/runs/ (e.g. `eval-forecast gemini_full_14m_v1`) --
#                forecast/concept/orthogonality metrics, written to
#                ~/runs/<run-name>/eval_forecast.json. max_shards/num_lanes
#                default to 20/16 (a first read, not the full held-out set),
#                overridable via GEMINI_EVAL_MAX_SHARDS/GEMINI_EVAL_NUM_LANES
#                so later ladder rungs don't need another code change. The
#                held-out dir itself defaults to data/held_out under
#                GEMINI_MEDS_OUTPUT_DIR but is overridable via
#                GEMINI_EVAL_HELD_OUT_DIR -- ladder runs (train-rung2 and
#                later) must point this at the same quarantined 12-table
#                held_out the training run itself used; train-rung2's own
#                launch report prints the exact command. Takes the run name
#                as a second positional argument, not a step of its own
#                config, so it serves every ladder rung the same way.
#                Also exports this run's own aggregate-only metrics to
#                scripts/gemini/out/evals/<run-name>_eval_forecast.json
#                (picked up by the usual small-output commit+push, same
#                mechanism as extraction_summary.json) after validating it
#                contains only known InferenceResults keys -- whitelist,
#                never blacklist, so anything unrecognized refuses loudly
#                instead of exporting. Also backfills any other
#                ~/runs/*/eval_forecast.json not yet exported (e.g. earlier
#                runs evaluated before this existed), so results stop
#                arriving in the shared record by manual paste.
#   train        not built yet (a general, non-GEMINI-specific full run).
#   eval         not built yet.
#   all          probe, schema, extract-dry, in order (default; deliberately
#                excludes env-gpu, extract, finalize, train-smoke*,
#                train-full, train-rung2, and eval-forecast -- see below)
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

    run_pipeline() {
        echo "=== pipeline (extract -> finalize -> export-codes) ==="
        echo "Each stage's own guards (manifest checks, disk/NOFILE preflight,"
        echo "finalize's completion sentinel) are unchanged -- this just"
        echo "sequences them. Two chain-level guards below, both from a real"
        echo "incident: finalize was OOM-killed 2026-08-22, 9 minutes into its"
        echo "repartition pass, running outside any Slurm allocation."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: this doesn't look like a tmux or screen session --" >&2
            echo "if the SSH connection drops, the whole chain dies with it," >&2
            echo "and extract alone can run for hours." >&2
            echo "Run it detached instead, e.g.:" >&2
            echo "  tmux new -s pipeline 'scripts/gemini/run.sh pipeline'" >&2
            echo "  # or: nohup scripts/gemini/run.sh pipeline > pipeline.log 2>&1 &" >&2
        fi

        # --- memory preflight ---------------------------------------------
        # Checked upfront, before extract even starts, not just before
        # finalize -- no point burning hours on extract only to hit the same
        # OOM at the finalize stage. Prefers SLURM_MEM_PER_NODE (MB, Slurm's
        # own convention) when set; falls back to /proc/meminfo's
        # MemAvailable when this isn't running inside an allocation at all,
        # which is exactly the condition that caused the real kill.
        local min_mem_gb=64
        # The refuse-below threshold (min_mem_gb) and the salloc amount we
        # actually recommend are deliberately different numbers: 64G is
        # where we refuse, but the only finalize run that ever SUCCEEDED at
        # the full 1.2B rows used a 96G allocation, and the 2026-08-22 kill
        # means the true peak is still unknown -- recommend the known-good
        # number, not just the refusal floor.
        local recommend_mem_gb=96
        local avail_kb avail_gb
        if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
            avail_gb=$((SLURM_MEM_PER_NODE / 1024))
            echo "Memory preflight: SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE} MB (~${avail_gb} GB) allocated."
        else
            avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
            avail_gb=$((${avail_kb:-0} / 1024 / 1024))
            echo "Memory preflight: no Slurm allocation detected (SLURM_MEM_PER_NODE"
            echo "unset) -- reading /proc/meminfo directly: ~${avail_gb} GB available."
        fi
        if ((avail_gb < min_mem_gb)); then
            echo "REFUSING to start: ~${avail_gb} GB available, need at least" >&2
            echo "${min_mem_gb} GB -- finalize's repartition pass was OOM-killed" >&2
            echo "by the kernel under exactly this condition (2026-08-22, 9" >&2
            echo "minutes into a 1.2B-row scan, no Slurm allocation). Get a real" >&2
            echo "allocation first, e.g.:" >&2
            echo "  salloc --mem=${recommend_mem_gb}G --cpus-per-task=8 --time=08:00:00" >&2
            echo "(${recommend_mem_gb}G is the known-good number -- the only" >&2
            echo "finalize run that succeeded at the full 1.2B rows used this" >&2
            echo "much; the true peak is still unknown after the kill, so this" >&2
            echo "is a safety margin above the ${min_mem_gb}G refusal floor," >&2
            echo "not the floor itself. (Adjust partition/account flags for" >&2
            echo "GEMINI's actual Slurm config if this doesn't match it -- not" >&2
            echo "documented in this repo.) Then re-run" >&2
            echo "'scripts/gemini/run.sh pipeline' inside that allocation's shell." >&2
            exit 1
        fi
        echo "Memory preflight OK (~${avail_gb} GB >= ${min_mem_gb} GB)."

        # --- partial-output guard -------------------------------------------
        # Same incident: the kill left data/ partially written under
        # GEMINI_MEDS_OUTPUT_DIR with no metadata/.finalize_complete sentinel
        # (finalize_meds.py's own completion marker). Run standalone,
        # finalize would auto-wipe this itself and start over (see
        # finalize_meds.py's _wipe_partial_output) -- fine for an operator
        # watching the run, but this chain is meant to run unattended, so a
        # partial, ambiguous prior attempt should stop and ask a human
        # instead of silently redoing possibly-hours of work on an
        # assumption. Mirrors finalize's own refusal doctrine (a specific
        # error naming the exact state and the exact fix), applied here
        # rather than left to finalize's internal auto-wipe.
        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        if [[ -d "$MEDS_DIR/data" && ! -f "$MEDS_DIR/metadata/.finalize_complete" ]]; then
            echo "REFUSING to start: $MEDS_DIR/data exists but" >&2
            echo "$MEDS_DIR/metadata/.finalize_complete does not -- a partial," >&2
            echo "incomplete finalize output from a prior run (e.g. the" >&2
            echo "2026-08-22 OOM kill), not a completed one. Confirm by hand" >&2
            echo "before re-running the unattended chain over it. If the" >&2
            echo "partial output is expected garbage, delete it and re-run:" >&2
            echo "  rm -rf $MEDS_DIR/data $MEDS_DIR/metadata" >&2
            echo "If it's something else, look before deleting." >&2
            exit 1
        fi

        echo "--- stage 1/3: extract ---" && run_extract &&
            echo "--- stage 2/3: finalize ---" && run_finalize &&
            echo "--- stage 3/3: export-codes ---" && run_export_codes &&
            echo "=== pipeline complete: extract -> finalize -> export-codes ==="
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

    run_train_cbm() {
        # G3: the concept-bottleneck run the paper actually needs. Every
        # other train step here is model_kind=baseline (forecasting only),
        # which produces no readout, no completeness cell and no lever --
        # none of the trust-audit results GEMINI is in the paper for.
        #
        # Mirrors R2/R6's recipe (full_run_v10 / eicu_full_v10 config.json)
        # so the three datasets' cells are comparable: bottleneck over
        # task_set v3, visit-scoped concept supervision, hazard heads with
        # a 256-wide readout, value embeddings, 64 lanes x 512 chunk,
        # reset_prob 0.0, 2 epochs. Two GEMINI-specific deltas, both
        # forced by the data rather than chosen:
        #   * task_set v3 resolves to 15 of the 29 concepts here (G2,
        #     scripts/gemini/out/concept_audit.json) -- concepts_for_source
        #     drops the rest automatically at load, and alert_events_for
        #     drops sepsis3's alert with its concept.
        #   * normalize_medications stays FALSE. The normalizer is built
        #     for MIMIC's sig-line/NDC fragmentation (and eICU's HICL
        #     dictionary); GEMINI's medication codes are already clean
        #     3-part strings, so running it would reshape codes for no
        #     gain. R2/R6 both set it true for their own sources.
        local max_train_shards="$1"   # empty = every shard
        local run_name="$2"
        local bottleneck_kind="${3:-mixture}"   # mixture (v10) | decomposed (v12)

        echo "=== $STEP ($run_name) ==="
        if [[ "$bottleneck_kind" == "decomposed" ]]; then
            echo "DECOMPOSED bottleneck (Steerling h = k + u + eps; the v12 recipe"
            echo "of MIMIC's full_run_DEC_v12 / eICU's eicu_full_DEC_v12): 3 unknown"
            echo "concepts per known one, residual dropout 0.3, reconstruction and"
            echo "independence losses on the unknown head, teacher forcing annealed"
            echo "1.0 -> 0.5 over 4,500 steps. This is the run the steering and"
            echo "atlas steps read."
        fi
        echo "CONCEPT BOTTLENECK run (model_kind=bottleneck), source=gemini,"
        echo "task_set=v3 -> 15 resolving concepts, hazard heads ON,"
        echo "concept_supervision=visit, num_lanes=64/chunk_size=512 --"
        echo "R2/R6 geometry, for cross-dataset comparability."
        if [[ -n "$max_train_shards" ]]; then
            echo "Smoke scale: max_train_shards=$max_train_shards. Run this"
            echo "FIRST -- the bottleneck path (concept labels, hazard"
            echo "targets, visit scoping) is a new failure surface on this"
            echo "data, and a crash surfaces in minutes here instead of"
            echo "hours into the full run."
        else
            echo "Full scale: every train shard."
        fi
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
            echo "$TUNING_SHARD_DIR -- run 'scripts/gemini/run.sh pipeline' first." >&2
            exit 1
        fi

        OUTPUT_DIR="$HOME/runs/$run_name"
        mkdir -p "$OUTPUT_DIR"
        CONFIG_JSON="$OUTPUT_DIR/train_cbm_config.json"

        # Shard count and bottleneck kind passed as argv, never interpolated
        # into the source.
        python3 - "$CONFIG_JSON" "$max_train_shards" "$bottleneck_kind" <<'PY'
import json
import sys

config_path, max_train_shards = sys.argv[1:3]
bottleneck_kind = sys.argv[3] if len(sys.argv) > 3 else "mixture"
overrides = {
    "model_kind": "bottleneck",
    "source": "gemini",
    "stream_shards": True,
    "task_set": "v3",
    "concept_supervision": "visit",
    "concept_pos_weight": True,
    "value_embeddings": True,
    "event_hazards": True,
    "event_head_hidden": 256,
    "time_to_event": True,
    "bundle_invariant_loss": True,
    "normalize_medications": False,
    "num_lanes": 64,
    "chunk_size": 512,
    "reset_prob": 0.0,
    "num_epochs": 2,
    "max_tuning_shards": 10,
    "checkpoint_every": 2000,
}
if max_train_shards:
    overrides["max_train_shards"] = int(max_train_shards)
if bottleneck_kind == "decomposed":
    # Verbatim from ~/runs/full_run_DEC_v12/config.json on the MIMIC VM
    # (commit 69c8cb0 and later); the eICU run used the same block.
    overrides.update({
        "bottleneck_kind": "decomposed",
        "unknown_ratio": 3,
        "residual_dropout": 0.3,
        "concept_global_pairs": False,
        "orthogonality_weight": 0.0,
        "observability_weight": 0.1,
        "reconstruction_weight": 1.0,
        "independence_weight": 1.0,
        "teacher_known_start": 1.0,
        "teacher_known_end": 0.5,
        "teacher_unknown_start": 1.0,
        "teacher_unknown_end": 0.5,
        "teacher_anneal_steps": 4500,
    })
elif bottleneck_kind != "mixture":
    raise SystemExit(f"unknown bottleneck kind {bottleneck_kind!r}")
with open(config_path, "w") as f:
    json.dump(overrides, f, indent=2)
PY

        echo "Run dir: $OUTPUT_DIR"
        echo "Config ($CONFIG_JSON):"
        cat "$CONFIG_JSON"
        echo "Watch the startup lines: the concept loader logs which"
        echo "concepts drop for this source, and the hazard heads log"
        echo "their event names -- both should match G2's audit."
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

    _prune_step_checkpoints() {
        # Deletes all but the KEEP most recent checkpoint_<N>.pt files under
        # DIR, by numeric global_step (not mtime -- a resumed run's mtimes
        # don't necessarily sort the same way its steps do). Can only ever
        # match checkpoint_<digits>.pt: checkpoint_best.pt/checkpoint_final.pt/
        # checkpoint_epoch_<N>.pt all have a non-digit character right after
        # "checkpoint_", so the glob below never matches them -- this must
        # never prune any of those three.
        #
        # Deliberately avoids `ls | ...` (a real pitfall hit writing this: an
        # interactive shell's own `ls` alias reformatted the listing and
        # silently broke the parsing) and GNU-only `head -n -N` (not
        # guaranteed on GEMINI's actual coreutils) -- a plain glob loop plus
        # `head -n <positive count>` is portable to both GNU and BSD.
        local dir="$1"
        local keep="$2"
        local f n pairs sorted total to_delete old
        pairs=""
        for f in "$dir"/checkpoint_[0-9]*.pt; do
            [[ -e "$f" ]] || continue
            n="${f##*/checkpoint_}"
            n="${n%.pt}"
            pairs="$pairs$n $f
"
        done
        [[ -z "$pairs" ]] && return 0
        sorted=$(printf '%s' "$pairs" | sort -n)
        total=$(printf '%s\n' "$sorted" | grep -c .)
        to_delete=$((total - keep))
        if [[ "$to_delete" -gt 0 ]]; then
            printf '%s\n' "$sorted" | head -n "$to_delete" | awk '{print $2}' | while IFS= read -r old; do
                rm -f -- "$old"
            done
        fi
    }

    _start_checkpoint_pruner() {
        # Backgrounds a loop that calls _prune_step_checkpoints every
        # $interval seconds; prints the loop's own PID so the caller can
        # kill it once training exits (success or failure) -- it must never
        # outlive the training process it was started for.
        local dir="$1"
        local keep="$2"
        local interval="${3:-300}"
        (
            while true; do
                sleep "$interval"
                _prune_step_checkpoints "$dir" "$keep"
            done
        ) &
        echo $!
    }

    run_train_rung2() {
        local run_name="gemini_rung2_60m_v1"
        local keep_checkpoints="${GEMINI_RUNG2_KEEP_CHECKPOINTS:-3}"

        echo "=== train-rung2 ($run_name) ==="
        echo "Ladder rung 2 (G4, docs/experiment_plan.md): same geometry as"
        echo "train-full (num_lanes=64/chunk_size=512, model_kind=baseline,"
        echo "source=gemini, concepts/alerts off, num_epochs=2), one"
        echo "deliberate delta -- hidden_size 256 -> 512, an ESTIMATED ~60M"
        echo "parameters (arithmetic, not measured -- the real, measured"
        echo "count logs at training init: \"[model] N.NM parameters...\", same"
        echo "as every step)."
        echo "Pruning periodic checkpoints beyond the most recent"
        echo "$keep_checkpoints in the background (override via"
        echo "GEMINI_RUNG2_KEEP_CHECKPOINTS) -- checkpoint_best.pt,"
        echo "checkpoint_final.pt, and checkpoint_epoch_*.pt are never pruned."
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

        # CRITICAL: the scaling curve holds data fixed while varying model
        # scale -- G2's 14M point (train-full) trained on this same
        # quarantined 12-table dataset, so rung 2/3 must too, never the newer
        # 18-table finalize output, or model scale and schema change get
        # confounded. Pinnable via GEMINI_LADDER_DATA_DIR since the exact
        # directory name is a real, one-off operator artifact (Amrit's
        # finalize-quarantine step), not something derivable from code.
        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        LADDER_DATA_DIR="${GEMINI_LADDER_DATA_DIR:-$MEDS_DIR/data_12table_trainfull}"
        TRAIN_SHARD_DIR="$LADDER_DATA_DIR/train"
        TUNING_SHARD_DIR="$LADDER_DATA_DIR/tuning"
        if [[ ! -d "$TRAIN_SHARD_DIR" || ! -d "$TUNING_SHARD_DIR" ]]; then
            echo "Expected the quarantined 12-table MEDS data at" >&2
            echo "$TRAIN_SHARD_DIR and $TUNING_SHARD_DIR -- override the path" >&2
            echo "via GEMINI_LADDER_DATA_DIR if the quarantine snapshot lives" >&2
            echo "somewhere else." >&2
            exit 1
        fi
        echo "Data: $LADDER_DATA_DIR (quarantined 12-table dataset -- NOT"
        echo "the current finalize output; override via GEMINI_LADDER_DATA_DIR)"

        OUTPUT_DIR="$HOME/runs/$run_name"
        mkdir -p "$OUTPUT_DIR"
        CONFIG_JSON="$OUTPUT_DIR/train_rung2_config.json"

        # Same as train-full's config, plus hidden_size -- the one
        # deliberate delta this rung exists to measure.
        cat > "$CONFIG_JSON" <<'JSON'
{
  "model_kind": "baseline",
  "source": "gemini",
  "stream_shards": true,
  "event_hazards": false,
  "num_lanes": 64,
  "chunk_size": 512,
  "num_epochs": 2,
  "checkpoint_every": 2000,
  "hidden_size": 512
}
JSON

        echo "Run dir: $OUTPUT_DIR"
        echo "Config ($CONFIG_JSON):"
        cat "$CONFIG_JSON"
        echo "Once running, tail progress with:"
        echo "  tail -f $OUTPUT_DIR/loss_log.jsonl"

        PRUNER_PID=""
        if [[ "$keep_checkpoints" -gt 0 ]]; then
            PRUNER_PID=$(_start_checkpoint_pruner "$OUTPUT_DIR" "$keep_checkpoints" 300)
        fi

        source "$GPU_VENV/bin/activate"
        python -m odyssey.training.train \
            --train-shard-dir "$TRAIN_SHARD_DIR" \
            --tuning-shard-dir "$TUNING_SHARD_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --config-json "$CONFIG_JSON"
        TRAIN_EXIT=$?
        [[ -n "$PRUNER_PID" ]] && kill "$PRUNER_PID" 2>/dev/null
        deactivate
        source "$VENV/bin/activate"

        if [[ "$TRAIN_EXIT" -ne 0 ]]; then
            echo "$STEP failed (exit $TRAIN_EXIT)" >&2
            exit "$TRAIN_EXIT"
        fi
        echo "$STEP complete. Checkpoints and loss_log.jsonl under $OUTPUT_DIR"
        echo "Eval: GEMINI_EVAL_HELD_OUT_DIR=$LADDER_DATA_DIR/held_out scripts/gemini/run.sh eval-forecast $run_name"
    }

    _export_eval_summary() {
        # Validates a run's eval_forecast.json contains ONLY known,
        # whitelisted aggregate InferenceResults keys (odyssey/inference/
        # run_inference.py's own dataclass shape) -- whitelist, never
        # blacklist, so anything not explicitly known refuses loudly instead
        # of silently passing through. If it passes, copies it verbatim to
        # scripts/gemini/out/evals/<run>_eval_forecast.json, where the
        # generic "commit and push only the small, allowed outputs" step at
        # the end of this script picks it up automatically -- the same
        # mechanism extraction_summary.json already rides, no new commit/push
        # logic needed here.
        #
        # Real motivation: eval numbers have been arriving in the shared
        # record by manual paste (sometimes in multiple fragments) because
        # nothing copies them out of ~/runs/ automatically -- this closes
        # that permanently, the same way extraction_summary.json already
        # closed it for extraction.
        local run_name="$1"
        local source_json="$2"
        local dest_json="scripts/gemini/out/evals/${run_name}_eval_forecast.json"

        mkdir -p scripts/gemini/out/evals
        if ! python3 - "$source_json" <<'PY'
import json
import sys

# Mirrors odyssey/inference/run_inference.py's InferenceResults/
# odyssey/training/metrics.py's TaskMetrics/ConceptMetrics/
# ObservabilityMetrics/TimeMetrics dataclasses exactly -- keep in sync by
# hand if those ever gain a field; the whole point of a whitelist is that
# an unrecognized field refuses instead of silently passing through.
TASK_METRICS_KEYS = {
    "cross_entropy", "perplexity", "top1_accuracy", "top5_accuracy",
    "n_predictions", "set_top1_accuracy",
}
CONCEPT_METRICS_KEYS = {
    "name", "n_observed", "prevalence", "auroc", "auprc",
    "brier_score", "accuracy_at_0_5",
}
OBSERVABILITY_METRICS_KEYS = {
    "name", "n_subjects", "observed_rate", "auroc", "accuracy_at_0_5",
}
TIME_METRICS_KEYS = {
    "nll", "n_positions", "same_instant_accuracy", "same_instant_rate",
    "calibration", "calibration_after_bundle",
}
CALIBRATION_ENTRY_KEYS = {"predicted", "observed"}
TOP_LEVEL_KEYS = {
    "task_metrics", "task_metrics_by_code_type", "concept_metrics",
    "observability_metrics", "orthogonality", "n_patient_ends_scored",
    "time_metrics", "tail_slice",
}


def _require_dict(obj, label):
    if not isinstance(obj, dict):
        raise ValueError(f"{label}: expected an object, got {type(obj).__name__}")


def _check_no_extra_keys(obj, allowed, label):
    _require_dict(obj, label)
    extra = set(obj) - allowed
    if extra:
        raise ValueError(
            f"{label}: unexpected key(s) {sorted(extra)} -- not in the known "
            "aggregate-metric whitelist"
        )


def _check_task_metrics(obj, label):
    if obj is None:
        return
    _check_no_extra_keys(obj, TASK_METRICS_KEYS, label)


def _check_calibration_map(obj, label):
    if obj is None:
        return
    _require_dict(obj, label)
    for horizon, entry in obj.items():
        _check_no_extra_keys(entry, CALIBRATION_ENTRY_KEYS, f"{label}[{horizon!r}]")


def _check_time_metrics(obj, label):
    if obj is None:
        return
    _check_no_extra_keys(obj, TIME_METRICS_KEYS, label)
    _check_calibration_map(obj.get("calibration"), f"{label}.calibration")
    _check_calibration_map(
        obj.get("calibration_after_bundle"), f"{label}.calibration_after_bundle"
    )


def validate(obj, label="root"):
    _check_no_extra_keys(obj, TOP_LEVEL_KEYS, label)
    _check_task_metrics(obj.get("task_metrics"), f"{label}.task_metrics")

    by_code_type = obj.get("task_metrics_by_code_type")
    if by_code_type is not None:
        _require_dict(by_code_type, f"{label}.task_metrics_by_code_type")
        for code_type, entry in by_code_type.items():
            _check_task_metrics(
                entry, f"{label}.task_metrics_by_code_type[{code_type!r}]"
            )

    concept_metrics = obj.get("concept_metrics")
    if concept_metrics is not None:
        if not isinstance(concept_metrics, list):
            raise ValueError(f"{label}.concept_metrics: expected a list")
        for i, entry in enumerate(concept_metrics):
            _check_no_extra_keys(
                entry, CONCEPT_METRICS_KEYS, f"{label}.concept_metrics[{i}]"
            )

    observability_metrics = obj.get("observability_metrics")
    if observability_metrics is not None:
        if not isinstance(observability_metrics, list):
            raise ValueError(f"{label}.observability_metrics: expected a list")
        for i, entry in enumerate(observability_metrics):
            _check_no_extra_keys(
                entry,
                OBSERVABILITY_METRICS_KEYS,
                f"{label}.observability_metrics[{i}]",
            )

    _check_time_metrics(obj.get("time_metrics"), f"{label}.time_metrics")

    tail_slice = obj.get("tail_slice")
    if tail_slice is not None:
        validate(tail_slice, f"{label}.tail_slice")


source_path = sys.argv[1]
with open(source_path) as f:
    data = json.load(f)
try:
    validate(data)
except ValueError as exc:
    print(f"REFUSING to export: {exc}", file=sys.stderr)
    sys.exit(1)
PY
        then
            echo "WARNING: not exporting eval summary for $run_name --" >&2
            echo "it contains something outside the known aggregate-metric" >&2
            echo "whitelist (see above); continuing without it rather than" >&2
            echo "failing this step, but this needs a human look." >&2
            return 1
        fi

        cp "$source_json" "$dest_json"
        echo "Exported aggregate eval summary: $dest_json"
    }

    _backfill_eval_summaries() {
        # Sweeps every ~/runs/*/eval_forecast.json not yet exported to
        # scripts/gemini/out/evals/ -- runs evaluated before this export step
        # existed (gemini_smoke_2, gemini_full_14m_v1) had their results
        # reach the shared record only by manual paste. Runs every time
        # eval-forecast does (cheap: a handful of run dirs at most), so it
        # backfills on the first invocation after this step exists without
        # needing a separate one-off script -- and stays a correct no-op on
        # every later invocation once everything is caught up.
        local eval_json run_dir run_name dest_json
        for eval_json in "$HOME"/runs/*/eval_forecast.json; do
            [[ -e "$eval_json" ]] || continue
            run_dir=$(dirname "$eval_json")
            run_name=$(basename "$run_dir")
            dest_json="scripts/gemini/out/evals/${run_name}_eval_forecast.json"
            [[ -f "$dest_json" ]] && continue
            echo "Backfilling eval summary for $run_name (found on node, never exported)..."
            if ! _export_eval_summary "$run_name" "$eval_json"; then
                echo "WARNING: backfill skipped for $run_name (see above)." >&2
            fi
        done
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
        echo "Forecast/concept/orthogonality metrics against held_out."
        echo "max_shards=$max_shards, num_lanes=$num_lanes (override via"
        echo "GEMINI_EVAL_MAX_SHARDS/GEMINI_EVAL_NUM_LANES for later ladder rungs)."
        echo "Held-out dir defaults to data/held_out under GEMINI_MEDS_OUTPUT_DIR;"
        echo "override via GEMINI_EVAL_HELD_OUT_DIR for a ladder run (must match"
        echo "the quarantined dataset that run trained on) -- resolved path"
        echo "printed below once computed."
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
        # Ladder runs (train-rung2 and later) must point this at the same
        # quarantined 12-table held_out their training run used, never the
        # current finalize output -- see train-rung2's own launch report,
        # which prints the exact GEMINI_EVAL_HELD_OUT_DIR command.
        HELD_OUT_SHARD_DIR="${GEMINI_EVAL_HELD_OUT_DIR:-$MEDS_DIR/data/held_out}"
        if [[ ! -d "$HELD_OUT_SHARD_DIR" ]]; then
            echo "Expected finalized MEDS data at $HELD_OUT_SHARD_DIR -- run" >&2
            echo "'scripts/gemini/run.sh finalize' first (or, for a ladder" >&2
            echo "run, check GEMINI_EVAL_HELD_OUT_DIR points at the right" >&2
            echo "quarantined dataset)." >&2
            exit 1
        fi

        OUTPUT_JSON="$RUN_DIR/eval_forecast.json"
        echo "Run dir: $RUN_DIR"
        echo "Held-out: $HELD_OUT_SHARD_DIR"
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

        if ! _export_eval_summary "$run_name" "$OUTPUT_JSON"; then
            echo "WARNING: this run's own eval summary was not exported (see above)." >&2
        fi
        _backfill_eval_summaries
    }

    _export_alerts_summary() {
        # Same mechanism and same reasoning as _export_eval_summary above:
        # validate against a whitelist, then copy verbatim into
        # scripts/gemini/out/evals/, where the generic commit-and-push step
        # at the bottom of this script picks it up. alerts.json is a BARE
        # LIST of one record per (event, horizon, scorer) -- not an object --
        # so the validator walks the list rather than a top-level dict.
        #
        # Everything in it is an aggregate over at-risk rows (counts, AUROC,
        # decile calibration). No patient-level field exists in the record
        # shape, which is what makes this safe to commit; alerts_rows.parquet,
        # which IS patient-level, stays on the node and is refused by the
        # data-like guard at the bottom regardless.
        local run_name="$1"
        local source_json="$2"
        local dest_json="scripts/gemini/out/evals/${run_name}_alerts.json"

        mkdir -p scripts/gemini/out/evals
        if ! python3 - "$source_json" <<'PY'
import json
import sys

# Mirrors odyssey/inference/alerts.py's AlertMetrics dataclass, plus the
# landmark_protocol_version the writer attaches per record. Keep in sync by
# hand if that dataclass gains a field -- an unrecognized key must refuse
# rather than silently ride along into the shared record.
RECORD_KEYS = {
    "event", "horizon_hours", "scorer", "n_at_risk", "n_positive",
    "n_censored", "auroc", "brier", "calibration", "baseline_feature_set",
    "baseline_n_features", "baseline_params", "landmark_protocol_version",
}
CALIBRATION_KEYS = {"predicted", "observed", "n"}

with open(sys.argv[1]) as f:
    obj = json.load(f)

if not isinstance(obj, list):
    raise SystemExit(
        f"alerts.json: expected a bare list of records, got {type(obj).__name__}"
    )
if not obj:
    raise SystemExit("alerts.json: empty -- no cell was scored, refusing to export")

for i, rec in enumerate(obj):
    if not isinstance(rec, dict):
        raise SystemExit(f"record {i}: expected an object, got {type(rec).__name__}")
    extra = set(rec) - RECORD_KEYS
    if extra:
        raise SystemExit(f"record {i}: unrecognized key(s) {sorted(extra)}")
    for required in ("event", "horizon_hours", "scorer", "n_at_risk"):
        if required not in rec:
            raise SystemExit(f"record {i}: missing required key {required!r}")
    cal = rec.get("calibration")
    if cal is not None:
        if not isinstance(cal, list):
            raise SystemExit(f"record {i}: calibration must be a list")
        for j, b in enumerate(cal):
            if not isinstance(b, dict):
                raise SystemExit(f"record {i} calibration[{j}]: expected an object")
            cal_extra = set(b) - CALIBRATION_KEYS
            if cal_extra:
                raise SystemExit(
                    f"record {i} calibration[{j}]: unrecognized key(s) {sorted(cal_extra)}"
                )

scorers = sorted({r["scorer"] for r in obj})
cells = len({(r["event"], r["horizon_hours"]) for r in obj})
print(f"alerts.json validated: {len(obj)} records, {cells} cells, scorers {scorers}")
PY
        then
            echo "REFUSING to export $source_json -- it did not validate (see above)." >&2
            return 1
        fi

        cp "$source_json" "$dest_json"
        echo "Exported alerts summary to $dest_json"
    }

    run_alerts() {
        # The GBM/TabICL baseline leg (experiment_plan G4). Runs the alerts
        # stage only -- NOT the full scripts/eval_run.sh chain -- because the
        # baselines are what this is for. The hazard column comes along free
        # (the stage scores both), but on a pre-decomposition checkpoint it is
        # a stale architecture and must not be quoted as a CBM result; the
        # baseline columns are model-independent given a fixed protocol and
        # row set, which is exactly why they can be banked now.
        local run_name="$1"
        if [[ -z "$run_name" ]]; then
            echo "alerts needs a run name: scripts/gemini/run.sh alerts <run-name>" >&2
            echo "e.g.: scripts/gemini/run.sh alerts gemini_full_v10" >&2
            exit 1
        fi

        # Held-out shard count is the one number that must match between this
        # step and the tabicl step. On MIMIC they did not (alerts covered 37
        # shards, TabICL 4), n differed 8.8x, and the two could never share a
        # table -- see tab:tabicl's separate existence. One knob feeds both.
        local alert_shards="${GEMINI_ALERT_SHARDS:-4}"
        local baseline_shards="${GEMINI_BASELINE_SHARDS:-30}"
        local num_lanes="${GEMINI_EVAL_NUM_LANES:-16}"
        local chunk="${GEMINI_EVAL_CHUNK:-512}"
        local checkpoint="${GEMINI_ALERT_CHECKPOINT:-checkpoint_best.pt}"

        echo "=== alerts ($run_name) ==="
        echo "Hazard heads vs the strong tuned per-event GBM on the landmark rows."
        echo "alert_shards=$alert_shards (held-out), baseline_shards=$baseline_shards (train),"
        echo "num_lanes=$num_lanes, chunk=$chunk, checkpoint=$checkpoint."
        echo "Override via GEMINI_ALERT_SHARDS / GEMINI_BASELINE_SHARDS /"
        echo "GEMINI_EVAL_NUM_LANES / GEMINI_EVAL_CHUNK / GEMINI_ALERT_CHECKPOINT."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: this doesn't look like a tmux or screen session --" >&2
            echo "if the SSH connection drops, alerts dies with it." >&2
            echo "Run it detached instead, e.g.:" >&2
            echo "  tmux new -s alerts-$run_name 'scripts/gemini/run.sh $STEP $run_name'" >&2
        fi

        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "GPU venv not found at $GPU_VENV -- run 'scripts/gemini/run.sh env-gpu' first." >&2
            exit 1
        fi

        RUN_DIR="$HOME/runs/$run_name"
        if [[ ! -d "$RUN_DIR" ]]; then
            echo "No run directory at $RUN_DIR -- run the training step that" >&2
            echo "produces it first (e.g. train-full-cbm)." >&2
            exit 1
        fi

        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        HELD_OUT_SHARD_DIR="${GEMINI_EVAL_HELD_OUT_DIR:-$MEDS_DIR/data/held_out}"
        TRAIN_SHARD_DIR="${GEMINI_TRAIN_SHARD_DIR:-$MEDS_DIR/data/train}"
        for d in "$HELD_OUT_SHARD_DIR" "$TRAIN_SHARD_DIR"; do
            if [[ ! -d "$d" ]]; then
                echo "Expected finalized MEDS data at $d -- run" >&2
                echo "'scripts/gemini/run.sh finalize' first." >&2
                exit 1
            fi
        done

        OUTPUT_JSON="$RUN_DIR/alerts.json"
        DUMP_ROWS="$RUN_DIR/alerts_rows.parquet"
        echo "Run dir: $RUN_DIR"
        echo "Held-out: $HELD_OUT_SHARD_DIR"
        echo "Train (GBM fit): $TRAIN_SHARD_DIR"
        echo "Output: $OUTPUT_JSON"
        echo "Row dump: $DUMP_ROWS (patient-level, stays on this node)"

        source "$GPU_VENV/bin/activate"
        # --stream-baseline-shards is not optional at GEMINI scale: the
        # whole-frame GBM fit path loads BASELINE_SHARDS entirely into memory
        # and OOMs on a corpus this size (894 train shards).
        python -m odyssey.inference.alerts \
            --run-dir "$RUN_DIR" \
            --held-out-shard-dir "$HELD_OUT_SHARD_DIR" \
            --baseline-shard-dir "$TRAIN_SHARD_DIR" \
            --max-shards "$alert_shards" \
            --max-baseline-shards "$baseline_shards" \
            --num-lanes "$num_lanes" \
            --chunk-size "$chunk" \
            --output-json "$OUTPUT_JSON" \
            --dump-rows "$DUMP_ROWS" \
            --checkpoint "$checkpoint" \
            --stream-baseline-shards
        deactivate
        source "$VENV/bin/activate"

        echo "$STEP complete. Results at $OUTPUT_JSON"

        if ! _export_alerts_summary "$run_name" "$OUTPUT_JSON"; then
            echo "WARNING: this run's alerts summary was not exported (see above)." >&2
        fi
    }

    run_tabicl() {
        # TabICLv2 on the SAME rows the alerts step already dumped, so its
        # column can sit in one table beside hazard and the GBM instead of
        # needing a separate one. --existing-dump is what enforces that.
        local run_name="$1"
        if [[ -z "$run_name" ]]; then
            echo "tabicl needs a run name: scripts/gemini/run.sh tabicl <run-name>" >&2
            echo "e.g.: scripts/gemini/run.sh tabicl gemini_full_v10" >&2
            exit 1
        fi

        local alert_shards="${GEMINI_ALERT_SHARDS:-4}"
        local train_shards="${GEMINI_TABICL_TRAIN_SHARDS:-8}"

        echo "=== tabicl ($run_name) ==="
        echo "TabICLv2, strong feature panel, at full capability: n_estimators=8"
        echo "(TabICLClassifier's own default) and a 50,000-row context"
        echo "(TABICL_MAX_ROWS, a module constant in tabicl_baseline.py --"
        echo "neither is a CLI knob, so this matches the MIMIC/eICU legs by"
        echo "construction rather than by remembering to pass a flag)."
        echo "Held-out shards=$alert_shards (GEMINI_ALERT_SHARDS -- MUST be the"
        echo "same value the alerts step ran with, or the two columns describe"
        echo "different samples)."
        echo "Budget roughly half an hour of inference PER CELL."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: this doesn't look like a tmux or screen session --" >&2
            echo "if the SSH connection drops, tabicl dies with it." >&2
            echo "Run it detached instead, e.g.:" >&2
            echo "  tmux new -s tabicl-$run_name 'scripts/gemini/run.sh $STEP $run_name'" >&2
        fi

        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "GPU venv not found at $GPU_VENV -- run 'scripts/gemini/run.sh env-gpu' first." >&2
            exit 1
        fi

        RUN_DIR="$HOME/runs/$run_name"
        EXISTING_DUMP="$RUN_DIR/alerts_rows.parquet"
        if [[ ! -f "$EXISTING_DUMP" ]]; then
            echo "No row dump at $EXISTING_DUMP -- run" >&2
            echo "'scripts/gemini/run.sh alerts $run_name' first. TabICL is scored" >&2
            echo "on the alerts stage's own rows on purpose; there is no" >&2
            echo "standalone path that would produce a comparable column." >&2
            exit 1
        fi

        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        HELD_OUT_SHARD_DIR="${GEMINI_EVAL_HELD_OUT_DIR:-$MEDS_DIR/data/held_out}"
        TRAIN_SHARD_DIR="${GEMINI_TRAIN_SHARD_DIR:-$MEDS_DIR/data/train}"
        OUTPUT_JSON="$RUN_DIR/tabicl_compare.json"

        source "$GPU_VENV/bin/activate"
        if ! python -c "import tabicl" 2>/dev/null; then
            echo "tabicl is not installed in $GPU_VENV." >&2
            echo "It is an optional extra and env-gpu does not install it." >&2
            echo "Install it into that venv first:" >&2
            echo "  source $GPU_VENV/bin/activate && pip install 'tabicl>=2.0.0'" >&2
            deactivate
            source "$VENV/bin/activate"
            exit 1
        fi
        # The guard's 16GB default refuses the strong panel outright; the
        # MIMIC and eICU legs both needed this raised on a 170GB host. If this
        # node has less memory than the budget claims, the fit will be killed
        # rather than silently reduced -- that is the intended failure.
        export ODYSSEY_TABICL_MEMORY_BUDGET_GB="${ODYSSEY_TABICL_MEMORY_BUDGET_GB:-120}"
        echo "ODYSSEY_TABICL_MEMORY_BUDGET_GB=$ODYSSEY_TABICL_MEMORY_BUDGET_GB"
        python scripts/tabicl_strong_compare.py \
            --run-dir "$RUN_DIR" \
            --train-shard-dir "$TRAIN_SHARD_DIR" \
            --held-out-shard-dir "$HELD_OUT_SHARD_DIR" \
            --existing-dump "$EXISTING_DUMP" \
            --max-train-shards "$train_shards" \
            --max-held-out-shards "$alert_shards" \
            --output-json "$OUTPUT_JSON"
        deactivate
        source "$VENV/bin/activate"

        echo "$STEP complete. Results at $OUTPUT_JSON"
    }

    _export_aggregate_json() {
        # Copies an aggregate-only JSON into scripts/gemini/out/evals/ after
        # checking its top-level keys against a whitelist (same reasoning as
        # _export_eval_summary: an unknown key refuses rather than riding
        # along). Both callers' files carry per-concept and per-dial
        # aggregates only -- no subject ids, no per-row fields.
        local dest_json="$1"
        local source_json="$2"
        local allowed_keys="$3"   # space-separated

        mkdir -p scripts/gemini/out/evals
        if ! python3 - "$source_json" "$allowed_keys" <<'PY'
import json
import sys

path, allowed = sys.argv[1], set(sys.argv[2].split())
with open(path) as f:
    obj = json.load(f)
if not isinstance(obj, dict):
    raise SystemExit(f"{path}: expected an object, got {type(obj).__name__}")
unknown = set(obj) - allowed
if unknown:
    raise SystemExit(f"{path}: unrecognized top-level keys {sorted(unknown)}")
FORBIDDEN = {"subject_id", "subject_ids", "rows", "per_subject"}

def scan(node, where):
    if isinstance(node, dict):
        hit = FORBIDDEN & set(node)
        if hit:
            raise SystemExit(f"{path}: patient-level key {sorted(hit)} at {where}, refusing")
        for k, v in node.items():
            scan(v, f"{where}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node[:50]):
            scan(v, f"{where}[{i}]")

scan(obj, "")
PY
        then
            echo "Refusing to export $source_json (see above)." >&2
            return 1
        fi
        cp "$source_json" "$dest_json"
        echo "Exported $dest_json"
    }

    _require_run_and_data() {
        # Shared preamble for the post-training analysis steps: GPU venv,
        # the run directory, and the finalized MEDS split directories.
        # Sets GPU_VENV, RUN_DIR, MEDS_DIR, HELD_OUT_SHARD_DIR,
        # TRAIN_SHARD_DIR, METADATA_DIR for the caller.
        local run_name="$1"
        GPU_VENV="${GEMINI_GPU_VENV:-$HOME/.venvs/odyssey-gemini-gpu}"
        if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
            echo "GPU venv not found at $GPU_VENV -- run 'scripts/gemini/run.sh env-gpu' first." >&2
            exit 1
        fi
        RUN_DIR="$HOME/runs/$run_name"
        if [[ ! -d "$RUN_DIR" ]]; then
            echo "No run directory at $RUN_DIR -- run the training step that" >&2
            echo "produces it first (train-full-dec for the decomposed model)." >&2
            exit 1
        fi
        MEDS_DIR="${GEMINI_MEDS_OUTPUT_DIR:-/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1}"
        HELD_OUT_SHARD_DIR="${GEMINI_EVAL_HELD_OUT_DIR:-$MEDS_DIR/data/held_out}"
        TRAIN_SHARD_DIR="${GEMINI_TRAIN_SHARD_DIR:-$MEDS_DIR/data/train}"
        METADATA_DIR="${GEMINI_METADATA_DIR:-$MEDS_DIR/metadata}"
        for d in "$HELD_OUT_SHARD_DIR" "$TRAIN_SHARD_DIR" "$METADATA_DIR"; do
            if [[ ! -d "$d" ]]; then
                echo "Expected finalized MEDS data at $d -- run" >&2
                echo "'scripts/gemini/run.sh finalize' first." >&2
                exit 1
            fi
        done
    }

    run_steering() {
        # The clinical dial benchmark (Steerling Sec. 6.2 steering, applied
        # to the decomposed model): for every concept with a clinical
        # expectation (odyssey/inference/steering.py CLINICAL_EXPECTATIONS),
        # push its direction into the residual stream at gamma = tau / max
        # alignment and read the change in every hazard head's risk over
        # at-risk patients, with a paired subject bootstrap. Also reports
        # the readout response (k_c) and the lifted-token mass. Same
        # settings as the MIMIC/eICU full-split runs: stream site, every
        # held-out shard, lifted sets from 4 train shards.
        local run_name="$1"
        if [[ -z "$run_name" ]]; then
            echo "steering needs a run name: scripts/gemini/run.sh steering <run-name>" >&2
            echo "e.g.: scripts/gemini/run.sh steering gemini_full_DEC_v12" >&2
            exit 1
        fi
        local num_lanes="${GEMINI_EVAL_NUM_LANES:-16}"
        local chunk="${GEMINI_EVAL_CHUNK:-512}"
        local max_shards="${GEMINI_STEERING_MAX_SHARDS:-}"   # empty = every held-out shard
        local tau="${GEMINI_STEERING_TAU:-1.0}"

        echo "=== steering ($run_name) ==="
        echo "Clinical dials on the decomposed model: stream site, tau=$tau,"
        echo "num_lanes=$num_lanes, chunk=$chunk, held-out shards=${max_shards:-all}."
        echo "Override via GEMINI_EVAL_NUM_LANES / GEMINI_EVAL_CHUNK /"
        echo "GEMINI_STEERING_MAX_SHARDS / GEMINI_STEERING_TAU."
        if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
            echo "WARNING: not a tmux/screen session; run detached, e.g.:" >&2
            echo "  tmux new -s steering-$run_name 'scripts/gemini/run.sh $STEP $run_name'" >&2
        fi
        _require_run_and_data "$run_name"
        if ! grep -q '"bottleneck_kind": "decomposed"' "$RUN_DIR/config.json" 2>/dev/null; then
            echo "WARNING: $RUN_DIR/config.json is not a decomposed run; the dials" >&2
            echo "are defined for the decomposition (train-full-dec). Continuing." >&2
        fi

        OUTPUT_JSON="$RUN_DIR/steering_full.json"
        echo "Run dir: $RUN_DIR"
        echo "Output: $OUTPUT_JSON"
        local extra=()
        if [[ -n "$max_shards" ]]; then extra+=(--max-shards "$max_shards"); fi

        source "$GPU_VENV/bin/activate"
        python -m odyssey.inference.steering \
            --run-dir "$RUN_DIR" \
            --held-out-shard-dir "$HELD_OUT_SHARD_DIR" \
            --lift-shard-dir "$TRAIN_SHARD_DIR" \
            --metadata-dir "$METADATA_DIR" \
            --output-json "$OUTPUT_JSON" \
            --site stream \
            --tau "$tau" \
            --lift-shards 4 \
            --num-lanes "$num_lanes" \
            --chunk-size "$chunk" \
            "${extra[@]}"
        deactivate
        source "$VENV/bin/activate"

        echo "$STEP complete. Results at $OUTPUT_JSON"
        _export_aggregate_json \
            "scripts/gemini/out/evals/${run_name}_steering_full.json" "$OUTPUT_JSON" \
            "site layer_index tau suppress_strength horizons_hours event_names gammas lifted_tokens summaries" \
            || echo "WARNING: steering output not exported (see above)." >&2
    }

    run_atlas() {
        # The concept atlas (paper appendix "What the Concepts Say"): for
        # every known and unknown concept, the vocabulary tokens its
        # direction promotes and suppresses through the next-event head
        # (W K_c / W U_j), the concept's mean activation, and the share of
        # the next-event logit carried by named / unknown / residual parts.
        # One held-out shard is enough for the shares; the token lists are
        # a property of the weights.
        local run_name="$1"
        if [[ -z "$run_name" ]]; then
            echo "atlas needs a run name: scripts/gemini/run.sh atlas <run-name>" >&2
            echo "e.g.: scripts/gemini/run.sh atlas gemini_full_DEC_v12" >&2
            exit 1
        fi
        local num_lanes="${GEMINI_EVAL_NUM_LANES:-16}"
        local chunk="${GEMINI_EVAL_CHUNK:-512}"
        local max_shards="${GEMINI_ATLAS_MAX_SHARDS:-1}"

        echo "=== atlas ($run_name) ==="
        echo "Concept atlas: num_lanes=$num_lanes, chunk=$chunk, shards=$max_shards."
        _require_run_and_data "$run_name"

        OUTPUT_JSON="$RUN_DIR/concept_atlas.json"
        echo "Run dir: $RUN_DIR"
        echo "Output: $OUTPUT_JSON"

        source "$GPU_VENV/bin/activate"
        python scripts/concept_atlas.py \
            --run-dir "$RUN_DIR" \
            --held-out-shard-dir "$HELD_OUT_SHARD_DIR" \
            --metadata-dir "$METADATA_DIR" \
            --output-json "$OUTPUT_JSON" \
            --max-shards "$max_shards" \
            --top 12 \
            --num-lanes "$num_lanes" \
            --chunk-size "$chunk"
        deactivate
        source "$VENV/bin/activate"

        echo "$STEP complete. Results at $OUTPUT_JSON"
        _export_aggregate_json \
            "scripts/gemini/out/evals/${run_name}_concept_atlas.json" "$OUTPUT_JSON" \
            "run_dir source n_positions contribution_share known unknown" \
            || echo "WARNING: atlas output not exported (see above)." >&2
    }

    case "$STEP" in
        probe) run_probe ;;
        schema) run_schema ;;
        env-gpu) run_env_gpu ;;
        extract-dry) run_extract_dry ;;
        extract) run_extract ;;
        finalize) run_finalize ;;
        export-codes) run_export_codes ;;
        pipeline) run_pipeline ;;
        train-smoke) run_train_smoke 5 2 gemini_smoke_1 ;;
        train-smoke-2) run_train_smoke 30 "" gemini_smoke_2 ;;
        train-full) run_train_full ;;
        train-smoke-cbm) run_train_cbm 5 gemini_smoke_cbm ;;
        train-full-cbm) run_train_cbm "" gemini_full_v10 ;;
        train-smoke-dec) run_train_cbm 5 gemini_smoke_dec decomposed ;;
        train-full-dec) run_train_cbm "" gemini_full_DEC_v12 decomposed ;;
        train-rung2) run_train_rung2 ;;
        eval-forecast) run_eval_forecast "${2:-}" ;;
        alerts) run_alerts "${2:-}" ;;
        tabicl) run_tabicl "${2:-}" ;;
        steering) run_steering "${2:-}" ;;
        atlas) run_atlas "${2:-}" ;;
        train) run_pending_stub train ;;
        eval) run_pending_stub eval ;;
        all) run_probe; run_schema; run_extract_dry ;;
        *)
            echo "unknown step: $STEP (expected probe, schema, env-gpu, extract-dry, extract, finalize, export-codes, pipeline, train-smoke, train-smoke-2, train-full, train-smoke-cbm, train-full-cbm, train-smoke-dec, train-full-dec, train-rung2, eval-forecast, alerts, tabicl, steering, atlas, train, eval, or all)" >&2
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
