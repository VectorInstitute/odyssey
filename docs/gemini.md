# GEMINI

[GEMINI](https://geminimedicine.ca/) is a multi-hospital (~30 sites) inpatient
database hosted on Unity Health Toronto-governed infrastructure, used here to
assess cross-hospital generalization once the MIMIC-IV/eICU pipeline is
validated (see the README's [Goal](../README.md#goal) and
[Roadmap](../README.md#roadmap)). This page documents how development against
GEMINI actually works, since it differs from MIMIC-IV/eICU in a way that
shapes every step: **nobody on this team can log into or run code inside
GEMINI.** Amrit is the only person with an account on the H200 node. The only
channel in or out is git.

## The git-only channel

Every GEMINI-facing script is designed around one workflow: **we push a
script, Amrit runs it on the node, the output comes back in a commit.** There
is no way to iterate interactively — a script has to be correct (or fail
loudly and safely) on the first real run, which is why GEMINI-facing code
gets the same test discipline as everything else in `odyssey/` (see
[Development](../README.md#development)), with a fake-connection unit test
standing in for the real database.

odyssey has two git remotes:

- `origin` — GitHub (`VectorInstitute/odyssey`), the canonical development
  remote. All history lives here.
- `gemini` — GEMINI's internal GitLab
  (`git@code.gemini-hpc.ca:vector-ai-engineering/odyssey.git`), reachable
  only from inside the GEMINI environment (Amrit's node). `main` on GitHub is
  mirrored to `main` on this remote after GitHub CI is green; GEMINI's git
  server has no CI of its own.

**The GEMINI remote enforces a 1 MiB cap per push** (a GitLab pre-receive
hook rejects any incoming pack over that size — this is a *pack* size limit,
not a per-file limit, so even a small file's diff can be rejected if it lands
in a commit whose surrounding history pushes the pack over the cap). A normal
incremental `git push gemini main` is usually fine; if it is rejected, push
in chunks -- but **never chunk-push directly to `refs/heads/main`**: a
mid-sequence force-push briefly leaves `main` pointing at an old, partial
commit, and anyone who happens to `git pull` in that window (Amrit has) sees
a broken-looking repo, even though the *final* state is correct. Stage the
full history on a throwaway ref first, then flip `main` to the real tip in
one atomic push:

```bash
# 1. Push history in increasing chunks to a staging ref, not main. Find
#    commits between the remote's current tip and your local tip:
git rev-list --reverse gemini/main..main

# 2. Push progressively further commits as the new tip of the STAGING ref,
#    smallest step that clears the cap; --force is safe here since each
#    step is strictly forward and nothing reads this ref but you:
git push gemini <sha>:refs/heads/_mirror-staging --force

# 3. Once the staging ref's tip is the real, current local tip, flip main
#    in one atomic, near-zero-transfer push (every object already exists
#    on the server from step 2):
git push gemini --force <local-tip-sha>:refs/heads/main

# 4. Clean up the staging ref:
git push gemini --delete _mirror-staging
```

If a single commit's own diff exceeds the cap (not just a multi-commit
range), it usually means an old, large blob is still in history somewhere
(a full-history mirror can drag in years of prior commits, not just recent
ones). Strip it in a scratch clone before mirroring, rather than trying to
chunk around it -- chunking can't help when *one* commit alone is over the
cap:

```bash
git clone --no-local <path-to-your-working-repo> /tmp/gemini-mirror-clone
cd /tmp/gemini-mirror-clone && git remote remove origin
git-filter-repo --force --strip-blobs-bigger-than 900K
# then push this clone's (rewritten-history) main via steps 1-4 above
```

`--strip-blobs-bigger-than` (size-based, not a named path) is deliberate:
the two offenders found in this repo's history so far (a 105 MB `.whl` file
and a 3.8 MB Jupyter notebook with embedded plot output, both from 2024)
were different files in different commits, so a general size cutoff catches
both -- and whatever the next one turns out to be -- without needing to
know its name in advance. This only rewrites the *scratch clone*; it never
touches `origin`'s real history, which is why `origin` stays canonical and
`gemini`'s own commit hashes are allowed to differ from GitHub's after every
mirror (they will not fast-forward against each other, which is expected --
`git push gemini <sha>:main --force` is the correct move each time, not a
sign something is wrong). GEMINI's own `legacy-*` branches (its git server's
history from before this project's current mirroring approach) are never
touched by any of the above -- only `refs/heads/main` (via
`_mirror-staging`) is ever written.

## Environment (H200 node)

From `scripts/gemini/out/env_probe.txt` (run 2026-08-18):

| | |
| --- | --- |
| GPU | 1x NVIDIA H200, 143,771 MiB (~140 GB) |
| Driver | 595.45.04 (reports CUDA 13.2 as the max the driver supports) |
| CUDA toolkit | no `nvcc` on `PATH`, `CUDA_HOME` unset, but `/usr/local/cuda-12.8` exists |
| Python | 3.12.3 (system `/usr/bin/python3`) |
| uv / poetry / conda | none usable: `uv` not found, `poetry`'s shim is broken (missing binary), no `conda`/`mamba` |
| glibc / libstdc++ | glibc 2.39, `GLIBCXX` up to 3.4.33 (modern; no old-libstdc++ build failures expected) |
| Disk | repo lives on NFS (`/mnt/nfs`), 8.0 TB total, 2.3 TB available |
| RAM | 966 GiB total, 954 GiB available |

**What this implies for torch/mamba-ssm** (see the `cuda` extra's comment in
`pyproject.toml` for the full reasoning already validated on the MIMIC/eICU
A100 host): the driver's CUDA 13.2 ceiling isn't the binding constraint —
drivers are backward compatible — the installed *toolkit* is, since
`mamba-ssm`'s `torch.utils.cpp_extension` build needs an `nvcc` whose major
CUDA version matches torch's. GEMINI has CUDA 12.8 installed (major version
12, just like the A100 host's 12.9), so the existing `torch==2.6.0+cu124` pin
should apply here too, once `PATH`/`CUDA_HOME` point at
`/usr/local/cuda-12.8` instead of `12.9`. The H200 is Hopper architecture
(compute capability 9.0, same family as H100), which `torch==2.6.0`'s cu124
builds already support, so no separate GPU-architecture concern beyond what's
already validated on A100.

Since `uv` isn't installed on the node, the CUDA-index pinning
(`[tool.uv.sources]`/`[[tool.uv.index]]` in `pyproject.toml`) isn't available
through plain `pip` — torch would need installing explicitly from PyTorch's
cu124 wheel index first (`pip install torch==2.6.0 --index-url
https://download.pytorch.org/whl/cu124`), then the rest of the project with
`--no-deps` so pip doesn't try to re-resolve torch, then `mamba-ssm`'s
existing two-step forced non-isolated rebuild exactly as documented in
`pyproject.toml`, substituting `CUDA_HOME=/usr/local/cuda-12.8`. This is real
work, not a one-liner (the mamba-ssm rebuild alone is a genuine ~30 minute
CUDA compile) -- built as its own dedicated step, `env-gpu`, in a separate
venv from the lightweight one `probe`/`schema`/`extract-dry` share, since
none of those need torch/mamba-ssm at all.

## scripts/gemini/run.sh

The single entry point Amrit actually runs on the node, mirroring
`gemini-variation-study`'s `run.sh`: it activates (creating if missing) a
venv under his home directory, installs what the selected step needs, runs
the step, then commits and pushes only the small output files it produced.

```bash
scripts/gemini/run.sh probe        # scripts/gemini/probe_env.sh -> env_probe.txt
scripts/gemini/run.sh schema       # scripts/gemini/explore_schema.py -> schema.json/.md
scripts/gemini/run.sh env-gpu      # builds the H200 training venv (torch + mamba-ssm),
                                    # writes env_fingerprint.json; separate GPU venv,
                                    # idempotent -- skips reinstall/rebuild if already done
scripts/gemini/run.sh extract-dry  # scripts/gemini/extract_dry.py -> extract_dry.{json,md};
                                    # needs schema.json first, else prints "pending
                                    # schema report" and does nothing
scripts/gemini/run.sh extract      # not built yet -- same "pending schema report" stub
scripts/gemini/run.sh train        # not built yet -- same stub
scripts/gemini/run.sh eval         # not built yet -- same stub
scripts/gemini/run.sh              # default: probe, schema, extract-dry, in order
                                    # (deliberately excludes env-gpu -- see below)
```

`env-gpu` is left out of the default `all` run on purpose: it's a real,
multi-minute, GPU-dependent build step, not a quick sanity check, so it only
runs when explicitly asked for (`scripts/gemini/run.sh env-gpu`), the same
way the not-yet-built `extract`/`train`/`eval` steps aren't wired into `all`
either -- `all` stays a fast, safe, no-surprises default.

Safe to re-run: venv creation and package installs are idempotent, each step
overwrites its own output deterministically, and if a step produces nothing
new since the last run there is nothing to commit. It refuses to commit
anything outside `scripts/gemini/out/` or `docs/gemini*`, anything over
900 KB, or any path that looks like data (`.parquet`, `.csv`, `.pt`,
`.ckpt`, ...) — see the script itself for the exact checks. Future steps
(extraction, training, eval) are meant to slot in as additional `case`
branches, not as separate scripts Amrit has to remember to run in order.

## Credentials

Never hard-code GEMINI database credentials. The pattern (mirrored from the
existing `gemini-variation-study` repo, which has already validated it in
production against this exact database) is: a git-ignored `.env` file at the
repository root, loaded automatically at import time, with real environment
variables always taking precedence over `.env` values. Copy `.env.example`
(added alongside `odyssey/data/gemini/config.py`) to `.env` and fill in all
four required variables -- none of them have a default:

```bash
# Secrets -- no defaults in code, ever.
GEMINI_DB_USER=your_username
GEMINI_DB_PASS=your_password

# Not secrets, but also no defaults: which database and data cut odyssey
# actually queries is a project decision, not something the code should
# guess at.
GEMINI_DB_NAME=your_database
GEMINI_DATACUT=your_datacut
```

Only `GEMINI_DB_HOST` and `GEMINI_DB_PORT` have real defaults
(`db.gemini-hpc.ca` / `5432`) the environment can override; `GEMINI_DB_USER`,
`GEMINI_DB_PASS`, `GEMINI_DB_NAME`, and `GEMINI_DATACUT` all have none, and
connecting fails with a clear message naming exactly which one is missing
(`odyssey.data.gemini.config.credentials_help`), rather than a generic
"incomplete" message or a confusing driver error. If the database connection
itself is configured but `GEMINI_DATACUT` specifically isn't set yet,
`scripts/gemini/run.sh schema` lists the schemata actually visible in the
database instead of raising, so there's something concrete to set it to.
See `odyssey/data/gemini/config.py` and `db.py`.

## Governance: what may leave GEMINI

All analyses run under Unity Health Toronto governance. The rule is
conservative by default — when unsure whether something is safe to commit,
it stays inside.

**May leave** (small, aggregate, cell-suppressed):

- Aggregate metrics and evaluation numbers (AUROC, calibration, loss curves)
  computed over the full held-out set, never per-patient.
- Schema/metadata reports: table and column names, types, and row counts —
  but row counts are rounded to the nearest 1,000, and any count under 6 is
  shown as `<6` rather than the real number (small-cell suppression: a small
  count can itself identify a patient). See `scripts/gemini/explore_schema.py`.
- Reports as aggregate HTML or JSON, with the same suppression applied to
  every printed table — not raw text dumps.
- Small plain-text environment/diagnostic output (e.g.
  `scripts/gemini/out/env_probe.txt`): GPU/driver/CUDA info, tool versions,
  disk/RAM — no data of any kind.

**Must stay inside GEMINI, always:**

- Any patient-level data, in any form (raw, de-identified, aggregated below
  the suppression threshold, or otherwise).
- **Model checkpoints.** These are not patient data, but they are large
  (hundreds of MB to GB) and there is no reason to move them through a
  1 MiB-capped git channel — they stay on the node. If a checkpoint needs to
  leave for external analysis, that is a deliberate, separate decision, not
  a routine push.
- Raw run logs with row-level output (frequency tables straight from a
  query, stack traces that echo query results). Suppression can be applied
  to numbers a script *chose* to print, but it cannot vouch for an entire
  captured log — the same reasoning `gemini-variation-study`'s `.gitignore`
  already encodes for `*_output.txt`.

## Run-there / commit-back workflow

1. We write and test a script (and, if it's a new step, wire it into
   `scripts/gemini/run.sh`) against a mocked connection, push it to both
   `origin` and `gemini` (`origin` first; mirror to `gemini` once GitHub CI
   is green).
2. Amrit pulls on the GEMINI node, runs `scripts/gemini/run.sh <step>`.
3. The step writes only what is allowed to leave (see Governance above) to
   `scripts/gemini/out/`.
4. `run.sh` commits and pushes that output back to `gemini main` directly
   (see the safety checks described under `scripts/gemini/run.sh` above) --
   this is the one case where `gemini main` legitimately moves ahead of
   `origin main`, since GEMINI's git server is the only place that commit
   could have been made.
5. Someone with GitHub push access fetches it back: `git fetch gemini main`,
   then copies only the new/changed files under `scripts/gemini/out/` (and
   `docs/gemini*` if Amrit's run touched docs) into a normal commit on
   `origin main` --

   ```bash
   git fetch gemini main
   git diff --stat main gemini/main -- scripts/gemini/out/ docs/gemini.md
   # copy out the specific files that changed, e.g.:
   git show gemini/main:scripts/gemini/out/schema.json > scripts/gemini/out/schema.json
   git add scripts/gemini/out/schema.json
   git commit -m "GEMINI: schema report from Amrit's run.sh schema"
   git push origin main
   ```

   **Never `git merge`/`git rebase` `gemini/main` into `main`** -- the two
   branches' histories are unrelated by design (see the mirroring section
   above: `gemini`'s commit hashes never match `origin`'s), so a graph merge
   would import GEMINI's entire divergent history into the canonical
   GitHub repo. Copying the specific output files as their own new commit
   keeps `origin main`'s history clean and is also naturally consistent
   with Governance below, since it's a deliberate look at exactly what's
   about to leave GEMINI, file by file, not a bulk import.
6. Once `origin main`'s CI is green, the *next* full mirror (GitHub → GEMINI,
   per the mirroring section above) carries this content back to `gemini
   main` too, so both sides end up consistent again -- `gemini main`
   temporarily leading between steps 4 and 6 is expected, not a conflict.
7. We iterate from the output — adjust the script, repeat. Every round trip
   costs Amrit's time on the node, so scripts should fail fast and report
   clearly rather than requiring a second run to fix an avoidable mistake.

## GEMINI to MEDS

The first real schema report has landed (`scripts/gemini/out/schema.json`,
`schema.md`, run 2026-08-20 against datacut `subdural_hematoma_v1_0_0`) --
the full MEDS mapping drafted against it, its open questions, and the
sharding/output rules now live in their own doc:
**[`docs/gemini_extraction.md`](gemini_extraction.md)**, kept separate from
this page since it's specific to the extraction design rather than the
git-only workflow this page documents. No extraction code exists yet.
