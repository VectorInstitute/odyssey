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
incremental `git push gemini main` is usually fine; if it is rejected:

```bash
# Push history in increasing chunks instead of the whole branch at once.
# Find commits between the remote's current tip and your local tip:
git rev-list --reverse gemini/main..main

# Push progressively further commits as the new tip, smallest step that
# clears the cap; --force is safe here since each step is strictly forward:
git push gemini <sha>:refs/heads/main --force
```

If a single commit's own diff exceeds the cap (not just a multi-commit
range), that commit itself needs breaking up or the offending large content
needs to be excluded from what gets mirrored — talk to whoever owns the
GEMINI sync before trying to force it through.

## Credentials

Never hard-code GEMINI database credentials. The pattern (mirrored from the
existing `gemini-variation-study` repo, which has already validated it in
production against this exact database) is: a git-ignored `.env` file at the
repository root, loaded automatically at import time, with real environment
variables always taking precedence over `.env` values. Copy `.env.example`
(added alongside `odyssey/data/gemini/config.py`) to `.env` and fill in:

```bash
GEMINI_DB_USER=your_username
GEMINI_DB_PASS=your_password
```

Non-secret connection parameters (`GEMINI_DB_HOST`, `GEMINI_DB_PORT`,
`GEMINI_DB_NAME`, `GEMINI_DATACUT`) have defaults the environment can
override; secrets have no defaults and connecting fails with a clear message
if they are missing, rather than a confusing driver error. See
`odyssey/data/gemini/config.py` and `db.py`.

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

1. We write and test a script against a mocked connection, push it to both
   `origin` and `gemini` (`origin` first; mirror to `gemini` once GitHub CI
   is green).
2. Amrit pulls on the GEMINI node, runs the script.
3. The script writes only what is allowed to leave (see Governance above) to
   a small, predictable output path (e.g. `scripts/gemini/out/`).
4. Amrit commits and pushes that output back to `gemini`, and it gets
   mirrored to `origin`/GitHub so the rest of the team can see it.
5. We iterate from the output — adjust the script, repeat. Every round trip
   costs Amrit's time on the node, so scripts should fail fast and report
   clearly rather than requiring a second run to fix an avoidable mistake.

## GEMINI to MEDS

*Pending the first schema report (`scripts/gemini/out/schema.json`,
`schema.md`) from `scripts/gemini/explore_schema.py`.* The intended shape,
to validate once the real schema is in hand:

- **Subject**: a patient, or a hospital encounter if patient-level linkage
  across encounters is not straightforward in this data cut — to be
  confirmed from the schema report.
- **Events**: drawn from admissions, diagnoses, labs, pharmacy, and
  radiology tables (see `gemini-variation-study`'s `admdad_subset`,
  `ipdiagnosis_subset`, `radiology_subset` for the table-naming convention
  in GEMINI's native schema, which is not MEDS or OMOP CDM).
- **Coding**: GEMINI maps some lab/measurement columns to OMOP concept IDs
  internally (e.g. `test_type_mapped_omop`, `measurement_mapped_omop`); the
  intent is to go through those OMOP-mapped columns to LOINC, the same
  vocabulary the existing MIMIC-IV/eICU concept rules are already keyed by
  (see the README's [Data pipeline](../README.md#data-pipeline) and
  `odyssey/data/code_mapping.py`), rather than writing a third,
  GEMINI-specific rule set from scratch.

No extraction code exists yet — this section is a target to validate against
the real schema, not a spec to implement blind.
