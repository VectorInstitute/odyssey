# Clinician demo ("Odyssey · Bedside Forecast")

A web app that replays one patient's admission and shows, moment by moment,
what the trained model forecasts: the chance of ICU admission, vasopressors,
acute kidney injury, Sepsis-3 and death within 8, 24 and 72 hours; when that
risk would have crossed an alert line; what the model thinks is going on (its
29 concept beliefs); what would move the forecast (what-if); what the forecast
leans on (evidence); and how good the model is (scorecard, against the tuned
GBM). Code: `apps/clinician_demo/`. Tests: `tests/apps/clinician_demo/`.

## Data modes and who may see them

| Mode | Data | Who may view |
| --- | --- | --- |
| `credentialed` | held-out MIMIC-IV 3.1 patients of the run's own extraction | only people holding PhysioNet MIMIC-IV credentials |
| `open` | MIMIC-IV Clinical Database Demo (100 patients, open licence) | anyone |

In open mode, 90 of the 100 demo patients were in the model's training or
tuning split (measured 2026-09-11 against `subject_splits.parquet`: 83 train,
7 tuning, 10 held-out). Every chart says whether the model saw the patient.
Only the 10 held-out ones are a fair test.

Patient-level data never leaves the GPU host. The server binds to loopback and
is viewed through an SSH tunnel. Do not screenshot credentialed patients into
chat tools, slides or LLM services; use open mode for anything shared.

## Run it (GPU host, repository root)

```bash
# credentialed mode, port 8765
.venv/bin/python -m apps.clinician_demo \
  --run-dir ~/runs/full_run_v10 \
  --data-dir ~/data/mimiciv_3.1_v1/data/held_out \
  --metadata-dir ~/data/mimiciv_3.1_v1/metadata \
  --splits ~/data/mimiciv_3.1_v1/metadata/subject_splits.parquet

# open mode, port 8766
.venv/bin/python -m apps.clinician_demo --data-mode open --port 8766 \
  --run-dir ~/runs/full_run_v10 \
  --data-dir ~/data/mimiciv_demo_meds/data \
  --metadata-dir ~/data/mimiciv_demo_meds/metadata \
  --splits ~/data/mimiciv_3.1_v1/metadata/subject_splits.parquet
```

Add `--self-check` to load everything, run one case end to end (trace,
what-if, gap to the banked landmark scores) and print a JSON report. Launch
long-running servers with `setsid nohup ... & disown`.

The open demo extraction is made once with
`PATH=$HOME/odyssey/.venv/bin:$PATH meds-extract-run spec=MIMIC-IV output_dir=~/data/mimiciv_demo_meds dataset_key=demo`
(the pipeline shells out to `MEDS_transform-stage`, so the venv must be on
`PATH`).

## View it (laptop)

```bash
gcloud compute ssh odyssey-cbm-a100 --zone us-central1-f \
  --project agentic-ai-evaluation-bootcamp --tunnel-through-iap -- \
  -N -L 8765:localhost:8765 -L 8766:localhost:8766
```

Then open http://localhost:8765 (credentialed) or http://localhost:8766 (open).

The server maps static URLs to files at start-up but reads each file on
every request, so a changed `index.html`, `styles.css` or `js/**` file can
be copied over the running deployment and takes effect on the next reload.
Adding or removing a static file needs a restart.

## What a clinician sees

The pages are written around a clinician's questions, in order. The
replay opens with the patient in one line (age, sex, admission type, length
of stay), a sticky "now" bar (play, scrub, clock time since admission),
then one tile per event with the risk in the chosen window, a one-word
status against its alert line (Low, Watch, Alert on, Happened) and how that
compares with the average patient. Below: the risk chart with what comes
after "now" faded, what happened in the stay in plain words, the
conditions the model believes are present right now, and the what-if and
evidence tools. The 29-row concept heat strip, the alert-line statistics
and the next-token forecast are behind disclosures. Light theme by default;
a footer link switches to dark.

## What is shown, and what is deliberately not

Shown: the hazard heads' risk (hidden at and after the event's onset), alert
lines set on the run's own held-out landmark rows (default: flag 5% of at-risk
moments; sensitivity and PPV, and the GBM's at the same flag rate, are shown
beside each line), concept beliefs, next-event forecast, what-if value edits
(`odyssey.inference.counterfactual`), occlusion evidence, and the banked
scorecard.

Not shown, on purpose: 30-day readmission (AUROC 0.59, the model's weakest
head), concept steering (does not apply to the mixture bottleneck of this
run), label overrides (wrong sign on this run), and generated future
timelines (untested on the hybrid backbone). A test enforces that the app
never imports those modules.

## Security

Loopback bind only; `Host` header allowlist (DNS rebinding); API calls need
`X-Odyssey-Demo: 1` (blocks cross-site requests); no CORS; `Cache-Control:
no-store`; same-origin CSP; static files from an allowlisted directory with
path containment.
