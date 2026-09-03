<p align="center">
  <img src="assets/logo.svg" width="70%">
</p>

<p align="center">
  <a href="https://github.com/VectorInstitute/odyssey/actions/workflows/code_checks.yml">
    <img src="https://github.com/VectorInstitute/odyssey/actions/workflows/code_checks.yml/badge.svg" alt="code checks">
  </a>
  <a href="https://github.com/VectorInstitute/odyssey/actions/workflows/unit_tests.yml">
    <img src="https://github.com/VectorInstitute/odyssey/actions/workflows/unit_tests.yml/badge.svg" alt="unit tests">
  </a>
  <a href="https://github.com/VectorInstitute/odyssey/actions/workflows/integration_tests.yml">
    <img src="https://github.com/VectorInstitute/odyssey/actions/workflows/integration_tests.yml/badge.svg" alt="integration tests">
  </a>
  <a href="https://codecov.io/gh/VectorInstitute/odyssey">
    <img src="https://codecov.io/gh/VectorInstitute/odyssey/branch/main/graph/badge.svg" alt="coverage">
  </a>
  <br>
  <a href="https://arxiv.org/abs/2405.14567">
    <img src="https://img.shields.io/badge/arXiv-2405.14567-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/python-≥3.12-blue.svg" alt="Python ≥ 3.12">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="ruff">
  </a>
  <a href="LICENSE.md">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="license">
  </a>
</p>

---

## What this is

Odyssey is a streaming EHR foundation model with a concept bottleneck, and the evaluation harness that asks whether such a bottleneck can be trusted. The model forecasts a patient's timeline as it unfolds (which events come next and when, plus a discrete-time hazard per clinical event: vasopressor start, acute kidney injury, ICU admission, death, Sepsis-3 onset, 30-day readmission), and its hidden state is decomposed into 29 rule-derived clinical concepts, unnamed concepts and a residual it is trained to shrink, following [Steerling](https://github.com/guidelabs/steerling). The same pipeline runs on MIMIC-IV, eICU-CRD and, inside its secure environment, the GEMINI general-medicine consortium.

The paper built on this repository (ML4H 2026, in preparation) asks four questions a clinician would ask before trusting the model, and each has a test here:

1. **Is the readout accurate?** Each concept head is scored by held-out AUROC against its rule label (mean 0.93 on MIMIC-IV, 0.87 on eICU-CRD).
2. **Is the bottleneck complete?** Zeroing probes remove the named, unnamed or residual channel and report how much next-event accuracy survives.
3. **Can the model be steered?** Two different levers. Overriding a concept's label with its true value never improves a jointly trained model's forecast: the model reacts to how unusual the override is, not to what it means. Pushing the model along a concept's *direction* (Steerling's steering) moves the outcome risks the way a clinician expects on 72 of 74 declared pairs of 26 eICU-CRD dials and on eight of ten MIMIC-IV dials, without retraining.
4. **What does the bottleneck cost?** Against a tuned gradient-boosted classifier on a 609-feature panel and against TabICLv2, on identical landmark rows under one versioned information boundary. The GBM wins most alert cells, mostly through explicit event counts; on GEMINI the hazard heads win 9 of 12.

Every number in the paper traces to a banked JSON under `research_journal/figure_data/` and a row in the experiment registry, [`docs/experiments.md`](docs/experiments.md).

**What this is not.** Not a diagnostic system, not a claim that the sequence model beats bespoke baselines, and not interpretability by construction: every claim has a test, and the registry records where the model falls short.

## Architecture

![Odyssey architecture](docs/figures/architecture.svg)

*Figure: data flows from MIMIC-IV, eICU-CRD or GEMINI through MEDS extraction and tokenization into the backbone and concept module, whose heads produce forecasts, survival curves and alerts, and concept readouts (editable source: `docs/figures/architecture.drawio`).*

**Inputs per token**: the event code with a clinical or quantile value bin folded in, an optional standardized numeric value channel, inter-event time, age, visit structure, and the patient's static facts as leading tokens. Sequences are each subject's complete history, streamed in chunks with carried recurrent state.

**Backbones** (`odyssey/models/backbones/`): the default `hybrid` (EHRMamba's Mamba-2 blocks interleaved with chunk-local attention, trained by truncated backpropagation over persistent packed patient lanes, so a record of any length streams at constant memory per step) and a standard `transformer` (packed context, `max_context` tokens) used in the paper to show the findings do not depend on the backbone.

**Concept module** (`odyssey/models/concept_bottleneck.py`), selected by `TrainingConfig.bottleneck_kind`:

- `decomposed` (the paper's model): $\bar h = \hat k + \hat u + \varepsilon$, with a known head over the supervised concepts, an unnamed head over $3n$ unnamed ones, each concept owning an embedding at the backbone's width, and Steerling's three regularizers (residual dropout, unnamed reconstruction, named/unnamed independence). Forcing a concept moves $\bar h$ by an exact, known displacement.
- `mixture` (the CBGM form, [Ismail et al. 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/9149fc44c95ce58e3ca529a1e34c2691-Paper-Conference.pdf)): each concept is a mixture of two learned embeddings; `concept_global_pairs=True` makes the pole pair independent of the hidden state. These are the paper's comparison arms (the additive arm reported there came from an earlier implementation and is not a current option).
- `model_kind="baseline"` removes the bottleneck with the same backbone and heads.

**Heads** (`odyssey/models/`): bundle-invariant next-event forecasting over same-timestamp bundles, time to the next event over log-spaced bins, one right-censored discrete-time hazard per clinical event (`time_to_event.py`), a concept observability head, and an optional value head.

**Steering** (`odyssey/models/steering.py`, `odyssey/training/steering_phase.py`): a concept's unit direction is added at every block from the middle of the backbone on, at a strength calibrated on the next-event head; the same push can be used as a training-time loss (`steering_phases`).

## Concepts

Concept labels come from a canonical LOINC-keyed rule registry (`odyssey/data/concepts.py`, `odyssey/data/sofa.py`) derived from validated clinical criteria: KDIGO AKI stages (all three legs), Sepsis-3 (validated against an independent reimplementation), SIRS and qSOFA, vasopressor exposure, and sustained or baseline-relative abnormality rules for vitals and labs. Labels are computed as the chart unfolds and never back-filled from later events. `concepts_for_source(source, task_set)` expands the one registry per data source through `odyssey/data/code_mapping.py`; a source resolves a rule when its code system carries the labs, vitals and drugs the rule needs (29 of 29 on MIMIC-IV and eICU-CRD, 25 on GEMINI). Bin edges for value tokens share thresholds with the rules and are versioned with each checkpoint. `scripts/make_concept_registry.py` renders the registry table.

## Installation

**Python ≥ 3.12** and [uv](https://github.com/astral-sh/uv) are required.

```bash
git clone https://github.com/VectorInstitute/odyssey.git
cd odyssey
uv sync --dev
```

The hybrid backbone depends on `mamba-ssm`, which needs CUDA/`nvcc` to build. On a CUDA-capable host:

```bash
uv sync --extra cuda --no-build-isolation
```

CPU/MPS development uses a lightweight stand-in backbone so the concept module and the harness can be built and tested without a GPU. The notes-sidecar text pipeline (`odyssey/text/`) needs `uv sync --extra text` (see `docs/sidecars_and_task_sets.md`).

## Data pipeline

**MEDS is the narrow waist.** Each source has its own extractor, but all converge on the [MEDS](https://github.com/Medical-Event-Data-Standard/meds) event schema, and everything downstream (binning, tokenization, concepts, training, evaluation, baselines) is written once against it. Conformance at the boundary is enforced mechanically (schema, `metadata/` layout, split directories).

**MIMIC-IV** uses the standard [`meds-extract`](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) tooling (`hosp` + `icu` modules):

```bash
uv run meds-extract-run spec=MIMIC-IV output_dir=<output_dir> dataset_key=demo   # public demo, no credentials
uv run meds-extract-run spec=MIMIC-IV output_dir=<output_dir> \
    do_download=false input_dir=<path_to_mimiciv_3.1>                             # full release
```

`do_download=false` also skips the ten auxiliary concept-mapping CSVs the pipeline fetches from `MIT-LCP/mimic-code`; fetch them into the input root first (`BASE=https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map`, files `meas_chartevents_main.csv inputevents_to_rxnorm.csv lab_itemid_to_loinc.csv meas_chartevents_value.csv numerics-summary.csv outputevents_to_loinc.csv d_labitems_to_loinc.csv proc_datetimeevents.csv waveforms-summary.csv proc_itemid.csv`).

**eICU-CRD** uses the same tooling with the project-local MESSY spec [`specs/eICU.yaml`](specs/eICU.yaml) (spec v2: HICL medication identity, named infusions, GCS, urine output; the header documents that subjects are single health-system stays with pseudotimes, so only within-subject relative times are meaningful). The `eicu-meds` PyPI package is incompatible and is not used. Set `TrainingConfig.source="eicu"` so normalization, concept expansion and clinical value ranges pick the eICU tables.

```bash
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> dataset_key=demo
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> \
    do_download=false input_dir=<path_to_eicu_2.0>
```

**GEMINI** ([geminimedicine.ca](https://geminimedicine.ca/), ~30 hospitals, general medicine) is extracted by a SQL-streaming extractor inside its governed environment (`odyssey/data/gemini/`, `scripts/gemini/`). GEMINI-facing work is git-mediated and privacy-preserving: code goes in through version control, runs there, and only aggregate, cell-suppressed outputs come back, never patient-level data or checkpoints. `scripts/gemini/run.sh` chains extraction, finalization, training, evaluation, alerts, steering and atlas steps, each export validated against a whitelist of aggregate keys. See [`docs/gemini.md`](docs/gemini.md).

**Tokenization** (`odyssey/data/value_binning.py`, `vocabulary.py`, `sequences.py`) folds each numeric value into its token as a clinical-range or quantile bin (`LAB//220045//bpm::HIGH`), normalizes medications to ingredient level (`code_normalization.py`), and builds one sequence per subject from the complete history. Optional sidecars carry microbiology labels and note embeddings (`odyssey/data/sidecars.py`).

## Running the experiments

Training and evaluation are plain modules; each prints its own `--help`.

```bash
# train (config fields: backbone, bottleneck_kind, task_set, source, steering_phases, ...)
uv run python -m odyssey.training.train --config-json <config.json> --train-shard-dir <dir> --tuning-shard-dir <dir> --output-dir <run>

# forecasting, readout and completeness metrics on the held-out shards
uv run python -m odyssey.inference.run_inference --run-dir <run> --held-out-shard-dir <dir>

# alerts: landmark rows, hazard heads vs the tuned GBM (and TabICLv2), per-row dumps, zero-channel probes
uv run python -m odyssey.inference.alerts --run-dir <run> --held-out-shard-dir <dir> --baseline-shard-dir <dir> [--zero-channel known|unknown|residual]

# label-override interventions (truth / flip / random / calibrated / zeroing) inside the uncertain band
uv run python -m odyssey.inference.interventions --run-dir <run> --held-out-shard-dir <dir>

# the dial benchmark: push each concept's direction up and down, read the 24 h risks over at-risk patients
uv run python -m odyssey.inference.steering --run-dir <run> --held-out-shard-dir <dir> [--control random|unknown] [--tau 1]

# input-level counterfactuals: edit the record, re-tokenize, re-score
uv run python -m odyssey.inference.counterfactual --run-dir <run> --held-out-shard-dir <dir>
```

Paired subject-clustered intervals come from `scripts/alerts_cis.py` and `scripts/intervention_cis.py` (`odyssey/inference/uncertainty.py`); `scripts/gbm_feature_ablation.py` refits the GBM with each feature group dropped; `scripts/long_history_compare.py` compares two backbones on identical rows split by truncation; `scripts/concept_atlas.py` renders what each concept promotes and suppresses. The `scripts/make_*_table.py` generators turn banked JSON into the paper's tables. `scripts/eval_run.sh` chains the full evaluation of one run.

Every run is registered in [`docs/experiments.md`](docs/experiments.md) (host, data, commit, purpose, outcome) and its aggregate outputs are banked under `research_journal/figure_data/<host>/<run>/`. Environment fingerprints and numeric canaries are written with every checkpoint; the landmark protocol is versioned and stamped on every row dump.

## Evaluation protocol

Alerts are scored at 4-hour landmark index times on every admission; a row is at risk only if the event has not onset, outcomes are onset within 8, 24 or 72 hours with explicit censoring, and every scorer in a table sees the identical row set under one information boundary (landmark protocol v4). Labels are anchored at the time a clinician could first have known them. Intervals are subject-clustered bootstraps; scorer-versus-scorer verdicts use a paired bootstrap of the AUROC difference. Every training arm is a single seed, so within-run claims carry paired inference and cross-run claims are labelled as hypotheses. Details: `docs/reeval_wave_v2.md`, `docs/missingness_protocol.md`, `docs/sidecars_and_task_sets.md`.

## Development

```bash
uv run pytest -m "not integration_test" tests/
uv run ruff format odyssey tests scripts && uv run ruff check odyssey tests scripts
uv run mypy odyssey
```

## Citation

If you use Odyssey or EHRMamba in your research, please cite:

```bibtex
@misc{fallahpour2024ehrmamba,
  title   = {EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records},
  author  = {Adibvafa Fallahpour and Mahshid Alinoori and Arash Afkanpour and Amrit Krishnan},
  year    = {2024},
  eprint  = {2405.14567},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
