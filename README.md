<p align="center">
  <img src="assets/logo.svg" width="70%">
</p>

<p align="center">
  <a href="https://github.com/VectorInstitute/odyssey/actions/workflows/code_checks.yml">
    <img src="https://github.com/VectorInstitute/odyssey/actions/workflows/code_checks.yml/badge.svg" alt="code checks">
  </a>
  <a href="https://arxiv.org/abs/2405.14567">
    <img src="https://img.shields.io/badge/arXiv-2405.14567-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/python-≥3.12-blue.svg" alt="Python ≥ 3.12">
</p>

---

## Goal

Odyssey's question is which kind of model is actually right for a general clinical foundation model: one that is strong, generalizes across hospitals, and is genuinely interpretable, not just explainable after the fact. That question is not settled by building one model and calling it done. It is settled by comparing model families honestly, on the same data, the same tasks, and the same held-out patients:

- **Gradient-boosted trees**, tuned per task on hand-built features. The strong baseline throughout this project (`odyssey/inference/baseline_features.py`, `odyssey/inference/alerts.py`), not a strawman: a substantial prior literature (Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022; Shwartz-Ziv & Armon, "Tabular Data: Deep Learning is Not All You Need", Information Fusion 2022) shows tree ensembles are genuinely hard to beat on tabular healthcare prediction, and this project treats that as a real bar to clear, not an assumption to wave past.
- **Tabular foundation models**, pretrained once and applied zero-shot via in-context learning rather than fit per task. [TabICLv2](https://github.com/soda-inria/tabicl) is the first one integrated (`odyssey/inference/tabicl_baseline.py`, optional dependency, not yet run against real data, see the research journal for the plan).
- **Additive models** (NAMs/GAMs), where every feature's contribution is a single learned curve a clinician can read directly, no post-hoc explainer required. Not yet in the codebase; a candidate next baseline precisely because their interpretability is structural rather than approximated.
- **The hybrid Mamba sequence model**, this project's own architecture: one model forecasting a patient's entire clinical timeline, with a concept bottleneck for interpretability and, in principle, causal-style test-time intervention.

Whichever family wins, it has to win on three axes at once, not one: **generalizability** (the same pipeline, unchanged, produces a strong model on more than one hospital system), **performance** (forecasting strength and calibrated time-to-event alerts that match or beat the tuned baselines above), and **interpretability** (a clinician can trust and act on what the model reports, not just read a number off it).

Interpretability itself needs a real definition, because the field mostly answers it with feature attribution (SHAP, feature importance) and stops there. Attribution answers "what did the model weigh" for a single prediction; it is not causal, does not license "if this were different, the forecast would change this way", and does not license overriding the model's belief about a patient and trusting the result. That gap is exactly why this project builds toward genuine causal capability rather than settling for attribution:

- The **concept bottleneck** (`odyssey/models/concept_bottleneck.py`) is the current causal-lever candidate: a clinician overrides a named clinical concept ("assume they are hypotensive") and the forecast should update, a well-defined intervention on a named concept rather than inserting a hypothetical treatment into an observational record (which the model would read as "a sick patient", confounding by indication). As of the latest investigation (`research_journal/experiments/23_concept_lever_leakage_investigation.html`), this lever is not yet trustworthy: the concept embeddings behave correlationally, tracking how unusual a forced value is rather than its causal content, and the standard training-time fix (RandInt) cannot repair this because it only ever trains on true concept values, never counterfactual ones. Two architectural variants are in flight (independent training of the bottleneck; the second, unrelated candidate is capping the residual channel) to see whether either produces a lever that actually deserves the word "causal".
- Beyond the bottleneck, the project is deliberately pulling in causal machine learning research rather than reinventing it, including amortized causal-effect estimation via in-context learning (e.g. Krishnan et al., "CausalPFN: Amortized Causal Effect Estimation via In-Context Learning", NeurIPS 2025) and related work on causal effect estimation and bounding under partial identification ([Rahul G. Krishnan's group](https://www.cs.toronto.edu/~rahulgk/)). Adding a genuine causal-inference capability to the pipeline, not just a concept readout, is a first-class open research question, not a solved feature.

The pipeline is built to travel. One codebase extracts, tokenizes, labels concepts, trains and evaluates on MIMIC-IV and on eICU (clinical knowledge is written once, keyed by LOINC, and expanded per source); cross-hospital generalization is then assessed on [GEMINI](https://geminimedicine.ca/), a multi-hospital inpatient dataset. Running the same pipeline unchanged across three real hospital systems, not just one, is itself part of the evidence for which model family actually generalizes.

The forecasting has to be strong across *every* kind of event, not only the frequent ones. Events arrive in bundles at one timestamp (a lab panel, a medication order set, the diagnoses coded at discharge) with no meaningful order inside a bundle, so the model is framed as a marked temporal point process over bundles: at each step it forecasts *when* the next bundle arrives (a hazard, so survival curves and "within N hours" fall out naturally) and *which* events it contains (a set, scored as such). Medications, procedures, diagnoses and billing are held to the same bar as labs.

**What success looks like.** A model, from whichever family earns it, that: forecasts strongly across every event family on patients it has never seen; produces calibrated time-to-event alerts that match or beat the tuned GBM and tabular-foundation-model baselines, not just a basic feature set; holds that performance unchanged across MIMIC-IV, eICU, and GEMINI; and supports an intervention a clinician can trust, verified with causal-intervention tests rather than assumed from architecture alone. On the current Mamba/CBM runs: set-based forecasting and calibrated hazard heads are measured and mostly competitive (within 0.01 to 0.09 AUROC of the tuned GBM on most event/horizon pairs, ahead on vasopressor start; see the research journal for exact numbers), concept readouts are faithful (AUROC 0.6 to 0.99), and the causal lever is not yet reliable, reported honestly rather than glossed over.

**What this is not.** Not a diagnostic system, not trained on outcome labels, not a claim that any one model family has already won, and not a claim of interpretability by construction: every interpretability property above has a test, and the research journal records where each model currently falls short.

**Sequencing.** Research questions first: which model family wins on generalizability, performance, and interpretability, and whether a genuine causal lever is achievable at all with the current architecture or requires a different one. Deployment-friendly packaging comes after those questions have real answers, not before.

**Status: active research.** Trained and evaluated on MIMIC-IV and eICU subsets; the eICU pipeline is validated end to end; TabICL integration and a from-scratch causal-lever investigation are both in flight; full-scale MIMIC-IV training is running. See [Roadmap](#roadmap).

## Architecture

![Odyssey architecture](docs/figures/architecture.svg)

*Figure: data flows from MIMIC-IV / eICU through MEDS extraction and tokenization into the hybrid Mamba-2 + attention backbone and concept bottleneck, whose heads produce forecasts, survival curves and alerts, and concept readouts (editable source: `docs/figures/architecture.drawio`).*

Inputs per token: the code (with a clinical or quantile value bin folded in), the
standardized numeric value (opt-in value channel), inter-event time, age, visit
structure, and the patient's static facts (sex, race, ...) placed as the first
tokens of the sequence. A no-bottleneck variant (`model_kind="baseline"`, same
backbone and heads) prices the bottleneck. Everything is streamed over each
patient's whole record in 512-token chunks with carried recurrent state.

The concept bottleneck implements Ismail, Adebayo, Bravo, Ra & Cho, ["Concept Bottleneck Generative Models"](https://proceedings.iclr.cc/paper_files/paper/2024/file/9149fc44c95ce58e3ca529a1e34c2691-Paper-Conference.pdf) (ICLR 2024): task loss plus supervised concept loss plus an orthogonality penalty, verified directly against the paper's Section 3.1/Eq. 5 and its official reference code ([prescient-design/CBGM](https://github.com/prescient-design/CBGM), [mateoespinosa/cem](https://github.com/mateoespinosa/cem)), not just the abstract. Each concept (including the unsupervised "unknown" one) is a *mixture of two learned embeddings*, not a scalar; see the module docstring for why that distinction is load-bearing (the paper's own ablation shows removing the unknown concept's embedding capacity, not merely having some free capacity, degrades FID 9.3 to 44.1).

## Installation

**Python ≥ 3.12** and [uv](https://github.com/astral-sh/uv) are required.

```bash
git clone https://github.com/VectorInstitute/odyssey.git
cd odyssey
uv sync --dev
```

The hybrid Mamba-2 + attention backbone depends on `mamba-ssm`, which requires CUDA/`nvcc` to build and cannot be installed on a Mac dev machine. On a CUDA-capable GPU host:

```bash
uv sync --extra cuda --no-build-isolation
```

Local (CPU/MPS) development uses a lightweight stand-in backbone so the concept-bottleneck logic can be built and tested without a GPU; see `tests/odyssey/models/test_concept_bottleneck.py`.

## Data pipeline

MIMIC-IV to MEDS extraction uses the standard [`meds-extract`](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) tooling (`hosp` + `icu` modules only; MIMIC-IV-ED is a separate dataset/DUA and is not yet wired in).

```bash
# No credentials needed, validates the pipeline against the public demo:
uv run meds-extract-run spec=MIMIC-IV output_dir=<output_dir> dataset_key=demo

# Full MIMIC-IV 3.1, already downloaded locally:
uv run meds-extract-run spec=MIMIC-IV output_dir=<output_dir> \
    do_download=false input_dir=<path_to_mimiciv_3.1>
```

`do_download=false` skips *all* downloads, including ten small auxiliary concept-mapping CSVs the pipeline fetches from `MIT-LCP/mimic-code` on GitHub (not PhysioNet) for `extract_code_metadata`; these aren't part of the MIMIC-IV release itself. If you're pointing `input_dir` at a manually-downloaded copy, fetch those into its root first:

```bash
BASE="https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map"
for f in meas_chartevents_main.csv inputevents_to_rxnorm.csv lab_itemid_to_loinc.csv \
         meas_chartevents_value.csv numerics-summary.csv outputevents_to_loinc.csv \
         d_labitems_to_loinc.csv proc_datetimeevents.csv waveforms-summary.csv proc_itemid.csv; do
  curl -sSL -o "<path_to_mimiciv_3.1>/$f" "$BASE/$f"
done
```

Validated end-to-end against the real, credentialed MIMIC-IV 3.1 (364,627 subjects, 148,193 distinct codes), not just the demo.

### eICU-CRD

eICU uses the same `meds-extract` tooling with a project-local MESSY spec at [`specs/eICU.yaml`](specs/eICU.yaml) (the reference `eicu-meds` PyPI package predates MESSY and pins an incompatible `meds-transforms`/`polars`, so the extraction is expressed declaratively there instead; see that file's header for the eICU-specific design notes: subjects are health-system stays, and all timestamps are pseudotimes reconstructed from minute offsets, so only intra-subject relative times are meaningful):

```bash
# No credentials needed, validates the pipeline against the public eICU demo:
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> dataset_key=demo

# Full eICU-CRD 2.0, already downloaded locally:
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> \
    do_download=false input_dir=<path_to_eicu_2.0>
```

`odyssey/data/code_mapping.py`'s eICU table translates the extraction's code prefixes (`VITALS//PERIODIC//...`, `LAB//{labname}//...`) to the same LOINC codes the canonical concept rules are grounded in, and `concepts_for_source("eicu")` expands one canonical rule set per source.

Spec v2 (the current file) also fixes two medication-identity gaps in the reference ETL's code shapes: eICU leaves `drugname` empty on 36% of medication rows, but 94% of those carry a HICL code, so medication codes are `MEDICATION//STARTED|STOPPED//{drugname}//{hicl}` and the normalizer resolves the HICL first through a shipped empirical dictionary (`odyssey/data/resources/eicu_hicl_ingredients.csv`, rebuilt from the raw tables by `scripts/build_eicu_hicl_lookup.py`); and infusions are `INFUSION_DRUG//{drugname}` instead of a bare token with the name only in `text_value`. Extractions made with spec v1 still load, they just keep the unnamed rows as `unk`. Set `TrainingConfig.source = "eicu"` so normalization, concept expansion and clinical value ranges all pick the eICU tables.

### Tokenization

`odyssey/data/vocabulary.py` and `odyssey/data/sequences.py` turn raw MEDS events into the batches the model consumes. `odyssey/data/value_binning.py` runs first, folding each numeric-valued event's magnitude into the token itself: `"LAB//220045//bpm"` (a heart-rate reading, any value) becomes `"LAB//220045//bpm::HIGH"`, via curated clinical ranges for the vitals/labs `odyssey/data/concepts.py` already defines thresholds for, and per-code quantile bins (fit on the training split only) elsewhere. Codes with no numeric value (a diagnosis, a procedure) pass through unchanged, since the event's occurrence is already the full signal:

```python
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.data.sequences import build_patient_sequence, collate_patient_sequences

binner = QuantileBinner.fit(train_events, n_bins=5, min_count=100)  # train split only
events = add_value_tokens(events, binner)

vocab = Vocabulary.build(events["code"].to_list(), min_count=10, max_size=20_000)
sequences = [
    build_patient_sequence(events.filter(pl.col("subject_id") == sid), vocab, max_seq_len=512)
    for sid in subject_ids
]
batch = collate_patient_sequences(sequences)  # -> ClinicalSequenceBatch, ready for the model
```

Validated at scale against the real extraction: 500 real patients tokenize in ~2s, mean sequence length ~301 events, 0.8% `[UNK]` rate. Visits are derived from `hadm_id` (events sharing one become one visit; events without one each get their own single-event visit), a documented v1 simplification, see the module docstring. Inter-event time (including gaps *between* admissions, not just within one) is already encoded regardless of value-binning: `PatientSequence.time_stamps` holds each event's absolute time since the sequence's first event, and `TimeEmbeddingLayer(is_time_delta=True)` computes real consecutive-event deltas from it, so it survives truncation and visit boundaries unchanged.

Two more per-token inputs exist alongside the bin token: with `TrainingConfig.value_embeddings=True` the binner's per-code standardized value (`numeric_z`, from `QuantileBinner.standardize`) is projected into the token embedding, so the model sees how far into a bin a reading is (a creatinine of 0.8 vs 1.4 are both `NORMAL` tokens); and timeless facts (`GENDER//F`, race, ...) lead every sequence at the first event's timestamp as inputs that are never prediction targets. Medication codes are normalized to ingredient level (`odyssey/data/code_normalization.py`), on eICU through a shipped HICL dictionary that resolves the 36% of medication rows with no drug name.

Sequences are built from each subject's **complete history**, not scoped to one admission or a fixed window; see `research_journal/02_sequence_scoping_methodology.html` (local-only) for why. The same pipeline runs unchanged on MIMIC-IV and eICU; cross-hospital/health-system generalization will be assessed on [GEMINI](https://geminimedicine.ca/) (~30 hospitals, inpatient), not between MIMIC and eICU.

## GEMINI

Cross-hospital generalization is assessed on [GEMINI](https://geminimedicine.ca/), a ~30-hospital inpatient database. Nobody on this team has a login on the GEMINI node except Amrit, so all GEMINI-facing work is git-mediated: we push a script to a second, 1 MiB-per-push-capped remote, Amrit runs it on the node, and only small aggregate/cell-suppressed output comes back in a commit, never patient-level data or model checkpoints. See [`docs/gemini.md`](docs/gemini.md) for the full workflow, credential pattern, and governance rules.

## Development

```bash
uv run pytest -m "not integration_test" tests/
uv run ruff check odyssey tests
uv run mypy odyssey
```

## Roadmap

1. ~~Validate the MEDS extraction pipeline (hosp + icu) end-to-end~~
2. ~~Implement and rigorously test the concept bottleneck layer~~
3. ~~Derive real clinical concept labels from MIMIC-IV codes (rule-based, e.g. SIRS criteria, AKI, hypotension)~~
4. ~~Wire the concept bottleneck into a real Mamba backbone; validate forward+backward on a real GPU~~
5. ~~Run the real MEDS extraction on full, credentialed MIMIC-IV 3.1 (364,627 subjects)~~
6. ~~Build patient-sequence tokenization (MEDS events -> the token/type/time/age/visit-order sequences the model consumes)~~
7. ~~Fold numeric lab/vital values into the token itself (clinical-range + quantile-bin fallback), not code-identity alone~~
8. ~~Streaming truncated-BPTT training and evaluation over full patient histories (subset runs on MIMIC-IV, with visit-scoped concept supervision, set-based next-event scoring, and causal-intervention evaluation)~~; leakage-control series (fixed pair embeddings, teacher-forced training, capped residual) run and analyzed, root cause identified (`research_journal/experiments/23_concept_lever_leakage_investigation.html`): the concept lever is correlational, not causal, and the standard fix (RandInt) cannot repair this by construction; independent training of the bottleneck is the current candidate fix, in progress
9. Extend extraction to MIMIC-IV-ED
10. ~~Multi-dataset pipeline: the same extraction/tokenization/concept pipeline running on eICU as well as MIMIC-IV~~, done: `specs/eICU.yaml`, validated on the full eICU-CRD 2.0 (166K stays, 856M events); concept rules and clinical value bins are canonical (LOINC-keyed) and expanded per source
11. Cross-hospital/health-system generalization: extend the pipeline to [GEMINI](https://geminimedicine.ca/) (~30 hospitals, inpatient/general internal medicine): the multi-hospital dataset where generalization is actually assessed; MIMIC-IV and eICU serve as pipeline-portability targets, not as a train-on-one/test-on-the-other experiment
12. ~~Bundle-aware forecasting: permutation-invariant loss within same-timestamp bundles (restricted to the target's own family) and family-balanced loss weighting~~ -- built and measured: same-family set top-1 69.9% -> 75.7% overall on the MIMIC-IV subset (labs 75 -> 81, medications 36 -> 44, diagnoses 31 -> 41, procedures 24 -> 38); with the value channel and static inputs 76.7%; a baseline without the bottleneck reaches 77.9%; eICU 86.4%
13. ~~Time-to-event: a hazard head for time to the next bundle, and per-event hazard heads (vasopressor start, ICU admission, AKI, death) trained with right censoring; alert evaluation harness scoring P(event within 8/24/72h) with time-dependent AUROC/Brier/calibration against per-event gradient-boosted baselines on hand features~~ -- built and measured: hazard heads reach 0.68-0.95 AUROC across events and horizons on the MIMIC-IV subset (calibrated), against a tuned 609-feature gradient-boosted baseline (`odyssey/inference/baseline_features.py`, fitted on the same training patients) at 0.78-0.97; the head leads only on vasopressor start, and the gap concentrates where a fresh precursor lab exists and late in long stays; per-event survival curves render in the report
14. Bundle-level set prediction head and hierarchical ICD (category, then code) for discharge diagnoses; prior-diagnosis history recap at admission (built, opt-in, untested at scale)
15. Phase 2: an LLM agent (e.g. MedGemma) that reads the concept-annotated forecast and assists a clinician; retrospective clinician validation on GEMINI
16. ~~Paper-grade bespoke baselines: best-effort feature panel (48 LOINC-keyed vitals/labs with window statistics and trends, drug-class exposures, ICU/visit context) with per-event, per-horizon tuning; per-index-row dumps for stratified error analysis~~ -- done
17. ~~eICU spec v2: medication identity via HICL, named infusions, GCS and urine output from the flowsheets~~ -- done; the eICU subset runs replicate the MIMIC-IV findings (forecasting up, concepts up, same alert picture against the tuned baseline)
18. Full-scale pretraining on all MIMIC-IV training shards (running), then eICU; manuscript in `paper/` (npj Digital Medicine)
19. TabICLv2 as a second strong baseline (`odyssey/inference/tabicl_baseline.py`, optional dependency, branch `feature/tabicl-baseline`): built and tested against a fake classifier, not yet run against real data; queued to run after the current causal-lever experiments finish on both VMs; plan and known scope caveats in `research_journal/experiments/24_tabicl_baseline_plan.html`
20. A NAM/GAM baseline, where each feature's contribution is a directly readable learned curve instead of a post-hoc explanation: not yet started, motivated by wanting a model whose interpretability is structural rather than approximated by SHAP/feature importance
21. Genuine causal-inference capability beyond the concept bottleneck's own lever: drawing on amortized causal-effect estimation via in-context learning (Krishnan et al., "CausalPFN", NeurIPS 2025, and related work from [Rahul G. Krishnan's group](https://www.cs.toronto.edu/~rahulgk/)); not yet started, an open research question rather than a scheduled feature

### Known concept-rule limitations

Concept labels are rule-derived, per visit, and evaluated over a visit's whole window (did this happen during the visit), with each concept's first-trigger time also recorded so a running "true as of now" label exists for interventions. Sustained/windowed criteria are used where a single reading over-triggers (`sustained_tachypnea`, KDIGO creatinine windows); GCS-dependent criteria are unavailable on eICU until its nurse-charting table is extracted; urine-output-based AKI staging and full SOFA/NEWS2 are not implemented.

### GPU notes

The real backbone (`EHRHybridBackbone`, `odyssey/models/backbones/hybrid.py`) runs a Mamba-2 mixer and an attention mixer in parallel on every position, fused by a small learned attention (`MergeAttention`), not a sequential stack, so it's built directly rather than through `mamba_ssm`'s high-level `MixerModel` dispatcher, which only supports one mixer per block. The Mamba branch carries real state across TBTT chunks (`hybrid.py` patches a minimal `Mamba2` subclass that seeds `mamba_chunk_scan_combined`'s `initial_states`, which upstream never wires up); the attention branch runs fresh, full attention over just the current chunk, with no cross-chunk memory, a deliberate trade-off, not a bug: Mamba handles compressed long-range recall across the whole sequence, attention handles precise local recall within a chunk. See `_make_mamba2_with_state_cls` in that module and `research_journal/03_backbone_architecture.html` (local-only) for the full writeup.

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
