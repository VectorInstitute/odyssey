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

Odyssey builds one general forecasting model of a patient's clinical timeline, and uses it to alert clinicians and to let them ask "what if".

Most clinical AI models are bespoke: one model for mortality, another for sepsis, another for deterioration, each trained on its own labels and useless outside its task. Odyssey instead trains a single model to forecast the patient's event sequence itself (labs, vitals, medications, diagnoses, procedures, admissions, outcomes) the way a language model forecasts the next word, over the whole record. Prior work ([EHRMamba](https://arxiv.org/abs/2405.14567)) showed this learns strong representations. Everything downstream is derived from the same model rather than trained separately:

- **Alerts.** Roll the model forward from a patient's current state and read off the probability of events that matter (vasopressor start, ICU transfer, acute kidney injury, death) within a horizon. One model, every alert.
- **Interaction.** Insert a hypothetical event ("we give this medication now") and roll forward again: how do the likely futures change? Or override the model's belief about the patient ("assume they are not septic") and see what the forecast does.

For that to be usable at the bedside the forecasts have to be inspectable, so Odyssey puts a **concept bottleneck** between the sequence model and its predictions: everything flows through a small set of named clinical concepts (tachycardia, hypotension, acute kidney injury, SIRS, on vasopressors, ...) plus one unnamed residual channel. Each concept probability is supervised against rule-derived clinical labels and scored on held-out patients, so the readout a clinician sees is measured, not assumed; each forecast can be traced to the concepts that drove it; and a clinician can override a concept and watch the forecast update. Whether the concepts really are that lever, rather than a decorative side channel, is itself tested with causal interventions, and the results are reported honestly either way.

The pipeline is built to travel. One codebase extracts, tokenizes, labels concepts, trains and evaluates on MIMIC-IV and on eICU (clinical knowledge is written once, keyed by LOINC, and expanded per source). Those two prove portability; cross-hospital generalization is assessed on [GEMINI](https://geminimedicine.ca/), a multi-hospital inpatient dataset.

**What success looks like.** On patients the model has never seen: forecasting quality close to an unconstrained model of the same size (the bottleneck costs little); alert probabilities read off the forecasts that match or beat bespoke single-task models on calibration and discrimination; concept probabilities that track the patient's real state; interventions that move forecasts in the clinically expected direction; and all of it holding across hospitals.

**What this is not, and one caveat.** Not a diagnostic system, not trained on outcome labels, and not a claim of interpretability by construction: every interpretability property above has a test, and the research journal records where the model currently falls short. And a "what if we give this drug" answer from a model trained on observational records is what the model expects to *see* next, not a causal treatment effect: drugs are given to sick patients, and the model knows it. That caveat is stated wherever what-if results are shown.

**Status: active research.** Trained and evaluated on MIMIC-IV subsets; the eICU pipeline is validated end to end; time-to-event forecasting, rollouts for alerts, and full-scale MIMIC-IV training are next. See [Roadmap](#roadmap).

## Architecture

```
MIMIC-IV 3.1 (hosp + icu)
    ↓  meds-extract  (Medical-Event-Data-Standard/MIMIC_IV_MEDS spec)
MEDS parquet  (subject_id · time · code · numeric_value)
    ↓  concept extraction (rule-derived labels from MEDS codes)
    ↓
Hybrid Mamba-2 + attention backbone  (next-token prediction over patient event sequences)
    ↓
ConceptBottleneck  (odyssey/models/concept_bottleneck.py)
    ├─ k known concepts    — each a soft mixture of a learned active/inactive
    │                        embedding pair, weighted by a supervised probability
    └─ 1 unknown concept   — same mixture structure, unsupervised, orthogonality-
                             penalized against the known concepts' embeddings
    ↓
task loss + concept loss + orthogonality loss
    ↓
forecast (adverse events, deterioration) — traceable to concept activations
```

The concept bottleneck implements Ismail, Adebayo, Bravo, Ra & Cho, ["Concept Bottleneck Generative Models"](https://proceedings.iclr.cc/paper_files/paper/2024/file/9149fc44c95ce58e3ca529a1e34c2691-Paper-Conference.pdf) (ICLR 2024) — task loss + supervised concept loss + an orthogonality penalty — verified directly against the paper's Section 3.1/Eq. 5 and its official reference code ([prescient-design/CBGM](https://github.com/prescient-design/CBGM), [mateoespinosa/cem](https://github.com/mateoespinosa/cem)), not just the abstract. Each concept (including the unsupervised "unknown" one) is a *mixture of two learned embeddings*, not a scalar — see the module docstring for why that distinction is load-bearing (the paper's own ablation shows removing the unknown concept's embedding capacity, not merely having some free capacity, degrades FID 9.3→44.1).

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

MIMIC-IV → MEDS extraction uses the standard [`meds-extract`](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) tooling (`hosp` + `icu` modules only — MIMIC-IV-ED is a separate dataset/DUA and is not yet wired in).

```bash
# No credentials needed — validates the pipeline against the public demo:
uv run meds-extract-run spec=MIMIC-IV output_dir=<output_dir> dataset_key=demo

# Full MIMIC-IV 3.1, already downloaded locally:
uv run meds-extract-run spec=MIMIC-IV output_dir=<output_dir> \
    do_download=false input_dir=<path_to_mimiciv_3.1>
```

`do_download=false` skips *all* downloads, including ten small auxiliary concept-mapping CSVs the pipeline fetches from `MIT-LCP/mimic-code` on GitHub (not PhysioNet) for `extract_code_metadata` — these aren't part of the MIMIC-IV release itself. If you're pointing `input_dir` at a manually-downloaded copy, fetch those into its root first:

```bash
BASE="https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map"
for f in meas_chartevents_main.csv inputevents_to_rxnorm.csv lab_itemid_to_loinc.csv \
         meas_chartevents_value.csv numerics-summary.csv outputevents_to_loinc.csv \
         d_labitems_to_loinc.csv proc_datetimeevents.csv waveforms-summary.csv proc_itemid.csv; do
  curl -sSL -o "<path_to_mimiciv_3.1>/$f" "$BASE/$f"
done
```

Validated end-to-end against the real, credentialed MIMIC-IV 3.1 (364,627 subjects, 148,193 distinct codes) — not just the demo.

### eICU-CRD

eICU uses the same `meds-extract` tooling with a project-local MESSY spec at [`specs/eICU.yaml`](specs/eICU.yaml) (the reference `eicu-meds` PyPI package predates MESSY and pins an incompatible `meds-transforms`/`polars`, so the extraction is expressed declaratively there instead — see that file's header for the eICU-specific design notes: subjects are health-system stays, and all timestamps are pseudotimes reconstructed from minute offsets, so only intra-subject relative times are meaningful):

```bash
# No credentials needed — validates the pipeline against the public eICU demo:
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> dataset_key=demo

# Full eICU-CRD 2.0, already downloaded locally:
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> \
    do_download=false input_dir=<path_to_eicu_2.0>
```

`odyssey/data/code_mapping.py`'s eICU table translates the extraction's code prefixes (`VITALS//PERIODIC//...`, `LAB//{labname}//...`) to the same LOINC codes the MIMIC-IV concept rules are grounded in. The concept *rules* themselves (`odyssey/data/concepts.py`) are still keyed on MIMIC-IV prefixes — parameterizing them per-source via the LOINC layer is the next step so one canonical rule set supervises every dataset.

### Tokenization

`odyssey/data/vocabulary.py` and `odyssey/data/sequences.py` turn raw MEDS events into the batches the model consumes. `odyssey/data/value_binning.py` runs first, folding each numeric-valued event's magnitude into the token itself — `"LAB//220045//bpm"` (a heart-rate reading, any value) becomes `"LAB//220045//bpm::HIGH"` — via curated clinical ranges for the vitals/labs `odyssey/data/concepts.py` already defines thresholds for, and per-code quantile bins (fit on the training split only) elsewhere. Codes with no numeric value (a diagnosis, a procedure) pass through unchanged, since the event's occurrence is already the full signal:

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

Validated at scale against the real extraction: 500 real patients tokenize in ~2s, mean sequence length ~301 events, 0.8% `[UNK]` rate. Visits are derived from `hadm_id` (events sharing one become one visit; events without one each get their own single-event visit) — a documented v1 simplification, see the module docstring. Inter-event time (including gaps *between* admissions, not just within one) is already encoded regardless of value-binning — `PatientSequence.time_stamps` holds each event's absolute time since the sequence's first event, and `TimeEmbeddingLayer(is_time_delta=True)` computes real consecutive-event deltas from it, so it survives truncation and visit boundaries unchanged.

Sequences are built from each subject's **complete history**, not scoped to one admission or a fixed window — see `research_journal/02_sequence_scoping_methodology.html` (local-only) for why. The same pipeline runs unchanged on MIMIC-IV and eICU; cross-hospital/health-system generalization will be assessed on [GEMINI](https://geminimedicine.ca/) (~30 hospitals, inpatient), not between MIMIC and eICU.

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
8. ~~Streaming truncated-BPTT training and evaluation over full patient histories (subset runs on MIMIC-IV, with visit-scoped concept supervision, set-based next-event scoring, and causal-intervention evaluation)~~; next: intervention-aware training so the concept probabilities become the causal lever, then full-scale MIMIC-IV pretraining
9. Extend extraction to MIMIC-IV-ED
10. ~~Multi-dataset pipeline: the same extraction/tokenization/concept pipeline running on eICU as well as MIMIC-IV~~ — done: `specs/eICU.yaml`, validated on the full eICU-CRD 2.0 (166K stays, 856M events); concept rules and clinical value bins are canonical (LOINC-keyed) and expanded per source
11. Cross-hospital/health-system generalization: extend the pipeline to [GEMINI](https://geminimedicine.ca/) (~30 hospitals, inpatient/general internal medicine) — the multi-hospital dataset where generalization is actually assessed; MIMIC-IV and eICU serve as pipeline-portability targets, not as a train-on-one/test-on-the-other experiment
12. Time-to-next-event forecasting alongside next-event prediction (the model conditions on time but does not yet predict it; alerts and what-ifs need "within N hours")
13. Rollouts: sample many futures from a patient's current state, read off event probabilities and concept trajectories; evaluate alert calibration/discrimination against bespoke single-task baselines on held-out patients
14. What-if interaction: insert hypothetical events and roll forward, with the observational-vs-causal caveat stated and sanity-checked (an antihypertensive should lower forecast BP bins)
15. Phase 2: an LLM agent (e.g. MedGemma) that reads the concept-annotated forecast and assists a clinician

### Known concept-rule limitations

Concept labels are rule-derived, per visit, and evaluated over a visit's whole window (did this happen during the visit), with each concept's first-trigger time also recorded so a running "true as of now" label exists for interventions. Sustained/windowed criteria are used where a single reading over-triggers (`sustained_tachypnea`, KDIGO creatinine windows); GCS-dependent criteria are unavailable on eICU until its nurse-charting table is extracted; urine-output-based AKI staging and full SOFA/NEWS2 are not implemented.

### GPU notes

The real backbone (`EHRHybridBackbone`, `odyssey/models/backbones/hybrid.py`) runs a Mamba-2 mixer and an attention mixer in parallel on every position, fused by a small learned attention (`MergeAttention`) — not a sequential stack, so it's built directly rather than through `mamba_ssm`'s high-level `MixerModel` dispatcher, which only supports one mixer per block. The Mamba branch carries real state across TBTT chunks (`hybrid.py` patches a minimal `Mamba2` subclass that seeds `mamba_chunk_scan_combined`'s `initial_states`, which upstream never wires up); the attention branch runs fresh, full attention over just the current chunk, with no cross-chunk memory — a deliberate trade-off, not a bug: Mamba handles compressed long-range recall across the whole sequence, attention handles precise local recall within a chunk. See `_make_mamba2_with_state_cls` in that module and `research_journal/03_backbone_architecture.html` (local-only) for the full writeup.

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
