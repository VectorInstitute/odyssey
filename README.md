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

Odyssey builds **interpretable** clinical foundation models from Electronic Health Records. Prior work (EHR-Mamba3, [arXiv:2405.14567](https://arxiv.org/abs/2405.14567)) showed that a Mamba SSM trained with next-token prediction over patient event sequences learns strong representations for forecasting — but, like most large sequence models, its predictions aren't inspectable. This iteration adds a **concept bottleneck**: the backbone's hidden state is split into (a) a small set of clinically-grounded, supervised concepts (e.g. "on vasopressors", "acute kidney injury"), and (b) a free "unknown concept" residual that absorbs whatever else the task needs. A patient-timeline forecast can then be explained in terms of the concepts that drove it, instead of an opaque embedding.

**Status: active research rebuild, not yet trained on real patient data.** See [Roadmap](#roadmap) below.

## Architecture (target)

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

eICU (Phase 2, cross-institution) uses the same `meds-extract` tooling with a project-local MESSY spec at [`specs/eICU.yaml`](specs/eICU.yaml) (the reference `eicu-meds` PyPI package predates MESSY and pins an incompatible `meds-transforms`/`polars`, so the extraction is expressed declaratively there instead — see that file's header for the eICU-specific design notes: subjects are health-system stays, and all timestamps are pseudotimes reconstructed from minute offsets, so only intra-subject relative times are meaningful):

```bash
# No credentials needed — validates the pipeline against the public eICU demo:
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> dataset_key=demo

# Full eICU-CRD 2.0, already downloaded locally:
uv run meds-extract-run spec=./specs/eICU.yaml output_dir=<output_dir> \
    do_download=false input_dir=<path_to_eicu_2.0>
```

`odyssey/data/code_mapping.py`'s eICU table translates the extraction's code prefixes (`VITALS//PERIODIC//...`, `LAB//{labname}//...`) to the same LOINC codes the MIMIC-IV concept rules are grounded in. The concept *rules* themselves (`odyssey/data/concepts.py`) are still keyed on MIMIC-IV prefixes — parameterizing them per-source via the LOINC layer is the next step for cross-institution concept supervision.

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

Sequences are built from each subject's **complete history**, not scoped to one admission or a fixed window — see `research_journal/02_sequence_scoping_methodology.html` (local-only) for why, and for the plan to test whether institution-specific patterns learned from MIMIC-IV's single hospital generalize once Phase 2 adds eICU and GEMINI.

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
8. Pretrain on real MIMIC-IV 3.1 at scale (GPU), on full patient history via Mamba chunked/TBTT training
9. Extend extraction to MIMIC-IV-ED
10. Cross-institution generalization: extend extraction to eICU and [GEMINI](https://geminimedicine.ca/) (multi-hospital, and ICU vs. internal medicine) to test whether patterns learned from MIMIC-IV's single hospital transfer — eICU MEDS extraction now in place (`specs/eICU.yaml`, validated on the public demo); next: run on full eICU-CRD 2.0, parameterize concept rules per-source via the LOINC mapping layer
11. Phase 2: an LLM agent (e.g. MedGemma) that reads the concept-annotated forecast and assists a clinician

### Known concept-rule limitations

Current v1 concept thresholds are single-timepoint, not sustained-criteria (real clinical definitions usually require e.g. persistence over a time window). On the full real dataset, `tachypnea` (RR > 20) triggers for 96.5% of subjects with respiratory-rate data — too loose to be a useful signal as-is, and worth recalibrating (e.g. against a sustained/windowed version, or a stricter threshold) before relying on it for supervision.

### GPU notes

The real backbone (`EHRHybridBackbone`, `odyssey/models/backbones/hybrid.py`) runs a Mamba-2 mixer and an attention mixer in parallel on every position, fused by a small learned attention (`MergeAttention`) — not a sequential stack, so it's built directly rather than through `mamba_ssm`'s high-level `MixerModel` dispatcher, which only supports one mixer per block. The Mamba branch carries real state across TBTT chunks (`hybrid.py` patches a minimal `Mamba2` subclass that seeds `mamba_chunk_scan_combined`'s `initial_states`, which upstream never wires up); the attention branch runs fresh, full attention over just the current chunk, with no cross-chunk memory — a deliberate trade-off, not a bug: Mamba handles compressed long-range recall across the whole sequence, attention handles precise local recall within a chunk. See `_make_mamba2_with_state_cls` in that module and `research_journal/03_backbone_architecture.html` (local-only) for the full writeup.

## Citation

If you use Odyssey or EHR-Mamba3 in your research, please cite:

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
