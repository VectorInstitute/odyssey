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
EHR-Mamba3 backbone  (next-token prediction over patient event sequences)
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

The EHR-Mamba3 backbone depends on `mamba-ssm`, which requires CUDA/`nvcc` to build and cannot be installed on a Mac dev machine. On a CUDA-capable GPU host:

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
4. ~~Wire the concept bottleneck into the EHR-Mamba3 backbone; validate forward+backward on a real GPU~~
5. ~~Run the real MEDS extraction on full, credentialed MIMIC-IV 3.1 (364,627 subjects)~~
6. Build patient-sequence tokenization (MEDS events -> the token/type/time/age/visit-order sequences the model consumes) — not yet implemented; needed before real pretraining
7. Pretrain on real MIMIC-IV 3.1 at scale (GPU)
8. Extend extraction to MIMIC-IV-ED
9. Phase 2: an LLM agent (e.g. MedGemma) that reads the concept-annotated forecast and assists a clinician

### Known concept-rule limitations

Current v1 concept thresholds are single-timepoint, not sustained-criteria (real clinical definitions usually require e.g. persistence over a time window). On the full real dataset, `tachypnea` (RR > 20) triggers for 96.5% of subjects with respiratory-rate data — too loose to be a useful signal as-is, and worth recalibrating (e.g. against a sustained/windowed version, or a stricter threshold) before relying on it for supervision.

### GPU notes

`mamba-ssm`'s high-level `MambaLMHeadModel`/`MixerModel` wrapper only dispatches `ssm_cfg={"layer": ...}` to Mamba1/Mamba2 (as of 2.3.2), even though the package ships real Mamba-3 kernels — `EHRMamba3Backbone` builds the block stack directly instead (see its module docstring). Mamba-3's MIMO kernels also require `seq_len % chunk_size == 0` and are only validated here at `headdim=64`; smaller `headdim` values hit TileLang warp-partitioning errors in the backward kernel for this `mamba-ssm` version.

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
