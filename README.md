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
    ├─ known concepts   — supervised, clinically interpretable
    └─ residual         — free, orthogonality-penalized against the concepts
    ↓
task loss + concept loss + orthogonality loss
    ↓
forecast (adverse events, deterioration) — traceable to concept activations
```

The concept-bottleneck loss recipe follows Ismail, Adebayo, Bravo, Ra & Cho, ["Concept Bottleneck Generative Models"](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9149fc44c95ce58e3ca529a1e34c2691-Abstract-Conference.html) (ICLR 2024): task loss + supervised concept loss + an orthogonality penalty that keeps the residual from silently re-encoding the known concepts.

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

## Development

```bash
uv run pytest -m "not integration_test" tests/
uv run ruff check odyssey tests
uv run mypy odyssey
```

## Roadmap

1. ~~Validate the MEDS extraction pipeline (hosp + icu) end-to-end~~
2. ~~Implement and rigorously test the concept bottleneck layer~~
3. Derive real clinical concept labels from MIMIC-IV codes (rule-based, e.g. SIRS criteria, AKI, hypotension)
4. Wire the concept bottleneck into the EHR-Mamba3 backbone; pretrain on real MIMIC-IV 3.1 (GPU-only)
5. Extend extraction to MIMIC-IV-ED
6. Phase 2: an LLM agent (e.g. MedGemma) that reads the concept-annotated forecast and assists a clinician

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
