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
  <img src="https://img.shields.io/badge/mamba--ssm-2.3.1-8A25C9.svg" alt="mamba-ssm 2.3.1">
</p>

---

Odyssey is a toolkit for building clinical foundation models from Electronic Health Records (EHR). It uses **EHR-Mamba3** — a Mamba-3 SSM backbone enriched with clinical embeddings (token types, timestamps, ages, visit structure) — and a **MEDS 0.4+ pipeline** that converts raw MIMIC-IV data to the model-ready format.

## Architecture

**EHR-Mamba3** wraps `mamba_ssm.MambaLMHeadModel` with `ssm_cfg={"layer": "Mamba3"}` and injects six EHR-specific embedding streams via a caching bridge (`CachedEHREmbeddings`) that replaces the backbone's standard token embedding layer.

```
MIMIC-IV CSVs
    ↓  scripts/meds/extract_mimic_iv.py
MEDS parquet  (subject_id · time · code · numeric_value)
    ↓  meds-transforms 0.6.1  (filter + normalize)
    ↓  scripts/meds/meds_to_odyssey.py
Odyssey parquet  (event_tokens · type_tokens · time_tokens · age_tokens · …)
    ↓
pretrain.py  →  Mamba3Pretrain  (next-token prediction)
finetune.py  →  Mamba3Finetune  (single- or multi-task classification)
```

## Installation

**Python ≥ 3.12** and [uv](https://github.com/astral-sh/uv) are required.

```bash
git clone https://github.com/VectorInstitute/odyssey.git
cd odyssey
uv sync --dev
```

On a CUDA-capable GPU host, also install mamba-ssm (requires `nvcc`):

```bash
# Install the torch wheel matching your CUDA version first, e.g. CUDA 12.8:
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Then build mamba-ssm from source:
uv sync --extra cuda --no-build-isolation
```

## Data Pipeline

### 1 — Download MIMIC-IV 3.1

PhysioNet credentials are required. Download using `wget` or the physionet client:

```bash
wget -r -N -c -np --user <physionet_user> --ask-password \
    https://physionet.org/files/mimiciv/3.1/ \
    -P data/
```

### 2 — Run the end-to-end pipeline

```bash
bash scripts/meds/run_pipeline.sh \
    --mimic_dir  data/physionet.org/files/mimiciv/3.1 \
    --output_dir data/pipeline_output \
    --max_seq_len 2048
```

This runs three steps:

| Step | Script | Input → Output |
|------|--------|----------------|
| Extract | `scripts/meds/extract_mimic_iv.py` | MIMIC-IV CSVs → MEDS parquet shards |
| Transform | `scripts/meds/pipeline.yaml` (meds-transforms 0.6.1) | Filter subjects/codes, normalize numeric values |
| Tokenize | `scripts/meds/meds_to_odyssey.py` | MEDS → odyssey token sequence parquets |

The output at `data/pipeline_output/odyssey_tokenized/` contains `train/`, `tuning/`, and `held_out/` splits ready for training.

## Training

### Pre-training

```bash
python pretrain.py \
    --data_dir       data/pipeline_output/odyssey_tokenized \
    --sequence_file  train.parquet \
    --id_file        subject_ids.json \
    --vocab_dir      data/vocab \
    --config_path    odyssey/models/configs/ehr_mamba3.yaml \
    --batch_size     32 \
    --max_len        2048
```

### Fine-tuning

```bash
python finetune.py \
    --pretrain_model_path checkpoints/pretrain.ckpt \
    --data_dir            data/pipeline_output/odyssey_tokenized \
    --vocab_dir           data/vocab \
    --task                mortality \
    --num_labels          2
```

## Model Configuration

Default Mamba-3 hyperparameters (`odyssey/models/configs/ehr_mamba3.yaml`):

```yaml
model:
  embedding_size: 768
  num_hidden_layers: 32
  state_size: 128       # d_state per SSM block
  headdim: 64           # head dimension
  is_mimo: true         # Multi-Input Multi-Output mode
  mimo_rank: 4
  chunk_size: 256
```

## Project Structure

```
odyssey/
├── data/
│   ├── dataset.py       # PretrainDatasetDecoder, FinetuneDatasetDecoder, …
│   └── tokenizer.py     # ConceptTokenizer
├── evals/
│   ├── evaluation.py    # calculate_metrics (AUROC, F1, …)
│   └── prediction.py    # Forecast (autoregressive token generation)
├── models/
│   ├── embeddings.py    # CachedEHREmbeddings, TimeEmbeddingLayer, VisitEmbedding
│   ├── ehr_mamba3/
│   │   └── model.py     # Mamba3Pretrain, Mamba3Finetune
│   └── configs/
│       └── ehr_mamba3.yaml
└── utils/
scripts/
└── meds/
    ├── extract_mimic_iv.py   # MIMIC-IV → MEDS
    ├── meds_to_odyssey.py    # MEDS → odyssey format
    ├── pipeline.yaml         # meds-transforms pipeline
    └── run_pipeline.sh       # end-to-end runner
pretrain.py
finetune.py
```

## Contributing

Issues and pull requests are welcome. Please open an issue before starting large changes.

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
