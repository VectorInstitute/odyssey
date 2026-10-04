# Using a trained checkpoint

This page is for people who receive a trained Odyssey checkpoint and want to
run it on their own data. It says what the checkpoint is, how it was
trained, and how to feed it data the way it expects.

## Older versions of this repository

Early versions of Odyssey had an `EHR-Mamba3` model, a default config
`odyssey/models/configs/ehr_mamba3.yaml` and a MEDS script
`scripts/meds/run_pipeline.sh`. These were removed in commit `3eef618`, when
the repository was rebuilt around the concept-bottleneck model. The Mamba-3
backbone was then replaced by Mamba-2 (commits `f220d2c`, `febbef5`): its
kernels could not carry state correctly across chunks, which streaming
training needs. Code, configs and checkpoints from that era are not
compatible with the current repository. Use the current code and this page.

## What a checkpoint directory holds

A run directory written by `odyssey.training.train` holds everything needed
to run the model:

| File | What it is |
| --- | --- |
| `checkpoint_best.pt` | weights (`model` key) from the step with the lowest validation loss; the other keys (`optimizer`, `step`, ...) are only for resuming training |
| `config.json` | the full training configuration (architecture, data source, task set) |
| `vocabulary.json` | the token vocabulary, built on the training split |
| `quantile_binner.json` | the value bins, fit on the training split |

`odyssey.inference.run_inference.load_run` rebuilds the model from these
four files. It reads the architecture from the checkpoint's own weights
where the config could be ambiguous, so older run directories still load.

## The MIMIC-IV checkpoint (`full_run_v10`)

**Model.** Hybrid backbone: 8 blocks, each with a Mamba-2 branch and a
chunk-local attention branch run in parallel, hidden size 256. A concept
bottleneck reads the hidden state into 29 clinical concepts (task set `v3`;
mixture form, see the README). Heads on top: next-event forecasting, time
to the next event, and a discrete-time hazard for six events (ICU
admission, vasopressor start, acute kidney injury, Sepsis-3, death, 30-day
readmission). 32.2 million parameters.

**Training.** One joint training run from random initialization. There is
no separate pretraining stage and no fine-tuning stage: the next-event,
timing, concept and hazard losses are all trained together from the start.
Two epochs over all 292 training shards; the best validation loss (2.056)
was at step 37,000. Trained at commit `cdbd4e7`; the registry row is in
[`experiments.md`](experiments.md) under `full_run_v10`.

**Data.** MIMIC-IV 3.1, `hosp` and `icu` modules, extracted to MEDS with the
standard `meds-extract` tooling (see the README's "Data pipeline"). Subjects
are split train / tuning / held_out by the extraction's
`metadata/subject_splits.parquet`. The model saw the train and tuning
subjects. **Score only held_out subjects** if you report numbers on
MIMIC-IV; use the split file that comes with the checkpoint, not one from a
fresh extraction.

**How well it does.** Held-out results (concept readout AUROCs, alert
AUROCs against a tuned gradient-boosted baseline) are in `alerts.json` and
`inference_results.json` beside the checkpoint, and in the paper. The
30-day readmission head is weak (AUROC about 0.59 at 7 days); do not use it
as a baseline.

## Install

Python 3.12 or later and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/VectorInstitute/odyssey.git
cd odyssey
uv sync --dev
```

That is enough to run the model on a CPU or an Apple-silicon GPU (MPS).
Without `mamba-ssm`, the hybrid backbone uses a pure-PyTorch version of its
layers (`odyssey/models/backbones/mamba_portable.py`) with the same weight
names, so the same checkpoint loads. It is for inference only. On an NVIDIA
GPU, install the CUDA kernels for speed (and for training):

```bash
uv sync --extra cuda --no-build-isolation
```

On 12 patients (about 165,000 positions), the portable layers on Apple MPS
match the CUDA kernels to a mean risk difference of 0.0002 (largest 0.009),
and the top next-event forecast agrees at 99.8% of positions.

## Prepare your data

1. **Extract to MEDS** with the same tooling and spec the checkpoint was
   trained on (for MIMIC-IV: `meds-extract-run spec=MIMIC-IV ...`, see the
   README). Another source needs its own spec and its own codes; a
   checkpoint trained on MIMIC-IV only knows MIMIC-IV codes.
2. **Do not rebuild the vocabulary or the value bins.** Tokenize with the
   checkpoint's `vocabulary.json` and `quantile_binner.json`. Codes the
   vocabulary does not know become an unknown token (with a coarser ICD
   back-off for diagnoses), so many unknown codes mean the data does not
   match the training extraction.
3. **Apply the run's own preprocessing.** `config.json` records it
   (`source`, `normalize_medications`, `history_recap`); the example below
   applies it from the loaded config.
4. **Sidecars are not model input.** `<meds root>/sidecars/` (for example
   microbiology for Sepsis-3) only feeds the labels used in evaluation. See
   [`sidecars_and_task_sets.md`](sidecars_and_task_sets.md).

## Run it

**Evaluate on a shard directory** (forecasting, concept readout,
completeness), the same way the paper did:

```bash
uv run python -m odyssey.inference.run_inference --run-dir <run> \
    --held-out-shard-dir <meds root>/data/held_out --output-json results.json
uv run python -m odyssey.inference.alerts --run-dir <run> \
    --held-out-shard-dir <meds root>/data/held_out --output-json alerts.json \
    --baseline-shard-dir <meds root>/data/train   # the GBM baseline is fit here
```

**Score one patient** and read the event risks and concept beliefs at every
position of the record:

```python
import torch

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import build_patient_sequence
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.patient_stream import risk_within, stream_patient
from odyssey.inference.run_inference import load_run
from odyssey.training.data import load_meds_subject
from odyssey.utils.device import default_device

run_dir, shard, subject_id = "full_run_v10", "<meds root>/data/held_out/0.parquet", 123
device = default_device()  # cuda, else Apple mps, else cpu
model, vocab, binner, config = load_run(
    run_dir, device=device, checkpoint_path=f"{run_dir}/checkpoint_best.pt"
)
model.eval()

# Prepare the record exactly as training did: the run's own normalization,
# value bins and vocabulary.
events = load_meds_subject(shard, subject_id)
events = maybe_normalize(events, enabled=config.normalize_medications, source=config.source)
events = maybe_history_recap(events, enabled=config.history_recap)
seq = build_patient_sequence(add_value_tokens(events, binner, source=config.source), vocab)

heads = model.event_heads
risks, concepts = [], []
for span in stream_patient(model, seq, device=device, chunk_size=config.chunk_size):
    n = span.n_real
    hazards = heads(span.fwd.features[0, :n])  # (positions, events, bins)
    risks.append(risk_within(hazards, heads.edges, [8.0, 24.0, 72.0]).cpu())
    concepts.append(span.fwd.bottleneck.concept_probs[0, :n].cpu())
risk = torch.cat(risks)  # (positions, events, horizons): P(event within h)
concept_probs = torch.cat(concepts)  # (positions, concepts)
for name, r in zip(heads.event_names, risk[-1, :, 1].tolist()):
    print(f"{name}: {r:.3f} within 24 h")
```

`stream_patient` feeds the record in chunks and carries the recurrent state
between them, so a record of any length runs in constant memory. On an
Apple M4 an 11,000-event record takes about 4 seconds.

**See it in a browser.** The clinician demo (`apps/clinician_demo`, see
[`clinician_demo.md`](clinician_demo.md)) replays one admission at a time
with the risks, concept beliefs, what-if edits and evidence.

## Data governance

A checkpoint trained on credentialed data (MIMIC-IV, eICU-CRD) is shared
only with people who hold that dataset's credentialed access and have
signed its data use agreement, and only for the purpose agreed. Do not
redistribute it. The `subject_splits.parquet` that comes with it lists
credentialed subject identifiers and is covered by the same terms.
