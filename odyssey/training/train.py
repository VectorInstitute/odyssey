"""End-to-end training: real MEDS shards -> a trained ConceptBottleneckSequenceModel.

Usage (on a CUDA host, from the repo root)::

    uv run python -m odyssey.training.train \\
        --train-shard-dir /path/to/data/train \\
        --tuning-shard-dir /path/to/data/tuning \\
        --output-dir runs/exp1 \\
        --max-train-shards 20 --max-tuning-shards 5

Every loss component (task, concept, orthogonality, observability) is
logged per step to ``<output_dir>/loss_log.jsonl``, one JSON object per
line -- the source the results HTML's loss-curve plots read from
directly, so training is auditable after the fact without re-running it.
Checkpoints are periodic ``torch.save`` dicts of the model/optimizer
state, plus a final one; the fitted quantile binner and vocabulary are
saved alongside so inference can reconstruct the exact same tokenization
without needing the train split again.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, TypeVar

import torch

from odyssey.data.concepts import CONCEPTS
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import PAD_ID
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.data import (
    build_concept_label_dicts,
    build_vocabulary,
    count_subjects,
    iter_patient_sequences,
    load_meds_shards,
)


@dataclass
class TrainingConfig:
    """All paths and hyperparameters for one training run."""

    train_shard_dir: str
    tuning_shard_dir: str
    output_dir: str
    max_train_shards: Optional[int] = None
    max_tuning_shards: Optional[int] = None

    # Backbone (EHRHybridBackbone). Defaults are modest, not the paper-scale
    # numbers -- see the training run's own README note on why.
    hidden_size: int = 256
    num_hidden_layers: int = 8
    mamba_state_size: int = 128
    mamba_headdim: int = 64
    mamba_chunk_size: int = 256
    attn_num_heads: int = 8
    embedding_dim: int = 32

    # Tokenization
    vocab_min_count: int = 5
    vocab_max_size: int = 20_000
    quantile_n_bins: int = 5
    quantile_min_count: int = 100
    max_seq_len: Optional[int] = None

    # Streaming TBTT
    num_lanes: int = 8
    chunk_size: int = 256
    reset_prob: float = 0.1

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_epochs: int = 3
    concept_weight: float = 1.0
    orthogonality_weight: float = 0.1
    observability_weight: float = 0.1

    # Logging/checkpointing cadence, in optimizer steps.
    log_every: int = 20
    eval_every: int = 200
    eval_max_chunks: int = 50
    checkpoint_every: int = 500

    seed: int = 0


class LossLogger:
    """Append-only JSONL logger for per-step loss components."""

    def __init__(self, path: Path) -> None:
        """Open ``path`` for appending; creates it if it doesn't exist."""
        self.path = path
        self._file = path.open("a")

    def log(self, **fields: object) -> None:
        """Write one JSON-serializable record as a line."""
        self._file.write(json.dumps(fields) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()


_Movable = TypeVar("_Movable")


def _move_chunk_to_device(chunk: _Movable, device: str) -> _Movable:
    """Move every tensor field of a (possibly nested) NamedTuple to ``device``.

    Works for :class:`~odyssey.data.streaming.StreamingChunk` and its
    nested :class:`~odyssey.data.types.ClinicalSequenceBatch`/
    :class:`~odyssey.data.types.AuxiliaryInputs` without depending on
    their exact field lists, so a new field added to any of them doesn't
    need a matching change here.
    """
    if isinstance(chunk, torch.Tensor):
        return chunk.to(device)  # type: ignore[return-value]
    if isinstance(chunk, tuple) and hasattr(chunk, "_fields"):  # NamedTuple
        return type(chunk)(*(_move_chunk_to_device(v, device) for v in chunk))
    return chunk


def _detach_state(state: TimeAwareState) -> TimeAwareState:
    """Truncate BPTT across chunks for the hybrid backbone's carried state."""
    from odyssey.models.backbones.hybrid import HybridState  # noqa: PLC0415

    recurrent = state.recurrent
    if not isinstance(recurrent, HybridState):
        raise TypeError(
            f"_detach_state expects EHRHybridBackbone's HybridState, got {type(recurrent)!r}"
        )
    detached = {
        layer_idx: tuple(t.detach() for t in cached)
        for layer_idx, cached in recurrent.mamba_states.items()
    }
    return TimeAwareState(
        recurrent=HybridState(detached),
        prev_time_stamps=state.prev_time_stamps.detach(),
    )


def build_model(
    config: TrainingConfig, *, vocab_size: int, num_concepts: int
) -> ConceptBottleneckSequenceModel:
    """Construct the real backbone + concept bottleneck model from ``config``."""
    from odyssey.models.backbones.hybrid import EHRHybridBackbone  # noqa: PLC0415

    backbone = EHRHybridBackbone(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        padding_idx=PAD_ID,
        num_hidden_layers=config.num_hidden_layers,
        mamba_state_size=config.mamba_state_size,
        mamba_headdim=config.mamba_headdim,
        mamba_chunk_size=config.mamba_chunk_size,
        attn_num_heads=config.attn_num_heads,
    )
    return ConceptBottleneckSequenceModel(
        backbone=backbone,
        vocab_size=vocab_size,
        num_concepts=num_concepts,
        embedding_dim=config.embedding_dim,
        padding_idx=PAD_ID,
    )


def evaluate_streaming(
    model: ConceptBottleneckSequenceModel,
    make_sampler: Callable[[], PackedLaneSampler],
    labels: Dict[int, torch.Tensor],
    masks: Dict[int, torch.Tensor],
    *,
    device: str,
    max_chunks: Optional[int] = None,
) -> Dict[str, float]:
    """Average loss components over one (partial), gradient-free sampler pass."""
    model.eval()
    sampler = make_sampler()
    state = None
    totals: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i, chunk in enumerate(sampler):
            if max_chunks is not None and i >= max_chunks:
                break
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            _, components, state = model.compute_streaming_loss(
                chunk, labels, masks, state=state
            )
            state = _detach_state(state)
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            n += 1
    model.train()
    return {key: value / max(n, 1) for key, value in totals.items()}


def train(config: TrainingConfig) -> Path:  # noqa: PLR0915
    """Run one full training job; returns the output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)

    print(f"[data] loading train shards from {config.train_shard_dir}")
    train_events = load_meds_shards(
        config.train_shard_dir, max_shards=config.max_train_shards
    )
    print(
        f"[data] train: {count_subjects(train_events)} subjects, {train_events.height} events"
    )

    print(f"[data] loading tuning shards from {config.tuning_shard_dir}")
    tuning_events = load_meds_shards(
        config.tuning_shard_dir, max_shards=config.max_tuning_shards
    )
    print(
        f"[data] tuning: {count_subjects(tuning_events)} subjects, {tuning_events.height} events"
    )

    print("[data] fitting quantile binner on train split")
    binner = QuantileBinner.fit(
        train_events, n_bins=config.quantile_n_bins, min_count=config.quantile_min_count
    )
    binner.save(output_dir / "quantile_binner.json")
    train_events_binned = add_value_tokens(train_events, binner)
    tuning_events_binned = add_value_tokens(tuning_events, binner)

    print("[data] building vocabulary from train split")
    vocab = build_vocabulary(
        train_events_binned,
        min_count=config.vocab_min_count,
        max_size=config.vocab_max_size,
    )
    vocab.save(output_dir / "vocabulary.json")
    print(f"[data] vocab size: {len(vocab)}")

    print("[data] labeling concepts")
    train_labels, train_masks = build_concept_label_dicts(train_events, CONCEPTS)
    tuning_labels, tuning_masks = build_concept_label_dicts(tuning_events, CONCEPTS)
    train_labels = {k: v.to(device) for k, v in train_labels.items()}
    train_masks = {k: v.to(device) for k, v in train_masks.items()}
    tuning_labels = {k: v.to(device) for k, v in tuning_labels.items()}
    tuning_masks = {k: v.to(device) for k, v in tuning_masks.items()}

    model = build_model(config, vocab_size=len(vocab), num_concepts=len(CONCEPTS)).to(
        device
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.1f}M parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_weights = ConceptBottleneckLossWeights(
        concept=config.concept_weight,
        orthogonality=config.orthogonality_weight,
        observability=config.observability_weight,
    )

    def make_train_sampler(epoch: int) -> PackedLaneSampler:
        patients = iter_patient_sequences(
            train_events_binned,
            vocab,
            max_seq_len=config.max_seq_len,
            shuffle_seed=config.seed + epoch,
        )
        return PackedLaneSampler(
            patients,
            num_lanes=config.num_lanes,
            chunk_size=config.chunk_size,
            reset_prob=config.reset_prob,
            seed=config.seed + epoch,
        )

    def make_tuning_sampler() -> PackedLaneSampler:
        patients = iter_patient_sequences(
            tuning_events_binned, vocab, max_seq_len=config.max_seq_len
        )
        return PackedLaneSampler(
            patients, num_lanes=config.num_lanes, chunk_size=config.chunk_size
        )

    logger = LossLogger(output_dir / "loss_log.jsonl")
    global_step = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        sampler = make_train_sampler(epoch)
        state = None
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            total, components, state = model.compute_streaming_loss(
                chunk, train_labels, train_masks, state=state, loss_weights=loss_weights
            )
            optimizer.zero_grad()
            total.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            state = _detach_state(state)
            global_step += 1

            if global_step % config.log_every == 0:
                fields = {
                    "step": global_step,
                    "epoch": epoch,
                    "elapsed_s": time.time() - start_time,
                    "split": "train",
                    **{k: v.item() for k, v in components.items()},
                }
                logger.log(**fields)
                summary = " ".join(f"{k}={v:.4f}" for k, v in components.items())
                print(f"[train] step={global_step} epoch={epoch} {summary}")

            if global_step % config.eval_every == 0:
                val = evaluate_streaming(
                    model,
                    make_tuning_sampler,
                    tuning_labels,
                    tuning_masks,
                    device=device,
                    max_chunks=config.eval_max_chunks,
                )
                fields = {
                    "step": global_step,
                    "epoch": epoch,
                    "elapsed_s": time.time() - start_time,
                    "split": "tuning",
                    **val,
                }
                logger.log(**fields)
                summary = " ".join(f"{k}={v:.4f}" for k, v in val.items())
                print(f"[val]   step={global_step} {summary}")

            if global_step % config.checkpoint_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "config": asdict(config),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )

    torch.save(
        {"model": model.state_dict(), "step": global_step, "config": asdict(config)},
        output_dir / "checkpoint_final.pt",
    )
    logger.close()
    elapsed = time.time() - start_time
    print(f"[done] {global_step} steps in {elapsed:.1f}s, output in {output_dir}")
    return output_dir


def _parse_args() -> TrainingConfig:
    """Build a :class:`TrainingConfig` from the required paths plus optional overrides.

    Every other hyperparameter is a plain code-level default on
    :class:`TrainingConfig`; ``--config-json`` accepts a JSON file with
    any subset of field overrides rather than one CLI flag per field --
    a hand-rolled per-field argparse builder from dataclass reflection is
    fragile for ``Optional[int]``-typed fields specifically (their
    ``field.type`` isn't a plain ``int``/``str``/``float`` to dispatch
    an argparse ``type=`` on), so this avoids that entirely.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--tuning-shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to a JSON file with any subset of TrainingConfig field overrides.",
    )
    args = parser.parse_args()

    overrides = (
        json.loads(Path(args.config_json).read_text()) if args.config_json else {}
    )
    return TrainingConfig(
        train_shard_dir=args.train_shard_dir,
        tuning_shard_dir=args.tuning_shard_dir,
        output_dir=args.output_dir,
        **overrides,
    )


if __name__ == "__main__":
    train(_parse_args())
