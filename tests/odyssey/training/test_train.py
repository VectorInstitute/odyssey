"""CPU-testable pieces of the training script.

Helpers only -- the real training loop needs EHRHybridBackbone/CUDA,
see test_train_gpu.py.
"""

import json
from pathlib import Path

import pytest
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.train import (
    LossLogger,
    TrainingConfig,
    _atomic_torch_save,
    _combined_val_loss,
    _detach_state,
    _move_chunk_to_device,
    _run_training,
    build_model,
)


# ---------------------------------------------------------------------------
# _combined_val_loss
# ---------------------------------------------------------------------------


def test_combined_val_loss_matches_combined_loss_weighting() -> None:
    weights = ConceptBottleneckLossWeights(
        concept=1.0, orthogonality=0.1, observability=0.2
    )
    components = {
        "task_loss": 2.0,
        "concept_loss": 0.5,
        "orthogonality_loss": 0.3,
        "observability_loss": 0.1,
    }

    got = _combined_val_loss(components, weights)

    assert got == pytest.approx(2.0 + 1.0 * 0.5 + 0.1 * 0.3 + 0.2 * 0.1)


def test_combined_val_loss_defaults_missing_auxiliary_terms_to_zero() -> None:
    # A chunk with no patient_end (or a whole eval pass with none) has no
    # concept/orthogonality/observability signal -- matches
    # evaluate_streaming's own dict, which only has whatever keys
    # compute_streaming_loss actually returned.
    weights = ConceptBottleneckLossWeights(concept=1.0, orthogonality=1.0)
    got = _combined_val_loss({"task_loss": 3.0}, weights)
    assert got == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# LossLogger
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _atomic_torch_save
# ---------------------------------------------------------------------------


def test_atomic_torch_save_writes_a_loadable_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save({"step": 42}, path)

    assert path.exists()
    assert torch.load(path, map_location="cpu") == {"step": 42}


def test_atomic_torch_save_leaves_no_tmp_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save({"step": 1}, path)

    assert list(tmp_path.iterdir()) == [path]


def test_atomic_torch_save_never_leaves_a_partial_file_at_the_target_path(
    tmp_path: Path,
) -> None:
    # The failure mode this exists to prevent: a reader must never see a
    # truncated file at the real path, even if the process died mid-write
    # -- confirmed for real during this project's own training run (a
    # crash mid checkpoint-write left a 0-byte checkpoint_N.pt, which
    # every subsequent resume attempt then failed to load, unable to
    # self-recover). Simulates that by making torch.save itself raise
    # partway through -- the target path must not exist, or if it
    # existed before, must still hold its old, complete contents.
    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save({"step": 1}, path)  # an existing, good checkpoint

    with pytest.MonkeyPatch.context() as mp:

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated crash mid-write")

        mp.setattr(torch, "save", _boom)
        with pytest.raises(RuntimeError):
            _atomic_torch_save({"step": 2}, path)

    # the old, good checkpoint is untouched -- never overwritten with a
    # partial file, and no leftover .tmp file either.
    assert torch.load(path, map_location="cpu") == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_loss_logger_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path)
    logger.log(step=1, task_loss=0.5)
    logger.log(step=2, task_loss=0.3)
    logger.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 1, "task_loss": 0.5}
    assert json.loads(lines[1]) == {"step": 2, "task_loss": 0.3}


def test_loss_logger_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "loss_log.jsonl"
    LossLogger(path).log(step=1)
    logger2 = LossLogger(path)
    logger2.log(step=2)
    logger2.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# _move_chunk_to_device
# ---------------------------------------------------------------------------


def test_move_chunk_to_device_preserves_structure_and_values() -> None:
    batch = ClinicalSequenceBatch(
        concept_ids=torch.tensor([[1, 2]]),
        aux=AuxiliaryInputs(
            type_ids=torch.tensor([[0, 1]]),
            time_stamps=torch.tensor([[0.0, 1.0]]),
            ages=torch.tensor([[30.0, 30.0]]),
            visit_orders=torch.tensor([[0, 0]]),
            visit_segments=torch.tensor([[0, 0]]),
        ),
    )

    moved = _move_chunk_to_device(batch, "cpu")

    assert isinstance(moved, ClinicalSequenceBatch)
    assert isinstance(moved.aux, AuxiliaryInputs)
    assert torch.equal(moved.concept_ids, batch.concept_ids)
    assert torch.equal(moved.aux.time_stamps, batch.aux.time_stamps)


def test_move_chunk_to_device_passes_through_non_tensor_values() -> None:
    assert _move_chunk_to_device(5, "cpu") == 5
    assert _move_chunk_to_device("x", "cpu") == "x"
    assert _move_chunk_to_device(None, "cpu") is None


# ---------------------------------------------------------------------------
# _detach_state
# ---------------------------------------------------------------------------


def test_detach_state_handles_tuple_of_tensors_backbones() -> None:
    # TinyGRUBackbone (and any lighter backbone) carries a plain tuple of
    # tensors -- the streaming loops must be able to truncate BPTT for it,
    # not only for EHRHybridBackbone's HybridState.
    h = torch.zeros(2, 4, requires_grad=True) + 1.0  # a non-leaf, grad-carrying tensor
    state = TimeAwareState(
        recurrent=(h,), prev_time_stamps=torch.tensor([1.0, 2.0], requires_grad=True)
    )

    detached = _detach_state(state)

    assert isinstance(detached.recurrent, tuple)
    assert not detached.recurrent[0].requires_grad
    assert not detached.prev_time_stamps.requires_grad
    assert torch.equal(detached.recurrent[0], h)


def test_detach_state_rejects_unknown_state_shapes() -> None:
    state = TimeAwareState(recurrent=object(), prev_time_stamps=torch.tensor([0.0]))
    try:
        _detach_state(state)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_requires_only_the_three_paths() -> None:
    config = TrainingConfig(
        train_shard_dir="/train", tuning_shard_dir="/tuning", output_dir="/out"
    )
    assert config.hidden_size == 256
    assert config.num_epochs == 3
    assert config.max_train_shards is None


def test_training_config_backbone_defaults_to_hybrid() -> None:
    config = TrainingConfig(
        train_shard_dir="/train", tuning_shard_dir="/tuning", output_dir="/out"
    )
    assert config.backbone == "hybrid"
    assert config.max_context == 4096


def test_training_config_rejects_checkpoint_every_below_one() -> None:
    # Real bug this guards against: checkpoint_every=0 used to raise
    # ZeroDivisionError from `global_step % config.checkpoint_every` the
    # first time a checkpoint was due -- hours into a real run, not at
    # config-load time. Validate at construction instead.
    with pytest.raises(ValueError, match="checkpoint_every"):
        TrainingConfig(
            train_shard_dir="/train",
            tuning_shard_dir="/tuning",
            output_dir="/out",
            checkpoint_every=0,
        )


# ---------------------------------------------------------------------------
# build_model: backbone="transformer" (backbone="hybrid" needs CUDA/mamba-ssm,
# see test_train_gpu.py)
# ---------------------------------------------------------------------------


def test_build_model_transformer_backbone_bottleneck() -> None:
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
        hidden_size=32,
        num_hidden_layers=2,
        attn_num_heads=4,
    )

    model = build_model(config, vocab_size=50, num_concepts=5)

    assert isinstance(model, ConceptBottleneckSequenceModel)
    backbone = model.backbone
    assert isinstance(backbone, TransformerBackbone)
    assert backbone.hidden_size == 32
    assert len(backbone.layers) == 2


def test_build_model_transformer_backbone_baseline() -> None:
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
        model_kind="baseline",
        hidden_size=16,
        num_hidden_layers=1,
        attn_num_heads=4,
    )

    model = build_model(config, vocab_size=50, num_concepts=5)

    assert isinstance(model, BaselineSequenceModel)
    assert isinstance(model.backbone, TransformerBackbone)


def test_build_model_unknown_backbone_raises() -> None:
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="not-a-real-backbone",
    )

    with pytest.raises(ValueError, match="backbone"):
        build_model(config, vocab_size=50, num_concepts=5)


def test_build_model_hybrid_and_transformer_are_directly_comparable_by_param_count() -> (
    None
):
    """Config plumbing's whole point: pick depth/width to match the hybrid's budget.

    Only the transformer side is buildable on this CPU-only host (the
    hybrid needs CUDA/mamba-ssm, see test_train_gpu.py), but build_model's
    signature and return type are identical either way, so the comparison
    this is meant to enable is just calling build_model twice and diffing
    parameter counts -- asserted here for the transformer side alone as a
    regression check that this path keeps producing a countable model.
    """
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
        hidden_size=32,
        num_hidden_layers=2,
        attn_num_heads=4,
    )

    model = build_model(config, vocab_size=50, num_concepts=5)
    n_params = sum(p.numel() for p in model.backbone.parameters())

    assert n_params > 0


# ---------------------------------------------------------------------------
# _run_training: backbone="transformer" is not yet wired into this loop
# ---------------------------------------------------------------------------


def test_run_training_rejects_transformer_backbone_before_touching_the_corpus() -> None:
    """Confirm the guard fires before touching any other (invalid) argument.

    That is the whole point of raising as the very first statement in
    _run_training, not deep inside the loop.
    """
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
    )

    with pytest.raises(NotImplementedError, match="PackedContextSampler"):
        _run_training(config, output_dir=None, device=None, corpus=None)  # type: ignore[arg-type]
