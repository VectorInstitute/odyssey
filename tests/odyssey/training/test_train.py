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
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.training.train import (
    LossLogger,
    TrainingConfig,
    _atomic_torch_save,
    _combined_val_loss,
    _detach_state,
    _move_chunk_to_device,
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
