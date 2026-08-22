"""CPU-testable pieces of the training script.

Helpers only -- the real training loop needs EHRHybridBackbone/CUDA,
see test_train_gpu.py.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import polars as pl
import pytest
import torch

import odyssey.training.train as train_module
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.vocabulary import Vocabulary
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.backbones.hybrid import HybridState
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.data import iter_patient_sequences
from odyssey.training.train import (
    LossLogger,
    PreparedCorpus,
    TrainingConfig,
    _atomic_torch_save,
    _batch_config_fields,
    _combined_val_loss,
    _detach_state,
    _move_chunk_to_device,
    build_model,
    build_objective,
    evaluate_streaming,
    train,
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


def test_detach_state_handles_none_for_stateless_backbones() -> None:
    # TransformerBackbone always returns TimeAwareState(recurrent=None, ...):
    # nothing to detach but prev_time_stamps, itself unused by that
    # backbone -- must not raise, so the training loop can call this
    # unconditionally after every chunk regardless of which backbone
    # config.backbone selected.
    state = TimeAwareState(
        recurrent=None, prev_time_stamps=torch.tensor([1.0], requires_grad=True)
    )

    detached = _detach_state(state)

    assert detached.recurrent is None
    assert not detached.prev_time_stamps.requires_grad


def test_detach_state_handles_hybrid_state() -> None:
    """EHRHybridBackbone's own state shape, the branch _detach_state was built for.

    HybridState itself has no CUDA/mamba-ssm dependency (a plain wrapper
    around per-layer tensor caches) even though the real backbone that
    produces one does -- see test_train_gpu.py for the full backbone.
    """
    cache_0 = (torch.zeros(2, 4, requires_grad=True) + 1.0,)
    cache_1 = (torch.zeros(2, 4, requires_grad=True) + 2.0,)
    state = TimeAwareState(
        recurrent=HybridState({0: cache_0, 1: cache_1}),
        prev_time_stamps=torch.tensor([1.0, 2.0], requires_grad=True),
    )

    detached = _detach_state(state)

    assert isinstance(detached.recurrent, HybridState)
    assert set(detached.recurrent.mamba_states) == {0, 1}
    for cached in detached.recurrent.mamba_states.values():
        for t in cached:
            assert not t.requires_grad
    assert torch.equal(detached.recurrent.mamba_states[0][0], cache_0[0])
    assert not detached.prev_time_stamps.requires_grad


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


def test_build_model_unknown_model_kind_raises() -> None:
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
        model_kind="not-a-real-kind",
    )

    with pytest.raises(ValueError, match="model_kind"):
        build_model(config, vocab_size=50, num_concepts=5)


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
# build_objective: family weights from streamed code_counts, not a held frame
# ---------------------------------------------------------------------------


def _small_vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update(
        {
            "LAB//220045//bpm": 2,
            "DIAGNOSIS//A047": 3,
            "MEDICATION//X": 4,
        }
    )
    return Vocabulary(tokens)


def test_build_objective_uses_code_counts_when_no_events_frame_is_held() -> None:
    """The shard-streaming path: family weights from counts, not a polars frame.

    build_objective's non-streaming caller passes train_events_binned
    instead (already covered indirectly by test_train_gpu.py's full runs)
    -- this is the other branch, exercised directly rather than only via
    a real streaming run.
    """
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        family_balance_alpha=1.0,
    )
    vocab = _small_vocab()
    code_counts = {
        "LAB//220045//bpm": 900,  # one dominant family
        "DIAGNOSIS//A047": 50,
        "MEDICATION//X": 50,
    }

    objective = build_objective(
        config, vocab, train_events_binned=None, device="cpu", code_counts=code_counts
    )

    assert objective.family_weights is not None
    # the rare families get an above-1 weight, the dominant one below --
    # alpha=1 inverse-frequency weighting, not left at the neutral default
    assert not torch.allclose(
        objective.family_weights, torch.ones_like(objective.family_weights)
    )


def test_build_objective_skips_family_weights_when_alpha_is_zero() -> None:
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        family_balance_alpha=0.0,
    )
    objective = build_objective(
        config, _small_vocab(), train_events_binned=None, device="cpu", code_counts={}
    )
    assert objective.family_weights is None


# ---------------------------------------------------------------------------
# evaluate_streaming: max_chunks cutoff and the BaselineSequenceModel dispatch
# ---------------------------------------------------------------------------


def _streaming_events() -> pl.DataFrame:
    codes = ["LAB//220045//bpm", "DIAGNOSIS//A047", "MEDICATION//X"]
    t0 = datetime(2024, 1, 1)
    rows = []
    for sid in (1, 2):
        for i in range(12):
            rows.append((sid, codes[i % 3], t0 + timedelta(hours=i), None, 100 + sid))
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def test_evaluate_streaming_respects_max_chunks() -> None:
    """A max_chunks cap must stop the pass early, not silently score everything."""
    vocab = _small_vocab()
    events = _streaming_events()
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        padding_idx=0,
    )

    def make_sampler() -> PackedLaneSampler:
        return PackedLaneSampler(
            iter_patient_sequences(events, vocab),
            num_lanes=1,
            chunk_size=4,
            reset_prob=0.0,
        )

    full = evaluate_streaming(model, make_sampler, {}, {}, device="cpu")
    capped = evaluate_streaming(model, make_sampler, {}, {}, device="cpu", max_chunks=1)

    # both must produce real (non-empty) loss components; the cap changes
    # which/how many chunks were seen, which is enough to move the mean
    assert full and capped
    assert full != capped


def test_evaluate_streaming_dispatches_on_baseline_model() -> None:
    """A BaselineSequenceModel's compute_streaming_loss takes no concept labels."""
    vocab = _small_vocab()
    events = _streaming_events()
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        padding_idx=0,
    )

    def make_sampler() -> PackedLaneSampler:
        return PackedLaneSampler(
            iter_patient_sequences(events, vocab),
            num_lanes=1,
            chunk_size=8,
            reset_prob=0.0,
        )

    # labels/masks passed as non-empty dicts a bottleneck model would use --
    # a baseline model must never look at them, only its own compute_streaming_loss
    result = evaluate_streaming(
        model,
        make_sampler,
        {1: torch.tensor([1.0])},
        {1: torch.tensor([1.0])},
        device="cpu",
    )
    assert "task_loss" in result


# ---------------------------------------------------------------------------
# train(): the small guards that run before any GPU-needing model
# construction -- reached on real (tiny, synthetic) data via a
# monkeypatched load_meds_shards, stopped before _run_training's own
# GPU-needing body via a monkeypatched _run_training, matching
# run_inference.py's load_run tests' established technique.
# ---------------------------------------------------------------------------


def _tiny_train_events() -> pl.DataFrame:
    codes = ["LAB//220045//bpm", "DIAGNOSIS//A047", "MEDICATION//X"]
    rows = [
        (sid, codes[i % 3], datetime(2024, 1, 1) + timedelta(hours=i), None, 100 + sid)
        for sid in (1, 2)
        for i in range(8)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def test_train_dispatches_to_streaming_when_stream_shards_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stream_shards=True must route to _train_streaming before touching any shards."""
    seen: Dict[str, object] = {}

    def fake_train_streaming(
        config: TrainingConfig,
        output_dir: Path,
        device: str,  # noqa: ARG001
    ) -> Path:
        seen["config"] = config
        seen["output_dir"] = output_dir
        return output_dir / "sentinel"

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not load whole-frame shards on the streaming path")

    monkeypatch.setattr(train_module, "_train_streaming", fake_train_streaming)
    monkeypatch.setattr(train_module, "load_meds_shards", _boom)

    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir=str(tmp_path),
        stream_shards=True,
    )

    result = train(config)

    assert seen["config"] is config
    assert result == tmp_path / "sentinel"


def test_train_rejects_unknown_concept_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train_module, "load_meds_shards", lambda *a, **k: _tiny_train_events()
    )

    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir=str(tmp_path),
        concept_supervision="bogus",
    )

    with pytest.raises(ValueError, match="concept_supervision"):
        train(config)


def test_train_resume_from_infers_heads_from_the_checkpoint_not_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint is the authority on which heads exist, not config.json's fields.

    A resume_from checkpoint with no time_head./event_heads. keys (a run
    predating those features) must flip config.time_to_event/event_hazards
    to False even though TrainingConfig's own dataclass defaults are both
    True -- see build_model's identical reasoning in load_run.
    """
    monkeypatch.setattr(
        train_module, "load_meds_shards", lambda *a, **k: _tiny_train_events()
    )
    seen = {}

    def fake_run_training(
        config: TrainingConfig,
        output_dir: Path,  # noqa: ARG001
        device: str,  # noqa: ARG001
        corpus: PreparedCorpus,  # noqa: ARG001
    ) -> Path:
        seen["time_to_event"] = config.time_to_event
        seen["event_hazards"] = config.event_hazards
        raise RuntimeError("stop here")

    monkeypatch.setattr(train_module, "_run_training", fake_run_training)

    checkpoint_path = tmp_path / "old_checkpoint.pt"
    torch.save(
        {"model": {"backbone.embeddings.weight": torch.zeros(1)}}, checkpoint_path
    )

    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir=str(tmp_path),
        resume_from=str(checkpoint_path),
    )
    assert config.time_to_event is True  # the dataclass default, before resume
    assert config.event_hazards is True

    with pytest.raises(RuntimeError, match="stop here"):
        train(config)

    assert seen["time_to_event"] is False
    assert seen["event_hazards"] is False


def test_train_reaches_run_training_with_stay_supervision_and_history_recap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """concept_supervision="stay" and history_recap=True, together: both real paths.

    The default config exercises concept_supervision="visit" and
    history_recap=False (see the other train() tests here) -- this is
    their counterpart, not covered anywhere else on CPU.
    """
    monkeypatch.setattr(
        train_module, "load_meds_shards", lambda *a, **k: _tiny_train_events()
    )

    def fake_run_training(
        config: TrainingConfig,
        output_dir: Path,  # noqa: ARG001
        device: str,  # noqa: ARG001
        corpus: PreparedCorpus,  # noqa: ARG001
    ) -> Path:
        raise RuntimeError("stop here")

    monkeypatch.setattr(train_module, "_run_training", fake_run_training)

    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir=str(tmp_path),
        concept_supervision="stay",
        history_recap=True,
    )

    with pytest.raises(RuntimeError, match="stop here"):
        train(config)


# ---------------------------------------------------------------------------
# _batch_config_fields: the resume-compatibility check
# ---------------------------------------------------------------------------


def test_batch_config_fields_covers_every_resume_relevant_field() -> None:
    """Pin the exact field set a resume checks for stream-position compatibility.

    Deliberately a literal set, not derived from TrainingConfig's own
    fields (e.g. via dataclasses.fields) -- that would auto-pass no
    matter what changed, defeating the point. A future field that
    changes what next_chunk() produces (another sampler knob, a new
    backbone-selection field, etc.) must be added to
    _batch_config_fields AND to this literal set consciously; this test
    is what forces that second step rather than letting it be forgotten.
    """
    config = TrainingConfig(
        train_shard_dir="/train", tuning_shard_dir="/tuning", output_dir="/out"
    )
    assert set(_batch_config_fields(config)) == {
        "backbone",
        "num_lanes",
        "chunk_size",
        "reset_prob",
        "max_context",
        "seed",
    }


def test_batch_config_fields_reflects_each_fields_actual_value() -> None:
    """Not just the right keys -- the right values, read from the given config."""
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
        num_lanes=17,
        chunk_size=99,
        reset_prob=0.42,
        max_context=1234,
        seed=7,
    )
    assert _batch_config_fields(config) == {
        "backbone": "transformer",
        "num_lanes": 17,
        "chunk_size": 99,
        "reset_prob": 0.42,
        "max_context": 1234,
        "seed": 7,
    }


def test_batch_config_fields_excludes_non_resume_relevant_hyperparameters() -> None:
    """learning_rate (and other model/optimizer knobs) must never appear.

    They don't affect what the data stream produces at a given position,
    so a resume must not treat a pure hyperparameter change as a reason
    to restart an epoch from its own beginning.
    """
    base = TrainingConfig(
        train_shard_dir="/train", tuning_shard_dir="/tuning", output_dir="/out"
    )
    different_lr = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        learning_rate=base.learning_rate * 100,
    )
    assert "learning_rate" not in _batch_config_fields(base)
    assert _batch_config_fields(base) == _batch_config_fields(different_lr)
