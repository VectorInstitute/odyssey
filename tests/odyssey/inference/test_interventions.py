"""Tests for the concept intervention / completeness harness (CPU)."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

import odyssey.inference.interventions as interventions_module
from odyssey.data.concepts import concepts_for_source
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.interventions import (
    INTERVENTION_MODES,
    _chunk_intervention,
    evaluate_interventions,
    run_streaming_intervention,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.concept_bottleneck import BottleneckIntervention
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.data import (
    build_visit_concept_first_times,
    iter_patient_sequences,
)
from odyssey.training.running_labels import position_running_labels
from odyssey.training.train import TrainingConfig


T0 = datetime(2024, 1, 1)
NUM_CONCEPTS = 3
CODES = [f"LAB//{i}//" for i in range(10)]


def _events() -> pl.DataFrame:
    rows = []
    for sid in (1, 2, 3):
        for i in range(20):
            rows.append(
                (
                    sid,
                    CODES[(sid * 3 + i) % len(CODES)],
                    T0 + timedelta(hours=i),
                    None,
                    100 + sid,
                )
            )
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


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({c: i + 2 for i, c in enumerate(CODES)})
    return Vocabulary(tokens)


def _model(vocab_size: int) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        padding_idx=0,
    )


def _labels_and_masks() -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    labels = {
        1: torch.tensor([1.0, 0.0, 1.0]),
        2: torch.tensor([0.0, 0.0, 0.0]),
        3: torch.tensor([1.0, 1.0, 0.0]),
    }
    masks = {sid: torch.ones(NUM_CONCEPTS) for sid in labels}
    return labels, masks


def _run(mode: str, seed: int = 0):
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    gammas = torch.full((NUM_CONCEPTS,), 0.25) if mode.endswith("_calibrated") else None
    return run_streaming_intervention(
        _model(len(vocab)),
        _events(),
        vocab,
        labels,
        masks,
        mode=mode,
        supervision="stay",
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        seed=seed,
        calibration_gammas=gammas,
    )


def test_every_mode_runs_and_scores_every_real_position() -> None:
    for mode in INTERVENTION_MODES:
        result = _run(mode)
        # 3 subjects x 20 events, minus one position per subject (the
        # final event has no next-token target).
        assert result.n_predictions == 3 * 19, mode
        assert 0.0 <= result.top1_accuracy <= 1.0, mode
        assert result.mean_task_loss > 0.0, mode


def test_probs_modes_move_the_logits_and_count_intervened_positions() -> None:
    baseline = _run("none")
    truth = _run("truth")
    flip = _run("flip")
    # Every position belongs to a fully-observed subject, so every
    # input position is intervened in truth/flip modes.
    assert truth.n_intervened_positions == 3 * 20
    assert baseline.n_intervened_positions == 0
    # An untrained model has no meaningful direction, but forcing
    # extreme (0/1) mixing probabilities must move the task loss.
    assert truth.mean_task_loss != baseline.mean_task_loss
    assert flip.mean_task_loss != truth.mean_task_loss


def test_flip_gated_reduces_to_none_when_nothing_is_observed() -> None:
    """With no observed concepts the gate has nothing to gate.

    This must hold EXACTLY, which also proves the mode's two forward
    passes thread the backbone state correctly: both passes start from
    the same input state, and the carried state matches the one the
    single-forward modes produce.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    masks = {sid: torch.zeros(NUM_CONCEPTS) for sid in masks}
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
    }
    baseline = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="none", **kwargs
    )
    gated = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="flip_gated", **kwargs
    )
    assert gated.n_intervened_positions == 0
    assert gated.mean_task_loss == baseline.mean_task_loss
    assert gated.top1_accuracy == baseline.top1_accuracy


def test_flip_gated_intervenes_like_flip_but_scores_differently() -> None:
    """The gated variant edits the same entries yet keeps only suppression."""
    flip = _run("flip")
    gated = _run("flip_gated")
    baseline = _run("none")
    assert gated.n_intervened_positions == flip.n_intervened_positions == 3 * 20
    assert gated.mean_abs_displacement == flip.mean_abs_displacement
    # An untrained model still moves under a hard 0/1 edit; the gate must
    # produce a third, distinct scoring (it discards flip's promotions).
    assert gated.mean_task_loss != baseline.mean_task_loss
    assert gated.mean_task_loss != flip.mean_task_loss


def test_zero_modes_change_task_loss() -> None:
    baseline = _run("none")
    assert _run("zero_known").mean_task_loss != baseline.mean_task_loss
    assert _run("zero_unknown").mean_task_loss != baseline.mean_task_loss


def test_random_mode_is_deterministic_per_seed() -> None:
    a = _run("random", seed=7)
    b = _run("random", seed=7)
    assert a.mean_task_loss == b.mean_task_loss
    assert a.top1_accuracy == b.top1_accuracy


def test_unobserved_concepts_are_never_intervened() -> None:
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    # Nothing observed anywhere: truth mode must reduce to the baseline.
    masks = {sid: torch.zeros(NUM_CONCEPTS) for sid in masks}
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
    }
    baseline = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="none", **kwargs
    )
    truth = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="truth", **kwargs
    )
    assert truth.n_intervened_positions == 0
    assert truth.mean_task_loss == baseline.mean_task_loss
    assert truth.top1_accuracy == baseline.top1_accuracy


def test_running_labels_are_false_before_first_trigger() -> None:
    """A concept is injected as 1 only from its first-trigger time on."""
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    # Subject 1's concept 0 first triggers at hour 10 (events are hourly),
    # concept 2 at hour 0; concept 1 never (label 0, inf).
    first = {
        1: torch.tensor([10.0, float("inf"), 0.0]),
        2: torch.tensor([float("inf")] * 3),
        3: torch.tensor([0.0, 5.0, float("inf")]),
    }
    seqs = iter_patient_sequences(_events(), vocab)
    sampler = PackedLaneSampler(seqs, num_lanes=1, chunk_size=64, reset_prob=0.0)
    chunk = next(iter(sampler))
    pos_labels, observed = position_running_labels(
        chunk, labels, masks, first, supervision="stay", num_concepts=NUM_CONCEPTS
    )
    sid = chunk.subject_ids[0]
    times = chunk.batch.aux.time_stamps[0]
    s1 = sid == 1
    # concept 0 for subject 1: 0 before hour 10, 1 from hour 10 on
    assert torch.equal(pos_labels[0, s1, 0], (times[s1] >= 10.0).float())
    # concept 2 for subject 1: 1 everywhere (triggered at hour 0)
    assert pos_labels[0, s1, 2].eq(1.0).all()
    # concept 1 for subject 1: label 0 -> 0 everywhere
    assert pos_labels[0, s1, 1].eq(0.0).all()
    assert observed[0, s1].eq(1.0).all()


def test_first_time_builders_align_with_sequence_time_origin() -> None:
    concepts = [c for c in concepts_for_source("mimic_iv") if c.name == "tachycardia"]
    events = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "code": ["LAB//50912//x", "LAB//220045//bpm", "LAB//220045//bpm"],
            "numeric_value": [1.0, 80.0, 130.0],
            "time": [T0, T0 + timedelta(hours=2), T0 + timedelta(hours=7)],
            "hadm_id": [10, 10, 10],
        }
    )
    first = build_visit_concept_first_times(events, concepts)
    # 7 hours after the subject's first event (the creatinine at T0).
    assert first[(1, 10)].tolist() == [7.0]


def test_uncertain_band_limits_replacement_and_reports_displacement() -> None:
    """A band replaces only entries near p=0.5 and reports their displacement."""
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
    }
    full = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="truth", **kwargs
    )
    banded = run_streaming_intervention(
        model,
        _events(),
        vocab,
        labels,
        masks,
        mode="truth",
        uncertain_band=0.05,
        **kwargs,
    )
    assert full.mean_abs_displacement is not None
    assert 0.0 < full.mean_abs_displacement <= 1.0
    assert banded.uncertain_band == 0.05
    assert banded.n_intervened_positions <= full.n_intervened_positions
    # Inside a +/-0.05 band around 0.5 no displacement can exceed 0.55.
    if banded.mean_abs_displacement is not None:
        assert banded.mean_abs_displacement <= 0.55


def test_run_streaming_intervention_rejects_an_unknown_mode() -> None:
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    with pytest.raises(ValueError, match="unknown intervention mode 'bogus'"):
        run_streaming_intervention(
            _model(len(vocab)),
            _events(),
            vocab,
            labels,
            masks,
            mode="bogus",
            supervision="stay",
            num_lanes=2,
            chunk_size=8,
            device="cpu",
        )


def test_zero_residual_is_exposed_and_maps_to_the_residual_flag() -> None:
    """The residual mode must be reachable from the CLI, not just the model.

    BottleneckIntervention grew ``zero_residual`` with the decomposed
    bottleneck, but INTERVENTION_MODES was not extended, so the mode the
    decomposition exists to measure -- whether the residual is where the
    predictive capacity actually went -- could not be requested from
    ``python -m odyssey.inference.interventions`` at all.
    """
    assert "zero_residual" in INTERVENTION_MODES
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    seqs = iter_patient_sequences(_events(), vocab)
    sampler = PackedLaneSampler(seqs, num_lanes=1, chunk_size=8, reset_prob=0.0)
    chunk = next(iter(sampler))
    built = _chunk_intervention(
        chunk,
        "zero_residual",
        labels,
        masks,
        {},
        supervision="stay",
        num_concepts=NUM_CONCEPTS,
        device="cpu",
        rng=torch.Generator(),
    )
    assert built is not None
    assert built.zero_residual is True
    # The zero_* modes are structural: they must not also inject values.
    assert built.zero_known is False and built.zero_unknown is False


def test_chunk_intervention_rejects_an_unknown_mode_directly() -> None:
    """The inner per-chunk builder's own guard, a second line of defense.

    run_streaming_intervention already rejects a bad mode before ever
    reaching this function; calling it directly proves this guard would
    still fire on its own if that outer check were ever bypassed or
    reordered.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    seqs = iter_patient_sequences(_events(), vocab)
    sampler = PackedLaneSampler(seqs, num_lanes=1, chunk_size=8, reset_prob=0.0)
    chunk = next(iter(sampler))
    with pytest.raises(ValueError, match="unknown intervention mode: 'bogus'"):
        _chunk_intervention(
            chunk,
            "bogus",
            labels,
            masks,
            {},
            supervision="stay",
            num_concepts=NUM_CONCEPTS,
            device="cpu",
            rng=torch.Generator(),
        )


def test_run_streaming_intervention_skips_a_chunk_with_no_real_positions() -> None:
    """A trailing chunk after the only subject finishes (all lanes NO_SUBJECT).

    More lanes than subjects, a chunk_size that leaves an all-padding
    chunk after the one subject's short sequence -- confirmed directly
    (chunk.real_mask.any() is False for the second chunk here) rather
    than assumed from the sampler's docs.
    """
    codes = CODES
    rows = [
        (1, codes[i % len(codes)], T0 + timedelta(hours=i), None, 101) for i in range(3)
    ]
    events = pl.DataFrame(
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
    vocab = _vocab()
    labels = {1: torch.tensor([1.0, 0.0, 1.0])}
    masks = {1: torch.ones(NUM_CONCEPTS)}
    model = _model(len(vocab))

    result = run_streaming_intervention(
        model,
        events,
        vocab,
        labels,
        masks,
        mode="none",
        supervision="stay",
        num_lanes=2,  # more lanes than the one subject
        chunk_size=2,  # leaves a fully-empty trailing chunk
        device="cpu",
    )

    assert result.n_predictions == 2  # 3 events, minus the last (no next target)


def test_run_streaming_intervention_handles_an_unmasked_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """intervention_apply_mask returning None ("everywhere") must not crash.

    _chunk_intervention always sets probs_mask, so this path is never
    reached through the normal mode-building code -- forced directly by
    replacing _chunk_intervention's output for one call, to prove the
    fallback (treat None as "applied everywhere") is handled correctly
    rather than left an untested assumption.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))

    def fake_chunk_intervention(*_args: object, **_kwargs: object):
        return BottleneckIntervention(probs=torch.full((NUM_CONCEPTS,), 0.5))

    monkeypatch.setattr(
        interventions_module, "_chunk_intervention", fake_chunk_intervention
    )

    result = run_streaming_intervention(
        model,
        _events(),
        vocab,
        labels,
        masks,
        mode="truth",
        supervision="stay",
        num_lanes=2,
        chunk_size=8,
        device="cpu",
    )

    # every real position counted as intervened, since apply=None means
    # "everywhere" rather than "nowhere"
    assert result.n_intervened_positions == result.n_predictions + 3  # +3 final events


# ---------------------------------------------------------------------------
# evaluate_interventions: backbone="transformer" gate
# ---------------------------------------------------------------------------


def test_evaluate_interventions_rejects_a_non_bottleneck_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = BaselineSequenceModel(
        TinyGRUBackbone(vocab_size=10, hidden_size=4), vocab_size=10
    )
    fake_config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        model_kind="baseline",
    )
    monkeypatch.setattr(
        interventions_module,
        "load_run",
        lambda *a, **k: (fake_model, object(), object(), fake_config),
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not read shards before the model_kind gate fires")

    monkeypatch.setattr(interventions_module, "load_meds_shards", _boom)

    with pytest.raises(ValueError, match="needs a concept bottleneck"):
        evaluate_interventions("/runs/x", "/data/held_out")


def test_evaluate_interventions_rejects_transformer_backbone_before_touching_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = ConceptBottleneckSequenceModel(
        TinyGRUBackbone(vocab_size=10, hidden_size=4),
        vocab_size=10,
        num_concepts=2,
        embedding_dim=4,
    )
    fake_config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        backbone="transformer",
    )
    monkeypatch.setattr(
        interventions_module,
        "load_run",
        lambda *a, **k: (fake_model, object(), object(), fake_config),
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not read shards before the backbone gate fires")

    monkeypatch.setattr(interventions_module, "load_meds_shards", _boom)

    with pytest.raises(NotImplementedError, match="backbone='transformer'"):
        evaluate_interventions("/runs/x", "/data/held_out")


def test_main_refuses_to_overwrite_an_existing_output_before_evaluating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guard must fire before evaluate_interventions runs, not just before the write.

    Real incident: a silent overwrite lost an irreplaceable row-level
    science output (2026-08-22, alerts.py's own dump-rows --
    interventions.py shares the same --output-json shape).
    """
    existing = tmp_path / "interventions_band15.json"
    existing.write_text("[]")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            str(existing),
        ],
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not evaluate before the overwrite guard fires")

    monkeypatch.setattr(interventions_module, "evaluate_interventions", _boom)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        interventions_module._main()


def test_per_subject_outcomes_sum_to_the_aggregate() -> None:
    """--dump-per-subject's accumulator must partition the totals exactly.

    The paired truth-vs-flip CI (scripts/intervention_cis.py) is only
    valid if per-subject hits/counts sum to the aggregate the mode
    reports and every subject contributes its full prediction count.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    per_subject: dict[int, list[int]] = {}
    result = run_streaming_intervention(
        _model(len(vocab)),
        _events(),
        vocab,
        labels,
        masks,
        mode="truth",
        supervision="stay",
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        seed=0,
        per_subject_out=per_subject,
    )
    assert set(per_subject) == {1, 2, 3}
    assert all(n == 19 for _, n in per_subject.values())
    assert sum(n for _, n in per_subject.values()) == result.n_predictions
    total_hits = sum(h for h, _ in per_subject.values())
    assert total_hits / result.n_predictions == result.top1_accuracy


# ---------------------------------------------------------------------------
# Output-calibrated modes (W7)
# ---------------------------------------------------------------------------


def test_calibrated_mode_requires_gammas() -> None:
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    with pytest.raises(ValueError, match="needs calibration_gammas"):
        run_streaming_intervention(
            _model(len(vocab)),
            _events(),
            vocab,
            labels,
            masks,
            mode="truth_calibrated",
            supervision="stay",
            num_lanes=2,
            chunk_size=8,
            device="cpu",
        )


def test_calibrated_modes_displace_by_at_most_gamma() -> None:
    """A calibrated edit displaces by gamma_i, less only at the [0, 1] clip.

    So the mean displacement can never exceed the largest gamma.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))
    gammas = torch.tensor([0.05, 0.10, 0.20])
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
        "calibration_gammas": gammas,
        "calibrated_tau": 1.0,
    }
    baseline = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="none", **kwargs
    )
    for mode in ("truth_calibrated", "flip_calibrated"):
        result = run_streaming_intervention(
            model, _events(), vocab, labels, masks, mode=mode, **kwargs
        )
        assert result.n_intervened_positions == 3 * 20, mode
        assert result.mean_abs_displacement is not None
        assert 0.0 < result.mean_abs_displacement <= float(gammas.max()) + 1e-6
        assert result.calibrated_tau == 1.0
        # A calibrated mode never uses (or reports) the uncertain band.
        assert result.uncertain_band is None
        assert result.mean_task_loss != baseline.mean_task_loss, mode


def test_calibrated_modes_accept_gammas_in_another_dtype() -> None:
    """Gammas are cast to the running labels' dtype, not assumed to match.

    The production gammas come from :func:`calibrated_gammas` over the LM
    head, so their dtype follows the model's parameters and need not equal
    the labels' float32. This pins the dtype half of that conversion; the
    device half cannot be exercised without a second device, which is what
    ``test_calibrated_modes_run_on_cuda`` below is for.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
        "calibrated_tau": 1.0,
    }
    for dtype in (torch.float64, torch.float16):
        result = run_streaming_intervention(
            model,
            _events(),
            vocab,
            labels,
            masks,
            mode="truth_calibrated",
            calibration_gammas=torch.full((NUM_CONCEPTS,), 0.25, dtype=dtype),
            **kwargs,
        )
        assert result.mean_abs_displacement is not None
        assert 0.0 < result.mean_abs_displacement <= 0.25 + 1e-3, dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_calibrated_modes_run_on_cuda() -> None:
    """Regression: calibrated modes died on every GPU run until 2026-08-31.

    ``calibration_gammas`` is built from the model's LM-head weights and so
    lives on the model's device, while the running labels are assembled on
    CPU. The offsets multiply cast dtype but not device, which raises
    "Expected all tensors to be on the same device" -- but only where those
    devices actually differ. The rest of this module is CPU-only, where
    they never do, so the whole calibrated-mode suite passed while no
    calibrated mode had ever completed a real run. This is gated rather
    than skipped-by-omission because the VMs run this suite on a GPU, which
    is where the bug was reachable.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab)).cuda()
    for mode in ("truth_calibrated", "flip_calibrated"):
        result = run_streaming_intervention(
            model,
            _events(),
            vocab,
            labels,
            masks,
            mode=mode,
            supervision="stay",
            num_lanes=2,
            chunk_size=8,
            device="cuda",
            seed=0,
            # deliberately on the model's device, as calibrated_gammas returns it
            calibration_gammas=torch.full((NUM_CONCEPTS,), 0.25).cuda(),
            calibrated_tau=1.0,
        )
        assert result.mean_abs_displacement is not None
        assert 0.0 < result.mean_abs_displacement <= 0.25 + 1e-6, mode


def test_calibrated_truth_and_flip_push_in_opposite_directions() -> None:
    """With no clipping possible, the two modes' displacements agree.

    Their scored logits still differ: they move the same entries in
    opposite directions.
    """
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))
    gammas = torch.full((NUM_CONCEPTS,), 0.01)  # tiny: clipping ~impossible
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
        "calibration_gammas": gammas,
    }
    truth = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="truth_calibrated", **kwargs
    )
    flip = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="flip_calibrated", **kwargs
    )
    assert truth.mean_abs_displacement == pytest.approx(0.01, abs=1e-6)
    assert flip.mean_abs_displacement == pytest.approx(0.01, abs=1e-6)
    assert truth.mean_task_loss != flip.mean_task_loss


def test_per_concept_coverage_partitions_the_replacement_counts() -> None:
    """W3 band coverage: per-concept counts must partition the totals.

    With every subject fully observed and no band, every concept is
    replaced at every input position; the per-concept mean displacements
    must average (count-weighted) to the aggregate displacement.
    """
    truth = _run("truth")
    assert truth.n_replaced_by_concept is not None
    assert all(n == 3 * 20 for n in truth.n_replaced_by_concept.values())
    assert len(truth.n_replaced_by_concept) == NUM_CONCEPTS
    assert truth.mean_abs_displacement_by_concept is not None
    total = sum(truth.n_replaced_by_concept.values())
    weighted = (
        sum(
            truth.mean_abs_displacement_by_concept[c] * truth.n_replaced_by_concept[c]
            for c in truth.n_replaced_by_concept
        )
        / total
    )
    # The aggregate accumulates in float32, the per-concept path in
    # float64, so agreement is to float32 precision, not exact.
    assert weighted == pytest.approx(truth.mean_abs_displacement, abs=1e-5)
    # Default names when the caller passes none.
    assert set(truth.n_replaced_by_concept) == {
        f"concept_{i}" for i in range(NUM_CONCEPTS)
    }
    # Modes that replace nothing report None, not empty dicts.
    baseline = _run("none")
    assert baseline.n_replaced_by_concept is None
    assert baseline.mean_abs_displacement_by_concept is None


def test_per_concept_coverage_shrinks_under_a_band() -> None:
    vocab = _vocab()
    labels, masks = _labels_and_masks()
    model = _model(len(vocab))
    kwargs = {
        "supervision": "stay",
        "num_lanes": 2,
        "chunk_size": 8,
        "device": "cpu",
        "seed": 0,
    }
    full = run_streaming_intervention(
        model, _events(), vocab, labels, masks, mode="truth", **kwargs
    )
    banded = run_streaming_intervention(
        model,
        _events(),
        vocab,
        labels,
        masks,
        mode="truth",
        uncertain_band=0.05,
        **kwargs,
    )
    assert full.n_replaced_by_concept is not None
    if banded.n_replaced_by_concept is None:
        return  # nothing entered the band at all: a legal, smaller outcome
    for name, n_banded in banded.n_replaced_by_concept.items():
        assert n_banded <= full.n_replaced_by_concept[name]
