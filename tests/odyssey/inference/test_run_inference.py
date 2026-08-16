"""CPU-testable pieces of the inference script.

The real streaming inference path needs EHRHybridBackbone/CUDA, see
test_run_inference_gpu.py.
"""

import json
from pathlib import Path

import pytest
import torch

from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.run_inference import (
    InferenceResults,
    _block_set_hits,
    _latest_checkpoint,
    _parse_args,
    _RunningTaskMetrics,
    results_to_dict,
)
from odyssey.training.metrics import (
    TaskMetrics,
    compute_task_metrics,
    compute_task_metrics_by_code_type,
)


def _vocab() -> Vocabulary:
    return Vocabulary.build(
        [
            "DIAGNOSIS//A",
            "MEDICATION//B",
            "LAB//220045//bpm",
            "PROCEDURE//C",
            "HOSPITAL_ADMISSION//D",
        ],
        min_count=1,
    )


def test_running_task_metrics_matches_computing_over_the_whole_tensor_at_once() -> None:
    # _RunningTaskMetrics exists specifically to avoid materializing the
    # full (N, vocab_size) logits tensor a real held-out pass would need
    # (see its docstring: an earlier version of this OOM-killed the
    # actual training job it ran alongside) -- this proves the
    # incremental, chunk-by-chunk path gives bit-for-bit the same
    # answer as the original compute-everything-at-once functions.
    vocab = _vocab()
    torch.manual_seed(0)
    n = 37
    logits = torch.randn(n, len(vocab))
    targets = torch.randint(2, len(vocab), (n,))  # never PAD(0)/UNK(1)

    expected_overall = compute_task_metrics(logits, targets, ignore_index=0)
    expected_by_type = compute_task_metrics_by_code_type(
        logits, targets, vocab, ignore_index=0
    )

    stats = _RunningTaskMetrics(vocab, device="cpu")
    chunk_sizes = [5, 12, 1, 19]
    assert sum(chunk_sizes) == n
    i = 0
    for size in chunk_sizes:
        stats.update(logits[i : i + size], targets[i : i + size])
        i += size
    got_overall, got_by_type = stats.finalize()

    assert got_overall.cross_entropy == pytest.approx(expected_overall.cross_entropy)
    assert got_overall.perplexity == pytest.approx(expected_overall.perplexity)
    assert got_overall.top1_accuracy == pytest.approx(expected_overall.top1_accuracy)
    assert got_overall.top5_accuracy == pytest.approx(expected_overall.top5_accuracy)
    assert got_overall.n_predictions == expected_overall.n_predictions

    assert set(got_by_type) == set(expected_by_type)
    for name in expected_by_type:
        assert got_by_type[name].cross_entropy == pytest.approx(
            expected_by_type[name].cross_entropy
        )
        assert got_by_type[name].n_predictions == expected_by_type[name].n_predictions


def test_running_task_metrics_handles_a_chunk_with_no_real_targets() -> None:
    vocab = _vocab()
    stats = _RunningTaskMetrics(vocab, device="cpu")
    stats.update(torch.empty(0, len(vocab)), torch.empty(0, dtype=torch.long))
    stats.update(torch.randn(3, len(vocab)), torch.randint(2, len(vocab), (3,)))
    overall, _ = stats.finalize()
    assert overall.n_predictions == 3


def test_latest_checkpoint_prefers_final(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_500.pt").touch()
    (tmp_path / "checkpoint_final.pt").touch()

    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint_final.pt"


def test_latest_checkpoint_picks_highest_step_when_no_final(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_500.pt").touch()
    (tmp_path / "checkpoint_2000.pt").touch()
    (tmp_path / "checkpoint_1000.pt").touch()

    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint_2000.pt"


def test_latest_checkpoint_raises_when_none_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _latest_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# results_to_dict / _parse_args
# ---------------------------------------------------------------------------


def test_results_to_dict_is_plain_json_serializable() -> None:
    results = InferenceResults(
        task_metrics=TaskMetrics(
            cross_entropy=1.0,
            perplexity=2.7,
            top1_accuracy=0.5,
            top5_accuracy=0.8,
            n_predictions=100,
        ),
        task_metrics_by_code_type={},
        concept_metrics=[],
        observability_metrics=[],
        orthogonality=0.1,
        n_patient_ends_scored=10,
    )

    got = json.loads(json.dumps(results_to_dict(results)))

    assert got["n_patient_ends_scored"] == 10
    assert got["task_metrics"]["cross_entropy"] == 1.0


def test_parse_args_defaults_checkpoint_to_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            "/out/results.json",
        ],
    )
    args = _parse_args()
    assert args.checkpoint_path == Path("/runs/x/checkpoint_best.pt")
    assert args.num_lanes == 8
    assert args.max_shards is None


def test_parse_args_honours_an_explicit_checkpoint_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-dir",
            "/runs/x",
            "--held-out-shard-dir",
            "/data/held_out",
            "--output-json",
            "/out/results.json",
            "--checkpoint",
            "checkpoint_final.pt",
        ],
    )
    args = _parse_args()
    assert args.checkpoint_path == Path("/runs/x/checkpoint_final.pt")


# ---------------------------------------------------------------------------
# _block_set_hits: set-based scoring over same-timestamp event blocks
# ---------------------------------------------------------------------------


def test_block_set_hits_counts_within_block_predictions() -> None:

    # One lane, 5 targets. Input times (positions 0..5): targets 0-2 share
    # time 1.0 (one block: tokens 11,12,13), targets 3-4 share time 2.0
    # (block: tokens 14,15).
    targets = torch.tensor([[11, 12, 13, 14, 15]])
    # times for target j = input time at j+1
    times_full = torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0, 2.0]])
    subject_ids = torch.ones(1, 6, dtype=torch.long)
    real = torch.ones(1, 5, dtype=torch.bool)

    # predictions: 13 (in block 1), 11 (in block 1), 99 (miss), 14
    # (block 2; note token 15 sits at the final position, whose target
    # time is outside the chunk, so it is invisible to block membership:
    # the documented boundary approximation)
    top1 = torch.tensor([[13, 11, 99, 14, 14]])
    set_valid, set_hit = _block_set_hits(
        top1,
        targets,
        times=times_full[:, :5],
        subject_ids=subject_ids[:, :5],
        real_mask=real,
        vocab_size=100,
    )
    # the final position has no in-chunk target time: excluded
    assert set_valid[0].tolist() == [True, True, True, True, False]
    assert set_hit[0].tolist() == [True, True, False, True, False]


def test_block_set_hits_blocks_never_span_subjects() -> None:

    # Two patients back to back, same timestamps: token 20 belongs to
    # subject 2's block, so subject 1's prediction of 20 must not count.
    targets = torch.tensor([[10, 20, 21]])
    times_full = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    subject_ids = torch.tensor([[1, 1, 2, 2]])
    real = torch.ones(1, 3, dtype=torch.bool)
    top1 = torch.tensor([[20, 10, 21]])
    _, set_hit = _block_set_hits(
        top1,
        targets,
        times=times_full[:, :3],
        subject_ids=subject_ids[:, :3],
        real_mask=real,
        vocab_size=100,
    )
    # position 0: predicted 20, but target block for subject 1 is {10} -> miss
    assert set_hit[0, 0].item() is False
