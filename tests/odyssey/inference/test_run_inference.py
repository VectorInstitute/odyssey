"""CPU-testable pieces of the inference script.

The real streaming inference path needs EHRHybridBackbone/CUDA, see
test_run_inference_gpu.py.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

import odyssey.inference.run_inference as ri
from odyssey.data.concepts import ConceptDefinition, ConceptRule
from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.run_inference import (
    InferenceResults,
    _block_set_hits,
    _latest_checkpoint,
    _parse_args,
    _RunningTaskMetrics,
    load_run,
    refuse_existing_output,
    results_to_dict,
    run_streaming_inference,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.backbones.transformer import TransformerBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneck
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS
from odyssey.training.data import (
    build_concept_label_dicts,
    build_visit_concept_label_dicts,
)
from odyssey.training.metrics import (
    TaskMetrics,
    compute_task_metrics,
    compute_task_metrics_by_code_type,
)
from odyssey.training.train import TrainingConfig


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


def test_load_and_bin_held_out_applies_the_given_binner_not_a_fresh_one(
    tmp_path: Path,
) -> None:
    """Held-out data must be binned with the train-fit binner, never re-fit.

    Re-fitting on held-out data would leak its own value distribution
    into the bin boundaries -- the exact leakage load_and_bin_held_out's
    docstring says this function exists to avoid.
    """
    shard_dir = tmp_path / "held_out"
    shard_dir.mkdir()
    pl.DataFrame(
        [(1, "LAB//UNMAPPED//x", datetime(2024, 1, 1), 999.0, 101)],
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    ).write_parquet(shard_dir / "0.parquet")

    # a binner whose boundaries were never fit on this value (999.0 is
    # far outside them) -- if this got re-fit on the held-out data
    # itself, the value would land in a normal middle bucket instead
    binner = QuantileBinner(boundaries={"LAB//UNMAPPED//x": [1.0, 2.0]}, n_bins=3)

    out = ri.load_and_bin_held_out(shard_dir, binner)

    assert out["code"].to_list() == ["LAB//UNMAPPED//x::Q3"]


def test_build_category_lookup_groups_icd_codes_by_three_character_category() -> None:
    """I5023 and I50 (the icd3 backoff token) must share one category id."""
    vocab = Vocabulary.build(
        [
            "DIAGNOSIS//ICD//10//I5023",
            "DIAGNOSIS//ICD//10//I50",
            "DIAGNOSIS//ICD//10//J449",
            "LAB//220045//bpm",  # not ICD-shaped: must get -1
        ],
        min_count=1,
    )
    lookup, n_categories = ri._build_category_lookup(vocab, "cpu")

    i5023 = vocab.token_to_id["DIAGNOSIS//ICD//10//I5023"]
    i50 = vocab.token_to_id["DIAGNOSIS//ICD//10//I50"]
    j449 = vocab.token_to_id["DIAGNOSIS//ICD//10//J449"]
    lab = vocab.token_to_id["LAB//220045//bpm"]

    assert lookup[i5023] == lookup[i50]
    assert lookup[i5023] != lookup[j449]
    assert lookup[lab] == -1
    assert n_categories == 2  # I50 and J449, the two distinct categories seen


def test_running_time_metrics_update_with_no_valid_positions_is_a_noop() -> None:
    """A chunk where every position is invalid (all-padding) must not perturb state."""
    stats = ri._RunningTimeMetrics(DEFAULT_TIME_BIN_EDGES_HOURS)
    n_bins = len(DEFAULT_TIME_BIN_EDGES_HOURS) + 1
    stats.update(
        hazard_logits=torch.zeros(4, n_bins),
        gap_hours=torch.zeros(4),
        valid=torch.zeros(4, dtype=torch.bool),
    )
    assert stats.n == 0
    assert stats.finalize() is None


def test_running_time_metrics_finalize_on_no_updates_returns_none() -> None:
    """Nothing was ever accumulated (an empty held-out split): finalize is None."""
    assert ri._RunningTimeMetrics(DEFAULT_TIME_BIN_EDGES_HOURS).finalize() is None


def test_running_bucket_update_with_empty_targets_is_a_noop() -> None:
    bucket = ri._RunningBucket()
    bucket.update(
        logits=torch.zeros(0, 10),
        targets=torch.zeros(0, dtype=torch.long),
        top_k=(1, 5),
    )
    assert bucket.n == 0


def test_running_bucket_finalize_with_nothing_accumulated_raises() -> None:
    """A bucket that never saw a non-ignored prediction must fail loudly."""
    with pytest.raises(ValueError, match="no non-ignored predictions"):
        ri._RunningBucket().finalize()


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


def test_parse_args_overwrite_defaults_to_false(
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
    assert _parse_args().overwrite is False


def test_parse_args_overwrite_flag_is_honoured(
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
            "--overwrite",
        ],
    )
    assert _parse_args().overwrite is True


# ---------------------------------------------------------------------------
# refuse_existing_output: append-only-by-default guard for science outputs
# ---------------------------------------------------------------------------


def test_refuse_existing_output_raises_on_an_existing_file_without_overwrite(
    tmp_path: Path,
) -> None:
    """Real incident: an overwrite lost an irreplaceable row-level dump (2026-08-22)."""
    existing = tmp_path / "alerts.json"
    existing.write_text("{}")

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        refuse_existing_output(existing, overwrite=False, kind="alerts")


def test_refuse_existing_output_allows_an_existing_file_with_overwrite(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "alerts.json"
    existing.write_text("{}")

    refuse_existing_output(existing, overwrite=True, kind="alerts")  # no raise


def test_refuse_existing_output_allows_a_nonexistent_file_either_way(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist.json"

    refuse_existing_output(missing, overwrite=False, kind="alerts")  # no raise
    refuse_existing_output(missing, overwrite=True, kind="alerts")  # no raise


# ---------------------------------------------------------------------------
# _block_set_hits: set-based scoring over same-timestamp event blocks
# ---------------------------------------------------------------------------


def test_block_set_hits_counts_within_block_predictions() -> None:
    # One lane, 5 targets. Input times (positions 0..5): targets 0-2 share
    # time 1.0 (one block: tokens 11,12,13), targets 3-4 share time 2.0
    # (block: tokens 14,15). All tokens are one family here.
    targets = torch.tensor([[11, 12, 13, 14, 15]])
    # times for target j = input time at j+1
    times_full = torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0, 2.0]])
    subject_ids = torch.ones(1, 6, dtype=torch.long)
    real = torch.ones(1, 5, dtype=torch.bool)
    one_family = torch.ones(100, dtype=torch.long)

    # predictions: 13 (in block 1), 11 (in block 1), 99 (miss), 14
    # (block 2; note token 15 sits at the final position, whose target
    # time is outside the chunk, so it is invisible to block membership:
    # the documented boundary approximation)
    top1 = torch.tensor([[13, 11, 99, 14, 14]])
    hits = _block_set_hits(
        top1,
        targets,
        times=times_full[:, :5],
        subject_ids=subject_ids[:, :5],
        real_mask=real,
        vocab_size=100,
        type_lookup=one_family,
    )
    # the final position has no in-chunk target time: excluded
    assert hits.set_valid[0].tolist() == [True, True, True, True, False]
    assert hits.set_hit[0].tolist() == [True, True, False, True, False]
    # no category lookup: category flags stay all-False
    assert not hits.category_valid.any()


def test_block_set_hits_blocks_never_span_subjects() -> None:
    # Two patients back to back, same timestamps: token 20 belongs to
    # subject 2's block, so subject 1's prediction of 20 must not count.
    targets = torch.tensor([[10, 20, 21]])
    times_full = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    subject_ids = torch.tensor([[1, 1, 2, 2]])
    real = torch.ones(1, 3, dtype=torch.bool)
    top1 = torch.tensor([[20, 10, 21]])
    hits = _block_set_hits(
        top1,
        targets,
        times=times_full[:, :3],
        subject_ids=subject_ids[:, :3],
        real_mask=real,
        vocab_size=100,
        type_lookup=torch.ones(100, dtype=torch.long),
    )
    # position 0: predicted 20, but target block for subject 1 is {10} -> miss
    assert hits.set_hit[0, 0].item() is False


def test_block_set_hits_require_the_targets_own_family() -> None:
    """A discharge block holds diagnoses AND the discharge/billing tokens.

    Predicting the discharge token at a diagnosis target must not count
    as a diagnosis set-hit: membership is restricted to the target's own
    family, so a set-hit says the model named an event of that family.
    """
    # block at time 1.0: tokens 11 (diagnosis), 12 (diagnosis), 30 (visit)
    targets = torch.tensor([[11, 12, 30, 40]])
    times_full = torch.tensor([[0.0, 1.0, 1.0, 1.0, 2.0]])
    subject_ids = torch.ones(1, 5, dtype=torch.long)
    real = torch.ones(1, 4, dtype=torch.bool)
    families = torch.zeros(100, dtype=torch.long)
    families[[11, 12]] = 1  # diagnosis
    families[[30]] = 5  # visit
    families[[40]] = 4  # lab
    # at target 11 predict 30 (visit token in the same block): NOT a hit;
    # at target 12 predict 11 (diagnosis in the block): hit;
    # at target 30 predict 12 (diagnosis, but target family is visit): NOT
    # a hit -- the family is the target's, in both directions.
    top1 = torch.tensor([[30, 11, 12, 40]])
    hits = _block_set_hits(
        top1,
        targets,
        times=times_full[:, :4],
        subject_ids=subject_ids[:, :4],
        real_mask=real,
        vocab_size=100,
        type_lookup=families,
    )
    assert hits.set_hit[0].tolist() == [False, True, False, False]


def test_block_set_hits_category_level_scoring() -> None:
    """Category flags credit the right 3-char ICD category within the block."""
    # tokens: 11 = I5023, 12 = I50 (its category token), 13 = E11 (other
    # category); 30 = a non-ICD token.
    targets = torch.tensor([[11, 13, 30]])
    times_full = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    subject_ids = torch.ones(1, 4, dtype=torch.long)
    real = torch.ones(1, 3, dtype=torch.bool)
    families = torch.zeros(100, dtype=torch.long)
    families[[11, 12, 13]] = 1
    families[30] = 5
    categories = torch.full((100,), -1, dtype=torch.long)
    categories[[11, 12]] = 0  # I50
    categories[13] = 1  # E11
    # at target I5023 predict I50 (parent): exact/set miss, category hit;
    # at target E11 predict I50: category miss (I50 in block, but E11 also
    # in block? no: block members' categories are {I50, E11}; predicted
    # I50's category IS in the block) -> category hit is a set-style hit.
    top1 = torch.tensor([[12, 12, 12]])
    hits = _block_set_hits(
        top1,
        targets,
        times=times_full[:, :3],
        subject_ids=subject_ids[:, :3],
        real_mask=real,
        vocab_size=100,
        type_lookup=families,
        category_lookup=categories,
        n_categories=2,
    )
    assert hits.set_hit[0].tolist() == [False, False, False]
    # target 30 is not ICD-coded: category metric does not apply there
    assert hits.category_valid[0].tolist() == [True, True, False]
    assert hits.category_hit[0].tolist() == [True, True, False]


def test_time_to_event_config_follows_the_checkpoint(tmp_path: Path) -> None:
    """Load a pre-time-head run whose config predates the field.

    The dataclass default would say True, but the checkpoint has no time
    head, and the checkpoint is the authority.
    """
    run_dir = tmp_path
    config = TrainingConfig(train_shard_dir="a", tuning_shard_dir="b", output_dir="c")
    payload = {k: v for k, v in config.__dict__.items() if k != "time_to_event"}
    (run_dir / "config.json").write_text(json.dumps(payload))
    Vocabulary({"[PAD]": 0, "[UNK]": 1, "LAB//220045//bpm": 2}).save(
        run_dir / "vocabulary.json"
    )
    (run_dir / "quantile_binner.json").write_text(
        json.dumps({"n_bins": 5, "boundaries": {}})
    )
    # a checkpoint without any time_head.* keys must load without a time head
    torch.save({"model": {}}, run_dir / "checkpoint_final.pt")
    try:
        load_run(run_dir, device="cpu")
    except RuntimeError as exc:
        # the real backbone needs mamba/CUDA; what matters is that no
        # time_head keys were demanded from the checkpoint
        assert "time_head" not in str(exc)
    except ImportError:
        pass  # mamba-ssm not installed: construction itself is CUDA-only


def test_streaming_inference_scores_a_baseline_model() -> None:
    """A no-bottleneck model gets task/set/time metrics and no concept metrics."""
    vocab = _vocab()
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
    )
    codes = ["DIAGNOSIS//A", "MEDICATION//B", "LAB//220045//bpm", "PROCEDURE//C"]
    rows = []
    for sid in (1, 2):
        for i in range(12):
            rows.append(
                (
                    sid,
                    codes[i % 4],
                    datetime(2024, 1, 1) + timedelta(hours=i),
                    None,
                    100 + sid,
                )
            )
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
    results = run_streaming_inference(
        model, events, vocab, {}, {}, num_lanes=1, chunk_size=8, device="cpu"
    )
    assert results.task_metrics.n_predictions == 2 * 11
    assert results.concept_metrics == []
    assert results.time_metrics is not None and results.time_metrics.n_positions > 0


def _concepts() -> list:
    return [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        ),
    ]


def _bottleneck_events(hadm_ids: dict) -> pl.DataFrame:
    """Two subjects, HR readings (one tachycardic, one not), plus filler codes."""
    codes = ["LAB//220045//bpm", "DIAGNOSIS//A", "MEDICATION//B"]
    rows = []
    for sid, hr in ((1, 120.0), (2, 80.0)):
        for i in range(6):
            code = codes[i % 3]
            rows.append(
                (
                    sid,
                    code,
                    datetime(2024, 1, 1) + timedelta(hours=i),
                    hr if code == "LAB//220045//bpm" else None,
                    hadm_ids[sid],
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


def _bottleneck_model(
    vocab: Vocabulary, num_concepts: int
) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=num_concepts,
        embedding_dim=4,
        padding_idx=0,
    )


def test_streaming_inference_stay_supervision_scores_concept_metrics() -> None:
    """A real bottleneck model, stay-scoped pooling: the entirely-untested pooled path.

    Every other CPU streaming test in this file uses a no-bottleneck
    BaselineSequenceModel, so _StreamingAccumulators' bottleneck-pooling
    branch and _finalize_inference_results' concept/observability/
    orthogonality computation had zero coverage.
    """
    concepts = _concepts()
    events = _bottleneck_events({1: 101, 2: 102})
    vocab = Vocabulary.build(events["code"].unique().to_list(), min_count=1)
    labels, masks = build_concept_label_dicts(events, concepts)
    model = _bottleneck_model(vocab, len(concepts))

    results = run_streaming_inference(
        model,
        events,
        vocab,
        labels,
        masks,
        num_lanes=1,
        chunk_size=8,
        device="cpu",
        supervision="stay",
        concepts=concepts,
    )

    assert results.n_patient_ends_scored == 2
    assert len(results.concept_metrics) == 1
    assert math.isfinite(results.orthogonality)


def test_streaming_inference_visit_supervision_scores_concept_metrics() -> None:
    """Same as the stay-scoped test, but exercising the visit-id gather branch."""
    concepts = _concepts()
    events = _bottleneck_events({1: 101, 2: 102})
    vocab = Vocabulary.build(events["code"].unique().to_list(), min_count=1)
    labels, masks = build_visit_concept_label_dicts(events, concepts)
    model = _bottleneck_model(vocab, len(concepts))

    results = run_streaming_inference(
        model,
        events,
        vocab,
        labels,
        masks,
        num_lanes=1,
        chunk_size=8,
        device="cpu",
        supervision="visit",
        concepts=concepts,
    )

    assert results.n_patient_ends_scored == 2
    assert len(results.concept_metrics) == 1


def test_streaming_inference_bottleneck_model_with_no_pooled_positions_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """visit-scoped pooling with no real hadm_id anywhere: no crash, a warning instead.

    Forecasting metrics must still come back (they never depend on
    pooling); concept/observability/orthogonality come back empty since
    chunk.visit_end never fires without a real visit boundary.
    """
    concepts = _concepts()
    events = _bottleneck_events({1: None, 2: None})  # type: ignore[dict-item]
    vocab = Vocabulary.build(events["code"].unique().to_list(), min_count=1)
    labels, masks = build_concept_label_dicts(events, concepts)
    model = _bottleneck_model(vocab, len(concepts))

    with caplog.at_level("WARNING"):
        results = run_streaming_inference(
            model,
            events,
            vocab,
            labels,
            masks,
            num_lanes=1,
            chunk_size=8,
            device="cpu",
            supervision="visit",
            concepts=concepts,
        )

    assert results.task_metrics.n_predictions > 0
    assert results.n_patient_ends_scored == 0
    assert results.concept_metrics == []
    assert any("no visit-scoped pool positions" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# backbone="transformer": PackedContextSampler dispatch and tail-slice reporting
# ---------------------------------------------------------------------------


def _events_for_subjects(subject_lengths: dict, codes: list) -> pl.DataFrame:
    rows = []
    for sid, n in subject_lengths.items():
        for i in range(n):
            rows.append(
                (
                    sid,
                    codes[i % len(codes)],
                    datetime(2024, 1, 1) + timedelta(hours=i),
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


def test_build_sampler_dispatches_on_backbone() -> None:
    patients_a = iter([])
    patients_b = iter([])

    hybrid = ri._build_sampler(
        patients_a, backbone="hybrid", num_lanes=2, chunk_size=8, max_context=16
    )
    transformer = ri._build_sampler(
        patients_b, backbone="transformer", num_lanes=2, chunk_size=8, max_context=16
    )

    assert isinstance(hybrid, PackedLaneSampler)
    assert isinstance(transformer, PackedContextSampler)


def _transformer_model(vocab: Vocabulary) -> BaselineSequenceModel:
    torch.manual_seed(0)
    return BaselineSequenceModel(
        backbone=TransformerBackbone(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=2,
            num_heads=4,
            padding_idx=0,
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
    )


def test_streaming_inference_backbone_transformer_no_truncation_has_no_tail_slice() -> (
    None
):
    vocab = _vocab()
    model = _transformer_model(vocab)
    codes = ["DIAGNOSIS//A", "MEDICATION//B", "LAB//220045//bpm", "PROCEDURE//C"]
    events = _events_for_subjects({1: 8, 2: 8}, codes)

    results = run_streaming_inference(
        model,
        events,
        vocab,
        {},
        {},
        device="cpu",
        backbone="transformer",
        num_lanes=2,
        max_context=64,
    )

    assert results.task_metrics.n_predictions > 0
    assert results.tail_slice is None


def test_streaming_inference_backbone_transformer_reports_tail_slice_separately() -> (
    None
):
    """A patient longer than max_context is truncated -- and scored twice.

    Once pooled into the headline numbers as usual (with only its kept
    tail visible), and again alone in ``tail_slice``, so the cost of
    losing the rest of that patient's history is visible rather than
    averaged away.
    """
    vocab = _vocab()
    model = _transformer_model(vocab)
    codes = ["DIAGNOSIS//A", "MEDICATION//B", "LAB//220045//bpm", "PROCEDURE//C"]
    # subject 1: 40 events, will be truncated at max_context=10;
    # subject 2: 6 events, comfortably fits, never truncated.
    events = _events_for_subjects({1: 40, 2: 6}, codes)

    results = run_streaming_inference(
        model,
        events,
        vocab,
        {},
        {},
        device="cpu",
        backbone="transformer",
        num_lanes=2,
        max_context=10,
    )

    assert results.tail_slice is not None
    # every real target the tail slice scores belongs to the truncated
    # subject alone, so it can never see more positions than the overall
    # pass scored in total.
    assert results.tail_slice.task_metrics.n_predictions > 0
    assert (
        results.tail_slice.task_metrics.n_predictions
        <= results.task_metrics.n_predictions
    )
    assert results.tail_slice.tail_slice is None


def test_results_to_dict_renders_nan_as_null() -> None:
    """A baseline run has no orthogonality; NaN must not reach JSON output."""
    tm = TaskMetrics(
        cross_entropy=1.0,
        perplexity=2.7,
        top1_accuracy=0.5,
        top5_accuracy=0.8,
        n_predictions=3,
    )
    results = InferenceResults(
        task_metrics=tm,
        task_metrics_by_code_type={},
        concept_metrics=[],
        observability_metrics=[],
        orthogonality=float("nan"),
        n_patient_ends_scored=0,
    )
    d = results_to_dict(results)
    assert d["orthogonality"] is None
    json.loads(json.dumps(d, allow_nan=False))  # strict JSON round-trips


def test_value_embeddings_flag_ignores_the_backbone_merge_attention(
    tmp_path: Path,
) -> None:
    """MergeAttention has its own value_proj; only the embeddings' one counts."""
    run_dir = tmp_path
    config = TrainingConfig(train_shard_dir="a", tuning_shard_dir="b", output_dir="c")
    (run_dir / "config.json").write_text(json.dumps(config.__dict__))
    Vocabulary({"[PAD]": 0, "[UNK]": 1, "LAB//220045//bpm": 2}).save(
        run_dir / "vocabulary.json"
    )
    (run_dir / "quantile_binner.json").write_text(
        json.dumps({"n_bins": 5, "boundaries": {}})
    )
    keys = {
        "backbone.layers.0.merge.value_proj.weight": torch.zeros(1),
        "backbone.layers.0.merge.value_proj.bias": torch.zeros(1),
    }
    torch.save({"model": keys}, run_dir / "checkpoint_final.pt")
    seen = {}

    def fake_build_model(cfg, *, vocab_size, num_concepts):  # noqa: ARG001
        seen["value_embeddings"] = cfg.value_embeddings
        seen["event_head_hidden"] = cfg.event_head_hidden
        seen["concept_global_pairs"] = cfg.concept_global_pairs
        seen["unknown_dim"] = cfg.unknown_dim
        raise RuntimeError("stop here")

    original = ri.build_model
    ri.build_model = fake_build_model
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            load_run(run_dir, device="cpu")
    finally:
        ri.build_model = original
    assert seen["value_embeddings"] is False
    assert seen["event_head_hidden"] == 0

    keys["backbone.embeddings.embeddings.value_proj.weight"] = torch.zeros(1)
    keys["event_heads.proj.0.weight"] = torch.zeros(32, 8)  # MLP readout, hidden 32
    torch.save({"model": keys}, run_dir / "checkpoint_final.pt")
    ri.build_model = fake_build_model
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            load_run(run_dir, device="cpu")
    finally:
        ri.build_model = original
    assert seen["value_embeddings"] is True
    assert seen["event_head_hidden"] == 32
    assert seen["concept_global_pairs"] is False

    # bottleneck variants read off parameter shapes: global pairs, unknown width 6
    keys["bottleneck.pair_embeddings"] = torch.zeros(3, 2, 4)
    keys["bottleneck.context_proj.weight"] = torch.zeros(12, 8)
    torch.save({"model": keys}, run_dir / "checkpoint_final.pt")
    ri.build_model = fake_build_model
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            load_run(run_dir, device="cpu")
    finally:
        ri.build_model = original
    assert seen["concept_global_pairs"] is True and seen["unknown_dim"] == 6


def test_unknown_dim_round_trips_for_the_non_global_pairs_unequal_width_case(
    tmp_path: Path,
) -> None:
    """Round-trip test for review finding 7 (test-only, per instruction).

    load_run's non-global-pairs unknown_dim reconstruction (rows -
    n_known*2*emb) // 2) had zero test coverage -- only the
    concept_global_pairs=True branch was exercised
    (test_value_embeddings_flag_ignores_the_backbone_merge_attention).
    This builds a *real* ConceptBottleneck (global_pairs=False,
    unknown_dim != embedding_dim -- the branch with a separate
    unknown_prob_weight, the least-covered shape combination), takes its
    actual state_dict, and checks the reconstructed config.unknown_dim
    matches what the module was actually built with.

    If this fails: the reconstruction formula is wrong for this shape
    combination -- stop and report to odyssey-db, do not fix it here.
    """
    num_concepts = 5
    embedding_dim = 8
    unknown_dim = 6  # deliberately != embedding_dim
    bottleneck = ConceptBottleneck(
        hidden_size=16,
        num_concepts=num_concepts,
        embedding_dim=embedding_dim,
        global_pairs=False,
        unknown_dim=unknown_dim,
    )
    assert "unknown_prob_weight" in dict(bottleneck.named_parameters()), (
        "fixture assumption broken: expected the separate-unknown-weight branch"
    )

    run_dir = tmp_path
    config = TrainingConfig(train_shard_dir="a", tuning_shard_dir="b", output_dir="c")
    (run_dir / "config.json").write_text(json.dumps(config.__dict__))
    Vocabulary({"[PAD]": 0, "[UNK]": 1, "LAB//220045//bpm": 2}).save(
        run_dir / "vocabulary.json"
    )
    (run_dir / "quantile_binner.json").write_text(
        json.dumps({"n_bins": 5, "boundaries": {}})
    )
    state = {f"bottleneck.{k}": v for k, v in bottleneck.state_dict().items()}
    torch.save({"model": state}, run_dir / "checkpoint_final.pt")

    seen = {}

    def fake_build_model(cfg, *, vocab_size, num_concepts):  # noqa: ARG001
        seen["unknown_dim"] = cfg.unknown_dim
        seen["concept_global_pairs"] = cfg.concept_global_pairs
        raise RuntimeError("stop here")

    original = ri.build_model
    ri.build_model = fake_build_model
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            load_run(run_dir, device="cpu")
    finally:
        ri.build_model = original

    assert seen["concept_global_pairs"] is False
    assert seen["unknown_dim"] == unknown_dim


def test_recency_features_reconstructs_for_a_baseline_model_kind(
    tmp_path: Path,
) -> None:
    """load_run's recency-shape check has a separate base for model_kind='baseline'.

    Bottleneck models measure the head input against
    n_concepts*embedding_dim + unknown_dim; a baseline (no bottleneck)
    model measures it against hidden_size directly instead -- that
    second branch had zero test coverage. Same monkeypatch-build_model
    technique as test_unknown_dim_round_trips_..., not a real model.
    """
    from odyssey.models.sequence_model import RECENCY_DIM  # noqa: PLC0415

    hidden_size = 16
    run_dir = tmp_path
    config = TrainingConfig(
        train_shard_dir="a",
        tuning_shard_dir="b",
        output_dir="c",
        model_kind="baseline",
        hidden_size=hidden_size,
    )
    (run_dir / "config.json").write_text(json.dumps(config.__dict__))
    Vocabulary({"[PAD]": 0, "[UNK]": 1, "LAB//220045//bpm": 2}).save(
        run_dir / "vocabulary.json"
    )
    (run_dir / "quantile_binner.json").write_text(
        json.dumps({"n_bins": 5, "boundaries": {}})
    )
    # in_features = hidden_size + RECENCY_DIM: the shape a real
    # recency-enabled baseline model's time head would have (see
    # BaselineSequenceModel.__init__'s head_in computation).
    state = {"time_head.proj.weight": torch.zeros(3, hidden_size + RECENCY_DIM)}
    torch.save({"model": state}, run_dir / "checkpoint_final.pt")

    seen = {}

    def fake_build_model(cfg, *, vocab_size, num_concepts):  # noqa: ARG001
        seen["recency_features"] = cfg.recency_features
        seen["model_kind"] = cfg.model_kind
        raise RuntimeError("stop here")

    original = ri.build_model
    ri.build_model = fake_build_model
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            load_run(run_dir, device="cpu")
    finally:
        ri.build_model = original

    assert seen["model_kind"] == "baseline"
    assert seen["recency_features"] is True


def test_default_checkpoint_prefers_best_matching_the_clis(tmp_path: Path) -> None:
    """Library default and CLI default must resolve the same checkpoint."""
    (tmp_path / "checkpoint_500.pt").touch()
    (tmp_path / "checkpoint_final.pt").touch()
    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint_final.pt"
    (tmp_path / "checkpoint_best.pt").touch()
    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint_best.pt"


# ---------------------------------------------------------------------------
# value_head: config round-trip through load_run, and streaming metrics
# ---------------------------------------------------------------------------


def _write_transformer_run(
    tmp_path: Path,
    vocab: Vocabulary,
    *,
    value_head: bool,
    value_fourier: bool = False,
    value_embeddings: bool = True,
    hidden_size: int = 16,
) -> Path:
    """Build a real (CPU-only) transformer BaselineSequenceModel run dir."""
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TransformerBackbone(
            vocab_size=len(vocab),
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_heads=4,
            padding_idx=0,
            use_values=value_embeddings,
            use_value_fourier=value_fourier,
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        value_head=value_head,
    )
    config = TrainingConfig(
        train_shard_dir="a",
        tuning_shard_dir="b",
        output_dir="c",
        model_kind="baseline",
        backbone="transformer",
        hidden_size=hidden_size,
        num_hidden_layers=2,
        attn_num_heads=4,
        time_to_event=True,
        value_head=value_head,
        value_fourier=value_fourier,
        value_embeddings=value_embeddings,
    )
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))
    vocab.save(tmp_path / "vocabulary.json")
    (tmp_path / "quantile_binner.json").write_text(
        json.dumps({"n_bins": 5, "boundaries": {}})
    )
    torch.save({"model": model.state_dict()}, tmp_path / "checkpoint_final.pt")
    return tmp_path


def test_load_run_reconstructs_value_head(tmp_path: Path) -> None:
    vocab = _vocab()
    _write_transformer_run(tmp_path, vocab, value_head=True, value_fourier=True)

    model, _, _, config = load_run(tmp_path, device="cpu")

    assert config.value_head is True
    assert config.value_fourier is True
    assert model.value_head is not None
    embeddings = model.backbone.embeddings.embeddings
    assert embeddings.value_proj is not None
    from odyssey.models.embeddings import N_FOURIER_FEATURES  # noqa: PLC0415

    assert embeddings.value_proj.in_features == N_FOURIER_FEATURES


def test_load_run_value_head_false_checkpoint_unaffected(tmp_path: Path) -> None:
    """No regression: a run with no value head loads exactly as before."""
    vocab = _vocab()
    _write_transformer_run(
        tmp_path, vocab, value_head=False, value_fourier=False, value_embeddings=False
    )

    model, _, _, config = load_run(tmp_path, device="cpu")

    assert config.value_head is False
    assert config.value_fourier is False
    assert model.value_head is None
    assert model.backbone.embeddings.embeddings.value_proj is None


def test_load_run_value_fourier_false_uses_three_features(tmp_path: Path) -> None:
    vocab = _vocab()
    _write_transformer_run(
        tmp_path, vocab, value_head=True, value_fourier=False, value_embeddings=True
    )

    model, _, _, config = load_run(tmp_path, device="cpu")

    assert config.value_fourier is False
    assert model.backbone.embeddings.embeddings.value_proj.in_features == 3


def test_load_run_round_trip_reproduces_identical_predictions(tmp_path: Path) -> None:
    """Save, reload via load_run, predict again: outputs must be identical."""
    vocab = _vocab()
    _write_transformer_run(tmp_path, vocab, value_head=True, value_fourier=True)

    model1, _, _, _ = load_run(tmp_path, device="cpu")
    model2, _, _, _ = load_run(tmp_path, device="cpu")

    torch.manual_seed(42)
    ids = torch.randint(1, len(vocab), (2, 6))
    from odyssey.data.types import (  # noqa: PLC0415
        AuxiliaryInputs,
        ClinicalSequenceBatch,
    )

    aux = AuxiliaryInputs(
        type_ids=torch.zeros(2, 6, dtype=torch.long),
        time_stamps=torch.arange(6).float().unsqueeze(0).repeat(2, 1),
        ages=torch.zeros(2, 6),
        visit_orders=torch.zeros(2, 6, dtype=torch.long),
        visit_segments=torch.zeros(2, 6, dtype=torch.long),
        values=torch.rand(2, 6),
    )
    batch = ClinicalSequenceBatch(concept_ids=ids, aux=aux)
    with torch.no_grad():
        fwd1 = model1.forward_with_features(batch)
        fwd2 = model2.forward_with_features(batch)
        emb1 = model1.backbone.embeddings.embeddings.word_embeddings(ids)
        emb2 = model2.backbone.embeddings.embeddings.word_embeddings(ids)
        q1 = model1.value_head(fwd1.features, emb1)
        q2 = model2.value_head(fwd2.features, emb2)

    assert torch.equal(fwd1.logits, fwd2.logits)
    assert torch.equal(q1, q2)


def test_streaming_inference_populates_value_metrics_for_transformer_backbone() -> None:
    from odyssey.models.value_head import DEFAULT_QUANTILE_LEVELS  # noqa: PLC0415

    vocab = _vocab()
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TransformerBackbone(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=2,
            num_heads=4,
            padding_idx=0,
            use_values=True,
        ),
        vocab_size=len(vocab),
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        value_head=True,
    )
    codes = ["DIAGNOSIS//A", "MEDICATION//B", "LAB//220045//bpm", "PROCEDURE//C"]
    events = _events_for_subjects({1: 10, 2: 10}, codes)
    # iter_patient_sequences reads a pre-binned numeric_z column directly
    # (VALUE_Z_COL) -- add_value_tokens's real output, faked here since
    # this test skips the full binning pipeline.
    events = events.with_columns(
        pl.when(pl.col("code") == "LAB//220045//bpm")
        .then(pl.Series(torch.randn(events.height).tolist()))
        .otherwise(None)
        .alias("numeric_z")
    )

    results = run_streaming_inference(
        model,
        events,
        vocab,
        {},
        {},
        device="cpu",
        backbone="transformer",
        num_lanes=2,
        max_context=64,
    )

    # LAB//220045//bpm carries a real value, so real value targets exist.
    assert results.value_metrics is not None
    assert results.value_metrics.n_positions > 0
    assert set(results.value_metrics.coverage.keys()) == {
        f"{lvl:g}" for lvl in DEFAULT_QUANTILE_LEVELS
    }


def test_streaming_inference_no_value_metrics_without_value_head() -> None:
    vocab = _vocab()
    model = _transformer_model(vocab)  # no value_head
    codes = ["DIAGNOSIS//A", "MEDICATION//B", "LAB//220045//bpm", "PROCEDURE//C"]
    events = _events_for_subjects({1: 8, 2: 8}, codes)

    results = run_streaming_inference(
        model, events, vocab, {}, {}, device="cpu", backbone="transformer", num_lanes=2
    )

    assert results.value_metrics is None
