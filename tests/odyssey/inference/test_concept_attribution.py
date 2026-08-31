"""Tests for the exact additive attribution / alignment analysis (CPU)."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

import odyssey.inference.concept_attribution as attribution_module
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.concept_attribution import (
    alignment_from_directions,
    run_streaming_attribution,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.data import iter_patient_sequences


T0 = datetime(2024, 1, 1)
NUM_CONCEPTS = 3
CONCEPT_NAMES = ["aki", "tachycardia", "on_vasopressors"]
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


def _model(
    vocab_size: int, *, global_pairs: bool = False
) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=4,
        padding_idx=0,
        concept_global_pairs=global_pairs,
    )


def _run(*, global_pairs: bool = False, top_k: int = 5):
    vocab = _vocab()
    return run_streaming_attribution(
        _model(len(vocab), global_pairs=global_pairs),
        _events(),
        vocab,
        CONCEPT_NAMES,
        top_k=top_k,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
    )


def test_shares_partition_to_one_over_every_real_position() -> None:
    result = _run()
    # 3 subjects x 20 events, minus one per subject (no next-token target).
    assert result.n_predictions == 3 * 19
    assert 0.0 <= result.mean_concept_contribution <= 1.0
    assert set(result.per_concept_share) == set(CONCEPT_NAMES)
    total = sum(result.per_concept_share.values()) + result.unknown_share
    assert total == pytest.approx(1.0, abs=1e-5)
    # The headline is exactly the known-slot side of that partition.
    assert result.mean_concept_contribution == pytest.approx(
        sum(result.per_concept_share.values()), abs=1e-5
    )


def test_slot_decomposition_reconstructs_the_models_own_logit() -> None:
    """The per-slot terms plus bias must equal the real logit exactly.

    This pins the column-layout assumption (slot i owns lm_head weight
    columns [i*d, (i+1)*d), unknown last) to the model's actual forward,
    not to a reading of the bottleneck code.
    """
    vocab = _vocab()
    model = _model(len(vocab))
    model.eval()
    seqs = iter_patient_sequences(_events(), vocab)
    sampler = PackedLaneSampler(seqs, num_lanes=2, chunk_size=8, reset_prob=0.0)
    chunk = next(iter(sampler))
    with torch.no_grad():
        logits, out, _ = model(chunk.batch)
    k, d = NUM_CONCEPTS, model.bottleneck.embedding_dim
    lane, pos = 0, 3
    y = int(logits[lane, pos].argmax())
    w_y = model.lm_head.weight[y].detach()
    known = sum(
        float(out.concept_embeddings[lane, pos, i] @ w_y[i * d : (i + 1) * d])
        for i in range(k)
    )
    unknown = float(out.unknown_embedding[lane, pos] @ w_y[k * d :])
    bias = float(model.lm_head.bias[y])
    assert known + unknown + bias == pytest.approx(
        float(logits[lane, pos, y]), abs=1e-5
    )


def test_direction_source_tracks_the_bottleneck_variant() -> None:
    assert _run().direction_source == "mean_context_pairs"
    assert _run(global_pairs=True).direction_source == "global_pairs"


def test_alignment_topk_matches_a_manual_weight_product() -> None:
    """Exact check on a global-pairs model, where the direction is a parameter."""
    vocab = _vocab()
    model = _model(len(vocab), global_pairs=True)
    model.eval()
    diff = (
        model.bottleneck.pair_embeddings[:, 0, :]
        - model.bottleneck.pair_embeddings[:, 1, :]
    ).detach()
    alignment = alignment_from_directions(
        model, diff.double(), vocab, CONCEPT_NAMES, top_k=3
    )
    assert [a.concept for a in alignment] == CONCEPT_NAMES
    d = model.bottleneck.embedding_dim
    for i, entry in enumerate(alignment):
        shifts = (
            model.lm_head.weight[:, i * d : (i + 1) * d].detach().double()
            @ diff[i].double()
        )
        top_token, top_value = entry.activate_promotes[0]
        assert top_token == vocab.decode(int(shifts.argmax()))
        assert top_value == pytest.approx(float(shifts.max()))
        bottom_token, bottom_value = entry.deactivate_promotes[0]
        assert bottom_token == vocab.decode(int(shifts.argmin()))
        assert bottom_value == pytest.approx(float(-shifts.min()))
        assert len(entry.activate_promotes) == 3
        # Scores are sorted, largest shift first.
        values = [v for _, v in entry.activate_promotes]
        assert values == sorted(values, reverse=True)


def test_run_rejects_a_name_count_mismatch() -> None:
    vocab = _vocab()
    with pytest.raises(ValueError, match="concept names"):
        run_streaming_attribution(
            _model(len(vocab)),
            _events(),
            vocab,
            ["only_one"],
            num_lanes=2,
            chunk_size=8,
            device="cpu",
        )


def test_main_refuses_to_overwrite_an_existing_output_before_evaluating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same append-only policy as interventions (real incident, 2026-08-22)."""
    existing = tmp_path / "attribution.json"
    existing.write_text("{}")
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

    monkeypatch.setattr(attribution_module, "evaluate_attribution", _boom)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        attribution_module._main()
