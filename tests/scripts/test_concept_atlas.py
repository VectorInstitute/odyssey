"""Concept atlas: exact head alignment per concept and the contribution shares."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from odyssey.data.vocabulary import Vocabulary
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from scripts.concept_atlas import alignment_table, contribution_pass


T0 = datetime(2024, 1, 1)
CODES = [f"LAB//{i}//" for i in range(8)]


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({c: i + 2 for i, c in enumerate(CODES)})
    return Vocabulary(tokens)


def _events() -> pl.DataFrame:
    rows = [
        (sid, CODES[(sid + i) % len(CODES)], T0 + timedelta(hours=i), None, 100 + sid)
        for sid in (1, 2, 3)
        for i in range(10)
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


def _model(vocab_size: int, kind: str = "decomposed") -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=vocab_size, hidden_size=8, padding_idx=0),
        vocab_size=vocab_size,
        num_concepts=3,
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind=kind,
    )


def test_alignment_table_ranks_tokens_by_the_unit_direction_alignment() -> None:
    torch.manual_seed(1)
    weight = torch.randn(6, 4)  # (vocab, hidden)
    embeddings = torch.randn(2, 4)
    id_to_token = {i: f"tok{i}" for i in range(6)}
    rows = alignment_table(weight, embeddings, id_to_token, {"tok0": "zero"}, top=2)
    assert [r["index"] for r in rows] == [0, 1]
    unit = embeddings[0] / embeddings[0].norm()
    align = weight @ unit
    best = int(align.argmax())
    assert rows[0]["promotes"][0]["token"] == f"tok{best}"
    assert rows[0]["promotes"][0]["shift"] == pytest.approx(
        float(align.max()), abs=1e-5
    )
    worst = int(align.argmin())
    assert rows[0]["suppresses"][0]["token"] == f"tok{worst}"
    assert rows[0]["suppresses"][0]["shift"] == pytest.approx(
        float(align.min()), abs=1e-5
    )
    assert rows[0]["norm"] == pytest.approx(float(embeddings[0].norm()))
    # the description map fills in names and falls back to the token
    for entry in rows[0]["promotes"] + rows[0]["suppresses"]:
        expected = "zero" if entry["token"] == "tok0" else entry["token"]
        assert entry["name"] == expected


def test_contribution_shares_sum_to_one_and_activations_are_per_concept() -> None:
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    stats = contribution_pass(
        model, _events(), vocab, num_lanes=2, chunk_size=8, device="cpu"
    )
    share = stats["contribution_share"]
    assert stats["n_positions"] == 27  # 3 subjects x 9 next-event targets
    assert share["named"] + share["unknown"] + share["residual"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert all(v >= 0 for v in share.values())
    assert len(stats["mean_known_activation"]) == 3
    assert len(stats["mean_unknown_activation"]) == model.bottleneck.num_unknown
    assert all(0.0 <= a <= 1.0 for a in stats["mean_known_activation"])


def test_contribution_pass_refuses_a_mixture_bottleneck() -> None:
    vocab = _vocab()
    with pytest.raises(AssertionError):
        contribution_pass(
            _model(len(vocab.token_to_id), kind="mixture"),
            _events(),
            vocab,
            num_lanes=2,
            chunk_size=8,
            device="cpu",
        )
