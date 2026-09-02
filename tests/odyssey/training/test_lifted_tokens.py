"""Lifted token sets: P(token | concept active) / P(token) over running labels."""

from datetime import datetime, timedelta

import polars as pl
import torch

from odyssey.data.vocabulary import Vocabulary
from odyssey.training.data import iter_patient_sequences
from odyssey.training.lifted_tokens import lifted_token_sets, rank_by_lift


T0 = datetime(2024, 1, 1)
CODES = [f"LAB//{i}//" for i in range(6)]
K = 2


def _events() -> pl.DataFrame:
    """Subjects 1 and 2 emit LAB//5 only after hour 4; subject 3 never does."""
    rows = []
    for sid in (1, 2, 3):
        for i in range(12):
            code = CODES[5] if sid != 3 and i >= 4 and i % 2 == 0 else CODES[i % 5]
            rows.append((sid, code, T0 + timedelta(hours=i), None, 100 + sid))
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


def test_lifted_sets_follow_the_running_label_and_skip_inactive_concepts() -> None:
    vocab = _vocab()
    # concept 0 triggers at hour 4 for subjects 1 and 2 and never for 3;
    # concept 1 never triggers anywhere.
    labels = {
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([1.0, 0.0]),
        3: torch.tensor([0.0, 0.0]),
    }
    masks = {sid: torch.ones(K) for sid in (1, 2, 3)}
    first = {
        1: torch.tensor([4.0, float("inf")]),
        2: torch.tensor([4.0, float("inf")]),
        3: torch.tensor([float("inf"), float("inf")]),
    }
    sets = lifted_token_sets(
        iter_patient_sequences(_events(), vocab),
        vocab_size=len(vocab.token_to_id),
        num_concepts=K,
        concept_labels=labels,
        concept_mask=masks,
        concept_first_times=first,
        supervision="stay",
        top_k=3,
        min_count=1,
        min_share=0.0,
        min_lift=1.0,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
    )
    assert set(sets) == {0, 1}
    # LAB//5 appears only while concept 0 is active, so it has the top lift.
    assert sets[0][0] == vocab.token_to_id[CODES[5]]
    assert len(sets[0]) <= 3
    assert sets[1] == []  # never active: no counts, no lifted tokens


def test_rank_by_lift_respects_top_k_and_the_share_floor() -> None:
    total = torch.tensor([100.0, 100.0, 100.0, 100.0])
    per_concept = torch.tensor([[50.0, 30.0, 2.0, 0.0]])
    sets = rank_by_lift(total, per_concept, top_k=1, min_count=1)
    assert sets == {0: [0]}
    # a 5% share floor drops token 2 (2 of 82 positions) even at min_count=1
    sets = rank_by_lift(total, per_concept, top_k=5, min_count=1, min_share=0.05)
    assert sets[0] == [0, 1]
