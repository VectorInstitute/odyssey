"""The steering pass and its orchestration on a tiny in-memory model.

``prepare`` and the CLI need a saved run and are exercised in
tests/odyssey/training/test_train_steering.py; here everything after
loading is driven directly so each piece is checked on known inputs.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from odyssey.data.concepts import concepts_for_source
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.steering import (
    CLINICAL_EXPECTATIONS,
    HORIZONS_HOURS,
    PositionStrata,
    SteeringPrepared,
    SteeringPush,
    SubjectReadout,
    _labels_for,
    _outcome_risk,
    _to_json,
    clinician_line,
    evaluate_steering,
    run_steering_pass,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.steering import steering_direction, steering_gamma


T0 = datetime(2024, 1, 1)
CODES = [f"LAB//{i}//" for i in range(8)]
EVENTS = ["death", "icu_admission"]
NAMES = ["tachycardia", "hypotension", "fever"]


def _vocab() -> Vocabulary:
    tokens = {"[PAD]": 0, "[UNK]": 1}
    tokens.update({c: i + 2 for i, c in enumerate(CODES)})
    return Vocabulary(tokens)


def _events() -> pl.DataFrame:
    rows = [
        (sid, CODES[(sid + i) % len(CODES)], T0 + timedelta(hours=i), None, 100 + sid)
        for sid in (1, 2, 3, 4)
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


def _model(vocab_size: int, with_heads: bool = True) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=vocab_size, hidden_size=8, padding_idx=0),
        vocab_size=vocab_size,
        num_concepts=len(NAMES),
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind="decomposed",
        event_names=EVENTS if with_heads else None,
    )
    # The tiny GRU has no block list; give the stream site a block to hook.
    model.backbone.layers = nn.ModuleList([nn.Identity()])  # type: ignore[attr-defined]
    model.eval()
    return model


def _prepared(
    model: ConceptBottleneckSequenceModel, vocab: Vocabulary
) -> SteeringPrepared:
    lifted = {0: [2, 3], 1: [4], 2: []}
    return SteeringPrepared(
        model=model,
        vocab=vocab,
        events_binned=_events(),
        concept_names=NAMES,
        event_names=EVENTS,
        lifted=lifted,
        gammas=[
            steering_gamma(model, steering_direction(model, c), tau=1.0)
            for c in range(3)
        ],
        supervision="stay",
        tables=None,
        token_names={},
    )


def test_pass_visits_every_subject_once_and_reads_at_risk_everywhere_without_tables() -> (
    None
):
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    lifted = [torch.tensor([2, 3]), torch.tensor([], dtype=torch.long)]
    readouts = run_steering_pass(
        model,
        _events(),
        vocab,
        push=None,
        lifted_sets=lifted,
        tables=None,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
    )
    assert set(readouts) == {1, 2, 3, 4}
    r = readouts[1]
    assert r.n == 9  # ten events, nine next-event positions
    assert r.risk_means().shape == (len(EVENTS), len(HORIZONS_HOURS))
    assert not np.isnan(r.risk_means()).any()  # every position counts as at risk
    mass = r.lifted_mass / r.n
    assert 0.0 < mass[0] < 1.0
    assert mass[1] == 0.0  # empty lifted set contributes no mass


@pytest.mark.parametrize("site", ["bottleneck", "stream"])
def test_pushed_pass_changes_the_readout_and_keeps_the_subjects(site: str) -> None:
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    lifted = [torch.tensor([2, 3])]
    base = run_steering_pass(
        model,
        _events(),
        vocab,
        push=None,
        lifted_sets=lifted,
        tables=None,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
    )
    push = SteeringPush(concept_index=0, gamma=2.0, site=site, layer_index=0)
    steered = run_steering_pass(
        model,
        _events(),
        vocab,
        push=push,
        lifted_sets=lifted,
        tables=None,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
    )
    assert set(steered) == set(base)
    if site == "bottleneck":
        # the bottleneck sum moves by gamma e_c, so the hazard heads see
        # different features and the risks move
        assert not np.allclose(steered[1].risk_means(), base[1].risk_means())
    else:
        # the identity block's hook adds the vector after the GRU; the
        # plumbing ran and produced finite readouts
        assert np.isfinite(steered[1].risk_means()).all()


def test_outcome_risk_needs_hazard_heads() -> None:
    vocab = _vocab()
    with pytest.raises(ValueError, match="hazard heads"):
        _outcome_risk(
            _model(len(vocab.token_to_id), with_heads=False), torch.zeros(2, 8)
        )


def test_readout_accumulates_across_chunks() -> None:
    r = SubjectReadout()
    probs = torch.tensor([[0.2, 0.8]])
    mass = torch.tensor([[0.5]])
    risk = torch.tensor([[[0.1, 0.2, 0.3]]])
    at_risk = torch.tensor([[True]])
    r.add(probs, mass, risk, at_risk)
    r.add(probs, mass, risk, torch.tensor([[False]]))
    assert r.n == 2
    assert r.risk_n.tolist() == [1.0]
    assert np.allclose(r.risk_means(), [[0.1, 0.2, 0.3]])
    assert np.allclose(r.concept_probs / r.n, [0.2, 0.8])


def test_evaluate_steering_runs_every_expected_dial_both_ways_and_serializes() -> None:
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    prepared = _prepared(model, vocab)
    progress: list[int] = []
    summaries = evaluate_steering(
        prepared,
        concepts=None,
        site="bottleneck",
        layer_index=None,
        suppress_strength=None,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        n_boot=20,
        on_progress=lambda done: progress.append(len(done)),
    )
    expected = [n for n in NAMES if n in CLINICAL_EXPECTATIONS]
    assert [s.concept for s in summaries] == [n for n in expected for _ in range(2)]
    # Called once per dial with the summaries so far, so a stopped run keeps them.
    assert progress == [2 * (i + 1) for i in range(len(expected))]
    assert [s.direction for s in summaries[:2]] == ["amplify", "suppress"]
    assert summaries[0].gamma > 0 > summaries[1].gamma
    assert summaries[0].n_subjects == 4
    for s in summaries:
        assert len(s.outcomes) == len(EVENTS) * len(HORIZONS_HOURS)
        assert "k_c" in clinician_line(s)
    payload = _to_json(summaries)
    assert len(payload) == len(summaries)
    first = payload[0]
    assert {"concept", "direction", "gamma", "outcomes", "sign_agreement"} <= set(first)
    assert {"relative_change", "as_expected", "separated"} <= set(first["outcomes"][0])


def test_evaluate_steering_rejects_unknown_concepts_and_honours_a_selection() -> None:
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    prepared = _prepared(model, vocab)
    with pytest.raises(ValueError, match="not in this run's registry"):
        evaluate_steering(
            prepared,
            concepts=["sepsis3"],
            site="bottleneck",
            layer_index=None,
            suppress_strength=None,
            num_lanes=2,
            chunk_size=8,
            device="cpu",
            n_boot=10,
        )
    summaries = evaluate_steering(
        prepared,
        concepts=["fever"],
        site="stream",
        layer_index=0,
        suppress_strength=0.3,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        n_boot=10,
    )
    assert [s.concept for s in summaries] == ["fever", "fever"]
    assert summaries[0].site == "stream"


def test_labels_for_builds_stay_and_visit_dictionaries() -> None:
    concepts = concepts_for_source("mimic_iv", task_set="v1")
    raw = _events()
    for supervision in ("stay", "visit"):
        labels, mask, first = _labels_for(raw, concepts, supervision)
        assert len(labels) == len(mask) == len(first)
        row = next(iter(labels.values()))
        assert row.shape[-1] == len(concepts)


def _strata() -> PositionStrata:
    """Concept 0 has triggered by hour 3 for subjects 1 and 2, never for 3 and 4."""
    inf = float("inf")
    return PositionStrata(
        name="tachycardia",
        concept_index=0,
        labels={
            1: torch.tensor([1.0, 0.0, 0.0]),
            2: torch.tensor([1.0, 0.0, 0.0]),
            3: torch.zeros(3),
            4: torch.zeros(3),
        },
        mask={sid: torch.ones(3) for sid in (1, 2, 3, 4)},
        first_times={
            1: torch.tensor([3.0, inf, inf]),
            2: torch.tensor([3.0, inf, inf]),
            3: torch.full((3,), inf),
            4: torch.full((3,), inf),
        },
        supervision="stay",
        num_concepts=3,
    )


def test_stratified_pass_splits_each_subject_by_the_running_label() -> None:
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    readouts = run_steering_pass(
        model,
        _events(),
        vocab,
        push=None,
        lifted_sets=[torch.tensor([2])],
        tables=None,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        strata=_strata(),
    )
    assert set(readouts) == {(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 0)}
    # subject 1: ten events, nine targets; the concept triggers at hour 3, so
    # three positions (hours 0-2) sit before it and six on or after it
    assert readouts[(1, 0)].n == 3 and readouts[(1, 1)].n == 6
    assert readouts[(3, 0)].n == 9


def test_evaluate_steering_reports_one_summary_per_stratum() -> None:
    vocab = _vocab()
    model = _model(len(vocab.token_to_id))
    prepared = _prepared(model, vocab)
    prepared.strata = _strata()
    summaries = evaluate_steering(
        prepared,
        concepts=["fever"],
        site="bottleneck",
        layer_index=None,
        suppress_strength=None,
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        n_boot=10,
        stratify=prepared.strata,
    )
    assert [(s.direction, s.stratum) for s in summaries] == [
        ("amplify", "tachycardia=0"),
        ("amplify", "tachycardia=1"),
        ("suppress", "tachycardia=0"),
        ("suppress", "tachycardia=1"),
    ]
    assert summaries[0].n_subjects == 4 and summaries[1].n_subjects == 2
    assert "[tachycardia=1]" in clinician_line(summaries[1])
    assert _to_json(summaries)[1]["stratum"] == "tachycardia=1"
