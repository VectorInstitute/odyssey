"""The benchmark must push the way Steerling pushes and score as a clinician reads.

Each test pins one definition from Madsen et al. (2026) Section 6.2 or one
piece of the clinical scoring: the unit direction and Eq. 19 calibration,
the accumulating layer injection of Eq. 18, the ReLU-gated suppression of
Eq. 21, the exact ``gamma e_c`` displacement at the bottleneck site, and
the paired, subject-level summary against declared expectations.
"""

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from odyssey.data.concepts import concepts_for_source
from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.inference.steering import (
    CLINICAL_EXPECTATIONS,
    HORIZONS_HOURS,
    SteeringPush,
    SubjectReadout,
    _forward_pushed,
    expectations_for,
    paired_delta,
    summarize_push,
    token_descriptions,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.steering import (
    concept_alignment,
    steering_direction,
    steering_gamma,
    stream_injection,
    suppress_logits,
)
from odyssey.training.lifted_tokens import rank_by_lift


HIDDEN, K, VOCAB = 8, 3, 11
EVENTS = ["death", "icu_admission"]


def _model() -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=VOCAB, hidden_size=HIDDEN, padding_idx=0),
        vocab_size=VOCAB,
        num_concepts=K,
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind="decomposed",
        event_names=EVENTS,
    )
    model.eval()
    return model


def _chunk() -> StreamingChunk:
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.tensor([[3, 5, 7]]),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(1, 3, dtype=torch.long),
                time_stamps=torch.tensor([[0.0, 1.0, 2.0]]),
                ages=torch.full((1, 3), 40.0),
                visit_orders=torch.zeros(1, 3, dtype=torch.long),
                visit_segments=torch.zeros(1, 3, dtype=torch.long),
            ),
        ),
        targets=torch.tensor([[4, 6, 8]]),
        reset_mask=torch.tensor([[True, False, False]]),
        real_mask=torch.tensor([[True, True, True]]),
        subject_ids=torch.tensor([[1, 1, 1]]),
        patient_end=torch.tensor([[False, False, True]]),
        visit_ids=torch.tensor([[-1, -1, -1]]),
        visit_end=torch.tensor([[False, False, False]]),
    )


# --- the clinical table -----------------------------------------------------


def test_every_expectation_names_a_registry_concept() -> None:
    """A dial that does not exist cannot be turned; catch typos at import time."""
    registry = {c.name for c in concepts_for_source("mimic_iv", task_set="v3")}
    assert set(CLINICAL_EXPECTATIONS) <= registry, sorted(
        set(CLINICAL_EXPECTATIONS) - registry
    )
    for concept, expected in CLINICAL_EXPECTATIONS.items():
        assert expected, concept
        assert set(expected.values()) <= {+1, -1}, concept


def test_expectations_for_drops_events_the_model_has_no_head_for() -> None:
    """Sepsis-3 has no head on eICU: out of scope for that source, not failed."""
    assert expectations_for("sirs", ["death", "icu_admission"]) == {
        "icu_admission": +1,
        "death": +1,
    }
    assert expectations_for("not a concept", EVENTS) == {}


# --- direction, calibration, suppression (Steerling 6.2) --------------------


def test_direction_is_unit_normalized_and_gamma_follows_eq19() -> None:
    model = _model()
    e = steering_direction(model, 1)
    assert torch.allclose(e.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(
        e,
        model.bottleneck.known_embeddings[1]
        / model.bottleneck.known_embeddings[1].norm(),
    )
    peak = float((model.lm_head.weight.detach() @ e).max())
    assert steering_gamma(model, e, tau=2.0) == pytest.approx(2.0 / peak)
    with pytest.raises(ValueError, match="positive"):
        steering_gamma(model, e, tau=0.0)


def test_suppression_gates_on_positive_alignment() -> None:
    """Eq. 21: aligned tokens are pushed down, anti-aligned tokens untouched."""
    logits = torch.zeros(2, 4)
    alignment = torch.tensor([1.0, -1.0, 0.5, 0.0])
    out = suppress_logits(logits, alignment, 2.0)
    assert torch.allclose(out[0], torch.tensor([-2.0, 0.0, -1.0, 0.0]))
    model = _model()
    e = steering_direction(model, 0)
    assert torch.allclose(
        concept_alignment(model, e), model.lm_head.weight.detach() @ e
    )


# --- layer injection (Eq. 18) -----------------------------------------------


class _Block(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x


class _Backbone(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])

    def run(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for layer in self.layers:
            x = layer(x)
        return x


def test_stream_injection_accumulates_from_the_layer_on_and_removes_its_hooks() -> None:
    backbone = _Backbone(4)
    x = torch.zeros(2, 3)
    vec = torch.tensor([1.0, 2.0, 3.0])
    with stream_injection(backbone, 1, vec):
        pushed = backbone.run(x)
    # Layers 1, 2, 3 each add the vector once: three copies reach the output.
    assert torch.allclose(pushed, 3 * vec.expand(2, 3))
    assert torch.equal(backbone.run(x), x)  # hooks gone


def test_stream_injection_rejects_unsuitable_backbones() -> None:
    with (
        pytest.raises(TypeError, match="layers"),
        stream_injection(nn.Linear(2, 2), 0, torch.zeros(2)),
    ):
        pass
    with pytest.raises(IndexError), stream_injection(_Backbone(2), 5, torch.zeros(3)):
        pass


# --- bottleneck-site push -----------------------------------------------------


def test_bottleneck_push_displaces_the_sum_by_exactly_gamma_e_c() -> None:
    model = _model()
    chunk = _chunk()
    _, _, base_features, _ = _forward_pushed(model, chunk, None, None)
    gamma = 0.7
    push = SteeringPush(concept_index=2, gamma=gamma, site="bottleneck")
    _, _, features, _ = _forward_pushed(model, chunk, None, push)
    expected = base_features + gamma * steering_direction(model, 2)
    assert torch.allclose(features, expected, atol=1e-5)


def test_negative_gamma_applies_the_relu_mask_to_the_logits() -> None:
    model = _model()
    chunk = _chunk()
    push = SteeringPush(concept_index=0, gamma=-0.5, site="bottleneck")
    logits, _, features, _ = _forward_pushed(model, chunk, None, push)
    unmasked = model.lm_head(features)
    e = steering_direction(model, 0)
    assert torch.allclose(
        logits, unmasked - 0.5 * torch.relu(concept_alignment(model, e)), atol=1e-5
    )


# --- summaries ---------------------------------------------------------------


def test_paired_delta_brackets_a_constant_shift() -> None:
    base = np.zeros(50)
    steered = np.full(50, 0.2)
    d = paired_delta(steered, base, n_boot=200)
    assert d.point == pytest.approx(0.2)
    assert d.ci_low == pytest.approx(0.2) and d.ci_high == pytest.approx(0.2)
    assert d.separated
    assert not paired_delta(base, base, n_boot=50).separated
    assert paired_delta(np.zeros(0), np.zeros(0)).n_subjects == 0


def _readouts(risk_death: float, k_shock: float) -> dict[int, SubjectReadout]:
    out: dict[int, SubjectReadout] = {}
    for sid in (1, 2, 3):
        r = SubjectReadout()
        probs = torch.tensor([[k_shock, 0.1, 0.1]] * 2)
        risk = torch.zeros(2, len(EVENTS), len(HORIZONS_HOURS))
        risk[:, 0, :] = risk_death
        risk[:, 1, :] = 0.3
        at_risk = torch.ones(2, len(EVENTS), dtype=torch.bool)
        r.add(probs, torch.full((2,), 0.05), risk, at_risk)
        out[sid] = r
    return out


def test_readout_counts_risk_only_where_the_patient_is_at_risk() -> None:
    """A patient already on pressors is not asked whether pressors will start."""
    r = SubjectReadout()
    probs = torch.zeros(3, K)
    risk = torch.tensor(
        [[[0.9], [0.1]], [[0.9], [0.1]], [[0.1], [0.1]]]
    )  # (N=3,E=2,H=1)
    at_risk = torch.tensor([[True, True], [False, True], [False, True]])
    r.add(probs, torch.zeros(3), risk, at_risk)
    means = r.risk_means()
    assert means[0, 0] == pytest.approx(0.9)  # only the first position counted
    assert means[1, 0] == pytest.approx(0.1)
    never = SubjectReadout()
    never.add(probs, torch.zeros(3), risk, torch.zeros(3, 2, dtype=torch.bool))
    assert np.isnan(never.risk_means()).all()
    # Subjects never at risk drop out of the paired delta rather than poisoning it.
    d = paired_delta(np.array([0.2, np.nan, 0.4]), np.array([0.1, 0.5, 0.3]), n_boot=20)
    assert d.n_subjects == 2 and d.point == pytest.approx(0.1)


def test_token_descriptions_use_meds_metadata_and_keep_the_bin(tmp_path) -> None:
    (tmp_path / "codes.parquet").touch()
    pl.DataFrame(
        {
            "code": ["LAB//50813//mmol/L", "MEDICATION//norepinephrine"],
            "description": ["Lactate", None],
        }
    ).write_parquet(tmp_path / "codes.parquet")
    tokens = ["LAB//50813//mmol/L::Q5", "MEDICATION//norepinephrine", "OTHER//x"]
    names = token_descriptions(tokens, tmp_path)
    assert names["LAB//50813//mmol/L::Q5"] == "Lactate (Q5)"
    assert names["MEDICATION//norepinephrine"] == "MEDICATION//norepinephrine"
    assert names["OTHER//x"] == "OTHER//x"
    assert token_descriptions(tokens, None) == {t: t for t in tokens}


def test_summary_scores_outcomes_against_the_declared_direction() -> None:
    baseline = _readouts(risk_death=0.10, k_shock=0.2)
    steered = _readouts(risk_death=0.15, k_shock=0.9)
    up = summarize_push(
        baseline,
        steered,
        concept="shock",
        concept_index=0,
        direction="amplify",
        gamma=1.0,
        site="bottleneck",
        event_names=EVENTS,
        n_boot=100,
    )
    assert up.n_subjects == 3
    assert up.respond_baseline == pytest.approx(0.2)
    assert up.respond_steered == pytest.approx(0.9)
    death = [o for o in up.outcomes if o.event == "death" and o.horizon_hours == 24.0][
        0
    ]
    assert death.as_expected is True
    assert death.relative_change == pytest.approx(1.5)
    assert death.agreement == pytest.approx(1.0)
    icu = [o for o in up.outcomes if o.event == "icu_admission"][0]
    assert icu.as_expected is False  # expected up, did not move
    assert up.sign_agreement == pytest.approx(0.5)

    down = summarize_push(
        baseline,
        steered,
        concept="shock",
        concept_index=0,
        direction="suppress",
        gamma=-1.0,
        site="bottleneck",
        event_names=EVENTS,
        n_boot=100,
    )
    # Suppression reverses the expectation: a rise in death risk is now wrong.
    assert [o for o in down.outcomes if o.event == "death"][0].as_expected is False


def test_rank_by_lift_applies_support_and_threshold() -> None:
    total = torch.tensor([100.0, 100.0, 100.0, 100.0])
    per_concept = torch.tensor([[50.0, 10.0, 25.0, 15.0]])
    # lifts: token 0 = 2.0, token 2 = 1.0 (excluded), token 3 = 0.6;
    # token 1 has 10 < 15 occurrences and is below support.
    got = rank_by_lift(total, per_concept, top_k=3, min_count=15)
    assert got == {0: [0]}
    # Token 0 has lift exactly 2.0: a strict min_lift of 2 excludes it too.
    assert rank_by_lift(total, per_concept, top_k=3, min_count=15, min_lift=2.0) == {
        0: []
    }
    # A share floor of 60% of the concept's 100 positions excludes token 0 too.
    assert rank_by_lift(total, per_concept, top_k=3, min_count=15, min_share=0.6) == {
        0: []
    }
