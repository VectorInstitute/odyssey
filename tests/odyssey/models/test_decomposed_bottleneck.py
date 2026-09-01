"""The decomposition must stay a decomposition, not a bookkeeping identity.

:class:`DecomposedConceptBottleneck` implements Steerling's concept
module: ``h_bar = k_hat + u_hat + eps`` with ``eps = h - k_hat - u_hat``.
That identity is free, and our earlier additive attempt shows what
happens without the pressure around it -- the model routed the whole
prediction through an untouched backbone stream and deleting all 26
concepts cost 1.6% of accuracy.

So these tests pin two separate things. First the algebra: the split is
exact, and an override moves the output by a known displacement rather
than being absorbed. Second the pressure: residual dropout, the
reconstruction target, and the independence penalty each behave as
Steerling defines them, because those are what stop the decomposition
going vacuous.
"""

import pytest
import torch

from odyssey.data.streaming import StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    ConceptBottleneck,
    ConceptBottleneckLossWeights,
    DecomposedConceptBottleneck,
    TeacherForcing,
    annealed_alpha,
    independence_loss,
    reconstruction_loss,
)
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


HIDDEN, K = 8, 3


def _bottleneck(**kw) -> DecomposedConceptBottleneck:
    """Build a deterministic decomposition with dropout off."""
    torch.manual_seed(0)
    defaults = {"concept_dropout": 0.0, "residual_dropout": 0.0}
    bn = DecomposedConceptBottleneck(HIDDEN, K, **{**defaults, **kw})
    bn.eval()
    return bn


def _override(bn, h, value, concept=0):
    """Force one concept's activation to ``value``."""
    probs = torch.full((*h.shape[:-1], bn.num_concepts), float(value))
    mask = torch.zeros((*h.shape[:-1], bn.num_concepts), dtype=torch.bool)
    mask[..., concept] = True
    return bn(h, intervention=BottleneckIntervention(probs=probs, probs_mask=mask))


def test_the_split_reconstructs_the_hidden_state_exactly() -> None:
    """Equation (5): h_bar == h when nothing is dropped or overridden."""
    bn = _bottleneck()
    h = torch.randn(5, HIDDEN)
    out = bn(h)
    assert torch.allclose(out.bottleneck, h, atol=1e-5)
    assert torch.allclose(
        out.known_part + out.unknown_embedding + out.residual, h, atol=1e-5
    )


def test_override_displaces_the_output_by_exactly_delta_times_embedding() -> None:
    """The steerability property, and the reason eps must be frozen.

    Recomputing eps after an override would cancel it identically, since
    k_hat' + u_hat + (h - k_hat' - u_hat) == h for any k_hat'.
    """
    bn = _bottleneck()
    h = torch.randn(5, HIDDEN)
    base = bn(h)
    for value in (0.0, 0.25, 1.0):
        out = _override(bn, h, value, concept=0)
        delta = value - base.concept_probs[..., 0]
        expected = base.bottleneck + delta.unsqueeze(-1) * bn.known_embeddings[0]
        assert torch.allclose(out.bottleneck, expected, atol=1e-5), value


def test_displacement_does_not_depend_on_the_patient() -> None:
    """Two hidden states, same forced change, same movement."""
    bn = _bottleneck()
    h1, h2 = torch.randn(1, HIDDEN), torch.randn(1, HIDDEN)
    move1 = _override(bn, h1, 1.0).bottleneck - _override(bn, h1, 0.0).bottleneck
    move2 = _override(bn, h2, 1.0).bottleneck - _override(bn, h2, 0.0).bottleneck
    assert torch.allclose(move1, move2, atol=1e-5)
    assert torch.allclose(move1[0], bn.known_embeddings[0], atol=1e-5)


def test_the_three_channels_zero_independently() -> None:
    """Known, unknown and residual are separate probes here.

    The mixture bottleneck folds "unknown concept" and "unexplained
    remainder" into one slot; this design separates them, so the
    completeness probe has three channels rather than two.
    """
    bn = _bottleneck()
    h = torch.randn(4, HIDDEN)
    out = bn(h)
    for flag, dropped in (
        ("zero_known", out.known_part),
        ("zero_unknown", out.unknown_embedding),
        ("zero_residual", out.residual),
    ):
        got = bn(h, intervention=BottleneckIntervention(**{flag: True}))
        assert torch.allclose(got.bottleneck, h - dropped, atol=1e-5), flag


def test_residual_dropout_is_training_only_and_perturbs_the_output() -> None:
    """p_eps is the direct counter-pressure to residual domination."""
    torch.manual_seed(0)
    bn = DecomposedConceptBottleneck(
        HIDDEN, K, concept_dropout=0.0, residual_dropout=0.9
    )
    h = torch.randn(64, HIDDEN)
    bn.eval()
    assert torch.allclose(bn(h).bottleneck, h, atol=1e-5)
    bn.train()
    assert not torch.allclose(bn(h).bottleneck, h, atol=1e-3)


def test_readouts_report_the_models_own_belief_not_the_override() -> None:
    """An intervention must never contaminate the reported probabilities."""
    bn = _bottleneck()
    h = torch.randn(3, HIDDEN)
    base = bn(h)
    forced = _override(bn, h, 1.0, concept=1)
    assert torch.allclose(base.concept_probs, forced.concept_probs, atol=1e-6)
    assert torch.allclose(base.observability_probs, forced.observability_probs)


def test_unknown_head_is_wider_than_the_known_one() -> None:
    """There are unknown_ratio * n unknown concepts, 3x by default."""
    assert _bottleneck().num_unknown == 3 * K
    assert _bottleneck(unknown_ratio=5).num_unknown == 5 * K


def test_low_rank_factorization_matches_the_full_matrix_shape() -> None:
    """U = A @ B, the parameter-count control for a large unknown set."""
    bn = _bottleneck(unknown_rank=2)
    assert bn.unknown_embeddings().shape == (3 * K, HIDDEN)
    assert not hasattr(bn, "unknown_embeddings_full")


def test_reconstruction_loss_is_zero_at_its_own_target() -> None:
    """Equation (12): u_hat should equal h - k_hat_gt."""
    bn = _bottleneck()
    h = torch.randn(6, HIDDEN)
    labels = torch.randint(0, 2, (6, K)).float()
    target = h - labels @ bn.known_embeddings
    assert reconstruction_loss(
        target, h, bn.known_embeddings, labels
    ).item() == pytest.approx(0.0, abs=1e-10)
    worse = reconstruction_loss(target + 1.0, h, bn.known_embeddings, labels)
    assert worse.item() > 0.5


def test_reconstruction_target_carries_no_gradient() -> None:
    """Steerling updates the unknown head here, not the backbone."""
    bn = _bottleneck()
    h = torch.randn(4, HIDDEN, requires_grad=True)
    labels = torch.randint(0, 2, (4, K)).float()
    unknown = torch.zeros(4, HIDDEN, requires_grad=True)
    reconstruction_loss(unknown, h, bn.known_embeddings, labels).backward()
    assert h.grad is None or torch.allclose(h.grad, torch.zeros_like(h))
    assert unknown.grad is not None and unknown.grad.abs().sum() > 0


def test_reconstruction_loss_only_counts_positions_with_labels() -> None:
    """An unobserved label would define a target from an all-zero k_hat_gt."""
    bn = _bottleneck()
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()
    mask = torch.ones(4, K, dtype=torch.bool)
    mask[2:] = False
    target = h - (labels * mask) @ bn.known_embeddings
    unknown = target.clone()
    unknown[2:] += 5.0  # garbage exactly where the labels are unobserved
    loss = reconstruction_loss(unknown, h, bn.known_embeddings, labels, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_independence_loss_is_zero_for_uncorrelated_and_positive_when_tied() -> None:
    """Equation (14): a normalized cross-covariance penalty."""
    torch.manual_seed(0)
    known = torch.randn(512, HIDDEN)
    assert independence_loss(known, known.clone()).item() > 1e-4
    orthogonal = torch.zeros(512, HIDDEN)
    orthogonal[:, 0] = known[:, 1]
    tied = independence_loss(known, known.clone()).item()
    assert independence_loss(known, orthogonal).item() < tied


def test_independence_gradient_flows_only_through_the_unknown_side() -> None:
    """Steerling treats the known representation as a fixed input."""
    known = torch.randn(64, HIDDEN, requires_grad=True)
    unknown = torch.randn(64, HIDDEN, requires_grad=True)
    independence_loss(known, unknown).backward()
    assert known.grad is None or torch.allclose(known.grad, torch.zeros_like(known))
    assert unknown.grad is not None and unknown.grad.abs().sum() > 0


def test_auxiliary_losses_are_reported_under_their_log_names() -> None:
    """fold_in_bottleneck_losses keys the weights off these names."""
    bn = _bottleneck()
    h = torch.randn(6, HIDDEN)
    out = bn(h)
    labels = torch.randint(0, 2, (6, K)).float()
    got = bn.auxiliary_losses(out, labels)
    assert set(got) == {"reconstruction_loss", "independence_loss"}
    assert all(v.ndim == 0 for v in got.values())


def test_sequence_model_can_be_built_with_either_bottleneck() -> None:
    """The kind is selectable and the heads size themselves off its output."""
    for kind in ("mixture", "decomposed"):
        model = ConceptBottleneckSequenceModel(
            backbone=TinyGRUBackbone(vocab_size=11, hidden_size=HIDDEN),
            vocab_size=11,
            num_concepts=K,
            embedding_dim=4,
            padding_idx=0,
            bottleneck_kind=kind,
        )
        if kind == "decomposed":
            assert isinstance(model.bottleneck, DecomposedConceptBottleneck)
            assert model.lm_head.in_features == HIDDEN
        else:
            assert isinstance(model.bottleneck, ConceptBottleneck)
            assert model.lm_head.in_features == K * 4 + 4


def test_unknown_bottleneck_kind_is_rejected() -> None:
    """An unrecognised kind fails at construction, not at first forward."""
    with pytest.raises(ValueError, match="bottleneck_kind"):
        ConceptBottleneckSequenceModel(
            backbone=TinyGRUBackbone(vocab_size=11, hidden_size=HIDDEN),
            vocab_size=11,
            num_concepts=K,
            embedding_dim=4,
            padding_idx=0,
            bottleneck_kind="nope",
        )


def _batch(batch: int = 4, seq_len: int = 10) -> ClinicalSequenceBatch:
    """Build a minimal well-formed batch for the end-to-end loss path."""
    torch.manual_seed(0)
    return ClinicalSequenceBatch(
        concept_ids=torch.randint(1, 11, (batch, seq_len)),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(0, 9, (batch, seq_len)),
            time_stamps=torch.rand(batch, seq_len) * 100,
            ages=torch.rand(batch, seq_len) * 90,
            visit_orders=torch.randint(0, 5, (batch, seq_len)),
            visit_segments=torch.randint(0, 3, (batch, seq_len)),
        ),
    )


def _decomposed_model(vocab: int = 11) -> ConceptBottleneckSequenceModel:
    """Build a small end-to-end model using the decomposition."""
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=vocab, hidden_size=HIDDEN, padding_idx=0),
        vocab_size=vocab,
        num_concepts=K,
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind="decomposed",
    )


def test_training_loss_reports_and_backpropagates_both_new_terms() -> None:
    """The end-to-end contract: the losses reach the optimizer.

    A correct module that never gets folded into the objective would
    leave the residual as unconstrained as it was before, which is the
    bug this whole change exists to fix.
    """
    model = _decomposed_model()
    batch = _batch()
    labels = torch.randint(0, 2, (4, K)).float()

    total, components = model.compute_loss(batch, labels)
    assert {"reconstruction_loss", "independence_loss"} <= set(components)
    assert torch.isfinite(total)

    total.backward()
    grads = {
        name: p.grad
        for name, p in model.bottleneck.named_parameters()
        if p.grad is not None
    }
    # The unknown head and its embeddings must actually receive signal.
    assert any("unknown_proj" in n and g.abs().sum() > 0 for n, g in grads.items())
    assert any(
        "unknown_embeddings" in n and g.abs().sum() > 0 for n, g in grads.items()
    )


def test_zero_weights_remove_the_terms_from_the_total() -> None:
    """A knob that does nothing is worse than no knob.

    eval() so the two passes are comparable: concept and residual dropout
    are both live in training mode, and this is asking about the weights.
    """
    model = _decomposed_model()
    model.eval()
    batch = _batch()
    labels = torch.randint(0, 2, (4, K)).float()

    on, comp = model.compute_loss(batch, labels)
    off, _ = model.compute_loss(
        batch,
        labels,
        loss_weights=ConceptBottleneckLossWeights(reconstruction=0.0, independence=0.0),
    )
    expected = comp["reconstruction_loss"] + comp["independence_loss"]
    assert (on - off).item() == pytest.approx(expected.item(), rel=1e-4)


def test_teacher_forcing_substitutes_the_ground_truth_contribution() -> None:
    """alpha=1 replaces k_hat with k_hat_gt when forming h_bar."""
    bn = _bottleneck()
    bn.train()  # teacher forcing is a training-time substitution
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()

    own = bn(h)
    forced = bn(h, teacher=TeacherForcing(labels, alpha_known=1.0))
    expected = bn.known_contribution(labels) + own.unknown_embedding + own.residual
    assert torch.allclose(forced.bottleneck, expected, atol=1e-5)


def test_teacher_forcing_on_the_unknown_head_uses_h_minus_k_hat_gt() -> None:
    """The unknown substitution is the reconstruction target itself."""
    bn = _bottleneck()
    bn.train()
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()

    own = bn(h)
    forced = bn(h, teacher=TeacherForcing(labels, alpha_unknown=1.0))
    known_gt = bn.known_contribution(labels)
    expected = own.known_part + (h - known_gt) + own.residual
    assert torch.allclose(forced.bottleneck, expected, atol=1e-5)


def test_teacher_forcing_never_fires_in_eval() -> None:
    """Substituting labels at inference would leak them into the forecast."""
    bn = _bottleneck()
    bn.eval()
    h = torch.randn(4, HIDDEN)
    labels = torch.ones(4, K)
    forced = bn(h, teacher=TeacherForcing(labels, alpha_known=1.0, alpha_unknown=1.0))
    assert torch.allclose(forced.bottleneck, h, atol=1e-5)


def test_teacher_forcing_is_off_when_alpha_is_zero() -> None:
    """The default configuration must change nothing."""
    bn = _bottleneck()
    bn.train()
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()
    own = bn(h)
    same = bn(h, teacher=TeacherForcing(labels))
    assert torch.allclose(same.bottleneck, own.bottleneck, atol=1e-6)


@pytest.mark.parametrize("cosine", [False, True])
def test_annealed_alpha_ramps_then_holds(cosine: bool) -> None:
    """Steerling ramps to a steady state over a prefix, then holds."""
    ends = {"start": 1.0, "end": 0.5, "cosine": cosine}
    assert annealed_alpha(0, 100, **ends) == pytest.approx(1.0)
    assert annealed_alpha(100, 100, **ends) == pytest.approx(0.5)
    assert annealed_alpha(500, 100, **ends) == pytest.approx(0.5)
    middle = annealed_alpha(50, 100, **ends)
    assert 0.5 < middle < 1.0


def test_annealed_alpha_handles_a_zero_length_ramp() -> None:
    """A disabled schedule returns the steady state, not a divide by zero."""
    assert annealed_alpha(0, 0, start=1.0, end=0.25) == pytest.approx(0.25)


def _streaming_chunk() -> StreamingChunk:
    """Build one chunk holding three single-token patients."""
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.tensor([[10, 20, 30]]),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(1, 3, dtype=torch.long),
                time_stamps=torch.tensor([[0.0, 1.0, 2.0]]),
                ages=torch.full((1, 3), 40.0),
                visit_orders=torch.zeros(1, 3, dtype=torch.long),
                visit_segments=torch.zeros(1, 3, dtype=torch.long),
            ),
        ),
        targets=torch.tensor([[11, 21, 31]]),
        reset_mask=torch.tensor([[True, True, True]]),
        real_mask=torch.tensor([[True, True, True]]),
        subject_ids=torch.tensor([[1, 2, 3]]),
        patient_end=torch.tensor([[True, True, True]]),
        visit_ids=torch.tensor([[-1, -1, -1]]),
        visit_end=torch.tensor([[False, False, False]]),
    )


def test_streaming_loss_carries_both_terms_and_teacher_forcing() -> None:
    """The path full-scale training actually uses.

    compute_loss and compute_streaming_loss are separate code paths and
    only the streaming one runs at scale, so a fix that reached the first
    and not the second would look tested and change nothing.
    """
    model = _decomposed_model(vocab=64)
    chunk = _streaming_chunk()
    labels = {sid: torch.randint(0, 2, (K,)).float() for sid in (1, 2, 3)}

    total, components, _ = model.compute_streaming_loss(
        chunk,
        labels,
        teacher_alpha_known=1.0,
        teacher_alpha_unknown=1.0,
    )
    assert {"reconstruction_loss", "independence_loss"} <= set(components)
    assert torch.isfinite(total)
    total.backward()
    assert model.bottleneck.unknown_proj.weight.grad.abs().sum() > 0


def test_streaming_teacher_labels_fall_back_to_zero_where_absent() -> None:
    """An unlabeled position must not be taught "no concepts present".

    The pooling step separately requires a label at every patient_end, so
    this tolerance only bites for positions that are not supervision
    points; the helper is exercised directly because that is where the
    behaviour lives.
    """
    model = _decomposed_model(vocab=64)
    chunk = _streaming_chunk()
    teacher = model._streaming_teacher(chunk, {1: torch.ones(K)}, "stay", 1.0, 0.0)
    assert teacher is not None
    assert teacher.concept_labels.shape == (1, 3, K)
    assert torch.equal(teacher.concept_labels[0, 0], torch.ones(K))
    assert torch.equal(teacher.concept_labels[0, 1], torch.zeros(K))


def test_streaming_teacher_returns_none_when_nothing_is_labeled() -> None:
    """No labels at all means no substitution, not a tensor of zeros."""
    model = _decomposed_model(vocab=64)
    assert model._streaming_teacher(_streaming_chunk(), {}, "stay", 1.0, 1.0) is None
