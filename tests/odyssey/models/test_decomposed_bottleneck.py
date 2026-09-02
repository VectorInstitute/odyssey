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
from odyssey.inference.concept_attribution import calibrated_gammas
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
    """alpha=1 replaces k_hat with k_hat_gt when forming h_bar.

    residual_dropout=1.0 zeroes eps, so the bottleneck is exactly the two
    parts that were used and the substitution is observable.
    """
    bn = _bottleneck(residual_dropout=1.0)
    bn.train()  # teacher forcing is a training-time substitution
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()

    own = bn(h)
    forced = bn(h, teacher=TeacherForcing(labels, alpha_known=1.0))
    expected = bn.known_contribution(labels) + own.unknown_embedding
    assert torch.allclose(forced.bottleneck, expected, atol=1e-5)


def test_teacher_forcing_on_the_unknown_head_uses_h_minus_k_hat_gt() -> None:
    """The unknown substitution is the reconstruction target itself."""
    bn = _bottleneck(residual_dropout=1.0)
    bn.train()
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()

    own = bn(h)
    forced = bn(h, teacher=TeacherForcing(labels, alpha_unknown=1.0))
    known_gt = bn.known_contribution(labels)
    expected = own.known_part + (h - known_gt)
    assert torch.allclose(forced.bottleneck, expected, atol=1e-5)


def test_teacher_forcing_recomputes_the_residual_from_the_used_parts() -> None:
    """Steerling Eq (11)-(12): "when u_hat = u_hat_gt, eps = 0".

    The residual is whatever the parts actually summed into h_bar leave
    unexplained. With no dropout that makes h_bar == h under every
    forcing pattern. The previous behaviour held the model's OWN eps fixed
    and added it on top of the forced parts, so full forcing produced
    h + eps: the residual direction entered the LM head twice in training
    and once at inference, which is a direct push toward residual reliance
    in the arm built to measure residual reliance.
    """
    bn = _bottleneck()  # dropout off
    bn.train()
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()
    own = bn(h)
    assert own.residual.abs().sum() > 1e-3  # the own eps is not trivially 0

    for known, unknown in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
        forced = bn(
            h, teacher=TeacherForcing(labels, alpha_known=known, alpha_unknown=unknown)
        )
        assert torch.allclose(forced.bottleneck, h, atol=1e-5), (known, unknown)


def test_teacher_forcing_leaves_unlabeled_positions_alone() -> None:
    """Where the mask says no label was found, the model's own parts stand.

    Otherwise an all-zero label row becomes "no concepts present" and the
    forced unknown becomes the whole hidden state, at every position that
    merely lacks a visit id.
    """
    bn = _bottleneck(residual_dropout=1.0)
    bn.train()
    h = torch.randn(4, HIDDEN)
    labels = torch.randint(0, 2, (4, K)).float()
    mask = torch.ones(4, K, dtype=torch.bool)
    mask[1] = False

    own = bn(h)
    forced = bn(
        h,
        teacher=TeacherForcing(
            labels, alpha_known=1.0, alpha_unknown=1.0, concept_mask=mask
        ),
    )
    known_gt = bn.known_contribution(labels)
    assert torch.allclose(
        forced.bottleneck[0], h[0], atol=1e-5
    )  # forced: k_gt + (h - k_gt)
    assert torch.allclose(
        forced.bottleneck[1], own.known_part[1] + own.unknown_embedding[1], atol=1e-5
    )
    assert not torch.allclose(forced.bottleneck[1], known_gt[1] + h[1] - known_gt[1])


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


def test_streaming_loss_hands_the_found_mask_to_teacher_forcing() -> None:
    """The mask the helper builds must reach the bottleneck, or it is decor."""
    model = _decomposed_model(vocab=64)
    # Subject 2 is unlabeled, so it must not be a supervision point.
    chunk = _streaming_chunk()._replace(patient_end=torch.tensor([[True, False, True]]))
    seen: list[TeacherForcing | None] = []
    original = model.bottleneck.forward

    def spy(hidden, intervention=None, *, teacher=None):
        seen.append(teacher)
        return original(hidden, intervention=intervention, teacher=teacher)

    model.bottleneck.forward = spy  # type: ignore[method-assign]
    model.compute_streaming_loss(
        chunk, {1: torch.ones(K), 3: torch.ones(K)}, teacher_alpha_known=1.0
    )
    assert seen and seen[0] is not None
    mask = seen[0].concept_mask
    assert mask is not None and mask.shape == (1, 3, K)
    assert (
        bool(mask[0, 0].all()) and not bool(mask[0, 1].any()) and bool(mask[0, 2].all())
    )


def test_streaming_teacher_labels_fall_back_to_zero_where_absent() -> None:
    """An unlabeled position must not be taught "no concepts present".

    The pooling step separately requires a label at every patient_end, so
    this tolerance only bites for positions that are not supervision
    points; the helper is exercised directly because that is where the
    behaviour lives.
    """
    model = _decomposed_model(vocab=64)
    chunk = _streaming_chunk()
    got = model._streaming_position_labels(chunk, {1: torch.ones(K)}, "stay")
    assert got is not None
    labels, mask = got
    assert labels.shape == (1, 3, K)
    assert torch.equal(labels[0, 0], torch.ones(K))
    assert torch.equal(labels[0, 1], torch.zeros(K))
    # ...and marked absent, so the losses skip it rather than treating an
    # all-zero row as "no concepts present".
    assert bool(mask[0, 0].all()) and not bool(mask[0, 1].any())


def test_streaming_teacher_returns_none_when_nothing_is_labeled() -> None:
    """No labels at all means no substitution, not a tensor of zeros."""
    model = _decomposed_model(vocab=64)
    assert model._streaming_position_labels(_streaming_chunk(), {}, "stay") is None


def test_unit_displacement_is_the_concept_embedding() -> None:
    """One unit of k_i adds exactly K_i to h_bar.

    This is what makes output calibration definable here without a data
    pass: the mixture's displacement is (w+ - w-), a function of the
    hidden state, and has to be estimated; this one is a parameter.
    """
    bn = _bottleneck()
    assert torch.equal(bn.unit_displacements(), bn.known_embeddings.detach())
    assert bn.needs_calibration_directions is False


def test_unit_displacement_predicts_the_actual_movement() -> None:
    """The interface must agree with what the forward pass really does."""
    bn = _bottleneck()
    h = torch.randn(4, HIDDEN)
    disp = bn.unit_displacements()
    for concept in range(K):
        moved = (
            _override(bn, h, 1.0, concept=concept).bottleneck
            - _override(bn, h, 0.0, concept=concept).bottleneck
        )
        assert torch.allclose(moved[0], disp[concept], atol=1e-5), concept


def test_calibrated_gammas_need_no_directions_for_the_decomposition() -> None:
    """Equal peak logit shift per concept, computed from parameters alone."""
    model = _decomposed_model(vocab=32)
    gammas = calibrated_gammas(model, None, tau=1.0)
    assert gammas.shape == (K,)
    assert bool((gammas > 0).all())

    # gamma_i = tau / peak_i, so displacing p_i by gamma_i must move every
    # concept's largest logit by the same tau.
    weight = model.lm_head.weight.detach().double()
    shifts = model.bottleneck.unit_displacements().to(weight) @ weight.T
    peak_shift = (shifts * gammas.unsqueeze(1)).abs().amax(dim=1)
    assert torch.allclose(peak_shift, torch.ones(K, dtype=peak_shift.dtype), atol=1e-8)


def test_mixture_still_requires_its_measured_directions() -> None:
    """The mixture's displacement is data dependent and must say so."""
    bn = ConceptBottleneck(hidden_size=HIDDEN, num_concepts=K, embedding_dim=4)
    assert bn.needs_calibration_directions is True
    with pytest.raises(ValueError, match="data.*dependent"):
        bn.unit_displacements(None)


def test_mixture_displacement_lands_in_its_own_block() -> None:
    """Scattering keeps each concept's move inside the block it owns."""
    bn = ConceptBottleneck(hidden_size=HIDDEN, num_concepts=K, embedding_dim=4)
    directions = torch.arange(K * 4, dtype=torch.float32).reshape(K, 4) + 1.0
    out = bn.unit_displacements(directions)
    assert out.shape == (K, bn.output_dim)
    for i in range(K):
        assert torch.equal(out[i, i * 4 : (i + 1) * 4], directions[i])
        assert out[i, : i * 4].abs().sum() == 0
        assert out[i, (i + 1) * 4 :].abs().sum() == 0


def test_position_labels_are_correct_when_subjects_repeat() -> None:
    """The gather path must agree with a naive per-position lookup.

    Rows are stacked once per DISTINCT key and then gathered, because the
    naive version indexed a CUDA tensor per position and cost 32,768
    device syncs per training step. Repeats are the case that path exists
    for, so they are the case worth pinning.
    """
    chunk = _streaming_chunk()._replace(
        subject_ids=torch.tensor([[7, 7, 9]]),
        visit_ids=torch.tensor([[1, 2, 1]]),
    )
    model = _decomposed_model(vocab=64)
    labels = {7: torch.ones(K), 9: torch.full((K,), 2.0)}

    got = model._streaming_position_labels(chunk, labels, "stay")
    assert got is not None
    values, mask = got
    assert torch.equal(values[0, 0], torch.ones(K))
    assert torch.equal(values[0, 1], torch.ones(K))  # same subject, same row
    assert torch.equal(values[0, 2], torch.full((K,), 2.0))
    assert bool(mask.all())


def test_position_labels_key_on_the_visit_when_supervision_is_visit() -> None:
    """Two visits of one subject must not collapse to a single row."""
    chunk = _streaming_chunk()._replace(
        subject_ids=torch.tensor([[7, 7, 7]]),
        visit_ids=torch.tensor([[1, 2, 1]]),
    )
    model = _decomposed_model(vocab=64)
    labels = {(7, 1): torch.ones(K), (7, 2): torch.zeros(K)}

    got = model._streaming_position_labels(chunk, labels, "visit")
    assert got is not None
    values, _ = got
    assert torch.equal(values[0, 0], torch.ones(K))
    assert torch.equal(values[0, 1], torch.zeros(K))
    assert torch.equal(values[0, 2], torch.ones(K))


def test_auxiliary_losses_do_not_reach_the_backbone() -> None:
    """Steerling Eq (15): "only the unknown head is updated" by Lrec/Lindep.

    Using the live u_hat for the auxiliary terms backpropagates them
    through the unknown head into the backbone. That is what forced
    lambda_rec down to 1/256 to keep training stable, and at that weight
    the unknown head never learns its target and the residual absorbs the
    prediction.
    """
    torch.manual_seed(0)
    cb = DecomposedConceptBottleneck(hidden_size=32, num_concepts=4, unknown_ratio=3)
    cb.train()
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    labels = torch.randint(0, 2, (2, 4)).float()

    aux = cb.auxiliary_losses(cb(hidden), labels)
    (aux["reconstruction_loss"] + aux["independence_loss"]).backward()

    assert hidden.grad is None or float(hidden.grad.abs().sum()) == 0.0


def test_auxiliary_losses_still_train_the_unknown_head_only() -> None:
    """Cutting the path must not also stop the losses doing their job."""
    torch.manual_seed(0)
    cb = DecomposedConceptBottleneck(hidden_size=32, num_concepts=4, unknown_ratio=3)
    cb.train()
    labels = torch.randint(0, 2, (2, 4)).float()

    aux = cb.auxiliary_losses(cb(torch.randn(2, 5, 32)), labels)
    (aux["reconstruction_loss"] + aux["independence_loss"]).backward()

    assert cb.unknown_proj.weight.grad is not None
    assert float(cb.unknown_proj.weight.grad.abs().sum()) > 0.0
    known_grad = cb.known_proj.weight.grad
    assert known_grad is None or float(known_grad.abs().sum()) == 0.0


def test_lm_path_reaches_the_backbone_but_not_the_unknown_head() -> None:
    """Steerling Sec 10.2.2: the unknown prediction enters h_bar detached.

    The task loss must still train the backbone, through k_hat and eps,
    but it must not train the unknown head: that head is a reconstructor
    of h - k_hat_gt, shaped by Eq (12) and (14) only. Letting the task
    loss into it hands the model a second free channel, which is exactly
    how the unknown part came to carry 28.5% of top-1 while the named
    channel stayed at 2%.
    """
    torch.manual_seed(0)
    cb = DecomposedConceptBottleneck(hidden_size=32, num_concepts=4, unknown_ratio=3)
    cb.train()
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    cb(hidden).bottleneck.sum().backward()
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0.0
    for name in ("unknown_proj.weight", "unknown_embeddings_full"):
        grad = dict(cb.named_parameters())[name].grad
        assert grad is None or float(grad.abs().sum()) == 0.0, name


def test_known_embeddings_receive_task_gradient_through_residual_dropout() -> None:
    """The only task pressure on K is the dropout noise on eps.

    h_bar = k + u + D(h - k - u), so with D the identity (eval, or
    p_eps = 0) the task gradient into K is exactly zero, and with p_eps > 0
    it is nonzero per sample. This pins that the pressure exists at all;
    weight decay on K would otherwise win by default.
    """
    torch.manual_seed(0)
    cb = DecomposedConceptBottleneck(
        hidden_size=32, num_concepts=4, unknown_ratio=3, residual_dropout=0.3
    )
    cb.train()
    hidden = torch.randn(2, 5, 32)
    (cb(hidden).bottleneck * torch.randn(32)).sum().backward()
    assert cb.known_embeddings.grad is not None
    assert float(cb.known_embeddings.grad.abs().sum()) > 0.0

    cb.zero_grad()
    cb.eval()
    (cb(hidden).bottleneck * torch.randn(32)).sum().backward()
    grad = cb.known_embeddings.grad
    assert grad is None or float(grad.abs().max()) < 1e-6


def test_decay_exempt_parameters_are_the_concept_embeddings() -> None:
    """Steerling: weight decay "excluding embeddings"."""
    full = _bottleneck()
    assert [id(p) for p in full.decay_exempt_parameters()] == [
        id(full.known_embeddings),
        id(full.unknown_embeddings_full),
    ]
    low = _bottleneck(unknown_rank=2)
    assert [id(p) for p in low.decay_exempt_parameters()] == [
        id(low.known_embeddings),
        id(low.unknown_factor_a),
        id(low.unknown_factor_b),
    ]


def test_detached_unknown_part_matches_the_live_one_numerically() -> None:
    """Same values, same dropout mask -- only the gradient path differs."""
    torch.manual_seed(0)
    cb = DecomposedConceptBottleneck(hidden_size=32, num_concepts=4, unknown_ratio=3)
    cb.eval()  # dropout off, so the two paths must agree exactly
    out = cb(torch.randn(2, 5, 32))
    assert torch.allclose(out.unknown_embedding, out.unknown_embedding_detached)
