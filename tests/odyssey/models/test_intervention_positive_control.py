"""Positive control for the concept-override path.

The paper's headline is a NULL -- overriding a concept does not move the
forecast in the direction the concept implies. A null result is only worth
reporting if the machinery that produced it demonstrably works, so this
module builds a case where the concept \\emph{provably} determines the next
token and asserts that the override flips it.

Construction: one concept whose pole embeddings are set by hand to $+u$ and
$-u$, and a language-model head that reads exactly that concept's slot,
scoring token A by $z_0 \\cdot u$ and token B by $-z_0 \\cdot u$. Under
Equation (1) of the paper, $z_0 = p_0 w_0^{+} + (1-p_0) w_0^{-}$, so forcing
$p_0=1$ must produce token A and $p_0=0$ must produce token B. Nothing is
learned and nothing is stochastic; if these fail, the intervention path is
broken and the paper's null is an artifact.

If instead they pass, the write path is sound end to end (override ->
mixture -> bottleneck -> logits -> argmax) and the null measured on real
data is a property of the trained model rather than of the harness. That is
the discrimination these tests exist to make.
"""

import torch

from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    ConceptBottleneck,
)


HIDDEN, K, D, U = 6, 2, 4, 2
DIRECTION = torch.tensor([1.0, -0.5, 0.25, 2.0])


def _bottleneck(global_pairs: bool) -> ConceptBottleneck:
    """Build a deterministic bottleneck whose concept 0 has hand-set poles."""
    torch.manual_seed(0)
    bn = ConceptBottleneck(
        hidden_size=HIDDEN,
        num_concepts=K,
        embedding_dim=D,
        concept_dropout=0.0,
        global_pairs=global_pairs,
        unknown_dim=U,
    )
    bn.eval()
    if global_pairs:
        with torch.no_grad():
            bn.pair_embeddings[0, 0, :] = DIRECTION
            bn.pair_embeddings[0, 1, :] = -DIRECTION
    return bn


def _head() -> torch.Tensor:
    """(2, k*D+U) head reading ONLY concept 0's slot: token A is +u, B is -u."""
    w = torch.zeros(2, K * D + U)
    w[0, 0:D] = DIRECTION
    w[1, 0:D] = -DIRECTION
    return w


def _forced(bn: ConceptBottleneck, h: torch.Tensor, value: float) -> torch.Tensor:
    """Bottleneck output with concept 0 forced to ``value`` at every position."""
    probs = torch.full((*h.shape[:-1], K), value)
    mask = torch.zeros((*h.shape[:-1], K), dtype=torch.bool)
    mask[..., 0] = True  # override concept 0 only
    out = bn(h, intervention=BottleneckIntervention(probs=probs, probs_mask=mask))
    return out.bottleneck


def test_forced_concept_flips_the_predicted_token() -> None:
    """The decisive control: p_0=1 must give token A, p_0=0 must give token B.

    This is the test that separates "the lever is inert in the trained
    model" from "the override never reaches the logits". A failure here
    would invalidate the paper's null.
    """
    bn = _bottleneck(global_pairs=True)
    head = _head()
    h = torch.randn(3, HIDDEN)

    logits_true = _forced(bn, h, 1.0) @ head.T
    logits_false = _forced(bn, h, 0.0) @ head.T

    assert torch.all(logits_true.argmax(dim=-1) == 0), "p=1 must select token A"
    assert torch.all(logits_false.argmax(dim=-1) == 1), "p=0 must select token B"
    # and the effect is a genuine sign reversal, not a tie broken by noise
    assert torch.all(logits_true[:, 0] > logits_true[:, 1])
    assert torch.all(logits_false[:, 1] > logits_false[:, 0])


def test_forced_concept_matches_equation_one_exactly() -> None:
    """z_0 = p w+ + (1-p) w- for the forced p, to floating-point tolerance.

    Pins the paper's Equation (1) against the implementation, so the printed
    equation cannot drift from the code that produced the results.
    """
    bn = _bottleneck(global_pairs=True)
    h = torch.randn(4, HIDDEN)
    for value in (0.0, 0.25, 0.5, 1.0):
        z0 = _forced(bn, h, value)[..., 0:D]
        expected = value * DIRECTION + (1.0 - value) * (-DIRECTION)
        assert torch.allclose(z0, expected.expand_as(z0), atol=1e-6), value


def test_override_reaches_the_bottleneck_with_context_pairs_too() -> None:
    """The default (context-dependent) bottleneck also propagates the write.

    Here the poles are functions of the hidden state, so the override cannot
    be checked against hand-set directions. What CAN be checked is that the
    forced value changes the bottleneck at all: the paper attributes the null
    to the poles carrying context rather than to the override being dropped,
    and that attribution requires the write to land.
    """
    bn = _bottleneck(global_pairs=False)
    h = torch.randn(3, HIDDEN)
    z_high = _forced(bn, h, 1.0)[..., 0:D]
    z_low = _forced(bn, h, 0.0)[..., 0:D]
    assert not torch.allclose(z_high, z_low), "override did not reach the bottleneck"


def test_unforced_concepts_are_left_alone() -> None:
    """Concept 1 is untouched when only concept 0 is overridden.

    Guards the probs_mask path: an override that silently rewrote every
    concept would make every intervention result meaningless.
    """
    bn = _bottleneck(global_pairs=True)
    h = torch.randn(3, HIDDEN)
    baseline = bn(h).bottleneck[..., D : 2 * D]
    forced = _forced(bn, h, 1.0)[..., D : 2 * D]
    assert torch.allclose(baseline, forced, atol=1e-6)
