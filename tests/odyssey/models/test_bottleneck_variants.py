"""Leakage-control variants of the concept bottleneck (global pairs, unknown width)."""

import torch

from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    ConceptBottleneck,
    orthogonality_loss,
)


def test_default_layout_is_checkpoint_compatible() -> None:
    """The default configuration keeps the original parameter names and shapes."""
    cb = ConceptBottleneck(hidden_size=16, num_concepts=3, embedding_dim=4)
    sd = cb.state_dict()
    assert sd["context_proj.weight"].shape == (4 * 2 * 4, 16)  # num_slots * 2 * d
    assert sd["prob_weight"].shape == (4, 8) and sd["prob_bias"].shape == (4,)
    assert "unknown_prob_weight" not in sd and "pair_embeddings" not in sd
    assert cb.output_dim == 4 * 4
    out = cb(torch.randn(2, 5, 16))
    assert out.bottleneck.shape == (2, 5, 16)


def test_global_pairs_make_the_concept_slot_a_pure_function_of_c() -> None:
    cb = ConceptBottleneck(
        hidden_size=16, num_concepts=3, embedding_dim=4, global_pairs=True
    ).eval()
    x = torch.randn(2, 5, 16)
    ones = BottleneckIntervention(probs=torch.ones(3))
    zeros = BottleneckIntervention(probs=torch.zeros(3))
    on = cb(x, intervention=ones)
    off = cb(x, intervention=zeros)
    w_pos, w_neg = cb.pair_embeddings[:, 0], cb.pair_embeddings[:, 1]
    # with c forced to 1 / 0 the known slots equal the global pair vectors,
    # regardless of the input: no context can leak through them
    assert torch.allclose(on.concept_embeddings, w_pos.expand(2, 5, 3, 4), atol=1e-6)
    assert torch.allclose(off.concept_embeddings, w_neg.expand(2, 5, 3, 4), atol=1e-6)
    # the unknown slot still depends on the input
    assert not torch.allclose(on.unknown_embedding[0, 0], on.unknown_embedding[0, 1])
    # the model's own probabilities are still predicted from the hidden state
    assert on.concept_probs.shape == (2, 5, 3) and (on.concept_probs != 1).any()


def test_unknown_width_and_zeroing() -> None:
    cb = ConceptBottleneck(
        hidden_size=16, num_concepts=3, embedding_dim=4, unknown_dim=2
    )
    assert cb.output_dim == 3 * 4 + 2
    x = torch.randn(2, 5, 16)
    out = cb(x)
    assert out.bottleneck.shape == (2, 5, 14) and out.unknown_embedding.shape == (
        2,
        5,
        2,
    )
    zk = cb(x, intervention=BottleneckIntervention(zero_known=True))
    assert (zk.concept_embeddings == 0).all() and not (zk.unknown_embedding == 0).all()
    zu = cb(x, intervention=BottleneckIntervention(zero_unknown=True))
    assert (zu.unknown_embedding == 0).all() and not (zu.concept_embeddings == 0).all()
    # orthogonality is undefined across widths and contributes nothing
    assert (
        orthogonality_loss(out.concept_embeddings, out.unknown_embedding).item() == 0.0
    )
    # global pairs with a narrow unknown slot
    cbg = ConceptBottleneck(
        hidden_size=16,
        num_concepts=3,
        embedding_dim=4,
        global_pairs=True,
        unknown_dim=2,
    )
    assert cbg(x).bottleneck.shape == (2, 5, 14)
