"""Weight decay must not touch the decomposed bottleneck's concept embeddings.

Steerling decays "excluding embeddings". The known embeddings receive no
task gradient in expectation, so decaying them is a steady shrink toward
an inert named channel; the v11 decomposed arms were trained that way.
"""

import torch
from torch import nn

from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.train import optimizer_param_groups


def _decomposed() -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=16, hidden_size=8, padding_idx=0),
        vocab_size=16,
        num_concepts=3,
        embedding_dim=4,
        padding_idx=0,
        bottleneck_kind="decomposed",
    )


def test_concept_embeddings_land_in_the_undecayed_group() -> None:
    model = _decomposed()
    groups = optimizer_param_groups(model, 0.01)
    assert [g["weight_decay"] for g in groups] == [0.01, 0.0]
    exempt = {id(p) for p in groups[1]["params"]}
    bn = model.bottleneck
    assert exempt == {id(bn.known_embeddings), id(bn.unknown_embeddings_full)}
    decayed = {id(p) for p in groups[0]["params"]}
    assert not (decayed & exempt)
    assert decayed | exempt == {id(p) for p in model.parameters() if p.requires_grad}


def test_optimizer_accepts_the_groups() -> None:
    model = _decomposed()
    opt = torch.optim.AdamW(optimizer_param_groups(model, 0.01), lr=1e-3)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0]


def test_models_without_exemptions_keep_one_group() -> None:
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(vocab_size=16, hidden_size=8, padding_idx=0),
        vocab_size=16,
        padding_idx=0,
    )
    groups = optimizer_param_groups(model, 0.05)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.05
    assert isinstance(model, nn.Module)
