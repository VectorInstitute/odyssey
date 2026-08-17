"""The value channel in ClinicalEventEmbeddings."""

import torch

from odyssey.data.types import AuxiliaryInputs
from odyssey.models.embeddings import ClinicalEventEmbeddings, value_features


def _aux(values=None) -> AuxiliaryInputs:
    b, t = 2, 5
    return AuxiliaryInputs(
        type_ids=torch.zeros(b, t, dtype=torch.long),
        time_stamps=torch.arange(t, dtype=torch.float).repeat(b, 1),
        ages=torch.full((b, t), 50.0),
        visit_orders=torch.zeros(b, t, dtype=torch.long),
        visit_segments=torch.ones(b, t, dtype=torch.long),
        values=values,
    )


def test_value_features_mask_nan() -> None:
    v = torch.tensor([[1.5, float("nan")]])
    f = value_features(v)
    assert f.shape == (1, 2, 3)
    assert torch.allclose(f[0, 0], torch.tensor([1.5, 2.25, 1.0]))
    assert torch.allclose(f[0, 1], torch.tensor([0.0, 0.0, 0.0]))


def test_values_change_embeddings_only_when_enabled() -> None:
    torch.manual_seed(0)
    ids = torch.randint(1, 10, (2, 5))
    off = ClinicalEventEmbeddings(10, 16, 0, hidden_dropout_prob=0.0).eval()
    on = ClinicalEventEmbeddings(
        10, 16, 0, hidden_dropout_prob=0.0, use_values=True
    ).eval()
    v1 = torch.full((2, 5), 0.5)
    v2 = torch.full((2, 5), 2.0)
    assert torch.allclose(off(ids, _aux(v1)), off(ids, _aux(v2)))
    assert torch.allclose(off(ids, _aux(None)), off(ids, _aux(v1)))
    assert not torch.allclose(on(ids, _aux(v1)), on(ids, _aux(v2)))
    # NaN (no value) is a distinct, well-defined input: finite, differs from a value
    nan = torch.full((2, 5), float("nan"))
    out_nan = on(ids, _aux(nan))
    assert torch.isfinite(out_nan).all()
    assert not torch.allclose(out_nan, on(ids, _aux(v1)))
    # missing channel entirely behaves like all-NaN
    assert torch.allclose(out_nan, on(ids, _aux(None)))


def test_value_proj_only_exists_when_enabled() -> None:
    assert ClinicalEventEmbeddings(10, 16, 0).value_proj is None
    keys = ClinicalEventEmbeddings(10, 16, 0, use_values=True).state_dict().keys()
    assert any("value_proj" in k for k in keys)
