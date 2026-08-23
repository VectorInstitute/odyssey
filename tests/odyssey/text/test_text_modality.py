"""Text modality probe: note features from the embeddings sidecar, embedder helpers."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from odyssey.data.sidecars import sidecar_context
from odyssey.inference.alerts import IndexRow, features_for_events
from odyssey.text.embed_notes import (
    PCA_COL,
    _windows,
    apply_pca,
    embed_texts,
    fit_pca,
    pool_last_hidden,
)
from odyssey.text.note_features import (
    NOTE_EMBEDDINGS,
    NoteFeatureBuilder,
    active_note_embeddings,
    note_feature_names,
    note_features_for_rows,
)


T0 = datetime(2024, 1, 1)


def _events() -> pl.DataFrame:
    rows = []
    for sid in (1, 2):
        for h in range(0, 48, 4):
            rows.append(
                (sid, "LAB//220045//bpm", T0 + timedelta(hours=h), 80.0, 10 + sid)
            )
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def _embeddings() -> pl.DataFrame:
    # subject 1: notes at 5h (vec 1,0), 20h (vec 0,1); subject 2: none
    return pl.DataFrame(
        {
            "note_id": ["a", "b"],
            "subject_id": [1, 1],
            "hadm_id": [11, 11],
            "note_type": ["RR", "RR"],
            "charttime": [T0 + timedelta(hours=5), T0 + timedelta(hours=20)],
            "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            PCA_COL: [[1.0, 0.0], [0.0, 1.0]],
        }
    )


def test_note_features_follow_the_strictly_before_rule_and_windows() -> None:
    events, emb = _events(), _embeddings()
    builder = NoteFeatureBuilder(events, emb, window_hours=48.0)
    assert builder.names == note_feature_names(2) and len(builder.names) == 3 + 4
    feats = builder.features([1, 1, 1, 2], [0.0, 0.0, 0.0, 0.0], [4.0, 5.0, 24.0, 24.0])
    # t=4h: no notes yet -> counts 0, staleness NaN, vectors NaN
    assert feats[0, 0] == 0 and feats[0, 1] == 0 and np.isnan(feats[0, 2])
    assert np.isnan(feats[0, 3:]).all()
    # t=5h: the 5h note is NOT visible (strictly before)
    assert feats[1, 0] == 0 and np.isnan(feats[1, 3:]).all()
    # t=24h: both notes visible; mean = (0.5, 0.5); last = (0, 1); staleness 4h
    assert feats[2, 0] == 2 and feats[2, 1] == 2  # both within 24h and the visit
    assert feats[2, 2] == pytest.approx(4.0)
    assert feats[2, 3:5].tolist() == pytest.approx([0.5, 0.5])
    assert feats[2, 5:7].tolist() == pytest.approx([0.0, 1.0])
    # subject 2 has no notes: all NaN
    assert np.isnan(feats[3]).all()
    # a 10h window at t=24h sees only the 20h note
    narrow = NoteFeatureBuilder(events, emb, window_hours=10.0).features(
        [1], [0.0], [24.0]
    )
    assert narrow[0, 3:5].tolist() == pytest.approx([0.0, 1.0])


def test_strong_text_feature_set_appends_note_columns_through_the_harness() -> None:
    from odyssey.data.value_binning import add_value_tokens  # noqa: PLC0415

    events = _events()
    binned = add_value_tokens(events)
    rows = {"death": [IndexRow(1, 11, 24.0), IndexRow(2, 12, 24.0)]}
    with sidecar_context({NOTE_EMBEDDINGS: _embeddings()}):
        strong = features_for_events(binned, rows, feature_set="strong")["death"]
        text = features_for_events(binned, rows, feature_set="strong_text")["death"]
    assert text.shape == (2, strong.shape[1] + 7)
    np.testing.assert_array_equal(text[:, : strong.shape[1]], strong)
    assert text[0, -7] == 2  # subject 1 note count 24h
    assert np.isnan(text[1, -7:]).all()
    with (
        pytest.raises(RuntimeError, match="note_embeddings sidecar"),
        sidecar_context({}),
    ):
        active_note_embeddings()
    with pytest.raises(ValueError, match="unknown baseline feature set"):
        features_for_events(binned, rows, feature_set="strong_notes")


def test_note_features_for_rows_convenience_uses_the_active_sidecar() -> None:
    with sidecar_context({NOTE_EMBEDDINGS: _embeddings()}):
        feats, names = note_features_for_rows(_events(), [1], [0.0], [24.0])
    assert feats.shape == (1, 7) and names[0] == "note_n_24h"


# ---------------------------------------------------------------------------
# Embedder helpers (no transformers needed: a stub model)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    pad_token_id = 0
    bos_token_id = 1

    def __call__(self, texts, add_special_tokens=False, truncation=False):  # noqa: ANN001, ANN204
        # one token per character code, so lengths are controllable
        return {"input_ids": [[2 + (ord(c) % 50) for c in t] for t in texts]}


class _StubModel:
    """Hidden state = one-hot-ish of the token id, so pooling is checkable."""

    def __call__(self, input_ids, attention_mask, output_hidden_states=True):  # noqa: ANN001, ANN204
        hidden = torch.zeros(*input_ids.shape, 4)
        hidden[..., 0] = input_ids.float()
        hidden[..., 1] = 1.0
        return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden)


def test_windowing_and_token_weighted_pooling() -> None:
    assert _windows(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert _windows([1, 2], 4) == [[1, 2]]
    tok, model = _StubTokenizer(), _StubModel()
    ids = torch.tensor([[5, 7, 0], [3, 0, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = pool_last_hidden(model, ids, mask)
    assert pooled[0, 0].item() == pytest.approx(6.0) and pooled[1, 0].item() == 3.0
    assert pooled[:, 1].tolist() == [1.0, 1.0]  # padding excluded from the mean
    vecs = embed_texts(
        ["abcdef", "a"], tok, model, device="cpu", max_tokens=4, batch_size=2
    )
    assert vecs.shape == (2, 4)
    # text 0 is split into windows [bos+3 chars] + [bos+3 chars]; the pooled
    # value is the token-weighted mean of the window means = overall mean
    ids0 = (
        [1]
        + [2 + (ord(c) % 50) for c in "abc"]
        + [1]
        + [2 + (ord(c) % 50) for c in "def"]
    )
    assert vecs[0, 0] == pytest.approx(np.mean(ids0), rel=1e-5)
    assert vecs[1, 0] == pytest.approx(np.mean([1, 2 + (ord("a") % 50)]), rel=1e-5)


def test_pca_round_trip_and_saved_projection() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32) @ rng.standard_normal(
        (8, 8)
    ).astype(np.float32)
    mean, comps = fit_pca(x, 3)
    assert mean.shape == (8,) and comps.shape == (3, 8)
    z = apply_pca(x, mean, comps)
    assert z.shape == (200, 3)
    # projection is centered and components are orthonormal
    assert np.abs(z.mean(0)).max() < 1e-4
    np.testing.assert_allclose(comps @ comps.T, np.eye(3), atol=1e-5)
