"""The per-signal staleness/last-value channel (v10): resolver, state, heads."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

import odyssey.inference.run_inference as ri
from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.sequences import PatientSequence, build_patient_sequence
from odyssey.data.signal_panel import (
    N_PANEL_SIGNALS,
    NO_SIGNAL,
    SIGNAL_PANEL,
    SignalPanelResolver,
)
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import VALUE_Z_COL
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.baseline_features import StrongFeatureBuilder
from odyssey.inference.run_inference import load_run
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    RECENCY_DIM,
    SIGNAL_DIM,
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.train import TrainingConfig, build_model


T0 = datetime(2024, 1, 1)
K = N_PANEL_SIGNALS
CREAT = [i for i, (name, _) in enumerate(SIGNAL_PANEL) if name == "creatinine"][0]
HR = [i for i, (name, _) in enumerate(SIGNAL_PANEL) if name == "heart_rate"][0]
# MIMIC prefixes for those two signals (code_mapping's mimic_iv table).
CREAT_CODE = "LAB//RESULT//50912//mg/dL::HIGH"
HR_CODE = "LAB//220045//bpm::NORMAL"


def _events() -> pl.DataFrame:
    rows = [
        (1, CREAT_CODE, T0, 1.5, 10),
        (1, "MEDICATION//X", T0 + timedelta(hours=2), None, 10),
        (1, HR_CODE, T0 + timedelta(hours=3), -0.5, 10),
        (1, CREAT_CODE, T0 + timedelta(hours=5), 2.5, 10),
        (1, "DIAGNOSIS//D", T0 + timedelta(hours=6), None, 10),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            VALUE_Z_COL: pl.Float64,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def test_resolver_matches_the_baseline_feature_builders_classification() -> None:
    """The model and the GBM must classify codes identically (matched inputs)."""
    resolver = SignalPanelResolver("mimic_iv")
    assert resolver.resolve(CREAT_CODE) == CREAT
    assert resolver.resolve(HR_CODE) == HR
    assert resolver.resolve("MEDICATION//X") == NO_SIGNAL
    assert resolver.resolve("LAB//RESULT//50912//mg/dL") == CREAT  # un-binned too
    # Same answer as the strong baseline's own classifier.
    events = _events().with_columns(pl.col("time").cast(pl.Datetime("us")))
    builder = StrongFeatureBuilder(events, source="mimic_iv")
    signal_of, _ = builder._classify_codes([CREAT_CODE, HR_CODE, "MEDICATION//X"])
    assert signal_of[CREAT_CODE][0] == CREAT and signal_of[HR_CODE][0] == HR
    assert "MEDICATION//X" not in signal_of


def test_signal_state_is_exclusive_staleness_and_last_value() -> None:
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab, signal_panel=SignalPanelResolver())
    st = seq.signal_state
    assert st is not None and st.shape == (5, 2 * K) and st.dtype == np.float32
    assert np.isnan(st[0]).all()  # nothing seen before the first token
    # t=2h: creatinine seen 2h ago with z=1.5; heart rate never
    assert st[1, CREAT] == 2.0 and st[1, K + CREAT] == 1.5
    assert math.isnan(st[1, HR]) and math.isnan(st[1, K + HR])
    # t=5h (second creatinine): previous creatinine was at t=0 (exclusive)
    assert st[3, CREAT] == 5.0 and st[3, K + CREAT] == 1.5
    assert st[3, HR] == 2.0 and st[3, K + HR] == -0.5
    # t=6h: refreshed creatinine at t=5 with z=2.5
    assert st[4, CREAT] == 1.0 and st[4, K + CREAT] == 2.5
    # without a resolver the channel is absent, not zeros
    assert build_patient_sequence(events, vocab).signal_state is None


def test_tail_truncation_keeps_signal_state_aligned() -> None:
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    full = build_patient_sequence(events, vocab, signal_panel=SignalPanelResolver())
    cut = build_patient_sequence(
        events, vocab, max_seq_len=2, signal_panel=SignalPanelResolver()
    )
    assert cut.signal_state is not None and cut.signal_state.shape[0] == 2
    np.testing.assert_array_equal(cut.signal_state, full.signal_state[-2:])
    rebased = full.tail(2, rebase_times=True)
    assert rebased.time_stamps[0] == 0.0
    np.testing.assert_array_equal(rebased.signal_state, full.signal_state[-2:])


def test_signal_state_survives_chunking_and_packing() -> None:
    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab, signal_panel=SignalPanelResolver())
    chunks = list(
        PackedLaneSampler(iter([seq]), num_lanes=1, chunk_size=2, reset_prob=0.0)
    )
    got = torch.cat([c.batch.aux.signal_state[0] for c in chunks], dim=0)
    assert got.shape[1] == 2 * K
    assert torch.isnan(got[0]).all()
    assert got[1, CREAT].item() == 2.0 and got[1, K + CREAT].item() == 1.5
    assert got[3, CREAT].item() == 5.0
    packed = next(iter(PackedContextSampler(iter([seq]), batch_size=1, max_context=8)))
    ps = packed.batch.aux.signal_state[0]
    assert ps[1, CREAT].item() == 2.0 and torch.isnan(ps[5]).all()  # padding is NaN
    # A patient built without the channel packs as all-NaN rows (zeros at the heads).
    plain = build_patient_sequence(events, vocab)
    chunk = next(
        iter(
            PackedLaneSampler(iter([plain]), num_lanes=1, chunk_size=8, reset_prob=0.0)
        )
    )
    assert torch.isnan(chunk.batch.aux.signal_state).all()


def _mk(model_cls, *, recency: bool = False, signals: bool = False, **kw):
    torch.manual_seed(0)
    backbone = TinyGRUBackbone(
        vocab_size=12, hidden_size=8, num_layers=1, padding_idx=0
    )
    return model_cls(
        backbone=backbone,
        vocab_size=12,
        padding_idx=0,
        time_bin_edges=(1.0, 8.0),
        event_names=["death"],
        recency_features=recency,
        signal_channels=signals,
        **kw,
    )


def test_head_widths_resolver_and_forward_shapes() -> None:
    base = _mk(BaselineSequenceModel, signals=True)
    assert base.time_head.proj.in_features == 8 + SIGNAL_DIM
    assert isinstance(base.signal_panel, SignalPanelResolver)
    assert _mk(BaselineSequenceModel).signal_panel is None
    both = _mk(BaselineSequenceModel, recency=True, signals=True)
    assert both.time_head.proj.in_features == 8 + RECENCY_DIM + SIGNAL_DIM
    cb = _mk(
        ConceptBottleneckSequenceModel, signals=True, num_concepts=3, embedding_dim=4
    )
    assert cb.time_head.proj.in_features == cb.bottleneck.output_dim + SIGNAL_DIM

    events = _events()
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab, signal_panel=base.signal_panel)
    chunk = next(
        iter(PackedLaneSampler(iter([seq]), num_lanes=1, chunk_size=8, reset_prob=0.0))
    )
    for m in (base, both, cb):
        m.eval()
        fwd = m.forward_with_features(chunk.batch, state=None, reset_mask=None)
        assert torch.isfinite(m.time_head(fwd.features)).all()
        assert torch.isfinite(m.event_heads(fwd.features)).all()
    # The block is [log1p(hours), seen, last value] per signal; check one cell.
    fwd = base.forward_with_features(chunk.batch, state=None, reset_mask=None)
    block = fwd.features[0, :, 8:]
    assert block[1, CREAT].item() == pytest.approx(math.log1p(2.0))
    assert block[1, K + CREAT].item() == 1.0
    assert block[1, 2 * K + CREAT].item() == pytest.approx(1.5)
    assert block[1, 2 * K + HR].item() == 0.0  # unseen -> 0, not NaN
    # A batch without the channel still works (zeros block).
    aux = chunk.batch.aux._replace(signal_state=None)
    fwd = base.forward_with_features(
        chunk.batch._replace(aux=aux), state=None, reset_mask=None
    )
    assert fwd.features.shape[-1] == 8 + SIGNAL_DIM
    assert (fwd.features[..., 8:] == 0).all()


def test_build_model_threads_the_flag_and_source() -> None:
    config = TrainingConfig(
        train_shard_dir="a",
        tuning_shard_dir="b",
        output_dir="c",
        model_kind="baseline",
        backbone="transformer",
        hidden_size=8,
        num_hidden_layers=1,
        attn_num_heads=2,
        signal_channels=True,
        source="eicu",
    )
    model = build_model(config, vocab_size=12, num_concepts=3)
    assert model.use_signal_channels and model.signal_panel.source == "eicu"


@pytest.mark.parametrize("recency", [False, True])
def test_load_run_reconstructs_signal_channels_from_head_width(
    tmp_path: Path, recency: bool
) -> None:
    hidden_size = 16
    config = TrainingConfig(
        train_shard_dir="a",
        tuning_shard_dir="b",
        output_dir="c",
        model_kind="baseline",
        hidden_size=hidden_size,
    )
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))
    Vocabulary({"[PAD]": 0, "[UNK]": 1, "LAB//220045//bpm": 2}).save(
        tmp_path / "vocabulary.json"
    )
    (tmp_path / "quantile_binner.json").write_text(
        json.dumps({"n_bins": 5, "boundaries": {}})
    )
    width = hidden_size + SIGNAL_DIM + (RECENCY_DIM if recency else 0)
    torch.save(
        {"model": {"time_head.proj.weight": torch.zeros(3, width)}},
        tmp_path / "checkpoint_final.pt",
    )
    seen = {}

    def fake_build_model(cfg, *, vocab_size, num_concepts):  # noqa: ARG001
        seen["signal_channels"] = cfg.signal_channels
        seen["recency_features"] = cfg.recency_features
        raise RuntimeError("stop here")

    original = ri.build_model
    ri.build_model = fake_build_model
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            load_run(tmp_path, device="cpu")
    finally:
        ri.build_model = original
    assert seen == {"signal_channels": True, "recency_features": recency}


def test_patient_sequence_equality_ignores_the_array_field() -> None:
    a = PatientSequence(
        1, [1], [1], [0.0], [0.0], [0], [0], signal_state=np.zeros((1, 2 * K))
    )
    b = PatientSequence(1, [1], [1], [0.0], [0.0], [0], [0], signal_state=None)
    assert a == b  # compare=False: arrays never make dataclass equality ambiguous
