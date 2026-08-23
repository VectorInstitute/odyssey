"""Tests for converting raw MEDS events into patient token sequences."""

import math
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from odyssey.data.sequences import (
    HOURS_PER_YEAR,
    NO_VISIT,
    _signal_state,
    build_patient_sequence,
    collate_patient_sequences,
)
from odyssey.data.signal_panel import N_PANEL_SIGNALS
from odyssey.data.vocabulary import PAD_ID, UNK_ID, Vocabulary
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


def _events(rows: list) -> pl.DataFrame:
    """rows: list of (subject_id, time_or_None, code, hadm_id_or_None)."""
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


VOCAB = Vocabulary.build(
    ["DIAGNOSIS//A", "LAB//220045//bpm", "MEDICATION//X"], min_count=1
)
T0 = datetime(2180, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# build_patient_sequence
# ---------------------------------------------------------------------------


def test_events_sorted_by_time_regardless_of_input_order() -> None:
    events = _events(
        [
            (1, T0 + timedelta(hours=2), "LAB//220045//bpm", None),
            (1, T0, "DIAGNOSIS//A", None),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.concept_ids == [
        VOCAB.encode("DIAGNOSIS//A"),
        VOCAB.encode("MEDICATION//X"),
        VOCAB.encode("LAB//220045//bpm"),
    ]
    assert seq.time_stamps == [0.0, 1.0, 2.0]


def test_same_timestamp_events_keep_a_deterministic_relative_order() -> None:
    # A lab panel drawn together has no true order, but the tokenized
    # sequence still needs a fixed, reproducible one (see
    # build_patient_sequence's docstring) -- ties must keep their
    # relative order from the input, not depend on Polars' default
    # (unstable) sort behavior.
    codes = ["LAB//E//", "LAB//D//", "LAB//C//", "LAB//B//", "LAB//A//"]
    vocab = Vocabulary.build(codes, min_count=1)
    events = _events([(1, T0, code, None) for code in codes])

    seq = build_patient_sequence(events, vocab)

    assert seq.concept_ids == [vocab.encode(c) for c in codes]
    assert seq.time_stamps == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_static_timeless_facts_lead_the_sequence() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", None),
            (1, None, "GENDER//F", None),
        ]
    )
    vocab = Vocabulary.build(["DIAGNOSIS//A", "GENDER//F"], min_count=1)
    seq = build_patient_sequence(events, vocab)
    assert len(seq) == 2
    assert vocab.decode(seq.concept_ids[0]) == "GENDER//F"
    assert seq.time_stamps == [0.0, 0.0]


def test_meds_birth_computes_ages_and_is_excluded_from_tokens() -> None:
    birth = T0 - timedelta(days=30 * 365.25)  # ~30 years before first event
    events = _events(
        [
            (1, birth, "MEDS_BIRTH", None),
            (1, T0, "DIAGNOSIS//A", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert len(seq) == 1  # MEDS_BIRTH itself isn't a sequence token
    assert abs(seq.ages[0] - 30.0) < 0.1


def test_ages_are_zero_without_a_birth_event() -> None:
    events = _events([(1, T0, "DIAGNOSIS//A", None)])
    seq = build_patient_sequence(events, VOCAB)
    assert seq.ages == [0.0]


def test_unknown_code_maps_to_unk_token() -> None:
    events = _events([(1, T0, "NEVER_SEEN//CODE", None)])
    seq = build_patient_sequence(events, VOCAB)
    assert seq.concept_ids == [UNK_ID]


def test_max_seq_len_keeps_most_recent_events() -> None:
    events = _events(
        [(1, T0 + timedelta(hours=i), f"DIAGNOSIS//{i}", None) for i in range(5)]
    )
    vocab = Vocabulary.build([f"DIAGNOSIS//{i}" for i in range(5)], min_count=1)
    seq = build_patient_sequence(events, vocab, max_seq_len=2)
    assert len(seq) == 2
    assert seq.time_stamps == [3.0, 4.0]  # kept the last two, times unshifted


def test_empty_events_produce_empty_sequence() -> None:
    events = _events([])
    seq = build_patient_sequence(events, VOCAB)
    assert len(seq) == 0


def test_multiple_subject_ids_raises() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", None),
            (2, T0, "MEDICATION//X", None),
        ]
    )
    with pytest.raises(ValueError, match="single subject_id"):
        build_patient_sequence(events, VOCAB)


def test_missing_hadm_id_column_treated_like_all_none() -> None:
    events = pl.DataFrame(
        [(1, T0, "DIAGNOSIS//A"), (1, T0 + timedelta(hours=1), "MEDICATION//X")],
        schema={"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.Utf8},
        orient="row",
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 1]  # each event its own solo visit


# ---------------------------------------------------------------------------
# Visit derivation
# ---------------------------------------------------------------------------


def test_same_hadm_id_forms_one_visit_with_first_middle_last_segments() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 100),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", 100),
            (1, T0 + timedelta(hours=2), "LAB//220045//bpm", 100),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 0, 0]
    assert seq.visit_segments == [0, 1, 2]


def test_events_without_hadm_id_each_get_their_own_visit() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", None),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 1]
    # FROZEN, DO NOT CHANGE (review finding 14): a single-event visit is
    # always coded segment 0 ("first"), never 2 ("last") -- _assign_visits'
    # if/elif only ever takes the `if k == i` branch when the group has
    # exactly one event, since k == i and k == j - 1 are the same index
    # there. This is now pinned as intentional behavior, not a bug:
    # changing it would alter tokenization for every source (MIMIC-IV,
    # eICU, GEMINI) mid-program, which is a far bigger cost than the
    # asymmetry itself.
    assert seq.visit_segments == [0, 0]


def test_distinct_hadm_ids_get_distinct_visit_orders() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 100),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", 200),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_orders == [0, 1]


def test_visit_order_capped_at_max_num_visits() -> None:
    events = _events(
        [(1, T0 + timedelta(hours=i), "DIAGNOSIS//A", 100 + i) for i in range(5)]
    )
    seq = build_patient_sequence(events, VOCAB, max_num_visits=3)
    assert max(seq.visit_orders) == 2  # capped at max_num_visits - 1


# ---------------------------------------------------------------------------
# collate_patient_sequences
# ---------------------------------------------------------------------------


def test_collate_pads_to_longest_sequence() -> None:
    events_a = _events([(1, T0, "DIAGNOSIS//A", None)])
    events_b = _events(
        [
            (2, T0, "DIAGNOSIS//A", None),
            (2, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq_a = build_patient_sequence(events_a, VOCAB)
    seq_b = build_patient_sequence(events_b, VOCAB)

    batch = collate_patient_sequences([seq_a, seq_b])

    assert batch.concept_ids.shape == (2, 2)
    assert batch.concept_ids[0, 1].item() == PAD_ID  # padded position
    assert batch.concept_ids[0, 0].item() == VOCAB.encode("DIAGNOSIS//A")
    assert batch.concept_ids[1, 0].item() == VOCAB.encode("DIAGNOSIS//A")
    assert batch.concept_ids[1, 1].item() == VOCAB.encode("MEDICATION//X")


def test_collate_output_types_and_aux_shapes_match() -> None:
    seq = build_patient_sequence(_events([(1, T0, "DIAGNOSIS//A", None)]), VOCAB)
    batch = collate_patient_sequences([seq])

    assert batch.concept_ids.dtype == torch.long
    assert batch.aux.type_ids.shape == batch.concept_ids.shape
    assert batch.aux.time_stamps.shape == batch.concept_ids.shape
    assert batch.aux.ages.shape == batch.concept_ids.shape
    assert batch.aux.visit_orders.shape == batch.concept_ids.shape
    assert batch.aux.visit_segments.shape == batch.concept_ids.shape


def test_collate_empty_list_produces_empty_batch() -> None:
    batch = collate_patient_sequences([])
    assert batch.concept_ids.shape == (0, 0)


def test_collate_mixes_empty_and_nonempty_sequences() -> None:
    empty_seq = build_patient_sequence(_events([]), VOCAB)
    seq = build_patient_sequence(_events([(1, T0, "DIAGNOSIS//A", None)]), VOCAB)

    batch = collate_patient_sequences([empty_seq, seq])

    assert batch.concept_ids.shape == (2, 1)
    assert batch.concept_ids[0, 0].item() == PAD_ID  # the empty sequence's only slot
    assert batch.concept_ids[1, 0].item() == VOCAB.encode("DIAGNOSIS//A")


def test_hours_per_year_constant_is_a_julian_year() -> None:
    assert abs(HOURS_PER_YEAR - 24.0 * 365.25) < 1e-9


# ---------------------------------------------------------------------------
# Real MEDS data, end to end: extraction -> vocab -> sequences -> model
# ---------------------------------------------------------------------------


@pytest.mark.integration_test
def test_real_meds_data_tokenizes_and_runs_through_the_model(tmp_path: Path) -> None:
    """The full path a training script will actually take, on real data.

    Runs the real meds-extract pipeline against the public MIMIC-IV demo,
    builds a vocabulary from the resulting events, tokenizes a batch of
    real patients, and runs that batch through
    ConceptBottleneckSequenceModel end to end -- proving the tokenization
    output is actually shaped and typed the way the model expects, not
    just that each piece works in isolation.

    ``MEDS_DEMO_CACHE_DIR``, if set, redirects ``output_dir`` from
    ``tmp_path`` (always fresh, so meds-extract-run's own PhysioNet fetch
    always hits the network) to a stable directory, and skips the fetch
    entirely if that directory already holds a complete extraction from
    an earlier run -- unset (the default for local/dev runs), behavior is
    unchanged. CI sets it to an actions/cache-restored path so the
    PhysioNet download only ever happens on a real cache miss: real
    incident this closes, PhysioNet's own ConnectTimeout flaking
    integration tests that have nothing to do with the network fetch
    itself. Everything after extraction -- vocab, tokenization, the model
    forward pass -- always runs for real regardless of cache hit or miss;
    only the extraction subprocess itself is ever skipped.
    """
    cache_dir = os.environ.get("MEDS_DEMO_CACHE_DIR")
    output_dir = Path(cache_dir) / "meds_demo" if cache_dir else tmp_path / "meds_demo"
    train_shards_glob = list((output_dir / "data" / "train").glob("*.parquet"))
    if cache_dir and train_shards_glob:
        pass  # cache hit -- reuse the earlier extraction, no network fetch
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "meds-extract-run",
                "spec=MIMIC-IV",
                f"output_dir={output_dir}",
                "dataset_key=demo",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-4000:]

    shards = sorted((output_dir / "data" / "train").glob("*.parquet"))[:3]
    events = pl.concat([pl.read_parquet(s) for s in shards])
    events = events.select(["subject_id", "time", "code", "hadm_id"])

    vocab = Vocabulary.build(events["code"].to_list(), min_count=2, max_size=2000)
    assert len(vocab) > 2  # more than just PAD/UNK

    subject_ids = events["subject_id"].unique().to_list()[:8]
    sequences = [
        build_patient_sequence(
            events.filter(pl.col("subject_id") == sid), vocab, max_seq_len=64
        )
        for sid in subject_ids
    ]
    sequences = [s for s in sequences if len(s) > 0]
    assert sequences, "expected at least one non-empty real patient sequence"

    batch = collate_patient_sequences(sequences)
    assert batch.concept_ids.shape[0] == len(sequences)
    assert batch.concept_ids.max().item() < len(vocab)
    assert batch.aux.visit_segments.max().item() <= 2

    backbone = TinyGRUBackbone(
        vocab_size=len(vocab), hidden_size=16, padding_idx=PAD_ID
    )
    model = ConceptBottleneckSequenceModel(
        backbone=backbone,
        vocab_size=len(vocab),
        num_concepts=4,
        embedding_dim=4,
        padding_idx=PAD_ID,
    )
    concept_labels = torch.zeros(len(sequences), 4)
    total, _ = model.compute_loss(batch, concept_labels)
    assert torch.isfinite(total)
    total.backward()
    assert model.bottleneck.context_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# visit_ids / visit_ends (visit-scoped concept supervision)
# ---------------------------------------------------------------------------


def test_visit_ids_and_ends_mark_each_real_visits_last_event() -> None:

    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 10),
            (1, T0 + timedelta(hours=1), "LAB//220045//bpm", 10),
            (1, T0 + timedelta(hours=2), "MEDICATION//X", 11),
            (1, T0 + timedelta(hours=3), "DIAGNOSIS//A", 11),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_ids == [10, 10, 11, 11]
    assert seq.visit_ends == [False, True, False, True]
    assert NO_VISIT not in seq.visit_ids


def test_visit_end_is_the_true_last_event_even_when_interleaved() -> None:
    # A solo event lands between two events of the same visit: the visit's
    # end must be its LAST event overall, not the end of the first run.
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 10),
            (1, T0 + timedelta(hours=1), "LAB//220045//bpm", None),
            (1, T0 + timedelta(hours=2), "MEDICATION//X", 10),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_ids == [10, -1, 10]
    assert seq.visit_ends == [False, False, True]


def test_solo_events_never_carry_visit_supervision() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", None),
            (1, T0 + timedelta(hours=1), "MEDICATION//X", None),
        ]
    )
    seq = build_patient_sequence(events, VOCAB)
    assert seq.visit_ids == [-1, -1]
    assert seq.visit_ends == [False, False]


def test_truncation_slices_visit_fields_consistently() -> None:
    events = _events(
        [
            (1, T0, "DIAGNOSIS//A", 10),
            (1, T0 + timedelta(hours=1), "LAB//220045//bpm", 10),
            (1, T0 + timedelta(hours=2), "MEDICATION//X", 11),
            (1, T0 + timedelta(hours=3), "DIAGNOSIS//A", 11),
        ]
    )
    seq = build_patient_sequence(events, VOCAB, max_seq_len=2)
    assert len(seq.visit_ids) == 2
    assert seq.visit_ids == [11, 11]
    assert seq.visit_ends == [False, True]


def test_static_events_lead_the_sequence_at_the_first_timestamp() -> None:
    t0 = datetime(2024, 1, 1)
    events = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "code": ["GENDER//F", "LAB//A", "RACE//X", "LAB//B"],
            "time": [None, t0, None, t0 + timedelta(hours=2)],
            "numeric_value": [None, 1.0, None, 2.0],
            "hadm_id": [None, 10, None, 10],
        },
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
    )
    vocab = Vocabulary.build(events["code"].to_list(), min_count=1)
    seq = build_patient_sequence(events, vocab)
    codes = [vocab.decode(i) for i in seq.concept_ids]
    assert codes == ["GENDER//F", "RACE//X", "LAB//A", "LAB//B"]
    assert seq.time_stamps[:3] == [0.0, 0.0, 0.0] and seq.time_stamps[3] == 2.0
    assert seq.visit_ids[:3] == [10, 10, 10]  # same visit as the first timed event
    # a static-only subject still yields an empty sequence
    only_static = events.filter(pl.col("time").is_null())
    assert len(build_patient_sequence(only_static, vocab)) == 0


# ---------------------------------------------------------------------------
# signal_state (real-data finding, research_journal/experiments/44_real_data_checks.html)
# ---------------------------------------------------------------------------


def test_signal_state_last_value_goes_nan_after_a_null_valued_repeat_reading() -> None:
    """Pins signal_state's current (surprising) last_value behavior -- NOT
    asserted as correct, flagged as an open design question.

    Real-data finding: a real held-out shard has cases where a panel
    signal is charted twice close together, the second time with a null
    numeric_value (e.g. an order/duplicate row under the same resolvable
    prefix). _signal_state's staleness channel correctly reports "this
    signal was seen recently" in that case, but the last_value channel
    goes NaN even though a real earlier value exists -- because "last
    occurrence" is tracked by position regardless of whether that
    occurrence's own value was null, then that position's (null) value is
    what gets read back.

    Minimal repro: signal 0 (e.g. creatinine) observed at t=0h with a real
    value, then again at t=1h with NO value, then an unrelated event at
    t=2h. At t=2h, staleness correctly says "1h" (last occurrence: t=1h),
    but last_value is NaN, not the real 1.5 from t=0h.

    Whether this should instead carry forward the last NON-null value is a
    real design question (a null reading could itself be meaningful, e.g.
    "ordered but not yet resulted") -- this test documents current
    behavior precisely so a deliberate choice can be made, not so this
    assertion is treated as the desired outcome.
    """
    signal_ids = [0, 0, -1]  # NO_SIGNAL sentinel for the third, unrelated row
    time_stamps = [0.0, 1.0, 2.0]
    values = [1.5, None, 0.0]

    state = _signal_state(signal_ids, time_stamps, values)
    staleness_col, last_value_col = 0, N_PANEL_SIGNALS

    assert state[2, staleness_col] == 1.0  # staleness: correctly "1h ago"
    # current behavior: the real t=0h value (1.5) is lost, not carried
    # forward through the null-valued t=1h occurrence.
    assert math.isnan(state[2, last_value_col])
