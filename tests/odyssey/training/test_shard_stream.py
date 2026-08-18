"""Shard-streaming corpus preparation agrees with the in-memory path."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.training.data import (
    build_visit_concept_label_dicts,
    build_vocabulary,
    family_loss_weights,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.shard_stream import (
    build_corpus_stats,
    family_loss_weights_from_counts,
    fit_binner_streaming,
    iter_patients_streaming,
    make_preparer,
    shard_paths,
)


T0 = datetime(2024, 1, 1)
SCHEMA = {
    "subject_id": pl.Int64,
    "code": pl.Utf8,
    "time": pl.Datetime,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _write_shards(shard_dir: Path, n_shards: int, subjects_per_shard: int) -> None:
    shard_dir.mkdir(parents=True)
    sid = 0
    for k in range(n_shards):
        rows = []
        for _ in range(subjects_per_shard):
            sid += 1
            hadm = 1000 + sid
            base = T0 + timedelta(days=sid)
            rows.append((sid, "GENDER//F" if sid % 2 else "GENDER//M", None, None, None))
            rows.append((sid, "HOSPITAL_ADMISSION//EW", base, None, hadm))
            for i in range(30):
                t = base + timedelta(hours=i)
                rows.append((sid, "LAB//220045//bpm", t, 60.0 + 3 * ((sid + i) % 20), hadm))
                rows.append((sid, "LAB//RESULT//50912//mg/dL", t, 0.6 + 0.1 * (i % 9), hadm))
                if i % 5 == 0:
                    rows.append((sid, "MEDICATION//furosemide//Administered", t, None, hadm))
                if sid % 3 == 0 and i == 12:
                    rows.append((sid, "MEDICATION//norepinephrine//Administered", t, None, hadm))
                    rows.append((sid, "ICU_ADMISSION//MICU", t, None, hadm))
            rows.append((sid, "DIAGNOSIS//ICD//10//I50", base + timedelta(hours=31), None, hadm))
        pl.DataFrame(rows, schema=SCHEMA, orient="row").write_parquet(shard_dir / f"{k}.parquet")


def test_streaming_matches_in_memory_preparation(tmp_path: Path) -> None:
    shard_dir = tmp_path / "train"
    _write_shards(shard_dir, n_shards=3, subjects_per_shard=6)
    paths = shard_paths(shard_dir)
    prepare = make_preparer(normalize_medications=True, history_recap=False, source="mimic_iv")
    concepts = concepts_for_source("mimic_iv")

    whole = prepare(load_meds_shards(shard_dir))
    ref_binner = QuantileBinner.fit(whole, n_bins=5, min_count=20)
    binner = fit_binner_streaming(
        paths, prepare, n_bins=5, min_count=20, sample_per_code=1_000_000, seed=0
    )
    # every code has fewer values than the per-shard cap: the sample is exhaustive
    assert set(binner.boundaries) == set(ref_binner.boundaries)
    for code, cuts in ref_binner.boundaries.items():
        assert binner.boundaries[code] == cuts
        c1, s1 = binner.value_stats[code]
        c2, s2 = ref_binner.value_stats[code]
        assert abs(c1 - c2) < 1e-6 and abs(s1 - s2) < 1e-6

    stats = build_corpus_stats(
        paths,
        prepare,
        binner,
        source="mimic_iv",
        concepts=concepts,
        concept_supervision="visit",
        with_first_times=False,
        alerts=ALERT_EVENTS,
    )
    whole_binned = add_value_tokens(whole, ref_binner, source="mimic_iv")
    ref_counts = dict(whole_binned.group_by("code").len().iter_rows())
    assert stats.code_counts == ref_counts
    assert stats.n_subjects == 18 and stats.n_events == whole.height

    ref_labels, ref_masks = build_visit_concept_label_dicts(whole, concepts)
    assert stats.labels.keys() == ref_labels.keys()
    for key in ref_labels:
        assert torch.equal(stats.labels[key], ref_labels[key])
        assert torch.equal(stats.masks[key], ref_masks[key])

    ref_times = all_event_times(whole, ALERT_EVENTS, "mimic_iv")
    assert stats.event_times.keys() == ref_times.keys()
    for name, times in ref_times.items():
        assert stats.event_times[name].onset == times.onset
        assert stats.event_times[name].censor == times.censor
        assert stats.event_times[name].subject_scoped == times.subject_scoped

    vocab_stream = Vocabulary.build_from_counts(stats.code_counts, min_count=1, max_size=1000)
    vocab_ref = build_vocabulary(whole_binned, min_count=1, max_size=1000)
    assert vocab_stream.token_to_id == vocab_ref.token_to_id

    w_stream = family_loss_weights_from_counts(stats.code_counts, alpha=0.5, n_families=9)
    w_ref = family_loss_weights(whole_binned, alpha=0.5, n_families=9)
    assert torch.allclose(w_stream, w_ref)

    streamed = {
        s.subject_id: s.concept_ids
        for s in iter_patients_streaming(paths, prepare, binner, vocab_ref, source="mimic_iv")
    }
    reference = {
        s.subject_id: s.concept_ids for s in iter_patient_sequences(whole_binned, vocab_ref)
    }
    assert streamed == reference

    # shuffled epochs visit every subject exactly once, in a seed-dependent order
    order_a = [
        s.subject_id
        for s in iter_patients_streaming(
            paths, prepare, binner, vocab_ref, source="mimic_iv", shuffle_seed=1
        )
    ]
    order_b = [
        s.subject_id
        for s in iter_patients_streaming(
            paths, prepare, binner, vocab_ref, source="mimic_iv", shuffle_seed=2
        )
    ]
    assert sorted(order_a) == sorted(reference) and order_a != order_b
