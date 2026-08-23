"""Shard-streaming training corpus: full-scale data without the full frame in RAM.

The in-memory path (:func:`~odyssey.training.data.load_meds_shards` and
friends) concatenates every shard of a split, labels concepts over the
whole frame and tokenizes from it. That is fine to ~100 shards; at the full
MIMIC-IV extraction (292 shards, 706M events) concept labeling alone was
OOM-killed at 82 GB on an 83 GB host. Every quantity training needs is
per-subject or a per-code aggregate, and subjects never span shards, so it
can all be computed one shard at a time:

- :func:`fit_binner_streaming`: per-code quantile cut points and value
  statistics from a fixed-size, seeded reservoir sample of values per code
  (exact per-code counts, approximate quantiles; documented on the binner).
- :func:`build_corpus_stats`: one pass over the (normalized, binned) shards
  collecting code counts (-> vocabulary and family weights), concept
  label/mask dicts, first-trigger times and alert-event time tables.
- :func:`iter_patients_streaming`: per epoch, shards in a seeded order,
  each loaded, prepared, binned and tokenized just in time for the
  :class:`~odyssey.data.streaming.PackedLaneSampler`.

Same rules as the in-memory path (train-split-only fitting, deterministic
shard order, identical normalization/recap/binning per shard) so a run's
outputs are reproducible and comparable across the two paths; the tests
check the two agree on synthetic shards.
"""

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import polars as pl
import torch

from odyssey.data.alert_events import AlertEvent, EventTimes, all_event_times
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import AnyConceptDefinition
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import PatientSequence
from odyssey.data.signal_panel import SignalPanelResolver
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
    build_visit_concept_first_times,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shard,
    shard_sort_key,
)


logger = logging.getLogger(__name__)

Preparer = Callable[[pl.DataFrame], pl.DataFrame]


def shard_paths(
    shard_dir: Union[str, Path], max_shards: Optional[int] = None
) -> List[Path]:
    """Numerically ordered shard paths of a split (same rule as load_meds_shards)."""
    paths = sorted(Path(shard_dir).glob("*.parquet"), key=shard_sort_key)
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"no .parquet shards found in {shard_dir}")
    return paths


def make_preparer(
    *, normalize_medications: bool, history_recap: bool, source: str
) -> Preparer:
    """Return the per-shard event preparation matching the in-memory path."""

    def prepare(events: pl.DataFrame) -> pl.DataFrame:
        events = maybe_normalize(events, enabled=normalize_medications, source=source)
        return maybe_history_recap(events, enabled=history_recap)

    return prepare


# ---------------------------------------------------------------------------
# Binner from a per-code reservoir sample
# ---------------------------------------------------------------------------


def fit_binner_streaming(
    paths: Sequence[Path],
    prepare: Preparer,
    *,
    n_bins: int,
    min_count: int,
    sample_per_code: int = 100_000,
    seed: int = 0,
    code_col: str = "code",
    value_col: str = "numeric_value",
) -> QuantileBinner:
    """Fit :class:`QuantileBinner` from a seeded per-code value sample across shards.

    Per-code observation counts are exact (so ``min_count`` eligibility is
    exact). Cut points and value statistics come from a sample of at most
    ``sample_per_code`` values per code, drawn uniformly within each shard
    (``sample_per_code // n_shards`` per shard, at least 200); shards are
    near-equal in size, so the pooled sample is close to uniform over all
    shards, and exact for codes with fewer values than the per-shard cap.
    """
    per_shard = max(sample_per_code // max(len(paths), 1), 200)
    counts: Counter[str] = Counter()
    samples: List[pl.DataFrame] = []
    for k, path in enumerate(paths):
        frame = prepare(load_meds_shard(path))
        if value_col not in frame.columns:
            continue
        numeric = frame.select(code_col, value_col).filter(
            pl.col(value_col).is_not_null()
        )
        if numeric.height == 0:
            continue
        counts.update(dict(numeric.group_by(code_col).len().iter_rows()))
        sampled = (
            numeric.group_by(code_col)
            .agg(
                pl.col(value_col).sample(
                    n=pl.min_horizontal(pl.len(), pl.lit(per_shard)),
                    seed=seed + k,
                )
            )
            .filter(pl.col(value_col).list.len() > 0)
            .explode(value_col)
        )
        samples.append(sampled.select(code_col, pl.col(value_col).cast(pl.Float64)))
    eligible = [code for code, n in counts.items() if n >= min_count]
    if not samples or not eligible:
        return QuantileBinner(boundaries={}, n_bins=n_bins)
    sample = pl.concat(samples).filter(pl.col(code_col).is_in(eligible))
    return QuantileBinner.fit(sample, n_bins=n_bins, min_count=1)


# ---------------------------------------------------------------------------
# One pass: code counts, concept labels, first times, event tables
# ---------------------------------------------------------------------------


@dataclass
class CorpusStats:
    """Everything training needs from the train split besides the token stream."""

    code_counts: Dict[str, int]
    n_subjects: int
    n_events: int
    labels: Dict[Any, torch.Tensor] = field(default_factory=dict)
    """Concept labels keyed by (subject, visit) or subject, like the trainer."""
    masks: Dict[Any, torch.Tensor] = field(default_factory=dict)
    first_times: Dict[Any, torch.Tensor] = field(default_factory=dict)
    event_times: Dict[str, EventTimes] = field(default_factory=dict)


def merge_event_times(into: Dict[str, EventTimes], part: Dict[str, EventTimes]) -> None:
    """Merge one shard's per-event onset/censor times into a running accumulator.

    Subjects never span shards, so each shard contributes disjoint keys;
    merging is a plain dict update per event. Shared by
    :func:`build_corpus_stats` (training corpus stats) and
    :func:`~odyssey.inference.alerts.fit_baselines_streaming` (baseline
    GBM fitting), the two streaming consumers of per-event time tables.
    """
    for name, times in part.items():
        if name not in into:
            into[name] = EventTimes(
                onset=dict(times.onset),
                censor=dict(times.censor),
                subject_scoped=times.subject_scoped,
            )
        else:
            into[name].onset.update(times.onset)
            into[name].censor.update(times.censor)


def build_corpus_stats(
    paths: Sequence[Path],
    prepare: Preparer,
    binner: QuantileBinner,
    *,
    source: str,
    concepts: Sequence[AnyConceptDefinition],
    concept_supervision: str,
    with_first_times: bool,
    alerts: Optional[Sequence[AlertEvent]],
    code_col: str = "code",
) -> CorpusStats:
    """Aggregate code counts, concept labels/masks/first times and event times."""
    counts: Counter[str] = Counter()
    stats = CorpusStats(code_counts={}, n_subjects=0, n_events=0)
    for i, path in enumerate(paths):
        raw = prepare(load_meds_shard(path))
        binned = add_value_tokens(raw, binner, source=source)
        counts.update(dict(binned.group_by(code_col).len().iter_rows()))
        stats.n_subjects += int(binned["subject_id"].n_unique())
        stats.n_events += binned.height
        labels: Dict[Any, torch.Tensor]
        masks: Dict[Any, torch.Tensor]
        if concept_supervision == "visit":
            labels, masks = build_visit_concept_label_dicts(raw, concepts)
            if with_first_times:
                stats.first_times.update(build_visit_concept_first_times(raw, concepts))
        elif concept_supervision == "stay":
            labels, masks = build_concept_label_dicts(raw, concepts)
            if with_first_times:
                stats.first_times.update(build_concept_first_times(raw, concepts))
        else:
            raise ValueError(f"unknown concept_supervision {concept_supervision!r}")
        stats.labels.update(labels)
        stats.masks.update(masks)
        if alerts:
            merge_event_times(stats.event_times, all_event_times(raw, alerts, source))
        if (i + 1) % 20 == 0:
            logger.info("[stream] stats: %d/%d shards", i + 1, len(paths))
    stats.code_counts = dict(counts)
    return stats


def family_loss_weights_from_counts(
    code_counts: Dict[str, int],
    *,
    alpha: float,
    cap: float = 20.0,
    n_families: Optional[int] = None,
) -> torch.Tensor:
    """Per-family loss weights from code counts (see ``family_loss_weights``)."""
    per_family: Counter[int] = Counter()
    for code, cnt in code_counts.items():
        per_family[code_type(code)] += cnt
    n_fam = max(max(per_family) + 1, n_families or 0)
    n = torch.zeros(n_fam, dtype=torch.float64)
    for fam, cnt in per_family.items():
        n[fam] = float(cnt)
    share = n / n.sum()
    raw = torch.where(
        share > 0, share.clamp_min(1e-12) ** (-alpha), torch.zeros_like(share)
    )
    scale = (share * raw).sum()
    weights = torch.where(scale > 0, raw / scale, raw)
    weights = torch.where(share > 0, weights, torch.ones_like(weights))
    return weights.clamp_max(cap).to(torch.float32)


# ---------------------------------------------------------------------------
# Token stream
# ---------------------------------------------------------------------------


def iter_patients_streaming(
    paths: Sequence[Path],
    prepare: Preparer,
    binner: QuantileBinner,
    vocab: Vocabulary,
    *,
    source: str,
    max_seq_len: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    signal_panel: Optional[SignalPanelResolver] = None,
) -> Iterator[PatientSequence]:
    """Yield patient sequences shard by shard; shards and subjects shuffled per seed."""
    order = list(paths)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    for k, path in enumerate(order):
        binned = add_value_tokens(prepare(load_meds_shard(path)), binner, source=source)
        yield from iter_patient_sequences(
            binned,
            vocab,
            max_seq_len=max_seq_len,
            shuffle_seed=None if shuffle_seed is None else shuffle_seed * 7919 + k,
            signal_panel=signal_panel,
        )


__all__ = [
    "CorpusStats",
    "Preparer",
    "build_corpus_stats",
    "family_loss_weights_from_counts",
    "fit_binner_streaming",
    "iter_patients_streaming",
    "make_preparer",
    "merge_event_times",
    "shard_paths",
]
