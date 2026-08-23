"""Streaming preparation of baseline fitting/scoring data, one shard at a time.

Why this module exists (confirmed incident, not hypothetical): the baseline
rescoring scripts used to load EVERY shard of a split into one polars frame
(``load_meds_shards`` -> one concat), hold that frame alive across three
consecutive baseline-family fits, and then load the ENTIRE raw split a
second time just to compute event times -- peaking at 30-40+ GB and getting
OOM-killed twice on the 83 GB eval boxes (2026-08-23, eICU fit_score legs at
35-37 GB anon-rss while "loading train shards"). Meanwhile
:func:`odyssey.inference.alerts.fit_baselines_streaming` had already solved
the identical problem for the GBM path: subjects never span shards (a MEDS
invariant), landmark rows and baseline features are entirely per-subject,
so everything can be built one shard at a time and only the small per-
landmark outputs kept.

:func:`prepare_baseline_data` is that same design, factored out so every
baseline family -- and both the fitting split and the held-out scoring
split -- goes through one shared, memory-bounded path:

- one shard in memory at a time (the raw and binned frames are dropped
  before the next shard loads);
- event times merged per shard from the PREPARED (post-normalization)
  events, matching ``fit_baselines_streaming`` -- not from a second
  un-normalized full load, which is both the memory bug and a subtle
  label-provenance inconsistency (code-normalization can affect which
  events an alert matcher sees);
- features built per shard as ``float32`` chunks and concatenated once at
  the end (the concatenated matrix is the small thing: one row per
  landmark, not per raw event);
- per-event feature matrices that are byte-identical across events (the
  landmark grid is shared; only outcomes differ) are stored as ONE array
  aliased under every event name, not four copies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from odyssey.data.alert_events import AlertEvent, all_event_times
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.inference.alerts import (
    BASELINE_FEATURE_SETS,
    EventTimes,
    IndexRow,
    _index_rows_from_events,
    features_for_events,
)
from odyssey.training.data import load_meds_shard
from odyssey.training.shard_stream import Preparer, merge_event_times


logger = logging.getLogger(__name__)

#: Loader signature, injectable for tests (e.g. tracking frame lifetimes).
ShardLoader = Callable[[Path], pl.DataFrame]


@dataclass
class BaselineData:
    """Everything the baseline families need from one split, sans raw frames.

    ``features`` is keyed ``feature_set -> event -> (n_rows, n_features)``
    float32 matrix, aligned row-for-row with ``rows[event]``. When every
    event shares the identical landmark grid (the normal case), the inner
    per-event values alias ONE array object per feature set.
    """

    rows: Dict[str, List[IndexRow]] = field(default_factory=dict)
    times: Dict[str, EventTimes] = field(default_factory=dict)
    features: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


def prepare_baseline_data(
    paths: Sequence[Path],
    prepare: Preparer,
    binner: Optional[QuantileBinner],
    *,
    alerts: Sequence[AlertEvent],
    feature_sets: Sequence[str] = ("strong",),
    source: str = "mimic_iv",
    landmark_hours: float = 4.0,
    loader: ShardLoader = load_meds_shard,
) -> BaselineData:
    """Build landmark rows, event times, and features one shard at a time.

    Semantically equivalent to loading every shard into one frame and
    calling ``_index_rows_from_events`` + ``all_event_times`` +
    ``features_for_events`` on it (subjects never span shards, and every
    one of those computations is per-subject) -- but with a peak memory
    footprint of ONE shard's frames plus the accumulated per-landmark
    outputs, instead of the whole split several times over. Verified
    equivalent by test, not by argument alone
    (``tests/odyssey/inference/test_baseline_prep.py``).
    """
    for fs in feature_sets:
        if fs not in BASELINE_FEATURE_SETS:
            raise ValueError(f"unknown baseline feature set {fs!r}")
    data = BaselineData()
    chunks: Dict[str, Dict[str, List[np.ndarray]]] = {fs: {} for fs in feature_sets}
    for path in paths:
        raw = prepare(loader(Path(path)))
        merge_event_times(data.times, all_event_times(raw, alerts, source))
        binned = add_value_tokens(raw, binner, source=source)
        del raw  # one shard's frames at a time -- the module's whole point
        shard_rows = _index_rows_from_events(
            binned, alerts, landmark_hours=landmark_hours
        )
        # The landmark grid is event-independent in the normal case (only
        # outcomes differ), so detect that and build ONE feature matrix per
        # set, appended as the same object under every event -- rather than
        # features_for_events' per-event fancy-indexed copies, which would
        # multiply the split's feature memory by the number of events.
        key_seqs = [
            [(r.subject_id, r.visit_id, r.time_hours) for r in event_rows]
            for event_rows in shard_rows.values()
        ]
        shared_grid = bool(key_seqs) and all(s == key_seqs[0] for s in key_seqs[1:])
        for fs in feature_sets:
            if shared_grid:
                canonical = next(iter(shard_rows))
                feats = features_for_events(
                    binned,
                    {canonical: shard_rows[canonical]},
                    source=source,
                    feature_set=fs,
                ).get(canonical)
                if feats is None:
                    continue
                feats = feats.astype(np.float32, copy=False)
                for event in shard_rows:
                    chunks[fs].setdefault(event, []).append(feats)
            else:
                shard_feats = features_for_events(
                    binned, shard_rows, source=source, feature_set=fs
                )
                for event, feats in shard_feats.items():
                    chunks[fs].setdefault(event, []).append(
                        feats.astype(np.float32, copy=False)
                    )
        for event, event_rows in shard_rows.items():
            data.rows.setdefault(event, []).extend(event_rows)
        del binned
    for fs in feature_sets:
        data.features[fs] = _concat_dedup(chunks[fs])
    n_rows = {event: len(rows) for event, rows in data.rows.items()}
    logger.info(
        "[baseline_prep] %d shards -> rows per event %s, feature sets %s",
        len(paths),
        n_rows,
        list(feature_sets),
    )
    return data


def _concat_dedup(
    per_event_chunks: Dict[str, List[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Concatenate per-event chunk lists, aliasing identical lists to one array.

    Events share the landmark grid, so their chunk lists are usually the
    same array objects in the same order -- concatenating each list
    separately would materialize N identical full-size matrices. Group by
    object identity and concatenate once per distinct list instead; a
    per-event list that genuinely differs still gets its own concat.
    """
    out: Dict[str, np.ndarray] = {}
    by_identity: Dict[Tuple[int, ...], np.ndarray] = {}
    for event, chunk_list in per_event_chunks.items():
        if not chunk_list:
            continue
        identity = tuple(id(c) for c in chunk_list)
        if identity not in by_identity:
            by_identity[identity] = (
                chunk_list[0]
                if len(chunk_list) == 1
                else np.concatenate(chunk_list, axis=0)
            )
        out[event] = by_identity[identity]
    return out
