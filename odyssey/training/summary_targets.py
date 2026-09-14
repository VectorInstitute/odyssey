"""Self-supervised window-summary targets for the summary head.

The tuned GBM the hazard heads are compared with is hand-fed windowed
statistics of the chart: per-signal minimum, maximum and mean over 6 h and
24 h, the change from the visit's first value, and per-family occurrence
counts. Frozen probes (2026-09-13, ``scripts/probe_summary_signal.py``)
showed the backbone's state holds current levels but not changes from
baseline and not fine window counts, linearly or otherwise. This module
turns those statistics into an auxiliary training target: at 4-hourly
landmark positions the model must *report* them from its own state. The
values are computed from the input stream by the GBM's own feature code,
never fed in as inputs, and the head that predicts them is discarded at
inference. The state is shaped to hold what the statistics need; nothing
is handed to it.

Pipeline: :func:`compute_summary_targets` runs once per shard (offline,
``scripts/build_summary_targets.py``) and writes one parquet per shard;
:func:`fit_summary_stats` standardizes columns over the training set
(counts through ``log1p``, everything winsorized at the 0.5/99.5
percentiles so sentinel values cannot dominate); :class:`SummaryTargetTables`
loads the parquets and :func:`summary_targets_for_chunk` looks up, for
every streaming chunk, which positions carry a target -- the last token of
the bundle at each landmark row -- and returns a ``(lanes, T, K)`` tensor
with a mask. Times are hours on the sequence origin, the same clock as
chunk time stamps and the alert harness' landmark rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from odyssey.data.alert_events import origin_hours
from odyssey.data.signal_panel import SIGNAL_PANEL
from odyssey.data.streaming import StreamingChunk
from odyssey.inference.baseline_features import (
    DRUG_CLASSES,
    FAMILY_LABELS,
    StrongFeatureBuilder,
    feature_names,
)


# Window statistics asked for per panel signal: the GBM's ``summary_stats``
# group without ``last``/``hours_since_last`` (recency probes already showed
# the state holds those) and without ``delta_prev``/``ratio_visit_min``
# (functions of the others).
SIGNAL_STATS: tuple[str, ...] = (
    "min_6h",
    "max_6h",
    "min_24h",
    "max_24h",
    "mean_24h",
    "delta_visit_first",
)

# Occurrence counts asked for per drug class and per code family: the GBM's
# ``counts_occurrence`` group at its two window lengths.
COUNT_STATS: tuple[str, ...] = ("n_6h", "n_24h")

DEFAULT_LANDMARK_HOURS = 4.0
WINSOR_PERCENTILES = (0.5, 99.5)
KEY_COLUMNS = ("subject_id", "visit_id", "time_hours")


def summary_target_names() -> list[str]:
    """Column names of the target panel, in the order the head predicts them."""
    names = [f"{label}.{stat}" for label, _ in SIGNAL_PANEL for stat in SIGNAL_STATS]
    names += [
        f"drug.{label}.{stat}" for label, _ in DRUG_CLASSES for stat in COUNT_STATS
    ]
    names += [
        f"family.{label}.{stat}" for label in FAMILY_LABELS for stat in COUNT_STATS
    ]
    return names


def count_target_mask(names: Sequence[str] | None = None) -> np.ndarray:
    """Boolean mask over the panel: which targets are occurrence counts."""
    names = list(names) if names is not None else summary_target_names()
    return np.array([n.rsplit(".", 1)[-1] in COUNT_STATS for n in names])


def landmark_rows(
    events: pl.DataFrame, landmark_hours: float = DEFAULT_LANDMARK_HOURS
) -> tuple[list[int], list[int], list[float]]:
    """``(subject_ids, visit_ids, times)`` of the landmark rows of every visit.

    Every ``landmark_hours`` from a visit's first event to its last, the
    row is the visit's last event time at or before that instant, so a row
    time always coincides with a real token's time stamp (hours since the
    subject's first event, like :class:`~odyssey.data.sequences.PatientSequence`).
    """
    origins = origin_hours(events)
    timed = (
        events.filter(pl.col("time").is_not_null() & pl.col("hadm_id").is_not_null())
        .join(origins, on="subject_id", how="left")
        .with_columns(
            ((pl.col("time") - pl.col("_origin")).dt.total_seconds() / 3600.0).alias(
                "_hours"
            )
        )
        .group_by("subject_id", "hadm_id")
        .agg(pl.col("_hours").unique().sort().alias("_hours"))
    )
    sids: list[int] = []
    vids: list[int] = []
    times: list[float] = []
    for sid, vid, hours in zip(
        timed["subject_id"].to_list(),
        timed["hadm_id"].to_list(),
        timed["_hours"].to_list(),
    ):
        arr = np.asarray(hours, dtype=np.float64)
        grid = np.arange(arr[0], arr[-1] + 1e-9, landmark_hours)
        idx = np.searchsorted(arr, grid, side="right") - 1
        picked = np.unique(arr[idx])
        sids.extend([int(sid)] * len(picked))
        vids.extend([int(vid)] * len(picked))
        times.extend(float(t) for t in picked)
    return sids, vids, times


def compute_summary_targets(
    events: pl.DataFrame,
    *,
    source: str = "mimic_iv",
    landmark_hours: float = DEFAULT_LANDMARK_HOURS,
) -> pl.DataFrame:
    """Raw (unstandardized) targets at every landmark row of ``events``.

    Columns: ``subject_id``, ``visit_id``, ``time_hours`` and one column per
    :func:`summary_target_names` entry; NaN where a window is empty.
    """
    names = summary_target_names()
    all_names = feature_names()
    idx = [all_names.index(n) for n in names]
    sids, vids, times = landmark_rows(events, landmark_hours)
    if not sids:
        return pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "visit_id": pl.Int64,
                "time_hours": pl.Float64,
                **dict.fromkeys(names, pl.Float32),
            }
        )
    builder = StrongFeatureBuilder(events, source=source)
    values = builder.features(sids, vids, times)[:, idx].astype(np.float32)
    frame = pl.DataFrame(
        {
            "subject_id": pl.Series(sids, dtype=pl.Int64),
            "visit_id": pl.Series(vids, dtype=pl.Int64),
            "time_hours": pl.Series(times, dtype=pl.Float64),
        }
    )
    return frame.with_columns(
        [pl.Series(n, values[:, i], dtype=pl.Float32) for i, n in enumerate(names)]
    )


@dataclass(frozen=True)
class SummaryTargetStats:
    """Per-column standardization fitted on the training targets."""

    names: list[str]
    lo: np.ndarray
    """Winsorizing floor per column (after ``log1p`` for counts)."""
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Standardize raw targets ``(n, K)``; NaN stays NaN."""
        x = np.asarray(raw, dtype=np.float64).copy()
        counts = count_target_mask(self.names)
        x[:, counts] = np.log1p(np.clip(x[:, counts], 0.0, None))
        x = np.clip(x, self.lo, self.hi)
        out: np.ndarray = ((x - self.mean) / self.std).astype(np.float32)
        return out

    def save(self, path: str | Path) -> None:
        """Write the stats as JSON."""
        Path(path).write_text(
            json.dumps(
                {
                    "names": self.names,
                    "lo": self.lo.tolist(),
                    "hi": self.hi.tolist(),
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> SummaryTargetStats:
        """Read stats written by :meth:`save`."""
        d = json.loads(Path(path).read_text())
        return cls(
            names=list(d["names"]),
            lo=np.asarray(d["lo"], dtype=np.float64),
            hi=np.asarray(d["hi"], dtype=np.float64),
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
        )


def fit_summary_stats(frames: Iterable[pl.DataFrame]) -> SummaryTargetStats:
    """Fit :class:`SummaryTargetStats` over the raw target frames of a split.

    Counts go through ``log1p``; every column is winsorized at
    :data:`WINSOR_PERCENTILES` of its finite values before mean/std, so a
    handful of sentinel readings cannot set the scale. A column with no
    finite value, or no spread, gets mean 0 and std 1.
    """
    names = summary_target_names()
    counts = count_target_mask(names)
    parts = [f.select(names).to_numpy().astype(np.float64) for f in frames]
    x = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(names)))
    x[:, counts] = np.log1p(np.clip(x[:, counts], 0.0, None))
    lo = np.zeros(len(names))
    hi = np.ones(len(names))
    mean = np.zeros(len(names))
    std = np.ones(len(names))
    for j in range(len(names)):
        col = x[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        lo[j], hi[j] = np.percentile(col, WINSOR_PERCENTILES)
        clipped = np.clip(col, lo[j], hi[j])
        mean[j] = clipped.mean()
        s = clipped.std()
        std[j] = s if s > 0 else 1.0
    return SummaryTargetStats(names=names, lo=lo, hi=hi, mean=mean, std=std)


class SummaryTargetTables:
    """Standardized targets per ``(subject, visit)``, queried per chunk.

    Each key maps to ``(times, values)``: sorted landmark row times and the
    matching ``(rows, K)`` standardized targets, stored as float16 to keep
    a whole split (millions of rows) in memory.
    """

    def __init__(self, stats: SummaryTargetStats) -> None:
        self.stats = stats
        self.num_targets = len(stats.names)
        self._rows: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        """Return the number of visits with targets."""
        return len(self._rows)

    def add_frame(self, frame: pl.DataFrame) -> None:
        """Ingest one raw target frame from :func:`compute_summary_targets`."""
        if frame.height == 0:
            return
        frame = frame.sort("subject_id", "visit_id", "time_hours")
        values = self.stats.transform(frame.select(self.stats.names).to_numpy())
        keys = frame.select("subject_id", "visit_id").to_numpy()
        times = frame["time_hours"].to_numpy().astype(np.float64)
        change = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(keys)]])
        for a, b in zip(starts, ends):
            key = (int(keys[a, 0]), int(keys[a, 1]))
            self._rows[key] = (times[a:b], values[a:b].astype(np.float16))

    @classmethod
    def from_dir(
        cls, directory: str | Path, stats: SummaryTargetStats
    ) -> SummaryTargetTables:
        """Load every ``*.parquet`` under ``directory``."""
        tables = cls(stats)
        for path in sorted(Path(directory).glob("*.parquet")):
            tables.add_frame(pl.read_parquet(path))
        return tables

    def lookup(
        self, subject_id: int, visit_id: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(times, values)`` for one visit, or ``None``."""
        return self._rows.get((subject_id, visit_id))


@dataclass
class SummaryTargets:
    """``(lanes, T, K)`` standardized targets and where they apply."""

    values: torch.Tensor
    mask: torch.Tensor


def _bundle_end_mask(
    time_stamps: np.ndarray, subject_ids: np.ndarray, real: np.ndarray
) -> np.ndarray:
    """Mark the last real token of each same-time bundle, per lane."""
    lanes, chunk_len = time_stamps.shape
    end = np.zeros_like(real)
    if chunk_len == 0:
        return end
    same_next = np.zeros((lanes, chunk_len), dtype=bool)
    same_next[:, :-1] = (
        (time_stamps[:, 1:] == time_stamps[:, :-1])
        & (subject_ids[:, 1:] == subject_ids[:, :-1])
        & real[:, 1:]
    )
    return real & ~same_next


def summary_targets_for_chunk(
    chunk: StreamingChunk, tables: SummaryTargetTables
) -> SummaryTargets | None:
    """Targets for one chunk, or ``None`` when no position carries one.

    A position carries a target when its ``(subject, visit, time)`` is a
    landmark row of the tables and it is the last real token of its
    bundle, so the state has read the whole bundle the row summarizes.
    """
    sids = chunk.subject_ids.detach().cpu().numpy()
    vids = chunk.visit_ids.detach().cpu().numpy()
    times = chunk.batch.aux.time_stamps.detach().cpu().numpy().astype(np.float64)
    real = chunk.real_mask.detach().cpu().numpy().astype(bool)
    lanes, chunk_len = sids.shape
    k = tables.num_targets
    values = np.zeros((lanes, chunk_len, k), dtype=np.float32)
    mask = np.zeros((lanes, chunk_len), dtype=bool)
    ends = _bundle_end_mask(times, sids, real)
    keys = np.stack([sids.reshape(-1), vids.reshape(-1)], axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    for i, (s, v) in enumerate(unique_keys.tolist()):
        rows = tables.lookup(int(s), int(v))
        if rows is None:
            continue
        row_times, row_values = rows
        flat = np.flatnonzero((inverse == i) & ends.reshape(-1))
        if flat.size == 0:
            continue
        t = times.reshape(-1)[flat]
        pos = np.searchsorted(row_times, t)
        pos = np.clip(pos, 0, len(row_times) - 1)
        hit = np.isclose(row_times[pos], t, rtol=0.0, atol=1e-6)
        if not hit.any():
            continue
        lane_idx, pos_idx = np.divmod(flat[hit], chunk_len)
        values[lane_idx, pos_idx] = row_values[pos[hit]].astype(np.float32)
        mask[lane_idx, pos_idx] = True
    if not mask.any():
        return None
    device = chunk.batch.concept_ids.device
    finite = np.isfinite(values)
    full_mask = finite & mask[:, :, None]
    values = np.where(full_mask, values, 0.0).astype(np.float32)
    return SummaryTargets(
        values=torch.from_numpy(values).to(device),
        mask=torch.from_numpy(full_mask).to(device),
    )


def load_summary_tables(
    directory: str | Path, stats_path: str | Path | None = None
) -> SummaryTargetTables:
    """Load a split's tables; the stats default to ``<directory>/../stats.json``."""
    directory = Path(directory)
    stats = SummaryTargetStats.load(
        stats_path if stats_path is not None else directory.parent / "stats.json"
    )
    return SummaryTargetTables.from_dir(directory, stats)


__all__ = [
    "COUNT_STATS",
    "DEFAULT_LANDMARK_HOURS",
    "SIGNAL_STATS",
    "SummaryTargetStats",
    "SummaryTargetTables",
    "SummaryTargets",
    "compute_summary_targets",
    "count_target_mask",
    "fit_summary_stats",
    "landmark_rows",
    "load_summary_tables",
    "summary_target_names",
    "summary_targets_for_chunk",
]
