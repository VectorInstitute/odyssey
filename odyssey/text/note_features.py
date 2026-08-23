"""Note-embedding features for the tabular baselines (the text-modality probe).

The cheap headroom test of Track A item 7: before any fusion work, hand the
tuned GBM the pooled note embeddings as extra features at each index row
and see whether the alerts move. Features are built from the active
``note_embeddings`` sidecar (:mod:`odyssey.data.sidecars`;
:mod:`odyssey.text.embed_notes` writes it) with the same "strictly before
the index time" rule every other baseline feature obeys:

- ``note_n_24h`` / ``note_n_visit``: notes charted in the trailing 24 h /
  since the visit start;
- ``note_hours_since_last``: staleness of the newest note (NaN if none);
- ``note_mean_pca_<k>``: mean PCA-reduced embedding over the notes of the
  trailing ``window_hours`` (default 7 days);
- ``note_last_pca_<k>``: the newest note's PCA-reduced embedding.

Only the PCA columns (``embedding_pca``) are used, never the full vector:
the probe question is "is there signal", and a few dozen dense columns is
what a GBM can use.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from odyssey.data.alert_events import origin_hours
from odyssey.data.sidecars import active_sidecar
from odyssey.text.embed_notes import PCA_COL


# Sidecar name the embedder writes (sidecars/note_embeddings.parquet).
NOTE_EMBEDDINGS = "note_embeddings"

NOTE_COUNT_STATS: Tuple[str, ...] = (
    "note_n_24h",
    "note_n_visit",
    "note_hours_since_last",
)


def note_feature_names(pca_dim: int) -> List[str]:
    """Column names of :func:`note_features` for a ``pca_dim``-wide sidecar."""
    return (
        list(NOTE_COUNT_STATS)
        + [f"note_mean_pca_{i}" for i in range(pca_dim)]
        + [f"note_last_pca_{i}" for i in range(pca_dim)]
    )


def active_note_embeddings() -> pl.DataFrame:
    """Return the active ``note_embeddings`` sidecar or raise a clear error."""
    table = active_sidecar(NOTE_EMBEDDINGS)
    if table is None:
        raise RuntimeError(
            "the 'strong_text' feature set needs the note_embeddings sidecar "
            "(<root>/sidecars/note_embeddings.parquet, from "
            "python -m odyssey.text.embed_notes); none is active -- see "
            "odyssey.data.sidecars.activate_sidecars"
        )
    if PCA_COL not in table.columns:
        raise RuntimeError(
            f"note_embeddings sidecar has no {PCA_COL!r} column (run with --pca)"
        )
    return table


def pca_dim_of(table: pl.DataFrame) -> int:
    """Width of the sidecar's PCA column."""
    first = table[PCA_COL][0]
    return int(len(first))


class NoteFeatureBuilder:
    """Per-subject sorted note arrays; feature rows by binary search."""

    def __init__(
        self,
        events: pl.DataFrame,
        embeddings: pl.DataFrame,
        *,
        window_hours: float = 24.0 * 7,
    ) -> None:
        """Index notes by subject, in hours on each subject's event time origin."""
        self.window_hours = window_hours
        self.pca_dim = pca_dim_of(embeddings)
        self.names = note_feature_names(self.pca_dim)
        origins = origin_hours(events)
        subjects = set(origins["subject_id"].to_list())
        joined = (
            embeddings.filter(pl.col("subject_id").is_in(sorted(subjects)))
            .join(origins, on="subject_id", how="inner")
            .with_columns(
                (
                    (pl.col("charttime") - pl.col("_origin")).dt.total_seconds()
                    / 3600.0
                ).alias("_hours")
            )
            .sort(["subject_id", "_hours"])
        )
        self._by_subject: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for sid, group in joined.group_by("subject_id", maintain_order=True):
            hours = group["_hours"].to_numpy().astype(np.float64)
            vecs = np.asarray(group[PCA_COL].to_list(), dtype=np.float32)
            self._by_subject[int(sid[0] if isinstance(sid, tuple) else sid)] = (
                hours,
                vecs,
            )

    def features(
        self,
        subject_ids: Sequence[int],
        visit_starts: Sequence[float],
        times: Sequence[float],
    ) -> np.ndarray:
        """Feature matrix ``(n_rows, len(self.names))``; NaN where no notes apply."""
        n = len(subject_ids)
        k = self.pca_dim
        out = np.full((n, len(self.names)), np.nan, dtype=np.float32)
        sids = np.asarray(subject_ids)
        now_all = np.asarray(times, dtype=np.float64)
        vs_all = np.asarray(visit_starts, dtype=np.float64)
        for sid in np.unique(sids):
            entry = self._by_subject.get(int(sid))
            if entry is None:
                continue
            hours, vecs = entry
            cum = np.vstack(
                [np.zeros((1, k), dtype=np.float64), np.cumsum(vecs, axis=0)]
            )
            rows = np.nonzero(sids == sid)[0]
            now = now_all[rows]
            hi = np.searchsorted(hours, now, side="left")  # strictly before
            lo24 = np.searchsorted(hours, now - 24.0, side="left")
            lo_v = np.searchsorted(hours, vs_all[rows], side="left")
            lo_w = np.searchsorted(hours, now - self.window_hours, side="left")
            block = np.full((len(rows), len(self.names)), np.nan, dtype=np.float64)
            block[:, 0] = hi - lo24
            block[:, 1] = hi - np.minimum(lo_v, hi)
            has = hi > 0
            block[has, 2] = now[has] - hours[hi[has] - 1]
            counts = (hi - lo_w).astype(np.float64)
            mean_ok = counts > 0
            if mean_ok.any():
                sums = cum[hi[mean_ok]] - cum[lo_w[mean_ok]]
                block[np.nonzero(mean_ok)[0], 3 : 3 + k] = sums / counts[mean_ok, None]
            if has.any():
                block[np.nonzero(has)[0], 3 + k : 3 + 2 * k] = vecs[hi[has] - 1]
            out[rows] = block.astype(np.float32)
        return out


def note_features_for_rows(
    events: pl.DataFrame,
    subject_ids: Sequence[int],
    visit_starts: Sequence[float],
    times: Sequence[float],
    *,
    embeddings: Optional[pl.DataFrame] = None,
    window_hours: float = 24.0 * 7,
) -> Tuple[np.ndarray, List[str]]:
    """Build once and return ``(matrix, names)`` for the given rows."""
    table = embeddings if embeddings is not None else active_note_embeddings()
    builder = NoteFeatureBuilder(events, table, window_hours=window_hours)
    return builder.features(subject_ids, visit_starts, times), builder.names


__all__ = [
    "NOTE_COUNT_STATS",
    "NOTE_EMBEDDINGS",
    "NoteFeatureBuilder",
    "active_note_embeddings",
    "note_feature_names",
    "note_features_for_rows",
    "pca_dim_of",
]
