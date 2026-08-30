"""Note-embedding features for the tabular baselines (the text-modality probe).

The cheap headroom test of Track A item 7: before any fusion work, hand the
tuned GBM the pooled note embeddings as extra features at each index row
and see whether the alerts move. Features are built from the active
``note_embeddings`` sidecar (:mod:`odyssey.data.sidecars`;
:mod:`odyssey.text.embed_notes` writes it), indexed by AVAILABILITY time:

- ``note_n_24h`` / ``note_n_visit``: notes available in the trailing 24 h /
  since the visit start;
- ``note_hours_since_last``: staleness of the newest available note (NaN
  if none);
- ``note_mean_pca_<k>``: mean PCA-reduced embedding over the notes of the
  trailing ``window_hours`` (default 7 days);
- ``note_last_pca_<k>``: the newest available note's PCA-reduced embedding.

Availability = ``max(charttime, storetime)`` (2026-08-30 fix): a note's
text only exists in the record at ``storetime`` (finalization), which on
the real MIMIC-IV-Note release lags ``charttime`` by a median 2.25 h for
radiology (22% > 6 h, 6% > 24 h) and ~20 h for discharge summaries (41%
> 24 h). Indexing by ``charttime`` alone -- the pre-fix behaviour --
handed the baseline text hours-to-days before the real system had it, the
classic MIMIC note-leakage trap. A sidecar without a ``storetime`` column
predates the fix and is refused with a rebuild instruction rather than
silently reintroducing the leak. Windows are half-open ``(t - w, t]``,
the same landmark-protocol-v4 boundary every other baseline feature uses.

Only the PCA columns (``embedding_pca``) are used, never the full vector:
the probe question is "is there signal", and a few dozen dense columns is
what a GBM can use.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl

from odyssey.data.alert_events import origin_hours
from odyssey.data.sidecars import active_sidecar
from odyssey.text.embed_notes import PCA_COL


# Sidecar name the embedder writes (sidecars/note_embeddings.parquet).
NOTE_EMBEDDINGS = "note_embeddings"

NOTE_COUNT_STATS: tuple[str, ...] = (
    "note_n_24h",
    "note_n_visit",
    "note_hours_since_last",
)


def note_feature_names(pca_dim: int) -> list[str]:
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
        """Index notes by subject, in hours on each subject's event time origin.

        Notes are indexed at their AVAILABILITY time,
        ``max(charttime, storetime)`` -- see the module docstring. A
        sidecar without ``storetime`` predates the 2026-08-30 fix and is
        refused: falling back to ``charttime`` would silently reintroduce
        the note-leakage this exists to prevent.
        """
        self.window_hours = window_hours
        self.pca_dim = pca_dim_of(embeddings)
        self.names = note_feature_names(self.pca_dim)
        if "storetime" not in embeddings.columns:
            raise RuntimeError(
                "note_embeddings sidecar has no 'storetime' column -- it was "
                "built before the availability-time fix (2026-08-30) and its "
                "charttime stamps precede the text's real existence by hours "
                "to days on MIMIC-IV-Note. Rebuild the notes sidecar "
                "(scripts/build_mimic_note_sidecar.py) and re-embed "
                "(python -m odyssey.text.embed_notes)."
            )
        origins = origin_hours(events)
        subjects = set(origins["subject_id"].to_list())
        joined = (
            embeddings.filter(pl.col("subject_id").is_in(sorted(subjects)))
            .join(origins, on="subject_id", how="inner")
            .with_columns(
                pl.max_horizontal("charttime", "storetime").alias("_available")
            )
            .with_columns(
                (
                    (pl.col("_available") - pl.col("_origin")).dt.total_seconds()
                    / 3600.0
                ).alias("_hours")
            )
            .sort(["subject_id", "_hours"])
        )
        self._by_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
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
            # (t - w, t] windows on availability time (protocol v4).
            hi = np.searchsorted(hours, now, side="right")
            lo24 = np.searchsorted(hours, now - 24.0, side="right")
            lo_v = np.searchsorted(hours, vs_all[rows], side="left")
            lo_w = np.searchsorted(hours, now - self.window_hours, side="right")
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
    embeddings: pl.DataFrame | None = None,
    window_hours: float = 24.0 * 7,
) -> tuple[np.ndarray, list[str]]:
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
