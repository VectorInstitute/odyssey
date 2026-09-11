"""Find and load one patient's record from a directory of MEDS shards.

Loading a whole split (37 held-out shards, tens of millions of rows) to
show one patient is wasteful. Instead the store reads only the
``subject_id`` column of every shard once, keeps a ``subject -> shard``
index, and loads a subject's rows on demand with a filtered scan. Loaded
records are normalized exactly as the run's training pipeline did
(medication normalization, history recap) and kept in a small LRU cache.
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import polars as pl

from apps.clinician_demo.schemas import PatientSummary, VisitSummary
from odyssey.data.alert_events import visit_envelope
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import BIRTH_CODE, HOURS_PER_YEAR
from odyssey.training.data import load_meds_subject, shard_sort_key


logger = logging.getLogger(__name__)

TRAINED_SPLITS = frozenset({"train", "tuning"})
_ADMISSION_PREFIX = "HOSPITAL_ADMISSION//"


class UnknownPatientError(KeyError):
    """The subject is not in any loaded shard."""


def discover_shards(
    data_dir: str | Path, *, max_shards: int | None = None
) -> list[Path]:
    """Find all ``*.parquet`` shards under ``data_dir``, in a stable order.

    Searches recursively, so a MEDS ``data/`` root with ``train/``,
    ``tuning/`` and ``held_out/`` beneath it works as well as one split
    directory; hidden directories (extractor working state) are skipped.
    Ordered by (directory, numeric shard index).

    Raises
    ------
    FileNotFoundError
        If no shard is found.
    """
    root = Path(data_dir)
    shards = [
        p
        for p in root.rglob("*.parquet")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]
    shards.sort(key=lambda p: (str(p.parent), shard_sort_key(p)))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"no .parquet shards under {root}")
    return shards


def build_shard_index(shards: Sequence[Path]) -> dict[int, Path]:
    """Map ``subject_id -> shard`` by reading only the ``subject_id`` column.

    Raises
    ------
    ValueError
        If a subject appears in two shards (MEDS shards partition subjects;
        a duplicate means the directory mixes extractions).
    """
    index: dict[int, Path] = {}
    for shard in shards:
        ids = pl.scan_parquet(shard).select("subject_id").unique().collect()
        for sid in ids["subject_id"].to_list():
            if sid in index:
                raise ValueError(f"subject {sid} is in both {index[sid]} and {shard}")
            index[int(sid)] = shard
    return index


def load_splits(path: str | Path | None) -> dict[int, str]:
    """Map ``subject_id -> split`` from ``subject_splits.parquet`` (empty if none)."""
    if path is None or not Path(path).exists():
        return {}
    frame = pl.read_parquet(path, columns=["subject_id", "split"])
    return dict(
        zip(frame["subject_id"].to_list(), frame["split"].to_list(), strict=True)
    )


class PatientStore:
    """Per-subject access to normalized raw MEDS events, with an LRU cache."""

    def __init__(
        self,
        index: Mapping[int, Path],
        *,
        source: str,
        normalize_medications: bool,
        history_recap: bool = False,
        splits: Mapping[int, str] | None = None,
        describe: Callable[[str], str] = str,
        cache_size: int = 32,
    ) -> None:
        """Wrap a shard index; ``describe`` turns an admission code into a label."""
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        self._index = dict(index)
        self._source = source
        self._normalize = normalize_medications
        self._recap = history_recap
        self._splits = dict(splits or {})
        self._describe = describe
        self._cache_size = cache_size
        self._cache: OrderedDict[int, pl.DataFrame] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def subject_ids(self) -> list[int]:
        """Every subject in the loaded shards, ascending."""
        return sorted(self._index)

    def __contains__(self, subject_id: object) -> bool:
        """Check whether ``subject_id`` is in a loaded shard."""
        return subject_id in self._index

    def __len__(self) -> int:
        """Return the number of subjects indexed."""
        return len(self._index)

    def split(self, subject_id: int) -> str | None:
        """Return the model's training split for this subject, if known."""
        return self._splits.get(subject_id)

    def seen_in_training(self, subject_id: int) -> bool:
        """Tell whether the model was trained or tuned on this subject."""
        return self._splits.get(subject_id) in TRAINED_SPLITS

    def raw_events(self, subject_id: int) -> pl.DataFrame:
        """Return the subject's events, normalized the way the run's training was.

        Raises
        ------
        UnknownPatientError
            If the subject is in no loaded shard.
        """
        with self._lock:
            cached = self._cache.get(subject_id)
            if cached is not None:
                self._cache.move_to_end(subject_id)
                return cached
        shard = self._index.get(subject_id)
        if shard is None:
            raise UnknownPatientError(subject_id)
        events = load_meds_subject(shard, subject_id)
        events = maybe_normalize(events, enabled=self._normalize, source=self._source)
        events = maybe_history_recap(events, enabled=self._recap)
        with self._lock:
            self._cache[subject_id] = events
            self._cache.move_to_end(subject_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return events

    def visits(self, subject_id: int) -> list[VisitSummary]:
        """Return the subject's admissions in time order (hours since first event)."""
        events = self.raw_events(subject_id)
        spans = visit_envelope(events)
        with_visit = events.filter(pl.col("hadm_id").is_not_null())
        counts = dict(
            with_visit.group_by("hadm_id").len().iter_rows()  # (hadm_id, n)
        )
        admissions = (
            with_visit.filter(pl.col("code").str.starts_with(_ADMISSION_PREFIX))
            .sort("time", maintain_order=True)
            .group_by("hadm_id", maintain_order=True)
            .agg(pl.col("code").first())
        )
        admission_code = dict(admissions.iter_rows())
        out = [
            VisitSummary(
                visit_id=vid,
                start_hours=start,
                end_hours=end,
                admission=(
                    self._describe(admission_code[vid])
                    if vid in admission_code
                    else "Admission"
                ),
                n_events=int(counts.get(vid, 0)),
            )
            for (sid, vid), (start, end) in spans.items()
            if sid == subject_id
        ]
        return sorted(out, key=lambda v: (v.start_hours, v.visit_id))

    def summary(self, subject_id: int) -> PatientSummary:
        """Return header facts (sex, age at first event, training split) and visits."""
        events = self.raw_events(subject_id)
        sex_rows = events.filter(pl.col("code").str.starts_with("GENDER//"))
        sex = sex_rows["code"][0].split("//", 1)[1] if sex_rows.height else None
        timed = events.filter(pl.col("time").is_not_null())
        birth = timed.filter(pl.col("code") == BIRTH_CODE)
        first = timed.filter(pl.col("code") != BIRTH_CODE)["time"].min()
        age = None
        if birth.height and first is not None:
            seconds = (first - birth["time"][0]).total_seconds()
            age = seconds / 3600.0 / HOURS_PER_YEAR
        return PatientSummary(
            subject_id=subject_id,
            split=self.split(subject_id),
            seen_in_training=self.seen_in_training(subject_id),
            sex=sex,
            age_years=age,
            visits=self.visits(subject_id),
        )


__all__ = [
    "TRAINED_SPLITS",
    "PatientStore",
    "UnknownPatientError",
    "build_shard_index",
    "discover_shards",
    "load_splits",
]
