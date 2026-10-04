"""'Why': which recorded items the forecast leans on, found by occlusion.

For a chosen moment and target (an event's risk or a concept's belief),
each candidate code recorded in the lookback window is removed in turn and
the record re-scored (:func:`odyssey.inference.concept_edit_attribution.occlude_codes`).
Codes whose removal moves the target most are what the forecast rests on.

One re-score per candidate makes this slow (seconds to a minute), so it
runs as a background job the UI polls, one job at a time on the GPU.
Removing a normal reading pushes a forecast toward the population
average, so the size of a change is more trustworthy than its direction.
"""

import itertools
import logging
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import polars as pl

from apps.clinician_demo.schemas import EvidenceItem, EvidenceJob


logger = logging.getLogger(__name__)

MAX_CANDIDATES = 40
TOP_ITEMS = 8
ProgressFn = Callable[[int, int], None]
EvidenceWork = Callable[[ProgressFn], list[EvidenceItem]]


def candidate_codes(
    raw_events: pl.DataFrame,
    *,
    index_time: object,
    lookback_hours: float,
    limit: int = MAX_CANDIDATES,
) -> list[str]:
    """Return the most frequent codes in ``(index_time - lookback, index_time]``.

    Ordered by count, then code, so the choice is deterministic; the cap
    bounds the job's cost (one re-score per candidate).
    """
    window = raw_events.filter(
        pl.col("time").is_not_null()
        & (pl.col("time") <= pl.lit(index_time))
        & (pl.col("time") > pl.lit(index_time) - pl.duration(hours=lookback_hours))
    )
    if window.height == 0:
        return []
    counts = (
        window.group_by("code").len().sort(["len", "code"], descending=[True, False])
    )
    return counts["code"].head(limit).to_list()


@dataclass
class _Job:
    job_id: str
    target: str
    note: str
    status: str = "pending"
    done: int = 0
    total: int = 0
    result: list[EvidenceItem] = field(default_factory=list)
    error: str | None = None

    def snapshot(self) -> EvidenceJob:
        return EvidenceJob(
            job_id=self.job_id,
            status=self.status,
            done=self.done,
            total=self.total,
            target=self.target,
            result=list(self.result),
            error=self.error,
            note=self.note,
        )


def disambiguate(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Suffix repeated labels with their code's last part so rows stay distinct.

    Two different recorded items can share a dictionary name (two infusion
    items both called "Dextrose 5%"); without a suffix they read as a
    duplicated row.
    """
    counts = Counter(i.label for i in items)
    return [
        replace(i, label=f"{i.label} ({i.code.rsplit('//', 1)[-1]})")
        if counts[i.label] > 1
        else i
        for i in items
    ]


class EvidenceRunner:
    """Runs evidence searches one at a time in a background thread."""

    def __init__(self, gpu_lock: threading.Lock, *, max_jobs: int = 64) -> None:
        """Share ``gpu_lock`` with every other model call of the service."""
        self._gpu_lock = gpu_lock
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[str, _Job] = OrderedDict()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence")

    def submit(self, target: str, note: str, work: EvidenceWork) -> EvidenceJob:
        """Queue ``work`` (called with a progress callback) and return its job."""
        with self._lock:
            job = _Job(job_id=f"ev{next(self._ids)}", target=target, note=note)
            self._jobs[job.job_id] = job
            self._evict()
            snapshot = job.snapshot()
        self._pool.submit(self._run, job, work)
        return snapshot

    def get(self, job_id: str) -> EvidenceJob:
        """Return the current state of a job.

        Raises
        ------
        KeyError
            If the job is unknown (never submitted, or evicted).
        """
        with self._lock:
            return self._jobs[job_id].snapshot()

    def shutdown(self) -> None:
        """Stop accepting work and wait for the running job."""
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _evict(self) -> None:
        finished = [k for k, j in self._jobs.items() if j.status in ("done", "error")]
        while len(self._jobs) > self._max_jobs and finished:
            self._jobs.pop(finished.pop(0))

    def _progress(self, job: _Job) -> ProgressFn:
        def update(done: int, total: int) -> None:
            with self._lock:
                job.done, job.total = done, total

        return update

    def _run(self, job: _Job, work: EvidenceWork) -> None:
        with self._lock:
            job.status = "running"
        try:
            with self._gpu_lock:
                items = work(self._progress(job))
        except Exception as exc:  # noqa: BLE001 -- reported to the UI, logged here
            logger.exception("[evidence] job %s failed", job.job_id)
            with self._lock:
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            job.result = disambiguate(items[:TOP_ITEMS])
            job.done = max(job.done, job.total)
            job.status = "done"


def cached_or_submit(
    cache: dict[tuple[object, ...], str],
    key: tuple[object, ...],
    runner: EvidenceRunner,
    submit: Callable[[], EvidenceJob],
) -> EvidenceJob:
    """Reuse the live job for ``key``; resubmit after an error or eviction."""
    job_id = cache.get(key)
    if job_id is not None:
        try:
            job = runner.get(job_id)
        except KeyError:
            job = None
        if job is not None and job.status != "error":
            return job
    job = submit()
    cache[key] = job.job_id
    return replace(job)


__all__ = [
    "MAX_CANDIDATES",
    "TOP_ITEMS",
    "EvidenceRunner",
    "cached_or_submit",
    "disambiguate",
    "candidate_codes",
]
