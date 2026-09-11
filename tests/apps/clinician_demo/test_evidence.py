"""Evidence: candidate choice and the background job runner."""

import threading
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from apps.clinician_demo.evidence import (
    TOP_ITEMS,
    EvidenceRunner,
    cached_or_submit,
    candidate_codes,
    disambiguate,
)
from apps.clinician_demo.schemas import EvidenceItem, EvidenceJob


T = datetime(2150, 1, 1, 12, 0)


def _wait(runner: EvidenceRunner, job_id: str, timeout: float = 5.0) -> EvidenceJob:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job.status in ("done", "error"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def _item(code: str, delta: float) -> EvidenceItem:
    return EvidenceItem(code, code, 1, 0.5, 0.5 + delta, delta)


def test_candidates_are_the_most_frequent_codes_inside_the_window() -> None:
    events = pl.DataFrame(
        {
            "time": [T - timedelta(hours=h) for h in (0, 0, 1, 2, 2, 2, 5, 30)]
            + [None],
            "code": ["B", "A", "A", "C", "C", "C", "D", "OLD", "GENDER//F"],
        }
    )
    assert candidate_codes(events, index_time=T, lookback_hours=24.0) == [
        "C",
        "A",
        "B",
        "D",
    ]
    assert candidate_codes(events, index_time=T, lookback_hours=24.0, limit=2) == [
        "C",
        "A",
    ]
    # the window is (T - lookback, T]: a row exactly lookback hours back is out
    assert candidate_codes(events, index_time=T, lookback_hours=5.0) == ["C", "A", "B"]
    assert (
        candidate_codes(events, index_time=T - timedelta(days=5), lookback_hours=1.0)
        == []
    )


def test_a_job_runs_under_the_gpu_lock_reports_progress_and_keeps_the_top_items() -> (
    None
):
    lock = threading.Lock()
    runner = EvidenceRunner(lock)
    seen_locked: list[bool] = []

    def work(progress):  # type: ignore[no-untyped-def]
        seen_locked.append(lock.locked())
        for i in range(1, 4):
            progress(i, 3)
        return [_item(f"C{i}", -i / 100) for i in range(TOP_ITEMS + 5)]

    job = runner.submit("target", "note", work)
    assert job.status == "pending" and job.target == "target" and job.note == "note"
    done = _wait(runner, job.job_id)
    assert done.status == "done" and (done.done, done.total) == (3, 3)
    assert len(done.result) == TOP_ITEMS
    assert seen_locked == [True]
    runner.shutdown()


def test_a_failing_job_reports_the_error_and_the_runner_keeps_going() -> None:
    runner = EvidenceRunner(threading.Lock())

    def boom(progress):  # type: ignore[no-untyped-def]
        raise RuntimeError("out of memory")

    failed = _wait(runner, runner.submit("t", "n", boom).job_id)
    assert failed.status == "error" and failed.error == "RuntimeError: out of memory"
    ok = _wait(runner, runner.submit("t", "n", lambda p: []).job_id)
    assert ok.status == "done" and ok.result == []
    runner.shutdown()


def test_unknown_jobs_raise_and_old_finished_jobs_are_evicted() -> None:
    runner = EvidenceRunner(threading.Lock(), max_jobs=2)
    with pytest.raises(KeyError):
        runner.get("ev999")
    ids = [runner.submit("t", "n", lambda p: []).job_id for _ in range(4)]
    _wait(runner, ids[-1])
    runner.submit("t", "n", lambda p: [])  # triggers eviction of finished jobs
    with pytest.raises(KeyError):
        runner.get(ids[0])
    runner.shutdown()


def test_cached_or_submit_reuses_live_jobs_and_retries_failed_or_evicted_ones() -> None:
    runner = EvidenceRunner(threading.Lock(), max_jobs=1)
    cache: dict[tuple[object, ...], str] = {}
    calls: list[int] = []

    def submit_ok() -> EvidenceJob:
        calls.append(1)
        return runner.submit("t", "n", lambda p: [])

    first = cached_or_submit(cache, ("k",), runner, submit_ok)
    _wait(runner, first.job_id)
    again = cached_or_submit(cache, ("k",), runner, submit_ok)
    assert again.job_id == first.job_id and len(calls) == 1

    def submit_fail() -> EvidenceJob:
        calls.append(1)
        return runner.submit("t", "n", lambda p: (_ for _ in ()).throw(ValueError("x")))

    failed = cached_or_submit(cache, ("f",), runner, submit_fail)
    _wait(runner, failed.job_id)
    retried = cached_or_submit(cache, ("f",), runner, submit_ok)
    assert retried.job_id != failed.job_id

    cache[("gone",)] = "ev-evicted"
    fresh = cached_or_submit(cache, ("gone",), runner, submit_ok)
    assert fresh.job_id != "ev-evicted" and cache[("gone",)] == fresh.job_id
    runner.shutdown()


def test_repeated_labels_get_their_item_number_and_unique_ones_stay() -> None:
    items = [
        EvidenceItem("INFUSION_START//220949", "Dextrose 5%", 7, 0.3, 0.29, -0.01),
        EvidenceItem("INFUSION_START//225823", "Dextrose 5%", 9, 0.3, 0.31, 0.01),
        EvidenceItem("LAB//220045//bpm", "Heart Rate", 27, 0.3, 0.27, -0.03),
    ]
    assert [i.label for i in disambiguate(items)] == [
        "Dextrose 5% (220949)",
        "Dextrose 5% (225823)",
        "Heart Rate",
    ]
    assert disambiguate([]) == []
