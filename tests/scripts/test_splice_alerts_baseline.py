"""Splicing a baseline across runs is refused unless the rows agree."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.splice_alerts_baseline import check_rows_align, splice


def _rec(
    event: str, h: float, scorer: str, *, n: int, pos: int, auroc: float
) -> dict[str, Any]:
    return {
        "event": event,
        "horizon_hours": h,
        "scorer": scorer,
        "n_at_risk": n,
        "n_positive": pos,
        "auroc": auroc,
    }


def _run(n: int = 100, pos: int = 10, gbm: float = 0.80) -> list[dict[str, Any]]:
    return [
        _rec("death", 8.0, "hazard", n=n, pos=pos, auroc=0.90),
        _rec("death", 8.0, "baseline_gbm", n=n, pos=pos, auroc=gbm),
    ]


def test_splice_takes_the_baseline_from_the_second_run() -> None:
    merged = splice(_run(gbm=0.80), _run(gbm=0.88))
    by = {r["scorer"]: r["auroc"] for r in merged}
    assert by["hazard"] == 0.90  # kept from the hazard side
    assert by["baseline_gbm"] == 0.88  # taken from the baseline side


def test_splice_refuses_when_the_at_risk_counts_differ() -> None:
    # The failure this exists to prevent: two runs whose cells look
    # comparable in a table but were scored on different rows.
    with pytest.raises(ValueError, match="did not score the same rows"):
        splice(_run(n=100), _run(n=112))


def test_splice_refuses_when_the_positive_counts_differ() -> None:
    with pytest.raises(ValueError, match="n_positive"):
        splice(_run(pos=10), _run(pos=11))


def test_check_names_the_offending_cell() -> None:
    with pytest.raises(ValueError, match=r"death@8h n_at_risk"):
        check_rows_align(_run(n=100), _run(n=112))


def test_splice_refuses_a_baseline_file_with_no_baseline_scorer() -> None:
    hazard_only = [_rec("death", 8.0, "hazard", n=100, pos=10, auroc=0.9)]
    with pytest.raises(ValueError, match="no baseline scorer"):
        splice(hazard_only, hazard_only)


def test_cells_absent_from_one_side_are_not_compared() -> None:
    # A baseline run may cover more cells than the hazard run; only the
    # shared ones constrain the splice.
    extra = _run() + [_rec("sepsis3", 8.0, "baseline_gbm", n=55, pos=5, auroc=0.7)]
    merged = splice(_run(), extra)
    assert any(r["event"] == "sepsis3" for r in merged)
