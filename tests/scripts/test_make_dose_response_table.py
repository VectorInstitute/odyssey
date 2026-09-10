"""The dose-response summary separates a graded response from a threshold."""

from __future__ import annotations

from typing import Any

from scripts.make_dose_response_table import (
    dose_response,
    null_reference,
    render,
    sweep_arms,
)


def _results(
    series_by_subject: dict[int, list[float]], rungs: list[int]
) -> dict[str, Any]:
    """Build a sweep JSON where subject s has ``series[i]`` risk at ``rungs[i]``."""
    edits: dict[str, Any] = {}
    for i, rung in enumerate(rungs):
        edits[f"sbp_noninvasive:set:{rung}:6"] = {
            "per_subject": [
                {
                    "subject_id": sid,
                    "rows_edited": 3,
                    "counterfactual": {"event_risk": {"death": {"24h": series[i]}}},
                }
                for sid, series in series_by_subject.items()
            ]
        }
    return {"edits": edits}


RUNGS = [110, 100, 90, 80, 70]


def test_sweep_arms_are_ordered_most_severe_last() -> None:
    arms = sweep_arms(_results({1: [0.0] * 5}, RUNGS), "sbp_noninvasive")
    assert [v for v, _ in arms] == [110.0, 100.0, 90.0, 80.0, 70.0]


def test_even_ramp_scores_one_over_k_on_top_step() -> None:
    # A perfectly linear response across 5 rungs has 4 equal steps, so the
    # largest step is a quarter of the range.
    stats = dose_response(
        _results({1: [0.1, 0.2, 0.3, 0.4, 0.5]}, RUNGS),
        "sbp_noninvasive",
        "death",
        "24h",
    )
    assert stats is not None
    assert stats["monotone_pct"] == 100.0
    assert abs(stats["median_top_step"] - 0.25) < 1e-9
    assert abs(stats["even_ramp"] - 0.25) < 1e-9


def test_pure_threshold_scores_one_on_top_step_despite_being_monotone() -> None:
    # The case the whole statistic exists for: flat, one jump, flat. It is
    # perfectly rank-monotonic, so a rank correlation would call it a clean
    # dose-response; top step must expose it as a trip point instead.
    stats = dose_response(
        _results({1: [0.1, 0.1, 0.1, 0.9, 0.9]}, RUNGS),
        "sbp_noninvasive",
        "death",
        "24h",
    )
    assert stats is not None
    assert stats["monotone_pct"] == 100.0
    assert abs(stats["median_top_step"] - 1.0) < 1e-9


def test_a_subject_that_reverses_is_not_counted_monotone() -> None:
    stats = dose_response(
        _results({1: [0.1, 0.2, 0.3, 0.4, 0.5], 2: [0.5, 0.4, 0.3, 0.2, 0.1]}, RUNGS),
        "sbp_noninvasive",
        "death",
        "24h",
    )
    assert stats is not None
    assert stats["n"] == 2
    assert stats["monotone_pct"] == 50.0


def test_only_subjects_present_in_every_rung_are_used() -> None:
    # Comparing a different subset per rung is the coverage-mismatch bug this
    # project has hit repeatedly; the cohort must be the intersection.
    results = _results({1: [0.1, 0.2, 0.3, 0.4, 0.5], 2: [0.1] * 5}, RUNGS)
    arm = results["edits"]["sbp_noninvasive:set:90:6"]
    arm["per_subject"] = [r for r in arm["per_subject"] if r["subject_id"] == 1]
    stats = dose_response(results, "sbp_noninvasive", "death", "24h")
    assert stats is not None
    assert stats["n"] == 1


def test_unedited_subjects_are_excluded() -> None:
    results = _results({1: [0.1, 0.2, 0.3, 0.4, 0.5]}, RUNGS)
    for arm in results["edits"].values():
        arm["per_subject"][0]["rows_edited"] = 0
    assert dose_response(results, "sbp_noninvasive", "death", "24h") is None


def test_render_emits_a_tabular_with_both_statistics() -> None:
    text = render(_results({1: [0.1, 0.2, 0.3, 0.4, 0.5]}, RUNGS))
    assert "\\begin{tabular}" in text and "\\end{tabular}" in text
    assert "monotone" in text and "top step" in text
    assert "100\\%" in text and "0.25" in text


def test_flat_subjects_are_reported_not_folded_into_monotone() -> None:
    # A subject the edit never moved is trivially non-decreasing. Counting it
    # as monotone without saying so would let "the model ignored the edit"
    # read as "the model responded monotonically".
    stats = dose_response(
        _results({1: [0.2] * 5, 2: [0.1, 0.2, 0.3, 0.4, 0.5]}, RUNGS),
        "sbp_noninvasive",
        "death",
        "24h",
    )
    assert stats is not None
    assert stats["monotone_pct"] == 100.0
    assert stats["flat_pct"] == 50.0
    # The flat subject has no shape, so it cannot drag the top-step median.
    assert abs(stats["median_top_step"] - 0.25) < 1e-9


def test_null_reference_is_far_from_an_even_ramp_in_both_statistics() -> None:
    # The point of the null row: "no response" does NOT look like an even
    # ramp. If a reader assumed noise sat at 1/k they would read a middling
    # top step as half-way to a dose-response, when it is nearer noise.
    mono, top = null_reference(44, 5, trials=200, seed=1)
    assert mono < 10.0
    assert top > 0.7


def test_null_reference_is_deterministic_for_a_seed() -> None:
    assert null_reference(44, 5, trials=50, seed=3) == null_reference(
        44, 5, trials=50, seed=3
    )
