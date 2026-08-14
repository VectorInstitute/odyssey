"""Tests for the prevalence/balance concept selection filter."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import pytest

from odyssey.data.concept_selection import (
    PrevalenceStats,
    compute_prevalence_stats,
    filter_by_prevalence,
)
from odyssey.data.concepts import (
    CONCEPTS,
    AnyConceptDefinition,
    ConceptDefinition,
    label_concepts,
)


def _labels(rows: Dict[str, List[Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _concepts(*names: str) -> List[AnyConceptDefinition]:
    return [ConceptDefinition(name, [], f"{name} description") for name in names]


def test_passes_when_well_balanced_and_well_supported() -> None:
    labels = _labels(
        {
            "subject_id": list(range(20)),
            "c": [1] * 10 + [0] * 10,
            "c_observed": [1] * 20,
        }
    )
    stats = compute_prevalence_stats(labels, _concepts("c"), min_support=10)
    assert stats[0].passes
    assert stats[0].n_observed == 20
    assert stats[0].n_triggered == 10
    assert stats[0].prevalence == 0.5


def test_fails_min_support_when_too_few_triggered() -> None:
    labels = _labels(
        {
            "subject_id": list(range(20)),
            "c": [1] * 3 + [0] * 17,
            "c_observed": [1] * 20,
        }
    )
    stats = compute_prevalence_stats(labels, _concepts("c"), min_support=10)
    assert not stats[0].passes_min_support
    assert not stats[0].passes


def test_fails_dominant_class_when_almost_always_triggered() -> None:
    # Mirrors the real tachypnea failure: 96.5% triggered.
    labels = _labels(
        {
            "subject_id": list(range(100)),
            "c": [1] * 96 + [0] * 4,
            "c_observed": [1] * 100,
        }
    )
    stats = compute_prevalence_stats(
        labels, _concepts("c"), min_support=10, max_dominant_class=0.95
    )
    assert stats[
        0
    ].passes_min_support  # 4 non-triggered subjects isn't the support issue
    assert not stats[0].passes_max_dominant_class
    assert not stats[0].passes


def test_fails_dominant_class_when_almost_never_triggered() -> None:
    labels = _labels(
        {
            "subject_id": list(range(100)),
            "c": [1] * 4 + [0] * 96,
            "c_observed": [1] * 100,
        }
    )
    stats = compute_prevalence_stats(labels, _concepts("c"), max_dominant_class=0.95)
    assert not stats[0].passes_max_dominant_class


def test_unobserved_subjects_are_excluded_from_the_denominator() -> None:
    """A concept observed in few subjects, but balanced among those.

    Should not be penalized by a large unobserved population.
    """
    labels = _labels(
        {
            "subject_id": list(range(1000)),
            "c": [1] * 10 + [0] * 10 + [0] * 980,
            "c_observed": [1] * 20 + [0] * 980,
        }
    )
    stats = compute_prevalence_stats(labels, _concepts("c"), min_support=10)
    assert stats[0].n_observed == 20
    assert stats[0].prevalence == 0.5
    assert stats[0].passes


def test_never_observed_concept_fails_both_filters() -> None:
    labels = _labels(
        {
            "subject_id": [1, 2, 3],
            "c": [0, 0, 0],
            "c_observed": [0, 0, 0],
        }
    )
    stats = compute_prevalence_stats(labels, _concepts("c"))
    assert stats[0].n_observed == 0
    assert stats[0].prevalence == 0.0
    assert not stats[0].passes


def test_filter_by_prevalence_keeps_only_passing_concepts() -> None:
    stats = [
        PrevalenceStats("good", 100, 50, 0.5, True, True),
        PrevalenceStats("too_rare", 100, 2, 0.02, False, True),
        PrevalenceStats("too_dominant", 100, 99, 0.99, True, False),
    ]
    assert filter_by_prevalence(stats) == ["good"]


@pytest.mark.integration_test
def test_prevalence_stats_on_real_mimic_iv_demo_extraction(tmp_path: Path) -> None:
    """Real prevalence numbers for the current CONCEPTS registry.

    Not asserting specific numbers (the demo cohort is tiny and any
    single concept's exact prevalence there isn't a stable property to
    pin a test to) -- just that the filter runs end-to-end against real
    labels and returns a sensible verdict shape.
    """
    output_dir = tmp_path / "meds_demo"
    result = subprocess.run(
        [
            "meds-extract-run",
            "spec=MIMIC-IV",
            f"output_dir={output_dir}",
            "dataset_key=demo",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    shards = list((Path(output_dir) / "data").rglob("*.parquet"))
    events = pl.concat([pl.read_parquet(s) for s in shards])
    labels = label_concepts(events)

    stats = compute_prevalence_stats(labels, CONCEPTS)
    assert len(stats) == len(CONCEPTS)
    for s in stats:
        assert 0.0 <= s.prevalence <= 1.0
        assert s.n_triggered <= s.n_observed
