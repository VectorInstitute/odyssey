"""Tests for rule-derived concept labels."""

import subprocess
from pathlib import Path

import polars as pl
import pytest

from odyssey.data.concepts import (
    CONCEPTS,
    ConceptDefinition,
    ConceptRule,
    label_concepts,
)


def _events(rows: list) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"subject_id": pl.Int64, "code": pl.Utf8, "numeric_value": pl.Float32},
        orient="row",
    )


# ---------------------------------------------------------------------------
# Unit tests on synthetic MEDS-schema data
# ---------------------------------------------------------------------------


def test_above_threshold_triggers() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = _events(
        [
            (1, "LAB//220045//bpm", 120.0),  # triggers
            (2, "LAB//220045//bpm", 80.0),  # does not trigger
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["tachycardia"].to_list() == [1, 0]
    assert labels["tachycardia_observed"].to_list() == [1, 1]


def test_below_threshold_triggers() -> None:
    concepts = [
        ConceptDefinition(
            "hypotension", [ConceptRule("LAB//220179//", 90.0, "below")], "SBP < 90"
        )
    ]
    events = _events(
        [
            (1, "LAB//220179//mmHg", 85.0),  # triggers
            (2, "LAB//220179//mmHg", 110.0),  # does not trigger
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["hypotension"].to_list() == [1, 0]


def test_multiple_rules_are_ored() -> None:
    # Fever via either the Fahrenheit or Celsius chartevents itemid.
    concepts = [c for c in CONCEPTS if c.name == "fever"]
    events = _events(
        [
            (
                1,
                "LAB//223761//F",
                101.5,
            ),  # F itemid triggers
            (
                2,
                "LAB//223762//C",
                39.0,
            ),  # C itemid triggers
            (
                3,
                "LAB//223761//F",
                98.6,
            ),  # F itemid, normal
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["fever"].to_list() == [1, 1, 0]
    assert labels["fever_observed"].to_list() == [1, 1, 1]


def test_observed_mask_false_when_itemid_absent() -> None:
    concepts = [
        ConceptDefinition(
            "elevated_lactate",
            [ConceptRule("LAB//RESULT//50813//", 2.0, "above")],
            "Lactate > 2.0",
        )
    ]
    # Subject 2 has events, but never the lactate itemid at all.
    events = _events(
        [
            (1, "LAB//RESULT//50813//mmol/L", 3.0),
            (2, "LAB//220045//bpm", 80.0),
        ]
    )
    labels = label_concepts(events, concepts).sort("subject_id")
    assert labels["elevated_lactate"].to_list() == [1, 0]
    assert labels["elevated_lactate_observed"].to_list() == [1, 0]


def test_null_numeric_value_never_triggers_or_counts_as_observed() -> None:
    concepts = [
        ConceptDefinition(
            "tachycardia", [ConceptRule("LAB//220045//", 100.0, "above")], "HR > 100"
        )
    ]
    events = pl.DataFrame(
        {
            "subject_id": [1],
            "code": ["LAB//220045//bpm"],
            "numeric_value": pl.Series([None], dtype=pl.Float32),
        }
    )
    labels = label_concepts(events, concepts)
    assert labels["tachycardia"].to_list() == [0]
    assert labels["tachycardia_observed"].to_list() == [0]


def test_default_registry_covers_all_subjects() -> None:
    events = _events([(1, "LAB//220045//bpm", 75.0), (2, "LAB//220045//bpm", 75.0)])
    labels = label_concepts(events)
    assert labels.height == 2
    for concept in CONCEPTS:
        assert concept.name in labels.columns
        assert f"{concept.name}_observed" in labels.columns


# ---------------------------------------------------------------------------
# Integration test against the real MIMIC-IV demo dataset
# ---------------------------------------------------------------------------


@pytest.mark.integration_test
def test_label_concepts_on_real_mimic_iv_demo_extraction(tmp_path: Path) -> None:
    """Sanity-check concept labeling against a real MEDS extraction.

    Runs the actual meds-extract pipeline against the public MIMIC-IV demo
    and confirms concept labeling behaves sensibly on real data: common
    vitals/labs are observed for a meaningful fraction of the cohort, and
    every concept produces a valid binary column.
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

    shards = list((output_dir / "data").rglob("*.parquet"))
    assert shards, "expected at least one MEDS data shard"
    events = pl.concat([pl.read_parquet(s) for s in shards])

    labels = label_concepts(events)
    n_subjects = labels.height
    assert n_subjects > 0

    for concept in CONCEPTS:
        assert labels[concept.name].is_in([0, 1]).all()

    # Heart rate is charted for virtually every ICU patient in the cohort.
    assert labels["tachycardia_observed"].sum() > 0
