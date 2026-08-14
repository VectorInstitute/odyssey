"""Tests for folding numeric event values into clinically meaningful tokens."""

import polars as pl
import pytest

from odyssey.data.concepts import CONCEPTS
from odyssey.data.value_binning import (
    _FALLBACK_LABEL,
    CLINICAL_RANGES,
    QuantileBinner,
    add_value_tokens,
)


def _events(rows: list) -> pl.DataFrame:
    """rows: list of (code, numeric_value_or_None)."""
    return pl.DataFrame(
        rows, schema={"code": pl.Utf8, "numeric_value": pl.Float64}, orient="row"
    )


# ---------------------------------------------------------------------------
# add_value_tokens: clinical-range path
# ---------------------------------------------------------------------------


def test_codes_without_numeric_value_pass_through_unchanged() -> None:
    events = _events([("PROCEDURE//CT_HEAD", None), ("DIAGNOSIS//A047", None)])
    out = add_value_tokens(events)
    assert out["code"].to_list() == ["PROCEDURE//CT_HEAD", "DIAGNOSIS//A047"]


def test_no_numeric_value_column_is_a_noop() -> None:
    events = pl.DataFrame({"code": ["DIAGNOSIS//A047"]})
    out = add_value_tokens(events)
    assert out["code"].to_list() == ["DIAGNOSIS//A047"]


def test_heart_rate_bins_into_low_normal_high() -> None:
    events = _events(
        [
            ("LAB//220045//bpm", 45.0),  # bradycardia
            ("LAB//220045//bpm", 80.0),  # normal
            ("LAB//220045//bpm", 130.0),  # tachycardia
        ]
    )
    out = add_value_tokens(events)
    assert out["code"].to_list() == [
        "LAB//220045//bpm::LOW",
        "LAB//220045//bpm::NORMAL",
        "LAB//220045//bpm::HIGH",
    ]


def test_bin_boundaries_are_exclusive_on_the_low_side() -> None:
    # threshold itself (60.0) is the bradycardia cutoff: values strictly
    # below it are LOW, the threshold value itself is NORMAL.
    events = _events([("LAB//220045//bpm", 60.0)])
    out = add_value_tokens(events)
    assert out["code"][0] == "LAB//220045//bpm::NORMAL"


def test_only_upper_threshold_defined_gives_two_bins() -> None:
    # respiratory rate only has a tachypnea (>20) rule -- no bradypnea rule.
    events = _events([("LAB//220210//insp/min", 12.0), ("LAB//220210//insp/min", 25.0)])
    out = add_value_tokens(events)
    assert out["code"].to_list() == [
        "LAB//220210//insp/min::NORMAL",
        "LAB//220210//insp/min::HIGH",
    ]


def test_numeric_valued_code_with_no_curated_range_and_no_binner_passes_through() -> None:
    events = _events([("LAB//99999//unmapped", 42.0)])
    out = add_value_tokens(events)
    assert out["code"][0] == "LAB//99999//unmapped"


# ---------------------------------------------------------------------------
# Consistency with odyssey/data/concepts.py's thresholds
# ---------------------------------------------------------------------------


def test_clinical_ranges_reproduce_every_concept_rule_threshold() -> None:
    """Every rule in concepts.py must correspond to a bin transition here.

    So a lab's bin label always means the same thing as the concept label
    it also supervises.
    """
    for concept in CONCEPTS:
        for rule in concept.rules:
            events = _events(
                [
                    (rule.code_prefix + "x", rule.threshold - 0.01),
                    (rule.code_prefix + "x", rule.threshold + 0.01),
                ]
            )
            out = add_value_tokens(events)
            below_bin, above_bin = out["code"].to_list()
            assert below_bin != above_bin, (
                f"{concept.name}: {rule.code_prefix} threshold {rule.threshold} "
                "produces no bin transition"
            )


# ---------------------------------------------------------------------------
# QuantileBinner
# ---------------------------------------------------------------------------


def test_quantile_binner_skips_codes_below_min_count() -> None:
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(10)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=100)
    assert binner.boundaries == {}


def test_quantile_binner_bins_a_well_observed_code() -> None:
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    assert "LAB//UNMAPPED//x" in binner.boundaries
    assert len(binner.boundaries["LAB//UNMAPPED//x"]) == 4  # n_bins - 1 cut points

    low = _events([("LAB//UNMAPPED//x", 0.0)])
    high = _events([("LAB//UNMAPPED//x", 99.0)])
    low_label = binner.apply(low)[0]
    high_label = binner.apply(high)[0]
    assert low_label == "Q1"
    assert high_label == "Q5"
    assert low_label != high_label


def test_quantile_binner_falls_back_when_code_unseen_at_fit_time() -> None:
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    unseen = _events([("LAB//OTHER//y", 5.0)])
    assert binner.apply(unseen)[0] is None


def test_quantile_binner_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    path = tmp_path / "binner.json"
    binner.save(path)
    loaded = QuantileBinner.load(path)
    assert loaded.boundaries == binner.boundaries
    assert loaded.n_bins == binner.n_bins


def test_add_value_tokens_uses_quantile_binner_when_no_clinical_range_applies() -> None:
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    out = add_value_tokens(events, binner)
    assert out["code"][0].startswith("LAB//UNMAPPED//x::Q")


def test_add_value_tokens_prefers_clinical_range_over_quantile_binner() -> None:
    # LAB//220045// (heart rate) has a curated range; the quantile binner
    # should never override it even if it also has boundaries for this code.
    hr_events = _events([("LAB//220045//bpm", float(v)) for v in range(50, 150)])
    binner = QuantileBinner.fit(hr_events, n_bins=5, min_count=10)
    out = add_value_tokens(_events([("LAB//220045//bpm", 45.0)]), binner)
    assert out["code"][0] == "LAB//220045//bpm::LOW"  # clinical bin, not "::Q1"


@pytest.mark.parametrize("prefix", list(CLINICAL_RANGES.keys()))
def test_every_clinical_range_prefix_has_a_fallback_label(prefix: str) -> None:
    assert prefix in _FALLBACK_LABEL
