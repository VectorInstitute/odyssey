"""Tests for folding numeric event values into clinically meaningful tokens."""

from typing import List, Optional, Tuple

import polars as pl
import pytest

from odyssey.data.concepts import (
    CONCEPTS,
    AnyOf,
    BaselineRelativeRule,
    ConceptDefinition,
    ConceptRule,
    DerivedGcsTotalRule,
    SustainedRule,
)
from odyssey.data.value_binning import (
    _FALLBACK_LABEL,
    CLINICAL_RANGES,
    QuantileBinner,
    add_value_tokens,
)


def _events(rows: List[Tuple[str, Optional[float]]]) -> pl.DataFrame:
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


def test_creatinine_bins_into_normal_high_critical() -> None:
    # Three tiers matching aki_stage_1's 1.5 threshold and aki_stage_3's
    # 4.0 absolute-value threshold (odyssey/data/concepts.py).
    events = _events(
        [
            ("LAB//RESULT//50912//mg/dL", 1.0),  # normal
            ("LAB//RESULT//50912//mg/dL", 2.5),  # aki_stage_1/2 territory
            ("LAB//RESULT//50912//mg/dL", 4.5),  # aki_stage_3 territory
        ]
    )
    out = add_value_tokens(events)
    assert out["code"].to_list() == [
        "LAB//RESULT//50912//mg/dL::NORMAL",
        "LAB//RESULT//50912//mg/dL::HIGH",
        "LAB//RESULT//50912//mg/dL::CRITICAL",
    ]


def test_only_upper_threshold_defined_gives_two_bins() -> None:
    # respiratory rate only has a tachypnea (>20) rule -- no bradypnea rule.
    events = _events([("LAB//220210//insp/min", 12.0), ("LAB//220210//insp/min", 25.0)])
    out = add_value_tokens(events)
    assert out["code"].to_list() == [
        "LAB//220210//insp/min::NORMAL",
        "LAB//220210//insp/min::HIGH",
    ]


def test_numeric_valued_code_with_no_curated_range_and_no_binner_passes_through() -> (
    None
):
    events = _events([("LAB//99999//unmapped", 42.0)])
    out = add_value_tokens(events)
    assert out["code"][0] == "LAB//99999//unmapped"


def test_empty_events_frame_is_a_noop() -> None:
    events = _events([])
    out = add_value_tokens(events)
    assert out["code"].to_list() == []


# ---------------------------------------------------------------------------
# Consistency with odyssey/data/concepts.py's thresholds
# ---------------------------------------------------------------------------


def _threshold_rules(concept: ConceptDefinition) -> List[Tuple[str, float]]:
    """Yield every (code_prefix, threshold) pair reachable from a concept.

    Only :class:`~odyssey.data.concepts.ConceptRule` and
    :class:`~odyssey.data.concepts.SustainedRule` represent a single
    instantaneous value crossing a fixed threshold -- the kind of thing
    a CLINICAL_RANGES bin transition can meaningfully correspond to.
    :class:`~odyssey.data.concepts.BaselineRelativeRule` is a delta from
    a personal baseline, not a fixed value, and
    :class:`~odyssey.data.concepts.DerivedGcsTotalRule` sums three
    different codes, not one -- neither maps to a single CLINICAL_RANGES
    prefix's bin edge, so both are skipped here. Recurses into
    :class:`~odyssey.data.concepts.AnyOf`, which nests further rules.

    :class:`~odyssey.data.concepts.CompositeConceptDefinition` (SIRS,
    qSOFA) is out of scope entirely, not just certain rule types within
    it: its component thresholds (e.g. SIRS's HR > 90) exist purely to
    feed a composite N-of-M score and are not, on their own, meant to be
    a standalone interpretable "this vital is abnormal" concept the way
    a plain :class:`~odyssey.data.concepts.ConceptDefinition` is -- they
    have no reason to share a bin edge with, say, ``tachycardia``'s
    HR > 100.
    """
    out: List[Tuple[str, float]] = []
    for rule in concept.rules:
        if isinstance(rule, AnyOf):
            for sub_rule in rule.rules:
                if isinstance(sub_rule, (ConceptRule, SustainedRule)):
                    out.append((sub_rule.code_prefix, sub_rule.threshold))
                elif not isinstance(
                    sub_rule, (BaselineRelativeRule, DerivedGcsTotalRule)
                ):
                    raise TypeError(f"unhandled rule type in AnyOf: {type(sub_rule)!r}")
        elif isinstance(rule, (ConceptRule, SustainedRule)):
            out.append((rule.code_prefix, rule.threshold))
        elif not isinstance(rule, (BaselineRelativeRule, DerivedGcsTotalRule)):
            raise TypeError(f"unhandled rule type: {type(rule)!r}")
    return out


def test_clinical_ranges_reproduce_every_concept_rule_threshold() -> None:
    """Every threshold-based rule in concepts.py must have a bin transition here.

    So a lab's bin label always means the same thing as the concept label
    it also supervises. Composite definitions and rule types with no
    single fixed threshold value are out of scope -- see
    :func:`_threshold_rules`.
    """
    plain_concepts = [c for c in CONCEPTS if isinstance(c, ConceptDefinition)]
    for concept in plain_concepts:
        for code_prefix, threshold in _threshold_rules(concept):
            events = _events(
                [
                    (code_prefix + "x", threshold - 0.01),
                    (code_prefix + "x", threshold + 0.01),
                ]
            )
            out = add_value_tokens(events)
            below_bin, above_bin = out["code"].to_list()
            assert below_bin != above_bin, (
                f"{concept.name}: {code_prefix} threshold {threshold} "
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


def test_quantile_binner_apply_on_empty_events_frame() -> None:
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    assert binner.apply(_events([])).to_list() == []


def test_quantile_binner_handles_constant_values_via_deduped_boundaries() -> None:
    # Every observation is identical, so every quantile collapses to one
    # cut point -- fewer than n_bins - 1. apply() must not index out of range.
    events = _events([("LAB//UNMAPPED//x", 5.0)] * 100)
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    assert binner.boundaries["LAB//UNMAPPED//x"] == [5.0]

    out = binner.apply(events)
    assert set(out.to_list()) == {"Q2"}


def test_quantile_binner_n_bins_one_gives_a_single_bin() -> None:
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=1, min_count=50)
    assert binner.boundaries["LAB//UNMAPPED//x"] == []

    out = binner.apply(events)
    assert set(out.to_list()) == {"Q1"}


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
