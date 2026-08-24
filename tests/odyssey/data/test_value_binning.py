"""Tests for folding numeric event values into clinically meaningful tokens."""

from typing import List, Optional, Tuple

import polars as pl
import pytest

from odyssey.data import code_mapping
from odyssey.data.concepts import (
    CONCEPTS,
    AnyOf,
    BaselineRelativeRule,
    CodeOccurrenceRule,
    ConceptDefinition,
    ConceptRule,
    DerivedGcsTotalRule,
    SustainedRule,
)
from odyssey.data.value_binning import (
    _FALLBACK_LABEL,
    CLINICAL_RANGES,
    SYMLOG_CEILING,
    SYMLOG_TAIL,
    VALUE_Z_CLIP,
    QuantileBinner,
    add_value_tokens,
    clinical_ranges_for_source,
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


def test_clinical_range_code_with_null_value_passes_through_unchanged() -> None:
    # A matching code whose reading is missing must NOT be binned: a null
    # value compares as False against every threshold, which without an
    # explicit guard falls through to the fallback label -- silently
    # tokenizing a missing heart-rate reading as "::HIGH".
    events = _events([("LAB//220045//bpm", None), ("LAB//220045//bpm", 75.0)])
    out = add_value_tokens(events)
    assert out["code"].to_list() == ["LAB//220045//bpm", "LAB//220045//bpm::NORMAL"]


def test_quantile_binned_code_with_null_value_passes_through_unchanged() -> None:
    # Same contract on the quantile path: a null value is below every cut
    # point, so without a guard it would silently land in "::Q1".
    train = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(train, n_bins=5, min_count=50)
    out = add_value_tokens(_events([("LAB//UNMAPPED//x", None)]), binner)
    assert out["code"].to_list() == ["LAB//UNMAPPED//x"]


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
                    sub_rule,
                    (BaselineRelativeRule, DerivedGcsTotalRule, CodeOccurrenceRule),
                ):
                    raise TypeError(f"unhandled rule type in AnyOf: {type(sub_rule)!r}")
        elif isinstance(rule, (ConceptRule, SustainedRule)):
            out.append((rule.code_prefix, rule.threshold))
        elif not isinstance(
            rule, (BaselineRelativeRule, DerivedGcsTotalRule, CodeOccurrenceRule)
        ):
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


def test_quantile_binner_apply_with_no_fitted_boundaries_returns_all_null() -> None:
    """An unfit (or fit-on-nothing-eligible) binner must not crash on real events."""
    binner = QuantileBinner(boundaries={}, n_bins=5)
    events = _events([("LAB//UNMAPPED//x", 1.0), ("LAB//UNMAPPED//x", 2.0)])
    out = binner.apply(events)
    assert out.to_list() == [None, None]
    assert out.dtype == pl.Utf8


def test_quantile_binner_standardize_with_no_value_stats_returns_all_null() -> None:
    """A binner saved before value_stats existed (or fit on nothing eligible)."""
    binner = QuantileBinner(boundaries={}, n_bins=5, value_stats={})
    events = _events([("LAB//UNMAPPED//x", 1.0), ("LAB//UNMAPPED//x", 2.0)])
    out = binner.standardize(events)
    assert out.to_list() == [None, None]
    assert out.dtype == pl.Float32


def test_quantile_binner_standardize_missing_value_column_returns_all_null() -> None:
    """Standardize on a frame with no numeric_value column at all, not a KeyError."""
    events = _events([("LAB//UNMAPPED//x", float(v)) for v in range(100)])
    binner = QuantileBinner.fit(events, n_bins=5, min_count=50)
    assert binner.value_stats  # sanity: this binner did fit real stats

    codes_only = pl.DataFrame({"code": ["LAB//UNMAPPED//x", "LAB//UNMAPPED//x"]})
    out = binner.standardize(codes_only)
    assert out.to_list() == [None, None]
    assert out.dtype == pl.Float32


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


# ---------------------------------------------------------------------------
# clinical_ranges_for_source: per-source expansion of the canonical ranges
# ---------------------------------------------------------------------------


def test_mimic_ranges_are_the_module_defaults() -> None:
    ranges, fallbacks = clinical_ranges_for_source("mimic_iv")
    assert ranges == CLINICAL_RANGES
    assert set(fallbacks) == set(ranges)


def test_eicu_ranges_cover_the_mapped_signals_with_the_same_cuts() -> None:
    """The same canonical range reaches each source through its own prefixes."""
    ranges, fallbacks = clinical_ranges_for_source("eicu")
    assert ranges["VITALS//PERIODIC//HEARTRATE"] == CLINICAL_RANGES["LAB//220045//"]
    # eICU temperature is Celsius: it gets the C cuts, and no F-range
    # prefix exists anywhere in the expansion.
    assert ranges["VITALS//PERIODIC//TEMPERATURE"] == CLINICAL_RANGES["LAB//223762//"]
    assert fallbacks["LAB//creatinine//"] == "CRITICAL"


def test_eicu_events_get_clinical_bins_via_source_parameter() -> None:
    events = pl.DataFrame(
        {
            "code": ["VITALS//PERIODIC//HEARTRATE", "VITALS//PERIODIC//TEMPERATURE"],
            "numeric_value": [120.0, 38.5],
        }
    )
    out = add_value_tokens(events, source="eicu")
    assert out["code"].to_list() == [
        "VITALS//PERIODIC//HEARTRATE::HIGH",
        "VITALS//PERIODIC//TEMPERATURE::HIGH",
    ]
    # Without the source, eICU prefixes are unknown and pass through.
    untouched = add_value_tokens(events)
    assert untouched["code"].to_list() == events["code"].to_list()


def test_clinical_ranges_skips_a_prefix_whose_unit_tag_is_not_curated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit-split LOINC (temperature: F/C) with a prefix tagged some other unit.

    Not reachable through any currently-registered source (every real
    prefix mapped to 8310-5 is tagged F or C), but a future or
    partially-configured source could add one -- prefixes_for_loinc and
    unit_for are wrapped (not replaced) to inject exactly one such prefix
    while every other LOINC's real expansion is untouched, so this tests
    the skip in isolation rather than emptying the whole result. The
    injected prefix must be silently excluded, not crash or fall back to
    the wrong cut points; the real F/C prefixes for the same LOINC must
    still come through normally.
    """
    real_prefixes_for_loinc = code_mapping.prefixes_for_loinc
    real_unit_for = code_mapping.unit_for

    def fake_prefixes_for_loinc(loinc: str, *, source: str) -> frozenset[str]:
        prefixes = real_prefixes_for_loinc(loinc, source=source)
        if loinc == "8310-5":
            prefixes = prefixes | {"LAB//TEMP//KELVIN"}
        return prefixes

    def fake_unit_for(prefix: str, *, source: str) -> Optional[str]:
        if prefix == "LAB//TEMP//KELVIN":
            return "K"
        return real_unit_for(prefix, source=source)

    monkeypatch.setattr(
        "odyssey.data.value_binning.prefixes_for_loinc", fake_prefixes_for_loinc
    )
    monkeypatch.setattr("odyssey.data.value_binning.unit_for", fake_unit_for)

    ranges, fallbacks = clinical_ranges_for_source("mimic_iv")

    assert "LAB//TEMP//KELVIN" not in ranges
    assert "LAB//TEMP//KELVIN" not in fallbacks
    # the real F/C prefixes for the same LOINC are unaffected by the injection
    assert ranges["LAB//223761//"] == CLINICAL_RANGES["LAB//223761//"]
    assert ranges["LAB//223762//"] == CLINICAL_RANGES["LAB//223762//"]


def test_gemini_creatinine_uses_si_clinical_range() -> None:
    """The umol/L range spec, not the mg/dL one, binds GEMINI's creatinine.

    mg/dL cuts applied to umol/L values would label nearly every reading
    CRITICAL (a normal 80 umol/L is far above the 4.0 mg/dL cut).
    """
    ranges, fallback = clinical_ranges_for_source("gemini")
    assert ranges["LAB//3020564//"] == [(132.6, "NORMAL"), (353.7, "HIGH")]
    assert fallback["LAB//3020564//"] == "CRITICAL"


def _tail_events(values: List[float]) -> pl.DataFrame:
    return pl.DataFrame({"code": ["LAB//X"] * len(values), "numeric_value": values})


def _tail_binner(transform: str) -> QuantileBinner:
    """Build a binner whose per-code stats are exactly center 0, scale 1."""
    return QuantileBinner(
        boundaries={"LAB//X": [0.0]},
        n_bins=2,
        value_stats={"LAB//X": (0.0, 1.0)},
        tail_transform=transform,
    )


def test_clip_tail_saturates_the_extremes_it_is_meant_to_distinguish() -> None:
    """The default flattens everything past the threshold to one number.

    Pinned as the behaviour symlog is the alternative to: because scale is
    robust (IQR / 1.349), +-5 lands inside the clinically abnormal range
    for skewed labs, so this is the tail the acute outcomes live in.
    """
    raw = [0.0, 2.0, VALUE_Z_CLIP + 1.0, 20.0, -20.0]
    z = _tail_binner("clip").standardize(_tail_events(raw)).to_list()
    assert z[:2] == [0.0, 2.0]  # inside the band, untouched
    assert z[2] == z[3] == VALUE_Z_CLIP  # 6 and 20 are indistinguishable
    assert z[4] == -VALUE_Z_CLIP


def test_symlog_tail_is_identity_inside_the_band_and_monotone_outside() -> None:
    """Symlog keeps ordering in the tail without moving the normal range."""
    inside = [0.0, 1.0, -2.5, VALUE_Z_CLIP]
    zi = _tail_binner(SYMLOG_TAIL).standardize(_tail_events(inside)).to_list()
    assert zi == pytest.approx(inside)  # nothing inside the band moves at all

    tail = [VALUE_Z_CLIP + 0.5, 6.0, 8.0, 20.0, 100.0]
    zt = _tail_binner(SYMLOG_TAIL).standardize(_tail_events(tail)).to_list()
    assert all(b > a for a, b in zip(zt, zt[1:]))  # strictly monotone: 4 != 8
    assert zt[0] > VALUE_Z_CLIP  # continuous, above the boundary
    assert zt[-1] < SYMLOG_CEILING  # bounded
    # and it is symmetric
    neg = _tail_binner(SYMLOG_TAIL).standardize(_tail_events([-20.0])).to_list()
    assert neg[0] == pytest.approx(-zt[3])


def test_tail_transform_round_trips_and_rejects_unknown_names(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Evaluation must rebuild the policy the run trained with, not the default."""
    path = tmp_path / "binner.json"
    _tail_binner(SYMLOG_TAIL).save(path)
    assert QuantileBinner.load(path).tail_transform == SYMLOG_TAIL
    # a binner written before the field existed reads back as the old behaviour
    path.write_text('{"n_bins": 2, "boundaries": {}}')
    assert QuantileBinner.load(path).tail_transform == "clip"
    with pytest.raises(ValueError, match="unknown tail_transform"):
        _tail_binner("logarithmic").standardize(_tail_events([9.0]))
    with pytest.raises(ValueError, match="unknown tail_transform"):
        QuantileBinner.fit(_tail_events([1.0, 2.0]), min_count=1, tail_transform="nope")


def test_symlog_separates_the_aki_threshold_it_was_built_for() -> None:
    """The point of the whole exercise, in the units it was measured in.

    Creatinine's fitted stats (median 0.900, scale 0.445, one real MIMIC
    train shard) put the +-5 clip at 3.12 mg/dL -- BELOW the 4.0 mg/dL
    the aki_stage_3 concept triggers on -- so under clip a borderline 3.2
    and a severe 15.6 reach the model as the same number.
    """
    binner_stats = {"LAB//RESULT//50912//mg/dL": (0.900, 0.445)}
    raw = [3.2, 4.0, 6.0, 9.0, 15.6]
    events = pl.DataFrame(
        {"code": ["LAB//RESULT//50912//mg/dL"] * len(raw), "numeric_value": raw}
    )
    clipped = QuantileBinner(
        boundaries={}, n_bins=2, value_stats=binner_stats
    ).standardize(events)
    assert clipped.to_list() == [VALUE_Z_CLIP] * len(raw)  # all indistinguishable

    logged = QuantileBinner(
        boundaries={}, n_bins=2, value_stats=binner_stats, tail_transform=SYMLOG_TAIL
    ).standardize(events)
    values = logged.to_list()
    assert all(b > a for a, b in zip(values, values[1:]))  # every level distinct
    assert values[-1] < SYMLOG_CEILING
    # the gap that matters: a borderline 3.2 and a stage-3 4.0 are now
    # nearly a full unit apart, where clip made them the same number
    assert values[1] - values[0] > 0.5


def test_symlog_ceiling_bounds_a_data_entry_error() -> None:
    """A garbage value must not hand the embedding an unseen input scale.

    value_features feeds ``[z, z^2, has]``, so an unbounded tail is
    squared on its way into the projection.
    """
    absurd = _tail_binner(SYMLOG_TAIL).standardize(_tail_events([1e6, -1e6]))
    assert absurd.to_list() == [SYMLOG_CEILING, -SYMLOG_CEILING]


def test_min_scale_floors_a_degenerate_iqr_and_is_off_by_default() -> None:
    """A near-zero training IQR turns float noise into an astronomical z.

    Measured on real held-out data 2026-08-24: INFUSION_END//228315 had a
    fitted scale of 4.524e-05, and a sentinel reading of 999999.0 on
    another code produced z above 1.2e7. Under the clip policy this was
    invisible (everything past 5 saturated to 5); under symlog such values
    reach SYMLOG_CEILING and their z^2 input feature quadruples. Off by
    default so the value-tail arms differ in exactly one respect.
    """
    # a code whose values are all but identical: IQR is tiny but positive
    # quartiles land on all-but-identical values: IQR is positive but tiny,
    # which is the real shape (an exactly-zero IQR falls back to std instead)
    events = pl.DataFrame(
        {
            "code": ["LAB//DEGENERATE"] * 8,
            "numeric_value": [100.0] * 4 + [100.00001] * 4,
        }
    )
    loose = QuantileBinner.fit(events, n_bins=2, min_count=1)
    _, scale = loose.value_stats["LAB//DEGENERATE"]
    assert scale < 1e-4  # the defect: a scale far below the value's magnitude

    floored = QuantileBinner.fit(events, n_bins=2, min_count=1, min_scale=0.01)
    _, floored_scale = floored.value_stats["LAB//DEGENERATE"]
    assert floored_scale == pytest.approx(1.0)  # 1% of a centre of 100

    # a mild real outlier is what the floor rescues: 30% above centre is a
    # z of ~4e6 on the degenerate scale and a sane 30 once floored
    mild = pl.DataFrame({"code": ["LAB//DEGENERATE"], "numeric_value": [130.0]})
    raw_loose = (130.0 - 100.0) / scale
    raw_floored = (130.0 - 100.0) / floored_scale
    assert raw_loose > 1e6
    assert raw_floored == pytest.approx(30.0)

    # and the limitation, asserted rather than assumed: the floor does NOT
    # rescue a sentinel value. 999999 is astronomically far from the centre
    # even on a sane scale, so it still pins the ceiling under symlog. That
    # is a data-quality problem and belongs upstream, not here.
    sentinel = pl.DataFrame({"code": ["LAB//DEGENERATE"], "numeric_value": [999999.0]})
    for stats in (loose.value_stats, floored.value_stats):
        symlogged = QuantileBinner(
            boundaries={}, n_bins=2, value_stats=stats, tail_transform=SYMLOG_TAIL
        )
        assert symlogged.standardize(sentinel).to_list()[0] == SYMLOG_CEILING
        assert symlogged.standardize(mild).to_list()[0] <= SYMLOG_CEILING
