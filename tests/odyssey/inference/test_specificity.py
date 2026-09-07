"""Specificity scoring over steering summaries: focus, opposites, two-sided outcomes."""

from odyssey.inference.specificity import score_dial, score_run, totals


def _delta(point: float) -> dict:
    return {"point": point, "ci_low": point - 0.01, "ci_high": point + 0.01, "n": 10}


def _shift(
    concept: str, point: float, expected: int | None, separated: bool = True
) -> dict:
    return {
        "concept": concept,
        "baseline": 0.3,
        "steered": 0.3 + point,
        "delta": _delta(point),
        "expected_sign": expected,
        "as_expected": None if expected is None else point * expected > 0,
        "separated": separated,
    }


def _outcome(
    event: str, ratio: float, expected: int | None, horizon: float = 24.0
) -> dict:
    return {
        "event": event,
        "horizon_hours": horizon,
        "relative_change": ratio,
        "expected_sign": expected,
        "as_expected": None if expected is None else (ratio - 1) * expected > 0,
        "separated": True,
    }


def test_a_focused_two_sided_dial_scores_perfectly() -> None:
    summary = {
        "concept": "hypotension",
        "direction": "amplify",
        "concept_shifts": [
            _shift("hypotension", +0.10, +1),
            _shift("hypertension", -0.02, -1),
            _shift("fever", 0.001, None, separated=False),
        ],
        "outcomes": [
            _outcome("death", 1.2, +1),
            _outcome("icu_discharge", 0.8, -1),
            _outcome("death", 1.1, +1, horizon=8.0),  # other horizon, ignored
        ],
    }
    d = score_dial(summary)
    assert d.concept == "hypotension" and d.direction == "amplify"
    assert d.own_shift == 0.10
    assert abs(d.focus - 0.10 / 0.121) < 1e-9
    assert (d.opposite_as_expected, d.opposite_declared) == (1, 1)
    assert (d.others_separated, d.others_total) == (0, 1)
    assert (d.good_as_expected, d.good_declared) == (1, 1)
    assert (d.bad_as_expected, d.bad_declared) == (1, 1)


def test_a_sicker_only_push_fails_the_good_outcomes_and_spreads_its_readout() -> None:
    summary = {
        "concept": "shock",  # legacy name maps to sustained_hypotension_map
        "direction": "amplify",
        "concept_shifts": [
            _shift("shock", +0.02, +1),
            _shift("hypertension", +0.02, -1),  # opposite went UP
            _shift("sepsis3", +0.05, None),
            _shift("fever", +0.05, None),
        ],
        "outcomes": [
            _outcome("death", 1.3, +1),
            _outcome("icu_discharge", 1.2, -1),  # good outcome went UP too
            _outcome("hospital_discharge_alive", 1.1, -1),
        ],
    }
    d = score_dial(summary)
    assert d.concept == "sustained_hypotension_map"
    assert abs(d.focus - 0.02 / 0.14) < 1e-9
    assert (d.opposite_as_expected, d.opposite_declared) == (0, 1)
    assert (d.others_separated, d.others_total) == (2, 2)
    assert (d.good_as_expected, d.good_declared) == (0, 2)
    assert (d.bad_as_expected, d.bad_declared) == (1, 1)


def test_totals_count_both_sided_dials_and_take_the_median_focus() -> None:
    good = {
        "concept": "hypotension",
        "direction": "amplify",
        "concept_shifts": [_shift("hypotension", 0.1, +1)],
        "outcomes": [_outcome("death", 1.2, +1), _outcome("icu_discharge", 0.9, -1)],
    }
    bad = {
        "concept": "anemia",
        "direction": "suppress",
        "concept_shifts": [_shift("anemia", -0.01, -1), _shift("fever", 0.03, None)],
        "outcomes": [_outcome("death", 0.9, -1), _outcome("icu_discharge", 0.95, +1)],
    }
    t = totals(score_run([good, bad]))
    assert t.n_dials == 2
    assert t.both_sided == 1
    assert (t.good_as_expected, t.good_declared) == (1, 2)
    assert (t.bad_as_expected, t.bad_declared) == (2, 2)
    assert abs(t.median_focus - (1.0 + 0.25) / 2) < 1e-9
