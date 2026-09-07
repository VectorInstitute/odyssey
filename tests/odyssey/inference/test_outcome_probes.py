"""Frozen state-transition probes: folding, the at-risk requirement, round trip."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import (
    STATE_TRANSITION_EVENTS,
    STATE_TRANSITION_REQUIRES,
    EventTimes,
)
from odyssey.data.concepts import concepts_for_source
from odyssey.inference.outcome_probes import (
    OutcomeProbes,
    fold_scaler,
    required_mask,
)
from odyssey.inference.steering import (
    READOUT_EXPECTATIONS,
    TRANSITION_EXPECTATIONS,
    expectations_for,
    readout_expectations_for,
)


def test_folded_probe_matches_the_scaled_logistic_regression() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6)) * np.array([1, 5, 0.2, 1, 3, 1])
    y = (x[:, 1] - 2 * x[:, 2] + rng.normal(size=500) > 0).astype(int)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(x), y)
    w, b = fold_scaler(scaler, clf)
    expected = clf.decision_function(scaler.transform(x))
    np.testing.assert_allclose(x @ w + b, expected, rtol=1e-6, atol=1e-6)


def test_required_mask_keeps_only_rows_after_the_prior_event() -> None:
    keys = [(1, 10, 2.0), (1, 10, 8.0), (2, 20, 5.0), (3, 30, 1.0)]
    prior = EventTimes(
        onset={(1, 10): 4.0, (2, 20): 5.0}, censor={}, subject_scoped=False
    )
    mask = required_mask(keys, prior)
    # subject 1: onset at 4 h, so only the 8 h row; subject 2: onset == t counts;
    # subject 3: never had the event, never at risk of leaving it
    assert mask.tolist() == [False, True, True, False]
    assert required_mask(keys, None).all()


def test_probes_round_trip_and_score_the_right_shape(tmp_path) -> None:
    probes = OutcomeProbes(
        event_names=["icu_discharge", "hospital_discharge_alive"],
        horizons_hours=[8.0, 24.0, 72.0],
        weight=torch.randn(2, 3, 5),
        bias=torch.randn(2, 3),
        requires={"icu_discharge": "icu_admission"},
        auroc={"icu_discharge": {"train@8h": 0.7}},
    )
    probes.save(tmp_path / "p.pt")
    loaded = OutcomeProbes.load(tmp_path / "p.pt")
    features = torch.randn(7, 5)
    risk = loaded.risk(features)
    assert risk.shape == (7, 2, 3)
    assert torch.all((risk >= 0) & (risk <= 1))
    torch.testing.assert_close(risk, probes.risk(features))
    assert loaded.requires == {"icu_discharge": "icu_admission"}


def test_transition_events_and_requirements_are_consistent() -> None:
    names = {ev.name for ev in STATE_TRANSITION_EVENTS}
    assert set(STATE_TRANSITION_REQUIRES) <= names
    # every transition expectation names a real transition event and a
    # registry concept
    registry = {c.name for c in concepts_for_source("mimic_iv", task_set="v3")}
    for concept, table in TRANSITION_EXPECTATIONS.items():
        assert concept in registry, concept
        assert set(table) <= names, concept
    merged = expectations_for(
        "hypotension", ["death", "icu_discharge", "unknown_event"]
    )
    assert merged == {"death": +1, "icu_discharge": -1}


def test_readout_expectations_name_registry_opposites() -> None:
    """Every readout expectation names two registry concepts and points down."""
    registry = {c.name for c in concepts_for_source("mimic_iv", task_set="v3")}
    for concept, table in READOUT_EXPECTATIONS.items():
        assert concept in registry, concept
        for other, sign in table.items():
            assert other in registry, (concept, other)
            assert sign == -1, (concept, other)
    got = readout_expectations_for(
        "hypokalemia", ["hypokalemia", "hyperkalemia", "fever"]
    )
    assert got == {"hypokalemia": +1, "hyperkalemia": -1}
