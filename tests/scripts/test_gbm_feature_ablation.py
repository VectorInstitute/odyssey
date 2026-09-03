"""The GBM feature-group ablation partitions the strong panel and refits on subsets."""

import numpy as np
import polars as pl
import pytest

from odyssey.data.alert_events import EventTimes
from odyssey.inference.alerts import IndexRow, _fit_baseline_grid
from odyssey.inference.baseline_features import feature_names
from scripts.gbm_feature_ablation import (
    GROUP_ORDER,
    _cell_rows,
    feature_groups,
    gap_share,
    held_out_rows,
    tuned_params,
    variants,
)


def test_groups_partition_the_strong_panel_exactly() -> None:
    names = feature_names()
    groups = feature_groups(names)
    sizes = {g: len(c) for g, c in groups.items()}
    assert sizes == {
        "static": 3,
        "recency": 64,
        "latest_value": 48,
        "summary_stats": 384,
        "counts_occurrence": 110,
    }
    seen = sorted(i for c in groups.values() for i in c)
    assert seen == list(range(len(names)))
    assert tuple(groups) == GROUP_ORDER
    # Each counting column is a count; each recency column a time-since.
    assert all(
        names[i].endswith((".n_6h", ".n_24h", ".n_visit", ".ever_visit"))
        or names[i] in {"n_prior_visits", "n_events_visit"}
        for i in groups["counts_occurrence"]
    )
    assert all(
        names[i].endswith(".hours_since_last")
        or names[i] in {"hours_into_visit", "in_icu", "hours_since_icu_admission"}
        for i in groups["recency"]
    )


def test_variants_are_complements() -> None:
    names = feature_names()
    groups = feature_groups(names)
    cols = variants(groups, len(names))
    assert cols["full"].tolist() == list(range(len(names)))
    for g in GROUP_ORDER:
        drop, keep = cols[f"drop:{g}"], cols[f"keep:{g}"]
        assert len(drop) + len(keep) == len(names)
        assert not set(drop.tolist()) & set(keep.tolist())


def test_gap_share_directions() -> None:
    # full 0.90, hazard 0.80: dropping a group to 0.85 explains half the gap
    # uniquely; keeping it alone at 0.85 also recovers half.
    assert gap_share(0.90, 0.80, 0.85, drop=True) == pytest.approx(0.5)
    assert gap_share(0.90, 0.80, 0.85, drop=False) == pytest.approx(0.5)
    assert gap_share(0.80, 0.80, 0.85, drop=True) is None


def test_fixed_params_skip_tuning_and_are_recorded() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 6)).astype(np.float32)
    rows = [
        IndexRow(subject_id=i // 4, visit_id=i // 4, time_hours=float(i % 4))
        for i in range(n)
    ]
    # onset for half the subjects at hour 10: every row of theirs is at risk
    # and positive within 24 h; the rest are followed to hour 100.
    onset = {(s, s): 10.0 for s in range(0, n // 4, 2)}
    censor = {(s, s): 100.0 for s in range(n // 4)}
    times = EventTimes(onset=onset, censor=censor, subject_scoped=False)
    fixed = {
        24.0: ({"learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 5}, 12)
    }
    models = _fit_baseline_grid(
        x,
        rows,
        times,
        horizons=[24.0],
        feature_set="strong",
        seed=0,
        tune=True,
        event_name="e",
        fixed=fixed,
    )
    model = models[24.0]
    assert model.params == {
        "learning_rate": 0.1,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 5,
        "n_rounds": 12.0,
    }
    assert model.clf.max_iter == 12
    assert model.predict_proba(x[:5]).shape == (5,)


def test_dump_helpers_read_rows_outcomes_and_hazard() -> None:
    dump = pl.DataFrame(
        {
            "event": ["death", "death", "death", "aki"],
            "subject_id": [1.0, 1.0, 2.0, 3.0],
            "visit_id": [1.0, 1.0, 2.0, 3.0],
            "time_hours": [4.0, 8.0, 4.0, 4.0],
            "y@24h": [1.0, None, 0.0, 0.0],
            "hazard@24h": [0.9, 0.5, 0.2, 0.1],
        }
    )
    rows = held_out_rows(dump, ["death"])
    assert [(r.subject_id, r.time_hours) for r in rows["death"]] == [
        (1, 4.0),
        (1, 8.0),
        (2, 4.0),
    ]
    mask, y, hazard = _cell_rows(dump, "death", 24.0)
    assert mask.tolist() == [True, False, True]
    assert y.tolist() == [1, 0] and hazard.tolist() == [0.9, 0.2]


def test_tuned_params_come_from_the_gbm_records(tmp_path) -> None:
    path = tmp_path / "alerts.json"
    path.write_text(
        '[{"event": "death", "horizon_hours": 8.0, "scorer": "baseline_gbm", '
        '"baseline_params": {"learning_rate": 0.05, "max_leaf_nodes": 31, '
        '"min_samples_leaf": 20, "n_rounds": 219.0}}, '
        '{"event": "death", "horizon_hours": 8.0, "scorer": "hazard"}]'
    )
    params = tuned_params(path)
    assert params == {
        ("death", 8.0): (
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
            219,
        )
    }
