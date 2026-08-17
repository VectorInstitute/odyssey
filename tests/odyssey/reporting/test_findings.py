"""Auto-interpreted findings in the concept-bottleneck report."""

from odyssey.reporting.concept_bottleneck_report import build_alert_finding


def _row(event: str, h: float, scorer: str, auroc: float, **extra):
    row = {
        "event": event,
        "horizon_hours": h,
        "scorer": scorer,
        "auroc": auroc,
        "brier": extra.pop("brier", None),
    }
    row.update(extra)
    return row


def test_alert_finding_reports_feature_set_and_wins_losses() -> None:
    alerts = [
        _row("death", 8.0, "hazard", 0.88, brier=0.005),
        _row(
            "death",
            8.0,
            "baseline_gbm",
            0.80,
            brier=0.007,
            baseline_feature_set="strong",
            baseline_n_features=609,
            baseline_params={"learning_rate": 0.05, "n_rounds": 120.0},
        ),
        _row("acute_kidney_injury", 8.0, "hazard", 0.70, brier=0.02),
        _row(
            "acute_kidney_injury",
            8.0,
            "baseline_gbm",
            0.75,
            brier=0.019,
            baseline_feature_set="strong",
            baseline_n_features=609,
            baseline_params={"learning_rate": 0.1, "n_rounds": 80.0},
        ),
    ]
    text = build_alert_finding(alerts)
    assert text is not None
    assert "<i>strong</i> feature set (609 features), tuned" in text
    assert "death at 8h (0.88 vs 0.80)" in text
    assert "GBM ahead on: acute kidney injury at 8h (0.70 vs 0.75)" in text
    assert "Brier" in text


def test_alert_finding_untuned_basic_and_no_hazard() -> None:
    alerts = [
        _row("death", 8.0, "concept", 0.6),
        _row(
            "death",
            8.0,
            "baseline_gbm",
            0.8,
            brier=0.01,
            baseline_feature_set="basic",
            baseline_n_features=17,
            baseline_params=None,
        ),
    ]
    text = build_alert_finding(alerts)
    assert text is not None
    assert "<i>basic</i> feature set (17 features), untuned" in text
    assert "no per-event hazard heads" in text
    assert build_alert_finding([]) is None
