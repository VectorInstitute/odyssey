"""Auto-interpreted findings in the concept-bottleneck report."""

from odyssey.reporting.concept_bottleneck_report import (
    build_alert_finding,
    build_findings,
)


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


def test_build_findings_degrades_gracefully_when_no_code_type_bucket_hits_threshold() -> (
    None
):
    # Real bug this guards against: `big = {k: v for ... if n_predictions >=
    # 50_000}` followed by max(big, ...)/min(big, ...) crashed report
    # generation entirely whenever no code-type bucket reached that
    # threshold (a real, plausible shape for a small/subset run) -- must
    # degrade to a "threshold not met" note instead of raising ValueError
    # on an empty dict.
    loss_curves = {
        "train": {"step": [1, 2, 3], "task_loss": [1.0, 0.8, 0.6]},
        "val": {"task_loss": [0.9, 0.7, 0.5]},
    }
    inference = {
        "task_metrics": {
            "cross_entropy": 1.2,
            "top1_accuracy": 0.5,
            "top5_accuracy": 0.8,
        },
        "task_metrics_by_code_type": {
            "lab": {"n_predictions": 100, "top1_accuracy": 0.6, "perplexity": 3.0},
            "diagnosis": {"n_predictions": 50, "top1_accuracy": 0.4, "perplexity": 5.0},
        },
        "concept_metrics": [],
        "observability_metrics": [],
    }

    findings = build_findings(loss_curves, inference, supervision="stay")

    assert "50,000-prediction threshold" in findings["task"]
    assert "by-type table above" in findings["task"]


def test_build_findings_reports_best_and_worst_type_when_threshold_is_met() -> None:
    # Pins the unchanged success-path phrasing (byte-identical to before
    # this fix) alongside the new empty-`big` fallback above.
    loss_curves = {
        "train": {"step": [1, 2, 3], "task_loss": [1.0, 0.8, 0.6]},
        "val": {"task_loss": [0.9, 0.7, 0.5]},
    }
    inference = {
        "task_metrics": {
            "cross_entropy": 1.2,
            "top1_accuracy": 0.5,
            "top5_accuracy": 0.8,
        },
        "task_metrics_by_code_type": {
            "lab": {"n_predictions": 60_000, "top1_accuracy": 0.7, "perplexity": 2.0},
            "diagnosis": {
                "n_predictions": 55_000,
                "top1_accuracy": 0.3,
                "perplexity": 6.0,
            },
        },
        "concept_metrics": [],
        "observability_metrics": [],
    }

    findings = build_findings(loss_curves, inference, supervision="stay")

    assert "lab is the most predictable" in findings["task"]
    assert "diagnosis the least" in findings["task"]
    assert "threshold" not in findings["task"]
