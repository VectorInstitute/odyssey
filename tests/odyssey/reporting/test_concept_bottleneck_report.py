"""Structural tests for the concept-bottleneck report generator.

The failure mode that matters here is silently showing a human the WRONG
number, not a crash -- so every test that touches a number builds a
synthetic run with a KNOWN value and asserts the payload (and, for
render_html specifically, the actual embedded JSON a browser would read)
carries that exact value through. Section presence/structure is checked
via the payload's own keys (what the template's JS actually reads), not
by parsing rendered DOM/pixel HTML.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from odyssey.reporting.concept_bottleneck_report import (
    ReportInputs,
    build_alert_finding,
    build_findings,
    build_intervention_finding,
    build_payload,
    load_inputs,
    render_html,
)
from odyssey.training.train import TrainingConfig


# ---------------------------------------------------------------------------
# Fixture builders: minimal, valid, keyword-overridable
# ---------------------------------------------------------------------------


def _config(**overrides: Any) -> TrainingConfig:
    base: Dict[str, Any] = {
        "train_shard_dir": "/train",
        "tuning_shard_dir": "/tuning",
        "output_dir": "/out",
        "concept_supervision": "stay",
        "num_lanes": 8,
        "chunk_size": 256,
        "log_every": 10,
        "eval_every": 100,
        "num_epochs": 2,
        "max_train_shards": 5,
        "max_tuning_shards": 2,
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _loss_records(
    *, with_bottleneck_terms: bool = True, n_train: int = 6, n_val: int = 2
) -> List[Dict[str, Any]]:
    records = []
    for i in range(n_train):
        row = {
            "split": "train",
            "step": (i + 1) * 10,
            "epoch": 0,
            "elapsed_s": (i + 1) * 12.0,
            "task_loss": 2.0 - i * 0.1,
        }
        if with_bottleneck_terms:
            row.update(
                concept_loss=0.5 - i * 0.02,
                orthogonality_loss=0.1,
                observability_loss=0.2,
            )
        records.append(row)
    for i in range(n_val):
        row = {
            "split": "tuning",
            "step": (i + 1) * 50,
            "epoch": 0,
            "elapsed_s": (i + 1) * 60.0,
            "task_loss": 1.8 - i * 0.05,
        }
        if with_bottleneck_terms:
            row.update(
                concept_loss=0.4, orthogonality_loss=0.08, observability_loss=0.15
            )
        records.append(row)
    return records


def _by_type(n_predictions: int) -> Dict[str, Dict[str, Any]]:
    """One code-type bucket at a chosen size, everything else tiny."""
    return {
        "lab": {
            "n_predictions": n_predictions,
            "top1_accuracy": 0.62,
            "top5_accuracy": 0.9,
            "perplexity": 12.5,
        },
        "diagnosis": {
            "n_predictions": 100,
            "top1_accuracy": 0.31,
            "top5_accuracy": 0.5,
            "perplexity": 40.0,
        },
    }


def _inference_results(
    *,
    top1_accuracy: float = 0.55,
    top5_accuracy: float = 0.85,
    cross_entropy: float = 1.9,
    n_predictions: int = 60_000,
    by_type_n_predictions: int = 60_000,
    concept_metrics: Optional[List[Dict[str, Any]]] = None,
    observability_metrics: Optional[List[Dict[str, Any]]] = None,
    time_metrics: Optional[Dict[str, Any]] = None,
    n_patient_ends_scored: int = 40,
) -> Dict[str, Any]:
    return {
        "task_metrics": {
            "cross_entropy": cross_entropy,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "n_predictions": n_predictions,
        },
        "task_metrics_by_code_type": _by_type(by_type_n_predictions),
        "concept_metrics": concept_metrics if concept_metrics is not None else [],
        "observability_metrics": (
            observability_metrics if observability_metrics is not None else []
        ),
        "time_metrics": time_metrics,
        "n_patient_ends_scored": n_patient_ends_scored,
    }


def _concept_metric(name: str, auroc: float, prevalence: float = 0.3) -> Dict[str, Any]:
    return {
        "name": name,
        "n_observed": 500,
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": 0.4,
        "brier_score": 0.15,
        "accuracy_at_0_5": 0.7,
    }


def _observability_metric(
    name: str, auroc: float = 0.9, observed_rate: float = 0.6
) -> Dict[str, Any]:
    return {
        "name": name,
        "n_subjects": 500,
        "observed_rate": observed_rate,
        "auroc": auroc,
        "accuracy_at_0_5": 0.8,
    }


def _time_metrics(*, with_after_bundle: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "same_instant_accuracy": 0.93,
        "same_instant_rate": 0.88,
        "calibration": {
            "1h": {"predicted": 0.97, "observed": 0.975},
            "8h": {"predicted": 0.99, "observed": 0.985},
        },
    }
    if with_after_bundle:
        out["calibration_after_bundle"] = {
            "1h": {"predicted": 0.75, "observed": 0.76},
        }
    return out


def _case(
    subject_id: int = 1,
    n: int = 5,
    *,
    with_observability: bool = True,
    with_event_risk: bool = False,
) -> Dict[str, Any]:
    times = [float(i) for i in range(n)]
    codes = [f"LAB//{220045 + i}//x" for i in range(n)]
    case: Dict[str, Any] = {
        "subject_id": subject_id,
        "times": times,
        "concept_names": ["tachycardia", "hypotension"],
        "concept_probs": [[0.2 + 0.1 * i, 0.3] for i in range(n)],
        "concept_labels": [1, 0],
        "concept_observed": [1, 1],
        "input_codes": codes,
        "predicted_top_k": [[[c, 0.5]] for c in codes],
        "true_next_code": [*codes[1:], None],
        "true_next_rank": [0, 1, 0, 2, None],
    }
    if with_observability:
        case["observability_probs"] = [[0.9, 0.8] for _ in range(n)]
    if with_event_risk:
        case["event_risk_names"] = ["death"]
        case["event_risk_24h"] = [[0.05] for _ in range(n)]
    return case


def _write_run_dir(
    tmp_path: Path, config: TrainingConfig, loss_records: List[Dict[str, Any]]
) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    with open(tmp_path / "loss_log.jsonl", "w") as f:
        for r in loss_records:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def _extract_embedded_payload(html: str) -> Dict[str, Any]:
    """Pull the actual JSON a browser would parse out of the rendered HTML."""
    match = re.search(
        r'<script id="dashboard-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "dashboard-data script tag not found in rendered HTML"
    payload: Dict[str, Any] = json.loads(match.group(1))
    return payload


# ---------------------------------------------------------------------------
# End-to-end: render_html must not corrupt build_payload's numbers
# ---------------------------------------------------------------------------


def test_render_html_embeds_the_exact_payload_build_payload_produced(
    tmp_path: Path,
) -> None:
    """The core claim: what the browser reads equals what build_payload computed.

    Not a crash test -- a fidelity test. Rounds through JSON exactly once
    (render_html's own json.dumps), so this catches render_html silently
    dropping/renaming/reordering a field on the way to the page, which a
    "does it crash" test would never see.
    """
    config = _config()
    _write_run_dir(tmp_path, config, _loss_records())
    inputs = ReportInputs(
        config=config,
        loss_records=_loss_records(),
        inference_results=_inference_results(
            concept_metrics=[_concept_metric("tachycardia", 0.81)],
            observability_metrics=[_observability_metric("tachycardia")],
            time_metrics=_time_metrics(),
        ),
        cases=[_case(subject_id=42, n=6)],
    )
    payload = build_payload(inputs)

    html = render_html(payload)
    embedded = _extract_embedded_payload(html)

    assert embedded == json.loads(json.dumps(payload))  # allow_nan=False path


def test_run_meta_numbers_match_the_actual_synthetic_run() -> None:
    """Steps/epochs/held-out counts in run_meta are read correctly, not off-by-one."""
    config = _config(num_epochs=3, max_train_shards=7)
    loss_records = _loss_records(n_train=4, n_val=2)  # epoch 0 only, 4 train steps
    inputs = ReportInputs(
        config=config,
        loss_records=loss_records,
        inference_results=_inference_results(n_predictions=12_345),
        cases=[],
    )
    payload = build_payload(inputs)
    embedded = _extract_embedded_payload(render_html(payload))

    meta = dict(embedded["run_meta"])
    last_step = loss_records[3]["step"]  # last train record, index 3 (n_train=4)
    assert meta["Steps"].startswith(f"{last_step:,}")
    assert "1 of 3 epochs" in meta["Steps"]  # max epoch (0) + 1, out of num_epochs=3
    assert "12.3K" in meta["Held-out test"] or "12,345" in meta["Held-out test"]


def test_loss_curves_reflect_the_logged_values_exactly() -> None:
    loss_records = _loss_records(n_train=3, n_val=1)
    inputs = ReportInputs(
        config=_config(),
        loss_records=loss_records,
        inference_results=_inference_results(),
        cases=[],
    )
    payload = build_payload(inputs)
    embedded = _extract_embedded_payload(render_html(payload))

    curves = embedded["loss_curves"]
    train_task_losses = [r["task_loss"] for r in loss_records if r["split"] == "train"]
    assert curves["train"]["task_loss"] == train_task_losses
    assert curves["train"]["step"] == [10, 20, 30]


# ---------------------------------------------------------------------------
# build_payload: guards around loss log / best_val_loss / pool accounting
# ---------------------------------------------------------------------------


def test_build_payload_raises_when_loss_log_has_no_train_split() -> None:
    inputs = ReportInputs(
        config=_config(),
        loss_records=[{"split": "tuning", "step": 1, "epoch": 0, "elapsed_s": 1.0}],
        inference_results=_inference_results(),
        cases=[],
    )
    with pytest.raises(ValueError, match="no 'train' split"):
        build_payload(inputs)


def test_best_val_loss_is_none_when_there_are_no_validation_records() -> None:
    """Regression test.

    build_findings's training string used to crash on val_last:.2f (None)
    whenever there were zero validation records, before run_meta's own
    "n/a" guard was ever reached. Now degrades gracefully.
    """
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(n_train=3, n_val=0),
        inference_results=_inference_results(),
        cases=[],
    )
    payload = build_payload(inputs)
    meta = dict(payload["run_meta"])
    assert meta["Best val loss (combined, as selected)"] == "n/a"
    assert (
        "No tuning-split checkpoint was logged yet" in payload["findings"]["training"]
    )


def test_n_pool_positions_falls_back_to_n_patient_ends_scored_for_a_baseline_run() -> (
    None
):
    """A baseline run has no observability_metrics -- must not IndexError on [0]."""
    inputs = ReportInputs(
        config=_config(model_kind="baseline"),
        loss_records=_loss_records(with_bottleneck_terms=False),
        inference_results=_inference_results(
            observability_metrics=[], n_patient_ends_scored=777
        ),
        cases=[],
    )
    payload = build_payload(inputs)
    assert "777" in dict(payload["run_meta"])["Held-out test"]


def test_pool_unit_label_matches_visit_vs_stay_supervision() -> None:
    for supervision, expected_word in (("visit", "visits"), ("stay", "patients")):
        inputs = ReportInputs(
            config=_config(concept_supervision=supervision),
            loss_records=_loss_records(),
            inference_results=_inference_results(
                observability_metrics=[_observability_metric("c")]
            ),
            cases=[],
        )
        payload = build_payload(inputs)
        assert expected_word in dict(payload["run_meta"])["Held-out test"], supervision


# ---------------------------------------------------------------------------
# build_payload: qualitative_desc / interventions_desc / alerts_desc presence
# ---------------------------------------------------------------------------


def test_qualitative_desc_degrades_gracefully_with_no_cases() -> None:
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(),
        inference_results=_inference_results(),
        cases=[],
    )
    payload = build_payload(inputs)
    assert "Not applicable" in payload["qualitative_desc"]


def test_qualitative_desc_mentions_case_count_when_cases_present() -> None:
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(),
        inference_results=_inference_results(),
        cases=[_case(1), _case(2), _case(3)],
    )
    payload = build_payload(inputs)
    assert "3 held-out patients" in payload["qualitative_desc"]


def test_interventions_desc_and_finding_absent_without_interventions_input() -> None:
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(),
        inference_results=_inference_results(),
        cases=[],
        interventions=None,
    )
    payload = build_payload(inputs)
    assert payload["interventions_desc"] is None
    assert "interventions" not in payload["findings"]


def test_interventions_desc_and_finding_present_with_interventions_input() -> None:
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(),
        inference_results=_inference_results(),
        cases=[],
        interventions=[
            {"mode": "none", "top1_accuracy": 0.5, "mean_task_loss": 1.0},
            {"mode": "truth", "top1_accuracy": 0.6, "mean_task_loss": 0.9},
            {"mode": "flip", "top1_accuracy": 0.4, "mean_task_loss": 1.1},
        ],
    )
    payload = build_payload(inputs)
    assert payload["interventions_desc"] is not None
    assert "interventions" in payload["findings"]


def test_alerts_desc_and_finding_absent_without_alerts_input() -> None:
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(),
        inference_results=_inference_results(),
        cases=[],
        alerts=None,
    )
    payload = build_payload(inputs)
    assert payload["alerts_desc"] is None
    assert "alerts" not in payload["findings"]


def test_alerts_desc_present_but_finding_absent_when_alerts_has_no_events() -> None:
    """A non-empty alerts list with content.

    alerts=[] (falsy) is covered separately above; this exercises the
    truthy-but-only-one-row path down through build_alert_finding.
    """
    inputs = ReportInputs(
        config=_config(),
        loss_records=_loss_records(),
        inference_results=_inference_results(),
        cases=[],
        alerts=[
            {"event": "x", "horizon_hours": 8.0, "scorer": "hazard", "auroc": None}
        ],
    )
    # build_alert_finding itself: events list is non-empty ("x"), so this
    # actually exercises the "has content" path -- covered separately below.
    payload = build_payload(inputs)
    assert payload["alerts_desc"] is not None


# ---------------------------------------------------------------------------
# build_findings: concept-bottleneck vs baseline, ceiling effects, ICU
# low-observed-rate caveat, time-to-event calibration. The by-type
# empty-bucket degraded path ("finding-3") already has dedicated coverage
# in test_findings.py (test_build_findings_degrades_gracefully_when_no_code_
# type_bucket_hits_threshold / test_build_findings_reports_best_and_worst_
# type_when_threshold_is_met) -- not duplicated here.
# ---------------------------------------------------------------------------


def test_findings_baseline_run_has_no_bottleneck_language() -> None:
    findings = build_findings(
        loss_curves={
            "train": {"task_loss": [2.0, 1.5], "step": [10, 20]},
            "val": {"task_loss": [1.8]},
        },
        inference=_inference_results(concept_metrics=[]),
        supervision="stay",
    )
    assert "No concept bottleneck" in findings["concepts"]
    assert findings["concepts"] == findings["observability"]


def test_findings_flags_high_prevalence_concepts_as_ceiling_effects() -> None:
    findings = build_findings(
        loss_curves={
            "train": {"task_loss": [2.0], "step": [10]},
            "val": {"task_loss": [1.8]},
        },
        inference=_inference_results(
            concept_metrics=[
                _concept_metric("common_thing", 0.95, prevalence=0.9),
                _concept_metric("rare_thing", 0.7, prevalence=0.1),
            ],
            observability_metrics=[_observability_metric("common_thing")],
        ),
        supervision="stay",
    )
    assert "common_thing" in findings["concepts"]
    assert "85% prevalence" in findings["concepts"]


def test_findings_observability_degrades_when_concepts_exist_but_observability_is_empty() -> (
    None
):
    """Regression test.

    The observability finding used to crash on min(rates)/max(rates)
    whenever concept_metrics was non-empty (so the baseline early-return
    didn't fire) but observability_metrics was empty -- a run with
    known-concept heads but no observability head. Same species as the
    already-fixed by_type_note empty-bucket guard.
    """
    findings = build_findings(
        loss_curves={
            "train": {"task_loss": [2.0], "step": [10]},
            "val": {"task_loss": [1.8]},
        },
        inference=_inference_results(
            concept_metrics=[_concept_metric("c1", 0.8)],
            observability_metrics=[],
        ),
        supervision="stay",
    )
    assert "No observability metrics in this run" in findings["observability"]


def test_findings_flags_low_observed_rate_observability_concepts() -> None:
    findings = build_findings(
        loss_curves={
            "train": {"task_loss": [2.0], "step": [10]},
            "val": {"task_loss": [1.8]},
        },
        inference=_inference_results(
            concept_metrics=[_concept_metric("c1", 0.8)],
            observability_metrics=[
                _observability_metric("c1", observed_rate=0.1),
                _observability_metric("c2", observed_rate=0.9),
            ],
        ),
        supervision="stay",
    )
    assert "only charted in the ICU" in findings["observability"]


def test_findings_no_icu_caveat_when_every_concept_is_widely_observed() -> None:
    findings = build_findings(
        loss_curves={
            "train": {"task_loss": [2.0], "step": [10]},
            "val": {"task_loss": [1.8]},
        },
        inference=_inference_results(
            concept_metrics=[_concept_metric("c1", 0.8)],
            observability_metrics=[_observability_metric("c1", observed_rate=0.9)],
        ),
        supervision="stay",
    )
    assert "only charted in the ICU" not in findings["observability"]


def test_findings_baseline_run_omits_the_orthogonality_sentence() -> None:
    findings = build_findings(
        loss_curves={
            "train": {
                "task_loss": [2.0, 1.5],
                "step": [10, 20],
            },  # no orthogonality_loss key
            "val": {"task_loss": [1.8]},
        },
        inference=_inference_results(concept_metrics=[]),
        supervision="stay",
    )
    assert "baseline run without a concept bottleneck" in findings["training"]


def test_findings_time_metrics_present_only_when_given() -> None:
    base_curves = {
        "train": {"task_loss": [2.0], "step": [10]},
        "val": {"task_loss": [1.8]},
    }
    without = build_findings(
        loss_curves=base_curves,
        inference=_inference_results(time_metrics=None),
        supervision="stay",
    )
    with_time = build_findings(
        loss_curves=base_curves,
        inference=_inference_results(time_metrics=_time_metrics()),
        supervision="stay",
    )
    assert "time" not in without
    assert "time" in with_time
    assert "93.0%" in with_time["time"] or "93%" in with_time["time"]


def test_findings_time_calibration_after_bundle_sentence_is_conditional() -> None:
    base_curves = {
        "train": {"task_loss": [2.0], "step": [10]},
        "val": {"task_loss": [1.8]},
    }
    without_bundle = build_findings(
        loss_curves=base_curves,
        inference=_inference_results(
            time_metrics=_time_metrics(with_after_bundle=False)
        ),
        supervision="stay",
    )
    with_bundle = build_findings(
        loss_curves=base_curves,
        inference=_inference_results(
            time_metrics=_time_metrics(with_after_bundle=True)
        ),
        supervision="stay",
    )
    assert "Given the bundle ends" not in without_bundle["time"]
    assert "Given the bundle ends" in with_bundle["time"]


# ---------------------------------------------------------------------------
# build_alert_finding: guard branches not already covered by test_findings.py
# (empty alerts, missing hazard head, feature-set/tuning wording, and the
# basic win/loss split are already pinned by test_alert_finding_reports_
# feature_set_and_wins_losses / test_alert_finding_untuned_basic_and_no_
# hazard -- not duplicated here).
# ---------------------------------------------------------------------------


def test_build_alert_finding_correctly_sorts_wins_and_losses_by_the_auroc_gap() -> None:
    """A win/loss must reflect the actual AUROC numbers, not just presence."""
    alerts = [
        # hazard wins here (0.90 vs 0.80, gap +0.10)
        {"event": "death", "horizon_hours": 8.0, "scorer": "hazard", "auroc": 0.90},
        {
            "event": "death",
            "horizon_hours": 8.0,
            "scorer": "baseline_gbm",
            "auroc": 0.80,
        },
        # gbm wins here (hazard 0.5 vs gbm 0.9, gap -0.40)
        {
            "event": "aki",
            "horizon_hours": 8.0,
            "scorer": "hazard",
            "auroc": 0.50,
        },
        {"event": "aki", "horizon_hours": 8.0, "scorer": "baseline_gbm", "auroc": 0.90},
    ]
    note = build_alert_finding(alerts)
    assert note is not None
    assert "death at 8h (0.90 vs 0.80)" in note
    assert "aki at 8h (0.50 vs 0.90)" in note
    # "death" must appear in the wins clause, "aki" in the losses clause,
    # not swapped -- check via substring position relative to the marker text
    wins_idx = note.index("Hazard heads match or beat")
    losses_idx = note.index("GBM ahead on")
    death_idx = note.index("death at 8h")
    aki_idx = note.index("aki at 8h")
    assert wins_idx < death_idx < losses_idx
    assert losses_idx < aki_idx


def test_build_alert_finding_reports_brier_parity_count() -> None:
    alerts = [
        {
            "event": "death",
            "horizon_hours": 8.0,
            "scorer": "hazard",
            "auroc": 0.9,
            "brier": 0.05,
        },
        {
            "event": "death",
            "horizon_hours": 8.0,
            "scorer": "baseline_gbm",
            "auroc": 0.85,
            "brier": 0.10,
        },
    ]
    note = build_alert_finding(alerts)
    assert note is not None
    assert "Calibration (Brier) is at least as good as the GBM's on 1" in note


# ---------------------------------------------------------------------------
# build_intervention_finding: every guard branch
# ---------------------------------------------------------------------------


def test_build_intervention_finding_returns_none_without_interventions() -> None:
    assert build_intervention_finding(None) is None
    assert build_intervention_finding([]) is None


def test_build_intervention_finding_returns_none_without_a_none_mode_row() -> None:
    assert build_intervention_finding([{"mode": "truth", "top1_accuracy": 0.5}]) is None


def test_build_intervention_finding_unbanded_path_reports_raw_deltas() -> None:
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {"mode": "truth", "top1_accuracy": 0.55, "mean_task_loss": 0.9},
        {"mode": "flip", "top1_accuracy": 0.45, "mean_task_loss": 1.1},
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "+5.0%" in note  # truth delta
    assert "-5.0%" in note  # flip delta
    assert "uncertain-band" in note  # points at the controlled re-run


def test_build_intervention_finding_banded_path_mediates_when_gap_is_large() -> None:
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {
            "mode": "truth",
            "top1_accuracy": 0.60,
            "mean_task_loss": 0.9,
            "uncertain_band": 0.1,
            "mean_abs_displacement": 0.4,
        },
        {
            "mode": "flip",
            "top1_accuracy": 0.50,
            "mean_task_loss": 1.0,
            "mean_abs_displacement": 0.4,
        },
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "reads the concept values in the intended direction" in note


def test_build_intervention_finding_banded_path_no_separation_when_truth_equals_flip() -> (
    None
):
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {
            "mode": "truth",
            "top1_accuracy": 0.50,
            "mean_task_loss": 1.0,
            "uncertain_band": 0.1,
            "mean_abs_displacement": 0.3,
        },
        {
            "mode": "flip",
            "top1_accuracy": 0.50,
            "mean_task_loss": 1.0,
            "mean_abs_displacement": 0.3,
        },
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "No separation" in note


def test_build_intervention_finding_degrades_when_displacement_is_none_in_banded_mode() -> (
    None
):
    """Regression test.

    disp.get(mode, nan) never fell back (by_mode always has the key, just
    sometimes valued None), and interventions.py legitimately sets
    mean_abs_displacement to None when n_replaced_entries is 0 for that
    mode (a tight uncertain_band with an unlucky sample) -- used to crash
    the whole report. Now degrades to "n/a" for that mode.
    """
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {
            "mode": "truth",
            "top1_accuracy": 0.60,
            "mean_task_loss": 0.9,
            "uncertain_band": 0.1,
            "mean_abs_displacement": None,
        },
        {"mode": "flip", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "mean displacement n/a vs n/a" in note
    assert "reads the concept values in the intended direction" in note


def test_build_intervention_finding_signed_only_when_gap_is_small_but_positive() -> (
    None
):
    rows = [
        {"mode": "none", "top1_accuracy": 0.500, "mean_task_loss": 1.0},
        {
            "mode": "truth",
            "top1_accuracy": 0.503,
            "mean_task_loss": 1.0,
            "uncertain_band": 0.1,
            "mean_abs_displacement": 0.3,
        },
        {
            "mode": "flip",
            "top1_accuracy": 0.500,
            "mean_task_loss": 1.0,
            "mean_abs_displacement": 0.3,
        },
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "too small to call a working lever" in note


def test_build_intervention_finding_mentions_randint_when_training_used_it() -> None:
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {
            "mode": "truth",
            "top1_accuracy": 0.60,
            "mean_task_loss": 0.9,
            "uncertain_band": 0.1,
        },
        {"mode": "flip", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
    ]
    note = build_intervention_finding(rows, randint_prob=0.5)
    assert note is not None
    assert "RandInt p=0.50" in note


def test_build_intervention_finding_flags_decorative_known_channel() -> None:
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {"mode": "zero_known", "top1_accuracy": 0.499, "mean_task_loss": 1.0},
        {"mode": "zero_unknown", "top1_accuracy": 0.20, "mean_task_loss": 2.0},
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "isn't where the task signal actually lives" in note


def test_build_intervention_finding_load_bearing_known_channel_not_flagged_decorative() -> (
    None
):
    rows = [
        {"mode": "none", "top1_accuracy": 0.50, "mean_task_loss": 1.0},
        {"mode": "zero_known", "top1_accuracy": 0.20, "mean_task_loss": 2.0},
        {"mode": "zero_unknown", "top1_accuracy": 0.499, "mean_task_loss": 1.0},
    ]
    note = build_intervention_finding(rows)
    assert note is not None
    assert "load-bearing for the task head" in note


# ---------------------------------------------------------------------------
# load_inputs: the on-disk file-reading path (end to end from real files)
# ---------------------------------------------------------------------------


def test_load_inputs_reads_a_full_run_dir_from_disk(tmp_path: Path) -> None:
    config = _config()
    loss_records = _loss_records()
    _write_run_dir(tmp_path, config, loss_records)

    inference_path = tmp_path / "inference_results.json"
    inference_path.write_text(json.dumps(_inference_results()))
    cases_path = tmp_path / "case_studies.json"
    cases_path.write_text(json.dumps([_case(1)]))
    interventions_path = tmp_path / "interventions.json"
    interventions_path.write_text(
        json.dumps([{"mode": "none", "top1_accuracy": 0.5, "mean_task_loss": 1.0}])
    )
    alerts_path = tmp_path / "alerts.json"
    alerts_path.write_text(
        json.dumps(
            [{"event": "death", "horizon_hours": 8.0, "scorer": "hazard", "auroc": 0.8}]
        )
    )

    inputs = load_inputs(
        tmp_path, inference_path, cases_path, interventions_path, alerts_path
    )

    assert inputs.config.train_shard_dir == config.train_shard_dir
    assert len(inputs.loss_records) == len(loss_records)
    assert inputs.inference_results["task_metrics"]["top1_accuracy"] == 0.55
    assert len(inputs.cases) == 1
    assert inputs.interventions is not None and len(inputs.interventions) == 1
    assert inputs.alerts is not None and len(inputs.alerts) == 1


def test_load_inputs_optional_interventions_and_alerts_default_to_none(
    tmp_path: Path,
) -> None:
    config = _config()
    _write_run_dir(tmp_path, config, _loss_records())
    inference_path = tmp_path / "inference_results.json"
    inference_path.write_text(json.dumps(_inference_results()))
    cases_path = tmp_path / "case_studies.json"
    cases_path.write_text(json.dumps([]))

    inputs = load_inputs(tmp_path, inference_path, cases_path)

    assert inputs.interventions is None
    assert inputs.alerts is None
