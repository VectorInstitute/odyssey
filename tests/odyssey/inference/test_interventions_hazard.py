"""Hazard-head scoring of the label-override test (CPU, tiny fixtures)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

import odyssey.inference.interventions as interventions_module
from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.alerts import (
    HORIZONS_HOURS,
    _visit_starts,
    collect_model_scores,
    score_alerts,
)
from odyssey.inference.interventions import (
    HazardLandmarkScorer,
    InterventionResult,
    LandmarkRiskTable,
    evaluate_interventions,
    hazard_paired_summary,
    hazard_summary,
    landmark_labels,
    paired_mean_delta,
    result_to_json,
    run_streaming_intervention,
    subject_bootstrap_means,
)
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
)
from odyssey.training.train import TrainingConfig


T0 = datetime(2024, 1, 1)
CONCEPTS = concepts_for_source("mimic_iv", task_set="v1")
EVENT_NAMES = [a.name for a in ALERT_EVENTS]

# The JSON keys interventions.py wrote before hazard scoring existed.
OLD_KEYS = [
    "mode",
    "n_predictions",
    "top1_accuracy",
    "mean_task_loss",
    "top1_by_code_type",
    "n_by_code_type",
    "n_intervened_positions",
    "uncertain_band",
    "mean_abs_displacement",
    "calibrated_tau",
    "n_replaced_by_concept",
    "mean_abs_displacement_by_concept",
    "calibration_gamma",
]


def _events(n_subjects: int = 16) -> pl.DataFrame:
    """Hourly heart rates; even subjects start norepinephrine at hour 14.

    Every fourth subject also gets an ICU admission at hour 6, so the
    vasopressor and ICU cells have both outcome classes at 8 h and 24 h.
    """
    rows: list[tuple[int, str, datetime, float | None, int]] = []
    for sid in range(1, n_subjects + 1):
        hadm = 1000 + sid
        for h in range(24):
            hr = 130.0 if sid % 2 == 0 and h >= 12 else 80.0
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), hr, hadm))
        if sid % 2 == 0:
            rows.append(
                (
                    sid,
                    "MEDICATION//norepinephrine//Administered",
                    T0 + timedelta(hours=14),
                    None,
                    hadm,
                )
            )
        if sid % 4 == 0:
            rows.append(
                (sid, "ICU_ADMISSION//MICU", T0 + timedelta(hours=6), None, hadm)
            )
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def _model(vocab_size: int, with_heads: bool = True) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=vocab_size, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=vocab_size,
        num_concepts=len(CONCEPTS),
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=EVENT_NAMES if with_heads else None,
    )


class _Fixture:
    """One synthetic held-out split, its model, labels and alert targets."""

    def __init__(self) -> None:
        self.raw = _events()
        self.binned = add_value_tokens(self.raw)
        self.vocab = Vocabulary.build(self.binned["code"].to_list(), min_count=1)
        self.model = _model(len(self.vocab))
        self.labels, self.mask = build_concept_label_dicts(self.raw, CONCEPTS)
        self.first_times = build_concept_first_times(self.raw, CONCEPTS)
        self.times = all_event_times(self.raw, ALERT_EVENTS, "mimic_iv")
        self.visit_start = _visit_starts(self.raw)

    def scorer(self) -> HazardLandmarkScorer:
        assert self.model.event_heads is not None
        return HazardLandmarkScorer(
            self.model.event_heads, ALERT_EVENTS, self.visit_start
        )

    def run(
        self, mode: str, *, hazard: bool
    ) -> tuple[InterventionResult, LandmarkRiskTable | None]:
        scorer = self.scorer() if hazard else None
        result = run_streaming_intervention(
            self.model,
            self.binned,
            self.vocab,
            self.labels,
            self.mask,
            mode=mode,
            concept_first_times=self.first_times,
            supervision="stay",
            num_lanes=2,
            chunk_size=16,
            device="cpu",
            seed=0,
            hazard_scorer=scorer,
        )
        return result, (scorer.table() if scorer is not None else None)


@pytest.fixture(scope="module")
def fx() -> _Fixture:
    return _Fixture()


# ---------------------------------------------------------------------------
# (i) hazard scoring off leaves the old output untouched
# ---------------------------------------------------------------------------


def test_hazard_off_keeps_the_old_schema_and_on_does_not_move_the_numbers(
    fx: _Fixture,
) -> None:
    plain, _ = fx.run("truth", hazard=False)
    scored, table = fx.run("truth", hazard=True)
    assert list(result_to_json(plain)) == OLD_KEYS
    assert plain.hazard is None and plain.hazard_paired is None
    # Reading the heads is a side readout: the serialised next-event
    # numbers are byte for byte the same (NaN entries compare as text).
    assert json.dumps(result_to_json(scored)) == json.dumps(result_to_json(plain))
    assert table is not None and table.n_rows > 0


def test_cli_writes_the_old_json_without_the_flag_and_passes_it_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}
    hazard = {"death": {"8h": {"auroc": None, "mean_risk": 0.1}}}

    def fake_evaluate(*_args: object, **kwargs: object) -> list[InterventionResult]:
        seen.update(kwargs)
        base = InterventionResult("none", 1, 1.0, 0.5)
        if kwargs["hazard_heads"]:
            base = InterventionResult(
                "none", 1, 1.0, 0.5, hazard=hazard, hazard_paired={}
            )
        return [base]

    monkeypatch.setattr(interventions_module, "evaluate_interventions", fake_evaluate)
    argv = [
        "prog",
        "--run-dir",
        "/runs/x",
        "--held-out-shard-dir",
        "/data/held_out",
        "--output-json",
    ]
    off = tmp_path / "off.json"
    monkeypatch.setattr("sys.argv", [*argv, str(off)])
    interventions_module._main()
    written = json.loads(off.read_text())
    assert list(written[0]) == OLD_KEYS
    assert seen["hazard_heads"] is False and seen["hazard_boot"] == 1000

    on = tmp_path / "on.json"
    monkeypatch.setattr(
        "sys.argv", [*argv, str(on), "--hazard-heads", "--hazard-boot", "7"]
    )
    interventions_module._main()
    written = json.loads(on.read_text())
    assert list(written[0]) == [*OLD_KEYS, "hazard", "hazard_paired"]
    assert written[0]["hazard"] == hazard
    assert seen["hazard_heads"] is True and seen["hazard_boot"] == 7


# ---------------------------------------------------------------------------
# (ii) the landmark rows are alerts.py's rows
# ---------------------------------------------------------------------------


def test_landmark_rows_and_counts_match_the_alert_protocol(fx: _Fixture) -> None:
    rows = collect_model_scores(
        fx.model,
        fx.binned,
        fx.vocab,
        [c.name for c in CONCEPTS],
        ALERT_EVENTS,
        visit_start=fx.visit_start,
        landmark_hours=4.0,
        num_lanes=2,
        chunk_size=16,
        device="cpu",
        horizons=HORIZONS_HOURS,
    )
    alert_metrics = {
        (m.event, m.horizon_hours): m
        for m in score_alerts(rows, fx.times, horizons=HORIZONS_HOURS)
        if m.scorer == "hazard"
    }
    assert alert_metrics, "the alert protocol scored no hazard cell"

    _, table = fx.run("none", hazard=True)
    assert table is not None
    mine = set(
        zip(
            table.subject_ids.tolist(),
            table.visit_ids.tolist(),
            table.time_hours.tolist(),
        )
    )
    for event in EVENT_NAMES:
        theirs = {(r.subject_id, r.visit_id, r.time_hours) for r in rows[event]}
        assert mine == theirs, event

    summary = hazard_summary(table, landmark_labels(table, fx.times))
    for event in EVENT_NAMES:
        for h in HORIZONS_HOURS:
            cell = summary[event][f"{h:g}h"]
            metric = alert_metrics.get((event, h))
            if metric is None:
                # alerts.py skips a single-class cell; we report it unscoreable
                assert cell["auroc"] is None
                continue
            assert cell["n_at_risk"] == metric.n_at_risk, (event, h)
            assert cell["n_positive"] == metric.n_positive, (event, h)
            assert cell["n_censored"] == metric.n_censored, (event, h)
            assert cell["auroc"] == pytest.approx(metric.auroc, abs=1e-6), (event, h)


# ---------------------------------------------------------------------------
# (iii) paired bootstrap: interval covers the point, collapses when identical
# ---------------------------------------------------------------------------


def test_paired_interval_covers_the_point_and_is_zero_width_when_identical(
    fx: _Fixture,
) -> None:
    _, none_table = fx.run("none", hazard=True)
    _, truth_table = fx.run("truth", hazard=True)
    assert none_table is not None and truth_table is not None
    labels = landmark_labels(none_table, fx.times)

    paired = hazard_paired_summary(truth_table, none_table, labels, n_boot=200, seed=0)
    scored = 0
    for event in EVENT_NAMES:
        for h in HORIZONS_HOURS:
            cell = paired[event][f"{h:g}h"]
            mean = cell["mean_risk"]
            if mean is None:
                assert cell["n_at_risk"] == 0, (event, h)
                continue
            assert mean["ci_low"] <= mean["point"] <= mean["ci_high"], (event, h)
            if cell["auroc"] is not None and cell["auroc"]["ci_low"] is not None:
                auc = cell["auroc"]
                assert auc["ci_low"] <= auc["point"] <= auc["ci_high"], (event, h)
                scored += 1
    assert scored > 0, "no two-class cell to check the AUROC interval on"

    same = hazard_paired_summary(none_table, none_table, labels, n_boot=200, seed=0)
    for event in EVENT_NAMES:
        for h in HORIZONS_HOURS:
            cell = same[event][f"{h:g}h"]
            if cell["mean_risk"] is None:
                continue
            assert cell["mean_risk"] == {
                "point": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "separated": False,
            }
            if cell["auroc"] is not None and cell["auroc"]["ci_low"] is not None:
                assert cell["auroc"]["point"] == 0.0
                assert cell["auroc"]["ci_low"] == 0.0 == cell["auroc"]["ci_high"]
                assert cell["auroc"]["separated"] is False


# ---------------------------------------------------------------------------
# (iv) resampling moves whole subjects
# ---------------------------------------------------------------------------


def test_bootstrap_resamples_whole_subjects() -> None:
    # Subject 1 has four rows of 1.0, subject 2 two rows of 10.0. Drawing
    # two subjects with replacement gives {1,1}, {1,2}, {2,2}: means of
    # 1, 4 (pooled: 24/6) or 10. Row resampling would produce other values.
    diff = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0])
    subjects = np.array([1, 1, 1, 1, 2, 2])
    boots = subject_bootstrap_means(diff, subjects, n_boot=300, seed=0, block=7)
    seen = set(np.round(boots, 12).tolist())
    assert seen <= {1.0, 4.0, 10.0}
    assert seen == {1.0, 4.0, 10.0}

    delta = paired_mean_delta(diff, np.zeros_like(diff), subjects, n_boot=300, seed=0)
    assert delta.point == pytest.approx(4.0)
    assert delta.n_rows == 6 and delta.n_subjects == 2
    assert delta.ci_low == 1.0 and delta.ci_high == 10.0
    assert delta.separated


def test_subject_bootstrap_is_deterministic_per_seed() -> None:
    diff = np.arange(20, dtype=float)
    subjects = np.repeat(np.arange(5), 4)
    a = subject_bootstrap_means(diff, subjects, n_boot=50, seed=3)
    b = subject_bootstrap_means(diff, subjects, n_boot=50, seed=3)
    c = subject_bootstrap_means(diff, subjects, n_boot=50, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# orchestration: refusals and the attached blocks
# ---------------------------------------------------------------------------


def test_hazard_scoring_refuses_a_run_without_heads_before_reading_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(10, with_heads=False)
    config = TrainingConfig(
        train_shard_dir="/train", tuning_shard_dir="/tuning", output_dir="/out"
    )
    monkeypatch.setattr(
        interventions_module, "load_run", lambda *a, **k: (model, None, None, config)
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not read shards before the hazard gate fires")

    monkeypatch.setattr(interventions_module, "load_meds_shards", _boom)
    with pytest.raises(ValueError, match="hazard heads"):
        evaluate_interventions("/runs/x", "/data/held_out", hazard_heads=True)


def test_hazard_scoring_refuses_a_baseline_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BaselineSequenceModel(
        TinyGRUBackbone(vocab_size=10, hidden_size=4), vocab_size=10
    )
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        model_kind="baseline",
    )
    monkeypatch.setattr(
        interventions_module, "load_run", lambda *a, **k: (model, None, None, config)
    )
    with pytest.raises(ValueError, match="needs a concept bottleneck"):
        evaluate_interventions("/runs/x", "/data/held_out", hazard_heads=True)


def test_evaluate_interventions_attaches_hazard_and_paired_blocks(
    fx: _Fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        source="mimic_iv",
        task_set="v1",
        concept_supervision="stay",
    )
    monkeypatch.setattr(
        interventions_module,
        "load_run",
        lambda *a, **k: (fx.model, fx.vocab, None, config),
    )
    monkeypatch.setattr(
        interventions_module, "load_meds_shards", lambda *a, **k: fx.raw.clone()
    )
    held_out = tmp_path / "held_out"
    held_out.mkdir()
    results = evaluate_interventions(
        tmp_path,
        held_out,
        modes=["none", "truth", "flip"],
        num_lanes=2,
        chunk_size=16,
        device="cpu",
        hazard_heads=True,
        hazard_boot=20,
    )
    by_mode = {r.mode: r for r in results}
    assert set(by_mode) == {"none", "truth", "flip"}
    for r in results:
        assert r.hazard is not None and set(r.hazard) == set(EVENT_NAMES)
        for event in EVENT_NAMES:
            assert set(r.hazard[event]) == {f"{h:g}h" for h in HORIZONS_HOURS}
    assert by_mode["none"].hazard_paired == {}
    assert set(by_mode["truth"].hazard_paired or {}) == {
        "truth_minus_none",
        "truth_minus_flip",
    }
    assert set(by_mode["flip"].hazard_paired or {}) == {"flip_minus_none"}
    cell = (by_mode["truth"].hazard_paired or {})["truth_minus_none"]
    vaso = cell["vasopressor_start"]["8h"]
    assert (
        vaso["n_at_risk"]
        == by_mode["truth"].hazard["vasopressor_start"]["8h"]["n_at_risk"]
    )
    assert vaso["mean_risk"]["point"] == pytest.approx(
        by_mode["truth"].hazard["vasopressor_start"]["8h"]["mean_risk"]
        - by_mode["none"].hazard["vasopressor_start"]["8h"]["mean_risk"],
        abs=1e-6,
    )
    # The whole thing round-trips through the CLI serialiser.
    dumped = json.loads(json.dumps([result_to_json(r) for r in results]))
    assert list(dumped[0]) == [*OLD_KEYS, "hazard", "hazard_paired"]

    # Without the flag on the same fixture: no hazard keys, same numbers.
    plain = evaluate_interventions(
        tmp_path,
        held_out,
        modes=["none"],
        num_lanes=2,
        chunk_size=16,
        device="cpu",
    )
    assert list(result_to_json(plain[0])) == OLD_KEYS
    assert plain[0].top1_accuracy == by_mode["none"].top1_accuracy
    assert plain[0].mean_task_loss == by_mode["none"].mean_task_loss
