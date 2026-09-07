"""End to end on CPU: probes fit from a saved run, then the steering CLI reads them.

A tiny decomposed run is trained on synthetic shards that carry visits,
ICU admission and discharge, hospital discharge and a vasopressor start
and stop, so every state-transition probe has positives; then
``odyssey.inference.outcome_probes`` fits and saves them and
``odyssey.inference.steering --outcome-probes`` scores a dial against
them with readout shifts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch

from odyssey.inference import outcome_probes as probes_cli
from odyssey.inference import steering as steering_cli
from odyssey.inference.outcome_probes import OutcomeProbes
from odyssey.inference.specificity import score_run, totals
from odyssey.training.train import TrainingConfig, train


T0 = datetime(2024, 1, 1)
N_EVENTS = 100


def _write_shards(shard_dir: Path, n_subjects: int) -> None:
    """One visit per subject with the transitions the probes need.

    Hourly heart rates (tachycardic on even subjects, bradycardic on odd)
    over ~100 h, ICU admission at 2 h, norepinephrine start at 4 h and a
    stop without an admission id (as MIMIC-IV records it), ICU discharge
    and hospital discharge. Even subjects transition early (stop 9 h, ICU
    discharge 12 h, home alive at 30 h); odd subjects late (stop 90 h, ICU
    discharge 80 h) and die at 96 h with a DIED discharge code, so every
    probe horizon sees both classes.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, str, datetime, float | None, int | None]] = []
    for sid in range(n_subjects):
        hadm = 1000 + sid
        base = T0 + timedelta(days=sid)
        for i in range(N_EVENTS):
            t = base + timedelta(hours=i)
            rate = 125.0 if sid % 2 == 0 else 45.0
            rows.append((sid, "LAB//220045//bpm", t, rate, hadm))
        early = sid % 2 == 0
        rows.append((sid, "ICU_ADMISSION//MICU", base + timedelta(hours=2), None, hadm))
        rows.append(
            (
                sid,
                "MEDICATION//START//norepinephrine",
                base + timedelta(hours=4),
                None,
                hadm,
            )
        )
        rows.append(
            (
                sid,
                "MEDICATION//STOP//norepinephrine",
                base + timedelta(hours=9 if early else 90),
                None,
                None,
            )
        )
        rows.append(
            (
                sid,
                "ICU_DISCHARGE//MICU",
                base + timedelta(hours=12 if early else 80),
                None,
                hadm,
            )
        )
        end = "HOSPITAL_DISCHARGE//HOME" if early else "HOSPITAL_DISCHARGE//DIED"
        rows.append((sid, end, base + timedelta(hours=30 if early else 96), None, hadm))
    frame = pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime,
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    ).sort("subject_id", "time")
    frame.write_parquet(shard_dir / "0.parquet")


def _config(train_dir: Path, tuning_dir: Path, output_dir: Path) -> TrainingConfig:
    values: dict[str, Any] = {
        "train_shard_dir": str(train_dir),
        "tuning_shard_dir": str(tuning_dir),
        "output_dir": str(output_dir),
        "backbone": "transformer",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "attn_num_heads": 4,
        "embedding_dim": 8,
        "vocab_min_count": 1,
        "quantile_min_count": 1,
        "num_lanes": 2,
        "chunk_size": 16,
        "max_context": 64,
        "num_epochs": 1,
        "log_every": 2,
        "eval_every": 1000,
        "eval_max_chunks": 1,
        "checkpoint_every": 1000,
        "model_kind": "bottleneck",
        "concept_supervision": "stay",
        "bottleneck_kind": "decomposed",
        "unknown_ratio": 2,
        "residual_dropout": 0.3,
        "event_hazards": True,
        "teacher_known_start": 1.0,
        "teacher_known_end": 0.5,
        "teacher_unknown_start": 1.0,
        "teacher_unknown_end": 0.5,
        "teacher_anneal_steps": 4,
        "orthogonality_weight": 0.0,
        "stream_shards": True,
        "lifted_patients": 6,
        "lifted_min_count": 1,
        "lifted_min_share": 0.0,
        "lifted_min_lift": 1.0,
    }
    return TrainingConfig(**values)


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("two_sided")
    train_dir, tuning_dir = root / "data" / "train", root / "data" / "tuning"
    _write_shards(train_dir, n_subjects=12)
    _write_shards(tuning_dir, n_subjects=6)
    output_dir = root / "run"
    assert train(_config(train_dir, tuning_dir, output_dir)) == output_dir
    return {"train": train_dir, "tuning": tuning_dir, "run": output_dir}


@pytest.fixture(scope="module")
def fitted_probes(saved_run: dict[str, Path]) -> Path:
    out = saved_run["run"] / "outcome_probes.pt"
    probes_cli.main(
        [
            "--run-dir",
            str(saved_run["run"]),
            "--train-shard-dir",
            str(saved_run["train"]),
            "--held-out-shard-dir",
            str(saved_run["tuning"]),
            "--max-train-shards",
            "1",
            "--max-held-out-shards",
            "1",
            "--num-lanes",
            "2",
            "--chunk-size",
            "16",
            "--output",
            str(out),
        ]
    )
    return out


def test_probe_cli_fits_every_transition_with_positives(fitted_probes: Path) -> None:
    probes = OutcomeProbes.load(fitted_probes)
    assert probes.event_names == [
        "icu_discharge",
        "hospital_discharge_alive",
        "vasopressor_stop",
    ]
    assert probes.requires == {
        "icu_discharge": "icu_admission",
        "vasopressor_stop": "vasopressor_start",
    }
    for event in probes.event_names:
        stats = probes.auroc[event]
        # every horizon was fit on rows with both classes present
        assert stats["positives_train@24h"] > 0
        assert 0.0 <= stats["train@24h"] <= 1.0
    # the vasopressor stop is only eligible after the start at 4 h and before
    # the stop, so it sees fewer rows than hospital discharge, which needs no
    # prior event
    assert (
        probes.auroc["vasopressor_stop"]["n_train@24h"]
        < probes.auroc["hospital_discharge_alive"]["n_train@24h"]
    )
    risk = probes.risk(torch.zeros(3, probes.weight.shape[-1]))
    assert risk.shape == (3, 3, 3)


def test_steering_cli_with_probes_reports_two_sided_outcomes_and_readout_shifts(
    saved_run: dict[str, Path],
    fitted_probes: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "steering_twosided.json"
    argv = [
        "steering",
        "--run-dir",
        str(saved_run["run"]),
        "--held-out-shard-dir",
        str(saved_run["tuning"]),
        "--lift-shard-dir",
        str(saved_run["train"]),
        "--output-json",
        str(out),
        "--concepts",
        "tachycardia",
        "--site",
        "stream",
        "--layer-index",
        "0",
        "--num-lanes",
        "2",
        "--chunk-size",
        "16",
        "--n-boot",
        "10",
        "--min-share",
        "0",
        "--min-lift",
        "1",
        "--lift-shards",
        "1",
        "--outcome-probes",
        str(fitted_probes),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    steering_cli._main()
    payload = json.loads(out.read_text())
    probe_events = ["icu_discharge", "hospital_discharge_alive", "vasopressor_stop"]
    assert payload["event_names"] == payload["trained_event_names"] + probe_events
    assert payload["outcome_probes"]["event_names"] == probe_events
    assert set(payload["outcome_probes"]["auroc"]) == set(probe_events)
    up, down = payload["summaries"]
    assert (up["direction"], down["direction"]) == ("amplify", "suppress")
    # the transition outcomes carry their declared sign: more tachycardia,
    # less chance of leaving the ICU or going home
    by_event = {(o["event"], o["horizon_hours"]): o for o in up["outcomes"]}
    assert by_event[("icu_discharge", 24.0)]["expected_sign"] == -1
    assert by_event[("hospital_discharge_alive", 24.0)]["expected_sign"] == -1
    assert by_event[("vasopressor_stop", 24.0)]["expected_sign"] is None  # undeclared
    assert by_event[("icu_discharge", 24.0)]["baseline_risk"] == pytest.approx(
        by_event[("icu_discharge", 24.0)]["baseline_risk"]
    )
    # every registry concept's readout shift is in the JSON, the pushed one
    # expected up under amplify and down under suppress
    shifts_up = {s["concept"]: s for s in up["concept_shifts"]}
    shifts_down = {s["concept"]: s for s in down["concept_shifts"]}
    assert set(shifts_up) == set(payload["gammas"])
    assert shifts_up["tachycardia"]["expected_sign"] == 1
    assert shifts_down["tachycardia"]["expected_sign"] == -1
    assert shifts_up["bradycardia"]["expected_sign"] == -1  # declared opposite
    assert shifts_up["hypotension"]["expected_sign"] is None
    # and the specificity scorer reads the file as written
    scored = score_run(payload["summaries"])
    t = totals(scored)
    assert t.n_dials == 2
    assert t.opposite_declared == 2
    assert t.good_declared == 4  # ICU discharge and discharge alive, both directions
