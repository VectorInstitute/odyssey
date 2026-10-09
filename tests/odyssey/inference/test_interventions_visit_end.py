"""Label override applied only at visit ends (CPU, tiny fixtures).

The concept head is supervised at each visit's last event; the default
override edits every position. ``override_positions="visit_end"`` edits
only the supervised positions and scores them on their own.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

import odyssey.inference.interventions as interventions_module
from odyssey.data.streaming import PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.interventions import (
    OVERRIDE_POSITIONS,
    InterventionResult,
    VisitEndScorer,
    _chunk_intervention,
    evaluate_interventions,
    result_to_json,
    run_streaming_intervention,
    score_visit_end_rows,
    visit_end_paired_summary,
    visit_end_positions,
)
from odyssey.models.concept_bottleneck import intervention_apply_mask
from odyssey.training.data import iter_patient_sequences
from odyssey.training.running_labels import position_running_labels
from odyssey.training.train import TrainingConfig
from tests.odyssey.inference.test_interventions import (
    CODES,
    NUM_CONCEPTS,
    T0,
    _model,
    _vocab,
)
from tests.odyssey.inference.test_interventions_hazard import OLD_KEYS
from tests.odyssey.inference.test_interventions_hazard import _Fixture as _HazardFixture
from tests.odyssey.inference.test_interventions_hazard import _model as _hazard_model


SUBJECTS = (1, 2, 3)
EVENTS_PER_VISIT = 10


def _events() -> pl.DataFrame:
    """Three subjects, two hourly visits each (hadm 10*sid + {1, 2})."""
    rows = []
    for sid in SUBJECTS:
        for visit in (1, 2):
            for i in range(EVENTS_PER_VISIT):
                hour = (visit - 1) * EVENTS_PER_VISIT + i
                rows.append(
                    (
                        sid,
                        CODES[(sid * 3 + hour) % len(CODES)],
                        T0 + timedelta(hours=hour),
                        None,
                        10 * sid + visit,
                    )
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


def _visit_labels() -> tuple[
    dict[tuple[int, int], torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
]:
    """Visit-scoped labels, masks and first times; every visit observed."""
    labels: dict[tuple[int, int], torch.Tensor] = {}
    masks: dict[tuple[int, int], torch.Tensor] = {}
    first: dict[tuple[int, int], torch.Tensor] = {}
    for sid in SUBJECTS:
        for visit in (1, 2):
            key = (sid, 10 * sid + visit)
            labels[key] = torch.tensor([float(visit == 1), 1.0, float(sid == 2)])
            masks[key] = torch.ones(NUM_CONCEPTS)
            first[key] = torch.where(
                labels[key] > 0, torch.tensor(0.0), torch.tensor(float("inf"))
            )
    return labels, masks, first


def _run(
    mode: str,
    *,
    override_positions: str | None = None,
    uncertain_band: float | None = None,
    scorer: VisitEndScorer | None = None,
) -> InterventionResult:
    vocab = _vocab()
    labels, masks, first = _visit_labels()
    kwargs = (
        {} if override_positions is None else {"override_positions": override_positions}
    )
    return run_streaming_intervention(
        _model(len(vocab)),
        _events(),
        vocab,
        labels,
        masks,
        mode=mode,
        concept_first_times=first,
        supervision="visit",
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        seed=0,
        uncertain_band=uncertain_band,
        visit_end_scorer=scorer,
        **kwargs,
    )


def _lane_chunks() -> list[StreamingChunk]:
    """Stream the whole fixture in one lane: one chunk per 20-event patient."""
    seqs = iter_patient_sequences(_events(), _vocab())
    sampler = PackedLaneSampler(seqs, num_lanes=1, chunk_size=20, reset_prob=0.0)
    return list(sampler)


# ---------------------------------------------------------------------------
# (i) the default reproduces the old output exactly
# ---------------------------------------------------------------------------


def test_default_override_positions_is_all_and_keeps_the_old_schema() -> None:
    for mode in ("none", "truth", "flip", "random"):
        default = _run(mode)
        explicit = _run(mode, override_positions="all")
        assert default.visit_end_positions is None
        assert json.dumps(result_to_json(default)) == json.dumps(
            result_to_json(explicit)
        )
        assert list(result_to_json(default)) == OLD_KEYS
    # The override edits every observed position by default: 2 visits of
    # 10 events per subject, all observed.
    assert _run("truth").n_intervened_positions == len(SUBJECTS) * 2 * EVENTS_PER_VISIT


def test_none_is_the_same_pass_under_both_settings() -> None:
    plain = _run("none")
    restricted = _run("none", override_positions="visit_end")
    assert restricted.n_predictions == plain.n_predictions
    assert restricted.top1_accuracy == plain.top1_accuracy
    assert restricted.mean_task_loss == plain.mean_task_loss
    assert restricted.n_intervened_positions == 0
    assert restricted.visit_end_positions is not None


# ---------------------------------------------------------------------------
# (ii) positions that are not visit ends receive no intervention
# ---------------------------------------------------------------------------


def test_visit_end_positions_are_each_visits_last_event() -> None:
    chunks = _lane_chunks()
    # One end per (subject, visit): six in all.
    assert sum(int(visit_end_positions(c).sum()) for c in chunks) == len(SUBJECTS) * 2
    for chunk in chunks:
        ends = visit_end_positions(chunk)[0]
        sids = chunk.subject_ids[0]
        vids = chunk.visit_ids[0]
        for i in torch.nonzero(ends).flatten().tolist():
            later = (vids[i + 1 :] == vids[i]) & (sids[i + 1 :] == sids[i])
            assert not later.any()


def test_only_visit_ends_are_intervened_and_other_logits_match_none() -> None:
    labels, masks, first = _visit_labels()
    model = _model(len(_vocab())).eval()
    state = None
    n_touched = 0
    for chunk in _lane_chunks():
        position_mask = visit_end_positions(chunk)
        intervention = _chunk_intervention(
            chunk,
            "truth",
            labels,
            masks,
            first,
            supervision="visit",
            num_concepts=NUM_CONCEPTS,
            device="cpu",
            rng=torch.Generator().manual_seed(0),
            position_mask=position_mask,
        )
        assert intervention is not None
        with torch.no_grad():
            base_logits, out, _ = model(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            edited_logits, _, state = model(
                chunk.batch,
                state=state,
                reset_mask=chunk.reset_mask,
                intervention=intervention,
            )
        applied = intervention_apply_mask(intervention, out.concept_probs)
        assert applied is not None
        touched = applied.any(dim=-1)
        # The intervention mask is exactly the visit-end mask (all observed).
        assert torch.equal(touched, position_mask)
        # Off the visit ends the forward is byte for byte the un-edited one.
        assert torch.equal(edited_logits[~touched], base_logits[~touched])
        # On them, feeding hard 0/1 values moves the logits.
        assert not torch.equal(edited_logits[touched], base_logits[touched])
        n_touched += int(touched.sum())
    assert n_touched == len(SUBJECTS) * 2


def test_without_a_position_mask_the_chunk_intervention_is_unchanged() -> None:
    chunk = _lane_chunks()[0]
    labels, masks, first = _visit_labels()
    plain = _chunk_intervention(
        chunk,
        "flip",
        labels,
        masks,
        first,
        supervision="visit",
        num_concepts=NUM_CONCEPTS,
        device="cpu",
        rng=torch.Generator().manual_seed(0),
    )
    assert plain is not None and plain.probs_mask is not None
    real = chunk.subject_ids != -1
    assert torch.equal(plain.probs_mask.any(dim=-1), real)


def test_random_and_calibrated_modes_respect_the_restriction() -> None:
    n_ends = len(SUBJECTS) * 2
    assert (
        _run("random", override_positions="visit_end").n_intervened_positions == n_ends
    )
    assert _run("flip", override_positions="visit_end").n_intervened_positions == n_ends
    vocab = _vocab()
    labels, masks, first = _visit_labels()
    calibrated = run_streaming_intervention(
        _model(len(vocab)),
        _events(),
        vocab,
        labels,
        masks,
        mode="truth_calibrated",
        concept_first_times=first,
        supervision="visit",
        num_lanes=2,
        chunk_size=8,
        device="cpu",
        calibration_gammas=torch.full((NUM_CONCEPTS,), 0.25),
        override_positions="visit_end",
    )
    assert calibrated.n_intervened_positions == n_ends


# ---------------------------------------------------------------------------
# (iii) the visit_end_positions block
# ---------------------------------------------------------------------------


def test_block_counts_visits_with_an_observed_label_and_a_target() -> None:
    truth = _run("truth", override_positions="visit_end")
    block = truth.visit_end_positions
    assert block is not None
    # Every visit end is overridden (six), but only the first visit of
    # each subject has a next-token target: the second visit's last
    # event is followed by another patient (or nothing).
    assert truth.n_intervened_positions == len(SUBJECTS) * 2
    assert block["n_positions"] == len(SUBJECTS)
    assert block["n_subjects"] == len(SUBJECTS)
    assert 0.0 <= block["top1_accuracy"] <= 1.0
    assert block["mean_task_loss"] > 0.0
    assert block["paired"] == {}
    assert list(result_to_json(truth)) == [*OLD_KEYS, "visit_end_positions"]


def test_block_rows_follow_the_band_and_match_a_single_chunk_pass() -> None:
    band = 0.05
    labels, masks, first = _visit_labels()
    model = _model(len(_vocab())).eval()
    state = None
    expected_rows = 0
    expected_overridden = 0
    for chunk in _lane_chunks():
        with torch.no_grad():
            _, out, state = model(chunk.batch, state=state, reset_mask=chunk.reset_mask)
        _, observed = position_running_labels(
            chunk, labels, masks, first, supervision="visit", num_concepts=NUM_CONCEPTS
        )
        in_band = (out.concept_probs - 0.5).abs() < band
        overridden = (observed.bool() & in_band).any(dim=-1) & visit_end_positions(
            chunk
        )
        expected_overridden += int(overridden.sum())
        expected_rows += int((overridden & chunk.real_mask).sum())
    # A band this narrow drops some visits, so the subset is a real subset.
    assert 0 < expected_rows < len(SUBJECTS) * 2
    result = _run("truth", override_positions="visit_end", uncertain_band=band)
    block = result.visit_end_positions
    assert block is not None
    assert block["n_positions"] == expected_rows
    assert result.n_intervened_positions == expected_overridden


def test_none_versus_none_gives_zero_deltas() -> None:
    scorer_a = VisitEndScorer(uncertain_band=None)
    scorer_b = VisitEndScorer(uncertain_band=None)
    _run("none", override_positions="visit_end", scorer=scorer_a)
    _run("none", override_positions="visit_end", scorer=scorer_b)
    left, right = scorer_a.table(), scorer_b.table()
    assert left.same_rows(right) and left.n_rows == len(SUBJECTS)
    paired = visit_end_paired_summary(left, right, n_boot=50, seed=0)
    for metric in ("top1_accuracy", "mean_task_loss"):
        assert paired[metric] == {
            "point": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "separated": False,
        }
    assert paired["n_positions"] == len(SUBJECTS)
    assert paired["n_subjects"] == len(SUBJECTS)


def test_paired_blocks_land_on_the_left_mode() -> None:
    tables = {}
    for mode in ("none", "truth", "flip"):
        scorer = VisitEndScorer(uncertain_band=None)
        _run(mode, override_positions="visit_end", scorer=scorer)
        tables[mode] = scorer.table()
    blocks = score_visit_end_rows(tables, n_boot=50, seed=0)
    assert set(blocks["truth"]["paired"]) == {"truth_minus_none", "truth_minus_flip"}
    assert set(blocks["flip"]["paired"]) == {"flip_minus_none"}
    assert blocks["none"]["paired"] == {}
    delta = blocks["truth"]["paired"]["truth_minus_none"]["top1_accuracy"]
    assert delta["point"] == pytest.approx(
        blocks["truth"]["top1_accuracy"] - blocks["none"]["top1_accuracy"]
    )
    assert delta["ci_low"] <= delta["point"] <= delta["ci_high"]
    # The rows are mode independent, so the per-mode tables line up.
    assert np.array_equal(tables["truth"].subject_ids, tables["flip"].subject_ids)


def test_mismatched_rows_are_refused() -> None:
    scorer = VisitEndScorer(uncertain_band=None)
    _run("none", override_positions="visit_end", scorer=scorer)
    full = scorer.table()
    half = interventions_module.VisitEndRowTable(
        subject_ids=full.subject_ids[:1],
        visit_ids=full.visit_ids[:1],
        time_hours=full.time_hours[:1],
        hits=full.hits[:1],
        losses=full.losses[:1],
    )
    with pytest.raises(ValueError, match="identical rows"):
        visit_end_paired_summary(full, half, n_boot=5, seed=0)
    with pytest.raises(ValueError, match="different visit-end rows"):
        score_visit_end_rows({"none": full, "truth": half})


def test_unknown_override_positions_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown override_positions"):
        _run("none", override_positions="landmark")


# ---------------------------------------------------------------------------
# (iv) CLI parsing
# ---------------------------------------------------------------------------


def test_cli_passes_override_positions_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}
    block = {
        "n_positions": 3,
        "n_subjects": 3,
        "top1_accuracy": 0.5,
        "mean_task_loss": 1.0,
        "paired": {},
    }

    def fake_evaluate(*_args: object, **kwargs: object) -> list[InterventionResult]:
        seen.update(kwargs)
        if kwargs["override_positions"] == "visit_end":
            return [InterventionResult("none", 1, 1.0, 0.5, visit_end_positions=block)]
        return [InterventionResult("none", 1, 1.0, 0.5)]

    monkeypatch.setattr(interventions_module, "evaluate_interventions", fake_evaluate)
    argv = [
        "prog",
        "--run-dir",
        "/runs/x",
        "--held-out-shard-dir",
        "/data/held_out",
        "--output-json",
    ]
    default = tmp_path / "default.json"
    monkeypatch.setattr("sys.argv", [*argv, str(default)])
    interventions_module._main()
    assert seen["override_positions"] == "all"
    assert list(json.loads(default.read_text())[0]) == OLD_KEYS

    ends = tmp_path / "ends.json"
    monkeypatch.setattr(
        "sys.argv", [*argv, str(ends), "--override-positions", "visit_end"]
    )
    interventions_module._main()
    assert seen["override_positions"] == "visit_end"
    written = json.loads(ends.read_text())
    assert list(written[0]) == [*OLD_KEYS, "visit_end_positions"]
    assert written[0]["visit_end_positions"] == block

    monkeypatch.setattr(
        "sys.argv", [*argv, str(tmp_path / "bad.json"), "--override-positions", "x"]
    )
    with pytest.raises(SystemExit):
        interventions_module._main()
    assert OVERRIDE_POSITIONS == ("all", "visit_end")


# ---------------------------------------------------------------------------
# --hazard-heads stays compatible
# ---------------------------------------------------------------------------


def test_hazard_heads_combine_with_the_visit_end_override() -> None:
    """Both blocks are filled; the landmark rows are the usual ones."""
    fx = _HazardFixture()
    hazard_only, landmark_only = fx.run("truth", hazard=True)
    scorer = fx.scorer()
    combined = run_streaming_intervention(
        fx.model,
        fx.binned,
        fx.vocab,
        fx.labels,
        fx.mask,
        mode="truth",
        concept_first_times=fx.first_times,
        supervision="stay",
        num_lanes=2,
        chunk_size=16,
        device="cpu",
        seed=0,
        hazard_scorer=scorer,
        override_positions="visit_end",
    )
    assert landmark_only is not None
    assert scorer.table().same_rows(landmark_only)
    assert combined.visit_end_positions is not None
    # Far fewer positions are overridden than under the default.
    assert 0 < combined.n_intervened_positions < hazard_only.n_intervened_positions


# ---------------------------------------------------------------------------
# the run-dir driver attaches the block, and the small guards
# ---------------------------------------------------------------------------


def test_score_visit_end_rows_of_nothing_is_empty() -> None:
    assert score_visit_end_rows({}) == {}


def test_check_override_positions_rejects_unknown_and_warns_on_stay(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="unknown override_positions"):
        interventions_module._check_override_positions("landmark", "visit")
    with caplog.at_level("WARNING"):
        interventions_module._check_override_positions("visit_end", "visit")
    assert not caplog.records
    with caplog.at_level("WARNING"):
        interventions_module._check_override_positions("visit_end", "stay")
    assert any("concept_supervision='stay'" in r.getMessage() for r in caplog.records)


def _two_visit_events(n_subjects: int = 16) -> pl.DataFrame:
    """Split the hazard fixture's hourly heart rates into two admissions.

    The hazard fixture has one admission per subject, so its only visit end
    is the last position of each stream, which has no next-token target.
    Splitting each record at hour 12 gives every subject one scorable visit
    end.
    """
    rows: list[tuple[int, str, datetime, float | None, int]] = []
    for sid in range(1, n_subjects + 1):
        for h in range(24):
            hadm = (1000 if h < 12 else 2000) + sid
            hr = 130.0 if sid % 2 == 0 and h >= 12 else 80.0
            rows.append((sid, "LAB//220045//bpm", T0 + timedelta(hours=h), hr, hadm))
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "code": pl.Utf8,
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "hadm_id": pl.Int64,
        },
        orient="row",
    )


def test_evaluate_interventions_attaches_visit_end_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _two_visit_events()
    vocab = Vocabulary.build(add_value_tokens(raw)["code"].to_list(), min_count=1)
    model = _hazard_model(len(vocab))
    config = TrainingConfig(
        train_shard_dir="/train",
        tuning_shard_dir="/tuning",
        output_dir="/out",
        source="mimic_iv",
        task_set="v1",
        concept_supervision="visit",
    )
    monkeypatch.setattr(
        interventions_module,
        "load_run",
        lambda *a, **k: (model, vocab, None, config),
    )
    monkeypatch.setattr(
        interventions_module, "load_meds_shards", lambda *a, **k: raw.clone()
    )
    held_out = tmp_path / "held_out"
    held_out.mkdir()
    with caplog.at_level("INFO"):
        results = evaluate_interventions(
            tmp_path,
            held_out,
            modes=["none", "truth", "flip"],
            num_lanes=2,
            chunk_size=16,
            device="cpu",
            override_positions="visit_end",
            hazard_boot=20,
        )
    by_mode = {r.mode: r for r in results}
    assert set(by_mode) == {"none", "truth", "flip"}
    assert all(r.visit_end_positions is not None for r in results)
    blocks: dict[str, dict[str, Any]] = {
        m: r.visit_end_positions or {} for m, r in by_mode.items()
    }
    n = blocks["none"]["n_positions"]
    assert n > 0
    assert all(b["n_positions"] == n for b in blocks.values())
    assert set(blocks["truth"]["paired"]) == {"truth_minus_none", "truth_minus_flip"}
    assert set(blocks["flip"]["paired"]) == {"flip_minus_none"}
    assert blocks["none"]["paired"] == {}
    assert all(r.hazard is None for r in results)
    messages = [r.getMessage() for r in caplog.records]
    assert any("paired visit-end bootstrap" in m for m in messages)
    assert any("none at visit ends: top1" in m for m in messages)
    # the JSON carries the block under the old keys
    written = result_to_json(by_mode["truth"])
    assert written["visit_end_positions"] == blocks["truth"]
