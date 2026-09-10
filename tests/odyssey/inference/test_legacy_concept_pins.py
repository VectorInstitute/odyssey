"""A checkpoint's own concept set wins over today's registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from odyssey.data.concepts import concepts_for_source
from odyssey.inference.legacy_concept_pins import (
    LEGACY_CONCEPT_PINS,
    check_concept_count,
    checkpoint_num_concepts,
    pinned_concept_names,
    resolve_concepts_for_run,
)


def _state(n_slots: int, *, separate_unknown_head: bool = False) -> dict[str, object]:
    state: dict[str, object] = {"bottleneck.prob_weight": torch.zeros(n_slots, 64)}
    if separate_unknown_head:
        state["bottleneck.unknown_prob_weight"] = torch.zeros(1, 64)
    return state


def test_pin_lookup_ignores_path_and_trailing_slash() -> None:
    want = LEGACY_CONCEPT_PINS["gemini_full_v10"]
    assert pinned_concept_names("gemini_full_v10") == want
    assert pinned_concept_names("/mnt/nfs/home/x/runs/gemini_full_v10") == want
    assert pinned_concept_names("/mnt/nfs/home/x/runs/gemini_full_v10/") == want
    assert pinned_concept_names("runs/never_trained") is None


def test_a_renamed_run_directory_keeps_its_pin() -> None:
    # Pins key on the directory name, so moving a run aside (as happens when
    # a retrain is attempted against the original path) would otherwise drop
    # the pin and make the checkpoint refuse to load.
    assert pinned_concept_names("gemini_full_v10_15c") == pinned_concept_names(
        "gemini_full_v10"
    )


def test_pinned_lists_match_the_widths_they_were_recovered_from() -> None:
    # The pins exist to match a frozen bottleneck width. GEMINI trained with
    # 15 concepts and eICU-CRD with 26; if either list is edited to a
    # different length it no longer describes the checkpoint it names.
    assert len(LEGACY_CONCEPT_PINS["gemini_full_v10"]) == 15
    assert len(LEGACY_CONCEPT_PINS["eicu_full_v10"]) == 26


def test_pinned_lists_have_no_duplicates() -> None:
    for name, names in LEGACY_CONCEPT_PINS.items():
        assert len(set(names)) == len(names), f"{name} repeats a concept"


def test_checkpoint_num_concepts_discounts_the_shared_unknown_row() -> None:
    # With no head of its own, the unknown slot rides in the shared weight,
    # so 16 slots is 15 named concepts. With its own head, 15 rows is 15.
    assert checkpoint_num_concepts(_state(16)) == 15
    assert checkpoint_num_concepts(_state(15, separate_unknown_head=True)) == 15
    assert checkpoint_num_concepts({}) is None


def test_check_passes_when_the_counts_agree() -> None:
    check_concept_count("runs/x", _state(16), ["c"] * 15)


def test_check_refuses_with_both_counts_named() -> None:
    # The whole point: one readable sentence instead of a page of shape
    # errors from load_state_dict.
    with pytest.raises(ValueError, match="trained with 15 concepts"):
        check_concept_count("runs/gemini_full_v10", _state(16), ["c"] * 25)
    with pytest.raises(ValueError, match="resolves 25"):
        check_concept_count("runs/gemini_full_v10", _state(16), ["c"] * 25)


def test_check_is_silent_for_a_model_with_no_bottleneck() -> None:
    check_concept_count("runs/baseline", {}, ["c"] * 25)


def test_resolve_concepts_returns_the_runs_own_set_in_slot_order() -> None:
    # The GEMINI checkpoint trained with 15 of the 25 its source resolves
    # today. A caller that rebuilt the list from the registry would hand a
    # 25-long list to a 15-slot bottleneck.
    got = resolve_concepts_for_run("runs/gemini_full_v10", "gemini", "v3")
    assert [c.name for c in got] == list(LEGACY_CONCEPT_PINS["gemini_full_v10"])


def test_resolve_concepts_follows_a_rename() -> None:
    # eICU's pin records "shock", which the registry now calls
    # sustained_hypotension_map; the pin must still resolve.
    got = resolve_concepts_for_run("runs/eicu_full_v10", "eicu", "v3")
    assert len(got) == len(LEGACY_CONCEPT_PINS["eicu_full_v10"])
    assert "sustained_hypotension_map" in {c.name for c in got}


def test_resolve_concepts_is_the_full_registry_when_unpinned() -> None:
    got = resolve_concepts_for_run("runs/not_pinned", "mimic_iv", "v3")
    assert len(got) == len(concepts_for_source("mimic_iv", task_set="v3"))


def test_no_checkpoint_caller_pairs_the_registry_with_a_loaded_model() -> None:
    """A module that loads a checkpoint must not build concepts from the registry.

    The bug this guards has now shipped three times (``interventions``,
    ``eval-forecast``, and the callers fixed alongside them): a run trained
    with N bottleneck slots is scored against a registry that has since
    grown to M > N concepts, and the two get zipped or indexed together.
    The failure is an ``IndexError`` deep in scoring, or -- worse -- a
    silent mislabelling when M < N.

    Any module that calls ``load_run`` must therefore go through
    :func:`resolve_concepts_for_run`, which reads the run's own pinned
    concept list. Legitimate exceptions are listed explicitly.
    """
    import ast  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[3]
    # run_inference defines load_run; its own two uses are the pin lookup
    # inside load_run and a documented default on a helper that every real
    # caller passes concepts to explicitly.
    allowed = {
        # Defines load_run. Its own two uses are the pin lookup inside
        # load_run and a documented default on a helper that every real
        # caller passes concepts to explicitly.
        "odyssey/inference/run_inference.py",
        # The resolver itself: this is where the registry is looked up.
        "odyssey/inference/legacy_concept_pins.py",
        # Training DEFINES a run's concept set rather than reading one
        # back, so there is nothing to pin against.
        "odyssey/training/train.py",
    }

    offenders: list[str] = []
    for path in sorted(repo_root.glob("odyssey/**/*.py")) + sorted(
        repo_root.glob("scripts/**/*.py")
    ):
        rel = path.relative_to(repo_root).as_posix()
        if rel in allowed:
            continue
        source = path.read_text()
        if "load_run" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "concepts_for_source"
            ):
                offenders.append(f"{rel}:{node.lineno}")

    assert offenders == [], (
        "these modules load a checkpoint but build concepts from the live "
        "registry; use resolve_concepts_for_run(run_dir, source, task_set) "
        f"instead: {offenders}"
    )
