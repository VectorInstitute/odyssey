"""A checkpoint's own concept set wins over today's registry."""

from __future__ import annotations

import pytest
import torch

from odyssey.inference.legacy_concept_pins import (
    LEGACY_CONCEPT_PINS,
    check_concept_count,
    checkpoint_num_concepts,
    pinned_concept_names,
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
