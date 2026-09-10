"""Concept lists for checkpoints trained before the registry grew.

``concepts_for_source`` answers with TODAY's registry. A checkpoint answers
with the registry it trained against, frozen in its bottleneck's parameter
shapes. Those two drift apart every time a source gains code mappings, and
when they do, :func:`odyssey.inference.run_inference.load_run` builds a model
of the wrong width and ``load_state_dict`` fails with a wall of shape errors
that names no cause.

That happened three times in one day (2026-09-09): eICU-CRD at 26 concepts
against a registry of 29, the same run again at 5 alert-event heads against
6, and GEMINI at 15 against 25. Each was worked around with a private pinned
list in a scratch script. This module is that workaround promoted to one
place, keyed by run directory name, so every consumer of ``load_run`` gets
the same answer and a new mismatch fails with a readable message instead of
archaeology.

Adding an entry: check out the run's training commit (its
``env_fingerprint.json`` records ``git_commit``) and print
``[c.name for c in concepts_for_source(source, task_set=...)]``. The ORDER is
load-bearing, the bottleneck's slots are positional, so paste the list as
printed rather than sorting it.
"""

from __future__ import annotations

from collections.abc import Sequence


# run directory basename -> the concept names that run trained with, in slot
# order. Verified against each checkpoint's own bottleneck width, not guessed.
LEGACY_CONCEPT_PINS: dict[str, tuple[str, ...]] = {
    # GEMINI, 15 of 29 resolved at training time; ten electrolyte and
    # haematology concepts plus hypoxemic_respiratory_failure, oliguria,
    # sepsis3 and shock gained GEMINI code mappings afterwards, taking
    # today's registry to 25. Recovered at training commit c1dadb9.
    "gemini_full_v10": (
        "tachycardia",
        "bradycardia",
        "hypotension",
        "hypertension",
        "hypoxia",
        "fever",
        "hypothermia",
        "elevated_lactate",
        "sustained_tachypnea",
        "acute_kidney_injury",
        "aki_stage_2",
        "aki_stage_3",
        "sirs",
        "qsofa",
        "on_vasopressors",
    ),
    # eICU-CRD, 26 of 29; microbiology mapping landed after this run, so
    # sepsis3 (and with it the sepsis3 alert head) resolves today but did
    # not then. "shock" is the pre-rename name of sustained_hypotension_map.
    "eicu_full_v10": (
        "tachycardia",
        "bradycardia",
        "hypotension",
        "hypertension",
        "hypoxia",
        "fever",
        "hypothermia",
        "elevated_lactate",
        "sustained_tachypnea",
        "acute_kidney_injury",
        "aki_stage_2",
        "aki_stage_3",
        "sirs",
        "qsofa",
        "on_vasopressors",
        "hyperkalemia",
        "hypokalemia",
        "hyponatremia",
        "hypernatremia",
        "hypoglycemia",
        "hyperglycemia",
        "anemia",
        "thrombocytopenia",
        "coagulopathy",
        "metabolic_acidosis",
        "shock",
    ),
}


def pinned_concept_names(run_dir: str) -> tuple[str, ...] | None:
    """Return the pinned concept list for ``run_dir``, or ``None`` if unpinned.

    Matches on the directory's final component, so an absolute path, a
    relative one and a trailing slash all resolve the same way.
    """
    name = str(run_dir).rstrip("/").rsplit("/", 1)[-1]
    return LEGACY_CONCEPT_PINS.get(name)


def checkpoint_num_concepts(state: dict[str, object]) -> int | None:
    """Return how many NAMED concepts a checkpoint's bottleneck carries.

    ``bottleneck.prob_weight``'s first dimension counts slots. When the
    unknown slot has no head of its own the shared weight includes its row,
    so it is subtracted here, mirroring how ``load_run`` recovers
    ``unknown_dim``. Returns ``None`` for a checkpoint with no bottleneck.
    """
    weight = state.get("bottleneck.prob_weight")
    if weight is None:
        return None
    n_slots = int(weight.shape[0])  # type: ignore[attr-defined]
    if "bottleneck.unknown_prob_weight" not in state:
        n_slots -= 1
    return n_slots


def check_concept_count(
    run_dir: str, state: dict[str, object], resolved: Sequence[str]
) -> None:
    """Raise if today's registry disagrees with the checkpoint's own width.

    Failing here, before ``load_state_dict``, turns a page of shape errors
    into one sentence naming both counts and where to fix it.
    """
    expected = checkpoint_num_concepts(state)
    if expected is None or expected == len(resolved):
        return
    name = str(run_dir).rstrip("/").rsplit("/", 1)[-1]
    raise ValueError(
        f"{name}: checkpoint was trained with {expected} concepts but this "
        f"code's registry resolves {len(resolved)} for its source. The "
        f"registry has changed since the run. Add {name}'s training-time "
        "concept list to odyssey/inference/legacy_concept_pins.py "
        "(its env_fingerprint.json records the training commit); loading it "
        "against today's registry would build a model of the wrong width."
    )
