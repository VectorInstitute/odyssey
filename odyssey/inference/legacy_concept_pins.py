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

Hazard EVENT heads drift the same way: ``alert_events_for`` drops a
concept-backed alert whose concept does not resolve for the source, so the
eICU sepsis3 head appeared the day sepsis3 became resolvable there (PR #222)
and every eICU checkpoint trained before it now has one head fewer than
today's registry builds. :data:`LEGACY_EVENT_PINS` and
:func:`pinned_event_names` are the event-side twins of the concept table.

Resolution order, for both concepts and events:

1. ``run_pins.json`` in the run directory (:data:`PIN_FILENAME`), which
   training writes since this module gained :func:`write_run_pins`, so any
   newer run is self-describing.
2. The legacy tables here, keyed by run directory name, for runs trained
   before the pin file existed.
3. Nothing: today's registry, with :func:`check_concept_count` and
   :func:`check_event_count` refusing before ``load_state_dict`` when the
   checkpoint's own widths disagree with it.

Adding a legacy entry: check out the run's training commit (its
``env_fingerprint.json`` records ``git_commit``; ``docs/experiments.md``
names it per run) and print
``[c.name for c in concepts_for_source(source, task_set=...)]`` and
``[a.name for a in hazard_events_for(task_set, source=source)]`` FROM THAT
CHECKOUT (a script run from another directory imports the editable install
instead, and reports today's list). The ORDER is load-bearing, the
bottleneck's slots and the hazard head's rows are positional, so paste the
list as printed rather than sorting it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from odyssey.data.concepts import canonical_concept_name, concepts_for_source
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


if TYPE_CHECKING:
    from odyssey.data.concepts import AnyConceptDefinition


#: Written into every run directory by training; read first by both
#: :func:`pinned_concept_names` and :func:`pinned_event_names`.
PIN_FILENAME = "run_pins.json"

#: Bins per hazard head: ``EventHazardHeads.num_bins`` is ``len(edges) + 2``
#: (a zero-gap bin, one per edge, and the open tail), so a checkpoint's
#: event count is its head's output width divided by this.
HAZARD_NUM_BINS = len(DEFAULT_TIME_BIN_EDGES_HOURS) + 2

# eICU-CRD before PR #222 (ca541d8, 2026-09-02): 26 of 29 concepts. SOFA had
# no eICU source config, so sepsis3, hypoxemic_respiratory_failure and
# oliguria dropped out; "shock" is the pre-rename name of
# sustained_hypotension_map (PR #225). Printed from
# concepts_for_source("eicu", task_set="v3") at cdbd4e7, the eICU flagship's
# training commit, and identical at 17d28fc, 1193aa3, a86616a, 69c8cb0 and
# f8633b3 (the other pre-#222 eICU runs' commits).
_EICU_26_CONCEPTS: tuple[str, ...] = (
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
)

# run directory basename -> the concept names that run trained with, in slot
# order. Verified against each checkpoint's own bottleneck width, not guessed.
LEGACY_CONCEPT_PINS: dict[str, tuple[str, ...]] = {
    # GEMINI, 15 of 29 resolved at training time; ten electrolyte and
    # haematology concepts plus hypoxemic_respiratory_failure, oliguria,
    # sepsis3 and shock gained GEMINI code mappings afterwards, taking
    # today's registry to 25. Recovered at training commit c1dadb9.
    # gemini_full_v10_15c is the same checkpoint under the name it was moved
    # to when a 25-concept retrain was attempted against the original path.
    # Pins key on the directory name, so a renamed run silently loses its
    # pin and refuses to load; alias rather than rename the canonical entry,
    # since either directory may hold the checkpoint on a given node.
    "gemini_full_v10_15c": (
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
    # eICU-CRD, 26 of 29; verified against eicu_full_v10's own bottleneck
    # width. sepsis3 (and with it the sepsis3 alert head) resolves today but
    # did not then.
    "eicu_full_v10": _EICU_26_CONCEPTS,
    # The other eICU runs banked under research_journal/figure_data/vm2 that
    # trained before PR #222, each on R6's config (docs/experiments.md) at a
    # commit whose registry printed the same 26 names: eicu_full_L_v10
    # (17d28fc), eicu_full_ADD_v10 (pre-2026-09-01), eicu_full_DEC_v10
    # (1193aa3), eicu_full_DEC_v11 (a86616a), eicu_full_DEC_v12 (69c8cb0) and
    # eicu_full_DEC_v12_steer (f8633b3, one epoch from the v12 checkpoint).
    # Recovered from the training commits, not read off the checkpoints.
    "eicu_full_L_v10": _EICU_26_CONCEPTS,
    "eicu_full_ADD_v10": _EICU_26_CONCEPTS,
    "eicu_full_DEC_v10": _EICU_26_CONCEPTS,
    "eicu_full_DEC_v11": _EICU_26_CONCEPTS,
    "eicu_full_DEC_v12": _EICU_26_CONCEPTS,
    "eicu_full_DEC_v12_steer": _EICU_26_CONCEPTS,
}


# eICU-CRD hazard heads before PR #222. Printed from
# alert_events_for("v3", source="eicu") at cdbd4e7 (eicu_full_v10's training
# commit) and identical at 17d28fc, 1193aa3, a86616a, 69c8cb0 and f8633b3:
#   ('vasopressor_start', 'icu_admission', 'acute_kidney_injury', 'death',
#    'readmission_30d')
# The same call on main today (ca541d8 and later) gives
#   ('vasopressor_start', 'icu_admission', 'acute_kidney_injury', 'death',
#    'sepsis3', 'readmission_30d')
# so sepsis3 lands in the MIDDLE of the list: without a pin the checkpoint's
# readmission_30d rows would be read as sepsis3 even if the width matched.
# Each checkpoint's event_heads.proj output width is 80 = 5 x 16 bins.
_EICU_PRE_SOFA_EVENTS: tuple[str, ...] = (
    "vasopressor_start",
    "icu_admission",
    "acute_kidney_injury",
    "death",
    "readmission_30d",
)

# run directory basename -> the hazard event names that run trained heads
# for, in head order. MIMIC-IV runs are absent on purpose: their event set
# has not changed, so today's registry still describes them.
LEGACY_EVENT_PINS: dict[str, tuple[str, ...]] = {
    "eicu_full_v10": _EICU_PRE_SOFA_EVENTS,
    "eicu_full_L_v10": _EICU_PRE_SOFA_EVENTS,
    "eicu_full_ADD_v10": _EICU_PRE_SOFA_EVENTS,
    "eicu_full_DEC_v10": _EICU_PRE_SOFA_EVENTS,
    "eicu_full_DEC_v11": _EICU_PRE_SOFA_EVENTS,
    "eicu_full_DEC_v12": _EICU_PRE_SOFA_EVENTS,
    "eicu_full_DEC_v12_steer": _EICU_PRE_SOFA_EVENTS,
}


def _run_name(run_dir: str | Path) -> str:
    """Return the directory's final component, the key the legacy tables use."""
    return str(run_dir).rstrip("/").rsplit("/", 1)[-1]


def write_run_pins(
    run_dir: str | Path,
    *,
    concept_names: Sequence[str],
    event_names: Sequence[str] | None,
) -> Path:
    """Record what a run trains with, in slot/head order, as ``run_pins.json``.

    Called by training right after the model is built, so the file
    describes the checkpoint's actual widths rather than whatever the
    registry resolves when the run is later loaded. ``event_names`` is
    ``None`` for a run without hazard heads and is stored as ``null``,
    which :func:`read_run_pins` reports as "no event pin".
    """
    path = Path(run_dir) / PIN_FILENAME
    payload = {
        "concepts": list(concept_names),
        "events": None if event_names is None else list(event_names),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_run_pins(run_dir: str | Path) -> dict[str, tuple[str, ...]]:
    """Return the pin file's non-null lists, or ``{}`` when there is none.

    A malformed file raises rather than falling through to the legacy
    tables: a pin that cannot be read is a broken run directory, not an
    unpinned one.
    """
    path = Path(run_dir) / PIN_FILENAME
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    out: dict[str, tuple[str, ...]] = {}
    for key in ("concepts", "events"):
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{path}: {key!r} must be a list of names or null")
        out[key] = tuple(value)
    return out


def pinned_concept_names(run_dir: str) -> tuple[str, ...] | None:
    """Return the pinned concept list for ``run_dir``, or ``None`` if unpinned.

    The run's own ``run_pins.json`` wins; otherwise the legacy table, which
    matches on the directory's final component, so an absolute path, a
    relative one and a trailing slash all resolve the same way.
    """
    from_file = read_run_pins(run_dir).get("concepts")
    if from_file is not None:
        return from_file
    return LEGACY_CONCEPT_PINS.get(_run_name(run_dir))


def pinned_event_names(run_dir: str) -> tuple[str, ...] | None:
    """Return the pinned hazard event list for ``run_dir``, or ``None``.

    Same resolution as :func:`pinned_concept_names`: ``run_pins.json``
    first, then :data:`LEGACY_EVENT_PINS` by directory name.
    """
    from_file = read_run_pins(run_dir).get("events")
    if from_file is not None:
        return from_file
    return LEGACY_EVENT_PINS.get(_run_name(run_dir))


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


def checkpoint_num_events(state: dict[str, object]) -> int | None:
    """Return how many hazard events a checkpoint's ``event_heads`` carry.

    The output layer is ``event_heads.proj.weight`` for the linear readout
    and the highest-indexed ``event_heads.proj.N.weight`` for the MLP one;
    its row count is ``num_events * HAZARD_NUM_BINS``. Returns ``None`` for
    a checkpoint without hazard heads.
    """
    weight = state.get("event_heads.proj.weight")
    if weight is None:
        layers = [
            k
            for k in state
            if k.startswith("event_heads.proj.") and k.endswith(".weight")
        ]
        if not layers:
            return None
        weight = state[max(layers, key=lambda k: int(k.split(".")[2]))]
    rows = int(weight.shape[0])  # type: ignore[attr-defined]
    if rows % HAZARD_NUM_BINS:
        raise ValueError(
            f"event_heads output width {rows} is not a multiple of "
            f"{HAZARD_NUM_BINS} hazard bins; the checkpoint's bin edges differ "
            "from DEFAULT_TIME_BIN_EDGES_HOURS"
        )
    return rows // HAZARD_NUM_BINS


def check_event_count(
    run_dir: str, state: dict[str, object], event_names: Sequence[str] | None
) -> None:
    """Raise if today's event list disagrees with the checkpoint's head width.

    The event-head twin of :func:`check_concept_count`, for the same reason:
    one sentence naming both counts and the fix, instead of a
    ``load_state_dict`` size-mismatch on ``event_heads.proj.2.weight``.
    """
    expected = checkpoint_num_events(state)
    resolved = len(event_names or ())
    if expected is None or expected == resolved:
        return
    name = _run_name(run_dir)
    raise ValueError(
        f"{name}: checkpoint has hazard heads for {expected} events but this "
        f"code resolves {resolved} for its source and task set. The alert "
        f"registry has changed since the run. Pin {name}'s training-time "
        f"event list: write it as the 'events' list in {name}/{PIN_FILENAME} "
        "or add it to LEGACY_EVENT_PINS in "
        "odyssey/inference/legacy_concept_pins.py (its env_fingerprint.json "
        "records the training commit; print hazard_events_for(task_set, "
        "source=source) there, in order). Loading it against today's list "
        "would build a head of the wrong width."
    )


def resolve_concepts_for_run(
    run_dir: str, source: str, task_set: str
) -> list[AnyConceptDefinition]:
    """Return the concept DEFINITIONS a run trained with, in its own slot order.

    :func:`pinned_concept_names` gives names; callers that build concept
    labels need the definitions behind them. Pinning only ``load_run`` is not
    enough: every caller that separately calls ``concepts_for_source`` gets
    today's larger list and then mismatches the model it was just handed. That
    is a real failure, not a hypothetical one -- it surfaced as
    ``zip() argument 2 is shorter than argument 1`` in interventions.py, where
    25 registry concepts met a 15-slot bottleneck's calibration gammas.

    Pinned names are resolved through :func:`canonical_concept_name`, since a
    pin records the name a checkpoint trained under and concepts get renamed
    (``shock`` is today's ``sustained_hypotension_map``).
    """
    definitions = {c.name: c for c in concepts_for_source(source, task_set=task_set)}
    pinned = pinned_concept_names(run_dir)
    if pinned is None:
        return list(definitions.values())
    resolved: list[AnyConceptDefinition] = []
    missing: list[str] = []
    for name in pinned:
        definition = definitions.get(name) or definitions.get(
            canonical_concept_name(name)
        )
        if definition is None:
            missing.append(name)
        else:
            resolved.append(definition)
    if missing:
        raise ValueError(
            f"{run_dir}: pinned concepts {missing} do not resolve for source "
            f"{source!r} under task_set {task_set!r}. The pin records what the "
            "checkpoint trained with; if the registry no longer defines those "
            "concepts the pin cannot be honoured."
        )
    return resolved
