"""Discover which raw codes an input-level edit should target, by occlusion.

``counterfactual.py``'s ``ValueEdit`` already moves hazards the clinically
expected way when the signal to edit is named by hand (hypotension ->
``sbp_noninvasive``, AKI -> ``creatinine``). That only covers concepts with
one obvious signal. Composite, criteria-based concepts (Sepsis-3, qSOFA, AKI
staging) have no single signal to name -- the input-level lever cannot reach
them today, not because editing doesn't work, but because nobody told it
what to edit.

This module finds the target automatically: for a concept and a patient
record, remove each candidate raw code from the window before the index
time in turn, re-score the concept head, and rank codes by how much
removing them shifts the concept's probability. The top codes are the
evidence the concept's own prediction rests on; removing them is then
usable as an automatically discovered edit, scored by the same
``score_record_at`` every hand-specified edit uses.

Occlusion, not gradient attribution, deliberately: it costs a forward pass
per candidate code rather than one backward pass, but it needs no autograd
wiring through tokenization and binning, and its output is trivial to sanity
check -- the top codes for "hypotension" should obviously be blood pressure
readings, not a lab panel three days away.

Removal here is by **exact code equality**, not ``ValueEdit``'s prefix
match. ``ValueEdit`` treats any signal containing ``//`` as a literal
prefix (``code.str.starts_with(signal)``), which is the right behavior for
a named panel edit ("remove every ``LAB//RESULT//`` reading") but wrong for
a single discovered code: if one candidate code happens to be a string
prefix of a different, unrelated code -- a real risk for hierarchical
coding schemes (ICD's ``E11`` is a prefix of ``E11.9``) -- prefix removal
would silently occlude both and contaminate the attribution. Exact
equality has no such collision.

Discovery (occlusion) and the edit it produces are two different
operations, deliberately. Removing a candidate's readings tells you
whether the concept's prediction rests on it; it does not tell you which
*direction* would make the concept more true, because a real patient's
actual readings are usually reassuring (normal), so deleting them makes
the record less informative and the prediction drifts toward the model's
higher unconditional base rate -- the same "reacts to surprise, not
meaning" effect the paper's label-override finding already documents,
just reached by deletion instead of a label patch. :func:`worsen_edits_from_attribution`
turns a discovered *code* into a discovered *signal* (via
:class:`~odyssey.data.signal_panel.SignalPanelResolver`, already
LOINC-keyed and per-source resolved, so this step is portable to eICU for
free) and sets it to a clinically worse value with the existing,
already-validated :class:`~odyssey.inference.counterfactual.ValueEdit`
machinery -- the same operation the hand-specified edits use, just aimed
automatically instead of by hand.

The cohort driver at the bottom (:func:`cohort_worsen`, ``python -m
odyssey.inference.concept_edit_attribution``) runs that discovery + edit
over held-out subjects and reports sign agreement of the hazard shift. It
also carries the reviewer-requested control: ``code_selection="random"``
draws the same number of codes per subject uniformly from the same
candidate pool the occlusion loop ranks, and pushes them through the
identical worsen-edit machinery, so the gap between the two arms is what
the concept ranking adds and nothing else.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

from odyssey.data.signal_panel import NO_SIGNAL, SIGNAL_PANEL, SignalPanelResolver
from odyssey.data.value_binning import QuantileBinner
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.counterfactual import (
    HORIZONS_HOURS,
    ForecastReadout,
    ValueEdit,
    _index_times_by_subject,
    apply_value_edits,
    score_record_at,
)
from odyssey.models.sequence_model import SequenceModel


logger = logging.getLogger(__name__)

# signal name (odyssey.data.signal_panel.SIGNAL_PANEL) -> the edit that
# pushes it toward a clinically worse value. Representative abnormal
# levels in the same spirit as counterfactual.py's STANDARD_EDITS
# (hypotension_6h sets SBP to 80), not per-patient calibrated. window_hours
# is filled in per call to match the attribution's own lookback window.
# Covers every SOFA component (respiration, coagulation, liver,
# cardiovascular, CNS, renal) plus lactate and respiratory rate (the
# third qSOFA criterion, alongside GCS and SBP already above); a signal
# absent here is one occlusion may point at but this module cannot yet
# act on.
WORSEN_EDIT_FOR_SIGNAL: dict[str, ValueEdit] = {
    "sbp_noninvasive": ValueEdit("sbp_noninvasive", "set", 80.0, None),
    "dbp_noninvasive": ValueEdit("dbp_noninvasive", "set", 40.0, None),
    "map_noninvasive": ValueEdit("map_noninvasive", "set", 55.0, None),
    "sbp_arterial": ValueEdit("sbp_arterial", "set", 80.0, None),
    "dbp_arterial": ValueEdit("dbp_arterial", "set", 40.0, None),
    "map_arterial": ValueEdit("map_arterial", "set", 55.0, None),
    "spo2": ValueEdit("spo2", "set", 85.0, None),
    "fio2": ValueEdit("fio2", "set", 0.60, None),  # more O2 support needed
    "resp_rate": ValueEdit("resp_rate", "set", 28.0, None),  # qSOFA: >=22/min
    "gcs_eye": ValueEdit("gcs_eye", "set", 1.0, None),
    "gcs_verbal": ValueEdit("gcs_verbal", "set", 1.0, None),
    "gcs_motor": ValueEdit("gcs_motor", "set", 1.0, None),
    "creatinine": ValueEdit("creatinine", "set", 3.0, None),
    "bilirubin_total": ValueEdit("bilirubin_total", "set", 3.0, None),
    "platelets": ValueEdit("platelets", "set", 50.0, None),
    "lactate": ValueEdit("lactate", "scale", 3.0, None),
}


@dataclass(frozen=True)
class CodeEdit:
    """Remove every exact-match reading of ``code`` inside the window."""

    code: str
    window_hours: float


@dataclass(frozen=True)
class CodeAttribution:
    """One candidate code's effect on a target readout when removed.

    The target is a concept's probability (:func:`occlusion_attribution`)
    or an event's risk at one horizon (:func:`event_occlusion_attribution`).
    """

    code: str
    n_rows: int
    """Readings of this code removed from the window."""
    baseline: float
    """The target's value with nothing removed."""
    occluded: float
    """The target's value with this code's readings removed."""

    @property
    def delta(self) -> float:
        """``occluded - baseline``.

        Negative means this code was pushing the target up; positive
        means it was suppressing it.
        """
        return self.occluded - self.baseline


def _in_window(
    *,
    index_time: object,
    window_hours: float,
    time_col: str = "time",
) -> pl.Expr:
    """Rows strictly before ``index_time``, inside ``window_hours`` of it.

    Shared by candidate discovery and removal so the two can never
    disagree about what "in the window" means.
    """
    return (
        pl.col(time_col).is_not_null()
        & (pl.col(time_col) <= pl.lit(index_time))
        & (pl.col(time_col) > pl.lit(index_time) - pl.duration(hours=window_hours))
    )


def _candidate_codes(
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    lookback_hours: float,
    code_col: str = "code",
) -> list[str]:
    """Distinct codes with a reading in the lookback window."""
    in_window = (
        raw_subject_events.filter(
            _in_window(index_time=index_time, window_hours=lookback_hours)
        )
        .select(code_col)
        .unique()
    )
    return sorted(in_window[code_col].to_list())


def remove_code_exact(
    raw_subject_events: pl.DataFrame,
    code: str,
    *,
    index_time: object,
    window_hours: float,
    code_col: str = "code",
) -> tuple[pl.DataFrame, int]:
    """Drop rows with ``code == code`` (exact) inside the window.

    Returns the edited frame and how many rows were dropped. Exact
    equality, never ``ValueEdit``'s prefix match -- see the module
    docstring for why that distinction matters here.
    """
    hit = (pl.col(code_col) == code) & _in_window(
        index_time=index_time, window_hours=window_hours
    )
    touched = int(raw_subject_events.select(hit.sum()).item())
    return raw_subject_events.filter(~hit), touched


def score_with_codes_removed(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    codes: Sequence[str],
    *,
    index_time: object,
    concept_names: Sequence[str],
    window_hours: float = 24.0,
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
) -> tuple[ForecastReadout, int]:
    """Remove every ``codes`` (exact match, each independently) and re-score.

    Returns the counterfactual readout and the total rows removed across
    all of ``codes`` combined.
    """
    edited = raw_subject_events
    total_touched = 0
    for code in codes:
        edited, touched = remove_code_exact(
            edited, code, index_time=index_time, window_hours=window_hours
        )
        total_touched += touched
    readout = score_record_at(
        model,
        vocab,
        binner,
        edited,
        index_time=index_time,
        concept_names=concept_names,
        source=source,
        device=device,
        chunk_size=chunk_size,
    )
    return readout, total_touched


def occlude_codes(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    value_of: Callable[[ForecastReadout], float],
    concept_names: Sequence[str],
    lookback_hours: float = 24.0,
    candidate_codes: Sequence[str] | None = None,
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
    horizons: Sequence[float] = HORIZONS_HOURS,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[CodeAttribution]:
    """Rank codes in the lookback window by their effect on ``value_of(readout)``.

    The target-agnostic core of :func:`occlusion_attribution` and
    :func:`event_occlusion_attribution`. ``candidate_codes`` restricts the
    search (e.g. to a curated pool); by default every distinct code with a
    reading in the window is tried, one at a time, by exact-match removal.
    Returned sorted by ``abs(delta)`` descending; codes with zero readings
    in the window (nothing removed) are silently excluded rather than
    reported as a zero-effect result. ``on_progress(done, total)`` is
    called after each candidate, for callers reporting progress on a long
    search (one full re-score per candidate).
    """

    def _score(events: pl.DataFrame) -> float:
        return value_of(
            score_record_at(
                model,
                vocab,
                binner,
                events,
                index_time=index_time,
                concept_names=concept_names,
                source=source,
                device=device,
                chunk_size=chunk_size,
                horizons=horizons,
            )
        )

    baseline = _score(raw_subject_events)
    codes = (
        list(candidate_codes)
        if candidate_codes is not None
        else _candidate_codes(
            raw_subject_events, index_time=index_time, lookback_hours=lookback_hours
        )
    )
    results: list[CodeAttribution] = []
    for done, code in enumerate(codes, start=1):
        edited, touched = remove_code_exact(
            raw_subject_events,
            code,
            index_time=index_time,
            window_hours=lookback_hours,
        )
        if touched > 0:
            results.append(
                CodeAttribution(
                    code=code,
                    n_rows=touched,
                    baseline=baseline,
                    occluded=_score(edited),
                )
            )
        if on_progress is not None:
            on_progress(done, len(codes))
    results.sort(key=lambda r: abs(r.delta), reverse=True)
    return results


def occlusion_attribution(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    concept_name: str,
    concept_names: Sequence[str],
    lookback_hours: float = 24.0,
    candidate_codes: Sequence[str] | None = None,
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
) -> list[CodeAttribution]:
    """Rank codes in the lookback window by their effect on one concept.

    See :func:`occlude_codes` for the search; the target is the concept's
    probability at the index position.
    """
    if concept_name not in concept_names:
        raise ValueError(f"{concept_name!r} not in concept_names {list(concept_names)}")
    return occlude_codes(
        model,
        vocab,
        binner,
        raw_subject_events,
        index_time=index_time,
        value_of=lambda readout: readout.concept_probs[concept_name],
        concept_names=concept_names,
        lookback_hours=lookback_hours,
        candidate_codes=candidate_codes,
        source=source,
        device=device,
        chunk_size=chunk_size,
    )


def event_occlusion_attribution(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    event: str,
    horizon_hours: float,
    concept_names: Sequence[str],
    lookback_hours: float = 24.0,
    candidate_codes: Sequence[str] | None = None,
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[CodeAttribution]:
    """Rank codes in the lookback window by their effect on one event's risk.

    See :func:`occlude_codes` for the search; the target is the hazard
    head's ``P(event within horizon_hours)`` at the index position.
    ``horizon_hours`` should be a hazard bin edge (8, 24, 72, ...) for an
    exact probability.

    Raises
    ------
    ValueError
        If the model has no hazard head named ``event``.
    """
    key = f"{horizon_hours:g}h"
    event_names = list(getattr(getattr(model, "event_heads", None), "event_names", []))
    if event not in event_names:
        raise ValueError(f"{event!r} is not a hazard head of this model: {event_names}")
    return occlude_codes(
        model,
        vocab,
        binner,
        raw_subject_events,
        index_time=index_time,
        value_of=lambda readout: readout.event_risk[event][key],
        concept_names=concept_names,
        lookback_hours=lookback_hours,
        candidate_codes=candidate_codes,
        source=source,
        device=device,
        chunk_size=chunk_size,
        horizons=(horizon_hours,),
        on_progress=on_progress,
    )


def auto_edit_from_attribution(
    attributions: Sequence[CodeAttribution],
    *,
    top_k: int = 3,
    window_hours: float = 24.0,
    min_abs_delta: float = 0.0,
) -> list[CodeEdit]:
    """Turn the top ``top_k`` attributed codes into exact-match removal edits.

    Assumes ``attributions`` is already sorted by ``abs(delta)`` descending
    (what :func:`occlusion_attribution` returns) -- this does not re-sort.
    ``min_abs_delta`` drops codes whose occlusion barely moved the concept
    (noise, not evidence) even if they were among the top ``top_k``.
    """
    chosen = [a for a in attributions[:top_k] if abs(a.delta) >= min_abs_delta]
    return [CodeEdit(code=a.code, window_hours=window_hours) for a in chosen]


def worsen_edits_from_attribution(
    attributions: Sequence[CodeAttribution],
    *,
    source: str = "mimic_iv",
    top_k: int = 4,
    min_abs_delta: float = 0.0,
    window_hours: float = 24.0,
) -> list[ValueEdit]:
    """Turn attributed codes into worsen-the-signal edits, where possible.

    Resolves each of the top ``top_k`` attributed codes to its named panel
    signal (:data:`~odyssey.data.signal_panel.SIGNAL_PANEL`, LOINC-keyed
    and resolved for ``source``) and looks up that signal's edit in
    :data:`WORSEN_EDIT_FOR_SIGNAL`. A code that resolves to no panel
    signal, or to one this module has no worsen direction for, is
    dropped rather than guessed at. Assumes ``attributions`` is already
    sorted by ``abs(delta)`` descending, as :func:`occlusion_attribution`
    returns; does not re-sort. Two different attributed codes that
    resolve to the same signal (e.g. two raw SBP item codes) collapse to
    one edit.
    """
    chosen = [a for a in attributions[:top_k] if abs(a.delta) >= min_abs_delta]
    return worsen_edits_for_codes(
        [a.code for a in chosen], source=source, window_hours=window_hours
    )


def worsen_signal_for_code(code: str, resolver: SignalPanelResolver) -> str | None:
    """Return the panel signal ``code`` resolves to, if this module can worsen it.

    ``None`` when the code is outside the panel or its signal has no entry
    in :data:`WORSEN_EDIT_FOR_SIGNAL`: the one rule that decides which
    codes the edit machinery can act on, shared by the attributed arm,
    the random-code control and the candidate pool they both draw from.
    """
    idx = resolver.resolve(code)
    if idx == NO_SIGNAL:
        return None
    signal_name = SIGNAL_PANEL[idx][0]
    return signal_name if signal_name in WORSEN_EDIT_FOR_SIGNAL else None


def worsen_edits_for_codes(
    codes: Sequence[str],
    *,
    source: str = "mimic_iv",
    window_hours: float = 24.0,
) -> list[ValueEdit]:
    """Turn raw codes into worsen-the-signal edits, in the order given.

    The edit-construction half of :func:`worsen_edits_from_attribution`,
    with no ranking of its own: a code that resolves to no worsenable
    signal is dropped, and two codes on the same signal collapse to one
    edit. The random-code control calls this on its uniform draw so the
    two arms differ only in which codes come in.
    """
    resolver = SignalPanelResolver(source=source)
    edits: dict[str, ValueEdit] = {}
    for code in codes:
        signal_name = worsen_signal_for_code(code, resolver)
        if signal_name is None:
            continue
        template = WORSEN_EDIT_FOR_SIGNAL[signal_name]
        edits[signal_name] = ValueEdit(
            signal=template.signal,
            mode=template.mode,
            value=template.value,
            window_hours=window_hours,
        )
    return list(edits.values())


def score_with_worsen_edits(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    edits: Sequence[ValueEdit],
    *,
    index_time: object,
    concept_names: Sequence[str],
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
) -> tuple[ForecastReadout, int]:
    """Apply the edits from :func:`worsen_edits_from_attribution` and re-score.

    Thin wrapper over :func:`~odyssey.inference.counterfactual.apply_value_edits`
    + :func:`~odyssey.inference.counterfactual.score_record_at`, kept here
    so callers don't need to import the edit-application step separately
    from the attribution step that produced the edits.
    """
    edited, touched = apply_value_edits(
        raw_subject_events, edits, index_time=index_time, source=source
    )
    readout = score_record_at(
        model,
        vocab,
        binner,
        edited,
        index_time=index_time,
        concept_names=concept_names,
        source=source,
        device=device,
        chunk_size=chunk_size,
    )
    return readout, touched


# ---------------------------------------------------------------------------
# Cohort validation: discovered (or random) codes -> worsen edits -> hazards
# ---------------------------------------------------------------------------

CodeSelection = Literal["attributed", "random"]
CandidatePool = Literal["mappable", "all"]


def mappable_candidate_codes(
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    lookback_hours: float,
    source: str = "mimic_iv",
    resolver: SignalPanelResolver | None = None,
) -> list[str]:
    """Distinct codes in the window that the worsen-edit machinery can act on.

    :func:`occlude_codes`'s default pool is every distinct code with a
    reading in the window; most of those (diagnoses, orders, drugs) have
    no worsen edit, so the attributed arm can only ever act on the subset
    that resolves to a signal in :data:`WORSEN_EDIT_FOR_SIGNAL`. This is
    that subset, sorted, and it is the pool both arms of the cohort
    validation draw from.
    """
    resolver = resolver or SignalPanelResolver(source=source)
    codes = _candidate_codes(
        raw_subject_events, index_time=index_time, lookback_hours=lookback_hours
    )
    return [c for c in codes if worsen_signal_for_code(c, resolver) is not None]


def select_random_codes(
    candidate_codes: Sequence[str],
    *,
    n: int,
    seed: int,
    subject_id: int,
) -> list[str]:
    """Draw ``n`` codes uniformly without replacement from ``candidate_codes``.

    Seeded per subject from ``(seed, subject_id)`` so the draw is
    reproducible and independent of the order subjects are visited in.
    Fewer than ``n`` candidates returns them all (shuffled), which is the
    same number the attributed arm would edit from that pool.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    pool = sorted(set(candidate_codes))
    rng = random.Random(f"{seed}:{subject_id}")
    return rng.sample(pool, min(n, len(pool)))


@dataclass
class SubjectWorsenRecord:
    """One subject's discovered (or drawn) codes, edit, and hazard shift."""

    subject_id: int
    index_time: str
    n_candidates: int
    """Size of the candidate pool both arms draw from."""
    codes: list[str]
    """Codes the arm selected (top-``k`` attributed or the random draw)."""
    signals: list[str]
    """Panel signals those codes resolved to, i.e. the edits applied."""
    rows_edited: int
    concept_before: float
    concept_after: float
    delta_event_risk: dict[str, dict[str, float]]
    """event -> horizon -> (worsened - factual) risk."""


@dataclass
class CohortWorsenResult:
    """The cohort validation's per-subject records plus how they were made."""

    concept: str
    source: str
    top_k: int
    lookback_hours: float
    index_hours: float
    index_frac: float | None
    code_selection: str
    random_seed: int | None
    candidate_pool: str
    min_abs_delta: float
    subjects: list[SubjectWorsenRecord] = field(default_factory=list)
    n_pool_above_top_k: int = 0
    """Subjects whose pool held more than ``top_k`` codes, i.e. the only
    subjects where the random draw can differ from the attributed set."""

    @property
    def scored(self) -> list[SubjectWorsenRecord]:
        """Subjects with at least one reading actually edited."""
        return [s for s in self.subjects if s.rows_edited > 0]

    def sign_agreement(self) -> dict[str, dict[str, tuple[int, int]]]:
        """Return event -> horizon -> (subjects whose risk rose, subjects scored)."""
        out: dict[str, dict[str, tuple[int, int]]] = {}
        scored = self.scored
        events = sorted({ev for s in scored for ev in s.delta_event_risk})
        for ev in events:
            horizons = [
                h
                for h in (f"{x:g}h" for x in HORIZONS_HOURS)
                if any(h in s.delta_event_risk.get(ev, {}) for s in scored)
            ]
            out[ev] = {}
            for h in horizons:
                ds = [
                    s.delta_event_risk[ev][h]
                    for s in scored
                    if h in s.delta_event_risk.get(ev, {})
                ]
                out[ev][h] = (sum(1 for d in ds if d > 0), len(ds))
        return out


def _candidate_pool(
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    lookback_hours: float,
    candidate_pool: CandidatePool,
    source: str,
    resolver: SignalPanelResolver,
) -> list[str]:
    if candidate_pool == "mappable":
        return mappable_candidate_codes(
            raw_subject_events,
            index_time=index_time,
            lookback_hours=lookback_hours,
            source=source,
            resolver=resolver,
        )
    return _candidate_codes(
        raw_subject_events, index_time=index_time, lookback_hours=lookback_hours
    )


def _resolve_signals(codes: Sequence[str], resolver: SignalPanelResolver) -> list[str]:
    seen: list[str] = []
    for code in codes:
        name = worsen_signal_for_code(code, resolver)
        if name is not None and name not in seen:
            seen.append(name)
    return seen


def cohort_worsen(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_events: pl.DataFrame,
    *,
    concept_name: str,
    concept_names: Sequence[str],
    top_k: int = 4,
    lookback_hours: float = 24.0,
    index_hours: float = 24.0,
    index_frac: float | None = None,
    max_subjects: int = 50,
    code_selection: CodeSelection = "attributed",
    random_seed: int = 0,
    candidate_pool: CandidatePool = "mappable",
    min_abs_delta: float = 0.0,
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
    log: Callable[[str], None] | None = None,
) -> CohortWorsenResult:
    """Discover (or draw) codes per subject, worsen them, and re-score hazards.

    Subjects are the first ``max_subjects`` held-out subjects, in the
    order the shards list them, whose first visit lasts at least
    ``index_hours`` and whose candidate pool is non-empty; the pool does
    not depend on the arm, so both arms score the same subjects. Per
    subject, ``attributed`` ranks the pool by occlusion of the concept
    head and keeps the top ``top_k`` codes; ``random`` draws
    ``min(top_k, pool size)`` codes uniformly from the same pool, seeded
    from ``(random_seed, subject_id)``. Both arms then build the same
    worsen edits (:func:`worsen_edits_for_codes`) and read the hazard
    heads before and after. ``candidate_pool="all"`` ranks every distinct
    code in the window instead, the way :func:`occlude_codes` does by
    default; a selected code with no worsen edit is then dropped, so a
    random draw there can leave a subject with nothing to edit, which is
    reported but not scored.
    """
    if concept_name not in concept_names:
        raise ValueError(f"{concept_name!r} not in concept_names {list(concept_names)}")
    if code_selection not in ("attributed", "random"):
        raise ValueError(f"unknown code_selection {code_selection!r}")
    if candidate_pool not in ("mappable", "all"):
        raise ValueError(f"unknown candidate_pool {candidate_pool!r}")
    emit = log or (lambda _msg: None)
    resolver = SignalPanelResolver(source=source)
    index_times = _index_times_by_subject(
        raw_events, index_hours=index_hours, index_frac=index_frac
    )
    ordered = [
        int(sid)
        for sid in raw_events["subject_id"].unique(maintain_order=True).to_list()
        if sid in index_times
    ]
    result = CohortWorsenResult(
        concept=concept_name,
        source=source,
        top_k=top_k,
        lookback_hours=lookback_hours,
        index_hours=index_hours,
        index_frac=index_frac,
        code_selection=code_selection,
        random_seed=random_seed if code_selection == "random" else None,
        candidate_pool=candidate_pool,
        min_abs_delta=min_abs_delta,
    )
    for sid in ordered:
        if len(result.subjects) >= max_subjects:
            break
        sub = raw_events.filter(pl.col("subject_id") == sid)
        index_time = index_times[sid]
        pool = _candidate_pool(
            sub,
            index_time=index_time,
            lookback_hours=lookback_hours,
            candidate_pool=candidate_pool,
            source=source,
            resolver=resolver,
        )
        if not pool:
            continue
        if len(pool) > top_k:
            result.n_pool_above_top_k += 1
        if code_selection == "attributed":
            attributions = occlusion_attribution(
                model,
                vocab,
                binner,
                sub,
                index_time=index_time,
                concept_name=concept_name,
                concept_names=concept_names,
                lookback_hours=lookback_hours,
                candidate_codes=pool,
                source=source,
                device=device,
                chunk_size=chunk_size,
            )
            codes = [
                a.code for a in attributions[:top_k] if abs(a.delta) >= min_abs_delta
            ]
        else:
            codes = select_random_codes(
                pool, n=min(top_k, len(pool)), seed=random_seed, subject_id=sid
            )
        edits = worsen_edits_for_codes(
            codes, source=source, window_hours=lookback_hours
        )
        factual = score_record_at(
            model,
            vocab,
            binner,
            sub,
            index_time=index_time,
            concept_names=concept_names,
            source=source,
            device=device,
            chunk_size=chunk_size,
        )
        if edits:
            worsened, touched = score_with_worsen_edits(
                model,
                vocab,
                binner,
                sub,
                edits,
                index_time=index_time,
                concept_names=concept_names,
                source=source,
                device=device,
                chunk_size=chunk_size,
            )
        else:
            worsened, touched = factual, 0
        before = factual.concept_probs.get(concept_name, float("nan"))
        after = worsened.concept_probs.get(concept_name, float("nan"))
        record = SubjectWorsenRecord(
            subject_id=sid,
            index_time=str(index_time),
            n_candidates=len(pool),
            codes=codes,
            signals=_resolve_signals(codes, resolver),
            rows_edited=touched,
            concept_before=before,
            concept_after=after,
            delta_event_risk={
                ev: {h: worsened.event_risk[ev][h] - p for h, p in hs.items()}
                for ev, hs in factual.event_risk.items()
                if ev in worsened.event_risk
            },
        )
        result.subjects.append(record)
        emit(
            f"subject {sid} [{len(result.subjects)}/{max_subjects}]: "
            f"{record.signals!r} concept {before:.3f}->{after:.3f}"
        )
    return result


def format_cohort_summary(result: CohortWorsenResult) -> str:
    """Render the fixed-format summary block the log parser reads.

    ``scripts/parse_edit_attribution_logs.py`` turns this into JSON. Same
    layout as the banked ``cohort_*.log`` files under
    ``research_journal/figure_data/edit_attribution/``, plus a
    ``selection=`` tag (and ``seed=`` for the random arm) in the header
    so the parser can label the arm.
    """
    n = len(result.subjects)
    header = (
        f"=== {result.concept} on {result.source}, n={n} subjects, top_k={result.top_k}"
    )
    if result.index_frac is not None:
        header += f", index_frac={result.index_frac:.4f}"
    header += f", selection={result.code_selection}"
    if result.code_selection == "random":
        header += f", seed={result.random_seed}"
    if result.candidate_pool != "mappable":
        header += f", pool={result.candidate_pool}"
    no_edit = n - len(result.scored)
    if no_edit:
        header += f", no_edit={no_edit}"
    header += " ==="
    saturated = sum(1 for s in result.subjects if s.concept_before >= 0.9)
    freq: dict[str, int] = {}
    for s in result.subjects:
        for name in s.signals:
            freq[name] = freq.get(name, 0) + 1
    scored = result.scored
    mean_delta = (
        sum(s.concept_after - s.concept_before for s in scored) / len(scored)
        if scored
        else float("nan")
    )
    lines = [
        header,
        f"baseline concept prob >= 0.9: {saturated}/{n} subjects (saturation check)",
        f"candidate pool larger than top_k: {result.n_pool_above_top_k}/{n} subjects",
        f"edit signal frequency: {freq!r}",
        f"mean concept-probability delta: {mean_delta:+.4f}",
        "",
        "Sign agreement (risk should INCREASE when discovered evidence is worsened):",
    ]
    for event, horizons in result.sign_agreement().items():
        for horizon, (agree, total) in horizons.items():
            pct = 100.0 * agree / total if total else 0.0
            lines.append(
                f"  {event:<22} {horizon:<4}: {agree:3d}/{total:3d} = {pct:5.1f}%"
            )
    return "\n".join(lines) + "\n"


def _main() -> None:
    from odyssey.data.code_normalization import maybe_normalize  # noqa: PLC0415
    from odyssey.data.history_recap import maybe_history_recap  # noqa: PLC0415
    from odyssey.inference.legacy_concept_pins import (  # noqa: PLC0415
        resolve_concepts_for_run,
    )
    from odyssey.inference.run_inference import load_run  # noqa: PLC0415
    from odyssey.training.data import load_meds_shards  # noqa: PLC0415
    from odyssey.utils.device import default_device  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description=(
            "Automated occlusion + worsen-edit cohort validation (paper "
            "tab:edit-attribution), with a random-code control arm."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument(
        "--concept", required=True, help="e.g. sepsis3, qsofa, aki_stage_3"
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--max-subjects", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--index-hours", type=float, default=24.0)
    parser.add_argument(
        "--index-frac",
        type=float,
        default=None,
        help="index at this fraction of the first qualifying visit instead of --index-hours into it",
    )
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument(
        "--code-selection",
        choices=("attributed", "random"),
        default="attributed",
        help="attributed: top-k by occlusion (the paper's arm); random: same count, uniform from the same pool",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--candidate-pool",
        choices=("mappable", "all"),
        default="mappable",
        help="mappable: codes with a worsen edit (default); all: every distinct code in the window",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or default_device()
    run_dir = Path(args.run_dir)
    model, vocab, binner, config = load_run(
        run_dir,
        device=device,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
    )
    source = getattr(config, "source", "mimic_iv")
    concept_names = [
        c.name
        for c in resolve_concepts_for_run(
            str(run_dir), source, getattr(config, "task_set", "v1")
        )
    ]
    raw = load_meds_shards(args.held_out_shard_dir, max_shards=args.max_shards)
    raw = maybe_normalize(
        raw, enabled=getattr(config, "normalize_medications", False), source=source
    )
    raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
    resolver = SignalPanelResolver(source=source)
    present = {
        name
        for name in (
            worsen_signal_for_code(c, resolver) for c in raw["code"].unique().to_list()
        )
        if name is not None
    }
    print(
        f"worsenable signals with a reading in the loaded shards: "
        f"{len(present)}/{len(WORSEN_EDIT_FOR_SIGNAL)}",
        flush=True,
    )
    result = cohort_worsen(
        model,
        vocab,
        binner,
        raw,
        concept_name=args.concept,
        concept_names=concept_names,
        top_k=args.top_k,
        lookback_hours=args.lookback_hours,
        index_hours=args.index_hours,
        index_frac=args.index_frac,
        max_subjects=args.max_subjects,
        code_selection=args.code_selection,
        random_seed=args.random_seed,
        candidate_pool=args.candidate_pool,
        min_abs_delta=args.min_abs_delta,
        source=source,
        device=device,
        chunk_size=args.chunk_size,
        log=lambda msg: print(msg, flush=True),
    )
    summary = format_cohort_summary(result)
    print()
    print(summary, end="", flush=True)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["run_dir"] = str(run_dir)
    payload["sign_agreement"] = {
        ev: {h: {"agree": a, "total": t} for h, (a, t) in hs.items()}
        for ev, hs in result.sign_agreement().items()
    }
    payload["summary_text"] = summary
    out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("[cohort_worsen] wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _main()
