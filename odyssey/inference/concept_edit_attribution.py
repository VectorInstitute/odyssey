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
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from odyssey.data.signal_panel import NO_SIGNAL, SIGNAL_PANEL, SignalPanelResolver
from odyssey.data.value_binning import QuantileBinner
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.counterfactual import (
    ForecastReadout,
    ValueEdit,
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
# cardiovascular, CNS, renal) plus lactate; a signal absent here is one
# occlusion may point at but this module cannot yet act on.
WORSEN_EDIT_FOR_SIGNAL: dict[str, ValueEdit] = {
    "sbp_noninvasive": ValueEdit("sbp_noninvasive", "set", 80.0, None),
    "dbp_noninvasive": ValueEdit("dbp_noninvasive", "set", 40.0, None),
    "map_noninvasive": ValueEdit("map_noninvasive", "set", 55.0, None),
    "sbp_arterial": ValueEdit("sbp_arterial", "set", 80.0, None),
    "dbp_arterial": ValueEdit("dbp_arterial", "set", 40.0, None),
    "map_arterial": ValueEdit("map_arterial", "set", 55.0, None),
    "spo2": ValueEdit("spo2", "set", 85.0, None),
    "fio2": ValueEdit("fio2", "set", 0.60, None),  # more O2 support needed
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
    """One candidate code's effect on a concept's probability when removed."""

    code: str
    n_rows: int
    """Readings of this code removed from the window."""
    baseline: float
    """The concept's probability with nothing removed."""
    occluded: float
    """The concept's probability with this code's readings removed."""

    @property
    def delta(self) -> float:
        """``occluded - baseline``.

        Negative means this code was pushing the concept up; positive
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

    ``candidate_codes`` restricts the search (e.g. to a curated pool); by
    default every distinct code with a reading in the window is tried, one
    at a time, by exact-match removal. Returned sorted by ``abs(delta)``
    descending; codes with zero readings in the window (nothing removed)
    are silently excluded rather than reported as a zero-effect result.
    """
    baseline_readout = score_record_at(
        model,
        vocab,
        binner,
        raw_subject_events,
        index_time=index_time,
        concept_names=concept_names,
        source=source,
        device=device,
        chunk_size=chunk_size,
    )
    if concept_name not in baseline_readout.concept_probs:
        raise ValueError(f"{concept_name!r} not in concept_names {list(concept_names)}")
    baseline = baseline_readout.concept_probs[concept_name]

    codes = (
        list(candidate_codes)
        if candidate_codes is not None
        else _candidate_codes(
            raw_subject_events, index_time=index_time, lookback_hours=lookback_hours
        )
    )
    results: list[CodeAttribution] = []
    for code in codes:
        edited, touched = remove_code_exact(
            raw_subject_events,
            code,
            index_time=index_time,
            window_hours=lookback_hours,
        )
        if touched == 0:
            continue
        occluded_readout = score_record_at(
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
        occluded = occluded_readout.concept_probs.get(concept_name, baseline)
        results.append(
            CodeAttribution(
                code=code, n_rows=touched, baseline=baseline, occluded=occluded
            )
        )
    results.sort(key=lambda r: abs(r.delta), reverse=True)
    return results


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
    resolver = SignalPanelResolver(source=source)
    chosen = [a for a in attributions[:top_k] if abs(a.delta) >= min_abs_delta]
    edits: dict[str, ValueEdit] = {}
    for a in chosen:
        idx = resolver.resolve(a.code)
        if idx == NO_SIGNAL:
            continue
        signal_name = SIGNAL_PANEL[idx][0]
        template = WORSEN_EDIT_FOR_SIGNAL.get(signal_name)
        if template is None:
            continue
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
