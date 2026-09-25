"""Concept intervention and completeness evaluation.

The one architectural claim a concept bottleneck makes beyond ordinary
sequence modeling is that the supervised concepts *mediate* prediction:
the task head reads a mixture steered by the concept probabilities, so
editing those probabilities should causally move the forecasts. This
module tests that claim directly, CEM/CBGM-style, by re-running the
streaming next-event evaluation under do()-style edits inside the
bottleneck (:class:`~odyssey.models.concept_bottleneck.BottleneckIntervention`)
and comparing task metrics across modes:

- ``none`` -- the unedited baseline; must reproduce the standard
  evaluation's numbers.
- ``truth`` -- replace each known concept's mixing probability with its
  ground-truth rule label wherever that label is observed. If concepts
  causally steer prediction, perfect concept information should *help*
  (or at minimum not hurt) next-event accuracy; a model that ignores its
  bottleneck shows no movement.
- ``flip`` -- feed ``1 - label`` on the same positions. The mirror
  image: reliance on the concept channel shows up as damage.
- ``flip_gated`` -- the same ``1 - label`` edit, but its logit changes
  pass through a suppression-only gate: ``logits = logits_none +
  min(0, logits_flip - logits_none)``, so the flip may lower token
  logits but never raise them. Guide Labs (arXiv:2608.07594, Fig. 19)
  show naive negative steering *promotes* anti-aligned or unrelated
  vocabulary rather than only suppressing the aligned direction; this
  mode is the control for that artifact. If ``flip_gated`` recovers
  most of ``flip``'s damage relative to ``none``, the damage came from
  spurious promotion (a steering artifact), not from losing the true
  concept's contribution.
- ``truth_calibrated`` / ``flip_calibrated`` -- the output-calibrated
  protocol (Guide Labs, adapted): instead of a hard 0/1 value, displace
  the model's own probability by a per-concept step ``gamma_i = tau /
  peak_i`` toward the true (resp. flipped) pole, clipped to [0, 1],
  where ``peak_i`` is concept ``i``'s largest per-token logit shift per
  unit of mixing probability (see
  :func:`~odyssey.inference.concept_attribution.calibrated_gammas`).
  Every concept then applies the same largest achievable logit shift
  ``tau``, so per-concept sensitivities are comparable regardless of
  how large the head's weights happen to be for each concept -- the
  targeted fix for band-population artifacts (rare concepts rarely
  enter the |p - 0.5| band). The ``uncertain_band`` restriction is
  deliberately NOT applied to these modes: calibration replaces the
  band as the equalizer.
- ``random`` -- feed coin-flip values on the same positions. Separates
  "any perturbation hurts" from "wrong information hurts": a gap
  between ``random`` and ``flip`` means the model reads the *direction*
  of the concept values, not just their stability.
- ``zero_known`` / ``zero_unknown`` -- zero the known concepts' (resp.
  the unknown channel's) mixed embeddings. The completeness probe: how
  the task signal is apportioned between the supervised, interpretable
  channel and the unsupervised one. A bottleneck whose entire task
  performance survives ``zero_known`` is interpretable-in-name-only --
  the concepts would be a decorative side channel.
- ``zero_residual`` -- zero the residual term of a decomposed
  bottleneck, leaving the named and unknown parts. Only the decomposed
  bottleneck has a residual; on the mixture bottleneck this mode is
  inert. It is the direct test for residual domination: the
  decomposition is algebraically exact by construction, so the question
  is whether the residual is where the predictive capacity actually
  went. Read it against ``zero_known``: a model that shrugs off losing
  its named channel but collapses without its residual has routed the
  task around the concepts.

Intervened values are applied per position as *running* labels: the
visit- (or stay-) scoped label, but true only from the concept's
first-trigger time onward, so what is fed at each position is what is
true as of that moment rather than a retrospective fact about the whole
visit (see :func:`_position_labels`). Everything is gated by the
observed mask: unobserved concepts keep the model's own probability, in
every mode -- there is no ground truth to feed there.

Hazard heads (``--hazard-heads``). The next-event scores above say
nothing about the clinical alert heads. With the flag on, the same pass
also reads each per-event hazard head at the landmark rows the alert
protocol scores (:mod:`odyssey.inference.alerts`: the first event of
every 4 h bucket in each visit, at-risk rows only, right-censored rows
dropped per horizon) and reports, per mode, event and horizon, the AUROC
against the landmark outcome and the mean predicted ``P(event within
h)``. Because the override is applied at every position, the readout is
what the alert would have said had the concept been overridden all
along. Three paired, subject-clustered bootstrap differences (truth
minus none, truth minus flip, flip minus none) are attached to the
left-hand mode's entry. For ``flip_gated`` the heads are read from the
flip-intervened features: the logit gate has no hazard analogue. Off by
default, so an existing run reproduces byte for byte.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import roc_auc_score

from odyssey.data.alert_events import (
    AlertEvent,
    EventTimes,
    alert_events_for,
    all_event_times,
)
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.inference.alerts import (
    HORIZONS_HOURS,
    LandmarkState,
    _landmark_mask,
    _visit_starts,
    position_visit_starts,
)
from odyssey.inference.concept_attribution import (
    calibrated_gammas,
    mean_concept_directions,
)
from odyssey.inference.legacy_concept_pins import resolve_concepts_for_run
from odyssey.inference.run_inference import (
    _CODE_TYPE_NAMES,
    _build_type_lookup,
    load_run,
    refuse_existing_output,
)
from odyssey.inference.uncertainty import BootstrapAUROCDelta, bootstrap_auroc_delta
from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    intervention_apply_mask,
)
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
)
from odyssey.models.time_to_event import EventHazardHeads, probability_within
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
    build_visit_concept_first_times,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.running_labels import position_running_labels
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

INTERVENTION_MODES = (
    "none",
    "truth",
    "flip",
    "flip_gated",
    "truth_calibrated",
    "flip_calibrated",
    "random",
    "zero_known",
    "zero_unknown",
    "zero_residual",
)

CALIBRATED_MODES = ("truth_calibrated", "flip_calibrated")

#: Paired hazard differences, (left, right), reported on the left mode's entry.
HAZARD_PAIRS: tuple[tuple[str, str], ...] = (
    ("truth", "none"),
    ("truth", "flip"),
    ("flip", "none"),
)


@dataclass(frozen=True)
class InterventionResult:
    """Task metrics for one intervention mode over the held-out stream."""

    mode: str
    n_predictions: int
    top1_accuracy: float
    mean_task_loss: float
    top1_by_code_type: dict[str, float] = field(default_factory=dict)
    n_by_code_type: dict[str, int] = field(default_factory=dict)
    n_intervened_positions: int = 0
    """Positions where at least one concept's mixing probability was
    actually replaced (0 for none/zero_* modes, which edit embeddings
    or nothing)."""

    uncertain_band: float | None = None
    """If set, values were only injected where the model's own probability
    was within this distance of 0.5 (see BottleneckIntervention)."""

    mean_abs_displacement: float | None = None
    """Mean ``|injected value - model's own probability|`` over the
    concept entries actually replaced: how far the intervention pushed
    the bottleneck. Truth and flip displace by ``1 - p`` and ``p``
    respectively, so comparing their accuracy deltas without this is
    comparing perturbations of different sizes."""

    calibrated_tau: float | None = None
    """For the *_calibrated modes: the shared peak logit shift every
    concept's step was calibrated to."""

    n_replaced_by_concept: dict[str, int] | None = None
    """Per-concept count of entries actually replaced (W3 band coverage:
    under an uncertain band, a rare concept whose probability hugs its
    base rate rarely enters the band at all -- this is the denominator
    that makes per-concept sensitivity claims honest). None for modes
    that replace nothing."""

    mean_abs_displacement_by_concept: dict[str, float] | None = None
    """Per-concept mean ``|injected - own|`` over that concept's replaced
    entries (NaN for a concept with zero replacements)."""

    calibration_gamma: dict[str, float] | None = None
    """For the *_calibrated modes: the per-concept mixing-probability
    step ``tau / peak_i`` (attached with concept names by
    :func:`evaluate_interventions`)."""

    hazard: dict[str, Any] | None = None
    """``{event: {"8h": {auroc, mean_risk, n_at_risk, n_positive,
    n_censored}}}`` from the hazard heads at landmark rows under this
    mode (``--hazard-heads``); None when hazard scoring is off."""

    hazard_paired: dict[str, Any] | None = None
    """``{"truth_minus_none": {event: {"8h": {auroc, mean_risk, ...}}}}``:
    the :data:`HAZARD_PAIRS` differences whose left mode is this one,
    each with a subject-clustered paired bootstrap interval. ``{}`` for a
    mode that is never a left operand; None when hazard scoring is off."""


def result_to_json(result: InterventionResult) -> dict[str, Any]:
    """Serialise one result; the hazard keys appear only when scored.

    With ``--hazard-heads`` off the dict is exactly what earlier versions
    wrote, so banked JSONs stay byte for byte reproducible.
    """
    out = asdict(result)
    for key in ("hazard", "hazard_paired"):
        if out[key] is None:
            del out[key]
    return out


# ---------------------------------------------------------------------------
# Hazard heads at landmark rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandmarkRiskTable:
    """Hazard-head risk at every landmark row of one intervention pass."""

    event_names: list[str]
    horizons: list[float]
    subject_ids: np.ndarray
    visit_ids: np.ndarray
    time_hours: np.ndarray
    risk: np.ndarray
    """``(n_rows, n_events, n_horizons)`` P(event within horizon)."""

    @property
    def n_rows(self) -> int:
        """Number of landmark rows."""
        return int(self.subject_ids.shape[0])

    def same_rows(self, other: "LandmarkRiskTable") -> bool:
        """Whether both tables hold the same (subject, visit, time) rows in order."""
        return (
            np.array_equal(self.subject_ids, other.subject_ids)
            and np.array_equal(self.visit_ids, other.visit_ids)
            and np.array_equal(self.time_hours, other.time_hours)
        )


class HazardLandmarkScorer:
    """Read P(event within h) from the hazard heads at landmark rows.

    Landmark rows are chosen with the alert protocol's own mask
    (:func:`odyssey.inference.alerts._landmark_mask`), state threaded
    across chunks, so the rows are the ones Tables 12 to 14 score. Only
    the landmark rows are kept: subject, visit, time and one risk per
    event and horizon.
    """

    def __init__(
        self,
        event_heads: EventHazardHeads,
        alerts: Sequence[AlertEvent],
        visit_start: dict[tuple[int, int], float],
        *,
        landmark_hours: float = 4.0,
        horizons: Sequence[float] = HORIZONS_HOURS,
    ) -> None:
        """Bind the heads, the events to read and the visit envelope."""
        missing = [a.name for a in alerts if a.name not in event_heads.event_names]
        if missing:
            raise ValueError(f"no hazard head for alert events {missing}")
        self.event_heads = event_heads
        self.event_names = [a.name for a in alerts]
        self.head_index = [event_heads.event_names.index(n) for n in self.event_names]
        self.visit_start = visit_start
        self.landmark_hours = landmark_hours
        self.horizons = [float(h) for h in horizons]
        self._state: LandmarkState | None = None
        self._sids: list[np.ndarray] = []
        self._vids: list[np.ndarray] = []
        self._times: list[np.ndarray] = []
        self._risk: list[np.ndarray] = []

    def add_chunk(self, chunk: StreamingChunk, features: torch.Tensor) -> None:
        """Score this chunk's landmark positions from ``features``."""
        sids, vids = chunk.subject_ids, chunk.visit_ids
        times = chunk.batch.aux.time_stamps
        starts = position_visit_starts(sids, vids, times, self.visit_start)
        keep, self._state = _landmark_mask(
            times, sids, vids, self.landmark_hours, starts, state=self._state
        )
        if not keep.any():
            return
        keep = keep.to(features.device)
        logits = self.event_heads(features[keep])[:, self.head_index]
        risk = torch.stack(
            [
                probability_within(logits, self.event_heads.edges, h)
                for h in self.horizons
            ],
            dim=-1,
        )
        self._sids.append(sids[keep].cpu().numpy().astype(np.int64))
        self._vids.append(vids[keep].cpu().numpy().astype(np.int64))
        self._times.append(times[keep].cpu().numpy().astype(np.float64))
        self._risk.append(risk.float().cpu().numpy())

    def table(self) -> LandmarkRiskTable:
        """Concatenate everything accumulated so far."""
        n_events, n_h = len(self.event_names), len(self.horizons)
        return LandmarkRiskTable(
            event_names=list(self.event_names),
            horizons=list(self.horizons),
            subject_ids=(
                np.concatenate(self._sids) if self._sids else np.zeros(0, np.int64)
            ),
            visit_ids=(
                np.concatenate(self._vids) if self._vids else np.zeros(0, np.int64)
            ),
            time_hours=(
                np.concatenate(self._times) if self._times else np.zeros(0, np.float64)
            ),
            risk=(
                np.concatenate(self._risk)
                if self._risk
                else np.zeros((0, n_events, n_h), np.float32)
            ),
        )


def landmark_labels(
    table: LandmarkRiskTable, times: dict[str, EventTimes]
) -> np.ndarray:
    """``(n_rows, n_events, n_horizons)`` int8 labels: 1, 0, or -1.

    -1 marks a row the alert protocol does not score for that horizon:
    not at risk (the event already happened) or censored (follow-up ends
    before ``t + h``). The rule is :func:`odyssey.inference.alerts.outcome_at_horizon`,
    vectorised per event.
    """
    n = table.n_rows
    labels = np.full((n, len(table.event_names), len(table.horizons)), -1, np.int8)
    t = table.time_hours
    for e, name in enumerate(table.event_names):
        ev = times[name]
        vids = np.full(n, -1, np.int64) if ev.subject_scoped else table.visit_ids
        keys = zip(table.subject_ids.tolist(), vids.tolist())
        onset = np.empty(n, np.float64)
        censor = np.empty(n, np.float64)
        for i, key in enumerate(keys):
            o = ev.onset.get(key)
            c = ev.censor.get(key)
            onset[i] = np.inf if o is None else o
            censor[i] = -np.inf if c is None else c
        at_risk = ~(onset <= t)
        for j, h in enumerate(table.horizons):
            positive = at_risk & (onset <= t + h)
            negative = at_risk & ~positive & (censor >= t + h)
            labels[positive, e, j] = 1
            labels[negative, e, j] = 0
    return labels


def _horizon_key(table: LandmarkRiskTable, j: int) -> str:
    return f"{table.horizons[j]:g}h"


def hazard_summary(table: LandmarkRiskTable, labels: np.ndarray) -> dict[str, Any]:
    """Per event and horizon: AUROC, mean risk and the row counts."""
    out: dict[str, Any] = {}
    for e, name in enumerate(table.event_names):
        out[name] = {}
        for j in range(len(table.horizons)):
            y = labels[:, e, j]
            ok = y >= 0
            p = table.risk[ok, e, j].astype(np.float64)
            y_ok = y[ok]
            two_class = ok.any() and y_ok.min() != y_ok.max()
            out[name][_horizon_key(table, j)] = {
                "auroc": float(roc_auc_score(y_ok, p)) if two_class else None,
                "mean_risk": float(p.mean()) if ok.any() else None,
                "n_at_risk": int(ok.sum()),
                "n_positive": int(y_ok.sum()),
                "n_censored": int((~ok).sum()),
            }
    return out


@dataclass(frozen=True)
class PairedMeanDelta:
    """Mean over rows of ``a - b`` with a subject-clustered bootstrap interval."""

    point: float
    ci_low: float
    ci_high: float
    n_rows: int
    n_subjects: int

    @property
    def separated(self) -> bool:
        """Whether the interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def subject_bootstrap_means(
    diff: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    block: int = 64,
) -> np.ndarray:
    """``(n_boot,)`` resampled means of ``diff``, drawing whole subjects.

    Rows are summed per subject once; each resample draws ``n_subjects``
    subjects with replacement and divides the drawn sums by the drawn
    counts, so a subject's rows always move together. Draws are made in
    blocks to bound memory.
    """
    diff = np.asarray(diff, dtype=np.float64)
    _, inverse = np.unique(np.asarray(subject_ids), return_inverse=True)
    n_subjects = int(inverse.max()) + 1 if inverse.size else 0
    if n_subjects == 0:
        return np.full(n_boot, np.nan)
    sums = np.bincount(inverse, weights=diff, minlength=n_subjects)
    counts = np.bincount(inverse, minlength=n_subjects).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, block):
        k = min(block, n_boot - start)
        drawn = rng.integers(0, n_subjects, size=(k, n_subjects))
        out[start : start + k] = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return out


def paired_mean_delta(
    a: np.ndarray,
    b: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> PairedMeanDelta:
    """Subject-clustered paired bootstrap of ``mean(a) - mean(b)`` on the same rows."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    subjects = np.asarray(subject_ids)
    n_subjects = int(np.unique(subjects).shape[0])
    if diff.size == 0:
        return PairedMeanDelta(float("nan"), float("nan"), float("nan"), 0, 0)
    boots = subject_bootstrap_means(diff, subjects, n_boot=n_boot, seed=seed)
    return PairedMeanDelta(
        float(diff.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        int(diff.size),
        n_subjects,
    )


def _auroc_delta_json(d: BootstrapAUROCDelta | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {
        "point": d.point_estimate,
        "ci_low": d.ci_low,
        "ci_high": d.ci_high,
        "n_boot_used": d.n_boot_used,
        "n_boot_skipped": d.n_boot_skipped,
        "separated": d.excludes_zero(),
    }


def hazard_paired_summary(
    left: LandmarkRiskTable,
    right: LandmarkRiskTable,
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Per event and horizon: paired bootstrap of AUROC and mean-risk differences.

    Both tables must hold the same rows (checked). AUROC differences use
    :func:`odyssey.inference.uncertainty.bootstrap_auroc_delta`, the
    helper behind ``scripts/alerts_cis.py``; mean-risk differences use
    :func:`paired_mean_delta`. Both draw whole subjects.
    """
    if not left.same_rows(right):
        raise ValueError("paired hazard scoring needs identical landmark rows")
    out: dict[str, Any] = {}
    for e, name in enumerate(left.event_names):
        out[name] = {}
        for j in range(len(left.horizons)):
            y = labels[:, e, j]
            ok = y >= 0
            y_ok = y[ok].astype(np.float64)
            p_left = left.risk[ok, e, j].astype(np.float64)
            p_right = right.risk[ok, e, j].astype(np.float64)
            subjects = left.subject_ids[ok]
            auroc = (
                bootstrap_auroc_delta(
                    y_ok, p_left, p_right, subjects, n_boot=n_boot, seed=seed
                )
                if ok.any()
                else None
            )
            mean = paired_mean_delta(
                p_left, p_right, subjects, n_boot=n_boot, seed=seed
            )
            out[name][_horizon_key(left, j)] = {
                "auroc": _auroc_delta_json(auroc),
                # None, like auroc, when the cell has no scoreable row.
                "mean_risk": (
                    {
                        "point": mean.point,
                        "ci_low": mean.ci_low,
                        "ci_high": mean.ci_high,
                        "separated": mean.separated,
                    }
                    if mean.n_rows
                    else None
                ),
                "n_at_risk": int(ok.sum()),
                "n_subjects": mean.n_subjects,
            }
    return out


def score_hazard_landmarks(
    tables: dict[str, LandmarkRiskTable],
    times: dict[str, EventTimes],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return ``(hazard by mode, hazard_paired by mode)`` for the JSON output.

    Labels are computed once from the first table; every other table must
    hold the same rows, which the shared sampler and landmark mask
    guarantee. Pairs come from :data:`HAZARD_PAIRS`, limited to the modes
    that were run, and land on the left mode's entry.
    """
    if not tables:
        return {}, {}
    first = next(iter(tables.values()))
    for mode, table in tables.items():
        if not table.same_rows(first):
            raise ValueError(f"mode {mode!r} scored different landmark rows")
    labels = landmark_labels(first, times)
    per_mode = {mode: hazard_summary(table, labels) for mode, table in tables.items()}
    paired: dict[str, dict[str, Any]] = {mode: {} for mode in tables}
    for left, right in HAZARD_PAIRS:
        if left in tables and right in tables:
            paired[left][f"{left}_minus_{right}"] = hazard_paired_summary(
                tables[left], tables[right], labels, n_boot=n_boot, seed=seed
            )
    return per_mode, paired


def _chunk_intervention(
    chunk: StreamingChunk,
    mode: str,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
    device: str,
    rng: torch.Generator,
    uncertain_band: float | None = None,
) -> BottleneckIntervention | None:
    """Build the per-position intervention for one chunk, or None."""
    if mode == "none":
        return None
    if mode == "zero_known":
        return BottleneckIntervention(zero_known=True)
    if mode == "zero_unknown":
        return BottleneckIntervention(zero_unknown=True)
    if mode == "zero_residual":
        return BottleneckIntervention(zero_residual=True)

    labels, observed = position_running_labels(
        chunk,
        concept_labels,
        concept_mask,
        concept_first_times,
        supervision=supervision,
        num_concepts=num_concepts,
    )
    if mode == "truth":
        values = labels
    elif mode in ("flip", "flip_gated"):
        values = 1.0 - labels
    elif mode == "random":
        values = (torch.rand(labels.shape, generator=rng) < 0.5).float()
    else:
        raise ValueError(f"unknown intervention mode: {mode!r}")
    return BottleneckIntervention(
        probs=values.to(device),
        probs_mask=observed.bool().to(device),
        uncertain_band=uncertain_band,
    )


def run_streaming_intervention(  # noqa: PLR0912, PLR0915 -- one linear scoring pass
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    mode: str,
    concept_first_times: ConceptLabelDict | None = None,
    supervision: ConceptSupervision = "stay",
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: int | None = None,
    seed: int = 0,
    uncertain_band: float | None = None,
    per_subject_out: dict[int, list[int]] | None = None,
    calibration_gammas: torch.Tensor | None = None,
    calibrated_tau: float | None = None,
    concept_names: Sequence[str] | None = None,
    hazard_scorer: HazardLandmarkScorer | None = None,
) -> InterventionResult:
    """Score next-event prediction under one intervention mode.

    ``per_subject_out``, if given, accumulates ``{subject_id: [top1_hits,
    n_predictions]}`` over the pass -- the raw material for a PAIRED
    subject-clustered bootstrap on a truth-vs-flip accuracy delta
    (scripts/intervention_cis.py), which the aggregate numbers alone
    cannot support.

    ``hazard_scorer``, if given, also reads the per-event hazard heads at
    the alert protocol's landmark rows from the intervened features (see
    :class:`HazardLandmarkScorer`); it changes nothing in the returned
    next-event numbers.

    The identical streaming pass as
    :func:`~odyssey.inference.run_inference.run_streaming_inference`
    (same sampler, same state carrying), with the bottleneck edited per
    :data:`INTERVENTION_MODES`. ``concept_first_times`` (from
    :func:`~odyssey.training.data.build_visit_concept_first_times` or
    its stay-scoped twin) turns the retrospective labels into running
    ones, see :func:`_position_labels`; without it the retrospective
    labels are injected as-is at every position, which is only valid for
    concepts that are constant across the sequence. Deterministic for a
    given ``seed`` (which only the ``random`` mode consumes).
    """
    if mode not in INTERVENTION_MODES:
        raise ValueError(
            f"unknown intervention mode {mode!r}; known: {INTERVENTION_MODES}"
        )
    if mode in CALIBRATED_MODES and calibration_gammas is None:
        raise ValueError(
            f"mode {mode!r} needs calibration_gammas (see "
            "odyssey.inference.concept_attribution.calibrated_gammas)"
        )
    if concept_first_times is None:
        if mode in ("truth", "flip", "flip_gated", *CALIBRATED_MODES):
            logger.warning(
                "[interventions] mode %r without concept_first_times: injecting "
                "retrospective labels at every position (not running labels)",
                mode,
            )
        concept_first_times = {}
    model.eval()
    num_concepts = model.bottleneck.num_concepts
    patients = iter_patient_sequences(
        events_binned,
        vocab,
        max_seq_len=max_seq_len,
    )
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    rng = torch.Generator().manual_seed(seed)
    type_lookup = _build_type_lookup(vocab, device)

    n = 0
    top1_hits = 0
    loss_sum = 0.0
    n_intervened = 0
    displacement_sum = 0.0
    n_replaced_entries = 0
    per_concept_n = torch.zeros(num_concepts, dtype=torch.long)
    per_concept_disp = torch.zeros(num_concepts, dtype=torch.float64)
    type_n: dict[int, int] = {}
    type_hits: dict[int, int] = {}

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            intervention = (
                None
                if mode in CALIBRATED_MODES  # built below, from the model's own probs
                else _chunk_intervention(
                    chunk,
                    mode,
                    concept_labels,
                    concept_mask,
                    concept_first_times,
                    supervision=supervision,
                    num_concepts=num_concepts,
                    device=device,
                    rng=rng,
                    uncertain_band=uncertain_band,
                )
            )
            if mode in CALIBRATED_MODES:
                # The backbone runs once; the bottleneck runs twice on its
                # hidden states -- first un-intervened to read the model's
                # own probabilities (the calibrated step is RELATIVE to
                # them), then with the calibrated absolute values. No
                # uncertain band: calibration replaces it as the equalizer.
                assert calibration_gammas is not None  # noqa: S101 -- checked above
                hidden, state = model.backbone(
                    chunk.batch, state=state, reset_mask=chunk.reset_mask
                )
                own_probs = model.bottleneck(hidden).concept_probs
                labels, observed = position_running_labels(
                    chunk,
                    concept_labels,
                    concept_mask,
                    concept_first_times,
                    supervision=supervision,
                    num_concepts=num_concepts,
                )
                pole = labels if mode == "truth_calibrated" else 1.0 - labels
                # calibration_gammas is derived from the model's own LM-head
                # weights, so it lives on the model's device, while `pole`
                # comes from the running labels on CPU. Match BOTH device and
                # dtype: `.to(dtype)` alone silently worked in CPU-only tests
                # (where the two already agree) and raised a device mismatch
                # on every GPU run, so no calibrated mode had ever completed.
                offsets = (2.0 * pole - 1.0) * calibration_gammas.to(
                    device=pole.device, dtype=pole.dtype
                )
                values = (own_probs + offsets.to(device)).clamp(0.0, 1.0)
                intervention = BottleneckIntervention(
                    probs=values, probs_mask=observed.bool().to(device)
                )
                bottleneck_out = model.bottleneck(hidden, intervention=intervention)
                logits = model.lm_head(bottleneck_out.bottleneck)
            elif mode == "flip_gated":
                # Two forwards from the SAME input state: the intervention
                # edits only the post-backbone bottleneck mixing, so both
                # calls produce identical hidden states and new_state; the
                # gate then keeps only the flip's suppressive logit changes.
                state_in = state
                base_logits, bottleneck_out, state = model(
                    chunk.batch, state=state_in, reset_mask=chunk.reset_mask
                )
                flip_logits, flip_out, _ = model(
                    chunk.batch,
                    state=state_in,
                    reset_mask=chunk.reset_mask,
                    intervention=intervention,
                )
                logits = base_logits + torch.clamp_max(flip_logits - base_logits, 0.0)
                # The gate has no hazard analogue: the heads read the
                # flip-intervened features, as in plain flip mode.
                hazard_features = flip_out.bottleneck
            else:
                logits, bottleneck_out, state = model(
                    chunk.batch,
                    state=state,
                    reset_mask=chunk.reset_mask,
                    intervention=intervention,
                )
            if mode != "flip_gated":
                hazard_features = bottleneck_out.bottleneck
            if hazard_scorer is not None:
                hazard_scorer.add_chunk(chunk, hazard_features)
            real = chunk.real_mask
            if intervention is not None and intervention.probs is not None:
                own = bottleneck_out.concept_probs
                applied = intervention_apply_mask(intervention, own)
                if applied is None:
                    applied = torch.ones_like(own, dtype=torch.bool)
                input_real = chunk.subject_ids != NO_SUBJECT
                applied = applied & input_real.unsqueeze(-1)
                n_intervened += int(applied.any(dim=-1).sum().item())
                n_replaced_entries += int(applied.sum().item())
                abs_diff = (intervention.probs.expand_as(own) - own).abs()
                displacement_sum += float(abs_diff[applied].sum().item())
                lead_dims = tuple(range(applied.dim() - 1))
                per_concept_n += applied.sum(dim=lead_dims).long().cpu()
                per_concept_disp += (
                    (abs_diff * applied).sum(dim=lead_dims).double().cpu()
                )
            if not real.any():
                continue
            real_logits = logits[real]
            real_targets = chunk.targets[real]
            n += int(real_targets.shape[0])
            preds = real_logits.argmax(dim=-1)
            hits = preds == real_targets
            top1_hits += int(hits.sum().item())
            if per_subject_out is not None:
                sids = chunk.subject_ids[real]
                for sid in torch.unique(sids).tolist():
                    sel = sids == sid
                    entry = per_subject_out.setdefault(int(sid), [0, 0])
                    entry[0] += int(hits[sel].sum().item())
                    entry[1] += int(sel.sum().item())
            loss_sum += float(
                F.cross_entropy(
                    real_logits, real_targets, ignore_index=PAD_ID, reduction="sum"
                ).item()
            )
            target_types = type_lookup[real_targets]
            for type_id in torch.unique(target_types).tolist():
                sel = target_types == type_id
                type_n[type_id] = type_n.get(type_id, 0) + int(sel.sum().item())
                type_hits[type_id] = type_hits.get(type_id, 0) + int(
                    hits[sel].sum().item()
                )

    names = (
        list(concept_names)
        if concept_names is not None
        else [f"concept_{i}" for i in range(num_concepts)]
    )
    if len(names) != num_concepts:
        raise ValueError(f"{len(names)} concept names for {num_concepts} concepts")
    by_concept_n: dict[str, int] | None = None
    by_concept_disp: dict[str, float] | None = None
    if n_replaced_entries:
        by_concept_n = {
            names[i]: int(per_concept_n[i].item()) for i in range(num_concepts)
        }
        by_concept_disp = {
            names[i]: (
                float(per_concept_disp[i].item() / per_concept_n[i].item())
                if per_concept_n[i]
                else float("nan")
            )
            for i in range(num_concepts)
        }

    return InterventionResult(
        mode=mode,
        n_predictions=n,
        top1_accuracy=top1_hits / n if n else float("nan"),
        mean_task_loss=loss_sum / n if n else float("nan"),
        top1_by_code_type={
            _CODE_TYPE_NAMES[tid]: type_hits[tid] / type_n[tid]
            for tid in sorted(type_n)
            if tid in _CODE_TYPE_NAMES
        },
        n_by_code_type={
            _CODE_TYPE_NAMES[tid]: type_n[tid]
            for tid in sorted(type_n)
            if tid in _CODE_TYPE_NAMES
        },
        n_intervened_positions=n_intervened,
        uncertain_band=None if mode in CALIBRATED_MODES else uncertain_band,
        mean_abs_displacement=(
            displacement_sum / n_replaced_entries if n_replaced_entries else None
        ),
        calibrated_tau=calibrated_tau if mode in CALIBRATED_MODES else None,
        n_replaced_by_concept=by_concept_n,
        mean_abs_displacement_by_concept=by_concept_disp,
    )


def evaluate_interventions(
    run_dir: str | Path,
    held_out_shard_dir: str | Path,
    *,
    modes: Sequence[str] = INTERVENTION_MODES,
    max_shards: int | None = None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    uncertain_band: float | None = None,
    per_subject_out: dict[str, dict[int, list[int]]] | None = None,
    calibrated_tau: float = 1.0,
    hazard_heads: bool = False,
    hazard_boot: int = 1000,
    landmark_hours: float = 4.0,
    horizons: Sequence[float] = HORIZONS_HOURS,
) -> list[InterventionResult]:
    """End-to-end: load a trained run, score every intervention mode.

    ``per_subject_out``, if given, is filled as ``{mode: {subject_id:
    [top1_hits, n_predictions]}}`` (see
    :func:`run_streaming_intervention`).

    ``hazard_heads`` adds the hazard-head readout at landmark rows to
    every mode (``InterventionResult.hazard``) and the
    :data:`HAZARD_PAIRS` paired bootstrap differences with ``hazard_boot``
    resamples (``InterventionResult.hazard_paired``). Events are the
    run's landmark alert events that have a hazard head; outcomes and
    landmark rows follow :mod:`odyssey.inference.alerts` exactly. The
    bootstrap seed is ``seed``.

    Data preparation matches
    :func:`~odyssey.inference.run_inference.evaluate_run` exactly (same
    normalization, binning, and label scoping from the run's own
    config), so the ``none`` mode is directly comparable to the standard
    evaluation and every other mode is directly comparable to ``none``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            "this evaluation needs a concept bottleneck; the run's model_kind is "
            f"{getattr(config, 'model_kind', 'bottleneck')!r}"
        )
    if hazard_heads and model.event_heads is None:
        raise ValueError(
            "hazard scoring needs a run trained with per-event hazard heads "
            "(event_hazards); this run has none"
        )
    if getattr(config, "backbone", "hybrid") == "transformer":
        # The TBTT stream below is the same one the transformer trained on:
        # its ``state`` is an inert sentinel and every chunk is its own
        # context window, so a stateless backbone sees exactly the context
        # it was optimized for. What differs from run_inference/alerts is
        # only that those hand the transformer whole-patient context
        # (PackedContextSampler); the lever test keeps training's view.
        logger.info(
            "[interventions] backbone='transformer': chunks of %d tokens are "
            "independent context windows, as in training",
            chunk_size,
        )

    logger.info("[interventions] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events,
        enabled=getattr(config, "normalize_medications", False),
        source=getattr(config, "source", "mimic_iv"),
    )
    raw_events = maybe_history_recap(
        raw_events, enabled=getattr(config, "history_recap", False)
    )
    source = getattr(config, "source", "mimic_iv")
    activate_sidecars(held_out_shard_dir)
    # The run's own concept set, not today's: a checkpoint trained against
    # an older registry has fewer bottleneck slots than concepts_for_source
    # now returns, and the two are zipped together below.
    concepts = resolve_concepts_for_run(
        str(run_dir), source, getattr(config, "task_set", "v1")
    )
    events_binned = add_value_tokens(raw_events, binner, source=source)

    supervision: ConceptSupervision = getattr(config, "concept_supervision", "stay")
    concept_labels, concept_mask, concept_first_times = _concept_labels(
        raw_events, concepts, supervision
    )
    hazard_targets = (
        _hazard_targets(model, config, raw_events, source=source)
        if hazard_heads
        else None
    )
    del raw_events

    calibration_gammas: torch.Tensor | None = None
    gamma_by_name: dict[str, float] | None = None
    if any(m in CALIBRATED_MODES for m in modes):
        calibration_gammas, gamma_by_name = _calibration(
            model,
            events_binned,
            vocab,
            concept_names=[c.name for c in concepts],
            calibrated_tau=calibrated_tau,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
        )

    results = []
    tables: dict[str, LandmarkRiskTable] = {}
    for mode in modes:
        logger.info("[interventions] scoring mode %r", mode)
        mode_subjects: dict[int, list[int]] | None = None
        if per_subject_out is not None:
            mode_subjects = per_subject_out.setdefault(mode, {})
        scorer: HazardLandmarkScorer | None = None
        if hazard_targets is not None:
            scorer = HazardLandmarkScorer(
                hazard_targets.event_heads,
                hazard_targets.alerts,
                hazard_targets.visit_start,
                landmark_hours=landmark_hours,
                horizons=horizons,
            )
        result = run_streaming_intervention(
            model,
            events_binned,
            vocab,
            concept_labels,
            concept_mask,
            mode=mode,
            concept_first_times=concept_first_times,
            supervision=supervision,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
            seed=seed,
            uncertain_band=uncertain_band,
            per_subject_out=mode_subjects,
            calibration_gammas=calibration_gammas,
            calibrated_tau=calibrated_tau,
            concept_names=[c.name for c in concepts],
            hazard_scorer=scorer,
        )
        if mode in CALIBRATED_MODES:
            result = replace(result, calibration_gamma=gamma_by_name)
        if scorer is not None:
            tables[mode] = scorer.table()
        results.append(result)
        baseline = results[0]
        latest = results[-1]
        logger.info(
            "[interventions] %s: top1 %.4f (delta vs none %+0.4f), loss %.4f",
            mode,
            latest.top1_accuracy,
            latest.top1_accuracy - baseline.top1_accuracy,
            latest.mean_task_loss,
        )
    if hazard_targets is not None:
        logger.info(
            "[interventions] paired hazard bootstrap: %d resamples over %d landmark rows",
            hazard_boot,
            next(iter(tables.values())).n_rows if tables else 0,
        )
        per_mode, paired = score_hazard_landmarks(
            tables, hazard_targets.times, n_boot=hazard_boot, seed=seed
        )
        results = [
            replace(r, hazard=per_mode[r.mode], hazard_paired=paired[r.mode])
            for r in results
        ]
        _log_hazard(results)
    return results


def _concept_labels(
    raw_events: pl.DataFrame, concepts: Sequence[Any], supervision: ConceptSupervision
) -> tuple[ConceptLabelDict, ConceptLabelDict, ConceptLabelDict]:
    """``(labels, mask, first_times)`` under the run's own label scoping."""
    if supervision == "visit":
        visit_labels, visit_mask = build_visit_concept_label_dicts(raw_events, concepts)
        first = build_visit_concept_first_times(raw_events, concepts)
        return visit_labels, visit_mask, first
    stay_labels, stay_mask = build_concept_label_dicts(raw_events, concepts)
    return stay_labels, stay_mask, build_concept_first_times(raw_events, concepts)


def _calibration(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    concept_names: Sequence[str],
    calibrated_tau: float,
    num_lanes: int,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-concept calibrated steps for the *_calibrated modes, with names."""
    # Only a bottleneck whose per-unit displacement is data dependent
    # needs the estimation pass. The decomposition's displacement is a
    # parameter, so asking for directions there would be a forward
    # pass over the whole split to recover something already stored.
    directions = (
        mean_concept_directions(
            model,
            events_binned,
            vocab,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
        )
        if getattr(model.bottleneck, "needs_calibration_directions", True)
        else None
    )
    calibration_gammas = calibrated_gammas(model, directions, tau=calibrated_tau)
    gamma_by_name = {
        name: float(g)
        for name, g in zip(concept_names, calibration_gammas.tolist(), strict=True)
    }
    logger.info(
        "[interventions] calibrated gammas (tau=%.3g): %s",
        calibrated_tau,
        {k: round(v, 4) for k, v in gamma_by_name.items()},
    )
    return calibration_gammas, gamma_by_name


@dataclass(frozen=True)
class _HazardTargets:
    """What the hazard readout needs from the held-out split, built once."""

    event_heads: EventHazardHeads
    alerts: list[AlertEvent]
    times: dict[str, EventTimes]
    visit_start: dict[tuple[int, int], float]


def _hazard_targets(
    model: ConceptBottleneckSequenceModel,
    config: object,
    raw_events: pl.DataFrame,
    *,
    source: str,
) -> _HazardTargets:
    """Alert events with a head, their onset tables and the visit envelope.

    The events are the run's landmark alert events as
    :func:`odyssey.inference.alerts.evaluate_alerts` scores them,
    restricted to those the model has a hazard head for.
    """
    if model.event_heads is None:
        raise ValueError(
            "hazard scoring needs a run trained with per-event hazard heads "
            "(event_hazards); this run has none"
        )
    task_set = getattr(config, "task_set", "v1")
    head_names = set(model.event_heads.event_names)
    alerts = [
        a
        for a in alert_events_for(task_set, source=source)
        if not a.next_visit and a.name in head_names
    ]
    if not alerts:
        raise ValueError(
            f"task_set {task_set!r} has no landmark alert event with a hazard head"
        )
    logger.info("[interventions] hazard heads for %s", [a.name for a in alerts])
    return _HazardTargets(
        event_heads=model.event_heads,
        alerts=alerts,
        times=all_event_times(raw_events, alerts, source, task_set=task_set),
        visit_start=_visit_starts(raw_events),
    )


def _log_hazard(results: Sequence[InterventionResult]) -> None:
    for r in results:
        for event, by_h in (r.hazard or {}).items():
            cells = {
                h: (
                    None if c["auroc"] is None else round(c["auroc"], 4),
                    None if c["mean_risk"] is None else round(c["mean_risk"], 4),
                )
                for h, c in by_h.items()
            }
            logger.info("[interventions] %s hazard %s: %s", r.mode, event, cells)


def _main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename within --run-dir (default: checkpoint_best.pt).",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--modes", nargs="*", default=list(INTERVENTION_MODES))
    parser.add_argument(
        "--uncertain-band",
        type=float,
        default=None,
        help=(
            "Only inject truth/flip/random values where the model's own concept "
            "probability is within this distance of 0.5, so truth and flip "
            "displace it equally (a pure direction test)."
        ),
    )
    parser.add_argument(
        "--calibrated-tau",
        type=float,
        default=1.0,
        help=(
            "peak logit shift every concept's step is calibrated to in the "
            "truth_calibrated/flip_calibrated modes (gamma_i = tau / peak_i "
            "over the LM head weights); ignored unless a calibrated mode is "
            "in --modes."
        ),
    )
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--hazard-heads",
        action="store_true",
        help=(
            "also read the per-event hazard heads at the alert protocol's "
            "landmark rows under every mode (AUROC and mean P(event within h) "
            "per event and horizon) and attach paired subject-clustered "
            "bootstrap differences for truth-none, truth-flip and flip-none. "
            "Off by default: without it the output is unchanged."
        ),
    )
    parser.add_argument(
        "--hazard-boot",
        type=int,
        default=1000,
        help="bootstrap resamples for the paired hazard differences.",
    )
    parser.add_argument(
        "--dump-per-subject",
        action="store_true",
        help=(
            "also write <output-json stem>_per_subject.json with "
            "{mode: {subject_id: [top1_hits, n_predictions]}} -- the input "
            "scripts/intervention_cis.py needs for a paired subject-"
            "clustered CI on mode-vs-mode accuracy deltas."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "allow clobbering an existing --output-json file. Protocol-"
            "versioned science outputs are append-only by default -- a "
            "real, irreplaceable row-level dump was lost to a silent "
            "overwrite on 2026-08-22. Pass this only when re-running the "
            "same run/protocol intentionally."
        ),
    )
    args = parser.parse_args()

    out = Path(args.output_json)
    refuse_existing_output(out, overwrite=args.overwrite, kind="interventions")
    run_dir = Path(args.run_dir)
    per_subject: dict[str, dict[int, list[int]]] | None = (
        {} if args.dump_per_subject else None
    )
    results = evaluate_interventions(
        run_dir,
        args.held_out_shard_dir,
        modes=args.modes,
        max_shards=args.max_shards,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        uncertain_band=args.uncertain_band,
        per_subject_out=per_subject,
        calibrated_tau=args.calibrated_tau,
        hazard_heads=args.hazard_heads,
        hazard_boot=args.hazard_boot,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([result_to_json(r) for r in results], indent=2))
    logger.info("[interventions] wrote %d modes to %s", len(results), out)
    if per_subject is not None:
        ps_out = out.with_name(out.stem + "_per_subject.json")
        refuse_existing_output(
            ps_out, overwrite=args.overwrite, kind="interventions per-subject"
        )
        ps_out.write_text(json.dumps(per_subject))
        logger.info(
            "[interventions] wrote per-subject outcomes for %d modes to %s",
            len(per_subject),
            ps_out,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _main()


__all__ = [
    "HAZARD_PAIRS",
    "INTERVENTION_MODES",
    "HazardLandmarkScorer",
    "InterventionResult",
    "LandmarkRiskTable",
    "PairedMeanDelta",
    "evaluate_interventions",
    "hazard_paired_summary",
    "hazard_summary",
    "landmark_labels",
    "paired_mean_delta",
    "result_to_json",
    "run_streaming_intervention",
    "score_hazard_landmarks",
    "subject_bootstrap_means",
]
