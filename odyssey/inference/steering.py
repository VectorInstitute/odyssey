"""Concept steering as a clinician would test it: turn one dial, watch the risks.

The lever a clinician wants is not "does the model predict the recorded
next token better when told the true state" (that is the trust test of
:mod:`odyssey.inference.interventions`). It is: push the model along the
*sustained hypotension* direction and vasopressor risk should rise, ICU-transfer risk
should rise, and the events the model expects next should look like
hypotension; pull it the other way and they should fall. This module measures
exactly that, concept by concept, on held-out patients, with the same
sign-agreement summary the input-level counterfactuals of
:mod:`odyssey.inference.counterfactual` report, so the two levers are
directly comparable.

Three readouts per concept and direction, each paired against the
unsteered pass on the same positions:

* **respond** -- the concept's own activation ``k_c`` after the push
  (Steerling's respond objective). A dial that does not light its own
  concept is not connected to anything.
* **express** -- probability mass the next-event distribution puts on
  the concept's *lifted* tokens, the events over-represented where the
  concept holds (Steerling's express objective, in the vocabulary of
  the timeline; see :func:`lifted_token_sets`).
* **outcome risk** -- each hazard head's probability of the event
  within 8, 24 and 72 hours, and whether it moved the way a clinician
  expects (:data:`CLINICAL_EXPECTATIONS`).

What is Steerling's and what is ours
-----------------------------------
From Madsen et al. (2026), Section 6.2 and Table 36: the steering
direction is the unit-normalized concept embedding ``e_c = K_c / ||K_c||``
(their 6.2.1); it is added to the hidden state at every position at every
layer from ``L_inj`` onward, so the signal accumulates toward the
bottleneck (their Eq. 18); the strength is calibrated per concept as
``gamma = tau / peak(e_c)`` with ``peak(e_c) = max_y e_c . W_y`` (their
Eq. 19); amplification is ``gamma > 0``; suppression is ``gamma < 0``
plus a ReLU-gated logit mask ``l_v -> l_v - s . ReLU(W_v . e_c)`` (their
Eqs. 20-21); the express target is the lifted token set, lift
``P(token | c) / P(token)`` with a minimum-support filter (their 10.2.4).

The paper does not state ``tau`` for steering, ``L_inj``, the suppression
strength ``s``, or the size of the lifted set. Our defaults: ``tau = 1``
(one logit of peak shift, the value the trust test's calibrated modes
use), ``L_inj`` = the middle block, ``s = |gamma|``, and the top 25 lifted
tokens with at least 20 occurrences. Each is a CLI flag.

Two injection sites. ``stream`` is the paper's layer injection; because
our backbone is recurrent, it is the site where a push is carried forward
by the state the way a real change in the patient would be.
``bottleneck`` pushes the concept's own activation where the decomposition
sums it, by ``gamma / ||K_c||`` so the sum receives the same ``gamma e_c``,
but only at the current position; it isolates the output-side effect.

Only the decomposed bottleneck is supported: it is the design in which
"one unit of concept c" is a parameter rather than an estimate.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.alert_events import all_event_times, hazard_events_for
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import canonical_concept_name, concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import PackedLaneSampler, StreamingChunk, move_to_device
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.run_inference import load_run, refuse_existing_output
from odyssey.models.concept_bottleneck import BottleneckIntervention
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
)
from odyssey.models.steering import (
    _known_embedding,
    concept_alignment,
    steering_direction,
    steering_gamma,
    stream_injection,
    suppress_logits,
)
from odyssey.models.time_to_event import probability_within
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
    build_visit_concept_first_times,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.event_targets import EventTimeTables, event_hazard_targets
from odyssey.training.lifted_tokens import lifted_token_sets


logger = logging.getLogger(__name__)

SteerSite = Literal["bottleneck", "stream"]
Direction = Literal["amplify", "suppress"]
HORIZONS_HOURS: tuple[float, ...] = (8.0, 24.0, 72.0)

# ---------------------------------------------------------------------------
# What a clinician expects each dial to do to each outcome.
# ---------------------------------------------------------------------------

#: concept -> {event: +1 (risk should rise when the concept is pushed UP)
#: or -1 (should fall)}. Suppression reverses every sign. Only outcomes
#: with a direct physiological or care-pathway link are listed: an
#: expectation is a claim we are prepared to be wrong about in print, so
#: weak or indirect links are left out rather than padded in.
#:
#: The reasoning, in the order a clinician would give it:
#: * circulatory failure (sustained MAP hypotension, hypotension,
#:   elevated lactate, metabolic
#:   acidosis, oliguria) -> pressors, ICU, death, and kidney injury from
#:   hypoperfusion;
#: * sepsis and its screens (sepsis3, sirs, qsofa, fever) -> pressors,
#:   ICU, death; sirs, qsofa and fever also raise the chance the sepsis
#:   rule fires;
#: * respiratory failure (hypoxia, hypoxemic respiratory failure,
#:   sustained tachypnea) -> ICU and death;
#: * kidney injury by stage and its precursors (aki stages, oliguria,
#:   hyperkalemia) -> further AKI and death; stage 3 and hyperkalemia
#:   -> ICU;
#: * dangerous chemistry and blood counts (sodium, glucose, hemoglobin,
#:   platelets, INR) -> death and ICU;
#: * hypertension is the one dial expected to LOWER a risk: a hypertensive
#:   patient is not the one about to need a pressor.
CLINICAL_EXPECTATIONS: dict[str, dict[str, int]] = {
    "sustained_hypotension_map": {
        "vasopressor_start": +1,
        "icu_admission": +1,
        "death": +1,
        "acute_kidney_injury": +1,
    },
    "hypotension": {
        "vasopressor_start": +1,
        "icu_admission": +1,
        "death": +1,
        "acute_kidney_injury": +1,
    },
    "elevated_lactate": {"vasopressor_start": +1, "icu_admission": +1, "death": +1},
    "metabolic_acidosis": {
        "vasopressor_start": +1,
        "icu_admission": +1,
        "death": +1,
        "acute_kidney_injury": +1,
    },
    "on_vasopressors": {"icu_admission": +1, "death": +1},
    "sepsis3": {
        "vasopressor_start": +1,
        "icu_admission": +1,
        "death": +1,
        "acute_kidney_injury": +1,
    },
    "sirs": {"sepsis3": +1, "icu_admission": +1, "death": +1},
    "qsofa": {"sepsis3": +1, "vasopressor_start": +1, "icu_admission": +1, "death": +1},
    "fever": {"sepsis3": +1, "icu_admission": +1},
    "hypothermia": {"icu_admission": +1, "death": +1},
    "tachycardia": {"vasopressor_start": +1, "icu_admission": +1, "death": +1},
    "bradycardia": {"death": +1},
    "hypoxia": {"icu_admission": +1, "death": +1},
    "hypoxemic_respiratory_failure": {"icu_admission": +1, "death": +1},
    "sustained_tachypnea": {"icu_admission": +1, "death": +1},
    "acute_kidney_injury": {"acute_kidney_injury": +1, "death": +1},
    "aki_stage_2": {"acute_kidney_injury": +1, "death": +1},
    "aki_stage_3": {"acute_kidney_injury": +1, "icu_admission": +1, "death": +1},
    "oliguria": {"acute_kidney_injury": +1, "vasopressor_start": +1, "death": +1},
    "hyperkalemia": {"acute_kidney_injury": +1, "icu_admission": +1, "death": +1},
    "hypokalemia": {"death": +1},
    "hyponatremia": {"icu_admission": +1, "death": +1},
    "hypernatremia": {"icu_admission": +1, "death": +1},
    "hypoglycemia": {"icu_admission": +1, "death": +1},
    "hyperglycemia": {"icu_admission": +1},
    "anemia": {"icu_admission": +1, "death": +1},
    "thrombocytopenia": {"icu_admission": +1, "death": +1},
    "coagulopathy": {"icu_admission": +1, "death": +1},
    "hypertension": {"vasopressor_start": -1},
}


def expectations_for(concept: str, event_names: Sequence[str]) -> dict[str, int]:
    """Return the declared expectations for ``concept`` this model can score.

    Events the model has no head for (Sepsis-3 on eICU, say) are dropped:
    an expectation the model cannot be tested on is out of scope for that
    source, not a failure.
    """
    known = set(event_names)
    declared = CLINICAL_EXPECTATIONS.get(concept, {})
    return {ev: sign for ev, sign in declared.items() if ev in known}


# ---------------------------------------------------------------------------
# One streaming pass -> per-subject readouts.
# ---------------------------------------------------------------------------


@dataclass
class SubjectReadout:
    """Sums over one subject's real positions; divide by the counts for means."""

    n: int = 0
    concept_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """(num_concepts,) summed activations over all real positions."""
    lifted_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """(num_lifted_sets,) summed next-event probability on each lifted set."""
    risk: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """(num_events, num_horizons) summed P(event within horizon), at-risk only."""
    risk_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """(num_events,) how many positions were at risk of each event.

    A position at or after an event's onset is not at risk: asking "will
    vasopressors start?" of a patient already on them is not a question,
    and averaging it in is what made the sustained-hypotension (MAP)
    dial, then named shock, lower the start hazard.
    """

    def add(
        self,
        probs: torch.Tensor,
        mass: torch.Tensor,
        risk: torch.Tensor,
        at_risk: torch.Tensor,
    ) -> None:
        """Accumulate one batch of positions belonging to this subject.

        ``mass`` is ``(N, num_lifted_sets)``, ``risk`` is ``(N, E, H)`` and
        ``at_risk`` is ``(N, E)``; only at-risk positions count toward an
        event's risk sum and its denominator.
        """
        p = probs.sum(dim=0).double().cpu().numpy()
        gate = at_risk.to(risk.dtype).unsqueeze(-1)
        r = (risk * gate).sum(dim=0).double().cpu().numpy()
        rn = at_risk.sum(dim=0).double().cpu().numpy()
        m = mass.sum(dim=0).double().cpu().numpy()
        if self.n == 0:
            self.concept_probs, self.risk, self.risk_n, self.lifted_mass = p, r, rn, m
        else:
            self.concept_probs = self.concept_probs + p
            self.risk = self.risk + r
            self.risk_n = self.risk_n + rn
            self.lifted_mass = self.lifted_mass + m
        self.n += int(probs.shape[0])

    def risk_means(self) -> np.ndarray:
        """(num_events, num_horizons) mean at-risk risk; NaN where never at risk."""
        with np.errstate(divide="ignore", invalid="ignore"):
            means: np.ndarray = self.risk / self.risk_n[:, None]
        return means


@dataclass(frozen=True)
class SteeringPush:
    """One dial setting: which concept, how far, and where it is applied."""

    concept_index: int
    gamma: float
    """Signed strength along the unit direction ``e_c``; negative suppresses."""
    site: SteerSite
    layer_index: int | None = None
    suppress_strength: float | None = None
    """``s`` for the ReLU-gated logit mask, applied when ``gamma < 0``;
    ``None`` means ``|gamma|``."""


def _outcome_risk(
    model: ConceptBottleneckSequenceModel, features: torch.Tensor
) -> torch.Tensor:
    """``(..., num_events, len(HORIZONS_HOURS))`` P(event within horizon)."""
    heads = model.event_heads
    if heads is None:
        raise ValueError("the run has no per-event hazard heads; nothing to steer")
    logits = heads(features)
    return torch.stack(
        [probability_within(logits, heads.edges, h) for h in HORIZONS_HOURS], dim=-1
    )


def _forward_pushed(
    model: ConceptBottleneckSequenceModel,
    chunk: StreamingChunk,
    state: Any,
    push: SteeringPush | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """Return ``(logits, concept_probs, features, new_state)`` under ``push``."""
    if push is None:
        logits, out, new_state = model(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        return logits, out.concept_probs, out.bottleneck, new_state
    direction = steering_direction(model, push.concept_index)
    if push.site == "bottleneck":
        hidden, new_state = model.backbone(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        own = model.bottleneck(hidden).concept_probs
        # k_c moves by gamma / ||K_c|| so the sum receives exactly gamma e_c.
        step = push.gamma / float(_known_embedding(model, push.concept_index).norm())
        values = own.clone()
        values[..., push.concept_index] += step
        mask = torch.zeros_like(own, dtype=torch.bool)
        mask[..., push.concept_index] = True
        out = model.bottleneck(
            hidden, intervention=BottleneckIntervention(probs=values, probs_mask=mask)
        )
        logits = model.lm_head(out.bottleneck)
    else:
        assert push.layer_index is not None  # noqa: S101 -- validated by the caller
        with stream_injection(model.backbone, push.layer_index, push.gamma * direction):
            logits, out, new_state = model(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
    if push.gamma < 0:
        strength = (
            abs(push.gamma)
            if push.suppress_strength is None
            else push.suppress_strength
        )
        logits = suppress_logits(logits, concept_alignment(model, direction), strength)
    return logits, out.concept_probs, out.bottleneck, new_state


def run_steering_pass(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    push: SteeringPush | None,
    lifted_sets: Sequence[torch.Tensor],
    tables: EventTimeTables | None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: int | None = None,
) -> dict[int, SubjectReadout]:
    """Stream every held-out patient once under ``push`` (``None`` = unsteered).

    The same sampler and state carrying as every other scorer, so a pushed
    pass and the unsteered pass visit identical positions and their
    per-subject means are paired. ``tables`` supplies each event's onset so
    risk is read only where the patient is still at risk of it; ``None``
    treats every position as at risk.
    """
    model.eval()
    sampler = PackedLaneSampler(
        iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len),
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        reset_prob=0.0,
    )
    readouts: dict[int, SubjectReadout] = {}
    lifted = [ids.to(device) for ids in lifted_sets]
    state: Any = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = move_to_device(chunk, device)  # noqa: PLW2901
            logits, probs, features, state = _forward_pushed(model, chunk, state, push)
            real = chunk.real_mask
            if not real.any():
                continue
            probs_next = F.softmax(logits[real].float(), dim=-1)
            mass = torch.stack(
                [
                    probs_next[:, ids].sum(dim=-1)
                    if ids.numel()
                    else probs_next.new_zeros(probs_next.shape[0])
                    for ids in lifted
                ],
                dim=-1,
            )
            risk = _outcome_risk(model, features[real])
            at_risk = (
                event_hazard_targets(chunk, tables).at_risk[real]
                if tables is not None
                else torch.ones(risk.shape[:2], dtype=torch.bool, device=risk.device)
            )
            sids = chunk.subject_ids[real]
            probs_real = probs[real]
            for sid in torch.unique(sids).tolist():
                sel = sids == sid
                readouts.setdefault(int(sid), SubjectReadout()).add(
                    probs_real[sel], mass[sel], risk[sel], at_risk[sel]
                )
    return readouts


def token_descriptions(
    tokens: Sequence[str], metadata_dir: str | Path | None
) -> dict[str, str]:
    """Human-readable names for vocabulary tokens from MEDS ``codes.parquet``.

    A token is a MEDS code plus an optional value-bin suffix (``::Q3``);
    the code's ``description`` is looked up and the bin kept, so
    ``LAB//50813//mmol/L::Q5`` reads as ``Lactate (Q5)``. Tokens without a
    description, or when no metadata is given, map to themselves.
    """
    names: dict[str, str] = {t: t for t in tokens}
    if metadata_dir is None:
        return names
    path = Path(metadata_dir) / "codes.parquet"
    if not path.exists():
        logger.warning("[steering] no %s; tokens stay as codes", path)
        return names
    codes = pl.read_parquet(path)
    if "description" not in codes.columns:
        logger.warning(
            "[steering] %s has no description column; tokens stay as codes", path
        )
        return names
    lookup = dict(
        zip(codes["code"].to_list(), codes["description"].to_list(), strict=True)
    )
    for token in tokens:
        code, _, suffix = token.partition("::")
        description = lookup.get(code)
        if description:
            names[token] = f"{description} ({suffix})" if suffix else str(description)
    return names


# ---------------------------------------------------------------------------
# Summaries with paired, subject-clustered uncertainty.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedDelta:
    """Mean over subjects of (steered - unsteered), with a bootstrap interval."""

    point: float
    ci_low: float
    ci_high: float
    n_subjects: int

    @property
    def separated(self) -> bool:
        """Whether the interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_delta(
    steered: np.ndarray, baseline: np.ndarray, *, n_boot: int = 1000, seed: int = 0
) -> PairedDelta:
    """Subject-clustered percentile bootstrap of the paired mean difference."""
    diff = np.asarray(steered, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    diff = diff[np.isfinite(diff)]  # subjects never at risk of this event drop out
    n = int(diff.shape[0])
    if n == 0:
        return PairedDelta(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    return PairedDelta(
        float(diff.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
        n,
    )


def _subject_means(
    readouts: Mapping[int, SubjectReadout], subjects: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(concept_probs (S,k), lifted_mass (S,), risk (S,E,H))`` per-subject means."""
    probs = np.stack([readouts[s].concept_probs / readouts[s].n for s in subjects])
    mass = np.stack([readouts[s].lifted_mass / readouts[s].n for s in subjects])
    risk = np.stack([readouts[s].risk_means() for s in subjects])
    return probs, mass, risk


@dataclass
class OutcomeShift:
    """One (event, horizon) under one push, against the clinician's expectation."""

    event: str
    horizon_hours: float
    baseline_risk: float
    steered_risk: float
    delta: PairedDelta
    expected_sign: int | None
    agreement: float | None
    """Fraction of subjects whose mean risk moved in the expected direction."""

    @property
    def relative_change(self) -> float:
        """Steered risk over unsteered risk, the number a clinician reads."""
        if self.baseline_risk <= 0:
            return float("nan")
        return self.steered_risk / self.baseline_risk

    @property
    def as_expected(self) -> bool | None:
        """Whether the paired delta has the declared sign (``None`` if undeclared)."""
        if self.expected_sign is None:
            return None
        return self.delta.point * self.expected_sign > 0


@dataclass
class ConceptSteeringSummary:
    """Everything one dial did, in one direction."""

    concept: str
    direction: Direction
    gamma: float
    site: SteerSite
    n_subjects: int
    respond_baseline: float
    respond_steered: float
    respond_delta: PairedDelta
    express_baseline: float
    express_steered: float
    express_delta: PairedDelta
    outcomes: list[OutcomeShift] = field(default_factory=list)

    @property
    def sign_agreement(self) -> float | None:
        """Share of declared expectations whose paired delta has the expected sign."""
        verdicts = [o.as_expected for o in self.outcomes if o.as_expected is not None]
        if not verdicts:
            return None
        return sum(verdicts) / len(verdicts)


def summarize_push(
    baseline: Mapping[int, SubjectReadout],
    steered: Mapping[int, SubjectReadout],
    *,
    concept: str,
    concept_index: int,
    lifted_column: int = 0,
    direction: Direction,
    gamma: float,
    site: SteerSite,
    event_names: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> ConceptSteeringSummary:
    """Pair the two passes by subject and score respond, express and outcomes."""
    subjects = sorted(set(baseline) & set(steered))
    p0, m0, r0 = _subject_means(baseline, subjects)
    p1, m1, r1 = _subject_means(steered, subjects)
    flip = 1 if direction == "amplify" else -1
    expected = expectations_for(concept, event_names)
    outcomes: list[OutcomeShift] = []
    for e, event in enumerate(event_names):
        for h, horizon in enumerate(HORIZONS_HOURS):
            delta = paired_delta(r1[:, e, h], r0[:, e, h], n_boot=n_boot, seed=seed)
            declared = expected.get(event)
            sign = None if declared is None else declared * flip
            agreement = None
            finite = np.isfinite(r0[:, e, h]) & np.isfinite(r1[:, e, h])
            if sign is not None and finite.any():
                moved = (r1[finite, e, h] - r0[finite, e, h]) * sign
                agreement = float((moved > 0).mean())
            outcomes.append(
                OutcomeShift(
                    event=event,
                    horizon_hours=horizon,
                    baseline_risk=float(np.nanmean(r0[:, e, h])),
                    steered_risk=float(np.nanmean(r1[:, e, h])),
                    delta=delta,
                    expected_sign=sign,
                    agreement=agreement,
                )
            )
    return ConceptSteeringSummary(
        concept=concept,
        direction=direction,
        gamma=gamma,
        site=site,
        n_subjects=len(subjects),
        respond_baseline=float(p0[:, concept_index].mean()),
        respond_steered=float(p1[:, concept_index].mean()),
        respond_delta=paired_delta(
            p1[:, concept_index], p0[:, concept_index], n_boot=n_boot, seed=seed
        ),
        express_baseline=float(m0[:, lifted_column].mean()),
        express_steered=float(m1[:, lifted_column].mean()),
        express_delta=paired_delta(
            m1[:, lifted_column], m0[:, lifted_column], n_boot=n_boot, seed=seed
        ),
        outcomes=outcomes,
    )


def clinician_line(summary: ConceptSteeringSummary) -> str:
    """One readable line per dial: what moved, which way, whether it should have."""
    ratio = summary.express_steered / max(summary.express_baseline, 1e-9)
    parts = [
        f"{summary.concept} {'up' if summary.direction == 'amplify' else 'down'}: "
        f"k_c {summary.respond_baseline:.2f}->{summary.respond_steered:.2f}, "
        f"lifted mass x{ratio:.2f}"
    ]
    for o in summary.outcomes:
        if o.expected_sign is None or o.horizon_hours != 24.0:
            continue
        verdict = "as expected" if o.as_expected else "WRONG WAY"
        share = 100 * (o.agreement or 0.0)
        parts.append(
            f"{o.event}@24h x{o.relative_change:.2f} "
            f"[{o.delta.ci_low:+.4f},{o.delta.ci_high:+.4f}] {verdict} "
            f"({share:.0f}% of patients)"
        )
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


@dataclass
class SteeringPrepared:
    """Everything a steering run needs, loaded once."""

    model: ConceptBottleneckSequenceModel
    vocab: Vocabulary
    events_binned: pl.DataFrame
    concept_names: list[str]
    event_names: list[str]
    lifted: dict[int, list[int]]
    gammas: list[float]
    """Per-concept calibrated strengths, Eq. 19 on the unit directions."""
    supervision: ConceptSupervision
    tables: EventTimeTables | None
    """Event onsets on the held-out split, for the at-risk restriction."""
    token_names: dict[str, str]


def _labels_for(
    raw_events: pl.DataFrame, concepts: Sequence[Any], supervision: ConceptSupervision
) -> tuple[ConceptLabelDict, ConceptLabelDict, ConceptLabelDict]:
    if supervision == "visit":
        visit_labels, visit_mask = build_visit_concept_label_dicts(raw_events, concepts)
        first = build_visit_concept_first_times(raw_events, concepts)
        return visit_labels, visit_mask, first
    stay_labels, stay_mask = build_concept_label_dicts(raw_events, concepts)
    return stay_labels, stay_mask, build_concept_first_times(raw_events, concepts)


def prepare(
    run_dir: str | Path,
    held_out_shard_dir: str | Path,
    lift_shard_dir: str | Path,
    *,
    max_shards: int | None,
    lift_shards: int,
    tau: float,
    device: str,
    checkpoint_path: str | Path | None,
    num_lanes: int,
    chunk_size: int,
    metadata_dir: str | Path | None = None,
    min_share: float = 0.005,
    min_lift: float = 2.0,
) -> SteeringPrepared:
    """Load the run, bin the held-out split, and build the lifted token sets."""
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError("steering needs a concept-bottleneck run")
    if model.event_heads is None:
        raise ValueError("the run has no per-event hazard heads; nothing to steer")
    source = getattr(config, "source", "mimic_iv")
    supervision = cast(
        "ConceptSupervision", getattr(config, "concept_supervision", "stay")
    )
    concepts = concepts_for_source(source, task_set=getattr(config, "task_set", "v1"))
    concept_names = [c.name for c in concepts]

    def binned(
        shard_dir: str | Path, shards: int | None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        raw = load_meds_shards(shard_dir, max_shards=shards)
        raw = maybe_normalize(
            raw,
            enabled=getattr(config, "normalize_medications", False),
            source=source,
        )
        raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
        activate_sidecars(shard_dir)
        return raw, add_value_tokens(raw, binner, source=source)

    lift_raw, lift_binned = binned(lift_shard_dir, lift_shards)
    labels, mask, first_times = _labels_for(lift_raw, concepts, supervision)
    del lift_raw
    lifted = lifted_token_sets(
        iter_patient_sequences(lift_binned, vocab),
        vocab_size=len(vocab.token_to_id),
        num_concepts=model.bottleneck.num_concepts,
        concept_labels=labels,
        concept_mask=mask,
        concept_first_times=first_times,
        supervision=supervision,
        min_share=min_share,
        min_lift=min_lift,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
    )
    del lift_binned
    token_names = token_descriptions(
        sorted({vocab.id_to_token[i] for ids in lifted.values() for i in ids}),
        metadata_dir,
    )
    for c, ids in lifted.items():
        logger.info(
            "[steering] lifted tokens for %s: %s",
            concept_names[c],
            [token_names[vocab.id_to_token[i]] for i in ids[:8]],
        )

    held_raw, events_binned = binned(held_out_shard_dir, max_shards)
    tables: EventTimeTables | None = None
    if getattr(config, "event_hazards", False):
        alerts = hazard_events_for(
            config.task_set, config.auxiliary_event_names, source=source
        )
        names = [a.name for a in alerts]
        if names != list(model.event_heads.event_names):
            raise ValueError(
                f"event heads {list(model.event_heads.event_names)} do not match "
                f"the alert events {names}; the at-risk masks would be misaligned"
            )
        tables = EventTimeTables(
            all_event_times(held_raw, alerts, source, task_set=config.task_set),
            names,
        )
    del held_raw
    return SteeringPrepared(
        model=model,
        vocab=vocab,
        events_binned=events_binned,
        concept_names=concept_names,
        event_names=list(model.event_heads.event_names),
        lifted=lifted,
        gammas=[
            steering_gamma(model, steering_direction(model, c), tau=tau)
            for c in range(len(concept_names))
        ],
        supervision=supervision,
        tables=tables,
        token_names=token_names,
    )


def evaluate_steering(
    prepared: SteeringPrepared,
    *,
    concepts: Sequence[str] | None,
    site: SteerSite,
    layer_index: int | None,
    suppress_strength: float | None,
    num_lanes: int,
    chunk_size: int,
    device: str,
    n_boot: int,
) -> list[ConceptSteeringSummary]:
    """Unsteered pass once, then amplify and suppress each requested concept."""
    names = prepared.concept_names
    chosen = [
        canonical_concept_name(c)
        for c in (concepts or [n for n in names if n in CLINICAL_EXPECTATIONS])
    ]
    unknown = sorted(set(chosen) - set(names))
    if unknown:
        raise ValueError(f"concepts not in this run's registry: {unknown}")
    every_lifted = sorted({i for ids in prepared.lifted.values() for i in ids})
    indices = [names.index(c) for c in chosen]
    lifted_sets = [
        torch.tensor(prepared.lifted.get(c) or every_lifted, dtype=torch.long)
        for c in indices
    ]
    # One unsteered pass serves every dial: its readouts carry the mass on
    # each chosen concept's lifted set, so the pass is not repeated per
    # concept (a third of the card time on a full-split run).
    baseline = run_steering_pass(
        prepared.model,
        prepared.events_binned,
        prepared.vocab,
        push=None,
        lifted_sets=lifted_sets,
        tables=prepared.tables,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
    )
    summaries: list[ConceptSteeringSummary] = []
    for column, (concept, c) in enumerate(zip(chosen, indices, strict=True)):
        gamma = prepared.gammas[c]
        directions: tuple[tuple[Direction, float], ...] = (
            ("amplify", gamma),
            ("suppress", -gamma),
        )
        for direction, signed in directions:
            push = SteeringPush(
                concept_index=c,
                gamma=signed,
                site=site,
                layer_index=layer_index,
                suppress_strength=suppress_strength,
            )
            steered = run_steering_pass(
                prepared.model,
                prepared.events_binned,
                prepared.vocab,
                push=push,
                lifted_sets=lifted_sets,
                tables=prepared.tables,
                num_lanes=num_lanes,
                chunk_size=chunk_size,
                device=device,
            )
            summary = summarize_push(
                baseline,
                steered,
                concept=concept,
                concept_index=c,
                lifted_column=column,
                direction=direction,
                gamma=signed,
                site=site,
                event_names=prepared.event_names,
                n_boot=n_boot,
            )
            logger.info("[steering] %s", clinician_line(summary))
            summaries.append(summary)
    return summaries


def _to_json(summaries: Sequence[ConceptSteeringSummary]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in summaries:
        d = asdict(s)
        d["sign_agreement"] = s.sign_agreement
        for o, od in zip(s.outcomes, d["outcomes"], strict=True):
            od["relative_change"] = o.relative_change
            od["as_expected"] = o.as_expected
            od["separated"] = o.delta.separated
        out.append(d)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument(
        "--lift-shard-dir",
        required=True,
        help="training shards to build the lifted token sets from",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="registry names; default: every concept with a declared expectation",
    )
    parser.add_argument("--site", choices=("bottleneck", "stream"), default="stream")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="stream site: first block whose output is pushed (default: middle)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="peak logit shift for the Eq. 19 calibration (our default)",
    )
    parser.add_argument(
        "--suppress-strength",
        type=float,
        default=None,
        help="s in the ReLU-gated mask; default |gamma| (our default)",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--lift-shards", type=int, default=4)
    parser.add_argument(
        "--min-share",
        type=float,
        default=0.005,
        help="lifted-set support: share of a concept's positions a token needs",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=2.0,
        help="lifted-set filter: minimum P(token|c)/P(token) (our default)",
    )
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="MEDS metadata dir with codes.parquet, for readable token names "
        "(default: <shard dir>/../../metadata when it exists)",
    )
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    refuse_existing_output(
        Path(args.output_json), overwrite=args.overwrite, kind="steering"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        candidate = Path(args.held_out_shard_dir).resolve().parent.parent / "metadata"
        metadata_dir = str(candidate) if candidate.exists() else None
    prepared = prepare(
        args.run_dir,
        args.held_out_shard_dir,
        args.lift_shard_dir,
        max_shards=args.max_shards,
        lift_shards=args.lift_shards,
        tau=args.tau,
        device=device,
        checkpoint_path=Path(args.run_dir) / args.checkpoint
        if args.checkpoint
        else None,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        metadata_dir=metadata_dir,
        min_share=args.min_share,
        min_lift=args.min_lift,
    )
    layer_index = args.layer_index
    if args.site == "stream" and layer_index is None:
        layers = getattr(prepared.model.backbone, "layers", None)
        if layers is None:
            raise TypeError("stream site needs a block-structured backbone")
        layer_index = len(layers) // 2
    summaries = evaluate_steering(
        prepared,
        concepts=args.concepts,
        site=args.site,
        layer_index=layer_index,
        suppress_strength=args.suppress_strength,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
        n_boot=args.n_boot,
    )
    payload = {
        "site": args.site,
        "layer_index": layer_index,
        "tau": args.tau,
        "suppress_strength": args.suppress_strength,
        "horizons_hours": list(HORIZONS_HOURS),
        "event_names": prepared.event_names,
        "gammas": dict(zip(prepared.concept_names, prepared.gammas, strict=True)),
        "lifted_tokens": {
            prepared.concept_names[c]: [
                prepared.token_names[prepared.vocab.id_to_token[i]] for i in ids
            ]
            for c, ids in prepared.lifted.items()
        },
        "at_risk_restricted": prepared.tables is not None,
        "summaries": _to_json(summaries),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=1))
    logger.info("[steering] wrote %s (%d summaries)", args.output_json, len(summaries))


if __name__ == "__main__":
    _main()
