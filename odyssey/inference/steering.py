"""Concept steering as a clinician would test it: turn one dial, watch the risks.

The lever a clinician wants is not "does the model predict the recorded
next token better when told the true state" (that is the trust test of
:mod:`odyssey.inference.interventions`). It is: push the model along the
*shock* direction and vasopressor risk should rise, ICU-transfer risk
should rise, and the events the model expects next should look like
shock; pull it the other way and they should fall. This module measures
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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.run_inference import load_run, refuse_existing_output
from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    DecomposedConceptBottleneck,
)
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
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
from odyssey.training.running_labels import position_running_labels
from odyssey.training.train import _move_chunk_to_device


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
#: * circulatory failure (shock, hypotension, elevated lactate, metabolic
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
    "shock": {
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
# Where and how hard to push (Steerling Section 6.2).
# ---------------------------------------------------------------------------


def _known_embedding(
    model: ConceptBottleneckSequenceModel, concept_index: int
) -> torch.Tensor:
    bottleneck = model.bottleneck
    if not isinstance(bottleneck, DecomposedConceptBottleneck):
        raise NotImplementedError(
            "steering needs the decomposed bottleneck, where a unit of a concept "
            "is the parameter K_c; the mixture's displacement is a function of "
            "the hidden state and would have to be estimated per position"
        )
    embedding: torch.Tensor = bottleneck.known_embeddings[concept_index].detach()
    return embedding


def steering_direction(
    model: ConceptBottleneckSequenceModel, concept_index: int
) -> torch.Tensor:
    """``e_c = K_c / ||K_c||_2``, Steerling's steering direction (their 6.2.1)."""
    embedding = _known_embedding(model, concept_index)
    unit: torch.Tensor = embedding / embedding.norm().clamp_min(1e-12)
    return unit


def steering_gamma(
    model: ConceptBottleneckSequenceModel, direction: torch.Tensor, *, tau: float
) -> float:
    """``gamma = tau / peak(e_c)`` with ``peak(e_c) = max_y e_c . W_y`` (Eq. 19).

    The maximum is signed, not absolute: the paper calibrates on the
    largest logit *increase* the direction can produce.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    weight = model.lm_head.weight.detach().to(direction)
    peak = float((weight @ direction).max().item())
    if peak <= 0:
        raise ValueError("the direction raises no logit; calibration is undefined")
    return tau / peak


def concept_alignment(
    model: ConceptBottleneckSequenceModel, direction: torch.Tensor
) -> torch.Tensor:
    """``a_c = W e_c``: the concept's contribution to every logit (Eq. 20)."""
    weight = model.lm_head.weight.detach().to(direction)
    alignment: torch.Tensor = weight @ direction
    return alignment


def suppress_logits(
    logits: torch.Tensor, alignment: torch.Tensor, strength: float
) -> torch.Tensor:
    """``l_v -> l_v - s . ReLU(a_c[v])``, the ReLU-gated mask (Eq. 21).

    Plain subtraction would promote tokens anti-aligned with the concept;
    the gate leaves them untouched.
    """
    return logits - strength * torch.relu(alignment).to(logits)


@contextmanager
def stream_injection(
    backbone: torch.nn.Module, layer_index: int, vector: torch.Tensor
) -> Iterator[None]:
    """Add ``vector`` to the hidden state at every block from ``layer_index`` on.

    Steerling's Eq. 18: ``h^(l) <- h^(l) + gamma e_c`` for ``l >= L_inj``, so
    the signal accumulates toward the bottleneck. Registered as forward
    hooks on those blocks; every position of every later chunk is pushed
    and the recurrent state carries the push forward. The hooks are
    removed on exit even if the pass raises.
    """
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError(
            f"{type(backbone).__name__} exposes no `layers` to inject into; the "
            "stream site needs a block-structured backbone"
        )
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"layer_index {layer_index} outside 0..{len(layers) - 1}")

    def push(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden, *rest = output
            return (hidden + vector.to(hidden), *rest)
        return output + vector.to(output)

    handles = [layer.register_forward_hook(push) for layer in layers[layer_index:]]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


# ---------------------------------------------------------------------------
# One streaming pass -> per-subject readouts.
# ---------------------------------------------------------------------------


@dataclass
class SubjectReadout:
    """Sums over one subject's real positions; divide by ``n`` for means."""

    n: int = 0
    concept_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """(num_concepts,) summed activations."""
    lifted_mass: float = 0.0
    """Summed next-event probability on the concept's lifted tokens."""
    risk: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """(num_events, num_horizons) summed P(event within horizon)."""

    def add(self, probs: torch.Tensor, mass: torch.Tensor, risk: torch.Tensor) -> None:
        """Accumulate one batch of positions belonging to this subject."""
        p = probs.sum(dim=0).double().cpu().numpy()
        r = risk.sum(dim=0).double().cpu().numpy()
        self.concept_probs = p if self.n == 0 else self.concept_probs + p
        self.risk = r if self.n == 0 else self.risk + r
        self.lifted_mass += float(mass.sum().item())
        self.n += int(probs.shape[0])


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
    lifted_ids: torch.Tensor,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: int | None = None,
) -> dict[int, SubjectReadout]:
    """Stream every held-out patient once under ``push`` (``None`` = unsteered).

    The same sampler and state carrying as every other scorer, so a pushed
    pass and the unsteered pass visit identical positions and their
    per-subject means are paired.
    """
    model.eval()
    sampler = PackedLaneSampler(
        iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len),
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        reset_prob=0.0,
    )
    readouts: dict[int, SubjectReadout] = {}
    lifted = lifted_ids.to(device)
    state: Any = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            logits, probs, features, state = _forward_pushed(model, chunk, state, push)
            real = chunk.real_mask
            if not real.any():
                continue
            mass = F.softmax(logits[real].float(), dim=-1)[:, lifted].sum(dim=-1)
            risk = _outcome_risk(model, features[real])
            sids = chunk.subject_ids[real]
            probs_real = probs[real]
            for sid in torch.unique(sids).tolist():
                sel = sids == sid
                readouts.setdefault(int(sid), SubjectReadout()).add(
                    probs_real[sel], mass[sel], risk[sel]
                )
    return readouts


# ---------------------------------------------------------------------------
# Lifted tokens: the events that say "this concept is present".
# ---------------------------------------------------------------------------


def lifted_token_sets(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    supervision: ConceptSupervision,
    top_k: int = 25,
    min_count: int = 20,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
) -> dict[int, list[int]]:
    """Per concept, the ``top_k`` next-event tokens with the highest lift.

    Lift is ``P(token | concept active) / P(token)`` over target positions
    of a labeled stream, with running labels so "active" means "has
    triggered by this position"; Steerling defines the express target the
    same way over its chunk tags, with a minimum-support filter. Tokens
    seen fewer than ``min_count`` times under the concept are ignored so a
    rare code cannot top the list on two occurrences; only tokens with
    lift above 1 are kept.
    """
    num_concepts = model.bottleneck.num_concepts
    vocab_size = len(vocab.token_to_id)
    total = torch.zeros(vocab_size, dtype=torch.float64)
    per_concept = torch.zeros(num_concepts, vocab_size, dtype=torch.float64)
    sampler = PackedLaneSampler(
        iter_patient_sequences(events_binned, vocab),
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        reset_prob=0.0,
    )
    for chunk in sampler:
        chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
        labels, observed = position_running_labels(
            chunk,
            concept_labels,
            concept_mask,
            concept_first_times,
            supervision=supervision,
            num_concepts=num_concepts,
        )
        real = chunk.real_mask
        if not real.any():
            continue
        targets = chunk.targets[real]
        active = (labels[real] * observed[real]).to(torch.float64)  # (N, k)
        one_hot = F.one_hot(targets, num_classes=vocab_size).to(torch.float64)
        total += one_hot.sum(dim=0).cpu()
        per_concept += (active.T @ one_hot).cpu()
    return rank_by_lift(total, per_concept, top_k=top_k, min_count=min_count)


def rank_by_lift(
    total: torch.Tensor, per_concept: torch.Tensor, *, top_k: int, min_count: int
) -> dict[int, list[int]]:
    """Top-``top_k`` tokens by ``P(token | c) / P(token)`` per concept.

    ``total`` is ``(vocab,)`` target counts over the stream and
    ``per_concept`` is ``(num_concepts, vocab)`` counts at positions where
    each concept is active. Tokens under ``min_count`` occurrences for a
    concept, and tokens with lift at or below 1, are excluded.
    """
    base = total / total.sum().clamp_min(1.0)
    sets: dict[int, list[int]] = {}
    for c in range(per_concept.shape[0]):
        counts = per_concept[c]
        cond = counts / counts.sum().clamp_min(1.0)
        lift = torch.where(
            counts >= min_count, cond / base.clamp_min(1e-12), torch.zeros_like(cond)
        )
        keep = int(min(top_k, int((lift > 1.0).sum().item())))
        sets[c] = (
            [int(i) for i in torch.topk(lift, k=keep).indices.tolist()] if keep else []
        )
    return sets


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
    mass = np.array([readouts[s].lifted_mass / readouts[s].n for s in subjects])
    risk = np.stack([readouts[s].risk / readouts[s].n for s in subjects])
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
            if sign is not None:
                moved = (r1[:, e, h] - r0[:, e, h]) * sign
                agreement = float((moved > 0).mean())
            outcomes.append(
                OutcomeShift(
                    event=event,
                    horizon_hours=horizon,
                    baseline_risk=float(r0[:, e, h].mean()),
                    steered_risk=float(r1[:, e, h].mean()),
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
        express_baseline=float(m0.mean()),
        express_steered=float(m1.mean()),
        express_delta=paired_delta(m1, m0, n_boot=n_boot, seed=seed),
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
        model,
        lift_binned,
        vocab,
        concept_labels=labels,
        concept_mask=mask,
        concept_first_times=first_times,
        supervision=supervision,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
    )
    del lift_binned
    for c, ids in lifted.items():
        logger.info(
            "[steering] lifted tokens for %s: %s",
            concept_names[c],
            [vocab.id_to_token[i] for i in ids[:8]],
        )

    _raw, events_binned = binned(held_out_shard_dir, max_shards)
    del _raw
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
    chosen = (
        list(concepts) if concepts else [n for n in names if n in CLINICAL_EXPECTATIONS]
    )
    unknown = sorted(set(chosen) - set(names))
    if unknown:
        raise ValueError(f"concepts not in this run's registry: {unknown}")
    every_lifted = sorted({i for ids in prepared.lifted.values() for i in ids})

    summaries: list[ConceptSteeringSummary] = []
    for concept in chosen:
        c = names.index(concept)
        lifted = torch.tensor(prepared.lifted.get(c) or every_lifted, dtype=torch.long)
        baseline = run_steering_pass(
            prepared.model,
            prepared.events_binned,
            prepared.vocab,
            push=None,
            lifted_ids=lifted,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
        )
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
                lifted_ids=lifted,
                num_lanes=num_lanes,
                chunk_size=chunk_size,
                device=device,
            )
            summary = summarize_push(
                baseline,
                steered,
                concept=concept,
                concept_index=c,
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
            prepared.concept_names[c]: [prepared.vocab.id_to_token[i] for i in ids]
            for c, ids in prepared.lifted.items()
        },
        "summaries": _to_json(summaries),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=1))
    logger.info("[steering] wrote %s (%d summaries)", args.output_json, len(summaries))


if __name__ == "__main__":
    _main()
