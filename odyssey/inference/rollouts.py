"""Sampled forward rollouts: what the model expects to happen next (item 12).

:mod:`odyssey.inference.counterfactual` asks how a one-step forecast moves
when the record is edited. This module asks the multi-step question a
clinician actually poses -- *"what is likely to happen over the next
day?"* -- by sampling whole continuations from the model and summarizing
them:

- sample the next event from the next-token distribution;
- sample the gap to it from the time head's discrete hazard (bin 0 is the
  same instant, so bundles fall out naturally: several events at one
  timestamp before the clock advances);
- feed the sampled event back in as the next input, with its structural
  metadata derived exactly as the tokenizer would (family id, time, age,
  and the recency / panel-signal channels advanced), and repeat.

The summary over samples is what is reportable: for each alert event, the
fraction of sampled futures in which it occurs within a horizon (a Monte
Carlo estimate of the same quantity the hazard heads emit in closed form,
which makes the two a consistency check on each other), plus the expected
number of events by family. Combined with a
:class:`~odyssey.inference.counterfactual.ValueEdit`, the same summary run
on an edited record is the interactive what-if: *"assume they had been
hypotensive for six hours -- what do the next 24 hours look like now?"*

Sampling is seeded and the sampler is a plain categorical/hazard draw with
optional top-k and temperature; nothing here is trained. Rollouts are a
model-behaviour readout, not a prediction claim: a sampled future is only
as good as the one-step distributions it chains, and chaining compounds
their errors, so treat horizon-level fractions as the model's own belief,
comparable against its hazard heads, not as a validated risk estimate.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from odyssey.data.alert_events import AlertEvent
from odyssey.data.sequences import N_RECENCY_FAMILIES, PatientSequence
from odyssey.data.signal_panel import N_PANEL_SIGNALS
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.vocabulary import PAD_ID, Vocabulary, code_type
from odyssey.models.sequence_model import SequenceModel
from odyssey.models.time_to_event import survival_curve
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 24.0 * 365.25


@dataclass
class RolloutSample:
    """One sampled continuation from an index position."""

    codes: List[str]
    times: List[float]
    """Hours since the subject's first event, per sampled token."""

    def events_within(self, index_time: float, horizon: float) -> List[str]:
        """Codes sampled in ``(index_time, index_time + horizon]``."""
        return [c for c, t in zip(self.codes, self.times) if t <= index_time + horizon]


@dataclass
class RolloutSummary:
    """Monte Carlo summary over samples for one subject."""

    subject_id: int
    index_time_hours: float
    n_samples: int
    horizons: List[float]
    event_probability: Dict[str, Dict[str, float]]
    """alert event -> horizon label -> fraction of samples containing it."""
    family_counts: Dict[str, Dict[str, float]]
    """horizon label -> code family -> mean number of sampled events."""
    hazard_probability: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """The model's own closed-form ``P(event within h)`` at the index
    position, for the same events and horizons -- the consistency check."""
    samples: List[RolloutSample] = field(default_factory=list)


def _sample_gap_hours(
    hazard_logits: torch.Tensor, edges: Sequence[float], generator: torch.Generator
) -> float:
    """Draw a gap from the discrete-time hazard: a bin, then a point in it.

    Bin 0 is the exact-zero gap (the current bundle continues). A positive
    bin is sampled uniformly within its (left, right] range; the open final
    bin samples exponentially beyond the last edge with a mean of one last
    bin width, so a rollout cannot stall forever at the boundary.
    """
    survival = survival_curve(hazard_logits.unsqueeze(0))[0]
    cdf = 1.0 - survival
    probs = torch.diff(cdf, prepend=torch.zeros(1, device=cdf.device))
    probs = torch.cat([probs, (1.0 - cdf[-1]).clamp_min(0.0).unsqueeze(0)])
    probs = probs.clamp_min(0.0)
    total = probs.sum()
    if float(total) <= 0:
        return 0.0
    idx = int(torch.multinomial(probs / total, 1, generator=generator).item())
    if idx == 0:
        return 0.0
    left = 0.0 if idx == 1 else float(edges[idx - 2])
    if idx - 1 < len(edges):
        right = float(edges[idx - 1])
        u = float(torch.rand(1, generator=generator).item())
        return left + u * (right - left)
    width = float(edges[-1]) - (float(edges[-2]) if len(edges) > 1 else 0.0)
    # Inverse-CDF draw (mean-`width` exponential) instead of
    # torch.distributions.Exponential: that constructor is untyped (mypy
    # no-untyped-call under this repo's strict config) and, worse, its
    # .sample() ignores `generator` entirely -- every other draw in this
    # function is seeded through it, so a torch.distributions call here
    # would silently break rollout reproducibility for exactly the
    # positions that land in the open final bin.
    u = float(torch.rand(1, generator=generator).item())
    tail = -max(width, 1e-6) * math.log1p(-u)
    return float(edges[-1]) + tail


def _sample_code(
    logits: torch.Tensor,
    generator: torch.Generator,
    *,
    temperature: float,
    top_k: Optional[int],
) -> int:
    """Draw a next-token id (temperature, optional top-k; PAD excluded)."""
    scaled = logits / max(temperature, 1e-6)
    scaled[PAD_ID] = float("-inf")
    if top_k is not None and 0 < top_k < scaled.numel():
        cutoff = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


class _RolloutState:
    """The per-token structural metadata a generated token needs.

    Mirrors what :func:`~odyssey.data.sequences.build_patient_sequence`
    computes for a real token -- family recency, panel-signal staleness,
    age, visit ids -- advanced one sampled token at a time so a rollout
    feeds the model the same shape of input it was trained on. Sampled
    tokens carry no numeric value (the model samples a bin token, which
    already encodes the range), so the value and signal-value channels see
    NaN for them, exactly as a valueless real event would.
    """

    def __init__(self, seq: PatientSequence, position: int, resolver: object) -> None:
        """Seed from a real sequence's state at ``position``."""
        self.time = float(seq.time_stamps[position])
        self.age = float(seq.ages[position])
        self.visit_order = int(seq.visit_orders[position])
        self.visit_id = int(seq.visit_ids[position]) if seq.visit_ids else -1
        self.resolver = resolver
        last = seq.family_recency[position] if seq.family_recency else None
        self.family_last: List[float] = [
            self.time - v if last is not None and not np.isnan(v) else float("nan")
            for v in (last or [float("nan")] * N_RECENCY_FAMILIES)
        ]
        self.signal_last: List[float] = [float("nan")] * N_PANEL_SIGNALS
        if seq.signal_state is not None:
            row = seq.signal_state[position]
            for k in range(N_PANEL_SIGNALS):
                if not np.isnan(row[k]):
                    self.signal_last[k] = self.time - float(row[k])

    def advance(self, code: str, gap_hours: float) -> Dict[str, object]:
        """Advance the clock by ``gap_hours`` and absorb ``code``; return its aux."""
        self.time += gap_hours
        self.age += gap_hours / HOURS_PER_YEAR
        type_id = code_type(code)
        recency = [
            self.time - last if not np.isnan(last) else float("nan")
            for last in self.family_last
        ]
        signal = [
            self.time - last if not np.isnan(last) else float("nan")
            for last in self.signal_last
        ] + [float("nan")] * N_PANEL_SIGNALS
        aux = {
            "type_id": type_id,
            "time": self.time,
            "age": self.age,
            "recency": recency,
            "signal": signal,
        }
        if 1 <= type_id <= N_RECENCY_FAMILIES:
            self.family_last[type_id - 1] = self.time
        resolve = getattr(self.resolver, "resolve", None)
        if resolve is not None:
            idx = resolve(code)
            if idx >= 0:
                self.signal_last[idx] = self.time
        return aux


def _step_batch(
    code_id: int, aux: Dict[str, object], visit_order: int, visit_id: int, device: str
) -> ClinicalSequenceBatch:
    """One-token batch for the streaming forward."""

    def t(value: object, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([[value]], dtype=dtype, device=device)

    return ClinicalSequenceBatch(
        concept_ids=t(code_id, torch.long),
        aux=AuxiliaryInputs(
            type_ids=t(aux["type_id"], torch.long),
            time_stamps=t(aux["time"], torch.double),
            ages=t(aux["age"], torch.float),
            visit_orders=t(visit_order, torch.long),
            visit_segments=t(1, torch.long),
            values=t(float("nan"), torch.float),
            family_recency=torch.tensor(
                [[aux["recency"]]], dtype=torch.float, device=device
            ),
            signal_state=torch.tensor(
                [[aux["signal"]]], dtype=torch.float, device=device
            ),
        ),
    )


@torch.no_grad()
def rollout_from_position(
    model: SequenceModel,
    seq: PatientSequence,
    vocab: Vocabulary,
    *,
    position: int,
    horizon_hours: float,
    n_samples: int = 32,
    max_steps: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    chunk_size: int = 256,
) -> List[RolloutSample]:
    """Sample ``n_samples`` continuations from ``position`` of ``seq``.

    Each sample stops at ``horizon_hours`` past the index position's time
    or after ``max_steps`` tokens, whichever comes first. The history up to
    ``position`` is streamed once and its recurrent state reused for every
    sample, so cost is one prefix pass plus ``n_samples * steps`` one-token
    forwards.
    """
    model.eval()
    time_head = getattr(model, "time_head", None)
    if time_head is None:
        raise ValueError("rollouts need a model with a time-to-event head")
    prefix = seq.tail(position + 1) if position + 1 < len(seq) else seq
    state = None
    logits_at_index: Optional[torch.Tensor] = None
    features_at_index: Optional[torch.Tensor] = None
    for chunk in PackedLaneSampler(
        iter([prefix]), num_lanes=1, chunk_size=chunk_size, reset_prob=0.0
    ):
        chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
        fwd = model.forward_with_features(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        state = fwd.state
        real = int((chunk.subject_ids[0] >= 0).sum().item())
        if real:
            logits_at_index = fwd.logits[0, real - 1].clone()
            features_at_index = fwd.features[0, real - 1].clone()
    if logits_at_index is None or features_at_index is None:
        raise ValueError("empty prefix: nothing to roll out from")

    index_time = float(prefix.time_stamps[-1])
    resolver = getattr(model, "signal_panel", None)
    samples: List[RolloutSample] = []
    for s in range(n_samples):
        generator = torch.Generator(device="cpu").manual_seed(seed * 100_003 + s)
        rs = _RolloutState(prefix, len(prefix) - 1, resolver)
        step_state = state
        logits, features = logits_at_index, features_at_index
        codes: List[str] = []
        times: List[float] = []
        for _ in range(max_steps):
            gap = _sample_gap_hours(
                time_head(features.unsqueeze(0))[0], time_head.edges, generator
            )
            if rs.time + gap > index_time + horizon_hours:
                break
            code_id = _sample_code(
                logits.clone(), generator, temperature=temperature, top_k=top_k
            )
            code = vocab.decode(code_id)
            aux = rs.advance(code, gap)
            codes.append(code)
            times.append(rs.time)
            batch = _step_batch(code_id, aux, rs.visit_order, rs.visit_id, device)
            fwd = model.forward_with_features(batch, state=step_state, reset_mask=None)
            step_state = fwd.state
            logits, features = fwd.logits[0, -1], fwd.features[0, -1]
        samples.append(RolloutSample(codes=codes, times=times))
    return samples


def summarize_rollouts(
    samples: Sequence[RolloutSample],
    alerts: Sequence[AlertEvent],
    vocab: Vocabulary,
    *,
    subject_id: int,
    index_time_hours: float,
    horizons: Sequence[float],
) -> RolloutSummary:
    """Fraction of samples containing each alert event, and family counts."""
    import re  # noqa: PLC0415

    patterns = {
        a.name: re.compile(
            a.token_regex or f"^{re.escape(a.code_prefix or a.name)}", re.I
        )
        for a in alerts
    }
    event_probability: Dict[str, Dict[str, float]] = {a.name: {} for a in alerts}
    family_counts: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        label = f"{h:g}h"
        within = [s.events_within(index_time_hours, h) for s in samples]
        for name, pattern in patterns.items():
            hits = sum(1 for codes in within if any(pattern.search(c) for c in codes))
            event_probability[name][label] = hits / max(len(samples), 1)
        counter: Counter[str] = Counter()
        for codes in within:
            for c in codes:
                counter[str(code_type(c))] += 1
        family_counts[label] = {
            k: v / max(len(samples), 1) for k, v in sorted(counter.items())
        }
    del vocab  # patterns are matched on decoded codes
    return RolloutSummary(
        subject_id=subject_id,
        index_time_hours=index_time_hours,
        n_samples=len(samples),
        horizons=list(horizons),
        event_probability=event_probability,
        family_counts=family_counts,
    )


def hazard_probabilities_at(
    model: SequenceModel,
    features: torch.Tensor,
    alerts: Sequence[AlertEvent],
    horizons: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Return the model's closed-form ``P(event within h)`` at this position."""
    from odyssey.models.time_to_event import probability_within  # noqa: PLC0415

    heads = getattr(model, "event_heads", None)
    if heads is None:
        return {}
    index = {name: i for i, name in enumerate(heads.event_names)}
    hz = heads(features.unsqueeze(0))
    out: Dict[str, Dict[str, float]] = {}
    for alert in alerts:
        if alert.name not in index:
            continue
        logits = hz[:, index[alert.name]]
        out[alert.name] = {
            f"{h:g}h": float(probability_within(logits, heads.edges, h)[0])
            for h in horizons
        }
    return out


__all__ = [
    "RolloutSample",
    "RolloutSummary",
    "hazard_probabilities_at",
    "rollout_from_position",
    "summarize_rollouts",
]
