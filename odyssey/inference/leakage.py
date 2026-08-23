"""Concept-channel leakage metrics on a frozen bottleneck (Track B item 11).

Two probe-based diagnostics for a trained :class:`ConceptBottleneckSequenceModel`,
computed entirely on frozen, banked features from one streaming pass over a
split -- the pattern in :mod:`odyssey.inference.time_head_probe`
(:func:`~odyssey.inference.time_head_probe.collect_feature_bank`), reused
here rather than reinvented: the same streaming (:class:`PackedLaneSampler`),
fp16-at-rest, and position-cap discipline.

**These exact operationalizations -- CTL and ICL below, their probe designs
and formulas -- are this project's own and were designed for this codebase.
They are not drawn from a paper.** The phenomenon they operationalize is
documented elsewhere: soft-concept leakage (Mahinpei et al. 2021,
"Promises and Pitfalls of Black-Box Concept Learning Models") and a
mitigation for it (Havasi et al. 2022, "Addressing Leakage in Concept
Bottleneck Models"). Neither paper's specific metric is what is computed
here; do not cite them as the source of these formulas.

- **CTL (concept-task leakage)**: how much task-predictive information the
  concept channel carries *beyond* the concept values a clinician would
  read off (:data:`ConceptBottleneckOutput.concept_probs`). Four
  multinomial-logistic probes -- from the concept probabilities alone, from
  the known-concept embedding block alone, from the unknown/residual
  embedding alone, and from the probabilities passed through a fixed random
  projection to the embedding block's own width -- predict a compact
  per-position task label: the next-token's code-family id
  (:func:`odyssey.data.vocabulary.code_type`, the same 8-way family
  taxonomy :mod:`odyssey.training.metrics` reports task accuracy by).
  Chosen over the raw next-token id itself because a vocab-sized softmax
  probe would mostly measure memorization capacity, not leaked
  task-relevant structure; the family id is compact, always defined, and
  already the codebase's own notion of "what kind of thing is happening
  next".

  ``embeddings_only`` has ``num_concepts * embedding_dim`` inputs (often
  hundreds) against ``probs_only``'s ``num_concepts`` -- a naive
  ``embeddings_only - probs_only`` delta is confounded with probe
  *capacity*, not just leaked information, since a wider linear probe can
  fit more even when its extra inputs carry nothing new. ``probs_projected``
  (:func:`_random_projection`, a fixed, seeded, non-trainable
  semi-orthogonal map -- information-preserving, dimension-matching) is the
  capacity control: it carries exactly what ``probs_only`` carries, at
  ``embeddings_only``'s width. Two deltas are reported, in both accuracy
  and cross-entropy: ``ctl_vs_probs = embeddings_only - probs_only`` (the
  uncontrolled upper bound -- capacity and leakage both contribute) and
  ``ctl_vs_projected = embeddings_only - probs_projected`` (**the leakage
  reading** -- capacity is matched, so a positive accuracy delta / negative
  CE delta here can only mean the embeddings carry information the
  probabilities did not). The unknown-only probe is reported alongside as
  the residual channel's own share of that same task signal.

  This complements, not replaces, the ``zero_known``/``zero_unknown``
  ablation in :mod:`odyssey.inference.interventions`: that measures how
  much the task *depends* on each channel (causal -- ablate and see what
  breaks); CTL measures how much task information a channel *carries*,
  independent of whether the model currently uses it (a probe's ceiling --
  a channel the model ignores today can still leak recoverable
  information). A bottleneck can score well on the ablation (task loss
  jumps when concepts are zeroed -- the model relies on them) and still
  leak on CTL (the embeddings also carry next-event information beyond
  their own probabilities) -- the two questions are independent.

- **ICL (inter-concept leakage)**: for each ordered pair of known concepts
  ``(i, j)`` with ``i != j``, how much of concept ``j``'s own running label
  is predictable from concept ``i``'s slot embedding, beyond what concept
  ``i``'s probability alone (and the label-label correlation it already
  captures) implies. Labels are the *running* (time-valid) labels from
  :mod:`odyssey.training.running_labels` -- true only from each concept's
  first-trigger time onward, not the whole-visit retrospective label --
  restricted to positions where concept ``j`` is observed (no ground
  truth exists to score otherwise, same convention as the interventions
  module). For each concept ``i``, one small linear probe from its
  embedding and one from its scalar probability are each fit *jointly*
  against every other concept's label (a multi-output logistic regression,
  independent per-output BCE) rather than fit pairwise -- ``O(num_concepts)``
  probe fits instead of ``O(num_concepts^2)``, reading every pair's AUROC
  off the same two fits' held-out outputs; the probes and their
  optimization are otherwise identical to the pairwise formulation. Report
  is the full ``AUROC(embedding_i -> label_j) - AUROC(probability_i ->
  label_j)`` matrix, both raw (can be negative -- the probability was
  already the better predictor) and clipped at 0 (the leakage reading:
  only positive extra predictability counts as leakage).
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import roc_auc_score
from torch import nn

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.inference.run_inference import (
    _CODE_TYPE_NAMES,
    _build_type_lookup,
    load_run,
    refuse_existing_output,
)
from odyssey.models.concept_bottleneck import ConceptBottleneckOutput
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
    SequenceModel,
)
from odyssey.training.data import iter_patient_sequences
from odyssey.training.running_labels import position_running_labels
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

DEFAULT_EPOCHS = 20


# ---------------------------------------------------------------------------
# Feature bank
# ---------------------------------------------------------------------------


@dataclass
class LeakageBank:
    """Frozen bottleneck outputs plus their CTL/ICL targets for one split.

    One row per banked position: a position with a real, non-padding
    next-token target (the CTL family label needs one; ICL doesn't, but
    sharing one valid-position mask keeps a single bank/single streaming
    pass serving both metrics, at the cost of the handful of
    sequence-final positions -- real but targetless -- that ICL alone
    could otherwise have used). See :func:`collect_leakage_bank`.
    """

    concept_probs: torch.Tensor
    """``(N, num_concepts)`` float16 at rest, on CPU."""
    concept_embeddings: torch.Tensor
    """``(N, num_concepts, embedding_dim)`` float16 at rest, on CPU."""
    unknown_embedding: torch.Tensor
    """``(N, unknown_dim)`` float16 at rest, on CPU."""
    family_labels: torch.Tensor
    """``(N,)`` long: the next token's code-family id (CTL's task label)."""
    concept_labels: torch.Tensor
    """``(N, num_concepts)`` float32: running (time-valid) concept labels."""
    concept_observed: torch.Tensor
    """``(N, num_concepts)`` bool: whether each concept's running label has
    ground truth at that position (see :mod:`odyssey.training.running_labels`)."""
    concept_names: Tuple[str, ...]
    n_positions_seen: int
    sample_rate: float

    def __len__(self) -> int:
        """Return the number of banked positions."""
        return int(self.family_labels.numel())

    def to(self, device: str) -> "LeakageBank":
        """Copy to ``device`` (for fitting); float tensors stay fp16 until batched."""
        return LeakageBank(
            self.concept_probs.to(device),
            self.concept_embeddings.to(device),
            self.unknown_embedding.to(device),
            self.family_labels.to(device),
            self.concept_labels.to(device),
            self.concept_observed.to(device),
            self.concept_names,
            self.n_positions_seen,
            self.sample_rate,
        )

    @staticmethod
    def concat(
        banks: Sequence["LeakageBank"], max_positions: Optional[int]
    ) -> "LeakageBank":
        """Stack per-shard banks (same concept names), capped at ``max_positions``."""
        if not banks:
            raise ValueError("no banks to concatenate")
        names = banks[0].concept_names
        if any(b.concept_names != names for b in banks):
            raise ValueError("banks disagree on concept_names -- different runs?")

        def cat(attr: str) -> torch.Tensor:
            joined = torch.cat([getattr(b, attr) for b in banks])
            return joined[:max_positions] if max_positions else joined

        return LeakageBank(
            cat("concept_probs"),
            cat("concept_embeddings"),
            cat("unknown_embedding"),
            cat("family_labels"),
            cat("concept_labels"),
            cat("concept_observed"),
            names,
            n_positions_seen=sum(b.n_positions_seen for b in banks),
            sample_rate=banks[0].sample_rate,
        )


def collect_leakage_bank(
    model: SequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    concept_names: Sequence[str],
    supervision: ConceptSupervision,
    sample_rate: float = 1.0,
    seed: int = 0,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cpu",
    max_positions: Optional[int] = None,
) -> LeakageBank:
    """One frozen streaming pass; bank bottleneck outputs and their CTL/ICL targets.

    Positions are those with a real, non-padding next-token target -- the
    same valid-position definition
    :meth:`~odyssey.models.sequence_model._SequenceModelBase._streaming_task_loss`
    trains on -- kept with probability ``sample_rate`` (seeded) and capped
    at ``max_positions``. Running concept labels
    (:func:`~odyssey.training.running_labels.position_running_labels`) are
    computed for every real position regardless of the next-token filter,
    then subset to the same kept rows as everything else.
    """
    model.eval()
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            f"leakage metrics need a concept bottleneck; got {type(model).__name__}"
        )
    num_concepts = model.bottleneck.num_concepts
    if len(concept_names) != num_concepts:
        raise ValueError(
            f"{len(concept_names)} concept names but the bottleneck has "
            f"{num_concepts} concepts -- source/task_set mismatch?"
        )
    gen = torch.Generator().manual_seed(seed)
    type_lookup = _build_type_lookup(vocab, device)
    patients: Iterator[PatientSequence] = iter_patient_sequences(
        events_binned, vocab, signal_panel=getattr(model, "signal_panel", None)
    )
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )

    parts: Dict[str, List[torch.Tensor]] = {
        "probs": [],
        "embed": [],
        "unknown": [],
        "family": [],
        "labels": [],
        "observed": [],
    }

    def bank_valid(
        bottleneck: ConceptBottleneckOutput,
        family: torch.Tensor,
        labels: torch.Tensor,
        observed: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        # fp16 at rest: a bank is millions of rows x (concepts x
        # embedding_dim); heads upcast per batch when fitting/scoring.
        parts["probs"].append(bottleneck.concept_probs[valid].to(torch.float16).cpu())
        parts["embed"].append(
            bottleneck.concept_embeddings[valid].to(torch.float16).cpu()
        )
        parts["unknown"].append(
            bottleneck.unknown_embedding[valid].to(torch.float16).cpu()
        )
        parts["family"].append(family[valid].long().cpu())
        parts["labels"].append(labels[valid].float().cpu())
        parts["observed"].append(observed[valid].bool().cpu())

    seen = 0
    kept = 0
    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            fwd = model.forward_with_features(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            state = fwd.state
            if fwd.bottleneck is None:
                raise ValueError("model.forward_with_features returned no bottleneck")
            valid = chunk.real_mask & (chunk.targets != PAD_ID)
            n_valid = int(valid.sum().item())
            seen += n_valid
            if n_valid == 0:
                continue
            if sample_rate < 1.0:
                draw = torch.rand(valid.shape, generator=gen) < sample_rate
                valid = valid & draw.to(valid.device)
            if not bool(valid.any()):
                continue
            labels, observed = position_running_labels(
                chunk,
                concept_labels,
                concept_mask,
                concept_first_times,
                supervision=supervision,
                num_concepts=num_concepts,
            )
            family = type_lookup[chunk.targets]
            bank_valid(fwd.bottleneck, family, labels, observed, valid)
            kept += n_valid if sample_rate >= 1.0 else int(valid.sum().item())
            if max_positions is not None and kept >= max_positions:
                break
    if not parts["family"]:
        raise ValueError("no valid positions collected -- empty split?")

    def cap(key: str) -> torch.Tensor:
        joined = torch.cat(parts[key])
        return joined[:max_positions] if max_positions else joined

    bank = LeakageBank(
        cap("probs"),
        cap("embed"),
        cap("unknown"),
        cap("family"),
        cap("labels"),
        cap("observed"),
        tuple(concept_names),
        n_positions_seen=seen,
        sample_rate=sample_rate,
    )
    logger.info(
        "[leakage] banked %d of %d positions (rate %.3f, %d concepts)",
        len(bank),
        seen,
        sample_rate,
        num_concepts,
    )
    return bank


# ---------------------------------------------------------------------------
# Shared probe-fitting machinery
# ---------------------------------------------------------------------------


@dataclass
class ProbeFitTrace:
    """Per-epoch tuning loss and the epoch early stopping picked."""

    tuning_loss: List[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0


class _StandardizedLinearProbe(nn.Module):
    """A linear head behind a fixed (train-set) feature standardization.

    Bottleneck feature scales are not controlled -- a raw concept
    probability lives in [0, 1], but an embedding dimension can be
    anything a LeakyReLU projection produces. A handful of Adam steps at a
    fixed ``lr`` converges far too slowly on badly-scaled inputs (large
    initial logits saturate softmax/sigmoid, so the loss's gradient w.r.t.
    the weights is tiny exactly when it most needs to be large); fitting
    every probe behind the same standardization makes the optimization
    well-conditioned regardless of the input's native scale, without the
    caller having to know or care what that scale was. Stats are fit once
    on the training split and frozen (buffers, not parameters) -- the
    tuning/held-out splits are transformed, never refit.
    """

    def __init__(
        self, in_features: int, out_features: int, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """Wrap ``nn.Linear(in_features, out_features)`` with fixed (mean, std)."""
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize then apply the linear layer."""
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        out: torch.Tensor = self.linear((x - mean) / std)
        return out


def _feature_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-feature (mean, std) from ``x``, std floored to avoid a divide by 0."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def _fit_categorical_probe(
    in_features: int,
    num_classes: int,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    tune_x: torch.Tensor,
    tune_y: torch.Tensor,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[_StandardizedLinearProbe, ProbeFitTrace]:
    """Multinomial-logistic probe (CTL): Adam on train, early-stopped on tuning CE."""
    torch.manual_seed(seed)
    mean, std = _feature_stats(train_x.float())
    head = _StandardizedLinearProbe(in_features, num_classes, mean, std).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    trace = ProbeFitTrace()
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best = float("inf")
    bad = 0
    n = train_x.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=train_x.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(head(train_x[idx].float()), train_y[idx])
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        with torch.no_grad():
            tune_loss = float(F.cross_entropy(head(tune_x.float()), tune_y).item())
        trace.tuning_loss.append(tune_loss)
        if tune_loss < best - 1e-6:
            best, bad = tune_loss, 0
            trace.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    trace.seconds = time.time() - t0
    return head, trace


@torch.no_grad()
def _score_categorical_probe(
    head: _StandardizedLinearProbe, x: torch.Tensor, y: torch.Tensor
) -> Tuple[float, float]:
    """Return (accuracy, mean cross-entropy) on ``(x, y)``."""
    head.eval()
    logits = head(x.float())
    ce = float(F.cross_entropy(logits, y).item())
    acc = float((logits.argmax(dim=-1) == y).float().mean().item())
    return acc, ce


def _fit_masked_multilabel_probe(
    in_features: int,
    out_dim: int,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_observed: torch.Tensor,
    tune_x: torch.Tensor,
    tune_y: torch.Tensor,
    tune_observed: torch.Tensor,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[_StandardizedLinearProbe, ProbeFitTrace]:
    """Multi-output logistic probe (ICL): one linear head, ``out_dim`` targets.

    Each position/output pair weighted by ``observed`` so an unobserved
    (position, concept) contributes no gradient and no tuning loss -- there
    is no ground truth to fit or score there.
    """

    def masked_bce(
        logits: torch.Tensor, y: torch.Tensor, observed: torch.Tensor
    ) -> torch.Tensor:
        per_elem = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        weight = observed.float()
        return (per_elem * weight).sum() / weight.sum().clamp_min(1.0)

    torch.manual_seed(seed)
    mean, std = _feature_stats(train_x.float())
    head = _StandardizedLinearProbe(in_features, out_dim, mean, std).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    trace = ProbeFitTrace()
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best = float("inf")
    bad = 0
    n = train_x.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=train_x.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = masked_bce(
                head(train_x[idx].float()), train_y[idx], train_observed[idx]
            )
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        with torch.no_grad():
            tune_loss = float(
                masked_bce(head(tune_x.float()), tune_y, tune_observed).item()
            )
        trace.tuning_loss.append(tune_loss)
        if tune_loss < best - 1e-6:
            best, bad = tune_loss, 0
            trace.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    trace.seconds = time.time() - t0
    return head, trace


# ---------------------------------------------------------------------------
# CTL
# ---------------------------------------------------------------------------


@dataclass
class ProbeScore:
    """Held-out accuracy/cross-entropy for one CTL probe."""

    probe: str
    n: int
    accuracy: float
    cross_entropy: float
    parameters: int
    fit: Optional[ProbeFitTrace] = None


@dataclass
class CTLResult:
    """Concept-task leakage: four probes of the next-token family.

    From ``concept_probs``/``concept_embeddings``/``unknown_embedding``/a
    fixed random projection of ``concept_probs`` (the capacity control).
    """

    n_classes: int
    class_names: Dict[int, str]
    probs_only: ProbeScore
    embeddings_only: ProbeScore
    unknown_only: ProbeScore
    probs_projected: ProbeScore
    """``probs_only`` passed through a fixed, seeded, non-trainable
    semi-orthogonal projection to ``embeddings_only``'s own input width
    (:func:`_random_projection`) -- carries no information ``probs_only``
    didn't already have, only matches its dimensionality."""
    ctl_vs_probs_accuracy: float
    """``embeddings_only.accuracy - probs_only.accuracy``: the UNCONTROLLED
    upper bound -- a positive value can come from leaked information, from
    ``embeddings_only`` simply having more probe capacity, or both."""
    ctl_vs_probs_cross_entropy: float
    """``embeddings_only.cross_entropy - probs_only.cross_entropy`` (lower CE
    is better, so a *negative* value here is the same direction as a
    positive ``ctl_vs_probs_accuracy``); same capacity caveat."""
    ctl_vs_projected_accuracy: float
    """``embeddings_only.accuracy - probs_projected.accuracy``: THE LEAKAGE
    READING. Capacity is matched (both probes have the same input width),
    so a positive value can only mean the embeddings carry next-event
    information the probabilities did not."""
    ctl_vs_projected_cross_entropy: float
    """``embeddings_only.cross_entropy - probs_projected.cross_entropy``;
    same capacity-controlled leakage reading, in cross-entropy (negative =
    leakage, matching a positive ``ctl_vs_projected_accuracy``)."""
    n_train: int
    n_tuning: int
    n_held_out: int


def _random_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    """Build a fixed ``(in_dim, out_dim)`` semi-orthogonal projection, seeded.

    CTL's capacity control (see the module docstring): a QR decomposition
    of a seeded Gaussian matrix gives orthonormal columns, so the map is
    information-preserving (linearly invertible on its range) -- it
    cannot manufacture information ``concept_probs`` didn't have, only
    re-express it at ``out_dim`` width to match ``embeddings_only``'s
    input dimensionality. Deterministic and self-contained: uses a local
    :class:`torch.Generator`, never the global RNG, so it has no side
    effect on any other seeded call in this module.
    """
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(out_dim, in_dim, generator=gen)
    q, _ = torch.linalg.qr(raw)
    out: torch.Tensor = q.T
    return out


def compute_ctl(
    train_bank: LeakageBank,
    tuning_bank: LeakageBank,
    held_out_bank: LeakageBank,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 4096,
    patience: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> CTLResult:
    """Fit and score the four CTL probes; see the module docstring.

    ``batch_size`` matters more than it looks: Adam's step size is close
    to ``lr`` regardless of gradient magnitude, so total optimizer steps
    (``epochs * ceil(n_train / batch_size)``), not epochs alone, is what
    determines whether a probe actually converges -- a small bank needs a
    smaller ``batch_size`` (more steps per epoch) to reach the same number
    of updates a large one gets for free.
    """
    num_classes = max(_CODE_TYPE_NAMES) + 1
    class_names = dict(_CODE_TYPE_NAMES)
    num_concepts = train_bank.concept_probs.shape[1]
    embedding_dim = train_bank.concept_embeddings.shape[-1]
    projection = _random_projection(num_concepts, num_concepts * embedding_dim, seed)

    def flat(bank: LeakageBank) -> Dict[str, torch.Tensor]:
        return {
            "probs_only": bank.concept_probs.float(),
            "embeddings_only": bank.concept_embeddings.float().reshape(
                len(bank), num_concepts * embedding_dim
            ),
            "unknown_only": bank.unknown_embedding.float(),
            "probs_projected": bank.concept_probs.float() @ projection,
        }

    train_x = flat(train_bank)
    tune_x = flat(tuning_bank)
    held_x = flat(held_out_bank)
    train_y = train_bank.family_labels
    tune_y = tuning_bank.family_labels
    held_y = held_out_bank.family_labels

    scores: Dict[str, ProbeScore] = {}
    for name, x_train in train_x.items():
        head, trace = _fit_categorical_probe(
            x_train.shape[1],
            num_classes,
            train_x=x_train,
            train_y=train_y,
            tune_x=tune_x[name],
            tune_y=tune_y,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            device=device,
        )
        acc, ce = _score_categorical_probe(head, held_x[name], held_y)
        scores[name] = ProbeScore(
            probe=name,
            n=len(held_out_bank),
            accuracy=acc,
            cross_entropy=ce,
            parameters=sum(p.numel() for p in head.parameters()),
            fit=trace,
        )
        logger.info(
            "[leakage] CTL %-14s acc=%.4f ce=%.4f (%d params, %.1fs)",
            name,
            acc,
            ce,
            scores[name].parameters,
            trace.seconds,
        )

    return CTLResult(
        n_classes=num_classes,
        class_names=class_names,
        probs_only=scores["probs_only"],
        embeddings_only=scores["embeddings_only"],
        unknown_only=scores["unknown_only"],
        probs_projected=scores["probs_projected"],
        ctl_vs_probs_accuracy=(
            scores["embeddings_only"].accuracy - scores["probs_only"].accuracy
        ),
        ctl_vs_probs_cross_entropy=(
            scores["embeddings_only"].cross_entropy - scores["probs_only"].cross_entropy
        ),
        ctl_vs_projected_accuracy=(
            scores["embeddings_only"].accuracy - scores["probs_projected"].accuracy
        ),
        ctl_vs_projected_cross_entropy=(
            scores["embeddings_only"].cross_entropy
            - scores["probs_projected"].cross_entropy
        ),
        n_train=len(train_bank),
        n_tuning=len(tuning_bank),
        n_held_out=len(held_out_bank),
    )


# ---------------------------------------------------------------------------
# ICL
# ---------------------------------------------------------------------------


@dataclass
class ICLPairScore:
    """One ordered ``(concept_i, concept_j)`` pair's leakage score."""

    concept_i: str
    concept_j: str
    n: int
    auroc_embedding: Optional[float]
    auroc_probability: Optional[float]
    icl_raw: Optional[float]
    """``auroc_embedding - auroc_probability``; can be negative."""
    icl: Optional[float]
    """``icl_raw`` clipped at 0 -- the leakage reading."""


@dataclass
class ICLResult:
    """Inter-concept leakage: the full ordered-pair matrix plus summary stats."""

    pairs: List[ICLPairScore]
    mean_off_diagonal_icl: float
    top_pairs: List[ICLPairScore]
    n_train: int
    n_tuning: int
    n_held_out: int


def compute_icl(
    train_bank: LeakageBank,
    tuning_bank: LeakageBank,
    held_out_bank: LeakageBank,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 4096,
    patience: int = 2,
    seed: int = 0,
    device: str = "cpu",
    top_k: int = 10,
) -> ICLResult:
    """Fit and score every ordered concept pair's leakage; see the module docstring.

    See :func:`compute_ctl` on why ``batch_size`` (total optimizer steps),
    not ``epochs`` alone, is what determines convergence.
    """
    names = train_bank.concept_names
    num_concepts = len(names)
    pairs: List[ICLPairScore] = []
    for i, concept_i in enumerate(names):
        embed_head, _ = _fit_masked_multilabel_probe(
            train_bank.concept_embeddings.shape[-1],
            num_concepts,
            train_x=train_bank.concept_embeddings[:, i, :].float(),
            train_y=train_bank.concept_labels,
            train_observed=train_bank.concept_observed,
            tune_x=tuning_bank.concept_embeddings[:, i, :].float(),
            tune_y=tuning_bank.concept_labels,
            tune_observed=tuning_bank.concept_observed,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            device=device,
        )
        prob_head, _ = _fit_masked_multilabel_probe(
            1,
            num_concepts,
            train_x=train_bank.concept_probs[:, i : i + 1].float(),
            train_y=train_bank.concept_labels,
            train_observed=train_bank.concept_observed,
            tune_x=tuning_bank.concept_probs[:, i : i + 1].float(),
            tune_y=tuning_bank.concept_labels,
            tune_observed=tuning_bank.concept_observed,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            device=device,
        )
        with torch.no_grad():
            embed_logits = embed_head(held_out_bank.concept_embeddings[:, i, :].float())
            prob_logits = prob_head(held_out_bank.concept_probs[:, i : i + 1].float())
        for j, concept_j in enumerate(names):
            if j == i:
                continue
            obs = held_out_bank.concept_observed[:, j]
            n_obs = int(obs.sum().item())
            y = held_out_bank.concept_labels[obs, j].numpy()
            if n_obs < 2 or y.min() == y.max():
                pairs.append(
                    ICLPairScore(concept_i, concept_j, n_obs, None, None, None, None)
                )
                continue
            auroc_embed = float(
                roc_auc_score(y, embed_logits[obs, j].sigmoid().numpy())
            )
            auroc_prob = float(roc_auc_score(y, prob_logits[obs, j].sigmoid().numpy()))
            raw = auroc_embed - auroc_prob
            pairs.append(
                ICLPairScore(
                    concept_i,
                    concept_j,
                    n_obs,
                    auroc_embed,
                    auroc_prob,
                    raw,
                    max(0.0, raw),
                )
            )
        logger.info(
            "[leakage] ICL fit concept %d/%d (%s)", i + 1, num_concepts, concept_i
        )

    scored = [p for p in pairs if p.icl is not None and p.icl_raw is not None]
    icl_values = [p.icl for p in scored if p.icl is not None]
    mean_icl = sum(icl_values) / len(icl_values) if icl_values else float("nan")
    top = sorted(scored, key=lambda p: p.icl_raw or 0.0, reverse=True)[:top_k]

    return ICLResult(
        pairs=pairs,
        mean_off_diagonal_icl=mean_icl,
        top_pairs=top,
        n_train=len(train_bank),
        n_tuning=len(tuning_bank),
        n_held_out=len(held_out_bank),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _leakage_bank_from_shards(
    model: ConceptBottleneckSequenceModel,
    vocab: Vocabulary,
    binner: object,
    config: object,
    shard_dir: Union[str, Path],
    *,
    max_shards: Optional[int],
    sample_rate: float,
    seed: int,
    num_lanes: int,
    chunk_size: int,
    device: str,
    max_positions: Optional[int],
) -> LeakageBank:
    from odyssey.data.concepts import concepts_for_source  # noqa: PLC0415
    from odyssey.data.sidecars import activate_sidecars  # noqa: PLC0415
    from odyssey.training.data import (  # noqa: PLC0415
        build_concept_first_times,
        build_concept_label_dicts,
        build_visit_concept_first_times,
        build_visit_concept_label_dicts,
        load_meds_shard,
    )
    from odyssey.training.shard_stream import shard_paths  # noqa: PLC0415

    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    supervision: ConceptSupervision = getattr(config, "concept_supervision", "visit")
    activate_sidecars(shard_dir)
    concepts = concepts_for_source(source, task_set=task_set)
    concept_names = [c.name for c in concepts]

    banks: List[LeakageBank] = []
    kept = 0
    for k, path in enumerate(shard_paths(shard_dir, max_shards=max_shards)):
        raw = load_meds_shard(path)
        concept_labels: ConceptLabelDict
        concept_mask: ConceptLabelDict
        concept_first_times: ConceptLabelDict
        if supervision == "visit":
            concept_labels, concept_mask = build_visit_concept_label_dicts(
                raw, concepts
            )
            concept_first_times = build_visit_concept_first_times(raw, concepts)
        else:
            concept_labels, concept_mask = build_concept_label_dicts(raw, concepts)
            concept_first_times = build_concept_first_times(raw, concepts)
        binned = add_value_tokens(raw, binner, source=source)  # type: ignore[arg-type]
        del raw
        bank = collect_leakage_bank(
            model,
            binned,
            vocab,
            concept_labels=concept_labels,
            concept_mask=concept_mask,
            concept_first_times=concept_first_times,
            concept_names=concept_names,
            supervision=supervision,
            sample_rate=sample_rate,
            seed=seed * 7919 + k,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
            max_positions=(max_positions - kept) if max_positions else None,
        )
        del binned
        banks.append(bank)
        kept += len(bank)
        if max_positions is not None and kept >= max_positions:
            break
    return LeakageBank.concat(banks, max_positions)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Concept-task and inter-concept leakage probes (Track B item 11)."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--tuning-shard-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-train-shards", type=int, default=10)
    parser.add_argument("--max-tuning-shards", type=int, default=2)
    parser.add_argument("--max-held-out-shards", type=int, default=4)
    parser.add_argument("--train-sample-rate", type=float, default=0.1)
    parser.add_argument("--max-train-positions", type=int, default=2_000_000)
    parser.add_argument("--tuning-sample-rate", type=float, default=0.2)
    parser.add_argument("--max-tuning-positions", type=int, default=1_000_000)
    parser.add_argument("--held-out-sample-rate", type=float, default=0.3)
    parser.add_argument("--max-held-out-positions", type=int, default=3_000_000)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=4096,
        help=(
            "Optimizer steps are epochs * ceil(n_train / this), not epochs "
            "alone -- lower it for a small bank (see compute_ctl's docstring)."
        ),
    )
    parser.add_argument("--probe-patience", type=int, default=2)
    parser.add_argument("--num-lanes", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow clobbering an existing --output-json file (see refuse_existing_output).",
    )
    args = parser.parse_args()

    out = Path(args.output_json)
    refuse_existing_output(out, overwrite=args.overwrite, kind="leakage")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    model, vocab, binner, config = load_run(
        run_dir,
        device=device,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            "leakage metrics need a concept bottleneck; the run's model_kind is "
            f"{getattr(config, 'model_kind', 'bottleneck')!r}"
        )
    common = {
        "num_lanes": args.num_lanes,
        "chunk_size": args.chunk_size,
        "device": device,
    }
    train_bank = _leakage_bank_from_shards(
        model,
        vocab,
        binner,
        config,
        args.train_shard_dir,
        max_shards=args.max_train_shards,
        sample_rate=args.train_sample_rate,
        seed=args.seed,
        max_positions=args.max_train_positions,
        **common,
    )
    tuning_bank = _leakage_bank_from_shards(
        model,
        vocab,
        binner,
        config,
        args.tuning_shard_dir,
        max_shards=args.max_tuning_shards,
        sample_rate=args.tuning_sample_rate,
        seed=args.seed + 1,
        max_positions=args.max_tuning_positions,
        **common,
    )
    held_out_bank = _leakage_bank_from_shards(
        model,
        vocab,
        binner,
        config,
        args.held_out_shard_dir,
        max_shards=args.max_held_out_shards,
        sample_rate=args.held_out_sample_rate,
        seed=args.seed + 2,
        max_positions=args.max_held_out_positions,
        **common,
    )
    ctl = compute_ctl(
        train_bank,
        tuning_bank,
        held_out_bank,
        epochs=args.epochs,
        batch_size=args.probe_batch_size,
        patience=args.probe_patience,
        seed=args.seed,
        device=device,
    )
    icl = compute_icl(
        train_bank,
        tuning_bank,
        held_out_bank,
        epochs=args.epochs,
        batch_size=args.probe_batch_size,
        patience=args.probe_patience,
        seed=args.seed,
        device=device,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "concept_names": list(train_bank.concept_names),
        "banks": {
            "train": {
                "positions": len(train_bank),
                "seen": train_bank.n_positions_seen,
                "sample_rate": train_bank.sample_rate,
            },
            "tuning": {
                "positions": len(tuning_bank),
                "seen": tuning_bank.n_positions_seen,
                "sample_rate": tuning_bank.sample_rate,
            },
            "held_out": {
                "positions": len(held_out_bank),
                "seen": held_out_bank.n_positions_seen,
                "sample_rate": held_out_bank.sample_rate,
            },
        },
        "ctl": asdict(ctl),
        "icl": asdict(icl),
    }
    out.write_text(json.dumps(payload, indent=2))
    logger.info("[leakage] wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _main()
