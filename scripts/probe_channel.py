"""Channel probes: how much of the forecast runs through k versus the poles.

The mixture bottleneck builds each named slot as ``z_i = k_i w+_i +
(1 - k_i) w-_i``, where ``k_i`` is the concept probability shown to a
clinician and the poles ``w+_i``/``w-_i`` are free functions of the
hidden state. Zeroing every ``z_i`` collapses next-event top-1 to a few
percent, but that does not say whether the information lives in the
probabilities or in the poles. This script measures it directly on a
frozen checkpoint: one streaming pass over the held-out shards dumps
``k``, the unnamed slot's probability ``u``, the poles, and the unnamed
embedding ``z_u`` at every scored position (or a seeded, shard-
stratified subsample), then fresh readouts are fit on a SUBJECT-level
split and scored on held-out subjects for exact next-event top-1, the
paper's definition (argmax over the full vocabulary equals the next
token).

Readouts (each a fresh head, trained from scratch on frozen features):

- ``k_only``: the concept probabilities alone (Yeh et al. completeness).
- ``k_plus_u``: ``k`` plus the unnamed slot's probability ``u``.
- ``z_named``: the concatenated named embeddings, what the LM head reads
  minus the unnamed slot.
- ``poles_mean_k``: ``z`` recomputed with every ``k_i`` held at its
  training-split mean, so only the poles vary across positions.
- ``poles_both``: the raw ``[w+, w-]`` concatenation, the ceiling of what
  the poles carry regardless of ``k``.
- ``h_bar``: the full bottleneck ``[z, z_u]``, which should recover the
  model's own accuracy and serves as the ceiling.
- ``model_head``: the model's own LM head, no refit, the reference.

The readout is linear by default: a standardized ``nn.Linear`` over the
full next-event vocabulary trained with Adam and early stopping on a
tuning slice of the training subjects, the same probe family
:mod:`odyssey.inference.leakage` uses, chosen because sklearn's
multinomial logistic regression does not scale to a vocabulary of
thousands of classes over millions of rows. ``--mlp`` swaps in a one-
hidden-layer MLP. Every accuracy carries a subject-clustered bootstrap
interval (1000 resamples, seeded) and its ratio to the model's own
accuracy on the same rows ("retained"), with a paired interval on the
ratio. The Yeh-style completeness score is also reported, with the
majority next-event rate as the random baseline (the binary
:func:`odyssey.training.metrics.compute_completeness` does not apply to a
vocabulary-sized target, but its formula does).

Reading the result: if ``k_only`` recovers most of ``h_bar``, the forecast
runs through the named STATES. If ``poles_mean_k`` does, it runs through
the poles, and the concept probabilities are a small part of the channel.

The CTL leakage probes of :mod:`odyssey.inference.leakage` (probs-only,
known-embeddings, residual, random-projected probs, on the next-token
code family) are fit on the same subject split from the same dump, via
``compute_ctl``. Hazard-head readouts at landmark positions are NOT
dumped: the landmark protocol (v4, one information boundary per landmark)
lives inside :mod:`odyssey.inference.alerts` and reproducing it here
would risk a number that disagrees with the paper's alert tables.

Usage::

    uv run python scripts/probe_channel.py \\
        --run-dir ~/runs/full_run_v10 \\
        --held-out-shard-dir ~/data/mimic/held_out \\
        --output-json ~/runs/full_run_v10/channel_probes.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import PatientSequence
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.inference.leakage import (
    CTLResult,
    LeakageBank,
    ProbeFitTrace,
    _StandardizedLinearProbe,
    compute_ctl,
)
from odyssey.inference.legacy_concept_pins import resolve_concepts_for_run
from odyssey.inference.run_inference import (
    _build_type_lookup,
    load_run,
    refuse_existing_output,
)
from odyssey.models.concept_bottleneck import ConceptBottleneck
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
)
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
    build_visit_concept_first_times,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shard,
)
from odyssey.training.running_labels import position_running_labels
from odyssey.training.shard_stream import shard_paths
from odyssey.training.train import TrainingConfig, _move_chunk_to_device


logger = logging.getLogger("probe_channel")

READOUT_NAMES = (
    "k_only",
    "k_plus_u",
    "z_named",
    "poles_mean_k",
    "poles_both",
    "h_bar",
)
MODEL_READOUT = "model_head"


# ---------------------------------------------------------------------------
# Feature bank
# ---------------------------------------------------------------------------


@dataclass
class ChannelBank:
    """Frozen bottleneck pieces at scored positions, one row per position.

    Float tensors are float16 at rest; readouts upcast per batch. ``z`` is
    not stored: it is ``k w+ + (1 - k) w-`` and is recomputed from the
    poles, which is what lets the ``poles_mean_k`` readout swap ``k`` out.
    """

    concept_probs: torch.Tensor
    """``(N, k)`` the concept probabilities."""
    unknown_prob: torch.Tensor
    """``(N,)`` the unnamed slot's mixing probability."""
    known_pos: torch.Tensor
    """``(N, k, d)`` the ``w+`` poles."""
    known_neg: torch.Tensor
    """``(N, k, d)`` the ``w-`` poles."""
    unknown_embedding: torch.Tensor
    """``(N, u)`` the unnamed slot's mixed embedding ``z_u``."""
    targets: torch.Tensor
    """``(N,)`` long: the next event token id."""
    subject_ids: torch.Tensor
    """``(N,)`` long."""
    model_pred: torch.Tensor
    """``(N,)`` long: the model's own top-1 next event."""
    family_labels: torch.Tensor
    """``(N,)`` long: the next token's code family (the CTL target)."""
    concept_labels: torch.Tensor
    """``(N, k)`` float32 running concept labels (for the CTL bank)."""
    concept_observed: torch.Tensor
    """``(N, k)`` bool."""
    shard_index: torch.Tensor
    """``(N,)`` long: which held-out shard the row came from."""
    concept_names: tuple[str, ...]
    n_positions_seen: int

    _TENSORS = (
        "concept_probs",
        "unknown_prob",
        "known_pos",
        "known_neg",
        "unknown_embedding",
        "targets",
        "subject_ids",
        "model_pred",
        "family_labels",
        "concept_labels",
        "concept_observed",
        "shard_index",
    )

    def __len__(self) -> int:
        """Return the number of banked positions."""
        return int(self.targets.numel())

    def _map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> ChannelBank:
        return _build_bank(
            {name: fn(getattr(self, name)) for name in self._TENSORS},
            concept_names=self.concept_names,
            n_positions_seen=self.n_positions_seen,
        )

    def to(self, device: str) -> ChannelBank:
        """Copy every tensor to ``device``."""
        return self._map(lambda t: t.to(device))

    def subset(self, idx: torch.Tensor) -> ChannelBank:
        """Rows ``idx`` (a long index tensor on this bank's device)."""
        return self._map(lambda t: t[idx])

    @staticmethod
    def concat(banks: Sequence[ChannelBank]) -> ChannelBank:
        """Stack per-shard banks with the same concept names."""
        if not banks:
            raise ValueError("no banks to concatenate")
        names = banks[0].concept_names
        if any(b.concept_names != names for b in banks):
            raise ValueError("banks disagree on concept_names")
        return _build_bank(
            {
                name: torch.cat([getattr(b, name) for b in banks])
                for name in ChannelBank._TENSORS
            },
            concept_names=names,
            n_positions_seen=sum(b.n_positions_seen for b in banks),
        )

    def mixed(self, idx: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """``(n, k * d)`` named embeddings for rows ``idx`` mixed with ``probs``."""
        p = probs.float().unsqueeze(-1)
        w_pos = self.known_pos[idx].float()
        w_neg = self.known_neg[idx].float()
        return (p * w_pos + (1.0 - p) * w_neg).flatten(1)


def _build_bank(
    t: dict[str, torch.Tensor], *, concept_names: tuple[str, ...], n_positions_seen: int
) -> ChannelBank:
    """Construct a bank from a name-keyed tensor dict (typed field by field)."""
    return ChannelBank(
        concept_probs=t["concept_probs"],
        unknown_prob=t["unknown_prob"],
        known_pos=t["known_pos"],
        known_neg=t["known_neg"],
        unknown_embedding=t["unknown_embedding"],
        targets=t["targets"],
        subject_ids=t["subject_ids"],
        model_pred=t["model_pred"],
        family_labels=t["family_labels"],
        concept_labels=t["concept_labels"],
        concept_observed=t["concept_observed"],
        shard_index=t["shard_index"],
        concept_names=concept_names,
        n_positions_seen=n_positions_seen,
    )


def collect_channel_bank(  # noqa: PLR0915 -- one linear streaming pass
    model: ConceptBottleneckSequenceModel,
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
    max_positions: int | None = None,
    shard_index: int = 0,
) -> ChannelBank:
    """One frozen streaming pass; bank the bottleneck pieces per position.

    Scored positions are those with a real, non-padding next-token
    target, the same rows the model's own top-1 is computed on. Each is
    kept with probability ``sample_rate`` (seeded); if more than
    ``max_positions`` survive, a seeded random subset of that size is
    kept, so the subsample is random within the shard rather than the
    first rows streamed.
    """
    model.eval()
    bottleneck = model.bottleneck
    if not isinstance(bottleneck, ConceptBottleneck):
        raise ValueError(
            "channel probes need the mixture bottleneck (poles are functions "
            f"of the hidden state); got {type(bottleneck).__name__}"
        )
    num_concepts = bottleneck.num_concepts
    if len(concept_names) != num_concepts:
        raise ValueError(
            f"{len(concept_names)} concept names but the bottleneck has "
            f"{num_concepts} concepts"
        )
    gen = torch.Generator().manual_seed(seed)
    type_lookup = _build_type_lookup(vocab, device)
    patients: Iterator[PatientSequence] = iter_patient_sequences(events_binned, vocab)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in ChannelBank._TENSORS}
    seen = 0
    checked = False
    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            hidden, state = model.backbone(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            out = bottleneck(hidden)
            mix = bottleneck.mixture_parts(hidden)
            if not checked:
                # The poles must reproduce the forward pass's own mixing.
                p = out.concept_probs.unsqueeze(-1)
                rebuilt = p * mix.known_pos + (1.0 - p) * mix.known_neg
                if not torch.allclose(rebuilt, out.concept_embeddings, atol=1e-4):
                    raise RuntimeError(
                        "mixture_parts does not reproduce concept_embeddings"
                    )
                checked = True
            pred = model.lm_head(out.bottleneck).argmax(dim=-1)
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
            valid_cpu = valid.to(labels.device)
            parts["concept_probs"].append(out.concept_probs[valid].half().cpu())
            parts["unknown_prob"].append(
                torch.sigmoid(mix.unknown_logit)[valid].half().cpu()
            )
            parts["known_pos"].append(mix.known_pos[valid].half().cpu())
            parts["known_neg"].append(mix.known_neg[valid].half().cpu())
            parts["unknown_embedding"].append(out.unknown_embedding[valid].half().cpu())
            parts["targets"].append(chunk.targets[valid].long().cpu())
            parts["subject_ids"].append(chunk.subject_ids[valid].long().cpu())
            parts["model_pred"].append(pred[valid].long().cpu())
            parts["family_labels"].append(
                type_lookup[chunk.targets][valid].long().cpu()
            )
            parts["concept_labels"].append(labels[valid_cpu].float())
            parts["concept_observed"].append(observed[valid_cpu].bool())
            parts["shard_index"].append(
                torch.full((int(valid.sum().item()),), shard_index, dtype=torch.long)
            )
    if not parts["targets"]:
        raise ValueError("no scored positions collected; empty split?")
    bank = _build_bank(
        {name: torch.cat(parts[name]) for name in ChannelBank._TENSORS},
        concept_names=tuple(concept_names),
        n_positions_seen=seen,
    )
    if max_positions is not None and len(bank) > max_positions:
        keep = torch.randperm(len(bank), generator=gen)[:max_positions]
        bank = bank.subset(keep)
    logger.info(
        "[probe_channel] shard %d: banked %d of %d scored positions",
        shard_index,
        len(bank),
        seen,
    )
    return bank


def bank_from_shards(
    model: ConceptBottleneckSequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner,
    config: TrainingConfig,
    shard_dir: str | Path,
    *,
    run_dir: str | Path,
    max_shards: int | None,
    sample_rate: float,
    seed: int,
    num_lanes: int,
    chunk_size: int,
    device: str,
    max_positions: int | None,
) -> ChannelBank:
    """Bank every held-out shard, stratified: an equal position cap per shard.

    Data preparation matches ``evaluate_interventions`` (normalization,
    history recap, sidecars, the run's own concept set, the train-fit
    binner) so ``model_head`` reproduces the paper's evaluation rows.
    """
    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    supervision: ConceptSupervision = getattr(config, "concept_supervision", "visit")
    activate_sidecars(shard_dir)
    concepts = resolve_concepts_for_run(str(run_dir), source, task_set)
    concept_names = [c.name for c in concepts]
    paths = shard_paths(shard_dir, max_shards=max_shards)
    if not paths:
        raise ValueError(f"no shards under {shard_dir}")
    per_shard_cap = (
        int(math.ceil(max_positions / len(paths))) if max_positions else None
    )
    banks: list[ChannelBank] = []
    for k, path in enumerate(paths):
        raw = load_meds_shard(path)
        raw = maybe_normalize(
            raw,
            enabled=getattr(config, "normalize_medications", False),
            source=source,
        )
        raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
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
        binned = add_value_tokens(raw, binner, source=source)
        del raw
        banks.append(
            collect_channel_bank(
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
                max_positions=per_shard_cap,
                shard_index=k,
            )
        )
        del binned
    return ChannelBank.concat(banks)


# ---------------------------------------------------------------------------
# Subject-level split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectSplit:
    """Row indices of one bank, partitioned by subject."""

    train: torch.Tensor
    tune: torch.Tensor
    test: torch.Tensor
    n_subjects: dict[str, int]


def split_by_subject(
    subject_ids: torch.Tensor,
    *,
    seed: int,
    train_frac: float = 0.7,
    tune_frac: float = 0.1,
) -> SubjectSplit:
    """Partition rows so no subject appears in two parts (seeded)."""
    if train_frac <= 0 or tune_frac < 0 or train_frac + tune_frac >= 1.0:
        raise ValueError("need 0 < train_frac, 0 <= tune_frac, sum < 1")
    subjects = torch.unique(subject_ids.cpu())
    if subjects.numel() < 3:
        raise ValueError(f"need at least 3 subjects to split, got {subjects.numel()}")
    gen = torch.Generator().manual_seed(seed)
    order = subjects[torch.randperm(subjects.numel(), generator=gen)]
    n = order.numel()
    n_train = max(1, int(round(n * train_frac)))
    n_tune = max(1, int(round(n * tune_frac))) if tune_frac > 0 else 0
    n_train = min(n_train, n - n_tune - 1)
    parts = {
        "train": order[:n_train],
        "tune": order[n_train : n_train + n_tune],
        "test": order[n_train + n_tune :],
    }
    sids = subject_ids.cpu()
    idx = {
        name: torch.nonzero(torch.isin(sids, members)).flatten()
        for name, members in parts.items()
    }
    return SubjectSplit(
        train=idx["train"],
        tune=idx["tune"],
        test=idx["test"],
        n_subjects={name: int(members.numel()) for name, members in parts.items()},
    )


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


FeatureFn = Callable[[ChannelBank, torch.Tensor], torch.Tensor]


def readout_features(
    bank: ChannelBank, train_idx: torch.Tensor
) -> dict[str, FeatureFn]:
    """Return the feature map of every refit readout, keyed by name.

    ``poles_mean_k`` needs the training-split mean of ``k`` per concept,
    computed here once so the test split never informs it.
    """
    k_mean = bank.concept_probs[train_idx].float().mean(dim=0)

    def k_only(b: ChannelBank, idx: torch.Tensor) -> torch.Tensor:
        return b.concept_probs[idx].float()

    def k_plus_u(b: ChannelBank, idx: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [b.concept_probs[idx].float(), b.unknown_prob[idx].float().unsqueeze(-1)],
            dim=-1,
        )

    def z_named(b: ChannelBank, idx: torch.Tensor) -> torch.Tensor:
        return b.mixed(idx, b.concept_probs[idx])

    def poles_mean_k(b: ChannelBank, idx: torch.Tensor) -> torch.Tensor:
        return b.mixed(idx, k_mean.to(b.concept_probs.device).expand(idx.numel(), -1))

    def poles_both(b: ChannelBank, idx: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [b.known_pos[idx].float().flatten(1), b.known_neg[idx].float().flatten(1)],
            dim=-1,
        )

    def h_bar(b: ChannelBank, idx: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [b.mixed(idx, b.concept_probs[idx]), b.unknown_embedding[idx].float()],
            dim=-1,
        )

    return {
        "k_only": k_only,
        "k_plus_u": k_plus_u,
        "z_named": z_named,
        "poles_mean_k": poles_mean_k,
        "poles_both": poles_both,
        "h_bar": h_bar,
    }


class _StandardizedMLPProbe(nn.Module):
    """One hidden layer behind the same fixed standardization as the linear probe."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        hidden: int = 512,
    ) -> None:
        """Standardize, then ``Linear -> GELU -> Linear``."""
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.GELU(), nn.Linear(hidden, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize then apply the MLP."""
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        out: torch.Tensor = self.net((x - mean) / std)
        return out


def _batches(idx: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, idx.numel(), batch_size):
        yield idx[start : start + batch_size]


def _running_stats(
    bank: ChannelBank, idx: torch.Tensor, feature_fn: FeatureFn, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature (mean, std) over ``idx`` without materializing every row."""
    total = 0
    s1: torch.Tensor | None = None
    s2: torch.Tensor | None = None
    for b in _batches(idx, batch_size):
        x = feature_fn(bank, b).double()
        s1 = x.sum(dim=0) if s1 is None else s1 + x.sum(dim=0)
        s2 = (x * x).sum(dim=0) if s2 is None else s2 + (x * x).sum(dim=0)
        total += x.shape[0]
    assert s1 is not None and s2 is not None  # noqa: S101 -- idx is non-empty
    mean = s1 / total
    var = (s2 / total - mean * mean).clamp_min(0.0)
    if total > 1:
        var = var * total / (total - 1)
    std = var.sqrt().clamp_min(1e-6)
    return mean.float().unsqueeze(0), std.float().unsqueeze(0)


def fit_readout(
    bank: ChannelBank,
    train_idx: torch.Tensor,
    tune_idx: torch.Tensor,
    feature_fn: FeatureFn,
    num_classes: int,
    *,
    mlp: bool = False,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[nn.Module, ProbeFitTrace]:
    """Adam on the training rows, early-stopped on tuning cross-entropy.

    Features are built per batch from the bank, so the poles are read
    once per epoch and never expanded to a full ``(N, k * d)`` matrix.
    """
    torch.manual_seed(seed)
    if train_idx.numel() < batch_size:
        # A small bank needs more optimizer steps per epoch to converge.
        batch_size = max(32, train_idx.numel() // 8)
    mean, std = _running_stats(bank, train_idx, feature_fn, batch_size)
    in_features = int(mean.shape[1])
    head: nn.Module = (
        _StandardizedMLPProbe(in_features, num_classes, mean, std)
        if mlp
        else _StandardizedLinearProbe(in_features, num_classes, mean, std)
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    trace = ProbeFitTrace()
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best = float("inf")
    bad = 0
    gen = torch.Generator(device=train_idx.device).manual_seed(seed)
    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        perm = train_idx[
            torch.randperm(train_idx.numel(), generator=gen, device=train_idx.device)
        ]
        for b in _batches(perm, batch_size):
            opt.zero_grad()
            x = feature_fn(bank, b).to(device)
            y = bank.targets[b].to(device)
            loss = F.cross_entropy(head(x), y)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        head.eval()
        with torch.no_grad():
            tune_loss_sum = 0.0
            for b in _batches(tune_idx, batch_size):
                x = feature_fn(bank, b).to(device)
                y = bank.targets[b].to(device)
                tune_loss_sum += float(
                    F.cross_entropy(head(x), y, reduction="sum").item()
                )
            tune_loss = tune_loss_sum / max(1, tune_idx.numel())
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
def score_readout(
    head: nn.Module,
    bank: ChannelBank,
    idx: torch.Tensor,
    feature_fn: FeatureFn,
    *,
    batch_size: int = 4096,
    device: str = "cpu",
) -> torch.Tensor:
    """``(n,)`` bool on CPU: exact next-event top-1 hit per row of ``idx``."""
    head.eval()
    hits = []
    for b in _batches(idx, batch_size):
        x = feature_fn(bank, b).to(device)
        y = bank.targets[b].to(device)
        hits.append((head(x).argmax(dim=-1) == y).cpu())
    return torch.cat(hits)


# ---------------------------------------------------------------------------
# Bootstrap and scoring
# ---------------------------------------------------------------------------


@dataclass
class ReadoutScore:
    """One readout's held-out accuracy and its intervals."""

    readout: str
    n_positions: int
    n_subjects: int
    top1_accuracy: float
    ci95: tuple[float, float]
    retained: float
    """``top1_accuracy / model top1_accuracy`` on the same rows."""
    retained_ci95: tuple[float, float]
    """Paired subject-clustered interval on ``retained``."""
    completeness_score: float
    """Yeh-style: ``(acc - majority) / (model_acc - majority)``."""
    in_features: int | None = None
    parameters: int | None = None
    fit: ProbeFitTrace | None = None


def subject_bootstrap(
    subject_ids: np.ndarray,
    hits: dict[str, np.ndarray],
    reference: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Subject-clustered percentile intervals on accuracy and on the ratio.

    One subject multiset is drawn per resample and reused for every
    readout, so the interval on ``readout / reference`` is paired.
    """
    _, inverse = np.unique(subject_ids, return_inverse=True)
    n_subj = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_subj).astype(np.float64)
    per_subject = {
        name: np.bincount(inverse, weights=h.astype(np.float64), minlength=n_subj)
        for name, h in hits.items()
    }
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=n_boot)
    weights = weights.astype(np.float64)
    denom = weights @ counts
    accs = {name: (weights @ per_subject[name]) / denom for name in hits}
    ref = accs[reference]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for name, acc in accs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ref > 0, acc / ref, np.nan)
        out[name] = {
            "accuracy": (
                float(np.percentile(acc, 2.5)),
                float(np.percentile(acc, 97.5)),
            ),
            "retained": (
                float(np.nanpercentile(ratio, 2.5)),
                float(np.nanpercentile(ratio, 97.5)),
            ),
        }
    return out


def _majority_rate(targets: torch.Tensor) -> float:
    counts = torch.bincount(targets.cpu())
    return float(counts.max().item() / targets.numel())


def _completeness(acc: float, model_acc: float, majority: float) -> float:
    denom = model_acc - majority
    return (acc - majority) / denom if denom > 1e-9 else float("nan")


# ---------------------------------------------------------------------------
# CTL bridge
# ---------------------------------------------------------------------------


def leakage_bank(bank: ChannelBank, idx: torch.Tensor) -> LeakageBank:
    """Rows ``idx`` as a :class:`LeakageBank` (named embeddings materialized)."""
    sub = bank.subset(idx)
    k, d = sub.known_pos.shape[1], sub.known_pos.shape[2]
    z = sub.mixed(torch.arange(len(sub), device=sub.targets.device), sub.concept_probs)
    return LeakageBank(
        sub.concept_probs.half().cpu(),
        z.reshape(len(sub), k, d).half().cpu(),
        sub.unknown_embedding.half().cpu(),
        sub.family_labels.cpu(),
        sub.concept_labels.float().cpu(),
        sub.concept_observed.bool().cpu(),
        sub.concept_names,
        n_positions_seen=len(sub),
        sample_rate=1.0,
    )


def _cap(idx: torch.Tensor, cap: int | None, seed: int) -> torch.Tensor:
    if cap is None or idx.numel() <= cap:
        return idx
    gen = torch.Generator().manual_seed(seed)
    keep = torch.randperm(idx.numel(), generator=gen)[:cap].to(idx.device)
    return idx[keep]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ProbeOptions:
    """Everything that shapes the dump and the fits."""

    max_positions: int | None = 2_000_000
    max_shards: int | None = None
    sample_rate: float = 1.0
    seed: int = 0
    num_lanes: int = 16
    chunk_size: int = 512
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 1e-3
    patience: int = 2
    mlp: bool = False
    n_boot: int = 1000
    train_frac: float = 0.7
    tune_frac: float = 0.1
    skip_ctl: bool = False
    ctl_max_positions: int | None = 500_000
    ctl_epochs: int = 20
    bank_on_cpu: bool = False
    checkpoint: str | None = None
    readouts: tuple[str, ...] = READOUT_NAMES
    notes: list[str] = field(default_factory=list)


def _fit_all_readouts(
    bank: ChannelBank,
    *,
    split_idx: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    hits: dict[str, torch.Tensor],
    num_classes: int,
    opts: ProbeOptions,
    device: str,
) -> dict[str, dict[str, Any]]:
    """Fit and score every requested readout; fill ``hits``, return fit metadata."""
    train_idx, tune_idx, test_idx = split_idx
    features = readout_features(bank, train_idx)
    meta: dict[str, dict[str, Any]] = {}
    for name in opts.readouts:
        if name not in features:
            raise ValueError(f"unknown readout {name!r}; known: {READOUT_NAMES}")
        fn = features[name]
        head, trace = fit_readout(
            bank,
            train_idx,
            tune_idx,
            fn,
            num_classes,
            mlp=opts.mlp,
            epochs=opts.epochs,
            batch_size=opts.batch_size,
            lr=opts.lr,
            patience=opts.patience,
            seed=opts.seed,
            device=device,
        )
        hits[name] = score_readout(
            head, bank, test_idx, fn, batch_size=opts.batch_size, device=device
        )
        meta[name] = {
            "in_features": int(cast(torch.Tensor, head.mean).shape[1]),
            "parameters": sum(p.numel() for p in head.parameters()),
            "fit": trace,
        }
        logger.info(
            "[probe_channel] %-13s top-1 %.4f in %.1fs",
            name,
            float(hits[name].float().mean().item()),
            trace.seconds,
        )
        del head
    return meta


def run_channel_probes(
    run_dir: str | Path,
    held_out_shard_dir: str | Path,
    *,
    options: ProbeOptions | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load the run, dump the bank, fit every readout, return the JSON payload."""
    opts = options or ProbeOptions()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / (opts.checkpoint or "checkpoint_best.pt")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            "this probe needs a concept bottleneck; the run's model_kind is "
            f"{getattr(config, 'model_kind', 'bottleneck')!r}"
        )
    if not isinstance(model.bottleneck, ConceptBottleneck):
        raise ValueError(
            "channel probes are defined for the mixture bottleneck, whose poles "
            "are functions of the hidden state; the run's bottleneck_kind is "
            f"{getattr(config, 'bottleneck_kind', 'mixture')!r}"
        )
    notes = list(opts.notes)
    if model.bottleneck.global_pairs:
        notes.append(
            "global_pairs=True: the poles are input-independent parameters, so "
            "poles_mean_k is a constant feature and reads the majority class"
        )
    notes.append(
        "hazard-head readouts at landmarks are not dumped; see the module docstring"
    )

    bank = bank_from_shards(
        model,
        vocab,
        binner,
        config,
        held_out_shard_dir,
        run_dir=run_dir,
        max_shards=opts.max_shards,
        sample_rate=opts.sample_rate,
        seed=opts.seed,
        num_lanes=opts.num_lanes,
        chunk_size=opts.chunk_size,
        device=device,
        max_positions=opts.max_positions,
    )
    del model
    logger.info(
        "[probe_channel] bank: %d positions of %d seen, %d subjects, %d shards",
        len(bank),
        bank.n_positions_seen,
        int(torch.unique(bank.subject_ids).numel()),
        int(torch.unique(bank.shard_index).numel()),
    )
    split = split_by_subject(
        bank.subject_ids,
        seed=opts.seed,
        train_frac=opts.train_frac,
        tune_frac=opts.tune_frac,
    )
    bank_device = "cpu" if opts.bank_on_cpu else device
    bank = bank.to(bank_device)
    train_idx = split.train.to(bank_device)
    tune_idx = split.tune.to(bank_device)
    test_idx = split.test.to(bank_device)

    test_targets = bank.targets[test_idx]
    model_hits = (bank.model_pred[test_idx] == test_targets).cpu()
    model_acc = float(model_hits.float().mean().item())
    majority = _majority_rate(test_targets)
    num_classes = len(vocab)
    hits: dict[str, torch.Tensor] = {MODEL_READOUT: model_hits}
    meta = _fit_all_readouts(
        bank,
        split_idx=(train_idx, tune_idx, test_idx),
        hits=hits,
        num_classes=num_classes,
        opts=opts,
        device=device,
    )

    subject_np = bank.subject_ids[test_idx].cpu().numpy()
    hits_np = {name: h.numpy() for name, h in hits.items()}
    intervals = subject_bootstrap(
        subject_np, hits_np, MODEL_READOUT, n_boot=opts.n_boot, seed=opts.seed
    )
    n_test_subjects = int(np.unique(subject_np).size)
    scores: dict[str, ReadoutScore] = {}
    for name, h in hits_np.items():
        acc = float(h.mean())
        extra = meta.get(name, {})
        scores[name] = ReadoutScore(
            readout=name,
            n_positions=int(h.size),
            n_subjects=n_test_subjects,
            top1_accuracy=acc,
            ci95=intervals[name]["accuracy"],
            retained=acc / model_acc if model_acc > 0 else float("nan"),
            retained_ci95=intervals[name]["retained"],
            completeness_score=_completeness(acc, model_acc, majority),
            in_features=extra.get("in_features"),
            parameters=extra.get("parameters"),
            fit=extra.get("fit"),
        )

    ctl: CTLResult | None = None
    if not opts.skip_ctl:
        cap = opts.ctl_max_positions
        ctl = compute_ctl(
            leakage_bank(bank, _cap(train_idx, cap, opts.seed + 11)).to(device),
            leakage_bank(bank, _cap(tune_idx, cap, opts.seed + 12)).to(device),
            leakage_bank(bank, _cap(test_idx, cap, opts.seed + 13)).to(device),
            epochs=opts.ctl_epochs,
            batch_size=opts.batch_size,
            patience=opts.patience,
            seed=opts.seed,
            device=device,
        )

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "held_out_shard_dir": str(held_out_shard_dir),
        "concept_names": list(bank.concept_names),
        "protocol": {
            "readout": "mlp" if opts.mlp else "linear",
            "target": "exact next event over the full vocabulary (top-1)",
            "split": "by subject",
            "train_frac": opts.train_frac,
            "tune_frac": opts.tune_frac,
            "seed": opts.seed,
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "lr": opts.lr,
            "n_boot": opts.n_boot,
            "max_positions": opts.max_positions,
            "sample_rate": opts.sample_rate,
            "max_shards": opts.max_shards,
            "ctl_max_positions": opts.ctl_max_positions,
        },
        "n_positions": {
            "seen": bank.n_positions_seen,
            "bank": len(bank),
            "train": int(train_idx.numel()),
            "tune": int(tune_idx.numel()),
            "test": int(test_idx.numel()),
        },
        "n_subjects": {
            "bank": int(torch.unique(bank.subject_ids).numel()),
            **split.n_subjects,
        },
        "n_shards": int(torch.unique(bank.shard_index).numel()),
        "vocab_size": num_classes,
        "majority_class_accuracy": majority,
        "model_top1_accuracy_all_banked": float(
            (bank.model_pred == bank.targets).float().mean().item()
        ),
        "model": asdict(scores[MODEL_READOUT]),
        "readouts": {name: asdict(scores[name]) for name in opts.readouts},
        "ctl": None if ctl is None else asdict(ctl),
        "hazards": None,
        "notes": notes,
    }
    return payload


def markdown_table(payload: dict[str, Any]) -> str:
    """Render a small table of every readout for the terminal."""
    rows = [
        "| readout | top-1 | 95% CI | retained | completeness | features |",
        "|---|---|---|---|---|---|",
    ]
    entries = [payload["model"], *payload["readouts"].values()]
    for s in entries:
        lo, hi = s["ci95"]
        feats = "" if s["in_features"] is None else str(s["in_features"])
        rows.append(
            f"| {s['readout']} | {100 * s['top1_accuracy']:.2f} | "
            f"[{100 * lo:.2f}, {100 * hi:.2f}] | {s['retained']:.3f} | "
            f"{s['completeness_score']:.3f} | {feats} |"
        )
    return "\n".join(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("Usage::")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-positions", type=int, default=2_000_000)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-lanes", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--mlp", action="store_true", help="one-hidden-layer readout")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--tune-frac", type=float, default=0.1)
    parser.add_argument("--skip-ctl", action="store_true")
    parser.add_argument("--ctl-max-positions", type=int, default=500_000)
    parser.add_argument("--ctl-epochs", type=int, default=20)
    parser.add_argument(
        "--bank-on-cpu",
        action="store_true",
        help="keep the dump in host memory and move batches to the GPU",
    )
    parser.add_argument("--readouts", nargs="*", default=list(READOUT_NAMES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = _parse_args(argv)
    out = Path(args.output_json)
    refuse_existing_output(out, overwrite=args.overwrite, kind="channel probes")
    options = ProbeOptions(
        max_positions=args.max_positions or None,
        max_shards=args.max_shards,
        sample_rate=args.sample_rate,
        seed=args.seed,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        mlp=args.mlp,
        n_boot=args.n_boot,
        train_frac=args.train_frac,
        tune_frac=args.tune_frac,
        skip_ctl=args.skip_ctl,
        ctl_max_positions=args.ctl_max_positions or None,
        ctl_epochs=args.ctl_epochs,
        bank_on_cpu=args.bank_on_cpu,
        checkpoint=args.checkpoint,
        readouts=tuple(args.readouts),
    )
    payload = run_channel_probes(args.run_dir, args.held_out_shard_dir, options=options)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    logger.info("[probe_channel] wrote %s", out)
    print(markdown_table(payload))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
