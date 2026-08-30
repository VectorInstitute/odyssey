"""Distributional time-head probe on a frozen backbone (roadmap item 13, N2).

The trained models forecast *when* the next bundle arrives with a
discrete-time hazard over log-spaced bins (:mod:`odyssey.models.time_to_event`).
Its horizon calibration is already within a fraction of a point, so an
alternative head has to beat a strong baseline, not rescue a weak one.
This probe asks, without retraining anything: on the SAME frozen features
the flagship's heads read, which parameterization of the gap distribution
fits best?

- ``hazard``: the production head refit (linear hazard per bin) -- the
  control for head capacity / optimizer effects.
- ``hazard_mlp``: the same hazard bins behind a small MLP -- is any gain
  capacity or parameterization?
- ``categorical``: a softmax over the same bins (Doctor AI's
  discretized-grid framing, modernized: the bins are log-spaced and the
  distribution is read off directly).
- ``lognormal<k>``: a zero-gap Bernoulli plus a ``k``-component log-normal
  mixture over positive gaps -- a smooth predictive density with
  closed-form quantiles (clinician-facing intervals) and the sampling
  machinery counterfactual rollouts need.

Every head is scored on the same discrete-bin negative log-likelihood
(the mixture's bin masses are exact CDF differences), so the comparison
is in one currency (note: the ``hazard`` control treats the open bin as
absorbing so its masses sum to 1, whereas the production likelihood in
:func:`odyssey.models.time_to_event.hazard_log_likelihood` keeps a
deliberate leftover ``(1 - h_last) * S(last-1)`` -- probe NLLs are
comparable among heads, not digit-for-digit with registry time NLLs),
plus same-instant accuracy and P(within 1h/8h/24h) calibration including
the after-bundle cells the design doc singles out;
the mixture additionally reports its continuous log-likelihood and the
median-gap absolute error. Features are collected once per split from a
streaming pass (positions Bernoulli-subsampled to bound memory), heads
are fit on the train bank with early stopping on the tuning bank, and
reported on the held-out bank. Decision rule (docs/track_b_designs.md):
a head graduates to a flagship-run A/B only if it beats the refit hazard
on after-bundle-1h calibration AND overall NLL.
"""

import argparse
import json
import logging
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from odyssey.data.sequences import PatientSequence
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.models.sequence_model import SequenceModel
from odyssey.models.time_to_event import (
    DEFAULT_TIME_BIN_EDGES_HOURS,
    gap_survival_valid_mask,
    gap_to_bin,
)
from odyssey.training.data import iter_patient_sequences
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

HORIZONS: dict[str, float] = {"1h": 1.0, "8h": 8.0, "24h": 24.0}
DEFAULT_HEADS: tuple[str, ...] = ("hazard", "hazard_mlp", "categorical", "lognormal3")


# ---------------------------------------------------------------------------
# Feature banks
# ---------------------------------------------------------------------------


@dataclass
class FeatureBank:
    """Frozen head-input features with their gap targets for one split."""

    features: torch.Tensor
    """``(N, D)`` float16 at rest (banks are large), on CPU."""
    gaps: torch.Tensor
    """``(N,)`` hours to the next event (0 = same bundle)."""
    n_positions_seen: int
    sample_rate: float

    def __len__(self) -> int:
        """Return the number of banked positions."""
        return int(self.gaps.numel())

    def to(self, device: str) -> "FeatureBank":
        """Copy to ``device`` (for fitting); features stay fp16 until batched."""
        return FeatureBank(
            self.features.to(device),
            self.gaps.to(device),
            self.n_positions_seen,
            self.sample_rate,
        )

    @staticmethod
    def concat(
        banks: Sequence["FeatureBank"], max_positions: int | None
    ) -> "FeatureBank":
        """Stack per-shard banks (same sample rate), capped at ``max_positions``."""
        if not banks:
            raise ValueError("no banks to concatenate")
        feats = torch.cat([b.features for b in banks])
        gaps = torch.cat([b.gaps for b in banks])
        if max_positions is not None:
            feats, gaps = feats[:max_positions], gaps[:max_positions]
        return FeatureBank(
            feats,
            gaps,
            n_positions_seen=sum(b.n_positions_seen for b in banks),
            sample_rate=banks[0].sample_rate,
        )


def collect_feature_bank(
    model: SequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    sample_rate: float = 1.0,
    seed: int = 0,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cpu",
    max_positions: int | None = None,
) -> FeatureBank:
    """One frozen streaming pass; bank head features and gap targets.

    Positions are those with a valid in-chunk next token (the exact set
    the training loss and :class:`_RunningTimeMetrics` score), kept with
    probability ``sample_rate`` (seeded) and capped at ``max_positions``.
    """
    from odyssey.data.streaming import PackedLaneSampler  # noqa: PLC0415

    model.eval()
    gen = torch.Generator().manual_seed(seed)
    patients: Iterator[PatientSequence] = iter_patient_sequences(events_binned, vocab)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    feats: list[torch.Tensor] = []
    gaps_out: list[torch.Tensor] = []
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
            gap, valid = gap_survival_valid_mask(
                chunk.batch.aux.time_stamps, chunk.real_mask
            )
            n_valid = int(valid.sum().item())
            seen += n_valid
            if n_valid == 0:
                continue
            if sample_rate < 1.0:
                draw = torch.rand(valid.shape, generator=gen) < sample_rate
                valid = valid & draw.to(valid.device)
            if not bool(valid.any()):
                continue
            # fp16 at rest: a bank is millions of rows x hundreds of dims;
            # heads upcast per batch when fitting/scoring.
            feats.append(fwd.features[valid].to(torch.float16).cpu())
            gaps_out.append(gap[valid].float().clamp_min(0.0).cpu())
            kept += int(valid.sum().item())
            if max_positions is not None and kept >= max_positions:
                break
    if not feats:
        raise ValueError("no valid positions collected -- empty split?")
    bank = FeatureBank(
        torch.cat(feats)[:max_positions] if max_positions else torch.cat(feats),
        torch.cat(gaps_out)[:max_positions] if max_positions else torch.cat(gaps_out),
        n_positions_seen=seen,
        sample_rate=sample_rate,
    )
    logger.info(
        "[time-probe] banked %d of %d positions (rate %.3f, dim %d)",
        len(bank),
        seen,
        sample_rate,
        bank.features.shape[1],
    )
    return bank


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def _bin_left_edges(edges: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (left, right) hour edges of bins 1..len(edges)+1 (open bin: right=inf)."""
    e = torch.tensor(list(edges), dtype=torch.float64)
    left = torch.cat([torch.zeros(1, dtype=torch.float64), e])
    right = torch.cat([e, torch.full((1,), float("inf"), dtype=torch.float64)])
    return left, right


class ProbeHead(nn.Module):
    """A head that turns frozen features into a distribution over gap bins.

    Subclasses implement :meth:`log_bin_probs` (``(N, num_bins)`` log
    masses, bin 0 = same instant, last bin open) and :meth:`loss` (their
    natural training objective). Scoring is shared and bin-based.
    """

    def __init__(self, in_features: int, edges: Sequence[float]) -> None:
        """Record the bin layout shared by every head."""
        super().__init__()
        self.in_features = in_features
        self.edges = list(edges)
        self.num_bins = len(self.edges) + 2

    def log_bin_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(N, num_bins)`` log P(bin)."""
        raise NotImplementedError

    def loss(self, features: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        """Mean training loss on a batch (default: bin NLL)."""
        target = gap_to_bin(gaps, self.edges)
        return -self.log_bin_probs(features).gather(-1, target.unsqueeze(-1)).mean()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`log_bin_probs` on float32-upcast features."""
        return self.log_bin_probs(features.float())


class HazardProbe(ProbeHead):
    """The production head's parameterization: one hazard logit per bin."""

    def __init__(
        self, in_features: int, edges: Sequence[float], hidden: int = 0
    ) -> None:
        """``hidden > 0`` puts a one-hidden-layer MLP in front of the logits."""
        super().__init__(in_features, edges)
        self.net: nn.Module = (
            nn.Linear(in_features, self.num_bins)
            if hidden <= 0
            else nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.num_bins),
            )
        )

    def log_bin_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-bin log mass from the hazards (the open bin takes the rest)."""
        logits = self.net(features.float())
        log_h = F.logsigmoid(logits)
        log_1mh = F.logsigmoid(-logits)
        cum = torch.cumsum(log_1mh, dim=-1)
        before = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=-1)
        out = log_h + before
        # the final (open) bin is certain given survival through the rest
        out[:, -1] = before[:, -1]
        return out


class CategoricalProbe(ProbeHead):
    """Softmax over the same bins (discretized-grid framing)."""

    def __init__(self, in_features: int, edges: Sequence[float]) -> None:
        """Linear logits over bins."""
        super().__init__(in_features, edges)
        self.net = nn.Linear(in_features, self.num_bins)

    def log_bin_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Log-softmax over bins."""
        return F.log_softmax(self.net(features.float()), dim=-1)


class LogNormalMixtureProbe(ProbeHead):
    """Zero-gap Bernoulli + ``k``-component log-normal mixture over positive gaps."""

    def __init__(self, in_features: int, edges: Sequence[float], k: int = 3) -> None:
        """Heads for p(zero), mixture logits, means and log-sigmas of log-hours."""
        super().__init__(in_features, edges)
        self.k = k
        self.zero = nn.Linear(in_features, 1)
        self.mix = nn.Linear(in_features, k)
        self.mu = nn.Linear(in_features, k)
        self.log_sigma = nn.Linear(in_features, k)
        left, right = _bin_left_edges(edges)
        self.register_buffer("_left", left)
        self.register_buffer("_right", right)

    def _params(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = features.float()
        log_p_zero = F.logsigmoid(self.zero(features)).squeeze(-1)
        log_p_pos = F.logsigmoid(-self.zero(features)).squeeze(-1)
        log_pi = F.log_softmax(self.mix(features), dim=-1)
        mu = self.mu(features)
        sigma = F.softplus(self.log_sigma(features)) + 1e-3
        params: torch.Tensor = torch.stack([mu, sigma], dim=-1)
        return log_p_zero, log_p_pos, log_pi, params

    def log_bin_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Bin masses from mixture CDF differences on the log-hour axis."""
        log_p_zero, log_p_pos, log_pi, ms = self._params(features)
        mu, sigma = ms[..., 0], ms[..., 1]  # (N, k)
        left = cast(torch.Tensor, self._left).to(features.dtype)  # (B,)
        right = cast(torch.Tensor, self._right).to(features.dtype)
        log_left = torch.log(left.clamp_min(1e-12))  # bin 1 left = 0 -> -inf
        log_left[0] = float("-inf")
        log_right = torch.log(right)  # last = inf
        zl = (log_left[None, None, :] - mu[..., None]) / sigma[..., None]  # (N,k,B)
        zr = (log_right[None, None, :] - mu[..., None]) / sigma[..., None]
        cdf_r = torch.special.ndtr(zr)
        cdf_l = torch.special.ndtr(zl)
        mass = (cdf_r - cdf_l).clamp_min(1e-12)  # (N,k,B)
        log_mass = torch.logsumexp(log_pi[..., None] + torch.log(mass), dim=1)  # (N,B)
        return torch.cat([log_p_zero[:, None], log_p_pos[:, None] + log_mass], dim=-1)

    def loss(self, features: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        """Natural objective: Bernoulli(zero) + mixture density on log(gap)."""
        return self.per_position_nll(features, gaps).mean()

    def per_position_nll(
        self, features: torch.Tensor, gaps: torch.Tensor
    ) -> torch.Tensor:
        """Per-position NLL under the mixed discrete/continuous model."""
        log_p_zero, log_p_pos, log_pi, ms = self._params(features)
        mu, sigma = ms[..., 0], ms[..., 1]
        positive = gaps > 0
        y = torch.log(gaps.clamp_min(1e-6))
        z = (y[:, None] - mu) / sigma
        log_dens = log_pi - torch.log(sigma) - 0.5 * z * z - 0.5 * math.log(2 * math.pi)
        log_dens = torch.logsumexp(log_dens, dim=-1) - y
        return -torch.where(positive, log_p_pos + log_dens, log_p_zero)

    def median_positive_gap(self, features: torch.Tensor) -> torch.Tensor:
        """Return the positive-gap mixture median (hours) by bisection on the CDF."""
        _, _, log_pi, ms = self._params(features)
        mu, sigma = ms[..., 0], ms[..., 1]
        pi = log_pi.exp()
        lo = torch.full_like(mu[:, 0], math.log(1e-4))
        hi = torch.full_like(mu[:, 0], math.log(24.0 * 365))
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cdf = (pi * torch.special.ndtr((mid[:, None] - mu) / sigma)).sum(-1)
            lo = torch.where(cdf < 0.5, mid, lo)
            hi = torch.where(cdf < 0.5, hi, mid)
        return torch.exp(0.5 * (lo + hi))


def make_head(name: str, in_features: int, edges: Sequence[float]) -> ProbeHead:
    """Build a probe head by name (see :data:`DEFAULT_HEADS`)."""
    if name == "hazard":
        return HazardProbe(in_features, edges)
    if name == "hazard_mlp":
        return HazardProbe(in_features, edges, hidden=256)
    if name == "categorical":
        return CategoricalProbe(in_features, edges)
    if name.startswith("lognormal"):
        k = int(name[len("lognormal") :] or 3)
        return LogNormalMixtureProbe(in_features, edges, k=k)
    raise ValueError(f"unknown probe head {name!r}")


# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------


@dataclass
class FitTrace:
    """Per-epoch tuning NLL and the epoch chosen by early stopping."""

    tuning_nll: list[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0


def fit_head(
    head: ProbeHead,
    train: FeatureBank,
    tuning: FeatureBank,
    *,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 2,
    seed: int = 0,
) -> FitTrace:
    """Adam on the head's own loss; early stopping on tuning bin-NLL."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    trace = FitTrace()
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best = float("inf")
    bad = 0
    n = len(train)
    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=train.features.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = head.loss(train.features[idx], train.gaps[idx])
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        tune_nll = bin_nll(head, tuning)
        trace.tuning_nll.append(tune_nll)
        if tune_nll < best - 1e-5:
            best, bad = tune_nll, 0
            trace.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    trace.seconds = time.time() - t0
    return trace


@torch.no_grad()
def bin_nll(head: ProbeHead, bank: FeatureBank, batch_size: int = 16384) -> float:
    """Mean discrete-bin NLL over a bank."""
    head.eval()
    total = 0.0
    for start in range(0, len(bank), batch_size):
        f = bank.features[start : start + batch_size]
        g = bank.gaps[start : start + batch_size]
        target = gap_to_bin(g, head.edges)
        total += float(-head.log_bin_probs(f).gather(-1, target.unsqueeze(-1)).sum())
    return total / max(len(bank), 1)


@dataclass
class ProbeMetrics:
    """Held-out scores for one head (all heads on the same bins and positions)."""

    head: str
    n_positions: int
    bin_nll: float
    same_instant_accuracy: float
    calibration: dict[str, dict[str, float]]
    calibration_after_bundle: dict[str, dict[str, float]]
    n_positive: int
    continuous_nll: float | None = None
    median_gap_abs_error_hours: float | None = None
    parameters: int = 0
    fit: FitTrace | None = None


@torch.no_grad()
def score_head(
    head: ProbeHead, bank: FeatureBank, name: str, batch_size: int = 16384
) -> ProbeMetrics:
    """Score bin NLL, same-instant accuracy and within-horizon calibration."""
    head.eval()
    edges = torch.tensor(head.edges, dtype=torch.float64)
    covered = {label: int((edges <= h).sum().item()) for label, h in HORIZONS.items()}
    nll = 0.0
    same_correct = 0
    pred_within = dict.fromkeys(HORIZONS, 0.0)
    obs_within = dict.fromkeys(HORIZONS, 0)
    pred_pos = dict.fromkeys(HORIZONS, 0.0)
    obs_pos = dict.fromkeys(HORIZONS, 0)
    n_pos = 0
    cont = 0.0
    med_err = 0.0
    for start in range(0, len(bank), batch_size):
        f = bank.features[start : start + batch_size]
        g = bank.gaps[start : start + batch_size]
        logp = head.log_bin_probs(f)
        target = gap_to_bin(g, head.edges)
        nll += float(-logp.gather(-1, target.unsqueeze(-1)).sum())
        p = logp.exp()
        p_same = p[:, 0]
        same = g <= 0
        same_correct += int(((p_same > 0.5) == same).sum())
        positive = ~same
        n_pos += int(positive.sum())
        cdf = torch.cumsum(p, dim=-1)
        for label, horizon in HORIZONS.items():
            within = cdf[:, covered[label]]
            pred_within[label] += float(within.sum())
            obs_within[label] += int((g <= horizon).sum())
            if bool(positive.any()):
                conditional = (
                    (within - p_same) / (1.0 - p_same).clamp_min(1e-6)
                ).clamp(0, 1)
                pred_pos[label] += float(conditional[positive].sum())
                obs_pos[label] += int((g[positive] <= horizon).sum())
        if isinstance(head, LogNormalMixtureProbe):
            cont += float(head.per_position_nll(f, g).sum())
            if bool(positive.any()):
                med = head.median_positive_gap(f[positive])
                med_err += float((med - g[positive]).abs().sum())
    n = len(bank)
    return ProbeMetrics(
        head=name,
        n_positions=n,
        bin_nll=nll / n,
        same_instant_accuracy=same_correct / n,
        calibration={
            label: {
                "predicted": pred_within[label] / n,
                "observed": obs_within[label] / n,
            }
            for label in HORIZONS
        },
        calibration_after_bundle={
            label: {
                "predicted": pred_pos[label] / max(n_pos, 1),
                "observed": obs_pos[label] / max(n_pos, 1),
            }
            for label in HORIZONS
        },
        n_positive=n_pos,
        continuous_nll=(cont / n) if isinstance(head, LogNormalMixtureProbe) else None,
        median_gap_abs_error_hours=(
            med_err / max(n_pos, 1) if isinstance(head, LogNormalMixtureProbe) else None
        ),
        parameters=sum(p.numel() for p in head.parameters()),
    )


def run_probe(
    model: SequenceModel,
    vocab: Vocabulary,
    banks: dict[str, FeatureBank],
    *,
    heads: Sequence[str] = DEFAULT_HEADS,
    edges: Sequence[float] = DEFAULT_TIME_BIN_EDGES_HOURS,
    device: str = "cpu",
    epochs: int = 20,
    seed: int = 0,
) -> list[ProbeMetrics]:
    """Fit every head on the train bank, early-stop on tuning, score held-out."""
    del model, vocab  # banks already carry the frozen features
    train = banks["train"].to(device)
    tuning = banks["tuning"].to(device)
    held = banks["held_out"].to(device)
    results: list[ProbeMetrics] = []
    for name in heads:
        head = make_head(name, train.features.shape[1], edges).to(device)
        trace = fit_head(head, train, tuning, epochs=epochs, seed=seed)
        metrics = score_head(head, held, name)
        metrics.fit = trace
        logger.info(
            "[time-probe] %-12s nll=%.4f same=%.4f after-1h pred/obs=%.4f/%.4f (%d params, %.0fs)",
            name,
            metrics.bin_nll,
            metrics.same_instant_accuracy,
            metrics.calibration_after_bundle["1h"]["predicted"],
            metrics.calibration_after_bundle["1h"]["observed"],
            metrics.parameters,
            trace.seconds,
        )
        results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bank_from_shards(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: object,
    config: object,
    shard_dir: str | Path,
    *,
    max_shards: int | None,
    sample_rate: float,
    seed: int,
    num_lanes: int,
    chunk_size: int,
    device: str,
    max_positions: int | None,
) -> FeatureBank:
    from odyssey.data.code_normalization import maybe_normalize  # noqa: PLC0415
    from odyssey.data.history_recap import maybe_history_recap  # noqa: PLC0415
    from odyssey.training.data import load_meds_shard  # noqa: PLC0415
    from odyssey.training.shard_stream import shard_paths  # noqa: PLC0415

    # One shard at a time (subjects never span shards): the whole-split
    # frame for 10 MIMIC shards plus value tokens was ~30M rows and
    # OOM-killed the first N2 attempt at ~45 GB RSS. Peak is now one shard's
    # frames plus the accumulated fp16 banks.
    source = getattr(config, "source", "mimic_iv")
    banks: list[FeatureBank] = []
    kept = 0
    for k, path in enumerate(shard_paths(shard_dir, max_shards=max_shards)):
        raw = load_meds_shard(path)
        raw = maybe_normalize(
            raw, enabled=getattr(config, "normalize_medications", False), source=source
        )
        raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
        binned = add_value_tokens(raw, binner, source=source)  # type: ignore[arg-type]
        del raw
        bank = collect_feature_bank(
            model,
            binned,
            vocab,
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
    return FeatureBank.concat(banks, max_positions)


def _main() -> None:
    from odyssey.inference.run_inference import load_run  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Distributional time-head probe (N2).")
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
    parser.add_argument("--heads", nargs="+", default=list(DEFAULT_HEADS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-lanes", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    model, vocab, binner, config = load_run(
        run_dir,
        device=device,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
    )
    time_head = getattr(model, "time_head", None)
    edges = (
        list(time_head.edges)
        if time_head is not None
        else list(DEFAULT_TIME_BIN_EDGES_HOURS)
    )
    common = {
        "num_lanes": args.num_lanes,
        "chunk_size": args.chunk_size,
        "device": device,
    }
    banks = {
        "train": _bank_from_shards(
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
        ),
        "tuning": _bank_from_shards(
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
        ),
        "held_out": _bank_from_shards(
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
        ),
    }
    results = run_probe(
        model,
        vocab,
        banks,
        heads=args.heads,
        edges=edges,
        device=device,
        epochs=args.epochs,
        seed=args.seed,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "edges": edges,
        "banks": {
            k: {
                "positions": len(v),
                "seen": v.n_positions_seen,
                "sample_rate": v.sample_rate,
            }
            for k, v in banks.items()
        },
        "heads": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2))
    logger.info("[time-probe] wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _main()
