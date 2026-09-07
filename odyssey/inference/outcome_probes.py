"""Frozen outcome probes: extra hazard-style heads read from the bottleneck.

The trained per-event hazard heads all forecast something bad (vasopressor
start, ICU admission, AKI, death, Sepsis-3, readmission), so a steering
push can only ever be wrong in one direction: every declared expectation
says "up means more risk", and a push that merely moves a patient toward
"sicker" scores well without capturing the state. This module adds
outcomes whose expectation runs the other way for a sicker state, ICU
discharge, hospital discharge alive and vasopressor stop
(:data:`odyssey.data.alert_events.STATE_TRANSITION_EVENTS`), as probes fit
on the frozen model's own bottleneck output at the same landmark rows the
alerts harness scores. No backbone weight changes: a probe is a
scaler-folded logistic regression per (event, horizon), so the trained
model, its checkpoints and every banked number stay as they are.

Fit once per run with the CLI below, then hand the ``.pt`` to
``odyssey.inference.steering --outcome-probes``; the steering pass reads
the probes on the same features as the trained heads and reports them as
additional outcomes with their own declared expectations.

    uv run python -m odyssey.inference.outcome_probes \
        --run-dir ~/runs/full_run_DEC_v12 \
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --max-train-shards 40 --max-held-out-shards 8 \
        --output ~/runs/full_run_DEC_v12/outcome_probes.pt
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import (
    STATE_TRANSITION_EVENTS,
    STATE_TRANSITION_REQUIRES,
    AlertEvent,
    EventTimes,
    alert_events_for,
    all_event_times,
)
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import HORIZONS_HOURS, _load_prepared_raw, _visit_starts
from odyssey.inference.embedding_probe import Key, collect_embeddings, labels_for
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


logger = logging.getLogger(__name__)

MAX_FIT_ROWS = 400_000
"""Rows per (event, horizon) fit; a landmark pass over 40 MIMIC-IV train
shards yields millions, and a logistic probe saturates long before that."""


def fold_scaler(
    scaler: StandardScaler, clf: LogisticRegression
) -> tuple[np.ndarray, float]:
    """Return ``(w, b)`` so that ``sigmoid(x @ w + b)`` equals the scaled probe.

    The probe was fit on ``(x - mean) / scale``; folding the scaler into the
    weights lets the steering pass apply the probe as one linear layer on
    the raw bottleneck output.
    """
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    w = coef / scale
    b = float(clf.intercept_[0] - np.sum(coef * mean / scale))
    return w, b


def required_mask(keys: Sequence[Key], required: EventTimes | None) -> np.ndarray:
    """Return True where the required prior event has happened by the row's time.

    ``None`` (no requirement) keeps every row. A row of a visit in which the
    required event never occurs is never at risk: a patient not on
    vasopressors cannot stop them.
    """
    if required is None:
        return np.ones(len(keys), dtype=bool)
    out = np.zeros(len(keys), dtype=bool)
    for i, (sid, vid, t) in enumerate(keys):
        key = (sid, -1 if required.subject_scoped else vid)
        onset = required.onset.get(key)
        out[i] = onset is not None and onset <= t
    return out


@dataclass
class OutcomeProbes:
    """Folded linear probes: ``risk(features) -> (N, E, H)`` P(event within horizon)."""

    event_names: list[str]
    horizons_hours: list[float]
    weight: torch.Tensor
    """(E, H, D)."""
    bias: torch.Tensor
    """(E, H)."""
    requires: dict[str, str]
    """event -> prior event that must already have happened for a row to be at risk."""
    auroc: dict[str, dict[str, float]]
    """event -> {"train@8h": ..., "held_out@8h": ...}; the probe's own quality."""

    def to(self, device: str | torch.device) -> OutcomeProbes:
        """Move the tensors; returns self for chaining."""
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def risk(self, features: torch.Tensor) -> torch.Tensor:
        """``(N, E, H)`` probabilities from ``(N, D)`` bottleneck features."""
        logits = torch.einsum("nd,ehd->neh", features.float(), self.weight.float())
        return torch.sigmoid(logits + self.bias.float())

    def save(self, path: str | Path) -> None:
        """Write the probes as a plain dict of tensors and metadata."""
        torch.save(
            {
                "event_names": self.event_names,
                "horizons_hours": self.horizons_hours,
                "weight": self.weight.cpu(),
                "bias": self.bias.cpu(),
                "requires": self.requires,
                "auroc": self.auroc,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> OutcomeProbes:
        """Read probes written by :meth:`save`."""
        blob: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            event_names=list(blob["event_names"]),
            horizons_hours=[float(h) for h in blob["horizons_hours"]],
            weight=blob["weight"],
            bias=blob["bias"],
            requires=dict(blob["requires"]),
            auroc=dict(blob["auroc"]),
        ).to(device)


def fit_probes(  # noqa: PLR0917
    train_keys: Sequence[Key],
    train_post: np.ndarray,
    train_times: Mapping[str, EventTimes],
    held_keys: Sequence[Key],
    held_post: np.ndarray,
    held_times: Mapping[str, EventTimes],
    *,
    events: Sequence[AlertEvent] = STATE_TRANSITION_EVENTS,
    requires: Mapping[str, str] = STATE_TRANSITION_REQUIRES,
    horizons: Sequence[float] = HORIZONS_HOURS,
    max_rows: int = MAX_FIT_ROWS,
    seed: int = 0,
) -> OutcomeProbes:
    """Fit one folded logistic probe per (event, horizon) on landmark rows.

    Rows are kept only where the event is at risk (label defined) and, for
    events with a requirement, where the prior event has happened. The
    held-out rows only score the probe; nothing is fit on them.
    """
    rng = np.random.default_rng(seed)
    dim = int(train_post.shape[1])
    weight = np.zeros((len(events), len(horizons), dim), dtype=np.float32)
    bias = np.zeros((len(events), len(horizons)), dtype=np.float32)
    auroc: dict[str, dict[str, float]] = {}
    for e, event in enumerate(events):
        req = requires.get(event.name)
        train_req = required_mask(train_keys, train_times.get(req) if req else None)
        held_req = required_mask(held_keys, held_times.get(req) if req else None)
        auroc[event.name] = {}
        for h, horizon in enumerate(horizons):
            y_tr = labels_for(train_keys, train_times[event.name], horizon)
            keep = ~np.isnan(y_tr) & train_req
            idx = np.flatnonzero(keep)
            if idx.size > max_rows:
                idx = rng.choice(idx, size=max_rows, replace=False)
            x, y = train_post[idx], y_tr[idx].astype(int)
            if y.min() == y.max():
                logger.warning(
                    "[probes] %s@%gh: one class only in %d rows; probe left at zero",
                    event.name,
                    horizon,
                    y.size,
                )
                continue
            scaler = StandardScaler().fit(x)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(scaler.transform(x), y)
            w, b = fold_scaler(scaler, clf)
            weight[e, h] = w.astype(np.float32)
            bias[e, h] = b
            train_auc = float(roc_auc_score(y, x @ w + b))
            y_ho = labels_for(held_keys, held_times[event.name], horizon)
            keep_ho = ~np.isnan(y_ho) & held_req
            held_auc = float("nan")
            if keep_ho.sum() > 0 and y_ho[keep_ho].min() != y_ho[keep_ho].max():
                held_auc = float(
                    roc_auc_score(y_ho[keep_ho].astype(int), held_post[keep_ho] @ w + b)
                )
            auroc[event.name][f"train@{horizon:g}h"] = train_auc
            auroc[event.name][f"held_out@{horizon:g}h"] = held_auc
            auroc[event.name][f"n_train@{horizon:g}h"] = float(y.size)
            auroc[event.name][f"positives_train@{horizon:g}h"] = float(y.sum())
            logger.info(
                "[probes] %s@%gh: %d rows (%d positive), train AUROC %.3f, held-out %.3f",
                event.name,
                horizon,
                y.size,
                int(y.sum()),
                train_auc,
                held_auc,
            )
    return OutcomeProbes(
        event_names=[ev.name for ev in events],
        horizons_hours=[float(h) for h in horizons],
        weight=torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
        requires={k: v for k, v in requires.items() if k in {ev.name for ev in events}},
        auroc=auroc,
    )


def _embeddings_for_split(  # noqa: PLR0917
    model: ConceptBottleneckSequenceModel,
    vocab: Any,
    binner: Any,
    config: Any,
    source: str,
    shard_dir: str,
    max_shards: int,
    *,
    events: Sequence[AlertEvent],
    landmark_hours: float,
    num_lanes: int,
    chunk_size: int,
    device: str,
) -> tuple[list[Key], np.ndarray, dict[str, EventTimes]]:
    """Landmark keys, post-bottleneck embeddings and event times for one split."""
    task_set = getattr(config, "task_set", "v1")
    activate_sidecars(shard_dir)
    raw = _load_prepared_raw(shard_dir, max_shards, config, source)
    visit_start = _visit_starts(raw)
    binned = add_value_tokens(raw, binner, source=source)
    # the probe events plus whatever prior events the requirements name
    needed = list(events) + [
        a
        for a in alert_events_for(task_set, source=source)
        if a.name in set(STATE_TRANSITION_REQUIRES.values())
    ]
    times = all_event_times(raw, needed, source, task_set=task_set)
    del raw
    keys, _pre, post, _ve_keys, _ve_pre, _ve_post = collect_embeddings(
        model,
        binned,
        vocab,
        landmark_alerts=list(events),
        visit_end_alerts=[],
        visit_start=visit_start,
        landmark_hours=landmark_hours,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
    )
    return keys, post, times


def main() -> None:
    """Fit the state-transition probes for one run and save them next to it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--max-train-shards", type=int, default=40)
    parser.add_argument("--max-held-out-shards", type=int, default=8)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--num-lanes", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, binner, config = load_run(
        args.run_dir,
        device=device,
        checkpoint_path=Path(args.run_dir) / args.checkpoint
        if args.checkpoint
        else None,
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError("outcome probes need a concept-bottleneck run")
    source = getattr(config, "source", "mimic_iv")
    events = [
        ev
        for ev in STATE_TRANSITION_EVENTS
        if not (source != "mimic_iv" and ev.name == "vasopressor_stop")
    ]
    logger.info("[probes] events: %s", [ev.name for ev in events])
    train = _embeddings_for_split(
        model,
        vocab,
        binner,
        config,
        source,
        args.train_shard_dir,
        args.max_train_shards,
        events=events,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    logger.info("[probes] train rows %d, dim %d", len(train[0]), train[1].shape[1])
    held = _embeddings_for_split(
        model,
        vocab,
        binner,
        config,
        source,
        args.held_out_shard_dir,
        args.max_held_out_shards,
        events=events,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    logger.info("[probes] held-out rows %d", len(held[0]))
    probes = fit_probes(
        train[0], train[1], train[2], held[0], held[1], held[2], events=events
    )
    probes.save(args.output)
    logger.info("[probes] wrote %s", args.output)


if __name__ == "__main__":
    main()
