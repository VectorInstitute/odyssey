"""Frozen pre/post-bottleneck linear probes, per alert task.

Diagnostic, not production eval machinery: reuses alerts.py's own landmark
selection, event-time construction, and IndexRow/outcome_at_horizon
plumbing so labels and positions are identical to the alerts harness, but
captures two embeddings per kept position instead of scoring heads --
the backbone hidden state (pre-bottleneck) and the mixed known+unknown
bottleneck output (post-bottleneck, what the event/task heads actually
read) -- then fits a frozen StandardScaler+LogisticRegression probe on
each, separately, per (event, horizon). Mirrors the precedent frozen-
feature probe cited in commit cd96842 (per-family recency recoverable at
R^2 0.925 pre-bottleneck / 0.916 post-bottleneck on subset_run_v8).

``collect_embeddings``/``labels_for`` now live in
:mod:`odyssey.inference.embedding_probe` as tested library code (this
script duplicated them, untested, until 2026-08-28) -- also the
foundation :mod:`odyssey.inference.probe_baseline`'s EHRSHOT-style
benchmark builds on. This script is now a thin CLI over that library.

Not wired into any CI/registry path. Run directly:

    uv run python scripts/probe_bottleneck_signal.py \
        --run-dir ~/runs/subset_run_v8_taskset_v3 \
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --max-train-shards 3 --max-held-out-shards 4
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import EventTimes, alert_events_for, all_event_times
from odyssey.data.concepts import concepts_for_source
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import HORIZONS_HOURS, _load_prepared_raw, _visit_starts
from odyssey.inference.embedding_probe import collect_embeddings, labels_for
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_bottleneck_signal")

READMISSION_HORIZONS = (168.0, 720.0)


def probe_auroc(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> float | None:
    """Fit a frozen StandardScaler+LogisticRegression probe, return held-out AUROC."""
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return None
    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(train_x), train_y)
    proba = clf.predict_proba(scaler.transform(test_x))[:, 1]
    return float(roc_auc_score(test_y, proba))


def main() -> None:  # noqa: PLR0915
    """Extract pre/post-bottleneck embeddings and report probe AUROC per task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--max-train-shards", type=int, default=3)
    parser.add_argument("--max-held-out-shards", type=int, default=4)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--num-lanes", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, binner, config = load_run(args.run_dir, device=device)
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            f"{args.run_dir} is not a concept-bottleneck run (model_kind must be "
            "'cbm'): this script probes pre/post-bottleneck embeddings, which "
            "only exist on ConceptBottleneckSequenceModel."
        )
    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    concept_names = [c.name for c in concepts_for_source(source, task_set=task_set)]
    del concept_names  # not needed here; kept for parity with alerts.py setup

    all_alerts = alert_events_for(task_set)
    landmark_alerts = [a for a in all_alerts if not a.next_visit]
    visit_end_alerts = [a for a in all_alerts if a.next_visit]
    logger.info(
        "landmark alerts: %s | visit-end alerts: %s",
        [a.name for a in landmark_alerts],
        [a.name for a in visit_end_alerts],
    )

    def load_split(
        shard_dir: str, max_shards: int
    ) -> tuple[
        pl.DataFrame,
        dict[tuple[int, int], float],
        dict[str, EventTimes],
        dict[str, EventTimes],
    ]:
        activate_sidecars(shard_dir)
        raw = _load_prepared_raw(shard_dir, max_shards, config, source)
        visit_start = _visit_starts(raw)
        binned = add_value_tokens(raw, binner, source=source)
        lm_times = (
            all_event_times(raw, landmark_alerts, source, task_set=task_set)
            if landmark_alerts
            else {}
        )
        ve_times = (
            all_event_times(raw, visit_end_alerts, source, task_set=task_set)
            if visit_end_alerts
            else {}
        )
        del raw
        return binned, visit_start, lm_times, ve_times

    logger.info(
        "loading %d train shard(s) from %s", args.max_train_shards, args.train_shard_dir
    )
    train_binned, train_visit_start, train_lm_times, train_ve_times = load_split(
        args.train_shard_dir, args.max_train_shards
    )
    logger.info(
        "loading %d held-out shard(s) from %s",
        args.max_held_out_shards,
        args.held_out_shard_dir,
    )
    held_binned, held_visit_start, held_lm_times, held_ve_times = load_split(
        args.held_out_shard_dir, args.max_held_out_shards
    )

    logger.info("extracting train embeddings")
    (
        train_lm_keys,
        train_lm_pre,
        train_lm_post,
        train_ve_keys,
        train_ve_pre,
        train_ve_post,
    ) = collect_embeddings(
        model,
        train_binned,
        vocab,
        landmark_alerts=landmark_alerts,
        visit_end_alerts=visit_end_alerts,
        visit_start=train_visit_start,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    logger.info(
        "train: %d landmark rows, %d visit-end rows",
        len(train_lm_keys),
        len(train_ve_keys),
    )

    logger.info("extracting held-out embeddings")
    (
        held_lm_keys,
        held_lm_pre,
        held_lm_post,
        held_ve_keys,
        held_ve_pre,
        held_ve_post,
    ) = collect_embeddings(
        model,
        held_binned,
        vocab,
        landmark_alerts=landmark_alerts,
        visit_end_alerts=visit_end_alerts,
        visit_start=held_visit_start,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    logger.info(
        "held-out: %d landmark rows, %d visit-end rows",
        len(held_lm_keys),
        len(held_ve_keys),
    )

    results = []
    for alert in landmark_alerts:
        for h in HORIZONS_HOURS:
            y_train = labels_for(train_lm_keys, train_lm_times[alert.name], h)
            y_held = labels_for(held_lm_keys, held_lm_times[alert.name], h)
            m_train = ~np.isnan(y_train)
            m_held = ~np.isnan(y_held)
            pre_auc = probe_auroc(
                train_lm_pre[m_train],
                y_train[m_train],
                held_lm_pre[m_held],
                y_held[m_held],
            )
            post_auc = probe_auroc(
                train_lm_post[m_train],
                y_train[m_train],
                held_lm_post[m_held],
                y_held[m_held],
            )
            results.append((alert.name, h, int(m_held.sum()), pre_auc, post_auc))
            logger.info(
                "%-22s %5.0fh  n=%d  pre=%s  post=%s",
                alert.name,
                h,
                int(m_held.sum()),
                pre_auc,
                post_auc,
            )

    for alert in visit_end_alerts:
        for h in READMISSION_HORIZONS:
            y_train = labels_for(train_ve_keys, train_ve_times[alert.name], h)
            y_held = labels_for(held_ve_keys, held_ve_times[alert.name], h)
            m_train = ~np.isnan(y_train)
            m_held = ~np.isnan(y_held)
            pre_auc = probe_auroc(
                train_ve_pre[m_train],
                y_train[m_train],
                held_ve_pre[m_held],
                y_held[m_held],
            )
            post_auc = probe_auroc(
                train_ve_post[m_train],
                y_train[m_train],
                held_ve_post[m_held],
                y_held[m_held],
            )
            results.append((alert.name, h, int(m_held.sum()), pre_auc, post_auc))
            logger.info(
                "%-22s %5.0fh  n=%d  pre=%s  post=%s",
                alert.name,
                h,
                int(m_held.sum()),
                pre_auc,
                post_auc,
            )

    print("\ntask,horizon_h,n,pre_bottleneck_auroc,post_bottleneck_auroc,delta")
    for name, h, n, pre_auc, post_auc in results:
        delta = (
            (post_auc - pre_auc)
            if (pre_auc is not None and post_auc is not None)
            else None
        )
        print(f"{name},{h:g},{n},{pre_auc},{post_auc},{delta}")


if __name__ == "__main__":
    main()
