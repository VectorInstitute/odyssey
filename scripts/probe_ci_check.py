"""Paired subject-clustered bootstrap: post-bottleneck probe vs hazard head.

Follow-up to scripts/probe_bottleneck_signal.py: reuses its embedding
extraction (same train/held-out shards, same landmark/visit-end
selection), fits the post-bottleneck probe, and joins its held-out
predictions against the existing alerts_rows_{v3,readmission_v3}.parquet
dumps to get the hazard head's per-row score on the identical held-out
positions. Both arms are then restricted to the SAME rows (the key
intersection: rows the dump scored with a non-null hazard and label) and
compared with odyssey.inference.uncertainty.bootstrap_auroc_delta -- a
PAIRED, subject-clustered bootstrap of the per-resample AUROC difference
(n_boot=1000, alpha=0.05). Per-arm CIs are reported alongside for
context, on those same rows; the verdict comes from whether the paired
delta's CI excludes 0, never from CI overlap (overlapping per-arm CIs do
not establish "no difference", and the two arms are highly correlated).

    uv run python scripts/probe_ci_check.py \
        --run-dir ~/runs/subset_run_v8_taskset_v3 \
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --alerts-rows ~/runs/subset_run_v8_taskset_v3/alerts_rows_v3.parquet \
        --alerts-rows-readmission \
            ~/runs/subset_run_v8_taskset_v3/alerts_rows_readmission_v3.parquet \
        --max-train-shards 5 --max-held-out-shards 4
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import EventTimes, alert_events_for, all_event_times
from odyssey.data.concepts import concepts_for_source
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import (
    HORIZONS_HOURS,
    _load_prepared_raw,
    _visit_starts,
)
from odyssey.inference.embedding_probe import Key, collect_embeddings, labels_for
from odyssey.inference.run_inference import load_run
from odyssey.inference.uncertainty import (
    BootstrapAUROC,
    BootstrapAUROCDelta,
    bootstrap_auroc,
    bootstrap_auroc_delta,
)
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from scripts.probe_bottleneck_signal import READMISSION_HORIZONS


_CellResult = tuple[
    str,
    float,
    int,
    BootstrapAUROC | None,
    BootstrapAUROC | None,
    BootstrapAUROCDelta | None,
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_ci_check")


def _fmt(b: BootstrapAUROC | None) -> str:
    if b is None:
        return "n/a"
    if b.ci_low is None or b.ci_high is None:
        return f"{b.point_estimate:.3f} [no usable resamples]"
    return f"{b.point_estimate:.3f} [{b.ci_low:.3f}, {b.ci_high:.3f}]"


def _fmt_delta(d: BootstrapAUROCDelta | None) -> str:
    if d is None:
        return "n/a"
    if d.ci_low is None or d.ci_high is None:
        return f"{d.point_estimate:+.3f} [no usable resamples]"
    return f"{d.point_estimate:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]"


def _verdict(delta: BootstrapAUROCDelta | None) -> str:
    """Read the verdict off the PAIRED delta CI, never off CI overlap."""
    if delta is None:
        return "unscoreable"
    excludes = delta.excludes_zero()
    if excludes is None:
        return "unscoreable"
    return "SEPARATED (paired diff)" if excludes else "WITHIN NOISE (paired)"


def main() -> None:  # noqa: PLR0915
    """Fit the post-bottleneck probe and compare it, paired, to the hazard head."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--alerts-rows", required=True)
    parser.add_argument("--alerts-rows-readmission", required=True)
    parser.add_argument("--max-train-shards", type=int, default=5)
    parser.add_argument("--max-held-out-shards", type=int, default=4)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--num-lanes", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, binner, config = load_run(args.run_dir, device=device)
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            f"{args.run_dir} is not a concept-bottleneck run: this script "
            "probes the post-bottleneck embedding, which only exists on "
            "ConceptBottleneckSequenceModel."
        )
    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    concepts_for_source(source, task_set=task_set)  # parity with alerts.py setup

    all_alerts = alert_events_for(task_set)
    landmark_alerts = [a for a in all_alerts if not a.next_visit]
    visit_end_alerts = [a for a in all_alerts if a.next_visit]

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

    logger.info("loading train/held-out splits")
    train_binned, train_visit_start, train_lm_times, train_ve_times = load_split(
        args.train_shard_dir, args.max_train_shards
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
    del train_binned, held_binned

    # Load the alerts dumps: event -> {(subject_id, visit_id, time_hours) -> row dict}
    dump = pl.concat(
        [
            pl.read_parquet(args.alerts_rows),
            pl.read_parquet(args.alerts_rows_readmission),
        ],
        how="diagonal_relaxed",
    )
    logger.info("loaded alerts dump: %d rows, columns=%s", dump.height, dump.columns)

    def dump_lookup(event_name: str) -> dict[tuple[int, int, float], dict[str, Any]]:
        sub = (
            dump.filter(pl.col("event") == event_name)
            if "event" in dump.columns
            else dump
        )
        out: dict[tuple[int, int, float], dict[str, Any]] = {}
        n_dup = 0
        for row in sub.iter_rows(named=True):
            key = (
                int(row["subject_id"]),
                int(row["visit_id"]),
                float(row["time_hours"]),
            )
            if key in out:
                n_dup += 1
            out[key] = row
        if n_dup:
            logger.warning(
                "%s: %d duplicate keys in the alerts dump (last row wins)",
                event_name,
                n_dup,
            )
        return out

    results: list[_CellResult] = []

    def run_cell(
        event_name: str,
        horizon: float,
        *,
        train_keys: list[Key],
        train_post: np.ndarray,
        train_times: dict[str, EventTimes],
        held_keys: list[Key],
        held_post: np.ndarray,
        held_times: dict[str, EventTimes],
        dump_by_key: dict[tuple[int, int, float], dict[str, Any]],
    ) -> None:
        y_train = labels_for(train_keys, train_times[event_name], horizon)
        y_held = labels_for(held_keys, held_times[event_name], horizon)
        m_train = ~np.isnan(y_train)
        m_held = ~np.isnan(y_held)
        if m_train.sum() < 10 or len(np.unique(y_train[m_train])) < 2:
            results.append((event_name, horizon, 0, None, None, None))
            return
        scaler = StandardScaler().fit(train_post[m_train])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(train_post[m_train]), y_train[m_train])
        held_idx = np.nonzero(m_held)[0]
        probe_p_all = clf.predict_proba(scaler.transform(held_post[held_idx]))[:, 1]

        # Both arms are scored on the SAME rows: the labeled held-out rows
        # the dump also scored (non-null hazard and label). Scoring the
        # probe on all labeled rows but the hazard only on the dump's
        # subset would compare two different samples -- unpaired, and
        # silently so whenever rows are missing from the dump.
        hz_col = f"hazard@{horizon:g}h"
        y_col = f"y@{horizon:g}h"
        y_pair, probe_p, hz_p, subj = [], [], [], []
        n_missing = 0
        n_label_mismatch = 0
        for j, i in enumerate(held_idx):
            key = held_keys[i]
            row = dump_by_key.get(key)
            if row is None or row.get(hz_col) is None or row.get(y_col) is None:
                n_missing += 1
                continue
            if float(row[y_col]) != float(y_held[i]):
                n_label_mismatch += 1
                continue
            y_pair.append(y_held[i])
            probe_p.append(probe_p_all[j])
            hz_p.append(row[hz_col])
            subj.append(key[0])
        if n_missing:
            logger.warning(
                "%s@%gh: %d/%d labeled held-out rows missing from dump -- "
                "both arms are scored only on the %d-row intersection",
                event_name,
                horizon,
                n_missing,
                len(held_idx),
                len(y_pair),
            )
        if n_label_mismatch:
            logger.warning(
                "%s@%gh: %d rows dropped, dump label disagrees with the "
                "recomputed label -- protocol drift between dump and this run?",
                event_name,
                horizon,
                n_label_mismatch,
            )
        if not y_pair:
            results.append((event_name, horizon, 0, None, None, None))
            return

        y_arr = np.array(y_pair)
        probe_arr = np.array(probe_p)
        hz_arr = np.array(hz_p)
        subj_arr = np.array(subj)
        probe_ci = bootstrap_auroc(y_arr, probe_arr, subj_arr)
        hazard_ci = bootstrap_auroc(y_arr, hz_arr, subj_arr)
        delta = bootstrap_auroc_delta(y_arr, probe_arr, hz_arr, subj_arr)
        results.append((event_name, horizon, len(y_pair), probe_ci, hazard_ci, delta))
        logger.info(
            "%-22s %5.0fh  probe=%s  hazard=%s  delta=%s  verdict=%s",
            event_name,
            horizon,
            _fmt(probe_ci),
            _fmt(hazard_ci),
            _fmt_delta(delta),
            _verdict(delta),
        )

    for alert in landmark_alerts:
        dump_by_key = dump_lookup(alert.name)
        for h in HORIZONS_HOURS:
            run_cell(
                alert.name,
                h,
                train_keys=train_lm_keys,
                train_post=train_lm_post,
                train_times=train_lm_times,
                held_keys=held_lm_keys,
                held_post=held_lm_post,
                held_times=held_lm_times,
                dump_by_key=dump_by_key,
            )

    for alert in visit_end_alerts:
        dump_by_key = dump_lookup(alert.name)
        for h in READMISSION_HORIZONS:
            run_cell(
                alert.name,
                h,
                train_keys=train_ve_keys,
                train_post=train_ve_post,
                train_times=train_ve_times,
                held_keys=held_ve_keys,
                held_post=held_ve_post,
                held_times=held_ve_times,
                dump_by_key=dump_by_key,
            )

    print(
        "\nevent,horizon_h,n,probe_point,probe_ci_low,probe_ci_high,"
        "hazard_point,hazard_ci_low,hazard_ci_high,"
        "delta_point,delta_ci_low,delta_ci_high,verdict"
    )
    for name, h, n, probe_ci, hazard_ci, delta in results:
        pp = f"{probe_ci.point_estimate:.4f}" if probe_ci else ""
        pl_ = f"{probe_ci.ci_low:.4f}" if probe_ci else ""
        ph = f"{probe_ci.ci_high:.4f}" if probe_ci else ""
        hp = f"{hazard_ci.point_estimate:.4f}" if hazard_ci else ""
        hl = f"{hazard_ci.ci_low:.4f}" if hazard_ci else ""
        hh = f"{hazard_ci.ci_high:.4f}" if hazard_ci else ""
        dp = f"{delta.point_estimate:+.4f}" if delta else ""
        dl = f"{delta.ci_low:+.4f}" if delta and delta.ci_low is not None else ""
        dh = f"{delta.ci_high:+.4f}" if delta and delta.ci_high is not None else ""
        print(
            f"{name},{h:g},{n},{pp},{pl_},{ph},{hp},{hl},{hh},"
            f"{dp},{dl},{dh},{_verdict(delta)}"
        )


if __name__ == "__main__":
    main()
