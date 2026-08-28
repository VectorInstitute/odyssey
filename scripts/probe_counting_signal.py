"""Frozen pre/post-bottleneck regression probes for the GBM's COUNT features.

Direct follow-up to scripts/probe_bottleneck_signal.py and the 2026-08-24 GBM
feature-group ablation (docs/experiments.md, "subset_run_v8 (GBM feature-group
ablation)"): that ablation found the GBM's `counts_occurrence` group (drug-class
and code-family occurrence counts over 6h/24h/the visit, plus n_prior_visits
and n_events_visit -- 110 of 609 strong features) explains 70-99% of the GBM's
margin over the hazard head on vasopressor_start and icu_admission. The
straightforward fix -- feed those counts to the model as input features -- is
explicitly OUT OF SCOPE (see the "no hand-engineered features" standing rule:
recency_features/signal_channels were removed on 2026-08-25 for the same
reason). Before reaching for an architecture or training-objective change,
this asks the same question the recency precedent (commit cd96842, R^2 0.925
pre-bottleneck / 0.916 post-bottleneck with no such feature ever supplied)
already answered for recency: can a frozen linear (Ridge) probe ALREADY
recover the GBM's own count features from the backbone's own hidden state,
with nothing handed in?

- High recoverable R^2 (pre and post bottleneck) => the representation
  already encodes counting; the GBM gap on vasopressor/ICU is a head or
  training-signal problem, not a representational one. Matches the
  `probe_ci_check.py` finding that the trained hazard head already matches
  the linear-probe ceiling on those tasks (see docs/experiments.md, "probe vs
  hazard-head CI check") -- if counting is ALSO already recoverable, closing
  the gap needs a better way to USE what's already encoded (task formulation,
  training signal), not more input signal or more head capacity.
- Low recoverable R^2 (especially pre-bottleneck, where there is no
  compression to blame) => a genuine representational limit: the backbone
  cannot count over its own context/state. That motivates either (a) an
  auxiliary self-supervised objective that asks the model to predict counts
  from its own state as a training signal (never as an input), or (b)
  capacity/context-length changes -- not feeding the counts in directly.

Reuses probe_bottleneck_signal.collect_embeddings for identical row selection
and embeddings, and odyssey.inference.baseline_features.StrongFeatureBuilder
(the GBM's own feature code) to compute the regression TARGETS at the same
(subject, visit, time) keys -- so "what the GBM would have seen" and "what
the backbone/bottleneck already represents" are measured on identical rows.

Not wired into any CI/registry path. Run directly:

    uv run python scripts/probe_counting_signal.py \
        --run-dir ~/runs/subset_run_v8_taskset_v3 \
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --max-train-shards 5 --max-held-out-shards 4
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import alert_events_for
from odyssey.data.concepts import concepts_for_source
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import _load_prepared_raw, _visit_starts
from odyssey.inference.baseline_features import StrongFeatureBuilder, feature_names
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from scripts.probe_bottleneck_signal import collect_embeddings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_counting_signal")


def _counting_columns() -> list[int]:
    """Indices of the pure occurrence-COUNT columns in the strong feature matrix.

    Drug-class n_6h/n_24h/ever_visit and family n_6h/n_24h/n_visit (drops
    `hours_since_last`, which is recency, already probed by the cd96842
    precedent), plus n_prior_visits/n_events_visit from the context block.
    Deliberately excludes the signal-panel value stats (last/mean/min/max/
    delta/ratio): those are the `summary_stats` group, a separate question.
    """
    names = feature_names()
    keep_suffixes = (".n_6h", ".n_24h", ".n_visit", ".ever_visit")
    idx = [
        i
        for i, n in enumerate(names)
        if (n.startswith("drug.") or n.startswith("family.")) and n.endswith(keep_suffixes)
    ]
    idx += [names.index("n_prior_visits"), names.index("n_events_visit")]
    return sorted(idx)


def probe_r2(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit one frozen StandardScaler+Ridge probe, multi-output, per-column R^2."""
    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y)
    reg = Ridge(alpha=alpha)
    reg.fit(x_scaler.transform(train_x), y_scaler.transform(train_y))
    pred = reg.predict(x_scaler.transform(test_x))
    return r2_score(y_scaler.transform(test_y), pred, multioutput="raw_values")


def main() -> None:
    """Extract pre/post-bottleneck embeddings and report count-recovery R^2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--max-train-shards", type=int, default=5)
    parser.add_argument("--max-held-out-shards", type=int, default=4)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--num-lanes", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
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
    del concept_names  # not needed; kept for parity with the sibling script

    landmark_alerts = [a for a in alert_events_for(task_set) if not a.next_visit]

    def load_split(shard_dir: str, max_shards: int):
        activate_sidecars(shard_dir)
        raw = _load_prepared_raw(shard_dir, max_shards, config, source)
        visit_start = _visit_starts(raw)
        binned = add_value_tokens(raw, binner, source=source)
        builder = StrongFeatureBuilder(raw, source=source)
        del raw
        return binned, visit_start, builder

    logger.info(
        "loading %d train shard(s) from %s", args.max_train_shards, args.train_shard_dir
    )
    train_binned, train_visit_start, train_builder = load_split(
        args.train_shard_dir, args.max_train_shards
    )
    logger.info(
        "loading %d held-out shard(s) from %s",
        args.max_held_out_shards,
        args.held_out_shard_dir,
    )
    held_binned, held_visit_start, held_builder = load_split(
        args.held_out_shard_dir, args.max_held_out_shards
    )

    logger.info("extracting train embeddings")
    train_lm_keys, train_lm_pre, train_lm_post, _, _, _ = collect_embeddings(
        model,
        train_binned,
        vocab,
        landmark_alerts=landmark_alerts,
        visit_end_alerts=[],
        visit_start=train_visit_start,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    logger.info("train: %d landmark rows", len(train_lm_keys))

    logger.info("extracting held-out embeddings")
    held_lm_keys, held_lm_pre, held_lm_post, _, _, _ = collect_embeddings(
        model,
        held_binned,
        vocab,
        landmark_alerts=landmark_alerts,
        visit_end_alerts=[],
        visit_start=held_visit_start,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    logger.info("held-out: %d landmark rows", len(held_lm_keys))

    names = feature_names()
    count_idx = _counting_columns()
    count_names = [names[i] for i in count_idx]
    logger.info("%d counting-feature targets", len(count_idx))

    train_sids = [k[0] for k in train_lm_keys]
    train_vids = [k[1] for k in train_lm_keys]
    train_times = [k[2] for k in train_lm_keys]
    held_sids = [k[0] for k in held_lm_keys]
    held_vids = [k[1] for k in held_lm_keys]
    held_times = [k[2] for k in held_lm_keys]

    train_y = train_builder.features(train_sids, train_vids, train_times)[:, count_idx]
    held_y = held_builder.features(held_sids, held_vids, held_times)[:, count_idx]

    # Rows where every counting target is finite (StrongFeatureBuilder returns
    # NaN only for subjects/positions it never saw, which should not happen
    # here since the keys came from the same binned frame it was built on).
    m_train: np.ndarray = np.asarray(np.isfinite(train_y).all(axis=1))
    m_held: np.ndarray = np.asarray(np.isfinite(held_y).all(axis=1))
    logger.info(
        "usable rows: train %d/%d, held-out %d/%d",
        int(m_train.sum()),
        m_train.shape[0],
        int(m_held.sum()),
        m_held.shape[0],
    )

    pre_r2 = probe_r2(
        train_lm_pre[m_train],
        train_y[m_train],
        held_lm_pre[m_held],
        held_y[m_held],
        alpha=args.ridge_alpha,
    )
    post_r2 = probe_r2(
        train_lm_post[m_train],
        train_y[m_train],
        held_lm_post[m_held],
        held_y[m_held],
        alpha=args.ridge_alpha,
    )

    order = np.argsort(pre_r2)
    logger.info(
        "pooled mean R^2: pre-bottleneck %.4f  post-bottleneck %.4f  (n_held=%d)",
        float(np.mean(pre_r2)),
        float(np.mean(post_r2)),
        int(m_held.sum()),
    )
    logger.info("worst-recovered (by pre-bottleneck R^2):")
    for i in order[:10]:
        logger.info(
            "  %-40s pre=%.4f post=%.4f", count_names[i], pre_r2[i], post_r2[i]
        )
    logger.info("best-recovered (by pre-bottleneck R^2):")
    for i in order[-10:][::-1]:
        logger.info(
            "  %-40s pre=%.4f post=%.4f", count_names[i], pre_r2[i], post_r2[i]
        )

    print("\nfeature,pre_bottleneck_r2,post_bottleneck_r2,delta")
    for i, name in enumerate(count_names):
        print(f"{name},{pre_r2[i]},{post_r2[i]},{post_r2[i] - pre_r2[i]}")


if __name__ == "__main__":
    main()
