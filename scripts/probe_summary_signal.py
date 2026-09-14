"""Frozen pre/post-bottleneck probes for the GBM's window-statistic features.

Companion to scripts/probe_counting_signal.py. That script asks whether the
backbone's own hidden state already encodes the GBM's occurrence COUNTS
(the `counts_occurrence` group). This one asks the same question for the
`summary_stats` group -- per-signal window minimum/maximum/mean over 6 h and
24 h, the change from the previous value, the change from the visit's first
value and the ratio to the visit's minimum -- which the 2026-08-24
feature-group ablation found leads the GBM's margin on acute kidney injury
(and is literally what KDIGO defines AKI on). Same frozen Ridge probe, same
row selection (4-hourly landmarks), same GBM feature code for the targets.

Two differences from the counting script, both forced by the targets:

- A window statistic is NaN when the signal was never measured in that
  window, so each target is probed on its own finite rows and the row
  count is reported beside its R^2. Multi-output fitting would discard
  almost every row.
- Ratios and deltas have heavy tails, so Spearman rank correlation is
  reported next to R^2 as a scale-free companion.

`--feature-group both` also runs the counting group on the same extracted
embeddings, so one invocation gives a run both baselines at once.

`--probe hgb` fits a HistGradientBoosting regressor beside the Ridge probe
(same rows, same targets), the nonlinear check the counting work used on
2026-08-29: a target the linear probe cannot read but the tree probe can is
"encoded, but not linearly". `--targets change` restricts to the change
statistics (delta_prev, delta_visit_first, ratio_visit_min) plus the level
anchors of a few key signals. Targets are winsorized at the train 0.5/99.5
percentiles for BOTH probes when `--winsorize` is set, so artifact values
(sentinel MAPs) cannot dominate R^2.

Not wired into any CI/registry path. Run directly:

    uv run python scripts/probe_summary_signal.py \
        --run-dir ~/runs/full_run_v10 \
        --train-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --max-train-shards 5 --max-held-out-shards 4 --feature-group both
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from odyssey.data.alert_events import alert_events_for
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.alerts import _load_prepared_raw, _visit_starts
from odyssey.inference.baseline_features import (
    CONTEXT_FEATURES,
    StrongFeatureBuilder,
    feature_names,
)
from odyssey.inference.embedding_probe import collect_embeddings
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_summary_signal")

SUMMARY_STATS: tuple[str, ...] = (
    "last",
    "mean_24h",
    "min_24h",
    "max_24h",
    "min_6h",
    "max_6h",
    "delta_prev",
    "delta_visit_first",
    "ratio_visit_min",
)
"""`last` is included as an anchor: the most recent value is the easiest
statistic to read off a sequence model, so its R^2 bounds the others."""

MIN_TRAIN_ROWS = 500
MIN_HELD_ROWS = 200

CHANGE_STATS: tuple[str, ...] = ("delta_prev", "delta_visit_first", "ratio_visit_min")
ANCHOR_SIGNALS: tuple[str, ...] = (
    "creatinine",
    "bun",
    "lactate",
    "map_noninvasive",
    "sbp_noninvasive",
    "heart_rate",
    "platelets",
    "hemoglobin",
    "urine_output",
)
ANCHOR_STATS: tuple[str, ...] = ("last", "min_6h", "mean_24h")


def _is_change_target(name: str) -> bool:
    """Change statistics for every signal, plus level anchors for key signals."""
    signal, stat = name.rsplit(".", 1)
    return stat in CHANGE_STATS or (signal in ANCHOR_SIGNALS and stat in ANCHOR_STATS)


def _counting_columns(names: list[str]) -> list[int]:
    """Occurrence-count columns, exactly as scripts/probe_counting_signal.py."""
    keep_suffixes = (".n_6h", ".n_24h", ".n_visit", ".ever_visit")
    idx = [
        i
        for i, n in enumerate(names)
        if (n.startswith("drug.") or n.startswith("family."))
        and n.endswith(keep_suffixes)
    ]
    idx += [names.index("n_prior_visits"), names.index("n_events_visit")]
    return sorted(idx)


def _summary_columns(names: list[str]) -> list[int]:
    """Per-signal window-statistic columns (the GBM's `summary_stats` group)."""
    context = set(CONTEXT_FEATURES)
    return [
        i
        for i, n in enumerate(names)
        if n not in context
        and not n.startswith(("drug.", "family."))
        and n.rsplit(".", 1)[-1] in SUMMARY_STATS
    ]


def probe_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    kind: str = "ridge",
) -> tuple[float, float]:
    """Fit one frozen probe (Ridge or HistGradientBoosting); return (R^2, rho)."""
    x_scaler = StandardScaler().fit(train_x)
    y_mean, y_std = float(train_y.mean()), float(train_y.std() or 1.0)
    reg: Ridge | HistGradientBoostingRegressor
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.1,
            max_leaf_nodes=31,
            early_stopping=True,
            random_state=0,
        )
    else:
        reg = Ridge(alpha=alpha)
    reg.fit(x_scaler.transform(train_x), (train_y - y_mean) / y_std)
    pred = reg.predict(x_scaler.transform(test_x))
    r2 = float(r2_score((test_y - y_mean) / y_std, pred))
    rho = float(spearmanr(test_y, pred).correlation)
    return r2, rho


def probe_group(  # noqa: PLR0913
    label: str,
    idx: list[int],
    names: list[str],
    *,
    train_y_all: np.ndarray,
    held_y_all: np.ndarray,
    train_pre: np.ndarray,
    train_post: np.ndarray,
    held_pre: np.ndarray,
    held_post: np.ndarray,
    alpha: float,
    kinds: tuple[str, ...] = ("ridge",),
    winsorize: bool = False,
    train_subsample: int = 0,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Probe every column of a group on its own finite rows, per probe kind."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in idx:
        ty, hy = train_y_all[:, i], held_y_all[:, i]
        m_tr, m_he = np.isfinite(ty), np.isfinite(hy)
        if (
            m_tr.sum() < MIN_TRAIN_ROWS
            or m_he.sum() < MIN_HELD_ROWS
            or np.nanstd(hy) == 0
        ):
            logger.info(
                "  skip %-40s (train %d, held %d rows)",
                names[i],
                m_tr.sum(),
                m_he.sum(),
            )
            continue
        tr = np.flatnonzero(m_tr)
        if train_subsample and len(tr) > train_subsample:
            tr = rng.choice(tr, size=train_subsample, replace=False)
        ty_use, hy_use = ty[tr], hy[m_he]
        if winsorize:
            lo, hi = np.percentile(ty_use, [0.5, 99.5])
            ty_use, hy_use = np.clip(ty_use, lo, hi), np.clip(hy_use, lo, hi)
        row: dict[str, object] = {"feature": names[i], "n_held": int(m_he.sum())}
        for kind in kinds:
            pre = probe_one(
                train_pre[tr], ty_use, held_pre[m_he], hy_use, alpha=alpha, kind=kind
            )
            post = probe_one(
                train_post[tr], ty_use, held_post[m_he], hy_use, alpha=alpha, kind=kind
            )
            row[f"{kind}_pre_r2"], row[f"{kind}_pre_rho"] = pre
            row[f"{kind}_post_r2"], row[f"{kind}_post_rho"] = post
        rows.append(row)
        logger.info(
            "  %-40s "
            + "  ".join(
                f"{k}: pre={row[f'{k}_pre_r2']:.3f} post={row[f'{k}_post_r2']:.3f} rho={row[f'{k}_pre_rho']:.2f}"
                for k in kinds
            ),
            names[i],
        )
    logger.info("[%s] %d targets probed", label, len(rows))
    return rows


def report(label: str, rows: list[dict[str, object]], kinds: tuple[str, ...]) -> None:
    """Log per-statistic medians for every probe kind, then print the CSV block."""
    if not rows:
        logger.info("[%s] nothing to report", label)
        return
    cols = [
        f"{k}_{side}_{m}"
        for k in kinds
        for side in ("pre", "post")
        for m in ("r2", "rho")
    ]
    by_stat: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_stat[str(r["feature"]).rsplit(".", 1)[-1]].append(r)
    logger.info("[%s] MEDIAN by statistic; columns: %s", label, " ".join(cols))
    for stat, rs in sorted(by_stat.items()):
        meds = [float(np.median([float(r[c]) for r in rs])) for c in cols]  # type: ignore[arg-type]
        logger.info(
            "  %-20s %s  (n=%d)", stat, " ".join(f"{m:6.3f}" for m in meds), len(rs)
        )
    print(f"\n# group={label}")
    print("feature,n_held," + ",".join(cols))
    for r in rows:
        print(f"{r['feature']},{r['n_held']}," + ",".join(str(r[c]) for c in cols))


def main() -> None:  # noqa: PLR0915
    """Extract pre/post-bottleneck embeddings and report window-stat recovery."""
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
    parser.add_argument(
        "--feature-group", choices=("summary", "counts", "both"), default="summary"
    )
    parser.add_argument("--probe", choices=("ridge", "hgb", "both"), default="ridge")
    parser.add_argument(
        "--targets",
        choices=("all", "change"),
        default="all",
        help="'change' keeps delta_prev/delta_visit_first/ratio_visit_min plus level anchors",
    )
    parser.add_argument(
        "--winsorize",
        action="store_true",
        help="clip targets at train 0.5/99.5 percentiles",
    )
    parser.add_argument(
        "--train-subsample",
        type=int,
        default=0,
        help="rows per target for fitting (0 = all)",
    )
    args = parser.parse_args()
    kinds: tuple[str, ...] = ("ridge", "hgb") if args.probe == "both" else (args.probe,)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, binner, config = load_run(args.run_dir, device=device)
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(f"{args.run_dir} is not a concept-bottleneck run")
    source = getattr(config, "source", "mimic_iv")
    task_set = getattr(config, "task_set", "v1")
    landmark_alerts = [a for a in alert_events_for(task_set) if not a.next_visit]

    def load_split(
        shard_dir: str, max_shards: int
    ) -> tuple[pl.DataFrame, dict[tuple[int, int], float], StrongFeatureBuilder]:
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

    def embed(
        binned: pl.DataFrame, visit_start: dict[tuple[int, int], float]
    ) -> tuple[list[tuple[int, int, float]], np.ndarray, np.ndarray]:
        keys, pre, post, _, _, _ = collect_embeddings(
            model,
            binned,
            vocab,
            landmark_alerts=landmark_alerts,
            visit_end_alerts=[],
            visit_start=visit_start,
            landmark_hours=args.landmark_hours,
            num_lanes=args.num_lanes,
            chunk_size=args.chunk_size,
            device=device,
        )
        return keys, pre, post

    logger.info("extracting train embeddings")
    train_keys, train_pre, train_post = embed(train_binned, train_visit_start)
    logger.info("train: %d landmark rows", len(train_keys))
    logger.info("extracting held-out embeddings")
    held_keys, held_pre, held_post = embed(held_binned, held_visit_start)
    logger.info("held-out: %d landmark rows", len(held_keys))

    names = feature_names()
    train_y = train_builder.features(
        [k[0] for k in train_keys],
        [k[1] for k in train_keys],
        [k[2] for k in train_keys],
    )
    held_y = held_builder.features(
        [k[0] for k in held_keys], [k[1] for k in held_keys], [k[2] for k in held_keys]
    )

    groups = {"summary": _summary_columns, "counts": _counting_columns}
    chosen = list(groups) if args.feature_group == "both" else [args.feature_group]
    for label in chosen:
        idx = groups[label](names)
        if args.targets == "change" and label == "summary":
            idx = [i for i in idx if _is_change_target(names[i])]
        logger.info("[%s] %d candidate targets", label, len(idx))
        rows = probe_group(
            label,
            idx,
            names,
            train_y_all=train_y,
            held_y_all=held_y,
            train_pre=train_pre,
            train_post=train_post,
            held_pre=held_pre,
            held_post=held_post,
            alpha=args.ridge_alpha,
            kinds=kinds,
            winsorize=args.winsorize,
            train_subsample=args.train_subsample,
        )
        report(label, rows, kinds)


if __name__ == "__main__":
    main()
