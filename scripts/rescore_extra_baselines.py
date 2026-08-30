"""Rescore an already-dumped alerts table with TabICL / EBM / SurvivalPFN.

Companion to ``scripts/eval_run.sh``'s ``alerts`` stage: that stage already
dumps hazard-head and tuned-GBM scores via a full model forward pass
(``collect_model_scores``, GPU). This script adds three more baseline
families to the SAME held-out rows without re-running that pass -- fitting
a baseline (``fit_tabicl_baselines``/``fit_ebm_baselines``/
``fit_survivalpfn_baselines``) and scoring held-out features are both
model-free, CPU-only operations (``_index_rows_from_events``,
``features_for_events``), so the only work here is baseline fitting and
feature computation, joined onto the existing dump by (event, subject_id,
visit_id, time_hours) rather than assumed to line up positionally.

Two anti-lost-compute measures, both added after a real incident (a run
that fit all three baselines cleanly -- EBM alone took ~4.6h over 12
(event, horizon) pairs -- then crashed at the first held-out scoring
call; see ``docs/reeval_wave_v2.md``):

- Fits are cached to ``{run-dir}/rescore_cache/`` (see
  :mod:`odyssey.inference.fit_cache`) immediately as each one completes
  and reloaded on a rerun instead of refit, gated on an environment
  fingerprint so a cache built in a different venv is never trusted.
- Held-out scoring itself needs no batching logic here: the crash was an
  unbatched ``predict_proba`` call over 552K+ query rows inside the
  in-context/foundation-model baselines (TabICL, SurvivalPFN), fixed at
  the source in their own model wrappers
  (:class:`~odyssey.inference.tabicl_baseline.TabICLBaselineModel`,
  :class:`~odyssey.inference.survivalpfn_baseline.SurvivalPFNBaselineModel`),
  which now chunk the query dimension internally -- every caller,
  including this script's plain ``.predict_proba(...)`` calls below,
  gets that for free.

Usage:
    uv run python -m scripts.rescore_extra_baselines \
        --run-dir ~/runs/full_run_v8 \
        --held-out-shard-dir ~/data/mimiciv_3.1_v1/data/held_out \
        --baseline-shard-dir ~/data/mimiciv_3.1_v1/data/train \
        --existing-dump ~/runs/full_run_v8/alerts_rows_v2.parquet \
        --max-shards 4 --max-baseline-shards 30 \
        --output-parquet ~/runs/full_run_v8/alerts_rows_v2_rescored.parquet
"""

import argparse
import gc
import json
import logging
from pathlib import Path

import polars as pl

from odyssey.data.alert_events import ALERT_EVENTS
from odyssey.data.value_binning import QuantileBinner
from odyssey.inference.alerts import HORIZONS_HOURS, IndexRow
from odyssey.inference.baseline_prep import prepare_baseline_data
from odyssey.inference.ebm_baseline import fit_ebm_baselines
from odyssey.inference.fit_cache import FitCache
from odyssey.inference.survivalpfn_baseline import fit_survivalpfn_baselines
from odyssey.inference.tabicl_baseline import fit_tabicl_baselines
from odyssey.training.shard_stream import make_preparer, shard_paths
from odyssey.training.train import TrainingConfig
from odyssey.utils.joblib_tmp import ensure_joblib_temp_folder


logger = logging.getLogger(__name__)


def _rows_frame(
    rows: dict[str, list[IndexRow]],
    scores: dict[str, dict[str, dict[str, list[float]]]],
    horizons: tuple[float, ...],
) -> pl.DataFrame:
    """One row per (event, subject_id, visit_id, time_hours), plus score columns."""
    frames = []
    for event, event_rows in rows.items():
        if not event_rows:
            continue
        data = {
            "event": [event] * len(event_rows),
            "subject_id": [float(r.subject_id) for r in event_rows],
            "visit_id": [float(r.visit_id) for r in event_rows],
            "time_hours": [r.time_hours for r in event_rows],
        }
        for name, per_horizon in scores.items():
            for h in horizons:
                key = f"{name}@{h:g}h"
                if key in per_horizon.get(event, {}):
                    data[key] = per_horizon[event][key]
        frames.append(pl.DataFrame(data))
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


def _rss_human() -> str:
    """Return the current process RSS as a human string (from /proc)."""
    try:
        with open("/proc/self/status") as f:  # noqa: PTH123
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main() -> None:  # noqa: PLR0915
    """Fit TabICL/EBM/SurvivalPFN and join their scores onto an existing dump."""
    ensure_joblib_temp_folder()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--held-out-shard-dir", required=True, type=Path)
    parser.add_argument("--baseline-shard-dir", required=True, type=Path)
    parser.add_argument("--existing-dump", required=True, type=Path)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-baseline-shards", type=int, default=None)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--output-parquet", required=True, type=Path)
    args = parser.parse_args()

    config = TrainingConfig(**json.loads((args.run_dir / "config.json").read_text()))
    binner = QuantileBinner.load(args.run_dir / "quantile_binner.json")
    horizons = HORIZONS_HOURS
    cache = FitCache(cache_dir=args.run_dir / "rescore_cache")

    source = getattr(config, "source", "mimic_iv")
    prepare = make_preparer(
        normalize_medications=getattr(config, "normalize_medications", False),
        history_recap=getattr(config, "history_recap", False),
        source=source,
    )

    logger.info("[rescore] preparing held-out events (model-free, streaming)")
    held = prepare_baseline_data(
        shard_paths(args.held_out_shard_dir, max_shards=args.max_shards),
        prepare,
        binner,
        alerts=ALERT_EVENTS,
        feature_sets=("strong", "basic"),
        source=source,
        landmark_hours=args.landmark_hours,
    )
    held_rows = held.rows

    existing = pl.read_parquet(args.existing_dump)
    existing_key_rows = list(
        existing.select(["event", "subject_id", "visit_id", "time_hours"]).iter_rows()
    )
    existing_keys = set(existing_key_rows)
    held_key_rows = [
        (event, float(r.subject_id), float(r.visit_id), r.time_hours)
        for event, rows in held_rows.items()
        for r in rows
    ]
    held_keys = set(held_key_rows)

    missing_from_existing = held_keys - existing_keys
    extra_in_existing = existing_keys - held_keys
    duplicate_in_existing = len(existing_key_rows) != len(existing_keys)
    duplicate_in_held = len(held_key_rows) != len(held_keys)

    if (
        missing_from_existing
        or extra_in_existing
        or duplicate_in_existing
        or duplicate_in_held
    ):

        def _sample(
            keys: set[tuple[str, float, float, float]], n: int = 10
        ) -> list[tuple[str, float, float, float]]:
            return sorted(keys)[:n]

        raise RuntimeError(
            "model-free row set does not exactly match the existing dump's "
            f"keys -- held-out: {len(held_key_rows)} rows / {len(held_keys)} "
            f"unique keys, dump: {len(existing_key_rows)} rows / "
            f"{len(existing_keys)} unique keys. "
            f"{len(missing_from_existing)} keys in held-out but not in dump "
            f"(sample: {_sample(missing_from_existing)}). "
            f"{len(extra_in_existing)} keys in dump but not in held-out "
            f"(sample: {_sample(extra_in_existing)}). "
            "Landmark derivation has diverged from the dump (or the row set "
            "contains duplicate keys) -- refusing to join scores onto a "
            "mismatched row set."
        )
    logger.info(
        "[rescore] key match with existing dump: exact, %d rows", len(held_keys)
    )

    logger.info("[rescore] preparing baseline training events (model-free, streaming)")
    # One streaming pass replaces the old whole-split frame (held across all
    # three fits) plus a SECOND full raw load for event times -- the exact
    # shape that OOM-killed two eval boxes at 35-37GB (see module docstring).
    # Event times now come from the PREPARED (post-normalization) events,
    # matching fit_baselines_streaming's canonical behavior, where the old
    # code used un-normalized raw events for times only.
    train = prepare_baseline_data(
        shard_paths(args.baseline_shard_dir, max_shards=args.max_baseline_shards),
        prepare,
        binner,
        alerts=ALERT_EVENTS,
        feature_sets=("strong", "basic"),
        source=source,
        landmark_hours=args.landmark_hours,
    )
    empty = pl.DataFrame()

    # TabICL on the BASIC feature set, for two reasons that point the same
    # way: (1) comparability -- every completed TabICL run in this project
    # (the eICU rows in docs/experiments.md) used basic, so a strong-feature
    # MIMIC row would not be like-for-like; (2) cost -- TabICL re-reads its
    # whole context per call at O(n^2 + n*m^2), so 609 columns is ~70 GB per
    # call versus ~8 GB at 17 (three OOM-kills, 2026-08-23; see
    # tabicl_baseline.estimate_peak_gb, which now refuses such a fit).
    logger.info("[rescore] fitting TabICL (basic features)")
    tabicl_models = fit_tabicl_baselines(
        empty,
        train.rows,
        train.times,
        horizons=horizons,
        source=source,
        cache=cache,
        features=train.features["basic"],
        feature_set="basic",
    )
    logger.info("[rescore] fitting EBM (strong features)")
    ebm_models = fit_ebm_baselines(
        empty,
        train.rows,
        train.times,
        horizons=horizons,
        source=source,
        cache=cache,
        features=train.features["strong"],
    )
    logger.info("[rescore] fitting SurvivalPFN (basic features, 100-feature cap)")
    survivalpfn_models = fit_survivalpfn_baselines(
        empty,
        train.rows,
        train.times,
        horizons=horizons,
        source=source,
        cache=cache,
        features=train.features["basic"],
    )

    logger.info("[rescore] scoring held-out rows")
    strong_features = held.features["strong"]
    basic_features = held.features["basic"]

    scores: dict[str, dict[str, dict[str, list[float]]]] = {
        "tabicl": {},
        "ebm": {},
        "survivalpfn": {},
    }
    # One model at a time: score it, then drop the reference and collect.
    # A fitted TabICL model holds its whole in-context training set, and
    # inference allocates against it; keeping all 12 resident while scoring
    # 138k held-out rows each was part of a real OOM-kill (~71 GB RSS,
    # 2026-08-23). Released models are never needed again -- their scores
    # are already in `scores`.
    for event, rows in held_rows.items():
        if not rows:
            continue
        scores["tabicl"][event] = {}
        scores["ebm"][event] = {}
        scores["survivalpfn"][event] = {}
        for h in horizons:
            for name, models, feats in (
                ("tabicl", tabicl_models, basic_features),
                ("ebm", ebm_models, strong_features),
                ("survivalpfn", survivalpfn_models, basic_features),
            ):
                model = models.pop((event, h), None)
                if model is None:
                    continue
                p = model.predict_proba(feats[event])
                scores[name][event][f"{name}@{h:g}h"] = [float(v) for v in p]
                del model
                gc.collect()
                logger.info(
                    "[rescore] %s %s@%gh scored (%d rows, rss %s)",
                    name,
                    event,
                    h,
                    len(rows),
                    _rss_human(),
                )

    new_cols = _rows_frame(held_rows, scores, horizons)
    joined = existing.join(
        new_cols, on=["event", "subject_id", "visit_id", "time_hours"], how="left"
    )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(args.output_parquet)
    logger.info(
        "[rescore] wrote %d rows (%d matched a new score) to %s",
        joined.height,
        joined.filter(pl.col("tabicl@8h").is_not_null()).height
        if "tabicl@8h" in joined.columns
        else 0,
        args.output_parquet,
    )


if __name__ == "__main__":
    main()
