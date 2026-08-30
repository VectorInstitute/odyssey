#!/usr/bin/env python3
"""Missingness stress protocol sweep: one trained run, one held-out split, one command.

docs/missingness_protocol.md, Track A item 6. Builds nothing new -- the
degraded-shard generator (:mod:`odyssey.data.degrade`) and the harness glue
in :mod:`odyssey.inference.alerts`/:mod:`odyssey.inference.baseline_prep`
are already lead-owned, tested infrastructure (see their own module
docstrings). This just sequences them:

1. Generate the protocol's 8 degraded cells from ``--held-out-shard-dir``
   (:func:`odyssey.data.degrade.generate_cell`), under
   ``--output-root/degraded_shards/<cell>/``.
2. Score the CLEAN split once: :func:`odyssey.inference.alerts.evaluate_alerts`
   against the undegraded split, fitting the GBM baseline (if
   ``--baseline-shard-dir`` is given) and dumping the per-row table -- this
   becomes both the "vs clean" comparison point AND the
   ``verify_against_dump`` target every degraded cell is checked against.
3. Score each of the 8 cells against the SAME held-out split, but reading
   events from the degraded shard directory. The GBM baseline is REUSED,
   never refit (Principle 1: frozen models, and a frozen-fit baseline too --
   refitting 8 times would also confound the degradation signal with
   fit-to-fit hyperparameter-search variance).
4. Aggregate: AUROC/AUPRC/ECE and their delta from clean, per
   (scorer, event, horizon) x cell, as JSON + markdown
   (:mod:`odyssey.reporting.missingness_report`).

Usage (one command; nothing this script writes ever lands in git --
``--output-root`` must be outside the repo, same governance as
scripts/gemini/'s ``GEMINI_MEDS_OUTPUT_DIR``)::

    python scripts/missingness_sweep.py \\
        --run-dir ~/runs/<run> \\
        --held-out-shard-dir <data_root>/held_out \\
        --baseline-shard-dir <data_root>/train \\
        --output-root ~/missingness/<run>

Re-running with the same ``--output-root`` resumes after a crash: degraded
shard generation and already-scored degraded cells are skipped, and (when
no ``--baseline-shard-dir`` is involved) an existing clean result is
reused too. With ``--baseline-shard-dir``, an existing clean result stops
the run instead: the clean pass is also what fits the frozen GBM the
degraded cells reuse (Principle 1), so it cannot be skipped without
refitting -- pass ``--overwrite`` to redo everything from scratch.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from odyssey.data.degrade import all_cells, generate_cell, load_cell_metadata
from odyssey.inference.alerts import (
    BASELINE_FEATURE_SETS,
    LANDMARK_PROTOCOL_VERSION,
    AlertMetrics,
    BaselineModel,
    evaluate_alerts,
)
from odyssey.inference.run_inference import refuse_existing_output
from odyssey.reporting.missingness_report import (
    CLEAN_CELL,
    CellMetricRow,
    build_degradation_table,
    load_cell_metrics,
    write_json,
    write_markdown,
)
from odyssey.training.data import shard_sort_key
from odyssey.utils.joblib_tmp import ensure_joblib_temp_folder


logger = logging.getLogger(__name__)


def _write_cell_result(
    path: Path,
    *,
    cell: str,
    metadata: Optional[Dict[str, object]],
    run_dir: Path,
    held_out_shard_dir: Path,
    results: List[AlertMetrics],
    n_unscoreable: Optional[int] = None,
) -> None:
    """Write one cell's alerts.json, with the degrade.py cell metadata embedded.

    ``n_unscoreable`` is ``None`` for the clean baseline (which never scores
    against a degraded record) and an int -- possibly 0 -- for every degraded
    cell: the count of clean rows this cell's metrics had to drop because the
    degraded record had no visible token at/before the row's time (see
    ``evaluate_alerts``'s ``unscoreable_out``).
    """
    payload = {
        "cell": cell,
        "cell_metadata": metadata,  # None for the clean baseline
        "run_dir": str(run_dir),
        "held_out_shard_dir": str(held_out_shard_dir),
        "n_unscoreable": n_unscoreable,
        "metrics": [
            {**asdict(r), "landmark_protocol_version": LANDMARK_PROTOCOL_VERSION}
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _generate_cells(
    held_out_shard_dir: Path,
    degraded_root: Path,
    *,
    seed: int,
    cells: Optional[Sequence[str]],
    source: str,
    overwrite: bool,
) -> Dict[str, Path]:
    cell_specs = all_cells(seed)
    if cells:
        unknown = sorted(set(cells) - set(cell_specs))
        if unknown:
            raise ValueError(
                f"unknown cell name(s) {unknown}, expected one of {sorted(cell_specs)}"
            )
        cell_specs = {name: cell_specs[name] for name in cells}
    shard_files = sorted(held_out_shard_dir.glob("*.parquet"), key=shard_sort_key)
    if not shard_files:
        raise FileNotFoundError(f"no .parquet shards found in {held_out_shard_dir}")
    cell_dirs: Dict[str, Path] = {}
    for name, cell in cell_specs.items():
        cell_dir = degraded_root / name
        already_done = cell_dir.is_dir() and (cell_dir / "metadata.json").is_file()
        if already_done and not overwrite:
            logger.info(
                "[sweep] cell %s already generated at %s, skipping "
                "(pass --overwrite to redo)",
                name,
                cell_dir,
            )
        else:
            logger.info("[sweep] generating cell %s -> %s", name, cell_dir)
            generate_cell(cell, shard_files, cell_dir, source=source)
        cell_dirs[name] = cell_dir
    return cell_dirs


def run_sweep(
    run_dir: Path,
    held_out_shard_dir: Path,
    output_root: Path,
    *,
    baseline_shard_dir: Optional[Path] = None,
    seed: int = 0,
    cells: Optional[Sequence[str]] = None,
    max_shards: Optional[int] = None,
    max_baseline_shards: Optional[int] = None,
    landmark_hours: float = 4.0,
    num_lanes: int = 8,
    chunk_size: int = 256,
    checkpoint: str = "checkpoint_best.pt",
    baseline_feature_set: str = "strong",
    tune_baselines: bool = True,
    stream_baseline: bool = False,
    source: str = "mimic_iv",
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Run the full sweep; returns (degradation_table.json, .md) paths."""
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = output_root / "results"
    cell_dirs = _generate_cells(
        held_out_shard_dir,
        output_root / "degraded_shards",
        seed=seed,
        cells=cells,
        source=source,
        overwrite=overwrite,
    )

    checkpoint_path = run_dir / checkpoint
    clean_json = results_dir / "clean_alerts.json"
    clean_rows = results_dir / "clean_alerts_rows.parquet"

    fitted: Dict[Tuple[str, float], BaselineModel] = {}
    # Resume: an existing clean result can only be reused when the clean
    # pass fits nothing (no --baseline-shard-dir). With a baseline dir the
    # clean pass is also what fits the frozen GBM every degraded cell
    # reuses (Principle 1: one fit, never refit) -- skipping it and
    # refitting fresh would confound the degradation signal with
    # fit-to-fit variance, and mix provenance with cells scored earlier;
    # refuse_existing_output keeps that case a loud stop instead.
    resume_clean = (
        not overwrite
        and baseline_shard_dir is None
        and clean_json.is_file()
        and clean_rows.is_file()
    )
    if resume_clean:
        logger.info(
            "[sweep] clean result already at %s (no GBM to fit), reusing it",
            clean_json,
        )
    else:
        refuse_existing_output(
            clean_json, overwrite=overwrite, kind="missingness sweep clean alerts"
        )
        logger.info(
            "[sweep] scoring the clean baseline (fits the GBM once, if "
            "--baseline-shard-dir was given)"
        )
        clean_results = evaluate_alerts(
            run_dir,
            held_out_shard_dir,
            baseline_shard_dir=baseline_shard_dir,
            fitted_baselines_out=fitted,
            max_shards=max_shards,
            max_baseline_shards=max_baseline_shards,
            landmark_hours=landmark_hours,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            baseline_feature_set=baseline_feature_set,
            tune_baselines=tune_baselines,
            stream_baseline=stream_baseline,
            dump_rows_path=clean_rows,
        )
        _write_cell_result(
            clean_json,
            cell=CLEAN_CELL,
            metadata=None,
            run_dir=run_dir,
            held_out_shard_dir=held_out_shard_dir,
            results=clean_results,
        )
        if baseline_shard_dir is not None and not fitted:
            logger.warning(
                "[sweep] --baseline-shard-dir was given but no GBM was fit for "
                "any (event, horizon) -- degraded cells will score without a "
                "GBM baseline. Check the held-out split has enough positives."
            )

    per_cell_json: Dict[str, Path] = {CLEAN_CELL: clean_json}
    for name, cell_dir in cell_dirs.items():
        cell_json = results_dir / f"{name}_alerts.json"
        cell_rows = results_dir / f"{name}_alerts_rows.parquet"
        if not overwrite and cell_json.is_file():
            # Resume: this cell was fully scored (the JSON is written last,
            # after the rows dump) -- reuse it rather than aborting the
            # whole sweep, which is what refuse_existing_output used to do
            # despite the module docstring's resumability promise.
            logger.info(
                "[sweep] cell %s already scored at %s, skipping", name, cell_json
            )
            per_cell_json[name] = cell_json
            continue
        refuse_existing_output(
            cell_json, overwrite=overwrite, kind=f"missingness sweep {name} alerts"
        )
        logger.info("[sweep] scoring cell %s (GBM reused, not refit)", name)
        unscoreable: Set[Tuple[int, int, float]] = set()
        results = evaluate_alerts(
            run_dir,
            held_out_shard_dir,
            degraded_shard_dir=cell_dir,
            verify_against_dump=clean_rows,
            unscoreable_out=unscoreable,
            prefit_baselines=fitted or None,
            max_shards=max_shards,
            landmark_hours=landmark_hours,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            baseline_feature_set=baseline_feature_set,
            stream_baseline=stream_baseline,
            dump_rows_path=cell_rows,
        )
        if unscoreable:
            logger.warning(
                "[sweep] cell %s: %d clean rows unscoreable on the degraded "
                "record -- this cell's metrics are over a reduced row set",
                name,
                len(unscoreable),
            )
        _write_cell_result(
            cell_json,
            cell=name,
            metadata=load_cell_metadata(cell_dir),
            run_dir=run_dir,
            held_out_shard_dir=held_out_shard_dir,
            results=results,
            n_unscoreable=len(unscoreable),
        )
        per_cell_json[name] = cell_json

    return aggregate(results_dir, per_cell_json, output_root)


def aggregate(
    results_dir: Path, per_cell_json: Dict[str, Path], output_root: Path
) -> Tuple[Path, Path]:
    """Build the degradation table from already-written per-cell JSON files.

    Split out from :func:`run_sweep` so a sweep that already has all its
    per-cell JSON on disk can be re-aggregated (e.g. after a
    missingness_report.py change) without re-running any GPU passes: glob
    ``results_dir/*_alerts.json`` and call this directly.
    """
    clean_payload = json.loads(per_cell_json[CLEAN_CELL].read_text())
    clean_metrics = load_cell_metrics(
        CLEAN_CELL,
        clean_payload["metrics"],
        transform=None,
        rows_path=results_dir / "clean_alerts_rows.parquet",
    )
    cells_metrics: Dict[str, List[CellMetricRow]] = {}
    for name, path in per_cell_json.items():
        if name == CLEAN_CELL:
            continue
        payload = json.loads(path.read_text())
        transform = (payload.get("cell_metadata") or {}).get("transform")
        cells_metrics[name] = load_cell_metrics(
            name,
            payload["metrics"],
            transform=transform,
            rows_path=results_dir / f"{name}_alerts_rows.parquet",
            n_unscoreable=payload.get("n_unscoreable") or 0,
        )
    table = build_degradation_table(clean_metrics, cells_metrics)
    json_path = output_root / "degradation_table.json"
    md_path = output_root / "degradation_table.md"
    write_json(table, json_path)
    write_markdown(table, md_path)
    logger.info("[sweep] degradation table: %s / %s", json_path, md_path)
    return json_path, md_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--held-out-shard-dir", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="everything this script writes lands here -- keep it outside the repo",
    )
    parser.add_argument("--baseline-shard-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="subset of degrade.py's 8 cell names (default: all 8)",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-baseline-shards", type=int, default=None)
    parser.add_argument("--landmark-hours", type=float, default=4.0)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument(
        "--baseline-features", choices=BASELINE_FEATURE_SETS, default="strong"
    )
    parser.add_argument("--no-tune-baselines", action="store_true")
    parser.add_argument("--stream-baseline-shards", action="store_true")
    parser.add_argument(
        "--source", default="mimic_iv", choices=("mimic_iv", "eicu", "gemini")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow re-running over an existing --output-root's results",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: run the full sweep and write the degradation table."""
    ensure_joblib_temp_folder()
    args = _parse_args(argv)
    json_path, md_path = run_sweep(
        args.run_dir,
        args.held_out_shard_dir,
        args.output_root,
        baseline_shard_dir=args.baseline_shard_dir,
        seed=args.seed,
        cells=args.cells,
        max_shards=args.max_shards,
        max_baseline_shards=args.max_baseline_shards,
        landmark_hours=args.landmark_hours,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        checkpoint=args.checkpoint,
        baseline_feature_set=args.baseline_features,
        tune_baselines=not args.no_tune_baselines,
        stream_baseline=args.stream_baseline_shards,
        source=args.source,
        overwrite=args.overwrite,
    )
    logger.info("[sweep] done: %s / %s", json_path, md_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
