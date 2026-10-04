"""The model's report card, built from the run's banked held-out evaluation.

Reads three aggregate files the evaluation chain already wrote into the
run directory and turns them into plain statements a clinician can check:

- ``alerts.json``: per (event, horizon, scorer) AUROC, Brier, calibration
  deciles and at-risk counts, for the hazard heads and the tuned GBM;
- ``alerts_cis.json``: subject-clustered bootstrap intervals, including
  the paired hazard-minus-GBM difference and whether it is separated;
- ``inference_results.json``: per-concept readout AUROC.

The headline is computed from those files, never written by hand, so it
cannot drift from the numbers beneath it.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apps.clinician_demo.schemas import (
    CalibrationBin,
    ConceptInfo,
    Scorecard,
    ScoreCell,
)
from apps.clinician_demo.thresholds import horizon_key


logger = logging.getLogger(__name__)

ALERTS_FILENAME = "alerts.json"
CIS_FILENAME = "alerts_cis.json"
INFERENCE_FILENAME = "inference_results.json"
HAZARD_SCORER = "hazard"
GBM_SCORER = "baseline_gbm"

NOTES = [
    "AUROC: how well the risk ranks patients who go on to have the event above "
    "those who do not; 0.5 is chance, 1.0 is perfect.",
    "Every number comes from held-out patients the model never trained on, "
    "scored every 4 hours of each admission while the event had not yet happened.",
    "The comparison model is a gradient-boosted classifier tuned on 609 "
    "hand-built features from the same records. It is the stronger alert "
    "model on most cells; this model's value is that its forecast is "
    "decomposed into named clinical concepts.",
    "Intervals are 95% subject-clustered bootstrap intervals. 'Separated' "
    "means the paired difference's interval excludes zero.",
]


def read_json(path: Path) -> Any | None:  # noqa: ANN401 -- arbitrary JSON
    """Parse ``path`` or return ``None`` (logged) if it is missing or unreadable."""
    if not path.exists():
        logger.warning("[scorecard] missing %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("[scorecard] unreadable %s: %s", path, exc)
        return None


def _interval(block: Mapping[str, Any] | None) -> list[float] | None:
    if not block or block.get("ci_low") is None or block.get("ci_high") is None:
        return None
    return [float(block["ci_low"]), float(block["ci_high"])]


def _find(
    rows: Sequence[Mapping[str, Any]], event: str, h: float, scorer: str
) -> Mapping[str, Any] | None:
    for row in rows:
        if (
            row.get("event") == event
            and row.get("scorer") == scorer
            and float(row.get("horizon_hours", -1)) == h
        ):
            return row
    return None


def score_cells(
    alerts: Sequence[Mapping[str, Any]],
    cis: Mapping[str, Any] | None,
    events: Sequence[str],
    horizons: Sequence[float],
) -> list[ScoreCell]:
    """Build one cell per (event, horizon) that has a hazard row in ``alerts``."""
    cells: list[ScoreCell] = []
    ci_cells: Mapping[str, Any] = (cis or {}).get("cells", {})
    for event in events:
        for h in horizons:
            hazard = _find(alerts, event, h, HAZARD_SCORER)
            if hazard is None:
                continue
            gbm = _find(alerts, event, h, GBM_SCORER)
            ci = ci_cells.get(f"{event}@{horizon_key(h)}", {})
            scorers = ci.get("scorers", {})
            delta_block = (
                ci.get("paired_deltas", {}).get("hazard_minus_gbm", {}).get("auroc")
            )
            n_at_risk = int(hazard.get("n_at_risk") or 0)
            n_positive = int(hazard.get("n_positive") or 0)
            hazard_auroc = hazard.get("auroc")
            gbm_auroc = gbm.get("auroc") if gbm else None
            delta = (
                float(delta_block["point"])
                if delta_block and delta_block.get("point") is not None
                else (
                    hazard_auroc - gbm_auroc
                    if hazard_auroc is not None and gbm_auroc is not None
                    else None
                )
            )
            cells.append(
                ScoreCell(
                    event=event,
                    horizon_hours=h,
                    n_at_risk=n_at_risk,
                    n_positive=n_positive,
                    base_rate=n_positive / n_at_risk if n_at_risk else 0.0,
                    hazard_auroc=hazard_auroc,
                    hazard_ci=_interval(scorers.get("hazard", {}).get("auroc")),
                    gbm_auroc=gbm_auroc,
                    gbm_ci=_interval(scorers.get("gbm", {}).get("auroc")),
                    delta=delta,
                    delta_ci=_interval(delta_block),
                    separated=delta_block.get("separated") if delta_block else None,
                    calibration=[
                        CalibrationBin(
                            predicted=float(b["predicted"]),
                            observed=float(b["observed"]),
                            n=int(b["n"]),
                        )
                        for b in hazard.get("calibration") or []
                        if b.get("predicted") is not None
                        and b.get("observed") is not None
                    ],
                )
            )
    return cells


def headline(cells: Sequence[ScoreCell]) -> str:
    """Write one computed sentence comparing the model with the tuned GBM."""
    compared = [c for c in cells if c.delta is not None]
    if not compared:
        return "No GBM comparison is available for this run."
    gbm_ahead = [c for c in compared if c.delta is not None and c.delta < 0]
    model_ahead = [c for c in compared if c.delta is not None and c.delta > 0]
    sep_gbm = sum(1 for c in gbm_ahead if c.separated)
    sep_model = sum(1 for c in model_ahead if c.separated)
    return (
        f"Against the tuned GBM on {len(compared)} event-horizon cells: the GBM ranks "
        f"better on {len(gbm_ahead)} ({sep_gbm} clearly), this model on "
        f"{len(model_ahead)} ({sep_model} clearly)."
    )


def concept_readouts(
    inference: Mapping[str, Any] | None, concepts: Sequence[ConceptInfo]
) -> list[ConceptInfo]:
    """Fill ``readout_auroc`` of ``concepts`` from ``inference_results.json``."""
    by_name: dict[str, float] = {}
    for metric in (inference or {}).get("concept_metrics", []) or []:
        if metric.get("auroc") is not None:
            by_name[str(metric["name"])] = float(metric["auroc"])
    return [
        ConceptInfo(
            name=c.name,
            display=c.display,
            description=c.description,
            readout_auroc=by_name.get(c.name, c.readout_auroc),
        )
        for c in concepts
    ]


def build_scorecard(
    run_dir: str | Path,
    events: Sequence[str],
    horizons: Sequence[float],
    concepts: Sequence[ConceptInfo],
) -> Scorecard:
    """Assemble the report card from the run directory's banked JSON files."""
    run_dir = Path(run_dir)
    alerts = read_json(run_dir / ALERTS_FILENAME) or []
    cis = read_json(run_dir / CIS_FILENAME)
    inference = read_json(run_dir / INFERENCE_FILENAME)
    cells = score_cells(alerts, cis, events, horizons)
    notes = list(NOTES)
    if not alerts:
        notes.insert(0, "This run has no banked alert evaluation; no cells to show.")
    elif cis is None:
        notes.insert(
            0,
            "No bootstrap intervals are banked for this run; differences are point estimates.",
        )
    return Scorecard(
        headline=headline(cells),
        cells=cells,
        concepts=concept_readouts(inference, concepts),
        notes=notes,
    )


__all__ = [
    "ALERTS_FILENAME",
    "CIS_FILENAME",
    "INFERENCE_FILENAME",
    "build_scorecard",
    "concept_readouts",
    "headline",
    "read_json",
    "score_cells",
]
