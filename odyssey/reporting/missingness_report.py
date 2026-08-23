"""Missingness stress protocol: per-cell metric deltas vs the clean baseline.

docs/missingness_protocol.md. Pure computation over what
scripts/missingness_sweep.py already wrote to disk (each cell's
``{cell}_alerts.json`` from :func:`odyssey.inference.alerts.evaluate_alerts`,
already carrying the cell's own :mod:`odyssey.data.degrade` metadata) --
kept small and independently testable, never itself touches a live model or
shard directory.

Three metrics per (cell, scorer, event, horizon):

- **AUROC**: already computed by :func:`odyssey.inference.alerts.score_alerts`,
  read straight off the record.
- **AUPRC**: not in the aggregate record (only AUROC is) -- needs the raw
  (score, label) pairs, from the cell's own ``--dump-rows`` parquet. Best
  effort (:func:`auprc_from_rows`): ``None``, not a raise, if there's no row
  dump for this cell or the scorer's row-column naming isn't one of the two
  known conventions (``baseline_gbm``/``hazard``) -- AUROC/ECE still land
  either way.
- **ECE**: the weighted mean absolute gap between predicted and observed
  reused directly from the decile bins ``score_alerts`` already computed
  (:func:`ece_from_calibration``) -- no raw rows needed, unlike AUPRC.

Every metric also gets a delta (cell value minus the matching clean-baseline
value, same scorer/event/horizon) -- valid because the row set is identical
across every cell by construction (docs/missingness_protocol.md, Principle 3;
enforced by ``evaluate_alerts``'s own ``verify_against_dump``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import polars as pl


try:
    from sklearn.metrics import average_precision_score
except ImportError:  # pragma: no cover - sklearn is a real dependency elsewhere
    average_precision_score = None


#: The clean (undegraded) baseline's own cell name, shared between the sweep
#: script and this module so both sides of the JSON/markdown handoff agree.
CLEAN_CELL = "clean"

#: How to find a scorer's raw-score column in a --dump-rows parquet, keyed
#: by odyssey.inference.alerts.AlertMetrics.scorer. Only the two scorers the
#: missingness sweep actually produces (see scripts/missingness_sweep.py) --
#: any other scorer name (e.g. a future extra_baselines family) falls
#: through to auprc_from_rows' best-effort None rather than guessing a
#: naming convention that hasn't been confirmed.
_ROW_COLUMN_BY_SCORER = {
    "baseline_gbm": "gbm@{h:g}h",
    "hazard": "hazard@{h:g}h",
}


def ece_from_calibration(
    calibration: Optional[Sequence[Dict[str, float]]],
) -> Optional[float]:
    """Weighted mean ``|predicted - observed|`` over score_alerts' own decile bins."""
    if not calibration:
        return None
    total_n = sum(b["n"] for b in calibration)
    if total_n == 0:
        return None
    return (
        sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in calibration) / total_n
    )


def auprc_from_rows(
    rows_path: Optional[Path], *, scorer: str, horizon_hours: float
) -> Optional[float]:
    """AUPRC from a cell's per-row dump, or ``None`` if it can't be computed.

    See the module docstring: best-effort by design, not a raise -- a
    missing row dump or an unrecognized scorer naming convention just means
    this one cell's AUPRC column is empty, not that the whole aggregation
    fails.
    """
    if rows_path is None or not rows_path.is_file() or average_precision_score is None:
        return None
    template = _ROW_COLUMN_BY_SCORER.get(scorer)
    if template is None:
        return None
    score_col = template.format(h=horizon_hours)
    label_col = f"y@{horizon_hours:g}h"
    frame = pl.read_parquet(rows_path)
    if score_col not in frame.columns or label_col not in frame.columns:
        return None
    pair = frame.select(score_col, label_col).drop_nulls()
    if pair.height == 0:
        return None
    y = pair[label_col].to_numpy()
    if y.min() == y.max():
        return None
    return float(average_precision_score(y, pair[score_col].to_numpy()))


@dataclass(frozen=True)
class CellMetricRow:
    """One (cell, scorer, event, horizon)'s metrics, ready to compare against clean."""

    cell: str
    transform: Optional[str]
    scorer: str
    event: str
    horizon_hours: float
    n_at_risk: int
    auroc: Optional[float]
    auprc: Optional[float]
    ece: Optional[float]


def load_cell_metrics(
    cell: str,
    metrics: Sequence[Dict[str, Any]],
    *,
    transform: Optional[str],
    rows_path: Optional[Path],
) -> List[CellMetricRow]:
    """Build one cell's CellMetricRow list from its parsed AlertMetrics records.

    ``metrics`` is the ``"metrics"`` list a ``{cell}_alerts.json`` written by
    scripts/missingness_sweep.py carries (``AlertMetrics`` as dicts) --
    parsing the wrapper JSON is the caller's job, this only computes.
    """
    out = []
    for r in metrics:
        scorer = str(r["scorer"])
        horizon_hours = float(r["horizon_hours"])
        out.append(
            CellMetricRow(
                cell=cell,
                transform=transform,
                scorer=scorer,
                event=str(r["event"]),
                horizon_hours=horizon_hours,
                n_at_risk=int(r["n_at_risk"]),
                auroc=r.get("auroc"),
                auprc=auprc_from_rows(
                    rows_path, scorer=scorer, horizon_hours=horizon_hours
                ),
                ece=ece_from_calibration(r.get("calibration")),
            )
        )
    return out


def _delta(
    cell_value: Optional[float], clean_value: Optional[float]
) -> Optional[float]:
    if cell_value is None or clean_value is None:
        return None
    return cell_value - clean_value


def build_degradation_table(
    clean: Sequence[CellMetricRow], cells: Mapping[str, Sequence[CellMetricRow]]
) -> List[Dict[str, Any]]:
    """One row per (cell, scorer, event, horizon): metrics plus their delta from clean.

    The clean-baseline row is matched by (scorer, event, horizon) alone --
    valid because the row set is identical across every cell (see the module
    docstring). A cell row with no matching clean row (shouldn't happen if
    verify_against_dump passed, kept as a safe fallback) gets ``None`` deltas
    rather than raising.
    """
    clean_by_key = {(r.scorer, r.event, r.horizon_hours): r for r in clean}
    table: List[Dict[str, Any]] = []
    for cell_name, rows in cells.items():
        for r in rows:
            base = clean_by_key.get((r.scorer, r.event, r.horizon_hours))
            table.append(
                {
                    "cell": cell_name,
                    "transform": r.transform,
                    "scorer": r.scorer,
                    "event": r.event,
                    "horizon_hours": r.horizon_hours,
                    "n_at_risk": r.n_at_risk,
                    "auroc": r.auroc,
                    "auroc_delta": _delta(r.auroc, base.auroc if base else None),
                    "auprc": r.auprc,
                    "auprc_delta": _delta(r.auprc, base.auprc if base else None),
                    "ece": r.ece,
                    "ece_delta": _delta(r.ece, base.ece if base else None),
                }
            )
    return sorted(
        table,
        key=lambda row: (
            row["cell"],
            row["scorer"],
            row["event"],
            row["horizon_hours"],
        ),
    )


def write_json(table: List[Dict[str, Any]], path: Path) -> None:
    """Write the degradation table as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2))


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(table: List[Dict[str, Any]]) -> str:
    """Render the degradation table as a markdown table."""
    lines = [
        "# Missingness stress protocol: degradation table",
        "",
        "AUROC/AUPRC/ECE and their delta from the clean baseline "
        "(docs/missingness_protocol.md), per cell x scorer x event x horizon.",
        "",
        "| cell | transform | scorer | event | horizon (h) | n | "
        "AUROC | ΔAUROC | AUPRC | ΔAUPRC | ECE | ΔECE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            "| {cell} | {transform} | {scorer} | {event} | {horizon_hours:g} | "
            "{n_at_risk} | {auroc} | {auroc_delta} | {auprc} | {auprc_delta} | "
            "{ece} | {ece_delta} |".format(
                cell=row["cell"],
                transform=row["transform"] or "-",
                scorer=row["scorer"],
                event=row["event"],
                horizon_hours=row["horizon_hours"],
                n_at_risk=row["n_at_risk"],
                auroc=_fmt(row["auroc"]),
                auroc_delta=_fmt(row["auroc_delta"]),
                auprc=_fmt(row["auprc"]),
                auprc_delta=_fmt(row["auprc_delta"]),
                ece=_fmt(row["ece"]),
                ece_delta=_fmt(row["ece_delta"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_markdown(table: List[Dict[str, Any]], path: Path) -> None:
    """Render and write the degradation table as a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(table))
