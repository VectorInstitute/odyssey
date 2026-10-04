"""Alert lines: the risk threshold per (event, horizon) and what it buys.

A risk number alone does not tell a clinician whether to act. The demo
draws an alert line instead: the threshold at which the model would flag a
fixed share of at-risk moments (default 5%), measured on the run's own
held-out landmark rows (``alerts_rows.parquet``) -- the same rows its
published AUROCs come from. For that line it reports how many of the
moments later followed by the event were flagged (sensitivity), how many
flags were right (PPV), and the tuned GBM's sensitivity/PPV when it flags
the same share of moments, so the comparison is like for like.

Only aggregates leave this module; the row file is patient-level and is
read in place on the host.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from apps.clinician_demo.schemas import OperatingPoint, to_jsonable


logger = logging.getLogger(__name__)

ALERTS_ROWS_FILENAME = "alerts_rows.parquet"
CACHE_VERSION = 1


def horizon_key(horizon_hours: float) -> str:
    """Format a horizon the way the banked files do: ``24.0`` -> ``"24h"``."""
    return f"{horizon_hours:g}h"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def flag_threshold(scores: pl.Series, alert_rate: float) -> float:
    """Return the score at or above which ``alert_rate`` of ``scores`` are flagged.

    Uses the upper quantile (``interpolation="higher"``) so the threshold is
    an observed score; ties at the threshold can make the flagged share a
    little larger than ``alert_rate``, which callers report as measured.
    """
    value = scores.quantile(1.0 - alert_rate, interpolation="higher")
    if value is None:
        raise ValueError("cannot set a threshold on an empty score column")
    return float(value)


def _flag_stats(
    scores: pl.Series, outcome: pl.Series, threshold: float
) -> tuple[float, float | None, float | None]:
    """Return ``(flagged share, sensitivity, PPV)`` for ``scores >= threshold``."""
    flagged = scores >= threshold
    positive = outcome == 1
    n_flagged = int(flagged.sum())
    n_pos = int(positive.sum())
    hits = int((flagged & positive).sum())
    return n_flagged / len(scores), _rate(hits, n_pos), _rate(hits, n_flagged)


def operating_point(
    rows: pl.DataFrame, event: str, horizon_hours: float, alert_rate: float
) -> OperatingPoint | None:
    """Compute the alert line for one (event, horizon) over at-risk rows.

    ``rows`` needs ``event``, ``hazard@{h}h``, ``y@{h}h`` and optionally
    ``gbm@{h}h``. Rows whose outcome is null (not at risk, or censored
    before the horizon) are excluded, as in the published metrics. Returns
    ``None`` when no at-risk row exists.
    """
    key = horizon_key(horizon_hours)
    hazard, outcome, gbm = f"hazard@{key}", f"y@{key}", f"gbm@{key}"
    at_risk = rows.filter(
        (pl.col("event") == event)
        & pl.col(outcome).is_not_null()
        & pl.col(hazard).is_not_null()
    )
    if at_risk.height == 0:
        return None
    threshold = flag_threshold(at_risk[hazard], alert_rate)
    share, sensitivity, ppv = _flag_stats(at_risk[hazard], at_risk[outcome], threshold)
    gbm_sens: float | None = None
    gbm_ppv: float | None = None
    if gbm in at_risk.columns:
        scored = at_risk.filter(pl.col(gbm).is_not_null())
        if scored.height:
            gbm_threshold = flag_threshold(scored[gbm], alert_rate)
            _, gbm_sens, gbm_ppv = _flag_stats(
                scored[gbm], scored[outcome], gbm_threshold
            )
    return OperatingPoint(
        event=event,
        horizon_hours=horizon_hours,
        threshold=threshold,
        alert_rate=share,
        sensitivity=sensitivity,
        ppv=ppv,
        base_rate=float(at_risk[outcome].mean() or 0.0),  # type: ignore[arg-type]
        n_rows=at_risk.height,
        gbm_sensitivity=gbm_sens,
        gbm_ppv=gbm_ppv,
    )


def compute_operating_points(
    rows_path: str | Path,
    events: Sequence[str],
    horizons: Sequence[float],
    alert_rate: float,
) -> list[OperatingPoint]:
    """Compute operating points for every (event, horizon) from a row dump.

    Reads only the columns needed, one event at a time, so the multi-million
    row dump is never held whole in memory.
    """
    lf = pl.scan_parquet(rows_path)
    available = set(lf.collect_schema().names())
    points: list[OperatingPoint] = []
    for event in events:
        for h in horizons:
            key = horizon_key(h)
            columns = [
                c
                for c in ("event", f"hazard@{key}", f"y@{key}", f"gbm@{key}")
                if c in available
            ]
            if f"hazard@{key}" not in columns or f"y@{key}" not in columns:
                logger.warning("[thresholds] %s has no %s columns", rows_path, key)
                continue
            rows = lf.select(columns).filter(pl.col("event") == event).collect()
            point = operating_point(rows, event, h, alert_rate)
            if point is not None:
                points.append(point)
    return points


def _signature(
    rows_path: Path, events: Sequence[str], horizons: Sequence[float], alert_rate: float
) -> dict[str, object]:
    stat = rows_path.stat()
    return {
        "version": CACHE_VERSION,
        "rows": str(rows_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "events": list(events),
        "horizons": list(horizons),
        "alert_rate": alert_rate,
    }


def load_or_compute_operating_points(
    rows_path: str | Path,
    cache_path: str | Path,
    events: Sequence[str],
    horizons: Sequence[float],
    alert_rate: float,
) -> list[OperatingPoint]:
    """Run :func:`compute_operating_points`, cached in a JSON file of aggregates.

    The cache is reused only when the row file (path, size, mtime) and the
    request (events, horizons, alert rate) match exactly; anything else
    recomputes and rewrites it.
    """
    rows_path, cache_path = Path(rows_path), Path(cache_path)
    signature = _signature(rows_path, events, horizons, alert_rate)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("signature") == signature:
                return [OperatingPoint(**p) for p in cached["points"]]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "[thresholds] ignoring unreadable cache %s: %s", cache_path, exc
            )
    points = compute_operating_points(rows_path, events, horizons, alert_rate)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"signature": signature, "points": to_jsonable(points)}, indent=1)
    )
    return points


__all__ = [
    "ALERTS_ROWS_FILENAME",
    "compute_operating_points",
    "flag_threshold",
    "horizon_key",
    "load_or_compute_operating_points",
    "operating_point",
]
