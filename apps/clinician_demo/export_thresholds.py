"""Export a run's alert lines as aggregates, for serving the demo elsewhere.

The demo sets its alert lines from the run's patient-level
``alerts_rows.parquet``, which stays on the GPU host. Run this there once
per run; copy the resulting JSON next to the checkpoint and the demo uses
it on a host without the rows (for example a laptop)::

    python -m apps.clinician_demo.export_thresholds --run-dir ~/runs/full_run_v10
"""

import argparse
import sys
from pathlib import Path

from apps.clinician_demo.config import HORIZONS_HOURS
from apps.clinician_demo.thresholds import (
    AGGREGATE_THRESHOLDS_FILENAME,
    ALERTS_ROWS_FILENAME,
    export_operating_points,
)


def main(argv: list[str] | None = None) -> int:
    """Write ``<run-dir>/demo_thresholds_aggregate.json``; return the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m apps.clinician_demo.export_thresholds", description=__doc__
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alert-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser()
    rows = run_dir / ALERTS_ROWS_FILENAME
    if not rows.exists():
        parser.error(f"no {ALERTS_ROWS_FILENAME} in {run_dir}")
    out = run_dir / AGGREGATE_THRESHOLDS_FILENAME
    points = export_operating_points(rows, out, HORIZONS_HOURS, args.alert_rate)
    print(f"wrote {len(points)} alert lines to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
