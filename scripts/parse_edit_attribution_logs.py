"""Parse cohort_worsen.py's stdout logs into one structured results JSON.

cohort_worsen.py (paper/ml4h/../scratch validation harness, not part of the
package) prints a fixed-format summary block per run: a header line with
the concept/source/n, an edit-signal-frequency dict, a mean concept-delta,
and one sign-agreement line per (event, horizon). This turns that text
into JSON so the paper's table/figure generators never hand-transcribe a
number from a terminal log.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_HEADER_RE = re.compile(
    r"===\s*(?P<concept>\S+)\s+on\s+(?P<source>\S+),\s+n=(?P<n>\d+)\s+subjects"
)
_ROW_RE = re.compile(
    r"^\s*(?P<event>\S+)\s+(?P<horizon>\S+)\s*:\s*(?P<agree>\d+)/\s*(?P<total>\d+)\s*=\s*(?P<pct>[\d.]+)%"
)
_SATURATION_RE = re.compile(
    r"baseline concept prob >= 0\.9:\s*(?P<n_sat>\d+)/(?P<n_total>\d+)"
)


def parse_log(text: str) -> dict[str, Any]:
    """Extract one run's summary from cohort_worsen.py's stdout.

    Raises ``ValueError`` if the header line (concept/source/n) is
    missing -- a log with no header is not a completed run and should
    not silently produce an empty result.
    """
    header = _HEADER_RE.search(text)
    if header is None:
        raise ValueError("no '=== <concept> on <source>, n=... ===' header found")
    sat = _SATURATION_RE.search(text)
    cells: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if m is None:
            continue
        cells.append(
            {
                "event": m["event"],
                "horizon": m["horizon"],
                "agree": int(m["agree"]),
                "total": int(m["total"]),
                "pct": float(m["pct"]),
            }
        )
    return {
        "concept": header["concept"],
        "source": header["source"],
        "n_subjects": int(header["n"]),
        "n_baseline_saturated": int(sat["n_sat"]) if sat else None,
        "n_baseline_scored": int(sat["n_total"]) if sat else None,
        "cells": cells,
    }


def main() -> None:
    """Parse the given log files and write their combined results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="cohort_worsen.py log files")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    runs = [parse_log(log.read_text()) for log in args.logs]
    args.out.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(f"wrote {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
