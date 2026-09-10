"""Combine one run's hazard scores with another run's baseline scores.

The tuned GBM is a property of the training data and the landmark rows,
not of the sequence model it is compared against, so a baseline fitted
once can serve every arm scored on the same rows. That is worth doing when
refitting is expensive: GEMINI's fit materializes every landmark row and
needs more memory than the node reliably has, so its arms would otherwise
each carry whichever baseline happened to be affordable that day.

The danger is pairing scores computed on DIFFERENT rows, which looks
identical in a table and is silently wrong. This project has already
shipped one table whose arms disagreed about which rows were at risk. So
every shared cell is checked: same at-risk count, same positive count, or
the splice refuses.

Usage::

    python scripts/splice_alerts_baseline.py \
        --hazard-from runs/a/alerts_full.json \
        --baseline-from runs/b/alerts_allshards.json \
        --out runs/a/alerts_spliced.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_SCORERS = {"baseline_gbm", "gbm"}


def _cells(
    records: list[dict[str, Any]],
) -> dict[tuple[str, float, str], dict[str, Any]]:
    return {(r["event"], float(r["horizon_hours"]), r["scorer"]): r for r in records}


def check_rows_align(
    hazard: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> None:
    """Raise unless every cell in both agrees on its at-risk row set.

    Row counts are the only handle the aggregate records give on whether
    two runs scored the same landmark rows. Equal counts do not prove
    identical rows, but unequal ones prove the opposite, and that is the
    failure worth refusing.
    """
    h, b = _cells(hazard), _cells(baseline)
    shared = {(e, ho) for e, ho, _ in h} & {(e, ho) for e, ho, _ in b}
    problems: list[str] = []
    for event, horizon in sorted(shared):
        hs = [r for (e, ho, _), r in h.items() if (e, ho) == (event, horizon)]
        bs = [r for (e, ho, _), r in b.items() if (e, ho) == (event, horizon)]
        for field in ("n_at_risk", "n_positive"):
            hv = {r[field] for r in hs}
            bv = {r[field] for r in bs}
            if hv != bv:
                problems.append(
                    f"{event}@{horizon:g}h {field}: hazard side {sorted(hv)}, "
                    f"baseline side {sorted(bv)}"
                )
    if problems:
        raise ValueError(
            "refusing to splice: the two runs did not score the same rows.\n  "
            + "\n  ".join(problems)
        )


def splice(
    hazard: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return hazard's records with its baseline scorers replaced by baseline's."""
    check_rows_align(hazard, baseline)
    kept = [r for r in hazard if r["scorer"] not in BASELINE_SCORERS]
    taken = [r for r in baseline if r["scorer"] in BASELINE_SCORERS]
    if not taken:
        raise ValueError(
            f"no baseline scorer in the baseline file; looked for {sorted(BASELINE_SCORERS)}"
        )
    return kept + taken


def main() -> None:
    """Write the spliced alerts records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard-from", type=Path, required=True)
    parser.add_argument("--baseline-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    hazard = json.loads(args.hazard_from.read_text())
    baseline = json.loads(args.baseline_from.read_text())
    merged = splice(hazard, baseline)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"wrote {args.out}: {len(merged)} records")
    print(f"  hazard side   : {args.hazard_from}")
    print(f"  baseline side : {args.baseline_from}")


if __name__ == "__main__":
    main()
