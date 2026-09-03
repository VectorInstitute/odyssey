"""Render the backbone comparison table from a banked long_history_compare.json.

Per event and horizon: both models' hazard AUROC and the paired
subject-clustered difference (a minus b) on subjects the transformer saw
whole and on subjects it saw truncated.

Usage::

    python scripts/make_backbone_table.py \
        <banked long_history_compare.json> \
        --output paper/ml4h/tables/backbone_long.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVENT_ORDER = (
    "death",
    "vasopressor_start",
    "acute_kidney_injury",
    "icu_admission",
    "sepsis3",
)
EVENT_NAMES = {
    "death": "Death",
    "vasopressor_start": "Vasopressor",
    "acute_kidney_injury": "AKI",
    "icu_admission": "ICU adm.",
    "sepsis3": "Sepsis-3",
}
HORIZONS = ("8h", "24h", "72h")
STRATA = ("whole", "truncated")


def _delta(cell: dict[str, Any]) -> str:
    d = cell["delta_a_minus_b"]
    if d is None:
        return "--"
    text = f"{d['point_estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
    excludes_zero = d["ci_low"] > 0 or d["ci_high"] < 0
    return f"\\textbf{{{text}}}" if excludes_zero else text


def render(result: dict[str, Any]) -> str:
    """Return the LaTeX tabular for one long_history_compare result."""
    la, lb = result["label_a"], result["label_b"]
    cells = {(c["event"], c["horizon"], c["stratum"]): c for c in result["cells"]}
    lines = [
        "\\begin{tabular}{@{}llrrlrrl@{}}",
        "\\toprule",
        f"& & \\multicolumn{{3}}{{c}}{{seen whole ({result['n_subjects'] - result['n_truncated_subjects']:,} subjects)}}"
        f" & \\multicolumn{{3}}{{c}}{{truncated ({result['n_truncated_subjects']:,} subjects)}} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        f"Event & $h$ & {la} & {lb} & $\\Delta$ [95\\% CI] & {la} & {lb} & $\\Delta$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    for event in EVENT_ORDER:
        first = True
        for h in HORIZONS:
            row = [EVENT_NAMES[event] if first else "", h.replace("h", "")]
            present = False
            for stratum in STRATA:
                c = cells.get((event, h, stratum))
                if c is None:
                    row += ["--", "--", "--"]
                    continue
                present = True
                row += [f"{c[f'auroc_{la}']:.3f}", f"{c[f'auroc_{lb}']:.3f}", _delta(c)]
            if present:
                lines.append(" & ".join(row) + " \\\\")
                first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.result.read_text())))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
