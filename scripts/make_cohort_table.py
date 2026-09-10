"""Render the cohort-sensitivity table from the banked cohort_check.json files.

For each dataset and alert event: the share of at-risk rows and of
positives that come from patients carrying an end-stage-renal-disease or
dialysis marker, or a palliative-care or hospice marker, and the 24-hour
hazard AUROC on all rows and with each group excluded.

Usage::

    python scripts/make_cohort_table.py \
        research_journal/figure_data/vm1/full_run_DEC_v12/cohort_check.json MIMIC-IV \
        research_journal/figure_data/vm2/eicu_full_DEC_v13/cohort_check.json eICU \
        --output paper/ml4h/tables/cohort_check.tex
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
    "vasopressor_start": "Vasopressor start",
    "acute_kidney_injury": "AKI",
    "icu_admission": "ICU admission",
    "sepsis3": "Sepsis-3",
}
GROUPS = (
    ("esrd_dialysis", "ESRD/dialysis"),
    ("palliative_hospice", "palliative/hospice"),
)


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}\\%" if den else "--"


def render(blocks: list[tuple[str, dict[str, Any]]]) -> str:
    """Return the LaTeX tabular body for the (label, cohort_check) pairs."""
    lines = [
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{ESRD/dialysis} & \\multicolumn{2}{c}{palliative/hospice} & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Dataset & Event & pos. & AUROC & pos. & AUROC & AUROC all \\\\",
        "\\midrule",
    ]
    for label, check in blocks:
        n_subj = check["n_held_out_subjects"]
        prev = check["flag_prevalence_subjects"]
        first = True
        for event in EVENT_ORDER:
            rec = check["rows_by_event"].get(event)
            if rec is None:
                continue
            cells = []
            for key, _ in GROUPS:
                grp = rec[key]
                cells.append(_pct(grp["positives_flagged"], rec["positives_24h"]))
                cells.append(f"{grp['auroc_24h_excluding_flagged']:.3f}")
            head = (
                f"{label} ({_pct(prev['esrd_dialysis'], n_subj)}, "
                f"{_pct(prev['palliative_hospice'], n_subj)} of patients)"
                if first
                else ""
            )
            first = False
            lines.append(
                f"{head} & {EVENT_NAMES[event]} & "
                + " & ".join(cells)
                + f" & {rec['auroc_24h_all']:.3f} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs", nargs="+", help="alternating cohort_check.json path and dataset label"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.pairs) % 2:
        parser.error("pairs must alternate path and label")
    blocks = [
        (args.pairs[i + 1], json.loads(Path(args.pairs[i]).read_text()))
        for i in range(0, len(args.pairs), 2)
    ]
    args.output.write_text(render(blocks))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
