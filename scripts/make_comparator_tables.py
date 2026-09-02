"""Generate the paper's comparator table body from run JSONs, never by hand.

The comparator tables (``tab:mimic``, ``tab:eicu`` in paper/ml4h/main.tex)
were hand-transcribed from run outputs. On 2026-08-31 that produced a wrong
cell in ``tab:decomp`` (a loss delta printed +0.024 when the raw values give
+0.023) which survived two review passes, and left both comparator tables on
prior-generation numbers while post-fix ones sat in files. This script makes
the tables a build product of the JSONs instead.

Inputs (only ``--alerts`` is required, so it is usable before every baseline
has landed):

* ``--alerts``  ``alerts.json`` from the eval chain -- hazard and GBM point
  estimates, ``n_at_risk``/``n_positive``, and the landmark protocol stamp.
* ``--cis``     ``alerts_cis.json`` from ``scripts/alerts_cis.py`` -- adds
  subject-clustered CIs and the paired hazard-vs-GBM delta. Bold means "best
  in the row"; with this file, bold is only applied when the paired delta
  actually separates, which is what the table captions promise.
* ``--tabicl``  ``tabicl_strong_v4.json`` from
  ``scripts/tabicl_strong_compare.py`` -- adds the TabICL column.

Anything absent is reported in the emitted LaTeX as a comment rather than
silently dropped, so a half-built table cannot be mistaken for a finished one.

Usage::

    uv run python scripts/make_comparator_tables.py \\
        --alerts research_journal/figure_data/vm1/full_run_v10/alerts.json \\
        --tabicl research_journal/figure_data/vm1/full_run_v10/tabicl_strong_v4.json \\
        --cis research_journal/figure_data/vm1/full_run_v10/alerts_cis.json \\
        --output-tex paper/ml4h/tables/mimic_comparator.tex
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("comparator_tables")

# Display names and the row order the paper uses. Events absent from a given
# run (sepsis3 on eICU, which the source resolution drops) are skipped.
EVENT_ORDER: tuple[tuple[str, str], ...] = (
    ("acute_kidney_injury", "AKI"),
    ("death", "Death"),
    ("icu_admission", "ICU adm."),
    ("vasopressor_start", "Vasopressor"),
    ("sepsis3", "Sepsis-3"),
    ("readmission_30d", "Readmission"),
)
HAZARD = "hazard"
GBM = "baseline_gbm"
# tabicl_strong_compare.py names the chain's GBM "gbm"; alerts.json calls the
# same scorer "baseline_gbm".
_MATCHED_NAMES = {"hazard": HAZARD, "gbm": GBM, "tabicl": "tabicl"}
# Every spelling a paired-delta key might use for a scorer, for the bold rule.
_ALIASES: dict[str, tuple[str, ...]] = {GBM: (GBM, "gbm"), HAZARD: (HAZARD,)}


def _fmt(value: float | None) -> str:
    """Three decimals, or an em-free placeholder when the cell has no value."""
    return "--" if value is None else f"{value:.3f}"


def _cell_tex(scores: dict[str, float | None], bold: set[str], name: str) -> str:
    """One table cell, bolded when this scorer wins the row."""
    text = _fmt(scores.get(name))
    return f"\\textbf{{{text}}}" if name in bold else text


def load_alerts(path: Path) -> tuple[dict[str, dict[str, Any]], set[int]]:
    """Point estimates and row counts, keyed ``{event}@{horizon:g}h``."""
    records = json.loads(path.read_text())
    cells: dict[str, dict[str, Any]] = {}
    protocols: set[int] = set()
    for rec in records:
        if rec.get("scorer") not in (HAZARD, GBM):
            continue
        key = f"{rec['event']}@{rec['horizon_hours']:g}h"
        cell = cells.setdefault(
            key,
            {
                "n": rec.get("n_at_risk"),
                "n_positive": rec.get("n_positive"),
                "scores": {},
            },
        )
        cell["scores"][rec["scorer"]] = rec.get("auroc")
        if rec.get("landmark_protocol_version") is not None:
            protocols.add(int(rec["landmark_protocol_version"]))
    return cells, protocols


def load_tabicl(path: Path) -> tuple[dict[str, float | None], dict[str, int | None]]:
    """TabICL point estimates, and the row count each was scored on.

    The row counts are returned so the caller can refuse to print a
    TabICL score beside an ``n`` it was not computed on; see
    :func:`check_row_sets`.
    """
    raw = json.loads(path.read_text())
    out: dict[str, float | None] = {}
    rows: dict[str, int | None] = {}
    for key, cell in raw.items():
        entry = cell.get("tabicl")
        out[key] = None if entry is None else entry.get("point_estimate")
        rows[key] = cell.get("n")
    return out, rows


def check_row_sets(
    cells: dict[str, dict[str, Any]], tabicl_rows: dict[str, int | None]
) -> None:
    """Refuse to merge TabICL scored on a different row set than the table.

    TabICL costs about half an hour of inference per cell, so it is run on
    a subsample while the alerts table covers every shard. Only the score
    crosses over in the merge: the ``n`` column keeps saying the alerts
    figure, so a subsample AUROC lands beside a row count it was never
    computed on and the table looks finished. That is this project's most
    repeated analysis bug, and the "pass --tabicl" note this script prints
    on a full-coverage table walks straight into it -- the matched tables
    exist precisely so the two coverages never share a column.
    """
    mismatched = {
        key: (cells[key]["n"], n)
        for key, n in tabicl_rows.items()
        if key in cells and n is not None and cells[key]["n"] not in (None, n)
    }
    if mismatched:
        detail = ", ".join(
            f"{key}: table {tab:,} vs TabICL {tic:,}"
            for key, (tab, tic) in sorted(mismatched.items())
        )
        raise SystemExit(
            f"TabICL was scored on a different row set than this table in "
            f"{len(mismatched)} cell(s): {detail}. Point estimates from "
            "different coverages must not share a table. Build the matched "
            "table with --matched instead of merging with --tabicl"
        )


def load_matched(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Merge ``tabicl_strong_compare.py`` outputs into alerts-shaped cells.

    That script emits one file per pass over the same rows (hazard+GBM
    from the ``--skip-tabicl`` pass, GBM+TabICL from the fitted one), each
    keyed ``{event}@{h}h`` with a CI record per scorer. Merging them is
    only legitimate if the passes really did score the same rows, so the
    GBM column both files carry is checked against itself: it is the same
    model on the same rows, and must agree to the last digit. When it
    does not, the row sets differ and the merge is refused rather than
    producing a table whose columns describe different samples.
    """
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for key, cell in json.loads(path.read_text()).items():
            slot = merged.setdefault(
                key,
                {
                    "n": cell.get("n"),
                    "n_positive": cell.get("n_positive"),
                    "scores": {},
                },
            )
            if slot["n"] != cell.get("n"):
                raise SystemExit(
                    f"{key}: {path} scored {cell.get('n')} rows but an earlier "
                    f"file scored {slot['n']}; these are not the same row set"
                )
            for scorer in ("hazard", "gbm", "tabicl"):
                entry = cell.get(scorer)
                if entry is None:
                    continue
                point = entry.get("point_estimate")
                previous = slot["scores"].get(_MATCHED_NAMES[scorer])
                if previous is not None and abs(previous - point) > 1e-9:
                    raise SystemExit(
                        f"{key}: {scorer} is {point} in {path} but {previous} in "
                        "an earlier file. The same model on the same rows cannot "
                        "disagree, so the row sets differ; refusing to merge"
                    )
                slot["scores"][_MATCHED_NAMES[scorer]] = point
    return merged


def load_cis(path: Path) -> dict[str, dict[str, Any]]:
    """Per-cell CI/paired-delta records from scripts/alerts_cis.py."""
    return dict(json.loads(path.read_text()).get("cells", {}))


def _bold_targets(
    scores: dict[str, float | None], ci_cell: dict[str, Any] | None
) -> set[str]:
    """Which scorers to bold in this row.

    Without CIs, the plain arg-max. With them, bold only when the leader's
    paired delta against the runner-up actually excludes zero -- the
    captions promise bold means a separated cell, not merely a larger
    number, and a 0.002 lead inside a CI that spans 0.03 is not a win.

    The delta is looked up under both scorer vocabularies and in both
    directions: alerts.json calls the boosted baseline ``baseline_gbm``
    while tabicl_strong_compare.py calls the same scorer ``gbm``, and
    alerts_cis.py keys a pair by whichever scorer was passed first. A
    lookup that assumed one spelling would not find the delta and would
    silently fall back to arg-max, which is precisely the bold the
    captions forbid.
    """
    present = {k: v for k, v in scores.items() if v is not None}
    if not present:
        return set()
    best = max(present, key=lambda k: present[k])
    if ci_cell is None or len(present) < 2:
        return {best}
    runner = max((k for k in present if k != best), key=lambda k: present[k])
    deltas = ci_cell.get("paired_deltas", {})
    for a, b in ((best, runner), (runner, best)):
        for a_name in _ALIASES.get(a, (a,)):
            for b_name in _ALIASES.get(b, (b,)):
                delta = deltas.get(f"{a_name}_minus_{b_name}")
                if delta is None:
                    continue
                auroc = delta.get("auroc")
                if auroc is None:
                    continue
                return {best} if auroc.get("separated", False) else set()
    # No paired delta for this pair: the CI file does not cover it, so the
    # promise behind bold cannot be checked and must not be implied.
    return set()


def build_rows(
    cells: dict[str, dict[str, Any]],
    tabicl: dict[str, float | None],
    cis: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """LaTeX row strings plus any provenance notes worth emitting."""
    rows: list[str] = []
    notes: list[str] = []
    for event, display in EVENT_ORDER:
        horizons = sorted(
            {
                float(key.split("@")[1].rstrip("h"))
                for key in cells
                if key.split("@")[0] == event
            }
        )
        if not horizons:
            continue
        for i, horizon in enumerate(horizons):
            key = f"{event}@{horizon:g}h"
            cell = cells[key]
            scores = dict(cell["scores"])
            if key in tabicl:
                scores["tabicl"] = tabicl[key]
            bold = _bold_targets(scores, cis.get(key))

            label = f"\\multirow{{{len(horizons)}}}{{*}}{{{display}}}" if i == 0 else ""
            n_txt = (
                "--" if cell["n"] is None else f"{cell['n']:,} ({cell['n_positive']:,})"
            )
            columns = [
                label,
                f"{horizon:g}h",
                n_txt,
                _cell_tex(scores, bold, HAZARD),
                _cell_tex(scores, bold, GBM),
            ]
            if tabicl:
                columns.append(_cell_tex(scores, bold, "tabicl"))
            rows.append(" & ".join(columns) + r" \\")
        rows.append("\\midrule" if event != EVENT_ORDER[-1][0] else "")
    while rows and rows[-1] in ("", "\\midrule"):
        rows.pop()
    if not tabicl:
        notes.append(
            "TabICL column absent. Only merge --tabicl when it was scored on "
            "THIS table's rows; a subsample belongs in a --matched table"
        )
    if not cis:
        notes.append(
            "NO CIs: bold is plain arg-max, which the caption's bold rule does "
            "not permit at submission. Run scripts/alerts_cis.py and pass --cis"
        )
    # A PARTIAL supplementary file is more dangerous than a missing one: the
    # column renders, most cells quietly show "--", and the table looks
    # finished. tabicl_strong_compare.py in particular rewrites its output
    # after every cell, so pointing at a still-running job yields exactly
    # this. Warn on coverage, not just presence.
    for label, supplied in (("TabICL", tabicl), ("CI", cis)):
        if not supplied:
            continue
        missing = sorted(set(cells) - set(supplied))
        if missing:
            notes.append(
                f"{label} data is PARTIAL: {len(supplied)} of {len(cells)} cells "
                f"covered, missing {', '.join(missing)}. If the producing job is "
                "still running, this table is premature"
            )
    return rows, notes


def main() -> None:
    """Emit the comparator table body for one dataset arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--alerts", type=Path)
    source.add_argument(
        "--matched",
        type=Path,
        action="append",
        help=(
            "tabicl_strong_compare.py output(s) over one matched row set; "
            "repeat to merge the --skip-tabicl pass with the fitted one"
        ),
    )
    parser.add_argument("--tabicl", type=Path, default=None)
    parser.add_argument("--cis", type=Path, default=None)
    parser.add_argument("--output-tex", required=True, type=Path)
    args = parser.parse_args()

    cells: dict[str, dict[str, Any]]
    protocols: set[int]
    if args.matched:
        # The matched files carry no protocol stamp; they are built from a
        # dump that does, and the header says so via its source line.
        cells, protocols = load_matched(args.matched), set()
        tabicl = {
            key: cell["scores"]["tabicl"]
            for key, cell in cells.items()
            if "tabicl" in cell["scores"]
        }
    else:
        cells, protocols = load_alerts(args.alerts)
        tabicl = {}
        if args.tabicl:
            tabicl, tabicl_rows = load_tabicl(args.tabicl)
            check_row_sets(cells, tabicl_rows)
    cis = load_cis(args.cis) if args.cis else {}

    rows, notes = build_rows(cells, tabicl, cis)
    if len(protocols) > 1:
        raise SystemExit(
            f"alerts.json mixes landmark protocol versions {sorted(protocols)}; "
            "v4 and v1-v3 cells are not comparable and must not share a table"
        )
    protocol = next(iter(protocols), None)

    header = [
        "% GENERATED by scripts/make_comparator_tables.py -- do not hand-edit.",
        f"% source: {args.alerts or ', '.join(str(p) for p in args.matched)}",
        f"% landmark protocol: {'unstamped' if protocol is None else f'v{protocol}'}",
        f"% events: {len({k.split('@')[0] for k in cells})}, cells: {len(cells)}",
    ]
    header += [f"% WARNING: {n}" for n in notes]
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    # Emit the COMPLETE tabular, not just its rows. \input-ing a bare row
    # body inside a tabular breaks LaTeX's alignment scanning ("Misplaced
    # \noalign" at the following \bottomrule), and the generator is the
    # thing that knows how many columns it produced, so it should own the
    # preamble and header too.
    has_tabicl = bool(tabicl)
    spec = "llrrrr" if has_tabicl else "llrrr"
    head_cells = ["Event", "$h$", "$n$ (pos)", "Hazard", "GBM"]
    if has_tabicl:
        head_cells.append("TabICL")
    table = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(head_cells) + r" \\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    args.output_tex.write_text("\n".join(header + table) + "\n")
    for note in notes:
        logger.warning("%s", note)
    logger.info(
        "wrote %s (%d cells, protocol %s)",
        args.output_tex,
        len(cells),
        "unstamped" if protocol is None else f"v{protocol}",
    )


if __name__ == "__main__":
    main()
