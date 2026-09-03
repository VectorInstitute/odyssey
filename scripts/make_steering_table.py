"""Turn steering-benchmark JSON into the paper's dial table.

One row per concept and direction; one column per declared outcome at the
chosen horizon, showing the steered-over-unsteered risk ratio, bold when
the paired subject-clustered delta separates from zero and has the
clinically expected sign, and marked with a dagger when it separates the
WRONG way. A second JSON (``--after``) adds a matching block so a
steering-trained model can sit beside its untrained parent on the same
shard and dials.

Usage::

    uv run python scripts/make_steering_table.py \\
        --before figure_data/vm1/full_run_DEC_v12/steering_smoke_v2.json \\
        --after figure_data/vm1/full_run_DEC_v12_steer/steering_smoke.json \\
        --output-tex paper/ml4h/tables/steering_dials.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from odyssey.data.concepts import canonical_concept_name


logger = logging.getLogger(__name__)

EVENT_LABELS = {
    "vasopressor_start": "Pressors",
    "icu_admission": "ICU",
    "acute_kidney_injury": "AKI",
    "death": "Death",
    "sepsis3": "Sepsis-3",
    "readmission_30d": "Readmit",
}


def _cell(outcome: dict[str, Any]) -> str:
    """One risk ratio, bold if separated as expected, dagger if separated wrong."""
    ratio = outcome["relative_change"]
    if math.isnan(ratio):  # never at risk of this event
        return "--"
    text = f"{ratio:.2f}"
    if outcome.get("expected_sign") is None:
        return text
    if outcome["separated"] and outcome["as_expected"]:
        return f"\\textbf{{{text}}}"
    if outcome["separated"] and not outcome["as_expected"]:
        return f"{text}$^\\dagger$"
    return text


# Long registry names that would not fit a half-width dial table.
DISPLAY_NAMES = {
    "sustained hypotension map": "sust.\\ hypotension (MAP)",
    "sustained tachypnea": "sust.\\ tachypnea",
    "hypoxemic respiratory failure": "hypox.\\ resp.\\ failure",
    "sirs": "SIRS",
    "qsofa": "qSOFA",
    "sepsis3": "Sepsis-3",
    "aki stage 2": "AKI stage 2",
    "aki stage 3": "AKI stage 3",
    "acute kidney injury": "AKI (any stage)",
    # severe-range thresholds (Hb < 7 g/dL, K < 3.0 mmol/L): say so in the name
    "anemia": "severe anemia",
    "hypokalemia": "severe hypokalemia",
}


def rows_for(
    payload: dict[str, Any], *, horizon: float
) -> list[tuple[str, str, str, list[str]]]:
    """``(concept, direction, respond, cells)`` per summary at ``horizon``."""
    events = [e for e in payload["event_names"] if e in EVENT_LABELS]
    out = []
    for s in payload["summaries"]:
        by_event = {(o["event"], o["horizon_hours"]): o for o in s["outcomes"]}
        cells = []
        for e in events:
            o = by_event.get((e, horizon))
            if o is None:
                cells.append("\\textcolor{gray}{--}")
            elif o.get("expected_sign") is None:
                cells.append(f"\\textcolor{{gray}}{{{o['relative_change']:.2f}}}")
            else:
                cells.append(_cell(o))
        respond = f"{s['respond_baseline']:.2f}$\\to${s['respond_steered']:.2f}"
        arrow = "$\\uparrow$" if s["direction"] == "amplify" else "$\\downarrow$"
        name = canonical_concept_name(s["concept"]).replace("_", " ")
        name = DISPLAY_NAMES.get(name, name)
        out.append((name, arrow, respond, cells))
    return out


def _tabular(
    events: list[str], rows: list[tuple[str, str, str, list[str]]], label: str | None
) -> list[str]:
    cols = "ll" + "l" + "r" * len(events)
    head = (
        " & ".join(["Dial", "", "$k_c$", *[EVENT_LABELS[e] for e in events]]) + " \\\\"
    )
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head, "\\midrule"]
    if label is not None:
        lines.append(
            f"\\multicolumn{{{3 + len(events)}}}{{l}}{{\\emph{{{label}}}}} \\\\"
        )
    for concept, arrow, respond, cells in rows:
        lines.append(" & ".join([concept, arrow, respond, *cells]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def render_side_by_side(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    *,
    horizon: float,
    label_before: str = "before steering training",
    label_after: str = "after steering training",
) -> str:
    """Two tabulars in minipages: before | after, or the rows split in halves.

    Half the height of :func:`render` for the same content, so a 40-row
    dial table can share a page with another float.
    """
    events = [e for e in before["event_names"] if e in EVENT_LABELS]
    if after is not None:
        panels: list[tuple[str | None, list[tuple[str, str, str, list[str]]]]] = [
            (label_before, rows_for(before, horizon=horizon)),
            (label_after, rows_for(after, horizon=horizon)),
        ]
    else:
        rows = rows_for(before, horizon=horizon)
        half = (len(rows) + 1) // 2
        # keep a dial's up/down pair together
        if half % 2:
            half += 1
        panels = [(None, rows[:half]), (None, rows[half:])]
    lines = [
        "% GENERATED by scripts/make_steering_table.py --side-by-side -- do not hand-edit.",
        f"% site: {before.get('site')} layer {before.get('layer_index')} tau {before.get('tau')}; horizon {horizon:g}h",
        "% bold: paired delta separates from zero in the clinically expected direction;",
        "% dagger: separates the wrong way; gray: no expectation declared for that dial.",
    ]
    for i, (label, rows) in enumerate(panels):
        lines.append("\\begin{minipage}[t]{0.5\\textwidth}\\centering")
        lines += _tabular(events, rows, label)
        lines.append("\\end{minipage}" + ("\\hfill" if i == 0 else ""))
    return "\n".join(lines) + "\n"


def render(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    *,
    horizon: float,
    label_before: str = "before steering training",
    label_after: str = "after steering training",
) -> str:
    """Render the tabular body with a header comment recording provenance."""
    events = [e for e in before["event_names"] if e in EVENT_LABELS]
    cols = "ll" + "l" + "r" * len(events)
    head = (
        " & ".join(["Dial", "", "$k_c$", *[EVENT_LABELS[e] for e in events]]) + " \\\\"
    )
    lines = [
        "% GENERATED by scripts/make_steering_table.py -- do not hand-edit.",
        f"% site: {before.get('site')} layer {before.get('layer_index')} tau {before.get('tau')}; horizon {horizon:g}h",
        "% bold: paired delta separates from zero in the clinically expected direction;",
        "% dagger: separates the wrong way; gray: no expectation declared for that dial.",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        head,
        "\\midrule",
    ]
    blocks: list[tuple[str, dict[str, Any]]] = [(label_before, before)]
    if after is not None:
        blocks.append((label_after, after))
    for label, payload in blocks:
        if after is not None:
            lines.append(
                f"\\multicolumn{{{3 + len(events)}}}{{l}}{{\\emph{{{label}}}}} \\\\"
            )
        for concept, arrow, respond, cells in rows_for(payload, horizon=horizon):
            lines.append(" & ".join([concept, arrow, respond, *cells]) + " \\\\")
        if after is not None and label == label_before:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the dial table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", type=Path, default=None)
    parser.add_argument("--horizon", type=float, default=24.0)
    parser.add_argument("--output-tex", required=True, type=Path)
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--label-before", default="before steering training")
    parser.add_argument("--label-after", default="after steering training")
    parser.add_argument(
        "--exclude-event", action="append", default=[], help="drop this outcome column"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text()) if args.after else None
    if after is not None and after["event_names"] != before["event_names"]:
        raise SystemExit(
            "before/after files score different event heads; refusing to align columns"
        )
    for payload in (before, after):
        if payload is not None and args.exclude_event:
            payload["event_names"] = [
                e for e in payload["event_names"] if e not in args.exclude_event
            ]
    renderer = render_side_by_side if args.side_by_side else render
    args.output_tex.write_text(
        renderer(
            before,
            after,
            horizon=args.horizon,
            label_before=args.label_before,
            label_after=args.label_after,
        )
    )
    logger.info(
        "wrote %s (%d dial rows)",
        args.output_tex,
        len(before["summaries"]) + (len(after["summaries"]) if after else 0),
    )


if __name__ == "__main__":
    main()
