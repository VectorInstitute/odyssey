"""Summarize a pressure dose-response sweep as a LaTeX table.

A sign-agreement number says only that the hazard moved the right way. It
cannot tell a graded response from a threshold: a step function that is flat
at four rungs and jumps once is perfectly rank-monotonic, so a rank
correlation near 1 is equally consistent with the threshold it is supposed
to rule out. Two statistics are therefore reported per cell, and neither is
sufficient alone:

``monotone``
    Share of subjects whose hazard is non-decreasing across ALL consecutive
    rungs as the pressure falls. This is the direction claim, made once per
    subject over the whole sweep rather than once per edit.

``top step``
    Median over subjects of the largest single consecutive change divided by
    that subject's full range across the rungs. With ``k`` steps, a perfectly
    even ramp sits at ``1/k`` (0.25 for the five-rung sweeps here) and a pure
    threshold sits at 1. This is the shape claim, and it is the one that
    separates a dose-response from a trip point.

``flat``
    Share of subjects whose hazard does not move at all across the sweep.
    These are trivially non-decreasing, so they inflate ``monotone`` without
    being evidence for it; a high ``monotone`` beside a high ``flat`` means
    the model mostly ignored the edit, not that it responded to it.

Subjects are used only where every rung actually edited their record, so the
same cohort is compared across rungs rather than a different subset per rung
(the coverage-mismatch failure this project has hit before).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from statistics import median
from typing import Any


def null_reference(
    n_subjects: int, n_rungs: int, *, trials: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Return (monotone %, median top step) when the rungs carry no signal.

    Neither statistic is interpretable on its own scale. An even ramp puts
    the top step at ``1/k``, but what does a top step of 0.55 mean? The
    answer needs the other end of the scale, so this simulates the null the
    sweep is tested against: five exchangeable draws per subject, no
    ordering by rung at all. The result is the "no response" row the real
    cells are read against, and it is far from an even ramp in both
    statistics, which is why a cell can be well clear of noise and still be
    nowhere near a clean dose-response.
    """
    rng = random.Random(seed)
    monos: list[float] = []
    tops: list[float] = []
    for _ in range(trials):
        mono = 0
        per_trial: list[float] = []
        for _ in range(n_subjects):
            series = [rng.gauss(0.0, 1.0) for _ in range(n_rungs)]
            steps = [b - a for a, b in zip(series, series[1:])]
            if all(x >= 0 for x in steps):
                mono += 1
            per_trial.append(max(abs(x) for x in steps) / (max(series) - min(series)))
        monos.append(100.0 * mono / n_subjects)
        tops.append(median(per_trial))
    return median(monos), median(tops)


SWEEPS = {
    "sbp_noninvasive": ("SBP", "mmHg"),
    "map_noninvasive": ("MAP", "mmHg"),
}
EVENT_LABELS = {
    "death": "Death",
    "vasopressor_start": "Vasopressor",
    "icu_admission": "ICU adm.",
    "acute_kidney_injury": "AKI",
}
_SPEC = re.compile(r"^(?P<signal>[a-z_]+):set:(?P<value>[0-9.]+):(?P<window>[0-9.]+)$")


def sweep_arms(results: dict[str, Any], signal: str) -> list[tuple[float, str]]:
    """Return (value, arm_name) for one signal's set-mode arms, most severe last."""
    arms = []
    for name in results["edits"]:
        m = _SPEC.match(name)
        if m and m["signal"] == signal:
            arms.append((float(m["value"]), name))
    return sorted(arms, reverse=True)


def _risk_by_subject(
    results: dict[str, Any], arm: str, event: str, horizon: str
) -> dict[int, float]:
    """Counterfactual hazard per subject for one arm, edited subjects only."""
    out: dict[int, float] = {}
    for r in results["edits"][arm]["per_subject"]:
        if r["rows_edited"] <= 0:
            continue
        risk = r["counterfactual"]["event_risk"].get(event)
        if risk is not None and horizon in risk:
            out[int(r["subject_id"])] = float(risk[horizon])
    return out


def dose_response(
    results: dict[str, Any], signal: str, event: str, horizon: str
) -> dict[str, Any] | None:
    """Monotone share and median top-step share over subjects common to every rung."""
    arms = sweep_arms(results, signal)
    if len(arms) < 3:
        return None
    per_arm = [_risk_by_subject(results, arm, event, horizon) for _, arm in arms]
    common = set(per_arm[0])
    for d in per_arm[1:]:
        common &= set(d)
    if not common:
        return None
    monotone = 0
    flat = 0
    shares: list[float] = []
    for sid in common:
        series = [d[sid] for d in per_arm]
        steps = [b - a for a, b in zip(series, series[1:])]
        if all(s >= 0 for s in steps):
            monotone += 1
        spread = max(series) - min(series)
        # A subject whose hazard never moves is trivially non-decreasing and
        # would inflate the monotone share without being evidence of
        # anything, so it is counted and reported separately rather than
        # folded in silently. It has no shape either, so it cannot enter the
        # top-step median.
        if spread > 0:
            shares.append(max(abs(s) for s in steps) / spread)
        else:
            flat += 1
    return {
        "n": len(common),
        "rungs": [v for v, _ in arms],
        "monotone_pct": 100.0 * monotone / len(common),
        "flat_pct": 100.0 * flat / len(common),
        "median_top_step": median(shares) if shares else float("nan"),
        "even_ramp": 1.0 / len(steps),
    }


def render(results: dict[str, Any], horizon: str = "24h") -> str:
    """Return the LaTeX tabular, one block per swept signal."""
    lines = [
        "% GENERATED by scripts/make_dose_response_table.py -- do not hand-edit.",
        f"% Horizon {horizon}. monotone = share of subjects non-decreasing across",
        "% every rung; top step = median largest single step as a share of the",
        "% subject's full range (1/k = even ramp, 1 = pure threshold).",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Event & $n$ & monotone & flat & top step \\\\",
    ]
    for signal, (label, unit) in SWEEPS.items():
        arms = sweep_arms(results, signal)
        if not arms:
            continue
        rungs = ", ".join(f"{v:g}" for v, _ in arms)
        even = 1.0 / (len(arms) - 1)
        lines += [
            "\\midrule",
            f"\\multicolumn{{5}}{{l}}{{\\rlap{{\\emph{{{label}}} set to {rungs} {unit}"
            f" over 6 h; an even ramp gives {even:.2f}}}}} \\\\",
        ]
        first = next(
            (dose_response(results, signal, e, horizon) for e in EVENT_LABELS), None
        )
        if first is not None:
            null_mono, null_top = null_reference(first["n"], len(arms))
            lines.append(
                f"\\emph{{no response}} & {first['n']} &"
                f" {null_mono:.0f}\\% & -- & {null_top:.2f} \\\\"
            )
        for event, event_label in EVENT_LABELS.items():
            stats = dose_response(results, signal, event, horizon)
            if stats is None:
                continue
            lines.append(
                f"{event_label} & {stats['n']} &"
                f" {stats['monotone_pct']:.0f}\\% &"
                f" {stats['flat_pct']:.0f}\\% &"
                f" {stats['median_top_step']:.2f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the dose-response table from a counterfactual sweep JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", default="24h")
    args = parser.parse_args()

    results = json.loads(args.results_json.read_text())
    args.out.write_text(render(results, horizon=args.horizon))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
