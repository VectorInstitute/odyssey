"""How many of the GBM feature panel's signals resolve on one source.

The tuned tabular baseline reads :data:`odyssey.data.signal_panel.SIGNAL_PANEL`
(48 vitals and labs, keyed by LOINC). A signal resolves on a source when
that source's LOINC table in :mod:`odyssey.data.code_mapping` maps at
least one code prefix to the signal's LOINC; a signal with no prefix
never produces a feature there, so the baseline on that source runs with
a smaller panel. This script reports the resolved and unresolved names
for one source, which is the number the ML4H rebuttal needs for GEMINI
(review W3: "how many of the 48 signals does the GEMINI GBM actually
see?").

With ``--codes-parquet`` (a MEDS ``metadata/codes.parquet`` with a
``code`` column) the resolved signals are further split by whether any
code in that inventory starts with one of the signal's prefixes, so a
prefix that is in the table but never charted is reported too. Only
signal and prefix NAMES are written, never counts, so the output is safe
to export from a closed environment.

Usage::

    uv run python scripts/panel_coverage.py --source gemini \
        [--codes-parquet <meds_dir>/metadata/codes.parquet] \
        --output-json ~/runs/<run>/panel_coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from odyssey.data.code_mapping import prefixes_for_loinc
from odyssey.data.signal_panel import N_PANEL_SIGNALS, SIGNAL_PANEL


SOURCES: tuple[str, ...] = ("mimic_iv", "eicu", "gemini")

#: Top-level keys of the JSON this script writes. run.sh's export
#: whitelist for the panel-coverage step must list exactly these.
OUTPUT_KEYS: tuple[str, ...] = (
    "source",
    "n_panel_signals",
    "n_resolved",
    "unresolved",
    "resolved",
    "prefixes",
    "codes_inventory",
    "resolved_observed",
    "resolved_unobserved",
)


def panel_coverage(
    source: str, *, observed_codes: Iterable[str] | None = None
) -> dict[str, Any]:
    """Resolve every panel signal against ``source``'s LOINC table.

    Returns the names that resolve and the names that do not, plus the
    prefixes each resolved signal matches. When ``observed_codes`` is
    given (the distinct MEDS codes of that source), the resolved signals
    are further split into those with at least one observed code and
    those whose prefixes never occur. Codes carrying a ``::<bin>`` suffix
    are matched on their un-binned form, the rule
    :class:`~odyssey.data.signal_panel.SignalPanelResolver` uses.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    prefixes: dict[str, list[str]] = {}
    for name, loinc in SIGNAL_PANEL:
        hits = sorted(prefixes_for_loinc(loinc, source=source))
        if hits:
            resolved.append(name)
            prefixes[name] = hits
        else:
            unresolved.append(name)

    out: dict[str, Any] = {
        "source": source,
        "n_panel_signals": N_PANEL_SIGNALS,
        "n_resolved": len(resolved),
        "unresolved": unresolved,
        "resolved": resolved,
        "prefixes": prefixes,
        "codes_inventory": False,
        "resolved_observed": None,
        "resolved_unobserved": None,
    }
    if observed_codes is None:
        return out

    bases = {c.rsplit("::", 1)[0] if "::" in c else c for c in observed_codes}
    observed: list[str] = []
    unobserved: list[str] = []
    for name in resolved:
        if any(base.startswith(p) for p in prefixes[name] for base in bases):
            observed.append(name)
        else:
            unobserved.append(name)
    out["codes_inventory"] = True
    out["resolved_observed"] = observed
    out["resolved_unobserved"] = unobserved
    return out


def load_codes(path: str | Path) -> list[str]:
    """Distinct ``code`` strings from a MEDS ``codes.parquet`` (or any parquet)."""
    frame = pl.read_parquet(path, columns=["code"])
    return frame["code"].drop_nulls().unique().to_list()


def format_table(report: dict[str, Any]) -> str:
    """Render the report as a plain-text table for the log."""
    lines = [
        f"source: {report['source']}",
        f"panel signals: {report['n_panel_signals']}",
        f"resolved: {report['n_resolved']}",
        "",
        f"{'signal':<24}{'status':<12}prefixes",
    ]
    observed = report.get("resolved_observed")
    unobserved = set(report.get("resolved_unobserved") or [])
    for name, _ in SIGNAL_PANEL:
        if name in report["prefixes"]:
            status = "resolved"
            if observed is not None and name in unobserved:
                status = "no codes"
            lines.append(f"{name:<24}{status:<12}{', '.join(report['prefixes'][name])}")
        else:
            lines.append(f"{name:<24}{'unresolved':<12}")
    return "\n".join(lines)


def main() -> None:
    """Report panel coverage for one source and write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument(
        "--codes-parquet",
        default=None,
        help="MEDS metadata/codes.parquet; splits resolved signals by whether "
        "any charted code matches (names only are written)",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    codes = load_codes(args.codes_parquet) if args.codes_parquet else None
    report = panel_coverage(args.source, observed_codes=codes)
    print(format_table(report))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
