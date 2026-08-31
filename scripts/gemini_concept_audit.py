"""G2: GEMINI concept-resolution and binning-portability audit (local, CPU).

Runs on a normal checkout against ``scripts/gemini/out/codes_inventory.json``
(the suppressed code inventory ``scripts/gemini/run.sh export-codes`` commits
back from the node) -- no GEMINI access, no GPU, no patient data: every input
is already cell-suppressed vocabulary metadata.

Three questions, matching docs/experiment_plan.md row G2:

1. **Concept resolution** (paper Sec 3 portability RESULT): which of the
   full MIMIC concept set resolve on GEMINI through the LOINC layer
   (:func:`odyssey.data.concepts.concepts_for_source`), and which drop for
   want of a code mapping.
2. **Mapping-vs-reality check**: every ``GEMINI_TO_LOINC`` prefix should
   match at least one code in the real inventory (a prefix with zero
   matches means the mapping was built against a different cut than the
   extraction produced), and the distinct unit variants seen per prefix
   are listed so unit assumptions (SI creatinine in umol/L especially)
   are verified against data rather than trusted.
3. **Binning portability** (reviewer E4): the token-weighted fraction of
   GEMINI LAB// and VITALS// events whose codes fall in curated
   shared-threshold clinical bins
   (:func:`odyssey.data.value_binning.clinical_ranges_for_source`) versus
   source-fit quantile bins. Curated bins are the semantically portable
   subset; quantile bins are re-fit per source and carry no cross-source
   meaning. Suppressed counts make this a bounded estimate: ``"<1000"``
   entries contribute ``[0, 999]``, so the fraction is reported as a
   ``[lower, upper]`` interval, not a point.

Also reports the top-N unmapped codes by count -- the concrete candidate
list for extending ``GEMINI_TO_LOINC`` (and with it, concept resolution)
in a later pass.

Usage (after the export-codes output has been copied back to origin):

    uv run python scripts/gemini_concept_audit.py \
        --output-json scripts/gemini/out/concept_audit.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from odyssey.data.code_mapping import GEMINI_TO_LOINC
from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import clinical_ranges_for_source


logger = logging.getLogger(__name__)

DEFAULT_INVENTORY = (
    Path(__file__).resolve().parent / "gemini" / "out" / "codes_inventory.json"
)

#: Families whose events carry numeric values and therefore get bin tokens;
#: the binning-portability denominator (matches the curated-range surface:
#: every CANONICAL_CLINICAL_RANGES prefix is a LAB// or VITALS// code).
VALUE_FAMILIES = ("LAB//", "VITALS//")


def count_bounds(suppressed: str) -> tuple[int, int]:
    """(lower, upper) bounds for one suppressed count string.

    Numeric strings are already rounded to the nearest 1000 by
    ``export_codes`` -- treated as exact here (the rounding error is
    symmetric and immaterial at the aggregate level this audit reports);
    ``"<N"`` contributes ``[0, N-1]``.
    """
    if suppressed.startswith("<"):
        return 0, int(suppressed[1:]) - 1
    n = int(suppressed)
    return n, n


def audit(inventory: dict[str, str], *, top_unmapped: int = 40) -> dict[str, Any]:
    """Run the full audit against one suppressed code inventory."""
    # -- 1. concept resolution --------------------------------------------
    mimic_names = [c.name for c in concepts_for_source("mimic_iv", task_set="v3")]
    gemini_names = {c.name for c in concepts_for_source("gemini", task_set="v3")}
    resolution = {name: name in gemini_names for name in mimic_names}

    # -- 2. mapping vs reality --------------------------------------------
    ranges, _fallbacks = clinical_ranges_for_source("gemini")
    mapping_rows: dict[str, Any] = {}
    for prefix, loinc in sorted(GEMINI_TO_LOINC.items()):
        matches = {c: s for c, s in inventory.items() if c.startswith(prefix)}
        lo = sum(count_bounds(s)[0] for s in matches.values())
        hi = sum(count_bounds(s)[1] for s in matches.values())
        mapping_rows[prefix] = {
            "loinc": loinc,
            "n_codes": len(matches),
            "token_count_bounds": [lo, hi],
            "unit_variants": sorted({c[len(prefix) :] or "(none)" for c in matches}),
            "curated_bins": prefix in ranges,
        }
    unmatched_prefixes = [p for p, row in mapping_rows.items() if row["n_codes"] == 0]

    # -- 3. binning portability -------------------------------------------
    portability: dict[str, Any] = {}
    curated_prefixes = tuple(ranges)
    for family in VALUE_FAMILIES:
        fam_lo = fam_hi = cur_lo = cur_hi = 0
        n_codes = n_curated_codes = 0
        for code, s in inventory.items():
            if not code.startswith(family):
                continue
            lo, hi = count_bounds(s)
            fam_lo += lo
            fam_hi += hi
            n_codes += 1
            if code.startswith(curated_prefixes):
                cur_lo += lo
                cur_hi += hi
                n_curated_codes += 1
        portability[family.rstrip("/")] = {
            "n_codes": n_codes,
            "n_codes_curated": n_curated_codes,
            "token_count_bounds": [fam_lo, fam_hi],
            "curated_token_count_bounds": [cur_lo, cur_hi],
            # Conservative interval: fewest curated over most total, and
            # vice versa (guarding the zero-denominator corners).
            "curated_fraction_bounds": [
                cur_lo / fam_hi if fam_hi else 0.0,
                (cur_hi / fam_lo) if fam_lo else (1.0 if cur_hi else 0.0),
            ],
        }

    # -- extension candidates ---------------------------------------------
    mapped_prefixes = tuple(GEMINI_TO_LOINC)
    unmapped = [
        (code, count_bounds(s)[0], s)
        for code, s in inventory.items()
        if code.startswith(VALUE_FAMILIES) and not code.startswith(mapped_prefixes)
    ]
    unmapped.sort(key=lambda t: (-t[1], t[0]))
    candidates = [
        {"code": code, "count": s} for code, _lo, s in unmapped[:top_unmapped]
    ]

    return {
        "n_inventory_codes": len(inventory),
        "concept_resolution": resolution,
        "n_concepts_mimic_v3": len(mimic_names),
        "n_concepts_resolving": sum(resolution.values()),
        "mapping": mapping_rows,
        "unmatched_mapping_prefixes": unmatched_prefixes,
        "binning_portability": portability,
        "top_unmapped_value_codes": candidates,
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes-inventory",
        default=str(DEFAULT_INVENTORY),
        help="codes_inventory.json from run.sh export-codes (copied back).",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-unmapped", type=int, default=40)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow clobbering an existing --output-json (append-only default).",
    )
    args = parser.parse_args()

    out = Path(args.output_json)
    if out.exists() and not args.overwrite:
        sys.exit(f"refusing to overwrite existing {out} (pass --overwrite)")
    inventory_path = Path(args.codes_inventory)
    if not inventory_path.is_file():
        sys.exit(
            f"{inventory_path} not found -- run.sh export-codes output has "
            "not been copied back from the gemini remote yet"
        )
    inventory = json.loads(inventory_path.read_text())

    result = audit(inventory, top_unmapped=args.top_unmapped)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    resolving = [n for n, ok in result["concept_resolution"].items() if ok]
    dropped = [n for n, ok in result["concept_resolution"].items() if not ok]
    logger.info(
        "[gemini_concept_audit] %d/%d concepts resolve on GEMINI; dropped: %s",
        len(resolving),
        result["n_concepts_mimic_v3"],
        ", ".join(dropped) or "(none)",
    )
    for family, row in result["binning_portability"].items():
        lo, hi = row["curated_fraction_bounds"]
        logger.info(
            "[gemini_concept_audit] %s: curated-bin token fraction in "
            "[%.3f, %.3f] (%d of %d codes curated)",
            family,
            lo,
            hi,
            row["n_codes_curated"],
            row["n_codes"],
        )
    if result["unmatched_mapping_prefixes"]:
        logger.warning(
            "[gemini_concept_audit] mapping prefixes with NO inventory match "
            "(mapping/extraction drift): %s",
            ", ".join(result["unmatched_mapping_prefixes"]),
        )
    logger.info("[gemini_concept_audit] wrote %s", out)


if __name__ == "__main__":
    main()
