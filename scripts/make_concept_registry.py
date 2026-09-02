"""Generate the paper's concept-registry table.

One table, not two: the previous version was split across "1 of 2" and
"2 of 2" purely because the definitions were long enough to overrun a
page, which also made the second half look like a separate table when it
is a continuation. Terse definitions fit all 29 concepts on one page.

Three resolution columns, not one. The registry's portability across
sources is a result the paper claims, so the table should show it: which
concepts survive each source's code mapping. Those columns are read from
the code and from the G2 audit rather than transcribed, because they are
the part most likely to drift:

* MIMIC-IV and eICU from ``odyssey.data.concepts.concepts_for_source``,
  which drops a concept when the source lacks its ingredients;
* GEMINI from ``scripts/gemini/out/concept_audit.json`` (plan G2), the
  concept-resolution audit against the GEMINI code inventory.

The definitions here are deliberately one-liners. The rules themselves
are the code; a registry table's job is to let a reader see the whole
vocabulary at once, and the rationale for individual thresholds belongs
in the module docstrings where it can be maintained.

Usage::

    uv run python scripts/make_concept_registry.py \\
        --audit scripts/gemini/out/concept_audit.json \\
        --output-tex paper/ml4h/tables/concept_registry.tex
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from odyssey.data.concepts import concepts_for_source


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("concept_registry")

# Display name and a one-line rule per concept, keyed by the code's own
# concept name so a rename shows up as a missing entry rather than a
# silently stale row.
DEFINITIONS: dict[str, tuple[str, str]] = {
    "tachycardia": ("tachycardia", r"Heart rate $>$ 100 bpm"),
    "bradycardia": ("bradycardia", r"Heart rate $<$ 60 bpm"),
    "hypotension": ("hypotension", r"Systolic blood pressure $<$ 90 mmHg"),
    "hypertension": ("hypertension", r"Systolic blood pressure $>$ 140 mmHg"),
    "hypoxia": ("hypoxia", r"SpO$_2$ $<$ 92\%"),
    "fever": ("fever", r"Temperature $>$ 38.0\,$^\circ$C"),
    "hypothermia": ("hypothermia", r"Temperature $<$ 36.0\,$^\circ$C"),
    "elevated_lactate": ("elevated lactate", r"Serum lactate $>$ 2.0 mmol/L"),
    "sustained_tachypnea": (
        "sustained tachypnea",
        r"Respiratory rate $>$ 20/min, recurring at least 1 hour apart",
    ),
    "acute_kidney_injury": (
        "acute kidney injury",
        r"KDIGO stage 1: creatinine $+0.3$ mg/dL in 48h, $\times 1.5$ in 7d, "
        r"or urine $<$ 0.5 mL/kg/h for 6h",
    ),
    "aki_stage_2": (
        "aki stage 2",
        r"KDIGO stage 2: creatinine $\times 2.0$ in 7d, or urine $<$ 0.5 "
        r"mL/kg/h for 12h",
    ),
    "aki_stage_3": (
        "aki stage 3",
        r"KDIGO stage 3: creatinine $\times 3.0$ in 7d or $\ge$ 4.0 mg/dL, "
        r"renal-replacement therapy, urine $<$ 0.3 mL/kg/h for 24h, or anuria "
        r"for 12h",
    ),
    "sirs": (
        "sirs",
        r"$\ge$ 2 of abnormal temperature, HR $>$ 90, RR $>$ 20, abnormal WBC",
    ),
    "qsofa": (
        "qsofa",
        r"$\ge$ 2 of RR $\ge$ 22, systolic BP $\le$ 100 mmHg, GCS $<$ 15",
    ),
    "on_vasopressors": (
        "on vasopressors",
        r"Any vasopressor given (norepinephrine, epinephrine, vasopressin, "
        r"phenylephrine, dopamine, angiotensin II)",
    ),
    "hypoxemic_respiratory_failure": (
        "hypoxemic respiratory failure",
        r"PaO$_2$/FiO$_2$ $<$ 300 mmHg, each gas paired with an FiO$_2$ "
        r"within 4h",
    ),
    "oliguria": (
        "oliguria",
        r"Under 500 mL of urine over a trailing 24h, scored only once 24h of "
        r"record exists",
    ),
    "sepsis3": (
        "sepsis3",
        r"Suspected infection (culture and antibiotic within 72h/24h of each "
        r"other) with SOFA $\ge$ 2",
    ),
    "hyperkalemia": ("hyperkalemia", r"Potassium $>$ 5.5 mEq/L"),
    "hypokalemia": ("hypokalemia", r"Potassium $<$ 3.0 mEq/L"),
    "hyponatremia": ("hyponatremia", r"Sodium $<$ 130 mEq/L"),
    "hypernatremia": ("hypernatremia", r"Sodium $>$ 150 mEq/L"),
    "hypoglycemia": ("hypoglycemia", r"Glucose $<$ 70 mg/dL"),
    "hyperglycemia": ("hyperglycemia", r"Glucose $>$ 250 mg/dL"),
    "anemia": ("anemia", r"Hemoglobin $<$ 7 g/dL"),
    "thrombocytopenia": (
        "thrombocytopenia",
        r"Platelet count $<$ 100 $\times 10^{9}$/L",
    ),
    "coagulopathy": ("coagulopathy", r"INR $>$ 1.5"),
    "metabolic_acidosis": (
        "metabolic acidosis",
        r"Bicarbonate $<$ 18 mEq/L, or arterial/venous pH $<$ 7.3",
    ),
    "sustained_hypotension_map": (
        "sustained hypotension (MAP)",
        r"Mean arterial pressure $<$ 65 mmHg, recurring",
    ),
}

TICK = r"\checkmark"
DASH = r"--"


def resolution(task_set: str, audit: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Concept order plus the set of concepts resolving on each source."""
    mimic = [c.name for c in concepts_for_source("mimic_iv", task_set=task_set)]
    eicu = {c.name for c in concepts_for_source("eicu", task_set=task_set)}
    audited = json.loads(audit.read_text())["concept_resolution"]
    gemini = {name for name, ok in audited.items() if ok}
    missing = set(audited) ^ set(mimic)
    if missing:
        raise SystemExit(
            f"the G2 audit and task set {task_set} disagree on which concepts "
            f"exist: {sorted(missing)}. One of them is stale; refusing to "
            "build a portability table from a mismatched pair"
        )
    return mimic, {"mimic": set(mimic), "eicu": eicu, "gemini": gemini}


def build(order: list[str], resolves: dict[str, set[str]]) -> list[str]:
    """Build the table rows, one concept each."""
    rows = []
    for name in order:
        if name not in DEFINITIONS:
            raise SystemExit(
                f"no definition for concept {name!r}; add it to DEFINITIONS "
                "rather than letting the table quietly omit a concept"
            )
        display, rule = DEFINITIONS[name]
        marks = " & ".join(
            TICK if name in resolves[src] else DASH
            for src in ("mimic", "eicu", "gemini")
        )
        rows.append(f"{display} & {rule} & {marks} \\\\")
    return rows


def main() -> None:
    """Write the single-table concept registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", default="v3")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    args = parser.parse_args()

    order, resolves = resolution(args.task_set, args.audit)
    rows = build(order, resolves)
    counts = {k: len(v) for k, v in resolves.items()}

    header = [
        "% GENERATED by scripts/make_concept_registry.py -- do not hand-edit.",
        f"% task set: {args.task_set}; GEMINI column from {args.audit} (plan G2)",
        f"% resolving: MIMIC-IV {counts['mimic']}, eICU {counts['eicu']}, "
        f"GEMINI {counts['gemini']}, of {len(order)}",
    ]
    table = [
        # p{} for the definition: a plain l column takes the longest rule
        # as its natural width and ran 255pt past the page.
        r"\begin{tabular}{@{}l p{0.49\textwidth} ccc@{}}",
        r"\toprule",
        r"Concept & Definition & MIMIC & eICU & GEMINI \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(header + table) + "\n")
    logger.info(
        "wrote %s (%d concepts; MIMIC %d, eICU %d, GEMINI %d)",
        args.output_tex,
        len(order),
        counts["mimic"],
        counts["eicu"],
        counts["gemini"],
    )


if __name__ == "__main__":
    main()
