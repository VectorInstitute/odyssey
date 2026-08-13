"""Rule-derived clinical concept labels from MEDS events.

A small, editable registry of clinically-recognizable vital-sign and lab
abnormalities, derived directly from a MEDS event table to supervise
:class:`odyssey.models.concept_bottleneck.ConceptBottleneck`. These are
deliberately simple single-threshold rules for v1, not full KDIGO/SOFA-grade
clinical criteria — see each definition's ``description`` for its specific
simplification. Itemids are MIMIC-IV's, matching the itemids used by the
widely-cited `MIT-LCP/mimic-code <https://github.com/MIT-LCP/mimic-code>`_
concept queries; the MEDS ETL keys chartevents on ``LAB//{itemid}//`` and
labevents on ``LAB//RESULT//{itemid}//``.

Labels are currently aggregated over each subject's full extracted window
(did this ever happen), not per fixed time-bucket — extend once the patient
sequence windowing scheme is finalized.
"""

from dataclasses import dataclass
from typing import List, Literal

import polars as pl


Direction = Literal["above", "below"]


@dataclass(frozen=True)
class ConceptRule:
    """One (MEDS code prefix, threshold) test contributing to a concept."""

    code_prefix: str
    """Matched via ``code.str.starts_with(code_prefix)``, e.g. ``"LAB//220045//"``."""

    threshold: float
    direction: Direction


@dataclass(frozen=True)
class ConceptDefinition:
    """A clinical concept, derived by OR-ing one or more :class:`ConceptRule`."""

    name: str
    rules: List[ConceptRule]
    description: str


CONCEPTS: List[ConceptDefinition] = [
    ConceptDefinition(
        "tachycardia",
        [ConceptRule("LAB//220045//", 100.0, "above")],
        "Heart rate > 100 bpm.",
    ),
    ConceptDefinition(
        "bradycardia",
        [ConceptRule("LAB//220045//", 60.0, "below")],
        "Heart rate < 60 bpm.",
    ),
    ConceptDefinition(
        "hypotension",
        [
            ConceptRule("LAB//220179//", 90.0, "below"),  # non-invasive systolic BP
            ConceptRule("LAB//220050//", 90.0, "below"),  # arterial systolic BP
        ],
        "Systolic blood pressure < 90 mmHg.",
    ),
    ConceptDefinition(
        "hypertension",
        [
            ConceptRule("LAB//220179//", 140.0, "above"),
            ConceptRule("LAB//220050//", 140.0, "above"),
        ],
        "Systolic blood pressure > 140 mmHg.",
    ),
    ConceptDefinition(
        "tachypnea",
        [ConceptRule("LAB//220210//", 20.0, "above")],
        "Respiratory rate > 20 breaths/min.",
    ),
    ConceptDefinition(
        "hypoxia",
        [ConceptRule("LAB//220277//", 92.0, "below")],
        "SpO2 < 92%.",
    ),
    ConceptDefinition(
        "fever",
        [
            ConceptRule("LAB//223761//", 100.4, "above"),  # Fahrenheit
            ConceptRule("LAB//223762//", 38.0, "above"),  # Celsius
        ],
        "Temperature > 100.4F / 38.0C.",
    ),
    ConceptDefinition(
        "hypothermia",
        [
            ConceptRule("LAB//223761//", 96.8, "below"),
            ConceptRule("LAB//223762//", 36.0, "below"),
        ],
        "Temperature < 96.8F / 36.0C.",
    ),
    ConceptDefinition(
        "acute_kidney_injury",
        [ConceptRule("LAB//RESULT//50912//", 1.5, "above")],
        "Serum creatinine > 1.5 mg/dL. Simplified single-value proxy — "
        "not KDIGO delta-from-baseline criteria.",
    ),
    ConceptDefinition(
        "elevated_lactate",
        [ConceptRule("LAB//RESULT//50813//", 2.0, "above")],
        "Serum lactate > 2.0 mmol/L.",
    ),
]


def label_concepts(
    events: pl.DataFrame,
    concepts: List[ConceptDefinition] = CONCEPTS,
    subject_id_col: str = "subject_id",
    code_col: str = "code",
    value_col: str = "numeric_value",
) -> pl.DataFrame:
    """Derive per-subject concept labels and an observed-mask from MEDS events.

    Every subject in ``events`` gets a ``{name}`` binary label (1 if any
    matching measurement crossed the threshold) and a ``{name}_observed``
    mask (1 if at least one matching measurement exists at all, vs. that
    lab/vital simply never having been drawn for that subject — the
    distinction :func:`odyssey.models.concept_bottleneck.concept_loss`'s
    masking is meant to exclude from supervision).
    """
    subject_ids = events.select(subject_id_col).unique()
    out = subject_ids

    for concept in concepts:
        observed_frames = []
        triggered_frames = []
        for rule in concept.rules:
            matched = events.filter(
                pl.col(code_col).str.starts_with(rule.code_prefix)
                & pl.col(value_col).is_not_null()
            )
            observed_frames.append(matched.select(subject_id_col))
            comparison = (
                pl.col(value_col) > rule.threshold
                if rule.direction == "above"
                else pl.col(value_col) < rule.threshold
            )
            triggered_frames.append(matched.filter(comparison).select(subject_id_col))

        observed_ids = pl.concat(observed_frames).unique()[subject_id_col].to_list()
        triggered_ids = pl.concat(triggered_frames).unique()[subject_id_col].to_list()

        out = out.with_columns(
            pl.col(subject_id_col)
            .is_in(triggered_ids)
            .cast(pl.Int8)
            .alias(concept.name),
            pl.col(subject_id_col)
            .is_in(observed_ids)
            .cast(pl.Int8)
            .alias(f"{concept.name}_observed"),
        )

    return out
