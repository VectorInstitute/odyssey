"""Empirical filtering of candidate concepts (decision (d)).

See research_journal/04_concept_pipeline.html for the full decision.
Not every clinically-plausible candidate concept belongs in the
supervised set. Two checks, both reusable from Koh et al. (ICML 2020,
the original Concept Bottleneck Models paper)'s own pragmatic filtering
of noisy CUB-200 attributes: drop a concept with too little support
(fewer than a handful of subjects ever triggered it -- not enough signal
to learn from) or too little balance (one class, triggered or not,
covers almost every observed subject -- not enough contrast to be
useful supervision). The second is exactly what would have caught v1's
``tachypnea`` problem immediately: it triggered for 96.5% of subjects
with respiratory-rate data, a dominant-class fraction this filter is
built to catch.

This is only the half of decision (d) that doesn't need a trained model.
The other half -- a completeness/marginal-contribution check (does a
concept meaningfully help predict the downstream task, via a
ConceptSHAP-style probe) -- needs real labeled training data and a
trained forecasting model to evaluate against, neither of which exist
yet; see research_journal/04_concept_pipeline.html, "still open". This
module deliberately stops at prevalence/balance, which can run today
against nothing but the concept labels themselves.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from odyssey.data.concepts import AnyConceptDefinition


@dataclass(frozen=True)
class PrevalenceStats:
    """Prevalence/balance statistics for one concept, among observed subjects."""

    name: str
    n_observed: int
    """Subjects with at least one matching measurement for this concept at all."""

    n_triggered: int
    """Of those observed, how many the concept fired for."""

    prevalence: float
    """``n_triggered / n_observed``, or 0.0 if never observed."""

    passes_min_support: bool
    """At least ``min_support`` subjects triggered it."""

    passes_max_dominant_class: bool
    """Neither class (triggered/not, among observed) exceeds ``max_dominant_class``."""

    @property
    def passes(self) -> bool:
        """Both filters passed -- this concept is a reasonable supervision target."""
        return self.passes_min_support and self.passes_max_dominant_class


def compute_prevalence_stats(
    labels: pl.DataFrame,
    concepts: Sequence[AnyConceptDefinition],
    *,
    min_support: int = 10,
    max_dominant_class: float = 0.95,
) -> list[PrevalenceStats]:
    """Compute :class:`PrevalenceStats` for each concept.

    Consumes :func:`odyssey.data.concepts.label_concepts`'s output.

    ``min_support``: Koh et al.'s CUB-200 filter dropped attributes
    present in fewer than 10 classes; the same bar applied here to
    subjects, not classes -- fewer than 10 subjects ever triggering a
    concept is too little signal to learn a useful supervised
    probability from.

    ``max_dominant_class``: Koh et al.'s OAI knee-osteoarthritis filter
    dropped concepts where one class covered >= 95% of training data.
    Applied here to the observed (not all) subjects specifically, since
    an unobserved subject reveals nothing about the concept's true
    balance -- diluting the denominator with them would hide a genuinely
    imbalanced concept behind a large "never measured" population.
    """
    stats = []
    for concept in concepts:
        observed_col = f"{concept.name}_observed"
        observed = labels.filter(pl.col(observed_col) == 1)
        n_observed = observed.height
        n_triggered = int(observed[concept.name].sum()) if n_observed > 0 else 0
        prevalence = n_triggered / n_observed if n_observed > 0 else 0.0
        dominant_class = max(prevalence, 1.0 - prevalence) if n_observed > 0 else 1.0
        stats.append(
            PrevalenceStats(
                name=concept.name,
                n_observed=n_observed,
                n_triggered=n_triggered,
                prevalence=prevalence,
                passes_min_support=n_triggered >= min_support,
                passes_max_dominant_class=dominant_class < max_dominant_class,
            )
        )
    return stats


def filter_by_prevalence(stats: Sequence[PrevalenceStats]) -> list[str]:
    """Return the names of concepts that pass both prevalence/balance checks."""
    return [s.name for s in stats if s.passes]
