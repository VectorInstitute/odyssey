"""Fold numeric event values into the token itself.

:class:`~odyssey.data.vocabulary.Vocabulary` and
:func:`~odyssey.data.sequences.build_patient_sequence` tokenize on MEDS
``code`` identity alone -- "a heart-rate reading was taken" is one token
whether the value was 60 or 180. :func:`add_value_tokens` rewrites ``code``
to fold in a clinically meaningful bin (e.g. ``"LAB//220045//bpm::HIGH"``)
wherever a numeric value is present, so the two become distinct tokens.
Codes with no numeric value (a diagnosis, a procedure, a CT scan with no
attached result) pass through unchanged -- the event's occurrence is
already the full signal for those, exactly as before.

Two sources of bins, applied in priority order:

1. :data:`CANONICAL_CLINICAL_RANGES` -- hand-curated reference ranges,
   keyed by LOINC code and expanded to one source's concrete code
   prefixes by :func:`clinical_ranges_for_source` (the same
   canonical-then-expand pattern as
   :func:`odyssey.data.concepts.concepts_for_source`), for the same
   handful of vitals/labs the concept registry defines thresholds for
   (kept in sync deliberately: a lab token's bin should mean the same
   thing as the concept label it also supervises).
   ``test_value_binning.py`` asserts this consistency directly.
   :data:`CLINICAL_RANGES` is the MIMIC-IV expansion.
2. :class:`QuantileBinner` -- per-code quantile boundaries fit from the
   training corpus, for the much larger set of numeric-valued codes with no
   curated clinical range (most distinct LAB itemids). Must be fit once on
   train data only and reused for val/test, the same leakage discipline as
   :meth:`~odyssey.data.vocabulary.Vocabulary.build`.

Run :func:`add_value_tokens` on the event stream *before* building the
vocabulary and before :func:`~odyssey.data.sequences.build_patient_sequence`
-- both already tokenize on ``code`` as given, so no changes to either are
needed. Run :func:`~odyssey.data.concepts.label_concepts` on the original,
un-rewritten events (or the rewritten ones -- both work, since rewriting
only appends a suffix and ``code_prefix`` matching is a ``starts_with``).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl

from odyssey.data.code_mapping import prefixes_for_loinc, unit_for


# The canonical clinical ranges, keyed by LOINC code (and, for
# unit-split signals, by unit tag -- None means the range applies to
# every prefix of the LOINC). Keep in sync with
# odyssey/data/concepts.py's CANONICAL_CONCEPTS thresholds --
# test_value_binning.py asserts consistency: a lab token's bin should
# mean the same thing as the concept label it also supervises. Each
# value: (ascending (threshold, label-for-values-below) cut points,
# fallback label for values at or above every threshold).
_RangeSpec = Tuple[List[Tuple[float, str]], str]
CANONICAL_CLINICAL_RANGES: Dict[str, Dict[Optional[str], _RangeSpec]] = {
    # heart rate
    "8867-4": {None: ([(60.0, "LOW"), (100.0, "NORMAL")], "HIGH")},
    # SBP: non-invasive cuff and arterial line, same range
    "76534-7": {None: ([(90.0, "LOW"), (140.0, "NORMAL")], "HIGH")},
    "8480-6": {None: ([(90.0, "LOW"), (140.0, "NORMAL")], "HIGH")},
    # respiratory rate (only an upper rule exists)
    "9279-1": {None: ([(20.0, "NORMAL")], "HIGH")},
    # SpO2 (only a lower rule exists)
    "59408-5": {None: ([(92.0, "LOW")], "NORMAL")},
    # temperature: unit-split (see code_mapping._PREFIX_UNITS)
    "8310-5": {
        "F": ([(96.8, "LOW"), (100.4, "NORMAL")], "HIGH"),
        "C": ([(36.0, "LOW"), (38.0, "NORMAL")], "HIGH"),
    },
    # creatinine: NORMAL / HIGH (aki_stage_1) / CRITICAL (aki_stage_3)
    "2160-0": {None: ([(1.5, "NORMAL"), (4.0, "HIGH")], "CRITICAL")},
    # lactate
    "32693-4": {None: ([(2.0, "NORMAL")], "HIGH")},
}


def clinical_ranges_for_source(
    source: str = "mimic_iv",
) -> Tuple[Dict[str, List[Tuple[float, str]]], Dict[str, str]]:
    """Expand :data:`CANONICAL_CLINICAL_RANGES` to one source's prefixes.

    Returns ``(ranges, fallback_labels)``, both keyed by concrete MEDS
    code prefix. A LOINC with no prefix in ``source`` contributes
    nothing (its codes fall back to quantile bins); a unit-tagged range
    only reaches prefixes carrying that unit tag in
    :mod:`odyssey.data.code_mapping`.
    """
    ranges: Dict[str, List[Tuple[float, str]]] = {}
    fallbacks: Dict[str, str] = {}
    for loinc, by_unit in CANONICAL_CLINICAL_RANGES.items():
        for prefix in sorted(prefixes_for_loinc(loinc, source=source)):
            spec = by_unit.get(unit_for(prefix, source=source))
            if spec is None:
                continue
            cuts, fallback = spec
            ranges[prefix] = list(cuts)
            fallbacks[prefix] = fallback
    return ranges, fallbacks


# The MIMIC-IV expansion, kept as the module-level default exactly as
# before the canonical layer existed. Other sources pass
# ``source=`` to :func:`add_value_tokens`.
CLINICAL_RANGES, _FALLBACK_LABEL = clinical_ranges_for_source("mimic_iv")


def _clinical_label_expr(
    value_col: str,
    ranges: Dict[str, List[Tuple[float, str]]],
    fallbacks: Dict[str, str],
) -> pl.Expr:
    """Build one polars expression giving the clinical bin label, or null.

    Null wherever ``value_col`` is null, even for a matching code: a
    threshold comparison against a null value is null, which
    ``pl.when`` treats as False, so without an explicit null guard every
    threshold branch would fall through to the *fallback* label -- and a
    heart-rate event with a missing reading would silently tokenize as
    ``::HIGH``.
    """
    label_expr = pl.lit(None, dtype=pl.Utf8)
    for prefix, cuts in ranges.items():
        prefix_label = pl.lit(fallbacks[prefix])
        for threshold, label in reversed(cuts):
            prefix_label = (
                pl.when(pl.col(value_col) < threshold)
                .then(pl.lit(label))
                .otherwise(prefix_label)
            )
        label_expr = (
            pl.when(
                pl.col("code").str.starts_with(prefix) & pl.col(value_col).is_not_null()
            )
            .then(prefix_label)
            .otherwise(label_expr)
        )
    return label_expr


@dataclass
class QuantileBinner:
    """Per-code quantile boundaries for numeric values with no curated clinical range.

    Fit once on the training split only -- reusing it for val/test avoids
    leaking their value distributions into the bin boundaries, the same
    discipline :class:`~odyssey.data.vocabulary.Vocabulary` follows for
    token frequencies.
    """

    boundaries: Dict[str, List[float]]
    """code -> ascending list of ``n_bins - 1`` quantile cut points."""
    n_bins: int

    @classmethod
    def fit(
        cls,
        events: pl.DataFrame,
        *,
        n_bins: int = 5,
        min_count: int = 100,
        code_col: str = "code",
        value_col: str = "numeric_value",
    ) -> "QuantileBinner":
        """Compute per-code quantile boundaries from numeric-valued events.

        Codes with fewer than ``min_count`` numeric observations are
        skipped -- too little data for a stable per-code estimate; they
        fall back to the code-identity-only token (see :func:`add_value_tokens`).
        """
        quantiles = [i / n_bins for i in range(1, n_bins)]
        # Project to the two needed columns BEFORE filtering: the filter
        # materializes a copy, and at full-extraction scale (706M rows) a
        # five-column copy is tens of GB where a two-column one is not.
        numeric = events.select(code_col, value_col).filter(
            pl.col(value_col).is_not_null()
        )
        counts = numeric.group_by(code_col).agg(pl.len().alias("n"))
        eligible = counts.filter(pl.col("n") >= min_count)[code_col].to_list()
        if not eligible:
            return cls(boundaries={}, n_bins=n_bins)

        qcols = [f"_q{i}" for i in range(len(quantiles))]
        stats = (
            numeric.filter(pl.col(code_col).is_in(eligible))
            .group_by(code_col)
            .agg(
                [
                    pl.col(value_col).quantile(q).alias(qcol)
                    for qcol, q in zip(qcols, quantiles)
                ]
            )
        )
        boundaries = {
            row[code_col]: sorted({row[c] for c in qcols})
            for row in stats.iter_rows(named=True)
        }
        return cls(boundaries=boundaries, n_bins=n_bins)

    def apply(
        self,
        events: pl.DataFrame,
        *,
        code_col: str = "code",
        value_col: str = "numeric_value",
    ) -> pl.Series:
        """Return a ``Utf8`` Series of bin labels.

        Null where no boundary exists for the code, and null where the
        value itself is null (a null value compares as below every cut
        point, so without an explicit guard it would silently land in the
        lowest bin, ``Q1``, instead of staying unbinned).
        """
        if not self.boundaries:
            return pl.Series([None] * events.height, dtype=pl.Utf8)

        rows = []
        for code, cuts in self.boundaries.items():
            row: Dict[str, object] = {"_code": code, "_found": True}
            for i in range(self.n_bins - 1):
                row[f"_cut{i}"] = cuts[i] if i < len(cuts) else None
            rows.append(row)
        bframe = pl.DataFrame(rows)

        joined = (
            events.select([code_col, value_col])
            .with_row_index("_row")
            .join(bframe, left_on=code_col, right_on="_code", how="left")
        )
        cut_cols = [f"_cut{i}" for i in range(self.n_bins - 1)]
        bin_index = pl.lit(0, dtype=pl.Int32)
        for c in cut_cols:
            bin_index = bin_index + (
                (pl.col(value_col) >= pl.col(c)).fill_null(False).cast(pl.Int32)
            )
        joined = joined.with_columns(
            pl.when(pl.col("_found") & pl.col(value_col).is_not_null())
            .then(pl.lit("Q") + (bin_index + 1).cast(pl.Utf8))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("_bin")
        )
        return joined.sort("_row")["_bin"]

    def save(self, path: Union[str, Path]) -> None:
        """Save as JSON."""
        Path(path).write_text(
            json.dumps({"n_bins": self.n_bins, "boundaries": self.boundaries})
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QuantileBinner":
        """Load from JSON written by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        return cls(boundaries=data["boundaries"], n_bins=data["n_bins"])


def add_value_tokens(
    events: pl.DataFrame,
    quantile_binner: Optional[QuantileBinner] = None,
    *,
    code_col: str = "code",
    value_col: str = "numeric_value",
    source: str = "mimic_iv",
) -> pl.DataFrame:
    """Rewrite ``code`` to fold in a value bin wherever a numeric value exists.

    A no-op for any row where ``value_col`` is null (procedures, diagnoses,
    admin events, or a numeric-valued code with a missing reading) -- the
    event's occurrence is the full signal there, unchanged from today.

    ``source`` picks which institution's code prefixes the curated
    clinical ranges apply to (see :func:`clinical_ranges_for_source`);
    everything else is source-independent.
    """
    if value_col not in events.columns:
        return events

    ranges, fallbacks = clinical_ranges_for_source(source)
    events = events.with_columns(
        _clinical_label_expr(value_col, ranges, fallbacks).alias("_bin_label")
    )

    if quantile_binner is not None:
        qbins = quantile_binner.apply(events, code_col=code_col, value_col=value_col)
        events = events.with_columns(
            pl.when(pl.col("_bin_label").is_null())
            .then(qbins)
            .otherwise(pl.col("_bin_label"))
            .alias("_bin_label")
        )

    return events.with_columns(
        pl.when(pl.col("_bin_label").is_not_null())
        .then(pl.col(code_col) + "::" + pl.col("_bin_label"))
        .otherwise(pl.col(code_col))
        .alias(code_col)
    ).drop("_bin_label")
