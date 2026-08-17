"""Collapse dose/form/route variants of medication codes to ingredient level.

Medication codes embed free-text sig lines: ``MEDICATION//Acetaminophen 325
mg PO TABS``, ``MEDICATION//Acetaminophen 650 mg RE SUPP``, and (eICU)
``MEDICATION//STARTED//1000 ML FLEX CONT : SODIUM CHLORIDE 0.9 % IV SOLN``
are all distinct tokens for two ingredients. That fragmentation splits one
drug's statistics across dozens of sparse tokens, starves the vocabulary
(each variant competes for a slot; the losers become ``[UNK]``), and is a
prime suspect for medication forecasting's weak top-1 accuracy.

:func:`normalize_medication_codes` rewrites the ``code`` column so every
medication event carries ``MEDICATION//<action?>//<ingredient>`` with the
ingredient lowercased and stripped of dose, concentration, container, form,
and route. Everything non-medication passes through untouched. Run it on
the event stream *before* vocabulary building and sequence tokenization,
alongside :func:`~odyssey.data.value_binning.add_value_tokens` (order
between the two is irrelevant; they touch disjoint code families).

Heuristics, deliberately conservative:

- Text after the last ``:`` wins (strips eICU's container prefix,
  ``1000 ML FLEX CONT :``).
- The name is cut at the first space-preceded digit (`` 325 mg...``),
  which preserves digits inside a name token (``B12``, ``D5W``).
- Trailing route/form vocabulary (PO/IV/TABS/SOLN/...) is stripped.
- If stripping would leave nothing, the original name is kept lowercased:
  never produce an empty ingredient.

Concept rules that match drug names (``on_vasopressors``) are unaffected:
the ingredient substring they search for survives normalization.

Two eICU-specific rules ride along, both keyed on ``source="eicu"``:

- eICU medication codes carry a trailing HICL segment
  (``MEDICATION//STARTED//{drugname}//{hicl}``, spec v2). 36% of eICU
  medication rows have no drug name at all but 94% of those carry a HICL,
  so the HICL is resolved FIRST through the empirical dictionary shipped
  as ``odyssey/data/resources/eicu_hicl_ingredients.csv`` (built by
  ``scripts/build_eicu_hicl_lookup.py``: majority ingredient per HICL over
  the rows that carry both). Resolving HICL-first also merges brand/salt
  spellings of one ingredient that the name heuristic would keep apart.
- eICU infusions are ``INFUSION_DRUG//{drugname}`` with the rate unit in
  parentheses (``Norepinephrine (mcg/min)``); the unit suffix is dropped
  and the remainder normalized like a medication name.
"""

import csv
import io
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Mapping, Optional

import polars as pl


MEDICATION_FAMILIES = ("MEDICATION",)
INFUSION_FAMILIES = ("INFUSION_DRUG",)

# HICL sequence number (as the string it appears as in a code) -> ingredient.
HiclIngredients = Mapping[str, str]

_HICL_RESOURCE = "eicu_hicl_ingredients.csv"

# One trailing route/form/schedule token. Applied repeatedly, so
# "PO TABS" strips in two passes. Word-bounded to avoid eating name parts.
_FORM_WORDS = (
    "PO|IV|IVPB|IM|SC|SUBCUT|SL|PR|TP|NEB|INH|INHL|HFA|ORAL|"
    "TABS?|CAPS?|TBEC|CPDR|TBDP|SOLN|SOLR|SOL|SUSP|SYP|SYRUP|ELIX|"
    "INJ|CREA|CREAM|OINT|GEL|PATCH|SPRAY|DROPS?|SUPP|PWDR|KIT|"
    "ER|CR|SR|XL|XR|DR|UD|UDCUP|VIAL|BAG|PREMIX|CONT|FLEX|DESI|"
    "RE|MISC|MEQ|INTRAVENOUS|PIGGYBACK"
)
_TRAILING_FORM_RE = re.compile(rf"\s+(?:{_FORM_WORDS})\s*$", re.IGNORECASE)
_CONTAINER_RE = re.compile(r"^.*:\s*")
_DOSE_RE = re.compile(r"\s\d.*$")
_WS_RE = re.compile(r"\s+")


def _normalize_name(name: str) -> str:
    """Reduce one free-text drug string to its lowercased ingredient."""
    stripped = _CONTAINER_RE.sub("", name)
    stripped = _DOSE_RE.sub("", stripped)
    for _ in range(6):
        shorter = _TRAILING_FORM_RE.sub("", stripped)
        if shorter == stripped:
            break
        stripped = shorter
    stripped = _WS_RE.sub(" ", stripped).strip(" -.,")
    if not stripped:
        stripped = _WS_RE.sub(" ", name).strip()
    return stripped.lower()


_ACTION_SEGMENTS = {"START", "STOP", "STARTED", "STOPPED"}
_ID_SEGMENT_RE = re.compile(r"\d+|unk", re.IGNORECASE)
_INFUSION_UNIT_RE = re.compile(r"\s*\([^()]*\)\s*$")


@lru_cache(maxsize=1)
def load_eicu_hicl_ingredients() -> Dict[str, str]:
    """Load the shipped eICU HICL -> ingredient dictionary (see module docstring)."""
    text = (
        resources.files("odyssey.data.resources")
        .joinpath(_HICL_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return {row["hicl"]: row["ingredient"] for row in csv.DictReader(io.StringIO(text))}


def hicl_ingredients_for_source(source: Optional[str]) -> Optional[HiclIngredients]:
    """Return the HICL dictionary for a data source: eICU has one, nothing else does."""
    return load_eicu_hicl_ingredients() if source == "eicu" else None


def _normalize_infusion_code(parts: List[str]) -> str:
    """Strip the rate unit and normalize the infusion drug name.

    ``INFUSION_DRUG//Norepinephrine (mcg/min)`` becomes
    ``INFUSION_DRUG//norepinephrine``.
    """
    if len(parts) < 2 or not parts[1]:
        return "//".join(parts)
    name = _INFUSION_UNIT_RE.sub("", parts[1]) or parts[1]
    return "//".join([parts[0], _normalize_name(name), *parts[2:]])


def normalize_medication_code(
    code: str, hicl_ingredients: Optional[HiclIngredients] = None
) -> str:
    """Normalize one medication code; the single source of truth for the rule.

    Real MIMIC-IV emar codes are ``MEDICATION//{drug}//{event_txt}//{ndc}``
    (the trailing NDC packaging code splits one drug administration into
    thousands of tokens) and pharmacy codes are
    ``MEDICATION//START|STOP//{drug}//{gsn}``; eICU uses
    ``MEDICATION//STARTED|STOPPED//{drug}//{hicl}`` (spec v2; v1 had no
    HICL segment) and ``INFUSION_DRUG//{drug (unit)}``. Verified against
    the real extractions' top medication codes, not assumed. The rule:
    drop a trailing pure-ID segment (digits or ``unk``), find the drug
    segment (first segment, or second when the first is a START/STOP
    action), and reduce it to its lowercased ingredient. When
    ``hicl_ingredients`` is given (eICU only; MIMIC's trailing IDs are
    NDC/GSN numbers, not HICLs) and the trailing ID resolves, the
    dictionary ingredient replaces the name-derived one. Action and
    event-text segments survive; they carry clinical signal (given vs.
    not given).
    """
    parts = code.split("//")
    if parts[0] in INFUSION_FAMILIES:
        return _normalize_infusion_code(parts)
    if parts[0] not in MEDICATION_FAMILIES or len(parts) < 2:
        return code
    segs = parts[1:]
    trailing_id: Optional[str] = None
    if len(segs) >= 2 and _ID_SEGMENT_RE.fullmatch(segs[-1]):
        trailing_id = segs[-1]
        segs = segs[:-1]
    drug_idx = 1 if segs[0].upper() in _ACTION_SEGMENTS and len(segs) > 1 else 0
    resolved = (
        hicl_ingredients.get(trailing_id)
        if hicl_ingredients is not None and trailing_id is not None
        else None
    )
    if resolved:
        segs[drug_idx] = resolved
    elif segs[drug_idx]:
        segs[drug_idx] = _normalize_name(segs[drug_idx])
    return "//".join([parts[0], *segs])


def normalize_medication_codes(
    events: pl.DataFrame,
    *,
    code_col: str = "code",
    hicl_ingredients: Optional[HiclIngredients] = None,
) -> pl.DataFrame:
    """Rewrite medication codes to ingredient level; pass everything else through.

    Applies :func:`normalize_medication_code` over the *distinct* codes
    (tens of thousands) and joins the mapping back onto the events
    (hundreds of millions) -- exact same rule as the scalar function with
    no vectorized reimplementation to drift out of sync, at hash-join
    cost instead of per-row Python. ``hicl_ingredients`` is threaded
    through unchanged (see :func:`hicl_ingredients_for_source`).
    """
    distinct = events.select(pl.col(code_col).unique().alias(code_col))
    mapping = distinct.with_columns(
        pl.col(code_col)
        .map_elements(
            lambda code: normalize_medication_code(code, hicl_ingredients),
            return_dtype=pl.Utf8,
        )
        .alias("_normalized")
    )
    return (
        events.join(mapping, on=code_col, how="left")
        .with_columns(pl.col("_normalized").fill_null(pl.col(code_col)).alias(code_col))
        .drop("_normalized")
    )


def maybe_normalize(
    events: pl.DataFrame,
    *,
    enabled: bool,
    code_col: str = "code",
    source: Optional[str] = None,
) -> pl.DataFrame:
    """Apply :func:`normalize_medication_codes` when ``enabled``, else pass through.

    ``source`` selects the source-specific dictionary (eICU's HICL table);
    pass the run's ``TrainingConfig.source`` so training and every
    inference path normalize identically.
    """
    if not enabled:
        return events
    return normalize_medication_codes(
        events, code_col=code_col, hicl_ingredients=hicl_ingredients_for_source(source)
    )


def icd_category_code(code: str) -> Optional[str]:
    """Return the 3-character-category backoff for an ICD-coded event, or None.

    ``DIAGNOSIS//ICD//10//I5023`` backs off to ``DIAGNOSIS//ICD//10//I50``,
    the chapter-level category clinicians themselves group by (heart
    failure, unspecified vs. one of its many fifth-character variants).
    Applies to DIAGNOSIS and PROCEDURE ICD codes; anything else, or a code
    already at (or below) category length, gets ``None``.
    """
    parts = code.split("//")
    if (
        len(parts) == 4
        and parts[0] in ("DIAGNOSIS", "PROCEDURE")
        and parts[1] == "ICD"
        and len(parts[3]) > 3
    ):
        return "//".join([*parts[:3], parts[3][:3]])
    return None
