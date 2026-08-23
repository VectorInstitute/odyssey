"""Token and type vocabularies for MEDS event codes.

Two separate, small vocabularies a patient sequence is built from:

- :class:`Vocabulary` maps each MEDS ``code`` string (e.g.
  ``"LAB//220045//bpm"``) to an integer token id, frequency-filtered so
  rare/noisy codes collapse to ``[UNK]`` instead of bloating the embedding
  table.
- :func:`code_type` maps a code to one of a small fixed set of event
  *types* (diagnosis, medication, lab, ...), matching
  :class:`odyssey.models.embeddings.ClinicalEventEmbeddings`'s
  ``type_vocab_size`` token-type embedding.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Union

import polars as pl

from odyssey.data.code_normalization import icd_category_code


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1
_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


# Named backoff functions a Vocabulary can carry. Referenced by name (not by
# function object) so the choice survives save()/load() round trips.
BACKOFFS: Dict[str, Callable[[str], Optional[str]]] = {
    "icd3": icd_category_code,
}


class Vocabulary:
    """A frequency-filtered mapping from MEDS event codes to token ids.

    With a ``backoff`` set (see :data:`BACKOFFS`), a code missing from the
    vocabulary falls back to its backoff code before giving up to
    ``[UNK]``: a rare ``DIAGNOSIS//ICD//10//I5023`` encodes as the
    ``DIAGNOSIS//ICD//10//I50`` category token instead of dissolving into
    an unpredictable catch-all. :meth:`build_from_counts` applies the same
    rule when building: sub-``min_count`` codes roll their counts up into
    their backoff code, so common categories earn real vocabulary slots
    from the mass of their rare children.
    """

    def __init__(
        self, token_to_id: Dict[str, int], *, backoff: Optional[str] = None
    ) -> None:
        """Initialize from an already-built token -> id mapping."""
        if backoff is not None and backoff not in BACKOFFS:
            raise ValueError(
                f"unknown backoff {backoff!r}; registered: {sorted(BACKOFFS)}"
            )
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.backoff = backoff

    @classmethod
    def build(
        cls, codes: Iterable[str], *, min_count: int = 5, max_size: int = 50_000
    ) -> "Vocabulary":
        """Build a vocabulary from an iterable of (possibly repeated) codes.

        Keeps the ``max_size`` most frequent codes with count >= ``min_count``,
        always reserving ids 0/1 for ``[PAD]``/``[UNK]``.

        ``codes`` is fully materialized into a :class:`collections.Counter`
        here, one Python object per element -- fine for the small,
        already-list-like inputs this is normally called with (tests, a
        codes-metadata file), but a real event stream with tens of
        millions of rows should use :meth:`build_from_counts` on a
        vectorized, Arrow-native frequency count instead (see
        :func:`odyssey.training.data.build_vocabulary`), never
        ``.to_list()`` the raw column here.
        """
        return cls.build_from_counts(
            Counter(codes), min_count=min_count, max_size=max_size
        )

    @classmethod
    def build_from_counts(
        cls,
        counts: Mapping[str, int],
        *,
        min_count: int = 5,
        max_size: int = 50_000,
        backoff: Optional[str] = None,
    ) -> "Vocabulary":
        """Build a vocabulary from already-aggregated ``code -> count`` pairs.

        Bounded by vocabulary cardinality, not by the number of raw
        events -- the entry point for real, large event streams (see
        :meth:`build`'s docstring on why this matters). With ``backoff``,
        sub-``min_count`` codes roll their counts into their backoff code
        first (see the class docstring).
        """
        rolled = Counter(counts)
        if backoff is not None:
            backoff_fn = BACKOFFS[backoff]
            rolled = Counter()
            for code, count in counts.items():
                if count >= min_count:
                    rolled[code] += count
                else:
                    rolled[backoff_fn(code) or code] += count
        # Deterministic order regardless of how the counts were assembled:
        # by count descending, then by code (most_common alone breaks ties
        # by insertion order, which differs between the in-memory and the
        # shard-streaming corpus paths).
        ranked = sorted(rolled.items(), key=lambda kv: (-kv[1], kv[0]))[:max_size]
        kept = [code for code, count in ranked if count >= min_count]
        token_to_id = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
        for code in kept:
            token_to_id[code] = len(token_to_id)
        return cls(token_to_id, backoff=backoff)

    @classmethod
    def from_meds_codes_metadata(
        cls,
        codes_parquet_path: Union[str, Path],
        *,
        min_count: int = 5,
        max_size: int = 50_000,
    ) -> "Vocabulary":
        """Build from a MEDS ``metadata/codes.parquet`` file.

        ``codes.parquet``'s schema is metadata (description, parent codes),
        not a frequency table, so every code here is treated as observed
        exactly once regardless of ``min_count``; prefer :meth:`build`
        directly from the event stream when true frequencies matter.
        """
        codes = pl.read_parquet(codes_parquet_path)["code"].to_list()
        # Every code has an unweighted count of 1 above, so any min_count > 1
        # would silently empty the vocabulary; clamp so the caller's default
        # (tuned for `build`'s real frequencies) doesn't do that here.
        return cls.build(codes, min_count=min(min_count, 1), max_size=max_size)

    def encode(self, code: str) -> int:
        """Map a code to its id, trying its backoff code before ``[UNK]``."""
        direct = self.token_to_id.get(code)
        if direct is not None:
            return direct
        if self.backoff is not None:
            fallback = BACKOFFS[self.backoff](code)
            if fallback is not None:
                return self.token_to_id.get(fallback, UNK_ID)
        return UNK_ID

    def decode(self, token_id: int) -> str:
        """Map a token id back to its code, or ``[UNK]`` if out of range."""
        return self.id_to_token.get(token_id, UNK_TOKEN)

    def __len__(self) -> int:
        """Return the vocabulary size, including special tokens."""
        return len(self.token_to_id)

    def save(self, path: Union[str, Path]) -> None:
        """Save as JSON: the token map plus the backoff name."""
        Path(path).write_text(
            json.dumps({"token_to_id": self.token_to_id, "backoff": self.backoff})
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """Load from JSON written by :meth:`save` (or the older bare-dict format)."""
        data = json.loads(Path(path).read_text())
        if "token_to_id" in data:
            return cls(data["token_to_id"], backoff=data.get("backoff"))
        return cls(data)


# Fixed event-type taxonomy. 0 is reserved for padding (matching
# odyssey.data.types/ClinicalEventEmbeddings' padding_idx convention);
# the remaining 8 slots fill the default type_vocab_size=9. Built from the
# code prefixes actually observed in the MIMIC-IV 3.1 MEDS extraction
# (odyssey/data/concepts.py's itemid-keyed LAB//... codes fall under LAB)
# plus the eICU extraction's families (specs/eICU.yaml): both sources map
# into the same small taxonomy, so a model sees "a lab is a lab"
# regardless of which institution charted it.
PAD_TYPE = 0
DIAGNOSIS_TYPE = 1
MEDICATION_TYPE = 2
PROCEDURE_TYPE = 3
LAB_TYPE = 4
VISIT_TYPE = 5
DEMOGRAPHIC_TYPE = 6
BILLING_TYPE = 7
OTHER_TYPE = 8

_PREFIX_TO_TYPE: Dict[str, int] = {
    "DIAGNOSIS": DIAGNOSIS_TYPE,
    "MEDICATION": MEDICATION_TYPE,
    "INFUSION_START": MEDICATION_TYPE,
    "INFUSION_END": MEDICATION_TYPE,
    "SUBJECT_WEIGHT_AT_INFUSION": MEDICATION_TYPE,
    "PROCEDURE": PROCEDURE_TYPE,
    "HCPCS": BILLING_TYPE,
    "LAB": LAB_TYPE,
    "HOSPITAL_ADMISSION": VISIT_TYPE,
    "HOSPITAL_DISCHARGE": VISIT_TYPE,
    "ICU_ADMISSION": VISIT_TYPE,
    "ICU_DISCHARGE": VISIT_TYPE,
    "TRANSFER_TO": VISIT_TYPE,
    "ED_REGISTRATION": VISIT_TYPE,
    "ED_OUT": VISIT_TYPE,
    "SUBJECT_FLUID_OUTPUT": VISIT_TYPE,
    "GENDER": DEMOGRAPHIC_TYPE,
    "RACE": DEMOGRAPHIC_TYPE,
    "LANGUAGE": DEMOGRAPHIC_TYPE,
    "INSURANCE": DEMOGRAPHIC_TYPE,
    "MARITAL_STATUS": DEMOGRAPHIC_TYPE,
    "MEDS_BIRTH": DEMOGRAPHIC_TYPE,
    "MEDS_DEATH": DEMOGRAPHIC_TYPE,
    "DRG": BILLING_TYPE,
    # eICU families (specs/eICU.yaml). CAREPLAN_*/ALLERGY deliberately fall
    # through to OTHER_TYPE -- no better slot exists in this taxonomy.
    "VITALS": LAB_TYPE,
    "ADMISSION_DIAGNOSIS": DIAGNOSIS_TYPE,
    "TREATMENT": PROCEDURE_TYPE,
    "INFUSION_DRUG": MEDICATION_TYPE,
    "ICU_ADMISSION_WEIGHT": DEMOGRAPHIC_TYPE,
    "ICU_ADMISSION_HEIGHT": DEMOGRAPHIC_TYPE,
    "ICU_DISCHARGE_WEIGHT": DEMOGRAPHIC_TYPE,
    # GEMINI families (scripts/gemini/extract_meds.py). LAB/VITALS/
    # MEDICATION/DIAGNOSIS/PROCEDURE/ICU_ADMISSION/ICU_DISCHARGE already
    # covered above -- GEMINI's own admdad_subset admission/discharge and
    # radiology events are the only genuinely new prefixes.
    "ADMISSION": VISIT_TYPE,
    "DISCHARGE": VISIT_TYPE,
    "IMAGING": PROCEDURE_TYPE,
    # Not a clinical event type -- physicians_subset provider-hash events,
    # kept option-preserving for a tabled (not abandoned) physician-
    # preference IV study, see docs/gemini_extraction.md's "Why provider
    # ids are preserved". Explicit OTHER_TYPE, not a silent fall-through:
    # deliberate, not an omission.
    "PROVIDER": OTHER_TYPE,
    # GEMINI's ER/transfer/billing families, added alongside the core
    # tables above. ED_REGISTRATION/ED_OUT/TRANSFER_TO already existed in
    # this table (MIMIC's own convention) and are reused as-is, not
    # redefined here -- GEMINI's extract_er/extract_transfers just produce
    # codes under those same prefixes. ED_TRIAGE, ED_DIAGNOSIS, ER_CONSULT,
    # BILLING_CMG, BILLING_HIG are genuinely new. ED_DIAGNOSIS is kept
    # distinct from ipdiagnosis_subset's DIAGNOSIS// (different clinical
    # context, same DIAGNOSIS_TYPE bucket); BILLING_CMG/BILLING_HIG are
    # kept distinct from each other and from DRG (CIHI's Canadian CMG/HIG
    # casemix systems, not the US DRG system -- same BILLING_TYPE bucket,
    # different code vocabularies, so collapsing them onto one prefix
    # would conflate real differences).
    "ED_TRIAGE": VISIT_TYPE,
    "ED_DIAGNOSIS": DIAGNOSIS_TYPE,
    "ER_CONSULT": OTHER_TYPE,
    "BILLING_CMG": BILLING_TYPE,
    "BILLING_HIG": BILLING_TYPE,
}


def code_type(code: str) -> int:
    """Map a MEDS code to a small fixed event-type id (see the constants above)."""
    prefix = code.split("//", 1)[0]
    return _PREFIX_TO_TYPE.get(prefix, OTHER_TYPE)


def code_types(codes: List[str]) -> List[int]:
    """Vectorized convenience wrapper around :func:`code_type`."""
    return [code_type(c) for c in codes]


def is_anchor(code: str) -> bool:
    """Return True for admission/discharge/demographic-static rows.

    Used by :mod:`odyssey.data.degrade` (docs/missingness_protocol.md) to
    protect the rows a degraded record cannot lose without becoming a
    different task: :data:`VISIT_TYPE` (admission/discharge/ICU
    admission-discharge/transfer/ED registration) and
    :data:`DEMOGRAPHIC_TYPE` (sex/race/language/insurance/marital
    status/birth/death) are exactly "the visit envelope plus who this
    patient is". Not specific to any one degradation axis -- any caller
    that needs to know "is this row the kind that must always survive"
    reuses this rather than re-deriving it from :func:`code_type`.
    """
    return code_type(code) in (VISIT_TYPE, DEMOGRAPHIC_TYPE)


#: Row-family names :func:`row_family` classifies into, for the
#: missingness protocol's family-blackout axis (docs/missingness_protocol.md,
#: axis B) -- kept here, next to :func:`code_type`, since row_family is
#: itself just a finer-grained read of the same code-prefix taxonomy.
LAB_FAMILY = "labs"
VITAL_FAMILY = "vitals"
MEDICATION_FAMILY = "medications"
ROW_FAMILIES = (LAB_FAMILY, VITAL_FAMILY, MEDICATION_FAMILY)


def row_family(code: str, *, source: str) -> Optional[str]:
    """Classify one MEDS code into a row family, or ``None`` if none applies.

    Reuses two existing, documented conventions rather than inventing a new
    classifier:

    1. Medications: :func:`code_type`'s :data:`MEDICATION_TYPE` bucket
       already unifies medication/infusion code prefixes across every
       source.
    2. Labs vs. vitals/charting: *not* distinguished by :func:`code_type`
       -- MIMIC-IV charts both under one :data:`LAB_TYPE` bucket -- but
       *is* distinguished by the raw code-prefix shape itself, already
       documented in :mod:`odyssey.data.code_mapping`'s module docstring:
       MIMIC-IV's ``hosp/labevents`` rows are ``LAB//RESULT//{itemid}//...``
       (real labs); ``icu/chartevents`` rows are ``LAB//{itemid}//...``,
       with no ``RESULT`` segment (vitals/charting). eICU and GEMINI don't
       have this ambiguity: vitals are already a separate top-level
       ``VITALS//...`` prefix (specs/eICU.yaml,
       scripts/gemini/extract_meds.py's ``extract_vitals``), disjoint from
       their own real-lab ``LAB//...`` prefix (``extract_labs``).
    """
    if code_type(code) == MEDICATION_TYPE:
        return MEDICATION_FAMILY
    top = code.split("//", 1)[0]
    if top == "VITALS":
        return VITAL_FAMILY
    if top == "LAB":
        if source == "mimic_iv":
            second = code.split("//", 2)[1] if code.count("//") >= 2 else ""
            return VITAL_FAMILY if second != "RESULT" else LAB_FAMILY
        return LAB_FAMILY
    return None
