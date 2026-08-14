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
from typing import Dict, Iterable, List, Mapping, Union

import polars as pl


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1
_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


class Vocabulary:
    """A frequency-filtered mapping from MEDS event codes to token ids."""

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        """Initialize from an already-built token -> id mapping."""
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

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
        cls, counts: Mapping[str, int], *, min_count: int = 5, max_size: int = 50_000
    ) -> "Vocabulary":
        """Build a vocabulary from already-aggregated ``code -> count`` pairs.

        Bounded by vocabulary cardinality, not by the number of raw
        events -- the entry point for real, large event streams (see
        :meth:`build`'s docstring on why this matters).
        """
        kept = [
            code
            for code, count in Counter(counts).most_common(max_size)
            if count >= min_count
        ]
        token_to_id = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
        for code in kept:
            token_to_id[code] = len(token_to_id)
        return cls(token_to_id)

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
        """Map a code to its token id, or ``[UNK]`` if not in the vocabulary."""
        return self.token_to_id.get(code, UNK_ID)

    def decode(self, token_id: int) -> str:
        """Map a token id back to its code, or ``[UNK]`` if out of range."""
        return self.id_to_token.get(token_id, UNK_TOKEN)

    def __len__(self) -> int:
        """Return the vocabulary size, including special tokens."""
        return len(self.token_to_id)

    def save(self, path: Union[str, Path]) -> None:
        """Save as JSON."""
        Path(path).write_text(json.dumps(self.token_to_id))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """Load from JSON written by :meth:`save`."""
        return cls(json.loads(Path(path).read_text()))


# Fixed event-type taxonomy. 0 is reserved for padding (matching
# odyssey.data.types/ClinicalEventEmbeddings' padding_idx convention);
# the remaining 8 slots fill the default type_vocab_size=9. Built from the
# code prefixes actually observed in the MIMIC-IV 3.1 MEDS extraction
# (odyssey/data/concepts.py's itemid-keyed LAB//... codes fall under LAB).
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
}


def code_type(code: str) -> int:
    """Map a MEDS code to a small fixed event-type id (see the constants above)."""
    prefix = code.split("//", 1)[0]
    return _PREFIX_TO_TYPE.get(prefix, OTHER_TYPE)


def code_types(codes: List[str]) -> List[int]:
    """Vectorized convenience wrapper around :func:`code_type`."""
    return [code_type(c) for c in codes]
