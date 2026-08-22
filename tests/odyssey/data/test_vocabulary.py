"""Tests for the code and type vocabularies."""

import json
from pathlib import Path

import polars as pl

from odyssey.data.vocabulary import (
    OTHER_TYPE,
    PAD_ID,
    UNK_ID,
    VISIT_TYPE,
    Vocabulary,
    code_type,
    code_types,
)


def test_build_keeps_frequent_drops_rare() -> None:
    codes = ["A"] * 10 + ["B"] * 5 + ["C"] * 1
    vocab = Vocabulary.build(codes, min_count=5)
    assert vocab.encode("A") != UNK_ID
    assert vocab.encode("B") != UNK_ID
    assert vocab.encode("C") == UNK_ID  # below min_count


def test_build_respects_max_size() -> None:
    codes = []
    for i in range(20):
        codes += [f"code_{i}"] * (20 - i)  # code_0 most frequent, code_19 least
    vocab = Vocabulary.build(codes, min_count=1, max_size=5)
    # Only the 5 most frequent codes plus PAD/UNK.
    assert len(vocab) == 5 + 2
    assert vocab.encode("code_0") != UNK_ID
    assert vocab.encode("code_19") == UNK_ID


def test_pad_and_unk_are_reserved_at_fixed_ids() -> None:
    vocab = Vocabulary.build(["A", "A", "B", "B"], min_count=1)
    assert vocab.token_to_id["[PAD]"] == PAD_ID
    assert vocab.token_to_id["[UNK]"] == UNK_ID


def test_unseen_code_maps_to_unk() -> None:
    vocab = Vocabulary.build(["A", "A"], min_count=1)
    assert vocab.encode("never_seen") == UNK_ID


def test_decode_roundtrip() -> None:
    vocab = Vocabulary.build(["A", "A", "B", "B"], min_count=1)
    token_id = vocab.encode("A")
    assert vocab.decode(token_id) == "A"


def test_save_load_roundtrip(tmp_path: Path) -> None:
    vocab = Vocabulary.build(["A", "A", "B", "B", "C"], min_count=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.token_to_id == vocab.token_to_id


def test_build_from_counts_matches_build_on_the_same_data() -> None:
    codes = ["A"] * 10 + ["B"] * 5 + ["C"] * 1
    from_counts = Vocabulary.build_from_counts({"A": 10, "B": 5, "C": 1}, min_count=5)
    from_list = Vocabulary.build(codes, min_count=5)
    assert from_counts.token_to_id == from_list.token_to_id


def test_build_from_counts_respects_max_size() -> None:
    counts = {f"code_{i}": 20 - i for i in range(20)}  # code_0 most frequent
    vocab = Vocabulary.build_from_counts(counts, min_count=1, max_size=5)
    assert len(vocab) == 5 + 2
    assert vocab.encode("code_0") != UNK_ID
    assert vocab.encode("code_19") == UNK_ID


def test_code_type_known_prefixes() -> None:
    assert code_type("LAB//220045//bpm") == code_type("LAB//RESULT//50813//mmol/L")
    assert code_type("HOSPITAL_ADMISSION//...") == VISIT_TYPE


def test_code_type_eicu_prefixes() -> None:
    # eICU code families (specs/eICU.yaml) map into the same taxonomy as
    # the equivalent MIMIC-IV families.
    assert code_type("VITALS//PERIODIC//HEARTRATE") == code_type("LAB//220045//bpm")
    assert code_type("LAB//creatinine//mg/dL") == code_type("LAB//RESULT//50912//mg/dL")
    assert code_type("ADMISSION_DIAGNOSIS//SEPSIS") == code_type(
        "DIAGNOSIS//ICD//9//0389"
    )
    assert code_type("INFUSION_DRUG") == code_type("MEDICATION//STARTED//X")
    assert code_type("ICU_ADMISSION//UNK//admit") == VISIT_TYPE


# Every code prefix scripts/gemini/extract_meds.py's extract_<table>
# functions actually emit (as of the 2026-08-21 provider-table addition;
# see extract_admissions/extract_icu/extract_labs/extract_vitals/
# extract_pharmacy/extract_diagnoses/extract_procedures/extract_radiology/
# extract_providers). Kept here, not derived from that file dynamically,
# since extract_meds.py is a standalone script loaded via importlib in its
# own tests, not an importable package module -- update this list by hand
# alongside any new GEMINI code prefix.
_GEMINI_PREFIXES = [
    "ADMISSION",
    "DISCHARGE",
    "ICU_ADMISSION",
    "ICU_DISCHARGE",
    "LAB",
    "VITALS",
    "MEDICATION",
    "DIAGNOSIS",
    "PROCEDURE",
    "IMAGING",
    "PROVIDER",
    "ED_REGISTRATION",
    "ED_TRIAGE",
    "ED_OUT",
    "ED_DIAGNOSIS",
    "ER_CONSULT",
    "TRANSFER_TO",
    "BILLING_CMG",
    "BILLING_HIG",
]


def test_code_type_every_gemini_prefix_has_an_intentional_mapping() -> None:
    # Real risk this guards against: a new GEMINI code prefix silently
    # falls through to OTHER_TYPE via code_type's dict.get(prefix,
    # OTHER_TYPE) default, weakening the "a lab is a lab regardless of
    # institution" cross-source type-token consistency this taxonomy
    # exists for -- with no test failure to flag that it happened. Every
    # prefix GEMINI actually emits must have its own _PREFIX_TO_TYPE
    # entry, even if that entry deliberately maps to OTHER_TYPE (PROVIDER
    # does, on purpose -- see vocabulary.py's comment) -- an *explicit*
    # OTHER_TYPE is fine; an *implicit* one (missing from the dict
    # entirely) is what this test catches.
    from odyssey.data.vocabulary import _PREFIX_TO_TYPE  # noqa: PLC0415

    missing = [p for p in _GEMINI_PREFIXES if p not in _PREFIX_TO_TYPE]
    assert missing == [], f"GEMINI prefixes with no _PREFIX_TO_TYPE entry: {missing}"


def test_code_type_unknown_prefix_falls_back_to_other() -> None:
    assert code_type("Blood Pressure Standing") == OTHER_TYPE


def test_code_types_vectorized_matches_scalar() -> None:
    codes = ["DIAGNOSIS//ICD//10//A047", "MEDICATION//12345", "unknown_prefix"]
    assert code_types(codes) == [code_type(c) for c in codes]


def test_from_meds_codes_metadata_keeps_every_code_regardless_of_min_count(
    tmp_path: Path,
) -> None:
    # codes.parquet is a metadata table, not a frequency table -- every code
    # appears once, so a min_count tuned for real frequencies (e.g. 5) must
    # not silently empty the vocabulary.
    path = tmp_path / "codes.parquet"
    pl.DataFrame({"code": ["A", "B", "C"]}).write_parquet(path)

    vocab = Vocabulary.from_meds_codes_metadata(path, min_count=5)
    assert vocab.encode("A") != UNK_ID
    assert vocab.encode("B") != UNK_ID
    assert vocab.encode("C") != UNK_ID


def test_from_meds_codes_metadata_respects_max_size(tmp_path: Path) -> None:
    path = tmp_path / "codes.parquet"
    pl.DataFrame({"code": [f"code_{i}" for i in range(10)]}).write_parquet(path)

    vocab = Vocabulary.from_meds_codes_metadata(path, max_size=3)
    assert len(vocab) == 3 + 2  # 3 codes + PAD/UNK


# ---------------------------------------------------------------------------
# ICD hierarchical backoff
# ---------------------------------------------------------------------------


def test_backoff_rolls_rare_codes_into_their_category() -> None:
    # Three rare fifth-character variants (2 each, below min_count=5)
    # roll into the I50 category, which then earns a real slot.
    counts = {
        "DIAGNOSIS//ICD//10//I5021": 2,
        "DIAGNOSIS//ICD//10//I5022": 2,
        "DIAGNOSIS//ICD//10//I5023": 2,
        "LAB//220045//bpm::HIGH": 50,
    }
    vocab = Vocabulary.build_from_counts(counts, min_count=5, backoff="icd3")
    cat = vocab.encode("DIAGNOSIS//ICD//10//I50")
    assert cat != UNK_ID
    # every rare variant encodes to the category token, not [UNK]
    assert vocab.encode("DIAGNOSIS//ICD//10//I5021") == cat
    assert vocab.encode("DIAGNOSIS//ICD//10//I5099") == cat  # unseen variant too


def test_backoff_keeps_frequent_full_codes_as_their_own_tokens() -> None:
    counts = {
        "DIAGNOSIS//ICD//10//I5023": 20,  # frequent: keeps its own slot
        "DIAGNOSIS//ICD//10//I5021": 2,  # rare: rolls to category
        "LAB//220045//bpm::HIGH": 50,
    }
    vocab = Vocabulary.build_from_counts(counts, min_count=5, backoff="icd3")
    full = vocab.encode("DIAGNOSIS//ICD//10//I5023")
    assert full != UNK_ID
    assert vocab.decode(full) == "DIAGNOSIS//ICD//10//I5023"
    # the rare sibling lands on the category, distinct from the full code
    assert vocab.encode("DIAGNOSIS//ICD//10//I5021") != full


def test_backoff_survives_save_load_roundtrip(tmp_path: Path) -> None:
    counts = {"DIAGNOSIS//ICD//10//I5021": 2, "DIAGNOSIS//ICD//10//I5022": 4}
    vocab = Vocabulary.build_from_counts(counts, min_count=5, backoff="icd3")
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.backoff == "icd3"
    assert loaded.encode("DIAGNOSIS//ICD//10//I5021") == vocab.encode(
        "DIAGNOSIS//ICD//10//I5021"
    )


def test_old_bare_dict_vocab_files_still_load(tmp_path: Path) -> None:
    # pre-backoff format: a plain token_to_id dict
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"[PAD]": 0, "[UNK]": 1, "A": 2}))
    loaded = Vocabulary.load(path)
    assert loaded.encode("A") == 2
    assert loaded.backoff is None


def test_unknown_backoff_name_raises() -> None:
    try:
        Vocabulary({"[PAD]": 0, "[UNK]": 1}, backoff="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
