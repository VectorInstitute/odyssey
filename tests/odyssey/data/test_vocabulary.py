"""Tests for the code and type vocabularies."""

from pathlib import Path

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
    assert loaded.encode("A") == vocab.encode("A")


def test_code_type_known_prefixes() -> None:
    assert code_type("LAB//220045//bpm") == code_type("LAB//RESULT//50813//mmol/L")
    assert code_type("HOSPITAL_ADMISSION//...") == VISIT_TYPE


def test_code_type_unknown_prefix_falls_back_to_other() -> None:
    assert code_type("Blood Pressure Standing") == OTHER_TYPE


def test_code_types_vectorized_matches_scalar() -> None:
    codes = ["DIAGNOSIS//ICD//10//A047", "MEDICATION//12345", "unknown_prefix"]
    assert code_types(codes) == [code_type(c) for c in codes]
