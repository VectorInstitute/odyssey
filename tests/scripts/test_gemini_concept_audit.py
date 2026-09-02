"""Tests for the G2 GEMINI concept/binning audit (local, no GEMINI access)."""

import json
from pathlib import Path

import pytest

from scripts.gemini_concept_audit import audit, count_bounds, main


#: Synthetic suppressed inventory in the real export_codes.py shape:
#: {code: "<1000" | "<rounded thousands as string>"}. Codes chosen to hit
#: real GEMINI_TO_LOINC prefixes (heart rate, SI creatinine, arterial
#: lactate) plus one high-count unmapped lab and one suppressed-count code.
INVENTORY = {
    "VITALS//3027018//bpm": "55000",
    "LAB//3020564//umol/l": "15000",
    "LAB//3018405//mmol/l": "1000",
    "LAB//9999999//g/l": "20000",
    "LAB//8888888//": "<1000",
    "MEDICATION//warfarin//": "3000",  # not a value family: excluded
}


def test_count_bounds_parses_exact_and_suppressed() -> None:
    assert count_bounds("55000") == (55000, 55000)
    assert count_bounds("<1000") == (0, 999)


def test_concept_resolution_matches_the_source_expansion() -> None:
    result = audit(INVENTORY)
    resolution = result["concept_resolution"]
    assert result["n_concepts_mimic_v3"] == len(resolution)
    # Vitals-threshold concepts resolve through the LOINC layer.
    assert resolution["tachycardia"] is True
    assert resolution["acute_kidney_injury"] is True
    # v3 lab-severity concepts resolve through the electrolyte/CBC/INR
    # mapping; the four that need signals the datacut lacks do not.
    assert resolution["hyperkalemia"] is True
    assert resolution["anemia"] is True
    assert resolution["shock"] is False  # no mean arterial pressure
    assert resolution["oliguria"] is False  # no urine output anywhere
    assert result["n_concepts_resolving"] == sum(resolution.values())


def test_mapping_rows_report_units_and_reality_drift() -> None:
    result = audit(INVENTORY)
    creat = result["mapping"]["LAB//3020564//"]
    assert creat["loinc"] == "2160-0"
    assert creat["unit_variants"] == ["umol/l"]
    assert creat["curated_bins"] is True
    assert creat["token_count_bounds"] == [15000, 15000]
    # Prefixes present in the inventory are not flagged as drift; absent
    # ones (this tiny inventory omits most) are.
    assert "LAB//3020564//" not in result["unmatched_mapping_prefixes"]
    assert "VITALS//3024171//" in result["unmatched_mapping_prefixes"]


def test_binning_portability_bounds_are_hand_checkable() -> None:
    result = audit(INVENTORY)
    lab = result["binning_portability"]["LAB"]
    # LAB codes: creatinine 15000 + lactate 1000 curated; unmapped 20000
    # exact + one "<1000" in [0, 999].
    assert lab["n_codes"] == 4
    assert lab["n_codes_curated"] == 2
    assert lab["token_count_bounds"] == [36000, 36999]
    assert lab["curated_token_count_bounds"] == [16000, 16000]
    lo, hi = lab["curated_fraction_bounds"]
    assert lo == pytest.approx(16000 / 36999)
    assert hi == pytest.approx(16000 / 36000)
    vitals = result["binning_portability"]["VITALS"]
    assert vitals["curated_fraction_bounds"] == [1.0, 1.0]


def test_unmapped_candidates_rank_by_count_and_skip_mapped_families() -> None:
    result = audit(INVENTORY, top_unmapped=1)
    assert result["top_unmapped_value_codes"] == [
        {"code": "LAB//9999999//g/l", "count": "20000"}
    ]
    full = audit(INVENTORY)
    codes = {row["code"] for row in full["top_unmapped_value_codes"]}
    assert "MEDICATION//warfarin//" not in codes  # not a value family
    assert "LAB//3020564//umol/l" not in codes  # mapped


def test_main_refuses_to_overwrite_an_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = tmp_path / "codes_inventory.json"
    inventory.write_text(json.dumps(INVENTORY))
    existing = tmp_path / "concept_audit.json"
    existing.write_text("{}")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--codes-inventory",
            str(inventory),
            "--output-json",
            str(existing),
        ],
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main()


def test_main_end_to_end_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = tmp_path / "codes_inventory.json"
    inventory.write_text(json.dumps(INVENTORY))
    out = tmp_path / "concept_audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--codes-inventory",
            str(inventory),
            "--output-json",
            str(out),
        ],
    )
    main()
    report = json.loads(out.read_text())
    assert report["n_inventory_codes"] == len(INVENTORY)
    assert report["n_concepts_resolving"] >= 1
