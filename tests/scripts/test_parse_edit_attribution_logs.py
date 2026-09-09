"""cohort_worsen.py's fixed-format stdout summary parses into structured JSON."""

from __future__ import annotations

import pytest

from scripts.parse_edit_attribution_logs import parse_log


SAMPLE = """\
subject 1 [1/2]: ['platelets'] concept 0.10->0.20
subject 2 [2/2]: ['creatinine'] concept 0.30->0.40

=== sepsis3 on mimic_iv, n=2 subjects, top_k=4, index_frac=0.3333 ===
baseline concept prob >= 0.9: 0/2 subjects (saturation check)
edit signal frequency: {'platelets': 1, 'creatinine': 1}
mean concept-probability delta: +0.1000

Sign agreement (risk should INCREASE when discovered evidence is worsened):
  death                  8h  :   2/  2 = 100.0%
  death                  24h :   1/  2 =  50.0%
  vasopressor_start      8h  :   2/  2 = 100.0%
"""


def test_parse_log_extracts_header() -> None:
    result = parse_log(SAMPLE)
    assert result["concept"] == "sepsis3"
    assert result["source"] == "mimic_iv"
    assert result["n_subjects"] == 2


def test_parse_log_extracts_saturation() -> None:
    result = parse_log(SAMPLE)
    assert result["n_baseline_saturated"] == 0
    assert result["n_baseline_scored"] == 2


def test_parse_log_extracts_all_cells() -> None:
    result = parse_log(SAMPLE)
    assert len(result["cells"]) == 3
    death_8h = next(
        c for c in result["cells"] if c["event"] == "death" and c["horizon"] == "8h"
    )
    assert death_8h == {"event": "death", "horizon": "8h", "agree": 2, "total": 2, "pct": 100.0}


def test_parse_log_partial_agreement_cell() -> None:
    result = parse_log(SAMPLE)
    death_24h = next(
        c for c in result["cells"] if c["event"] == "death" and c["horizon"] == "24h"
    )
    assert death_24h["agree"] == 1
    assert death_24h["total"] == 2
    assert death_24h["pct"] == 50.0


def test_parse_log_no_saturation_line_yields_none() -> None:
    no_sat = SAMPLE.replace(
        "baseline concept prob >= 0.9: 0/2 subjects (saturation check)\n", ""
    )
    result = parse_log(no_sat)
    assert result["n_baseline_saturated"] is None
    assert result["n_baseline_scored"] is None


def test_parse_log_missing_header_raises() -> None:
    with pytest.raises(ValueError, match="header"):
        parse_log("no header here, just noise\n")


def test_parse_log_ignores_per_subject_lines() -> None:
    # the per-subject progress lines ("subject N [i/n]: ...") must not be
    # mistaken for sign-agreement rows.
    result = parse_log(SAMPLE)
    assert not any(c["event"].startswith("subject") for c in result["cells"])
