"""The eval-export whitelist accepts every aggregate field the eval writes.

Regression guard for 2026-09-02: ``eval-forecast gemini_full_DEC_v12``
ran to completion on the H200 and then refused to export its own
summary, because ``value_metrics`` and the set-level task accuracies
were added to ``InferenceResults`` after the whitelist was written. The
whitelist must refuse anything unknown (that is its job: nothing
per-patient leaves GEMINI), so this test runs the shipped validator, not
a copy of it, on the exact shape ``run_inference`` writes.
"""

import json
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
RUN_SH = REPO / "scripts" / "gemini" / "run.sh"
BANKED = (
    REPO
    / "research_journal"
    / "figure_data"
    / "vm1"
    / "full_run_DEC_v12"
    / "inference_results.json"
)


def _validator_source() -> str:
    source = RUN_SH.read_text()
    match = re.search(
        r"python3 - \"\$source_json\" <<'PY'\n(.*?)\nPY\n", source, re.DOTALL
    )
    assert match, "export validator heredoc not found in run.sh"
    body = match.group(1)
    assert "TASK_METRICS_KEYS" in body
    # Drop the script's argv/file tail; we call validate() directly.
    return body.split("source_path = sys.argv[1]", 1)[0]


def _validate(obj: dict) -> None:
    namespace: dict = {}
    exec(_validator_source(), namespace)  # noqa: S102 - the shipped validator
    namespace["validate"](obj)


def _eval_shape() -> dict:
    """Build the shape run_inference writes today: every InferenceResults field set."""
    value = {
        "crps": 0.31,
        "n_positions": 12345,
        "coverage": {"0.1": 0.11, "0.5": 0.49, "0.9": 0.88},
        "median_absolute_error": 0.42,
        "by_signal": {},
    }
    return {
        "task_metrics": {
            "cross_entropy": 1.9,
            "perplexity": 6.7,
            "top1_accuracy": 0.55,
            "top5_accuracy": 0.8,
            "n_predictions": 1000,
            "set_top1_accuracy": 0.8,
            "n_set_predictions": 999,
            "category_set_top1_accuracy": 0.49,
            "n_category_predictions": 500,
        },
        "task_metrics_by_code_type": {},
        "concept_metrics": [],
        "observability_metrics": [],
        "orthogonality": None,
        "n_patient_ends_scored": 100,
        "time_metrics": None,
        "value_metrics": dict(value, by_signal={"creatinine": value}),
        "tail_slice": None,
    }


def test_current_eval_shape_exports() -> None:
    _validate(_eval_shape())


@pytest.mark.skipif(not BANKED.exists(), reason="banked eval JSON not present")
def test_a_banked_full_eval_exports() -> None:
    _validate(json.loads(BANKED.read_text()))


def test_unknown_keys_still_refuse() -> None:
    shape = _eval_shape()
    shape["per_subject_scores"] = [1, 2, 3]
    with pytest.raises(ValueError, match="per_subject_scores"):
        _validate(shape)
    shape = _eval_shape()
    shape["value_metrics"]["subject_ids"] = [1]
    with pytest.raises(ValueError, match="subject_ids"):
        _validate(shape)
    shape = _eval_shape()
    shape["value_metrics"]["by_signal"]["creatinine"]["by_signal"] = {"x": {}}
    with pytest.raises(ValueError, match="one level down"):
        _validate(shape)
