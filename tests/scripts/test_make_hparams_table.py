"""The hyperparameter table renders the banked configs and the GBM's code settings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import make_hparams_table
from scripts.make_hparams_table import NOT_EXPORTED, TBD, gbm_rows, render, run_name


REPO = Path(__file__).resolve().parents[2]
BANKED = {
    "MIMIC-IV": REPO / "research_journal/figure_data/vm1/full_run_v10/config.json",
    "eICU-CRD": REPO / "research_journal/figure_data/vm2/eicu_full_v10/config.json",
}

# The values the two flagship configs carry (research_journal is gitignored,
# so a clean checkout falls back to this copy of the exported keys).
_FLAGSHIP: dict[str, Any] = {
    "output_dir": "/home/amritkrishnan/runs/full_run_v10",
    "model_kind": "bottleneck",
    "backbone": "hybrid",
    "max_context": 4096,
    "hidden_size": 256,
    "num_hidden_layers": 8,
    "mamba_state_size": 128,
    "mamba_headdim": 64,
    "mamba_chunk_size": 256,
    "attn_num_heads": 8,
    "embedding_dim": 32,
    "vocab_min_count": 5,
    "vocab_max_size": 20000,
    "vocab_backoff": "icd3",
    "quantile_n_bins": 5,
    "quantile_min_count": 100,
    "num_lanes": 64,
    "chunk_size": 512,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "num_epochs": 2,
    "concept_weight": 1.0,
    "orthogonality_weight": 0.1,
    "observability_weight": 0.1,
    "task_weight": 1.0,
    "checkpoint_every": 2000,
    "time_weight": 1.0,
    "event_hazard_weight": 1.0,
    "randint_prob": 0.0,
    "early_stopping_patience": 15,
    "seed": 0,
}


def _config_paths(tmp_path: Path) -> dict[str, Path]:
    """Use the real banked configs when present, else fixtures with their values."""
    out: dict[str, Path] = {}
    for label, banked in BANKED.items():
        if banked.exists():
            out[label] = banked
            continue
        path = tmp_path / f"{label}.json"
        cfg = dict(_FLAGSHIP)
        if label == "eICU-CRD":
            cfg["output_dir"] = "/home/amritkrishnan/runs/eicu_full_v10"
        path.write_text(json.dumps(cfg))
        out[label] = path
    return out


def _rows(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for raw in text.splitlines():
        if not raw.endswith("\\\\") or "&" not in raw:
            continue
        label, *cells = raw[: -len(" \\\\")].split(" & ")
        rows[label] = cells
    return rows


def test_model_and_training_rows_carry_the_banked_values(tmp_path: Path) -> None:
    paths = _config_paths(tmp_path)
    runs = [(label, json.loads(p.read_text())) for label, p in paths.items()]
    rows = _rows(render([*runs, ("GEMINI", None)]))

    assert rows["Banked config"] == ["yes", "yes", NOT_EXPORTED]
    assert rows["Backbone"][:2] == ["hybrid Mamba-2 + chunk attention"] * 2
    assert rows["Hidden size"] == ["256", "256", "--"]
    assert rows["Layers"][0] == "8"
    assert rows["Attention heads"][0] == "8"
    assert rows["Mamba state size"][0] == "128"
    assert rows["Mamba head dim"][0] == "64"
    assert rows["Mamba chunk size"][0] == "256"
    assert rows["Concept embedding dim"][0] == "32"
    assert rows["Lanes $\\times$ chunk (tokens)"][0] == "64 $\\times$ 512"
    assert rows["Max context (tokens)"][0] == "4{,}096"
    assert rows["Optimizer"] == ["AdamW (constant LR)"] * 2 + ["--"]
    assert rows["Learning rate"][0] == "$3\\times10^{-4}$"
    assert rows["Weight decay"][0] == "0.01"
    assert rows["Gradient clip (norm)"][0] == "1"
    assert rows["Epochs"][0] == "2"
    assert rows["Early-stopping patience (evals)"][0] == "15"
    assert rows["Checkpoint every (steps)"][0] == "2{,}000"
    assert rows["Seed"][0] == "0"
    assert rows["RandInt probability"][0] == "0"
    assert rows["Concept"][0] == "1"
    assert rows["Orthogonality"][0] == "0.1"
    assert rows["Observability"][0] == "0.1"
    assert rows["Task (next token)"][0] == "1"
    assert rows["Time to event"][0] == "1"
    assert rows["Event hazard"][0] == "1"
    assert rows["Vocabulary min count"][0] == "5"
    assert rows["Vocabulary max size"][0] == "20{,}000"
    assert rows["Vocabulary backoff"][0] == "icd3"
    assert rows["Quantile bins per lab"][0] == "5"
    assert rows["Parameters"] == [TBD, TBD, TBD]


def test_parameter_counts_key_by_run_name_or_label(tmp_path: Path) -> None:
    paths = _config_paths(tmp_path)
    runs = [(label, json.loads(p.read_text())) for label, p in paths.items()]
    assert run_name(runs[0][1], "MIMIC-IV") == "full_run_v10"
    assert run_name(None, "GEMINI") == "GEMINI"
    rows = _rows(
        render(
            [*runs, ("GEMINI", None)],
            {"full_run_v10": 12_345_678, "GEMINI": 9_000_000},
        )
    )
    assert rows["Parameters"] == ["12{,}345{,}678", TBD, "9{,}000{,}000"]


def test_gbm_block_reads_the_code_not_a_config() -> None:
    rows = dict(gbm_rows())
    assert "HistGradientBoostingClassifier" in rows["Estimator"]
    assert (
        rows["Search grid (LR, max leaves, min leaf)"]
        == "(0.05, 31, 20); (0.05, 63, 100); (0.1, 15, 20); (0.1, 63, 100)"
    )
    assert rows["Boosting rounds"].startswith("up to 400")
    assert "subject-grouped" in rows["Validation"]
    assert "200{,}000 rows" in rows["Validation"]
    assert rows["Feature panel"] == "609 features, 110 counts"


def test_main_writes_the_table_with_one_column_per_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _config_paths(tmp_path)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"eicu_full_v10": 1_000_000}))
    out = tmp_path / "tables" / "hparams.tex"
    argv = ["make_hparams_table"]
    for label, path in paths.items():
        argv += ["--run", label, str(path)]
    argv += ["--missing", "GEMINI", "--params", str(params), "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    make_hparams_table.main()

    text = out.read_text()
    assert text.startswith("% GENERATED by scripts/make_hparams_table.py")
    assert "\\begin{tabular}{@{}lrrr@{}}" in text
    assert "Setting & MIMIC-IV & eICU-CRD & GEMINI \\\\" in text
    assert "\\toprule" in text and "\\bottomrule" in text
    assert "\\multicolumn{3}{l}{scikit-learn" in text
    rows = _rows(text)
    assert rows["Parameters"] == [TBD, "1{,}000{,}000", TBD]
    # the paper rule: no em dashes anywhere in a generated table
    assert chr(0x2014) not in text  # em dash


def test_main_refuses_an_empty_column_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["make_hparams_table", "--output", str(tmp_path / "x.tex")]
    )
    with pytest.raises(SystemExit, match="at least one"):
        make_hparams_table.main()
