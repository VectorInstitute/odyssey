"""Model size and training settings of the flagship runs, from banked configs.

The ML4H submission gives no hyperparameters: no hidden size, no
optimizer, no learning rate, no loss weights, no parameter count, and
no description of the GBM comparator's search. This script makes that
appendix table a build product of the banked ``config.json`` each
flagship training run wrote, so the numbers cannot drift from what was
actually trained.

One column per database. A run whose ``config.json`` was never exported
(GEMINI's ``gemini_full_v10_15c`` only banked its evaluation files)
gets a column of ``--`` under a "not exported" marker rather than being
silently dropped or, worse, filled in from another database's config.

The GBM block below the model rows comes from the code, not from a
config: the estimator class, the four configurations searched, the
round budget and the validation scheme are read from
``odyssey.inference.alerts`` (``GBM_GRID``, ``GBM_MAX_ITER``,
``_tune_gbm``), and the panel size and its count-feature share are
computed from ``odyssey.inference.baseline_features.feature_names`` and
the ablation's ``feature_groups`` partition.

Parameter counts are not in the configs; pass ``--params`` with a JSON
file mapping run name (the basename of the config's ``output_dir``, or
the column label) to an integer. Missing runs print ``tbd``.

Usage::

    uv run python scripts/make_hparams_table.py \\
        --run MIMIC-IV research_journal/figure_data/vm1/full_run_v10/config.json \\
        --run eICU-CRD research_journal/figure_data/vm2/eicu_full_v10/config.json \\
        --missing GEMINI \\
        --params research_journal/figure_data/param_counts.json \\
        --output paper/ml4h/tables/hparams.tex
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from odyssey.inference.alerts import GBM_GRID, GBM_MAX_ITER, GBM_TUNE_MAX_ROWS
from odyssey.inference.baseline_features import feature_names
from scripts.gbm_feature_ablation import feature_groups


logger = logging.getLogger(__name__)

#: Marker for a column whose config was never exported.
NOT_EXPORTED = "not exported"
#: Marker for a parameter count that has not been computed yet.
TBD = "tbd"

_BACKBONES = {
    "hybrid": "hybrid Mamba-2 + chunk attention",
    "mamba": "Mamba-2",
    "transformer": "transformer",
}


def _num(value: Any) -> str:
    """Numbers the way the other tables print them: no trailing zeros."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}".replace(",", "{,}")
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-2:
            mantissa, exponent = f"{value:.0e}".split("e")
            return f"${mantissa}\\times10^{{{int(exponent)}}}$"
        return f"{value:g}"
    return str(value)


def _key(name: str) -> Callable[[dict[str, Any]], str]:
    def get(config: dict[str, Any]) -> str:
        return _num(config[name])

    return get


def _backbone(config: dict[str, Any]) -> str:
    backbone = str(config["backbone"])
    return _BACKBONES.get(backbone, backbone)


def _lanes(config: dict[str, Any]) -> str:
    return f"{_num(config['num_lanes'])} $\\times$ {_num(config['chunk_size'])}"


def _lr(config: dict[str, Any]) -> str:
    return _num(float(config["learning_rate"]))


#: Model rows: (label, reader). Every reader takes the config dict.
MODEL_ROWS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("Backbone", _backbone),
    ("Hidden size", _key("hidden_size")),
    ("Layers", _key("num_hidden_layers")),
    ("Attention heads", _key("attn_num_heads")),
    ("Mamba state size", _key("mamba_state_size")),
    ("Mamba head dim", _key("mamba_headdim")),
    ("Mamba chunk size", _key("mamba_chunk_size")),
    ("Concept embedding dim", _key("embedding_dim")),
    ("Lanes $\\times$ chunk (tokens)", _lanes),
    ("Max context (tokens)", _key("max_context")),
)

#: Training rows. AdamW with torch defaults and a constant learning rate:
#: odyssey/training/train.py builds ``torch.optim.AdamW`` from
#: ``optimizer_param_groups`` and no scheduler.
TRAINING_ROWS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("Optimizer", lambda _: "AdamW (constant LR)"),
    ("Learning rate", _lr),
    ("Weight decay", _key("weight_decay")),
    ("Gradient clip (norm)", _key("grad_clip_norm")),
    ("Epochs", _key("num_epochs")),
    ("Early-stopping patience (evals)", _key("early_stopping_patience")),
    ("Checkpoint every (steps)", _key("checkpoint_every")),
    ("Seed", _key("seed")),
    ("RandInt probability", _key("randint_prob")),
)

LOSS_ROWS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("Concept", _key("concept_weight")),
    ("Orthogonality", _key("orthogonality_weight")),
    ("Observability", _key("observability_weight")),
    ("Task (next token)", _key("task_weight")),
    ("Time to event", _key("time_weight")),
    ("Event hazard", _key("event_hazard_weight")),
)

VOCAB_ROWS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("Vocabulary min count", _key("vocab_min_count")),
    ("Vocabulary max size", _key("vocab_max_size")),
    ("Vocabulary backoff", _key("vocab_backoff")),
    ("Quantile bins per lab", _key("quantile_n_bins")),
    ("Quantile min count", _key("quantile_min_count")),
)


def run_name(config: dict[str, Any] | None, label: str) -> str:
    """Return the run's directory name, which is how ``--params`` keys it."""
    if config is None or not config.get("output_dir"):
        return label
    return Path(str(config["output_dir"])).name


def _param_cell(
    config: dict[str, Any] | None, label: str, params: dict[str, int]
) -> str:
    for key in (run_name(config, label), label):
        if key in params:
            return _num(int(params[key]))
    return TBD


def gbm_rows() -> list[tuple[str, str]]:
    """Read the GBM comparator's settings from the code that fits it."""
    names = feature_names()
    n_counts = len(feature_groups(names)["counts_occurrence"])
    grid = "; ".join(
        f"({_num(p['learning_rate'])}, {_num(int(p['max_leaf_nodes']))}, "
        f"{_num(int(p['min_samples_leaf']))})"
        for p in GBM_GRID
    )
    return [
        ("Estimator", "scikit-learn \\texttt{HistGradientBoostingClassifier}"),
        (
            "Search grid (LR, max leaves, min leaf)",
            grid,
        ),
        (
            "Boosting rounds",
            f"up to {_num(GBM_MAX_ITER)}; best round by validation log loss, "
            "then refit at that count",
        ),
        (
            "Validation",
            "10\\% of training subjects held out (subject-grouped), "
            f"tuned on at most {_num(GBM_TUNE_MAX_ROWS)} rows",
        ),
        ("Feature panel", f"{_num(len(names))} features, {_num(n_counts)} counts"),
        ("Missing values", "native (columns observed in $<$200 rows filled with 0)"),
    ]


def render(
    runs: list[tuple[str, dict[str, Any] | None]],
    params: dict[str, int] | None = None,
) -> str:
    """One row per setting, one column per database, GBM block at the end."""
    params = params or {}
    n = len(runs)
    labels = [label for label, _ in runs]

    def line(label: str, cells: list[str]) -> str:
        return f"{label} & " + " & ".join(cells) + " \\\\"

    def row(
        label: str, reader: Callable[[dict[str, Any]], str], missing: str = "--"
    ) -> str:
        return line(label, [missing if cfg is None else reader(cfg) for _, cfg in runs])

    def block(
        title: str, rows: tuple[tuple[str, Callable[..., str]], ...]
    ) -> list[str]:
        return [
            f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\",
            *(row(label, reader) for label, reader in rows),
        ]

    lines = [
        "% GENERATED by scripts/make_hparams_table.py -- do not hand-edit.",
        "% Model and training settings from each flagship run's banked",
        "% config.json. '--': that run's config.json was not exported.",
        "% Parameter counts come from --params; 'tbd' means not computed yet.",
        "% GBM block: odyssey/inference/alerts.py (GBM_GRID, GBM_MAX_ITER,",
        "% GBM_TUNE_MAX_ROWS, _tune_gbm, BaselineModel) and the panel from",
        "% odyssey/inference/baseline_features.py (feature_names) partitioned",
        "% by scripts/gbm_feature_ablation.py (feature_groups).",
        "\\begin{tabular}{@{}l" + "r" * n + "@{}}",
        "\\toprule",
        "Setting & " + " & ".join(labels) + " \\\\",
        "\\midrule",
        row("Banked config", lambda _: "yes", missing=NOT_EXPORTED),
        *block("Model", MODEL_ROWS),
        line("Parameters", [_param_cell(cfg, label, params) for label, cfg in runs]),
        *block("Training", TRAINING_ROWS),
        *block("Loss weights", LOSS_ROWS),
        *block("Vocabulary and value bins", VOCAB_ROWS),
        "\\midrule",
        f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\emph{{GBM comparator "
        "(same on every database)}} \\\\",
    ]
    for label, value in gbm_rows():
        lines.append(f"{label} & \\multicolumn{{{n}}}{{l}}{{{value}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def load_config(path: Path) -> dict[str, Any]:
    """Load the training config a run banked, as a plain dict."""
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return config


def main() -> None:
    """Write the hyperparameter table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "CONFIG_JSON"),
        help="column label and the run's banked config.json (repeatable)",
    )
    parser.add_argument(
        "--missing",
        action="append",
        default=[],
        metavar="LABEL",
        help="column for a run whose config.json was not exported (repeatable)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        help="JSON mapping run name (output_dir basename or label) to parameter count",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.run and not args.missing:
        raise SystemExit("give at least one --run or --missing column")
    runs: list[tuple[str, dict[str, Any] | None]] = [
        (label, load_config(Path(path))) for label, path in args.run
    ]
    runs += [(label, None) for label in args.missing]
    params: dict[str, int] = {}
    if args.params is not None:
        params = {k: int(v) for k, v in json.loads(args.params.read_text()).items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(runs, params))
    logger.info("wrote %s (%d columns)", args.output, len(runs))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
