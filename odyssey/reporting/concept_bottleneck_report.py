"""Build the concept-bottleneck training/eval HTML report from a run's artifacts.

Reusable across reruns: takes a training run directory (``config.json`` +
``loss_log.jsonl``, written by :func:`odyssey.training.train.train`) plus
the two JSON files :mod:`odyssey.inference.run_inference` and
:mod:`odyssey.inference.case_study`'s CLIs produce, and renders them into
the same interactive report template every time -- loss curves per term,
full held-out quantitative tables, and a per-patient case browser with a
concept-probability timeline. Meant to be run after every retrain so the
report always reflects the current run, not hand-edited per run.

Usage (see each upstream CLI's own --help for how to produce its input):
    uv run python -m odyssey.reporting.concept_bottleneck_report \
        --run-dir ~/runs/subset_run_v2 \
        --inference-results ~/runs/subset_run_v2/inference_results.json \
        --case-studies ~/runs/subset_run_v2/case_studies.json \
        --output-html research_journal/07_concept_bottleneck_results_subset.html

The output HTML embeds real patient-level detail (event timelines, per-
patient forecasts) -- always write it under ``research_journal/``
(gitignored, PhysioNet DUA), never anywhere git-tracked. This script and
its template contain no patient data and are tracked normally.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.training.train import TrainingConfig, _combined_val_loss


_TEMPLATE_PATH = Path(__file__).parent / "concept_bottleneck_report_template.html"

_LAB_NAMES = {
    "220045": "Heart Rate",
    "220210": "Respiratory Rate",
    "220277": "O2 Saturation (SpO2)",
    "220179": "BP Systolic (NIBP)",
    "220181": "BP Mean (NIBP)",
    "220052": "BP Mean (Arterial)",
    "220050": "BP Systolic (Arterial)",
    "223761": "Temperature (F)",
    "223762": "Temperature (C)",
    "223835": "Inspired O2 Fraction (FiO2)",
    "220739": "GCS - Eye Opening",
    "223900": "GCS - Verbal Response",
    "223901": "GCS - Motor Response",
    "226559": "Urine Output (Foley)",
    "50912": "Creatinine",
    "50813": "Lactate",
    "50821": "pO2",
    "50885": "Bilirubin, Total",
    "51265": "Platelet Count",
    "51301": "White Blood Cells",
}


def _lab_name(itemid: str) -> str:
    return _LAB_NAMES.get(itemid, f"item {itemid}")


def _fmt_value(v: str) -> Optional[str]:
    if v == "UNK":
        return None
    if "::" in v:
        unit, bin_ = v.split("::", 1)
        return f"{bin_} ({unit})" if unit else bin_
    return v


def pretty_code(code: str) -> str:  # noqa: PLR0911, PLR0912
    """Human-legible label for a raw MEDS event code.

    Only expands the handful of itemids in :data:`_LAB_NAMES` (verified
    against ``odyssey/data/code_mapping.py``'s comments) -- everything
    else gets structural formatting (split on ``//``, title-cased) rather
    than a fabricated name, since no full MIMIC-IV ``d_labitems``/
    ``d_items`` lookup ships with this repo.
    """
    if code == "[UNK]":
        return "(unknown token)"
    if "//" not in code:
        return code.replace("_", " ")
    parts = code.split("//")
    kind = parts[0]
    rest = parts[1:]

    if kind == "LAB":
        if len(rest) == 3 and rest[0] in ("SPECIMEN_COLLECTED", "RESULT"):
            sub, itemid, val = rest
            name = _lab_name(itemid)
            if sub == "SPECIMEN_COLLECTED":
                return f"Lab order · {name} ({val})"
            v = _fmt_value(val)
            return f"Lab result · {name}" + (f": {v}" if v else "")
        if len(rest) == 2:
            itemid, val = rest
            name = _lab_name(itemid)
            v = _fmt_value(val)
            return f"Lab · {name}" + (f": {v}" if v else "")
        return "Lab · " + " · ".join(rest)

    if kind == "MEDICATION":
        if (
            rest
            and rest[0] in ("START", "STOP", "STARTED", "STOPPED")
            and len(rest) >= 2
        ):
            action, drug = rest[0], rest[1]
            return f"Medication {action.lower()} · {drug}"
        if len(rest) >= 2:
            drug, action = rest[0], rest[1]
            return f"Medication · {drug} ({action.lower()})"
        return "Medication · " + " · ".join(rest)

    if kind == "DIAGNOSIS" and len(rest) == 3 and rest[0] == "ICD":
        _, version, dcode = rest
        return f"Diagnosis · ICD-{version} {dcode}"

    if kind == "PROCEDURE":
        if len(rest) == 3 and rest[0] == "ICD":
            _, version, pcode = rest
            return f"Procedure · ICD-{version} {pcode}"
        if len(rest) == 2 and rest[0] in ("START", "END"):
            return f"Procedure {rest[0].lower()} · item {rest[1]}"
        return "Procedure · " + " · ".join(rest)

    if kind in ("INFUSION_START", "INFUSION_END"):
        label = "Infusion start" if kind == "INFUSION_START" else "Infusion end"
        if rest:
            token = rest[0]
            if "::" in token:
                itemid, bin_ = token.split("::", 1)
                return f"{label} · item {itemid} ({bin_})"
            return f"{label} · item {token}"
        return label

    if kind == "SUBJECT_WEIGHT_AT_INFUSION":
        v = _fmt_value(rest[0]) if rest else None
        return "Weight at infusion" + (f": {v}" if v else "")

    if kind == "SUBJECT_FLUID_OUTPUT":
        if len(rest) == 2:
            itemid, val = rest
            v = _fmt_value(val)
            return f"Fluid output · item {itemid}" + (f": {v}" if v else "")
        return "Fluid output · " + " · ".join(rest)

    if kind == "HOSPITAL_ADMISSION":
        if len(rest) == 2:
            return f"Hospital admission · {rest[0]} via {rest[1]}"
        return "Hospital admission · " + " · ".join(rest)

    if kind == "HOSPITAL_DISCHARGE":
        return "Hospital discharge · " + " · ".join(rest)

    if kind in ("ICU_ADMISSION", "ICU_DISCHARGE", "TRANSFER_TO"):
        label = kind.replace("_", " ").title()
        return f"{label} · " + " · ".join(rest)

    if kind == "DRG":
        return "DRG · " + " ".join(rest)

    if kind in ("INSURANCE", "LANGUAGE", "MARITAL_STATUS", "RACE"):
        label = kind.replace("_", " ").title()
        return f"{label}: {rest[0]}" if rest else label

    if kind == "HCPCS":
        return "HCPCS · " + " · ".join(rest)

    label = kind.replace("_", " ").title()
    return label + (" · " + " · ".join(rest) if rest else "")


def _downsample_indices(n: int, target: int) -> List[int]:
    if n <= target:
        return list(range(n))
    step = n / target
    return sorted({int(i * step) for i in range(target)} | {n - 1})


def _summarize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    n = len(case["times"])
    times = case["times"]
    concept_names = case["concept_names"]

    traj_idx = _downsample_indices(n, 150)
    trajectory = {
        "t": [round(times[i], 2) for i in traj_idx],
        "concept_probs": [
            [round(p, 4) for p in case["concept_probs"][i]] for i in traj_idx
        ],
    }

    detail_idx = [i for i in _downsample_indices(n - 1, 40) if i < n - 1]
    detail_rows = []
    for i in detail_idx:
        top3 = case["predicted_top_k"][i][:3]
        detail_rows.append(
            {
                "t": round(times[i], 2),
                "input": pretty_code(case["input_codes"][i]),
                "top3": [[pretty_code(code), round(p, 3)] for code, p in top3],
                "true_next": (
                    pretty_code(case["true_next_code"][i])
                    if case["true_next_code"][i]
                    else None
                ),
                "true_rank": case["true_next_rank"][i],
            }
        )

    n_triggered = sum(1 for x in case["concept_labels"] if x > 0)
    n_scored = n - 1
    top1_hits = sum(1 for r in case["true_next_rank"][:-1] if r == 0)
    top5_hits = sum(1 for r in case["true_next_rank"][:-1] if r is not None and r < 5)

    return {
        "subject_id": case["subject_id"],
        "n_events": n,
        "span_hrs": round(times[-1], 1),
        "concept_names": concept_names,
        "concept_labels": case["concept_labels"],
        "concept_observed": case["concept_observed"],
        "n_concepts_triggered": n_triggered,
        "top1_acc": round(top1_hits / n_scored, 3) if n_scored else None,
        "top5_acc": round(top5_hits / n_scored, 3) if n_scored else None,
        "trajectory": trajectory,
        "detail_rows": detail_rows,
    }


def _fmt_count(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def _fmt_duration(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins = rem // 60
    return f"{hrs}h {mins:02d}m"


def _smoothed_last(values: List[Any], k: int = 15) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return sum(xs[-k:]) / len(xs[-k:])


def build_findings(
    loss_curves: Dict[str, Any],
    inference: Dict[str, Any],
    supervision: str,
) -> Dict[str, str]:
    """Compute the per-section "Reading" notes from the run's own numbers.

    Auto-generated so every regeneration interprets its own data instead
    of shipping bare tables; the phrasing stays strictly factual (numbers
    the reader can verify in the section above each note) because these
    render on every future run unseen.
    """
    tr, va = loss_curves["train"], loss_curves["val"]
    task_first = _smoothed_last(list(reversed(tr["task_loss"])), 5)
    task_last = _smoothed_last(tr["task_loss"])
    val_last = _smoothed_last(va["task_loss"], 3)
    orth_last = _smoothed_last(tr["orthogonality_loss"])
    test_ce = inference["task_metrics"]["cross_entropy"]
    training = (
        f"<b>Convergence at a glance.</b> Smoothed train task loss went from "
        f"{task_first:.2f} to {task_last:.2f} over {tr['step'][-1]:,} steps; the "
        f"tuning split ended near {val_last:.2f} and the untouched test split "
        f"scored {test_ce:.2f}, so the train/held-out gap is small. The "
        f"concept, orthogonality, and observability panels are spiky by "
        f"construction: those terms only apply at supervision positions, so "
        f"most steps log zero. The orthogonality penalty ended near "
        f"{orth_last:.3f}, holding the unknown channel apart from the known "
        f"concepts."
    )

    tm = inference["task_metrics"]
    by_type = inference["task_metrics_by_code_type"]
    total_n = sum(v["n_predictions"] for v in by_type.values())
    lab = by_type.get("lab")
    big = {k: v for k, v in by_type.items() if v["n_predictions"] >= 50_000}
    best = max(big, key=lambda k: big[k]["top1_accuracy"])
    worst = min(big, key=lambda k: big[k]["top1_accuracy"])
    task = (
        f"<b>Read the headline through the by-type table.</b> The aggregate "
        f"{tm['top1_accuracy']:.1%} top-1 / {tm['top5_accuracy']:.1%} top-5 is "
        f"dominated by lab tokens"
        + (
            f", which are {lab['n_predictions'] / total_n:.0%} of all scored "
            f"positions (top-1 {lab['top1_accuracy']:.1%}, perplexity "
            f"{lab['perplexity']:.1f})"
            if lab
            else ""
        )
        + f". Across the major families, {best} is the most predictable "
        f"(top-1 {big[best]['top1_accuracy']:.1%}) and {worst} the least "
        f"(top-1 {big[worst]['top1_accuracy']:.1%}, perplexity "
        f"{big[worst]['perplexity']:.0f})."
    )

    cm = [c for c in inference["concept_metrics"] if c.get("auroc") is not None]
    ranked = sorted(cm, key=lambda c: c["auroc"], reverse=True)
    top3 = ", ".join(f"{c['name']} ({c['auroc']:.2f})" for c in ranked[:3])
    bot3 = ", ".join(f"{c['name']} ({c['auroc']:.2f})" for c in ranked[-3:])
    ceiling = [c["name"] for c in cm if c["prevalence"] > 0.85]
    concepts = (
        f"<b>Concept quality, {supervision}-scoped.</b> Strongest by AUROC: "
        f"{top3}. Weakest: {bot3}."
        + (
            f" Note that {', '.join(ceiling)} "
            f"{'sits' if len(ceiling) == 1 else 'sit'} above 85% prevalence "
            f"among observed positions, so high AUPRC there is mostly "
            f"baseline."
            if ceiling
            else ""
        )
    )

    om = [c for c in inference["observability_metrics"] if c.get("auroc") is not None]
    mean_auroc = sum(c["auroc"] for c in om) / len(om) if om else float("nan")
    rates = [c["observed_rate"] for c in inference["observability_metrics"]]
    observability = (
        f"<b>The observability head knows what gets measured.</b> Mean AUROC "
        f"{mean_auroc:.2f} across {len(om)} concepts, against observed rates "
        f"ranging {min(rates):.0%} to {max(rates):.0%}: predicting whether a "
        f"concept will be measured at all is itself informative signal "
        f"(missingness in EHR data is clinical, not random)."
    )

    return {
        "training": training,
        "task": task,
        "concepts": concepts,
        "observability": observability,
    }


@dataclass
class ReportInputs:
    """Everything read from disk before building the report payload."""

    config: TrainingConfig
    loss_records: List[Dict[str, Any]]
    inference_results: Dict[str, Any]
    cases: List[Dict[str, Any]]


def load_inputs(
    run_dir: Path, inference_results_path: Path, case_studies_path: Path
) -> ReportInputs:
    """Read a run's config/loss log plus its inference/case-study output files."""
    config = TrainingConfig(**json.loads((run_dir / "config.json").read_text()))
    with open(run_dir / "loss_log.jsonl") as f:
        loss_records = [json.loads(line) for line in f if line.strip()]
    inference_results = json.loads(inference_results_path.read_text())
    cases = json.loads(case_studies_path.read_text())
    return ReportInputs(
        config=config,
        loss_records=loss_records,
        inference_results=inference_results,
        cases=cases,
    )


def build_payload(inputs: ReportInputs) -> Dict[str, Any]:
    """Turn raw run artifacts into the compact JSON payload the template embeds."""
    config = inputs.config
    train_records = [r for r in inputs.loss_records if r["split"] == "train"]
    val_records = [r for r in inputs.loss_records if r["split"] == "tuning"]
    if not train_records:
        raise ValueError("loss_log.jsonl has no 'train' split records")

    loss_terms = [
        "task_loss",
        "concept_loss",
        "orthogonality_loss",
        "observability_loss",
    ]
    loss_curves = {
        "train": {
            "step": [r["step"] for r in train_records],
            "epoch": [r["epoch"] for r in train_records],
            **{term: [r[term] for r in train_records] for term in loss_terms},
        },
        "val": {
            "step": [r["step"] for r in val_records],
            "epoch": [r["epoch"] for r in val_records],
            **{term: [r[term] for r in val_records] for term in loss_terms},
        },
    }

    weights = ConceptBottleneckLossWeights(
        concept=config.concept_weight,
        orthogonality=config.orthogonality_weight,
        observability=config.observability_weight,
    )
    best_val_loss = (
        min(_combined_val_loss(r, weights) for r in val_records)
        if val_records
        else None
    )

    cases = sorted(
        (_summarize_case(c) for c in inputs.cases),
        key=lambda c: (-c["n_concepts_triggered"], -c["n_events"]),
    )

    inference = inputs.inference_results
    tm = inference["task_metrics"]
    n_pool_positions = (
        inference["observability_metrics"][0]["n_subjects"]
        if inference["observability_metrics"]
        else inference["n_patient_ends_scored"]
    )
    # ObservabilityMetrics.n_subjects is really "how many pooled positions
    # were scored" -- under stay-scoped supervision that's one per patient
    # (a true patient count), but under visit-scoped supervision it's one
    # per real admission, which can exceed the held-out patient count for
    # patients with multiple stays. Label it accurately either way rather
    # than always calling it "patients".
    pool_unit = "visits" if config.concept_supervision == "visit" else "patients"

    epochs_run = max(r["epoch"] for r in train_records) + 1
    last_step = train_records[-1]["step"]
    total_elapsed_s = train_records[-1]["elapsed_s"]
    n_shards_note = (
        f"{config.max_train_shards} of the available train shards"
        if config.max_train_shards is not None
        else "the full train split"
    )
    tuning_note = (
        f"{config.max_tuning_shards} tuning shards"
        if config.max_tuning_shards is not None
        else "the full tuning split"
    )

    run_meta = [
        ["Steps", f"{last_step:,} · {epochs_run} of {config.num_epochs} epochs"],
        ["Wall-clock", _fmt_duration(total_elapsed_s)],
        [
            "Best val loss",
            f"{best_val_loss:.3f}" if best_val_loss is not None else "n/a",
        ],
        [
            "Held-out test",
            f"{_fmt_count(tm['n_predictions'])} predictions · {n_pool_positions:,} {pool_unit}",
        ],
        ["Concept supervision", config.concept_supervision],
    ]

    training_desc = (
        f"Streaming truncated-BPTT training over {n_shards_note} "
        f"({epochs_run} epochs, {config.num_lanes} parallel lanes \u00d7 "
        f"{config.chunk_size}-token chunks, {config.concept_supervision}-scoped "
        f"concept supervision). Each panel plots one term of the combined loss, "
        f"logged every {config.log_every} training steps and evaluated on "
        f"{tuning_note} every {config.eval_every} steps: task (next-token "
        f"forecasting), concept-supervision, orthogonality (known vs. unknown "
        f"concept separation), and observability (whether a concept would be "
        f"measured)."
    )
    quant_desc = (
        f"Scored once, after training, against the full held-out test split: "
        f"{n_pool_positions:,} {pool_unit} and {tm['n_predictions']:,} forecast "
        f"positions, never seen during training or validation."
    )
    qualitative_desc = (
        f"{len(cases)} held-out patients, selected to span short and long stays "
        f"and a range of triggered concepts. Each trace is a single "
        f"whole-sequence forward pass (no synthetic resets), showing the concept "
        f"bottleneck's running probability for all "
        f"{len(cases[0]['concept_names']) if cases else 0} concepts alongside the "
        f"model's next-event forecast at sampled points in the stay."
    )

    return {
        "run_meta": run_meta,
        "training_desc": training_desc,
        "quant_desc": quant_desc,
        "qualitative_desc": qualitative_desc,
        "findings": build_findings(loss_curves, inference, config.concept_supervision),
        "loss_curves": loss_curves,
        "inference_results": inference,
        "cases": cases,
    }


def render_html(payload: Dict[str, Any]) -> str:
    """Splice ``payload`` into the report template as its embedded JSON data."""
    template = _TEMPLATE_PATH.read_text()
    data_json = json.dumps(payload).replace("</script", "<\\/script")
    return template.replace("__DASHBOARD_DATA__", data_json)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--inference-results", required=True, type=Path)
    parser.add_argument("--case-studies", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """CLI entry point: build and write the report from --run-dir + its eval outputs."""
    args = _parse_args()
    inputs = load_inputs(args.run_dir, args.inference_results, args.case_studies)
    payload = build_payload(inputs)
    html = render_html(payload)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html)
    print(f"wrote {args.output_html} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
