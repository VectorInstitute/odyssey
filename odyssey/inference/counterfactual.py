"""Input-level counterfactual forecasts: edit the observations, re-score (item 12).

The concept-level lever (forcing a bottleneck probability) moved forecasts
in proportion to surprise, not truth (research journal 23/25). The
clinician's actual what-if is one level down: *"assume this patient's
blood pressure had been 80 for the last six hours -- what does the model
now expect?"* That is an edit to the record itself, needs no special
training, and every family can be asked the same question. This module
implements it for the sequence model: take one subject's raw MEDS events,
rewrite the numeric values of a named signal inside a window before an
index time (set / add / scale / remove), re-bin with the run's own binner,
re-tokenize, stream both the factual and the counterfactual record through
the frozen model, and read both forecasts at the index position:

- per alert event, the hazard head's ``P(event within h)``;
- the bottleneck's concept probabilities (if the model has one);
- the next-event distribution's top-k.

A cohort wrapper applies the same edit to many subjects at a fixed point
into a visit and summarizes the shift (mean delta, fraction moving in the
clinically expected direction when one is declared). Deterministic
re-scoring, not sampling: this measures how the model's *current forecast*
responds to a changed record -- the honest precondition for any rollout
story. Sampling full futures (autoregressive rollouts) is a separate step.

Edits are expressed in the source's own units (MIMIC-IV SBP mmHg,
creatinine mg/dL); signals are named by the panel
(:data:`odyssey.data.signal_panel.SIGNAL_PANEL`) and resolved to code
prefixes through the LOINC tables.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl
import torch

from odyssey.data.code_mapping import prefixes_for_loinc
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sequences import BIRTH_CODE, build_patient_sequence
from odyssey.data.signal_panel import SIGNAL_PANEL
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.models.sequence_model import SequenceModel
from odyssey.models.time_to_event import probability_within
from odyssey.training.data import load_meds_shards
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

HORIZONS_HOURS: tuple[float, ...] = (8.0, 24.0, 72.0)
EditMode = Literal["set", "add", "scale", "remove"]
_PANEL_LOINC = dict(SIGNAL_PANEL)


@dataclass(frozen=True)
class ValueEdit:
    """One counterfactual edit of a signal's readings before the index time."""

    signal: str
    """Panel signal name (e.g. ``sbp_noninvasive``, ``creatinine``) or a raw
    code prefix (anything containing ``//``)."""
    mode: EditMode
    value: float = 0.0
    """``set``: new reading; ``add``: offset; ``scale``: multiplier;
    ``remove``: drop the readings (value unused)."""
    window_hours: float | None = 6.0
    """Only readings within this many hours before the index time are
    edited; ``None`` = every reading before it."""
    expected_direction: dict[str, int] = field(default_factory=dict)
    """Optional: event name -> +1/-1, the clinically expected sign of the
    hazard shift (e.g. hypotension edit -> vasopressor_start +1). Used
    for the sign-agreement summary only."""

    def prefixes(self, source: str) -> list[str]:
        """Code prefixes this edit touches in ``source``."""
        if "//" in self.signal:
            return [self.signal]
        loinc = _PANEL_LOINC.get(self.signal)
        if loinc is None:
            raise ValueError(
                f"unknown signal {self.signal!r}; panel names: {sorted(_PANEL_LOINC)}"
            )
        return sorted(prefixes_for_loinc(loinc, source=source))

    @classmethod
    def parse(cls, spec: str) -> "ValueEdit":
        """Parse ``signal:mode[:value[:window_hours]]`` (window ``all`` = None)."""
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"edit spec {spec!r}: need at least signal:mode")
        signal, mode = parts[0], parts[1]
        if mode not in ("set", "add", "scale", "remove"):
            raise ValueError(f"edit spec {spec!r}: unknown mode {mode!r}")
        value = float(parts[2]) if len(parts) > 2 and parts[2] != "" else 0.0
        window: float | None = 6.0
        if len(parts) > 3:
            window = None if parts[3] == "all" else float(parts[3])
        return cls(signal=signal, mode=mode, value=value, window_hours=window)  # type: ignore[arg-type]


def apply_value_edits(
    raw_events: pl.DataFrame,
    edits: Sequence[ValueEdit],
    *,
    index_time: "pl.datatypes.Datetime | object",
    source: str = "mimic_iv",
    code_col: str = "code",
    value_col: str = "numeric_value",
    time_col: str = "time",
) -> tuple[pl.DataFrame, int]:
    """Return the edited raw events (one subject) and how many rows were touched.

    Only rows strictly before or at ``index_time`` and inside each edit's
    window are eligible; rows after the index time are left alone (they are
    not visible at the index position anyway).
    """
    out = raw_events
    touched = 0
    for edit in edits:
        prefix_expr = pl.lit(False)
        for p in edit.prefixes(source):
            prefix_expr = prefix_expr | pl.col(code_col).str.starts_with(p)
        in_window = pl.col(time_col) <= pl.lit(index_time)
        if edit.window_hours is not None:
            in_window = in_window & (
                pl.col(time_col)
                > pl.lit(index_time) - pl.duration(hours=edit.window_hours)
            )
        hit = prefix_expr & in_window & pl.col(value_col).is_not_null()
        n_hit = int(out.select(hit.sum()).item())
        touched += n_hit
        if edit.mode == "remove":
            out = out.filter(~hit)
            continue
        if edit.mode == "set":
            new_value = pl.lit(edit.value)
        elif edit.mode == "add":
            new_value = pl.col(value_col) + edit.value
        else:
            new_value = pl.col(value_col) * edit.value
        out = out.with_columns(
            pl.when(hit)
            .then(new_value.cast(out.schema[value_col]))
            .otherwise(pl.col(value_col))
            .alias(value_col)
        )
    return out, touched


@dataclass
class ForecastReadout:
    """What the model forecasts at one position of one record."""

    subject_id: int
    index_time_hours: float
    position: int
    event_risk: dict[str, dict[str, float]]
    """event -> {"8h": p, "24h": p, "72h": p}."""
    concept_probs: dict[str, float]
    top_next: list[tuple[str, float]]


@torch.no_grad()
def score_record_at(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    *,
    index_time: object,
    concept_names: Sequence[str],
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
    top_k: int = 5,
    horizons: Sequence[float] = HORIZONS_HOURS,
) -> ForecastReadout:
    """Stream one record through the frozen model; read the forecast at ``index_time``.

    The index position is the last token at or before ``index_time``.
    Everything after it is still tokenized (the record is the record) but
    only positions up to the index are ever needed, so streaming stops once
    the index position has been scored.
    """
    model.eval()
    binned = add_value_tokens(raw_subject_events, binner, source=source)
    seq = build_patient_sequence(binned, vocab)
    if len(seq) == 0:
        raise ValueError("empty tokenized record")
    timed = binned.filter(pl.col("time").is_not_null() & (pl.col("code") != BIRTH_CODE))
    origin = timed["time"].min()
    index_hours = float(
        (
            pl.Series([index_time]).cast(pl.Datetime("us"))
            - pl.Series([origin]).cast(pl.Datetime("us"))
        ).dt.total_seconds()[0]
        / 3600.0
    )
    positions = [i for i, t in enumerate(seq.time_stamps) if t <= index_hours + 1e-9]
    if not positions:
        raise ValueError("index_time precedes the record's first event")
    index_pos = positions[-1]

    sampler = PackedLaneSampler(
        iter([seq]), num_lanes=1, chunk_size=chunk_size, reset_prob=0.0
    )
    event_heads = getattr(model, "event_heads", None)
    state = None
    offset = 0
    for chunk in sampler:
        chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
        fwd = model.forward_with_features(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        state = fwd.state
        n_real = int((chunk.subject_ids[0] != NO_SUBJECT).sum().item())
        if offset + n_real <= index_pos:
            offset += n_real
            continue
        i = index_pos - offset
        probs = torch.softmax(fwd.logits[0, i], dim=-1)
        top_p, top_i = probs.topk(min(top_k, probs.numel()))
        top_next = [
            (vocab.decode(int(t)), float(p))
            for t, p in zip(top_i.tolist(), top_p.tolist())
        ]
        risk: dict[str, dict[str, float]] = {}
        if event_heads is not None:
            hz = event_heads(fwd.features[0, i : i + 1])  # (1, E, B)
            for e_idx, name in enumerate(event_heads.event_names):
                risk[name] = {
                    f"{h:g}h": float(
                        probability_within(hz[:, e_idx], event_heads.edges, h)[0]
                    )
                    for h in horizons
                }
        concepts: dict[str, float] = {}
        if fwd.bottleneck is not None:
            cp = fwd.bottleneck.concept_probs[0, i].tolist()
            concepts = {name: float(p) for name, p in zip(concept_names, cp)}
        return ForecastReadout(
            subject_id=seq.subject_id,
            index_time_hours=index_hours,
            position=index_pos,
            event_risk=risk,
            concept_probs=concepts,
            top_next=top_next,
        )
    raise RuntimeError("index position was never streamed (internal error)")


@dataclass
class CounterfactualResult:
    """Factual vs counterfactual forecast for one subject and one edit set."""

    subject_id: int
    edits: list[dict[str, object]]
    rows_edited: int
    factual: ForecastReadout
    counterfactual: ForecastReadout
    delta_event_risk: dict[str, dict[str, float]]
    delta_concepts: dict[str, float]


def counterfactual_forecast(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_subject_events: pl.DataFrame,
    edits: Sequence[ValueEdit],
    *,
    index_time: object,
    concept_names: Sequence[str],
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
) -> CounterfactualResult:
    """Score the factual record and its edited twin at the same index time."""
    factual = score_record_at(
        model,
        vocab,
        binner,
        raw_subject_events,
        index_time=index_time,
        concept_names=concept_names,
        source=source,
        device=device,
        chunk_size=chunk_size,
    )
    edited, touched = apply_value_edits(
        raw_subject_events, edits, index_time=index_time, source=source
    )
    cf = score_record_at(
        model,
        vocab,
        binner,
        edited,
        index_time=index_time,
        concept_names=concept_names,
        source=source,
        device=device,
        chunk_size=chunk_size,
    )
    delta_risk = {
        ev: {h: cf.event_risk[ev][h] - p for h, p in hs.items()}
        for ev, hs in factual.event_risk.items()
        if ev in cf.event_risk
    }
    delta_c = {
        name: cf.concept_probs.get(name, 0.0) - p
        for name, p in factual.concept_probs.items()
    }
    return CounterfactualResult(
        subject_id=factual.subject_id,
        edits=[asdict(e) for e in edits],
        rows_edited=touched,
        factual=factual,
        counterfactual=cf,
        delta_event_risk=delta_risk,
        delta_concepts=delta_c,
    )


# ---------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------


def _index_times_by_subject(
    raw_events: pl.DataFrame, *, index_hours: float
) -> dict[int, object]:
    """Return subject -> last event time at or before ``index_hours`` into a visit.

    The first visit of each subject that lasts at least ``index_hours``.
    """
    timed = raw_events.filter(
        pl.col("time").is_not_null() & pl.col("hadm_id").is_not_null()
    )
    visits = (
        timed.group_by("subject_id", "hadm_id")
        .agg(pl.col("time").min().alias("_start"), pl.col("time").max().alias("_end"))
        .filter((pl.col("_end") - pl.col("_start")) >= pl.duration(hours=index_hours))
        .sort(["subject_id", "_start"])
        .unique(subset=["subject_id"], keep="first", maintain_order=True)
        .with_columns((pl.col("_start") + pl.duration(hours=index_hours)).alias("_cut"))
    )
    at = (
        timed.join(
            visits.select("subject_id", "hadm_id", "_cut"),
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        .filter(pl.col("time") <= pl.col("_cut"))
        .group_by("subject_id")
        .agg(pl.col("time").max().alias("_index"))
    )
    return dict(zip(at["subject_id"].to_list(), at["_index"].to_list()))


@dataclass
class CohortSummary:
    """Mean shift and sign agreement of one edit set over a cohort."""

    edits: list[dict[str, object]]
    n_subjects: int
    n_edited: int
    """Subjects where at least one reading was actually edited."""
    mean_delta_event_risk: dict[str, dict[str, float]]
    sign_agreement: dict[str, dict[str, float]]
    """event -> horizon -> fraction of edited subjects whose shift has the
    declared expected sign (only events with an expectation)."""
    mean_delta_concepts: dict[str, float]
    per_subject: list[CounterfactualResult] = field(default_factory=list)


def cohort_counterfactual(
    model: SequenceModel,
    vocab: Vocabulary,
    binner: QuantileBinner | None,
    raw_events: pl.DataFrame,
    edits: Sequence[ValueEdit],
    *,
    concept_names: Sequence[str],
    index_hours: float = 24.0,
    max_subjects: int = 200,
    source: str = "mimic_iv",
    device: str = "cpu",
    chunk_size: int = 256,
    keep_per_subject: bool = False,
    seed: int = 0,
) -> CohortSummary:
    """Apply ``edits`` to a cohort of subjects at a fixed point into a visit."""
    index_times = _index_times_by_subject(raw_events, index_hours=index_hours)
    subjects = sorted(index_times)
    if len(subjects) > max_subjects:
        gen = torch.Generator().manual_seed(seed)
        pick = torch.randperm(len(subjects), generator=gen)[:max_subjects].tolist()
        subjects = sorted(subjects[i] for i in pick)
    results: list[CounterfactualResult] = []
    for sid in subjects:
        sub = raw_events.filter(pl.col("subject_id") == sid)
        res = counterfactual_forecast(
            model,
            vocab,
            binner,
            sub,
            edits,
            index_time=index_times[sid],
            concept_names=concept_names,
            source=source,
            device=device,
            chunk_size=chunk_size,
        )
        results.append(res)
    edited = [r for r in results if r.rows_edited > 0]
    events = sorted({ev for r in edited for ev in r.delta_event_risk})
    mean_delta: dict[str, dict[str, float]] = {}
    agree: dict[str, dict[str, float]] = {}
    expected: dict[str, int] = {}
    for e in edits:
        expected.update(e.expected_direction)
    for ev in events:
        horizons = sorted({h for r in edited for h in r.delta_event_risk.get(ev, {})})
        mean_delta[ev] = {}
        for h in horizons:
            ds = [
                r.delta_event_risk[ev][h]
                for r in edited
                if h in r.delta_event_risk.get(ev, {})
            ]
            mean_delta[ev][h] = sum(ds) / len(ds) if ds else 0.0
            if ev in expected and ds:
                sign = expected[ev]
                agree.setdefault(ev, {})[h] = sum(1 for d in ds if d * sign > 0) / len(
                    ds
                )
    concept_keys = sorted({c for r in edited for c in r.delta_concepts})
    mean_c = (
        {
            c: sum(r.delta_concepts[c] for r in edited) / len(edited)
            for c in concept_keys
        }
        if edited
        else {}
    )
    return CohortSummary(
        edits=[asdict(e) for e in edits],
        n_subjects=len(results),
        n_edited=len(edited),
        mean_delta_event_risk=mean_delta,
        sign_agreement=agree,
        mean_delta_concepts=mean_c,
        per_subject=results if keep_per_subject else [],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Standard what-ifs with their clinically expected hazard directions.
STANDARD_EDITS: dict[str, ValueEdit] = {
    "hypotension_6h": ValueEdit(
        "sbp_noninvasive", "set", 80.0, 6.0, {"vasopressor_start": +1, "death": +1}
    ),
    "normotension_6h": ValueEdit(
        "sbp_noninvasive", "set", 120.0, 6.0, {"vasopressor_start": -1}
    ),
    "creatinine_plus_1": ValueEdit(
        "creatinine", "add", 1.0, 24.0, {"acute_kidney_injury": +1}
    ),
    "lactate_x3": ValueEdit(
        "lactate", "scale", 3.0, 12.0, {"death": +1, "vasopressor_start": +1}
    ),
    "remove_labs_24h": ValueEdit("LAB//RESULT//", "remove", 0.0, 24.0),
}


def _main() -> None:
    from odyssey.data.concepts import concepts_for_source  # noqa: PLC0415
    from odyssey.inference.run_inference import load_run  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Input-level counterfactual forecasts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--max-subjects", type=int, default=200)
    parser.add_argument("--index-hours", type=float, default=24.0)
    parser.add_argument(
        "--edits",
        nargs="+",
        default=list(STANDARD_EDITS),
        help="standard edit names (see STANDARD_EDITS) or signal:mode:value:window specs",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--keep-per-subject", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    model, vocab, binner, config = load_run(
        run_dir,
        device=device,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
    )
    source = getattr(config, "source", "mimic_iv")
    concept_names = [
        c.name
        for c in concepts_for_source(source, task_set=getattr(config, "task_set", "v1"))
    ]
    raw = load_meds_shards(args.held_out_shard_dir, max_shards=args.max_shards)
    raw = maybe_normalize(
        raw, enabled=getattr(config, "normalize_medications", False), source=source
    )
    raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
    summaries = {}
    for name in args.edits:
        edit = STANDARD_EDITS[name] if name in STANDARD_EDITS else ValueEdit.parse(name)
        logger.info("[counterfactual] edit %s: %s", name, edit)
        summary = cohort_counterfactual(
            model,
            vocab,
            binner,
            raw,
            [edit],
            concept_names=concept_names,
            index_hours=args.index_hours,
            max_subjects=args.max_subjects,
            source=source,
            device=device,
            chunk_size=args.chunk_size,
            keep_per_subject=args.keep_per_subject,
        )
        summaries[name] = asdict(summary)
        logger.info(
            "[counterfactual] %s: n=%d edited=%d mean delta %s sign agreement %s",
            name,
            summary.n_subjects,
            summary.n_edited,
            json.dumps(summary.mean_delta_event_risk),
            json.dumps(summary.sign_agreement),
        )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "index_hours": args.index_hours,
                "edits": summaries,
            },
            indent=2,
        )
    )
    logger.info("[counterfactual] wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _main()
