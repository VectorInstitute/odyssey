"""The demo's application layer: one method per API endpoint.

Owns the loaded run, the patient store, caches and the single GPU lock
every model call goes through (one card, one job at a time). It speaks
schema objects, never HTTP: :mod:`apps.clinician_demo.server` is a thin
adapter over it, and tests drive it directly with a tiny CPU model.
"""

import logging
import statistics
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from apps.clinician_demo.codebook import Codebook, admission_label
from apps.clinician_demo.config import DemoConfig
from apps.clinician_demo.evidence import (
    EvidenceRunner,
    cached_or_submit,
    candidate_codes,
)
from apps.clinician_demo.forecast import (
    ALERT_HORIZON_HOURS,
    PatientTrace,
    RunContext,
    bundle_ends,
    displayed_alerts,
    onsets_for,
    trace_patient,
    visit_stats_from_trace,
    visit_view,
    visit_window,
)
from apps.clinician_demo.patient_store import (
    PatientStore,
    UnknownPatientError,
    build_shard_index,
    discover_shards,
    load_splits,
)
from apps.clinician_demo.schemas import (
    BankedPoint,
    ConceptInfo,
    EventInfo,
    EvidenceItem,
    EvidenceJob,
    Gallery,
    GalleryCase,
    GallerySection,
    Meta,
    OperatingPoint,
    PatientSummary,
    Scorecard,
    VisitSummary,
    VisitTrace,
    WhatIfPreset,
    WhatIfResult,
)
from apps.clinician_demo.scorecard import build_scorecard
from apps.clinician_demo.showcase import (
    build_gallery,
    empty_visit_stats,
    visit_stats_from_alert_rows,
)
from apps.clinician_demo.thresholds import (
    ALERTS_ROWS_FILENAME,
    horizon_key,
    load_or_compute_operating_points,
)
from apps.clinician_demo.whatif import PRESETS, parse_edit_requests, run_whatif
from odyssey.data.alert_events import alert_events_for
from odyssey.data.concepts import canonical_concept_name, concept_display_name
from odyssey.data.sidecars import activate_sidecars
from odyssey.inference.concept_edit_attribution import occlude_codes
from odyssey.inference.legacy_concept_pins import resolve_concepts_for_run
from odyssey.inference.run_inference import load_run
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel


logger = logging.getLogger(__name__)

#: Display order and plain-language definitions of the forecast events.
EVENT_TEXT: dict[str, tuple[str, str, str]] = {
    "icu_admission": (
        "ICU admission",
        "ICU",
        "First transfer into an intensive care unit during the admission.",
    ),
    "vasopressor_start": (
        "Vasopressors",
        "Pressors",
        "First dose of a vasopressor (norepinephrine, epinephrine, vasopressin, "
        "phenylephrine, dopamine or angiotensin II) during the admission.",
    ),
    "acute_kidney_injury": (
        "Acute kidney injury",
        "AKI",
        "KDIGO stage 1 or worse: creatinine up 0.3 mg/dL within 48 h or to 1.5x "
        "baseline within 7 days, or urine output under 0.5 mL/kg/h for 6 h.",
    ),
    "sepsis3": (
        "Sepsis-3",
        "Sepsis",
        "Suspected infection (a culture plus antibiotics) with an acute rise in "
        "SOFA score of 2 or more.",
    ),
    "death": ("Death", "Death", "Death, during or after the admission."),
}
EVENT_ORDER = tuple(EVENT_TEXT)

DISCLAIMERS: dict[str, str] = {
    "banner": "Research prototype on retrospective data. Not for clinical use.",
    "credentialed": (
        "Held-out MIMIC-IV patients the model never trained on. Only for viewers "
        "holding PhysioNet MIMIC-IV credentials (data use agreement)."
    ),
    "open": (
        "Open MIMIC-IV Clinical Database Demo (100 patients). Most of these patients "
        "were in the model's training data; each chart says whether it was."
    ),
    "whatif": (
        "Shows how the model's forecast responds to a changed record: model "
        "sensitivity, not the effect of a treatment. The edit method was validated "
        "on an earlier model version."
    ),
    "evidence": (
        "Recorded items the forecast leans on, found by removing each one and "
        "re-scoring. Removing a normal reading pushes the forecast toward the "
        "population average, so trust the size of a change more than its direction."
    ),
    "concepts": (
        "The model's running belief that each condition occurs during this "
        "admission. A model reading, not a diagnosis."
    ),
    "risk": (
        "Chance the event first happens within the horizon, from this moment. "
        "Hidden once the event has happened."
    ),
}
TRACE_CACHE_SIZE = 16
BANKED_CACHE_SIZE = 64
MAX_LOOKBACK_HOURS = 72.0


#: Clinical acronyms and casing the registry's snake_case names lose.
CONCEPT_LABELS: dict[str, str] = {
    "sirs": "SIRS",
    "qsofa": "qSOFA",
    "sepsis3": "Sepsis-3",
    "acute_kidney_injury": "AKI (any stage)",
    "aki_stage_2": "AKI stage 2",
    "aki_stage_3": "AKI stage 3",
}


def concept_label(name: str) -> str:
    """Name a concept the way a clinician would write it."""
    canon = canonical_concept_name(name)
    if canon in CONCEPT_LABELS:  # hand-written: keep its exact casing (qSOFA)
        return CONCEPT_LABELS[canon]
    text = concept_display_name(canon)
    return text[:1].upper() + text[1:]


class NotFoundError(LookupError):
    """The requested patient, visit or job does not exist."""


class BadRequestError(ValueError):
    """The request is malformed or out of bounds."""


def event_infos(events: tuple[str, ...]) -> list[EventInfo]:
    """Return display names and definitions for ``events`` (fallback: raw name)."""
    out = []
    for name in events:
        display, short, definition = EVENT_TEXT.get(
            name, (name.replace("_", " "), name, "")
        )
        out.append(
            EventInfo(name=name, display=display, short=short, definition=definition)
        )
    return out


def _number(
    body: Mapping[str, Any],
    key: str,
    lo: float,
    hi: float,
    default: float | None = None,
) -> float:
    raw = body.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BadRequestError(f"{key} must be a number")
    value = float(raw)
    if not lo <= value <= hi:
        raise BadRequestError(f"{key} must be within [{lo:g}, {hi:g}]")
    return value


@dataclass(frozen=True)
class Components:
    """Everything a :class:`DemoService` is built from (injectable for tests)."""

    config: DemoConfig
    ctx: RunContext
    store: PatientStore
    codebook: Codebook
    operating_points: list[OperatingPoint]
    scorecard: Scorecard
    concepts: list[ConceptInfo]
    banked_rows_path: Path | None


class DemoService:
    """Serve the demo's API from a loaded run and a patient store."""

    def __init__(self, parts: Components) -> None:
        """Wire the service; call :meth:`prepare_gallery` before serving."""
        self.config = parts.config
        self.ctx = parts.ctx
        self.store = parts.store
        self.codebook = parts.codebook
        self.concepts = parts.concepts
        self.events = event_infos(parts.ctx.events)
        self._display = {e.name: e.display for e in self.events}
        self._points = parts.operating_points
        self._alert_points = {
            p.event: p
            for p in parts.operating_points
            if p.horizon_hours == ALERT_HORIZON_HOURS
        }
        self._scorecard = parts.scorecard
        self._banked_path = parts.banked_rows_path
        self._gpu_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._traces: OrderedDict[int, PatientTrace] = OrderedDict()
        self._banked: OrderedDict[tuple[int, int], list[BankedPoint]] = OrderedDict()
        self._evidence = EvidenceRunner(self._gpu_lock)
        self._evidence_jobs: dict[tuple[object, ...], str] = {}
        self._gallery = Gallery(sections=[])

    # -- construction -----------------------------------------------------

    @classmethod
    def from_config(cls, config: DemoConfig) -> "DemoService":
        """Load the run, index the data and precompute the aggregates."""
        model, vocab, binner, run_config = load_run(
            config.run_dir,
            device=config.device,
            checkpoint_path=config.run_dir / config.checkpoint,
        )
        heads = getattr(model, "event_heads", None)
        if not isinstance(model, ConceptBottleneckSequenceModel) or heads is None:
            raise ValueError(
                "the demo needs a concept-bottleneck model with event hazard heads"
            )
        source = getattr(run_config, "source", "mimic_iv")
        task_set = getattr(run_config, "task_set", "v1")
        definitions = resolve_concepts_for_run(str(config.run_dir), source, task_set)
        alerts, head_index = displayed_alerts(
            alert_events_for(task_set, source=source), heads.event_names, EVENT_ORDER
        )
        ctx = RunContext(
            model=model,
            vocab=vocab,
            binner=binner,
            source=source,
            task_set=task_set,
            chunk_size=int(run_config.chunk_size),
            device=config.device,
            concept_names=tuple(d.name for d in definitions),
            alerts=alerts,
            head_index=head_index,
            horizons=config.horizons,
        )
        activate_sidecars(config.data_dir)
        codebook = Codebook.from_metadata_dir(config.metadata_dir)
        store = PatientStore(
            build_shard_index(
                discover_shards(config.data_dir, max_shards=config.max_shards)
            ),
            source=source,
            normalize_medications=bool(
                getattr(run_config, "normalize_medications", False)
            ),
            history_recap=bool(getattr(run_config, "history_recap", False)),
            splits=load_splits(config.splits_path),
            describe=admission_label,
        )
        rows_path = config.run_dir / ALERTS_ROWS_FILENAME
        points = (
            load_or_compute_operating_points(
                rows_path,
                config.resolved_cache_dir / "thresholds.json",
                ctx.events,
                config.horizons,
                config.alert_rate,
            )
            if rows_path.exists()
            else []
        )
        concepts = [
            ConceptInfo(d.name, concept_label(d.name), d.description, None)
            for d in definitions
        ]
        scorecard = build_scorecard(
            config.run_dir, ctx.events, config.horizons, concepts
        )
        service = cls(
            Components(
                config=config,
                ctx=ctx,
                store=store,
                codebook=codebook,
                operating_points=points,
                scorecard=scorecard,
                concepts=scorecard.concepts or concepts,
                banked_rows_path=rows_path
                if rows_path.exists() and config.data_mode == "credentialed"
                else None,
            )
        )
        service.prepare_gallery()
        return service

    # -- gallery ----------------------------------------------------------

    def prepare_gallery(self) -> Gallery:
        """Build the gallery: from banked rows (credentialed) or by tracing (open)."""
        thresholds = {e: p.threshold for e, p in self._alert_points.items()}
        if self.config.data_mode == "credentialed" and self._banked_path is not None:
            stats = self._stats_from_banked(thresholds)
            gallery = build_gallery(
                stats,
                display=self._display,
                seen_in_training=self.store.seen_in_training,
            )
        else:
            stats = self._stats_from_traces(thresholds)
            gallery = build_gallery(
                stats,
                display=self._display,
                seen_in_training=self.store.seen_in_training,
                approximate_leads=False,
            )
            gallery = Gallery(
                sections=[*gallery.sections, self._all_patients_section()]
            )
        self._gallery = gallery
        return gallery

    def _stats_from_banked(self, thresholds: Mapping[str, float]) -> pl.DataFrame:
        assert self._banked_path is not None  # noqa: S101 -- checked by caller
        key = horizon_key(ALERT_HORIZON_HOURS)
        columns = [
            "subject_id",
            "visit_id",
            "time_hours",
            "event",
            f"hazard@{key}",
            f"y@{key}",
        ]
        rows = (
            pl.scan_parquet(self._banked_path)
            .select(columns)
            .filter(pl.col("subject_id").cast(pl.Int64).is_in(self.store.subject_ids))
            .collect()
        )
        return (
            visit_stats_from_alert_rows(rows, thresholds)
            if rows.height
            else empty_visit_stats()
        )

    def _stats_from_traces(self, thresholds: Mapping[str, float]) -> pl.DataFrame:
        frames = [empty_visit_stats()]
        for sid in self.store.subject_ids:
            try:
                trace = self._trace(sid)
                visits = self.store.visits(sid)
                raw = self.store.raw_events(sid)
            except ValueError as exc:
                logger.warning("[gallery] skipping subject %s: %s", sid, exc)
                continue
            onsets = {
                v.visit_id: onsets_for(
                    raw,
                    self.ctx.alerts,
                    source=self.ctx.source,
                    task_set=self.ctx.task_set,
                    subject_id=sid,
                    visit_id=v.visit_id,
                )
                for v in visits
            }
            frames.append(
                visit_stats_from_trace(
                    trace,
                    visits,
                    onsets,
                    events=self.ctx.events,
                    horizons=self.ctx.horizons,
                    thresholds=thresholds,
                )
            )
        return pl.concat(frames)

    def _all_patients_section(self) -> GallerySection:
        cases = []
        for sid in self.store.subject_ids:
            for visit in self.store.visits(sid):
                cases.append(
                    GalleryCase(
                        subject_id=sid,
                        visit_id=visit.visit_id,
                        kind="other",
                        event=None,
                        headline=visit.admission,
                        lead_hours=None,
                        los_hours=visit.end_hours - visit.start_hours,
                        seen_in_training=self.store.seen_in_training(sid),
                    )
                )
        return GallerySection(
            kind="other",
            title="All patients",
            summary=f"{len(cases)} admissions of {len(self.store)} patients",
            cases=cases,
        )

    # -- endpoints --------------------------------------------------------

    def meta(self) -> Meta:
        """Return the static facts the UI needs."""
        mode = self.config.data_mode
        return Meta(
            run_name=self.config.run_name,
            checkpoint=self.config.checkpoint,
            data_mode=mode,
            chunk_size=self.ctx.chunk_size,
            horizons=list(self.ctx.horizons),
            events=self.events,
            concepts=self.concepts,
            operating_points=self._points,
            disclaimers={
                k: v
                for k, v in DISCLAIMERS.items()
                if k not in ("credentialed", "open") or k == mode
            },
            searchable=mode == "credentialed",
        )

    def gallery(self) -> Gallery:
        """Return the curated gallery (built at start-up)."""
        return self._gallery

    def patient(self, subject_id: int) -> PatientSummary:
        """Return the header facts and admissions of one patient."""
        try:
            return self.store.summary(subject_id)
        except UnknownPatientError as exc:
            raise NotFoundError(
                f"patient {subject_id} is not in the loaded data"
            ) from exc

    def trace(self, subject_id: int, visit_id: int) -> VisitTrace:
        """Return the replay view of one visit."""
        visit = self._visit(subject_id, visit_id)
        trace = self._trace(subject_id)
        onsets = onsets_for(
            self.store.raw_events(subject_id),
            self.ctx.alerts,
            source=self.ctx.source,
            task_set=self.ctx.task_set,
            subject_id=subject_id,
            visit_id=visit_id,
        )
        return visit_view(
            trace,
            visit,
            events=self.ctx.events,
            horizons=self.ctx.horizons,
            onsets=onsets,
            points=self._alert_points,
            codebook=self.codebook,
            decode=self.ctx.vocab.decode,
            display=self._display,
            seen_in_training=self.store.seen_in_training(subject_id),
            banked=self._banked_points(subject_id, visit_id, visit.start_hours),
        )

    def presets(self) -> list[WhatIfPreset]:
        """Return the what-if controls."""
        return list(PRESETS)

    def whatif(
        self, subject_id: int, visit_id: int, body: Mapping[str, Any]
    ) -> WhatIfResult:
        """Compare the factual and edited forecast at a moment of the visit."""
        t_hours = _number(body, "t_hours", -1e6, 1e6)
        try:
            requests = parse_edit_requests(body.get("edits"))
        except ValueError as exc:
            raise BadRequestError(str(exc)) from exc
        index_time, t_snapped = self._moment(subject_id, visit_id, t_hours)
        raw = self.store.raw_events(subject_id)
        with self._gpu_lock:
            return run_whatif(
                self.ctx, raw, requests, index_time=index_time, t_hours=t_snapped
            )

    def evidence(
        self, subject_id: int, visit_id: int, body: Mapping[str, Any]
    ) -> EvidenceJob:
        """Start (or reuse) an evidence search for a target at a moment."""
        t_hours = _number(body, "t_hours", -1e6, 1e6)
        lookback = _number(
            body, "lookback_hours", 1.0, MAX_LOOKBACK_HOURS, default=24.0
        )
        target = body.get("target")
        if not isinstance(target, dict):
            raise BadRequestError("target must be an object")
        kind, name = target.get("kind"), str(target.get("name"))
        if kind == "event":
            if name not in self.ctx.events:
                raise BadRequestError(f"unknown event {name!r}")
            horizon = _number(
                target, "horizon_hours", 0.0, 1e4, default=ALERT_HORIZON_HOURS
            )
            if horizon not in self.ctx.horizons:
                raise BadRequestError(
                    f"horizon must be one of {list(self.ctx.horizons)}"
                )
            label = f"{self._display.get(name, name)} risk within {horizon:g} h"
            key_h = horizon_key(horizon)

            def value_of(r: Any) -> float:  # noqa: ANN401 -- ForecastReadout
                return float(r.event_risk[name][key_h])

        elif kind == "concept":
            if name not in self.ctx.concept_names:
                raise BadRequestError(f"unknown concept {name!r}")
            label = f"belief in {concept_label(name)}"

            def value_of(r: Any) -> float:  # noqa: ANN401 -- ForecastReadout
                return float(r.concept_probs[name])

        else:
            raise BadRequestError("target.kind must be 'event' or 'concept'")

        index_time, t_snapped = self._moment(subject_id, visit_id, t_hours)
        raw = self.store.raw_events(subject_id)
        candidates = candidate_codes(
            raw, index_time=index_time, lookback_hours=lookback
        )

        def work(progress: Any) -> list[EvidenceItem]:  # noqa: ANN401 -- ProgressFn
            ranked = occlude_codes(
                self.ctx.model,
                self.ctx.vocab,
                self.ctx.binner,
                raw,
                index_time=index_time,
                value_of=value_of,
                concept_names=self.ctx.concept_names,
                lookback_hours=lookback,
                candidate_codes=candidates,
                source=self.ctx.source,
                device=self.ctx.device,
                chunk_size=self.ctx.chunk_size,
                horizons=self.ctx.horizons,
                on_progress=progress,
            )
            return [
                EvidenceItem(
                    code=a.code,
                    label=self.codebook.label(a.code),
                    n_rows=a.n_rows,
                    baseline=a.baseline,
                    occluded=a.occluded,
                    delta=a.delta,
                )
                for a in ranked
            ]

        note = (
            f"{label} at hour {t_snapped:.1f}; {len(candidates)} most frequent items of the "
            f"last {lookback:g} h each removed in turn."
        )
        key = (
            subject_id,
            visit_id,
            round(t_snapped, 4),
            kind,
            name,
            key_h if kind == "event" else None,
            lookback,
        )
        return cached_or_submit(
            self._evidence_jobs,
            key,
            self._evidence,
            lambda: self._evidence.submit(label, note, work),
        )

    def job(self, job_id: str) -> EvidenceJob:
        """Poll an evidence job."""
        try:
            return self._evidence.get(job_id)
        except KeyError as exc:
            raise NotFoundError(f"no job {job_id!r}") from exc

    def scorecard(self) -> Scorecard:
        """Return the model's report card."""
        return self._scorecard

    def shutdown(self) -> None:
        """Release background workers."""
        self._evidence.shutdown()

    # -- internals ----------------------------------------------------------

    def _visit(self, subject_id: int, visit_id: int) -> VisitSummary:
        for visit in self.patient(subject_id).visits:
            if visit.visit_id == visit_id:
                return visit
        raise NotFoundError(f"patient {subject_id} has no visit {visit_id}")

    def _trace(self, subject_id: int) -> PatientTrace:
        with self._cache_lock:
            cached = self._traces.get(subject_id)
            if cached is not None:
                self._traces.move_to_end(subject_id)
                return cached
        try:
            raw = self.store.raw_events(subject_id)
        except UnknownPatientError as exc:
            raise NotFoundError(
                f"patient {subject_id} is not in the loaded data"
            ) from exc
        with self._gpu_lock:
            with self._cache_lock:  # another request may have traced it meanwhile
                cached = self._traces.get(subject_id)
            if cached is not None:
                return cached
            trace = trace_patient(self.ctx, raw)
        with self._cache_lock:
            self._traces[subject_id] = trace
            while len(self._traces) > TRACE_CACHE_SIZE:
                self._traces.popitem(last=False)
        return trace

    def _moment(
        self, subject_id: int, visit_id: int, t_hours: float
    ) -> tuple[object, float]:
        """Return the timestamp of the last bundle end at or before ``t_hours``.

        Snaps to a moment the replay actually shows, and returns the exact
        timestamp of that row (not a float round trip), so re-scoring reads
        the same position the replay drew.
        """
        visit = self._visit(subject_id, visit_id)
        trace = self._trace(subject_id)
        first, last = visit_window(trace.visit_ids, visit_id, trace.n_static)
        ends = [i for i in bundle_ends(trace.times) if first <= i <= last]
        target = visit.start_hours + t_hours
        eligible = [i for i in ends if trace.times[i] <= target + 1e-9]
        pos = eligible[-1] if eligible else ends[0]
        return trace.timestamps[pos], trace.times[pos] - visit.start_hours

    def _banked_points(
        self, subject_id: int, visit_id: int, start: float
    ) -> list[BankedPoint]:
        if self._banked_path is None:
            return []
        key = (subject_id, visit_id)
        with self._cache_lock:
            if key in self._banked:
                return self._banked[key]
        h = horizon_key(ALERT_HORIZON_HOURS)
        rows = (
            pl.scan_parquet(self._banked_path)
            .select(
                [
                    "subject_id",
                    "visit_id",
                    "time_hours",
                    "event",
                    f"hazard@{h}",
                    f"gbm@{h}",
                    f"y@{h}",
                ]
            )
            .filter(
                (pl.col("subject_id") == float(subject_id))
                & (pl.col("visit_id") == float(visit_id))
            )
            .filter(pl.col("event").is_in(list(self.ctx.events)))
            .sort("time_hours")
            .collect()
        )
        points = [
            BankedPoint(
                t=float(r["time_hours"]) - start,
                event=str(r["event"]),
                hazard_24h=r[f"hazard@{h}"],
                gbm_24h=r[f"gbm@{h}"],
                outcome_24h=r[f"y@{h}"],
            )
            for r in rows.iter_rows(named=True)
        ]
        with self._cache_lock:
            self._banked[key] = points
            while len(self._banked) > BANKED_CACHE_SIZE:
                self._banked.popitem(last=False)
        return points

    # -- operations ---------------------------------------------------------

    def warm_up(self) -> threading.Thread:
        """Trace every gallery patient in the background so first clicks are fast."""
        subjects = sorted(
            {c.subject_id for s in self._gallery.sections for c in s.cases}
        )

        def run() -> None:
            for sid in subjects:
                try:
                    self._trace(sid)
                except Exception:  # noqa: BLE001 -- warm-up must never kill the server
                    logger.exception("[warm-up] subject %s failed", sid)
            logger.info("[warm-up] traced %d gallery patients", len(subjects))

        thread = threading.Thread(target=run, name="warm-up", daemon=True)
        thread.start()
        return thread

    def self_check(self) -> dict[str, Any]:
        """Measure one gallery case end to end; raise on a broken invariant."""
        report: dict[str, Any] = {
            "run": self.config.run_name,
            "data_mode": self.config.data_mode,
            "chunk_size": self.ctx.chunk_size,
            "events": list(self.ctx.events),
            "subjects": len(self.store),
            "codebook_entries": len(self.codebook),
            "operating_points": len(self._points),
            "gallery": {s.kind: len(s.cases) for s in self._gallery.sections},
        }
        if "readmission_30d" in self.ctx.events:
            raise AssertionError("readmission must never be displayed")
        case = next((c for s in self._gallery.sections for c in s.cases), None)
        if case is None:
            report["warning"] = "gallery is empty; nothing to trace"
            return report
        started = time.perf_counter()
        view = self.trace(case.subject_id, case.visit_id)
        report["trace_seconds"] = round(time.perf_counter() - started, 2)
        trace = self._trace(case.subject_id)
        report["trace_positions"] = trace.n_positions
        report["unknown_token_share"] = round(
            trace.n_unknown / max(trace.n_positions, 1), 4
        )
        report["replay_points"] = len(view.times)
        if view.banked:
            report["banked_gap_24h"] = self._banked_gap(
                case.subject_id, case.visit_id, view.banked
            )
        t_mid = view.times[len(view.times) // 2]
        started = time.perf_counter()
        result = self.whatif(
            case.subject_id,
            case.visit_id,
            {"t_hours": t_mid, "edits": [{"preset": "sbp"}]},
        )
        report["whatif_seconds"] = round(time.perf_counter() - started, 2)
        report["whatif_rows_edited"] = result.rows_edited
        return report

    def _banked_gap(
        self, subject_id: int, visit_id: int, banked: list[BankedPoint]
    ) -> dict[str, float]:
        """Compare demo and banked 24 h risk at the banked landmark moments."""
        trace = self._trace(subject_id)
        visit = self._visit(subject_id, visit_id)
        h_index = list(self.ctx.horizons).index(ALERT_HORIZON_HOURS)
        gaps: list[float] = []
        for point in banked:
            if point.hazard_24h is None or point.event not in self.ctx.events:
                continue
            t_abs = point.t + visit.start_hours
            positions = [i for i, t in enumerate(trace.times) if abs(t - t_abs) < 1e-6]
            if positions:
                j = self.ctx.events.index(point.event)
                gaps.append(
                    abs(float(trace.risk[positions[0], j, h_index]) - point.hazard_24h)
                )
        if not gaps:
            return {"n": 0}
        return {"n": len(gaps), "median": statistics.median(gaps), "max": max(gaps)}


__all__ = [
    "DISCLAIMERS",
    "EVENT_ORDER",
    "EVENT_TEXT",
    "CONCEPT_LABELS",
    "BadRequestError",
    "Components",
    "DemoService",
    "NotFoundError",
    "concept_label",
    "event_infos",
]
