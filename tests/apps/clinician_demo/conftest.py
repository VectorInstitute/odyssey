"""Shared fixtures: a synthetic MEDS cohort and a tiny CPU model (no real data)."""

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from apps.clinician_demo.codebook import Codebook
from apps.clinician_demo.config import DemoConfig
from apps.clinician_demo.forecast import RunContext, displayed_alerts
from apps.clinician_demo.patient_store import (
    PatientStore,
    build_shard_index,
    discover_shards,
)
from apps.clinician_demo.schemas import ConceptInfo, OperatingPoint, Scorecard
from apps.clinician_demo.service import EVENT_ORDER, Components, DemoService
from apps.clinician_demo.whatif import PRESETS, EditRequest, to_value_edits
from odyssey.data.alert_events import alert_events_for
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.counterfactual import apply_value_edits
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS


T0 = datetime(2150, 3, 1, 8, 0)
SBP = "LAB//220179//mmHg"
HR = "LAB//220045//bpm"
CREAT = "LAB//RESULT//50912//mg/dL"
ICU = "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"
HEAD_NAMES = [
    "vasopressor_start",
    "icu_admission",
    "acute_kidney_injury",
    "death",
    "sepsis3",
    "readmission_30d",
]
CHUNK = 16  # deliberately not any API's default (256), so a dropped value shows
SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}

Row = tuple[int, datetime | None, str, float | None, int | None]


def cohort_rows() -> list[Row]:
    """Subject 1: an ICU transfer at hour 20 of visit 10, death 10 h into visit 11.

    Subject 2: one uneventful 60 h visit. Every hour of visit 10 holds a
    same-time bundle (SBP + HR); creatinine is drawn every 6 h.
    """
    rows: list[Row] = [
        (1, None, "GENDER//M", None, None),
        (1, T0 - timedelta(days=365.25 * 65), "MEDS_BIRTH", None, None),
        (1, T0, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None, 10),
    ]
    for h in range(1, 41):
        t = T0 + timedelta(hours=h)
        rows.append((1, t, SBP, 125.0 - 1.5 * h, 10))
        rows.append((1, t, HR, 80.0 + h, 10))
        if h % 6 == 0:
            rows.append((1, t, CREAT, 1.0 + 0.1 * (h // 6), 10))
    rows.append((1, T0 + timedelta(hours=20), ICU, None, 10))
    rows.append((1, T0 + timedelta(hours=41), "HOSPITAL_DISCHARGE//HOME", None, 10))
    second = T0 + timedelta(days=30)
    rows.append(
        (1, second, "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM HOSPITAL", None, 11)
    )
    for h in range(1, 9):
        rows.append((1, second + timedelta(hours=h), SBP, 90.0, 11))
    # an in-hospital death: MIMIC records it with a DIED discharge at the same time
    rows.append((1, second + timedelta(hours=10), "HOSPITAL_DISCHARGE//DIED", None, 11))
    rows.append((1, second + timedelta(hours=10), "MEDS_DEATH", None, None))

    rows += [
        (2, None, "GENDER//F", None, None),
        (2, T0, "HOSPITAL_ADMISSION//ELECTIVE//PHYSICIAN REFERRAL", None, 20),
    ]
    for h in range(1, 61):
        rows.append((2, T0 + timedelta(hours=h), SBP, 120.0, 20))
    rows.append((2, T0 + timedelta(hours=61), "HOSPITAL_DISCHARGE//HOME", None, 20))
    return rows


def cohort_frame() -> pl.DataFrame:
    """Return the synthetic cohort as a MEDS frame."""
    return pl.DataFrame(cohort_rows(), schema=SCHEMA, orient="row")


@pytest.fixture
def cohort_dir(tmp_path: Path) -> Path:
    """Two held-out shards, one subject each."""
    data = tmp_path / "held_out"
    data.mkdir()
    frame = cohort_frame()
    frame.filter(pl.col("subject_id") == 1).write_parquet(data / "0.parquet")
    frame.filter(pl.col("subject_id") == 2).write_parquet(data / "1.parquet")
    return data


def _vocabulary() -> Vocabulary:
    """Every token the cohort and its what-if edits can produce."""
    raw = maybe_normalize(cohort_frame(), enabled=True, source="mimic_iv")
    codes = add_value_tokens(raw)["code"].to_list()
    for preset in PRESETS:
        for value in (preset.min, preset.value, preset.max):
            edited, _ = apply_value_edits(
                raw,
                to_value_edits(EditRequest(preset.id, value)),
                index_time=T0 + timedelta(days=60),
            )
            codes += add_value_tokens(edited)["code"].to_list()
    return Vocabulary.build(codes, min_count=1)


CONCEPT_NAMES = tuple(c.name for c in concepts_for_source("mimic_iv", task_set="v3"))


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    """Return the cohort's vocabulary."""
    return _vocabulary()


@pytest.fixture(scope="session")
def model(vocab: Vocabulary) -> ConceptBottleneckSequenceModel:
    """Build a tiny deterministic bottleneck model with the real event heads."""
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=len(vocab), hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=len(vocab),
        num_concepts=len(CONCEPT_NAMES),
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=HEAD_NAMES,
    ).eval()


@pytest.fixture
def ctx(model: ConceptBottleneckSequenceModel, vocab: Vocabulary) -> RunContext:
    """Build the run context the service would build for this model."""
    alerts, head_index = displayed_alerts(
        alert_events_for("v3", source="mimic_iv"), HEAD_NAMES, EVENT_ORDER
    )
    return RunContext(
        model=model,
        vocab=vocab,
        binner=None,
        source="mimic_iv",
        task_set="v3",
        chunk_size=CHUNK,
        device="cpu",
        concept_names=CONCEPT_NAMES,
        alerts=alerts,
        head_index=head_index,
        horizons=(8.0, 24.0, 72.0),
    )


@pytest.fixture
def store(cohort_dir: Path) -> PatientStore:
    """Build the cohort's store; subject 1 is in the model's training split."""
    return PatientStore(
        build_shard_index(discover_shards(cohort_dir)),
        source="mimic_iv",
        normalize_medications=True,
        splits={1: "train", 2: "held_out"},
    )


def points_for(events: tuple[str, ...], threshold: float) -> list[OperatingPoint]:
    """One 24 h operating point per event at ``threshold``."""
    return [
        OperatingPoint(e, 24.0, threshold, 0.05, 0.5, 0.2, 0.01, 1000, 0.6, 0.25)
        for e in events
    ]


ServiceFactory = Callable[..., DemoService]


@pytest.fixture
def make_service(
    tmp_path: Path, cohort_dir: Path, ctx: RunContext, store: PatientStore
) -> Iterator[ServiceFactory]:
    """Build a service over the cohort; ``threshold`` sets every alert line."""
    built: list[DemoService] = []

    def factory(
        *,
        data_mode: str = "open",
        threshold: float = 0.0,
        banked_rows_path: Path | None = None,
        prepare: bool = True,
    ) -> DemoService:
        config = DemoConfig(
            run_dir=tmp_path / "run",
            data_dir=cohort_dir,
            data_mode=data_mode,
            device="cpu",  # type: ignore[arg-type]
        )
        service = DemoService(
            Components(
                config=config,
                ctx=ctx,
                store=store,
                codebook=Codebook(),
                operating_points=points_for(ctx.events, threshold),
                scorecard=Scorecard(headline="h", cells=[], concepts=[], notes=[]),
                concepts=[ConceptInfo(n, n, "", None) for n in ctx.concept_names],
                banked_rows_path=banked_rows_path,
            )
        )
        if prepare:
            service.prepare_gallery()
        built.append(service)
        return service

    yield factory
    for service in built:
        service.shutdown()
