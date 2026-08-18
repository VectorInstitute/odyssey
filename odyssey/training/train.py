"""End-to-end training: real MEDS shards -> a trained ConceptBottleneckSequenceModel.

Usage (on a CUDA host, from the repo root)::

    uv run python -m odyssey.training.train \\
        --train-shard-dir /path/to/data/train \\
        --tuning-shard-dir /path/to/data/tuning \\
        --output-dir runs/exp1 \\
        --max-train-shards 20 --max-tuning-shards 5

Every loss component (task, concept, orthogonality, observability) is
logged per step to ``<output_dir>/loss_log.jsonl``, one JSON object per
line -- the source the results HTML's loss-curve plots read from
directly, so training is auditable after the fact without re-running it.
Checkpoints are periodic ``torch.save`` dicts of the model/optimizer
state, one more at the end of every epoch, plus a final one; the fitted
quantile binner and vocabulary are saved alongside so inference can
reconstruct the exact same tokenization without needing the train split
again. Every evaluation that improves the combined validation loss (the
same task + weighted concept/orthogonality/observability combination
training itself optimizes) also saves ``checkpoint_best.pt`` -- the one
inference should generally load, since it need not be the same as
whatever the run happened to end on. Set ``early_stopping_patience`` to
stop once that hasn't improved for that many consecutive evaluations;
unset (the default), a run always completes every configured epoch, as
it always did before this existed.

To resume after an interruption (e.g. a spot-instance preemption), pass
``--config-json`` with ``"resume_from": "<output_dir>/checkpoint_N.pt"``
(any checkpoint, periodic or epoch-boundary). Resuming fast-forwards
the resumed epoch's own deterministically-seeded sampler back to the
same position the checkpoint was taken at (discarding chunks, no
gradient steps, so this is cheap) rather than restarting that epoch's
data from its beginning -- correct only if ``num_lanes``/``chunk_size``/
``reset_prob``/``seed`` are unchanged from the checkpoint (saved
alongside it for exactly this check); if they differ, the epoch
restarts from its own beginning instead, with a warning logged, since
fast-forwarding down a differently-configured sampler would land on a
different, not equivalent, position.
"""

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, TypeVar

import polars as pl
import torch

from odyssey.data.alert_events import ALERT_EVENTS, all_event_times
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
    ForecastObjective,
    SequenceModel,
)
from odyssey.models.time_to_event import DEFAULT_TIME_BIN_EDGES_HOURS
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
    build_visit_concept_first_times,
    build_visit_concept_label_dicts,
    build_vocabulary,
    count_subjects,
    family_loss_weights,
    iter_patient_sequences,
    load_meds_shards,
    token_type_lookup,
)
from odyssey.training.event_targets import EventTimeTables, event_hazard_targets
from odyssey.training.running_labels import randint_intervention


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """All paths and hyperparameters for one training run."""

    train_shard_dir: str
    tuning_shard_dir: str
    output_dir: str
    max_train_shards: Optional[int] = None
    max_tuning_shards: Optional[int] = None
    resume_from: Optional[str] = None

    model_kind: str = "bottleneck"
    """``"bottleneck"`` (ConceptBottleneckSequenceModel, the interpretable
    model this project is about) or ``"baseline"`` (BaselineSequenceModel:
    the same backbone and forecasting/time/event heads with no concept
    bottleneck and no concept supervision). Train both with identical
    settings to price the bottleneck: the README's "costs little" claim
    is measured, not assumed."""

    # Backbone (EHRHybridBackbone). Defaults are modest, not the paper-scale
    # numbers -- see the training run's own README note on why.
    hidden_size: int = 256
    num_hidden_layers: int = 8
    value_embeddings: bool = False
    event_head_hidden: int = 0
    """Hidden width of the per-event hazard heads' MLP readout; 0 = the
    single linear layer every run before v8 used."""
    concept_global_pairs: bool = False
    """Leakage control: input-independent (w+, w-) per known concept, so a
    concept slot carries only its probability (see ConceptBottleneck)."""
    unknown_dim: Optional[int] = None
    """Width of the unknown (residual) slot; None = embedding_dim."""
    """Feed standardized numeric values (``aux.values``) into the token
    embeddings alongside the bin tokens (see
    :class:`~odyssey.models.embeddings.ClinicalEventEmbeddings`). Opt-in;
    an A/B against the bin-only input."""
    mamba_state_size: int = 128
    mamba_headdim: int = 64
    mamba_chunk_size: int = 256
    attn_num_heads: int = 8
    embedding_dim: int = 32

    # Tokenization
    source: str = "mimic_iv"
    """Which institution's extraction the shards come from ("mimic_iv",
    "eicu", "gemini"). Picks the per-source expansion of the canonical
    concept registry (odyssey.data.concepts.concepts_for_source) and of
    the curated clinical bin ranges
    (odyssey.data.value_binning.clinical_ranges_for_source); the rest of
    the pipeline is source-independent."""

    vocab_min_count: int = 5
    vocab_max_size: int = 20_000
    vocab_backoff: Optional[str] = "icd3"
    """Named vocabulary backoff (see odyssey.data.vocabulary.BACKOFFS).
    "icd3" rolls rare ICD diagnosis/procedure codes up to their
    3-character category, both when building the vocabulary and when
    encoding, so rare codes become predictable category tokens instead
    of [UNK]. None disables."""

    history_recap: bool = False
    """At each hospital admission, inject HISTORY//DIAGNOSIS//... tokens
    for the patient's prior diagnosis categories (odyssey.data.history_recap)
    so chronic conditions are in the local context when discharge coding
    and in-visit forecasts are made. Off by default; a data-level
    experiment motivated by the bundle analysis (discharge diagnosis
    recall at bundle start ~11% vs ~35% by copying the previous
    admission's codes)."""

    normalize_medications: bool = True
    """Collapse medication codes to ingredient level (strip dose, form,
    route, container) before tokenization -- one drug, one token, instead
    of dozens of sparse sig-line variants. See
    odyssey.data.code_normalization."""
    quantile_n_bins: int = 5
    quantile_min_count: int = 100
    max_seq_len: Optional[int] = None

    # Streaming TBTT
    num_lanes: int = 8
    chunk_size: int = 256
    reset_prob: float = 0.1

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_epochs: int = 3
    concept_weight: float = 1.0
    orthogonality_weight: float = 0.1
    observability_weight: float = 0.1

    # Logging/checkpointing cadence, in optimizer steps.
    log_every: int = 20
    eval_every: int = 200
    eval_max_chunks: int = 50
    checkpoint_every: int = 500

    concept_pos_weight: bool = True
    """Weight each concept's positive BCE term by its training-split
    ``n_negative / n_positive`` among observed entries (clamped to
    [0.2, 10]), so rare concepts (AKI stage 2 at ~5% prevalence)
    contribute real positive gradient instead of sitting near their base
    rate. Disable to reproduce the original unweighted loss."""

    concept_supervision: str = "visit"
    """Where concept supervision applies and how labels are keyed:
    ``"visit"`` (the default) supervises at every real visit's last event
    with visit-scoped labels; ``"stay"`` reproduces the original
    whole-stay-label-at-patient-end behavior (the subset-run baseline in
    research journal entry 07). Visit scoping exists because entry 07's
    evaluation showed whole-stay labels demand long-range single-event
    recall the compressed recurrent state cannot guarantee."""

    bundle_invariant_loss: bool = True
    """Score each position by the total probability of every not-yet-
    emitted member of the target's same-timestamp bundle instead of the
    single next token in ETL order (see
    odyssey.models.sequence_model.ForecastObjective). Off reproduces
    plain next-token cross-entropy."""

    family_balance_alpha: float = 0.5
    """Per-family loss weights proportional to (family share of training
    targets) ** -alpha, normalized to mean 1 over targets and capped at
    family_weight_cap. 0 disables (labs, 85% of positions, then dominate
    the gradient); 1 would be full inverse frequency. 0.5 is a square-root
    tempering."""

    family_weight_cap: float = 20.0

    time_to_event: bool = True
    """Add the time-to-next-event hazard head (odyssey.models.time_to_event)
    and its loss, weighted by time_weight. Off reproduces the previous
    what-only model."""

    time_weight: float = 1.0

    event_hazards: bool = True
    """Add per-event hazard heads (time to vasopressor start, ICU admission,
    acute kidney injury, death; odyssey.data.alert_events.ALERT_EVENTS)
    trained with right censoring, so calibrated P(event within h) and
    survival curves read off the model directly. This is what makes the
    general forecaster usable for alerts; the alert evaluation harness
    scores these heads against bespoke per-event GBM baselines. Off
    reproduces a model with no alert-grade output."""

    event_hazard_weight: float = 1.0

    randint_prob: float = 0.25
    """Intervention-aware training (CEM's RandInt): at every training
    position, each observed concept's mixing probability is replaced by
    its running ground-truth value with this probability, so the task
    head learns to rely on the concept values and test-time
    interventions actually move forecasts. The concept and
    observability losses still supervise the model's own readouts. 0
    disables (the pre-Aug-16 behavior, under which the
    magnitude-controlled intervention test found the concept
    probabilities causally inert). 0.25 is the CEM default."""

    early_stopping_patience: Optional[int] = None
    """Stop once the combined validation loss (the same task + weighted
    concept/orthogonality/observability combination compute_streaming_loss
    trains against, evaluated on the tuning split) hasn't improved for
    this many consecutive ``eval_every`` checks. ``None`` disables early
    stopping -- the run always did every configured epoch before this
    existed, so opt-in keeps that the default. Every improvement saves
    ``checkpoint_best.pt``, independent of ``checkpoint_every``."""

    seed: int = 0


def _atomic_torch_save(obj: object, path: Path) -> None:
    """Save ``obj`` via ``torch.save``, atomically -- never a partial file.

    ``torch.save`` writes its target path directly and isn't atomic: a
    process killed mid-write (OOM, spot preemption, any crash) leaves a
    truncated file at that exact path. A later ``torch.load`` of that
    path then fails with an ``EOFError`` -- confirmed directly during
    this project's own training run, where the resulting resume loop
    kept picking the same corrupted checkpoint on every restart, unable
    to recover on its own. Writing to a sibling ``.tmp`` path first and
    renaming it into place afterward avoids this: ``Path.replace`` is
    atomic on the same filesystem (POSIX), so the target path always
    either has the previous complete file (if a crash happens before
    the rename) or the new complete one (after) -- never a partial one.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


class LossLogger:
    """Append-only JSONL logger for per-step loss components."""

    def __init__(self, path: Path) -> None:
        """Open ``path`` for appending; creates it if it doesn't exist."""
        self.path = path
        self._file = path.open("a")

    def log(self, **fields: object) -> None:
        """Write one JSON-serializable record as a line."""
        self._file.write(json.dumps(fields) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()


_Movable = TypeVar("_Movable")


def _move_chunk_to_device(chunk: _Movable, device: str) -> _Movable:
    """Move every tensor field of a (possibly nested) NamedTuple to ``device``.

    Works for :class:`~odyssey.data.streaming.StreamingChunk` and its
    nested :class:`~odyssey.data.types.ClinicalSequenceBatch`/
    :class:`~odyssey.data.types.AuxiliaryInputs` without depending on
    their exact field lists, so a new field added to any of them doesn't
    need a matching change here.
    """
    if isinstance(chunk, torch.Tensor):
        return chunk.to(device)  # type: ignore[return-value]
    if isinstance(chunk, tuple) and hasattr(chunk, "_fields"):  # NamedTuple
        return type(chunk)(*(_move_chunk_to_device(v, device) for v in chunk))
    return chunk


def _detach_state(state: TimeAwareState) -> TimeAwareState:
    """Truncate BPTT across chunks for a backbone's carried recurrent state.

    Backbone-agnostic so the streaming loops here don't require the
    hybrid backbone specifically: handles ``EHRHybridBackbone``'s
    :class:`~odyssey.models.backbones.hybrid.HybridState` and the plain
    tuple-of-tensors state lighter backbones (e.g. ``TinyGRUBackbone``)
    return. A new backbone with a different state shape must be added
    here explicitly -- silently not detaching would leak the autograd
    graph across every chunk of an epoch.
    """
    from odyssey.models.backbones.hybrid import HybridState  # noqa: PLC0415

    recurrent = state.recurrent
    detached_recurrent: object
    if isinstance(recurrent, HybridState):
        detached_recurrent = HybridState(
            {
                layer_idx: tuple(t.detach() for t in cached)
                for layer_idx, cached in recurrent.mamba_states.items()
            }
        )
    elif isinstance(recurrent, tuple) and all(
        isinstance(t, torch.Tensor) for t in recurrent
    ):
        detached_recurrent = tuple(t.detach() for t in recurrent)
    else:
        raise TypeError(
            f"_detach_state does not know this backbone's state shape: "
            f"{type(recurrent)!r}"
        )
    return TimeAwareState(
        recurrent=detached_recurrent,
        prev_time_stamps=state.prev_time_stamps.detach(),
    )


def build_model(
    config: TrainingConfig, *, vocab_size: int, num_concepts: int
) -> SequenceModel:
    """Construct the real backbone + heads from ``config`` (see ``model_kind``)."""
    from odyssey.models.backbones.hybrid import EHRHybridBackbone  # noqa: PLC0415

    kind = getattr(config, "model_kind", "bottleneck")
    if kind not in ("bottleneck", "baseline"):
        raise ValueError(f"model_kind must be 'bottleneck' or 'baseline', got {kind!r}")
    backbone = EHRHybridBackbone(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        padding_idx=PAD_ID,
        num_hidden_layers=config.num_hidden_layers,
        mamba_state_size=config.mamba_state_size,
        mamba_headdim=config.mamba_headdim,
        mamba_chunk_size=config.mamba_chunk_size,
        attn_num_heads=config.attn_num_heads,
        use_values=bool(getattr(config, "value_embeddings", False)),
    )
    time_bin_edges = (
        DEFAULT_TIME_BIN_EDGES_HOURS
        if getattr(config, "time_to_event", False)
        else None
    )
    event_names = (
        [a.name for a in ALERT_EVENTS]
        if getattr(config, "event_hazards", False)
        else None
    )
    event_head_hidden = int(getattr(config, "event_head_hidden", 0) or 0)
    if kind == "baseline":
        return BaselineSequenceModel(
            backbone=backbone,
            vocab_size=vocab_size,
            padding_idx=PAD_ID,
            time_bin_edges=time_bin_edges,
            event_names=event_names,
            event_head_hidden=event_head_hidden,
        )
    return ConceptBottleneckSequenceModel(
        backbone=backbone,
        vocab_size=vocab_size,
        num_concepts=num_concepts,
        embedding_dim=config.embedding_dim,
        padding_idx=PAD_ID,
        time_bin_edges=time_bin_edges,
        event_names=event_names,
        event_head_hidden=event_head_hidden,
        concept_global_pairs=bool(getattr(config, "concept_global_pairs", False)),
        unknown_dim=getattr(config, "unknown_dim", None),
    )


def build_objective(
    config: TrainingConfig,
    vocab: Vocabulary,
    train_events_binned: pl.DataFrame,
    device: str,
) -> ForecastObjective:
    """Construct the :class:`ForecastObjective` a run trains and validates with."""
    family_weights = None
    # token -> family is always needed: the bundle-invariant loss restricts
    # membership to the target's own family (see _bundle_log_likelihood).
    token_types = token_type_lookup(vocab).to(device)
    if config.family_balance_alpha > 0.0:
        family_weights = family_loss_weights(
            train_events_binned,
            alpha=config.family_balance_alpha,
            cap=config.family_weight_cap,
            n_families=int(token_types.max().item()) + 1,
        ).to(device)
        logger.info(
            "[loss] family weights (alpha=%.2f): %s",
            config.family_balance_alpha,
            {k: round(float(v), 2) for k, v in enumerate(family_weights.tolist()) if v},
        )
    return ForecastObjective(
        bundle_invariant=config.bundle_invariant_loss,
        family_weights=family_weights,
        token_types=token_types,
        time_weight=config.time_weight if config.time_to_event else 0.0,
        event_hazard_weight=config.event_hazard_weight if config.event_hazards else 0.0,
    )


def evaluate_streaming(
    model: SequenceModel,
    make_sampler: Callable[[], PackedLaneSampler],
    labels: ConceptLabelDict,
    masks: ConceptLabelDict,
    *,
    device: str,
    max_chunks: Optional[int] = None,
    supervision: ConceptSupervision = "stay",
    objective: Optional[ForecastObjective] = None,
    event_tables: Optional[EventTimeTables] = None,
) -> Dict[str, float]:
    """Average loss components over one (partial), gradient-free sampler pass."""
    model.eval()
    sampler = make_sampler()
    state = None
    totals: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i, chunk in enumerate(sampler):
            if max_chunks is not None and i >= max_chunks:
                break
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            event_targets = (
                event_hazard_targets(chunk, event_tables)
                if event_tables is not None
                else None
            )
            if isinstance(model, BaselineSequenceModel):
                _, components, state = model.compute_streaming_loss(
                    chunk, state=state, objective=objective, event_targets=event_targets
                )
            else:
                _, components, state = model.compute_streaming_loss(
                    chunk,
                    labels,
                    masks,
                    state=state,
                    supervision=supervision,
                    objective=objective,
                    event_targets=event_targets,
                )
            state = _detach_state(state)
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            n += 1
    model.train()
    return {key: value / max(n, 1) for key, value in totals.items()}


def _labels_to_device(labels: ConceptLabelDict, device: str) -> ConceptLabelDict:
    """Move every label tensor to ``device``, preserving the dict's key type."""
    return {k: v.to(device) for k, v in labels.items()}  # type: ignore[return-value]


def _combined_val_loss(
    components: Dict[str, float],
    weights: ConceptBottleneckLossWeights,
    time_weight: float = 0.0,
    event_hazard_weight: float = 0.0,
) -> float:
    """Compute the same task + weighted-auxiliary combination the training loss uses.

    ``evaluate_streaming``'s returned dict has the four loss components
    averaged separately, not combined -- this applies
    :func:`~odyssey.models.concept_bottleneck.combined_loss`'s exact
    weighting to them, so "is validation performance improving" tracks
    the actual thing being optimized, not just next-token loss alone.
    """
    return (
        components["task_loss"]
        + time_weight * components.get("time_loss", 0.0)
        + event_hazard_weight * components.get("event_loss", 0.0)
        + weights.concept * components.get("concept_loss", 0.0)
        + weights.orthogonality * components.get("orthogonality_loss", 0.0)
        + weights.observability * components.get("observability_loss", 0.0)
    )


def train(config: TrainingConfig) -> Path:  # noqa: PLR0912, PLR0915
    """Run one full training job; returns the output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)

    logger.info("[data] loading train shards from %s", config.train_shard_dir)
    train_events = load_meds_shards(
        config.train_shard_dir, max_shards=config.max_train_shards
    )
    logger.info(
        "[data] train: %d subjects, %d events",
        count_subjects(train_events),
        train_events.height,
    )

    logger.info("[data] loading tuning shards from %s", config.tuning_shard_dir)
    tuning_events = load_meds_shards(
        config.tuning_shard_dir, max_shards=config.max_tuning_shards
    )
    logger.info(
        "[data] tuning: %d subjects, %d events",
        count_subjects(tuning_events),
        tuning_events.height,
    )

    train_events = maybe_normalize(
        train_events, enabled=config.normalize_medications, source=config.source
    )
    tuning_events = maybe_normalize(
        tuning_events, enabled=config.normalize_medications, source=config.source
    )
    train_events = maybe_history_recap(train_events, enabled=config.history_recap)
    tuning_events = maybe_history_recap(tuning_events, enabled=config.history_recap)
    if config.history_recap:
        logger.info("[data] prior-diagnosis history recap injected at admissions")
    if config.normalize_medications:
        logger.info("[data] medication codes normalized to ingredient level")

    concepts = concepts_for_source(config.source)
    logger.info(
        "[data] labeling %d concepts (%s-scoped, source=%s)",
        len(concepts),
        config.concept_supervision,
        config.source,
    )
    train_labels: ConceptLabelDict
    train_masks: ConceptLabelDict
    tuning_labels: ConceptLabelDict
    tuning_masks: ConceptLabelDict
    train_first_times: ConceptLabelDict = {}
    if config.concept_supervision == "visit":
        train_labels, train_masks = build_visit_concept_label_dicts(
            train_events, concepts
        )
        tuning_labels, tuning_masks = build_visit_concept_label_dicts(
            tuning_events, concepts
        )
        if config.randint_prob > 0.0:
            train_first_times = build_visit_concept_first_times(train_events, concepts)
    elif config.concept_supervision == "stay":
        train_labels, train_masks = build_concept_label_dicts(train_events, concepts)
        tuning_labels, tuning_masks = build_concept_label_dicts(tuning_events, concepts)
        if config.randint_prob > 0.0:
            train_first_times = build_concept_first_times(train_events, concepts)
    else:
        raise ValueError(
            f"concept_supervision must be 'visit' or 'stay', got "
            f"{config.concept_supervision!r}"
        )
    train_event_tables: Optional[EventTimeTables] = None
    tuning_event_tables: Optional[EventTimeTables] = None
    if config.event_hazards:
        logger.info("[data] computing alert-event onset/censoring times")
        event_names = [a.name for a in ALERT_EVENTS]
        train_event_tables = EventTimeTables(
            all_event_times(train_events, ALERT_EVENTS, config.source), event_names
        )
        tuning_event_tables = EventTimeTables(
            all_event_times(tuning_events, ALERT_EVENTS, config.source), event_names
        )
    train_labels = _labels_to_device(train_labels, device)
    train_masks = _labels_to_device(train_masks, device)
    train_first_times = _labels_to_device(train_first_times, device)
    tuning_labels = _labels_to_device(tuning_labels, device)
    tuning_masks = _labels_to_device(tuning_masks, device)
    if config.randint_prob > 0.0:
        logger.info(
            "[loss] intervention-aware training on: RandInt prob %.2f over %d "
            "%s-scoped first-trigger records",
            config.randint_prob,
            len(train_first_times),
            config.concept_supervision,
        )

    logger.info("[data] fitting quantile binner on train split")
    binner = QuantileBinner.fit(
        train_events, n_bins=config.quantile_n_bins, min_count=config.quantile_min_count
    )
    binner.save(output_dir / "quantile_binner.json")
    train_events_binned = add_value_tokens(train_events, binner, source=config.source)
    tuning_events_binned = add_value_tokens(tuning_events, binner, source=config.source)
    # The unbinned events are never needed again -- everything for the
    # rest of this (potentially many-hour) run reads from the binned
    # copies instead. Without this, both copies of every split stay
    # alive as live locals for the whole run: 100 train shards alone
    # was enough to OOM-kill an 83GB VM (confirmed via dmesg) before
    # this fix.
    del train_events, tuning_events
    gc.collect()

    logger.info("[data] building vocabulary from train split")
    vocab = build_vocabulary(
        train_events_binned,
        min_count=config.vocab_min_count,
        max_size=config.vocab_max_size,
        backoff=config.vocab_backoff,
    )
    vocab.save(output_dir / "vocabulary.json")
    logger.info("[data] vocab size: %d", len(vocab))

    if config.resume_from is not None:
        # The checkpoint is the authority on which heads exist: a run
        # started before time_to_event / event_hazards existed must not be
        # rebuilt with heads its weights do not have (mirrors load_run).
        resume_keys = torch.load(config.resume_from, map_location="cpu")["model"].keys()
        config.time_to_event = any(k.startswith("time_head.") for k in resume_keys)
        config.event_hazards = any(k.startswith("event_heads.") for k in resume_keys)
        del resume_keys
    objective = build_objective(config, vocab, train_events_binned, device)
    model = build_model(config, vocab_size=len(vocab), num_concepts=len(concepts)).to(
        device
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[model] %.1fM parameters on %s", n_params / 1e6, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = None
    if config.concept_pos_weight and not train_labels:
        logger.warning(
            "[loss] concept_pos_weight requested but no %s-scoped labels were "
            "produced (e.g. no real hadm_id on any event) -- falling back to "
            "unweighted concept loss",
            config.concept_supervision,
        )
    elif config.concept_pos_weight:
        all_labels = torch.stack(list(train_labels.values()))
        all_masks = torch.stack(list(train_masks.values()))
        n_pos = (all_labels * all_masks).sum(dim=0)
        n_obs = all_masks.sum(dim=0)
        pos_weight = ((n_obs - n_pos) / n_pos.clamp_min(1.0)).clamp(0.2, 10.0)
        logger.info(
            "[loss] concept pos_weight: %s",
            [round(float(w), 2) for w in pos_weight],
        )
    loss_weights = ConceptBottleneckLossWeights(
        concept=config.concept_weight,
        orthogonality=config.orthogonality_weight,
        observability=config.observability_weight,
        concept_pos_weight=pos_weight,
    )

    # Fields that determine what PackedLaneSampler.next_chunk() actually
    # produces at a given position, for a given epoch's seed -- saved
    # alongside every periodic checkpoint so a resume can tell whether
    # fast-forwarding to the checkpoint's steps_into_epoch would land on
    # the same position it was taken at, or a different one (e.g. this
    # run manually restarted with a different num_lanes/chunk_size,
    # which is exactly what happened partway through the run that
    # motivated this: batch size was tuned up for GPU utilization mid
    # -training). Deliberately excludes model/optimizer hyperparameters
    # (learning_rate etc.), which don't affect the data stream.
    def _batch_config_fields(cfg: TrainingConfig) -> Dict[str, object]:
        return {
            "num_lanes": cfg.num_lanes,
            "chunk_size": cfg.chunk_size,
            "reset_prob": cfg.reset_prob,
            "seed": cfg.seed,
        }

    start_epoch = 0
    global_step = 0
    steps_into_epoch = 0
    best_val_loss = float("inf")
    evals_without_improvement = 0
    if config.resume_from is not None:
        logger.info("[resume] loading %s", config.resume_from)
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        evals_without_improvement = checkpoint.get("evals_without_improvement", 0)
        saved_steps_into_epoch = checkpoint.get("steps_into_epoch", 0)
        saved_batch_config = checkpoint.get("batch_config")
        if saved_steps_into_epoch > 0 and saved_batch_config != _batch_config_fields(
            config
        ):
            logger.warning(
                "[resume] batch config changed since this checkpoint (%s -> %s); "
                "restarting epoch %d from its own beginning instead of "
                "fast-forwarding to a now-meaningless position",
                saved_batch_config,
                _batch_config_fields(config),
                start_epoch,
            )
        else:
            steps_into_epoch = saved_steps_into_epoch
        logger.info(
            "[resume] resuming at epoch=%d, global_step=%d, steps_into_epoch=%d",
            start_epoch,
            global_step,
            steps_into_epoch,
        )

    def make_train_sampler(epoch: int) -> PackedLaneSampler:
        patients = iter_patient_sequences(
            train_events_binned,
            vocab,
            max_seq_len=config.max_seq_len,
            shuffle_seed=config.seed + epoch,
        )
        return PackedLaneSampler(
            patients,
            num_lanes=config.num_lanes,
            chunk_size=config.chunk_size,
            reset_prob=config.reset_prob,
            seed=config.seed + epoch,
        )

    def make_tuning_sampler() -> PackedLaneSampler:
        patients = iter_patient_sequences(
            tuning_events_binned, vocab, max_seq_len=config.max_seq_len
        )
        return PackedLaneSampler(
            patients, num_lanes=config.num_lanes, chunk_size=config.chunk_size
        )

    loss_logger = LossLogger(output_dir / "loss_log.jsonl")
    start_time = time.time()
    stop_early = False
    randint_rng = torch.Generator(device=device).manual_seed(config.seed + 1)

    for epoch in range(start_epoch, config.num_epochs):
        sampler = make_train_sampler(epoch)
        state = None
        steps_this_epoch = 0
        if epoch == start_epoch and steps_into_epoch > 0:
            logger.info(
                "[resume] fast-forwarding %d chunks (no gradient steps) to reach "
                "the resume position",
                steps_into_epoch,
            )
            for _ in range(steps_into_epoch):
                if sampler.next_chunk() is None:
                    break
            steps_this_epoch = steps_into_epoch

        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            event_targets = (
                event_hazard_targets(chunk, train_event_tables)
                if train_event_tables is not None
                else None
            )
            if isinstance(model, BaselineSequenceModel):
                total, components, state = model.compute_streaming_loss(
                    chunk, state=state, objective=objective, event_targets=event_targets
                )
            else:
                intervention = randint_intervention(
                    chunk,
                    train_labels,
                    train_masks,
                    train_first_times,
                    supervision=config.concept_supervision,  # type: ignore[arg-type]
                    num_concepts=len(concepts),
                    prob=config.randint_prob,
                    generator=randint_rng,
                )
                total, components, state = model.compute_streaming_loss(
                    chunk,
                    train_labels,
                    train_masks,
                    state=state,
                    loss_weights=loss_weights,
                    supervision=config.concept_supervision,  # type: ignore[arg-type]
                    intervention=intervention,
                    objective=objective,
                    event_targets=event_targets,
                )
            optimizer.zero_grad()
            total.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            state = _detach_state(state)
            global_step += 1
            steps_this_epoch += 1

            if global_step % config.log_every == 0:
                fields = {
                    "step": global_step,
                    "epoch": epoch,
                    "elapsed_s": time.time() - start_time,
                    "split": "train",
                    **{k: v.item() for k, v in components.items()},
                }
                loss_logger.log(**fields)
                summary = " ".join(f"{k}={v:.4f}" for k, v in components.items())
                logger.info("[train] step=%d epoch=%d %s", global_step, epoch, summary)

            if global_step % config.eval_every == 0:
                val = evaluate_streaming(
                    model,
                    make_tuning_sampler,
                    tuning_labels,
                    tuning_masks,
                    device=device,
                    max_chunks=config.eval_max_chunks,
                    supervision=config.concept_supervision,  # type: ignore[arg-type]
                    objective=objective,
                    event_tables=tuning_event_tables,
                )
                fields = {
                    "step": global_step,
                    "epoch": epoch,
                    "elapsed_s": time.time() - start_time,
                    "split": "tuning",
                    **val,
                }
                loss_logger.log(**fields)
                summary = " ".join(f"{k}={v:.4f}" for k, v in val.items())
                logger.info("[val]   step=%d %s", global_step, summary)

                val_loss = _combined_val_loss(
                    val,
                    loss_weights,
                    objective.time_weight,
                    objective.event_hazard_weight,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    evals_without_improvement = 0
                    _atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": global_step,
                            "epoch": epoch,
                            "steps_into_epoch": steps_this_epoch,
                            "batch_config": _batch_config_fields(config),
                            "best_val_loss": best_val_loss,
                            "evals_without_improvement": evals_without_improvement,
                            "config": asdict(config),
                        },
                        output_dir / "checkpoint_best.pt",
                    )
                    logger.info(
                        "[best]  new best val_loss=%.4f at step=%d",
                        best_val_loss,
                        global_step,
                    )
                else:
                    evals_without_improvement += 1
                    if config.early_stopping_patience is not None:
                        logger.info(
                            "[early-stop] %d/%d evals without improvement "
                            "(best=%.4f, current=%.4f)",
                            evals_without_improvement,
                            config.early_stopping_patience,
                            best_val_loss,
                            val_loss,
                        )

                if (
                    config.early_stopping_patience is not None
                    and evals_without_improvement >= config.early_stopping_patience
                ):
                    logger.info(
                        "[early-stop] stopping: no improvement in %d consecutive evals",
                        config.early_stopping_patience,
                    )
                    stop_early = True
                    break

            if global_step % config.checkpoint_every == 0:
                _atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "steps_into_epoch": steps_this_epoch,
                        "batch_config": _batch_config_fields(config),
                        "best_val_loss": best_val_loss,
                        "evals_without_improvement": evals_without_improvement,
                        "config": asdict(config),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )

        if stop_early:
            break

        # One checkpoint per completed epoch, independent of
        # checkpoint_every -- steps_into_epoch=0 here since resuming
        # from this checkpoint starts the *next* epoch at its own
        # beginning, not partway through this one.
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "evals_without_improvement": evals_without_improvement,
                "config": asdict(config),
            },
            output_dir / f"checkpoint_epoch_{epoch}.pt",
        )

    _atomic_torch_save(
        {"model": model.state_dict(), "step": global_step, "config": asdict(config)},
        output_dir / "checkpoint_final.pt",
    )
    loss_logger.close()
    elapsed = time.time() - start_time
    logger.info(
        "[done] %d steps in %.1fs (early_stop=%s, best_val_loss=%.4f), output in %s",
        global_step,
        elapsed,
        stop_early,
        best_val_loss,
        output_dir,
    )
    return output_dir


def _parse_args() -> TrainingConfig:
    """Build a :class:`TrainingConfig` from the required paths plus optional overrides.

    Every other hyperparameter is a plain code-level default on
    :class:`TrainingConfig`; ``--config-json`` accepts a JSON file with
    any subset of field overrides rather than one CLI flag per field --
    a hand-rolled per-field argparse builder from dataclass reflection is
    fragile for ``Optional[int]``-typed fields specifically (their
    ``field.type`` isn't a plain ``int``/``str``/``float`` to dispatch
    an argparse ``type=`` on), so this avoids that entirely.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard-dir", required=True)
    parser.add_argument("--tuning-shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to a JSON file with any subset of TrainingConfig field overrides.",
    )
    args = parser.parse_args()

    overrides = (
        json.loads(Path(args.config_json).read_text()) if args.config_json else {}
    )
    return TrainingConfig(
        train_shard_dir=args.train_shard_dir,
        tuning_shard_dir=args.tuning_shard_dir,
        output_dir=args.output_dir,
        **overrides,
    )


if __name__ == "__main__":
    # Only the top-level entry point configures logging -- odyssey.training.train
    # is also imported as a library (tests, odyssey.inference), which must
    # never clobber a caller's own logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train(_parse_args())
