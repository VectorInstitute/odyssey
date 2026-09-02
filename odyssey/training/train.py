"""End-to-end training: real MEDS shards -> a trained ConceptBottleneckSequenceModel.

Usage (on a CUDA host, from the repo root)::

    uv run python -m odyssey.training.train \\
        --train-shard-dir /path/to/data/train \\
        --tuning-shard-dir /path/to/data/tuning \\
        --output-dir runs/exp1 \\
        --config-json '{"max_train_shards": 20, "max_tuning_shards": 5}'

Those four are the only real CLI flags; every other
:class:`TrainingConfig` field is set through ``--config-json`` (inline
JSON or a path to a JSON file), never through its own flag.

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
import itertools
import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

import polars as pl
import torch

from odyssey.data.alert_events import all_event_times, hazard_events_for
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import AnyConceptDefinition, concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.packed_context import PackedContextSampler
from odyssey.data.sequences import PatientSequence
from odyssey.data.sidecars import activate_sidecars, active_sidecar_names
from odyssey.data.streaming import PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import CLIP_TAIL, QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.concept_bottleneck import (
    ConceptBottleneckLossWeights,
    annealed_alpha,
)
from odyssey.models.injection import middle_layer
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
    ForecastObjective,
    SequenceModel,
)
from odyssey.models.steering import steering_direction, steering_gamma
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
from odyssey.training.lifted_tokens import lifted_token_sets
from odyssey.training.running_labels import randint_intervention
from odyssey.training.shard_stream import (
    build_corpus_stats,
    family_loss_weights_from_counts,
    fit_binner_streaming,
    iter_patients_streaming,
    make_preparer,
    shard_paths,
)
from odyssey.training.steering_phase import (
    Injection,
    SteeringSchedule,
    choose_injection,
)
from odyssey.utils.env_fingerprint import write_run_provenance


logger = logging.getLogger(__name__)

#: Either sampler this loop can drive: PackedLaneSampler for the recurrent
#: hybrid backbone (persistent lanes, carried state), PackedContextSampler
#: for the stateless transformer backbone (whole/truncated patients packed
#: per row, no carried state). Both yield StreamingChunk, so the
#: training/eval loop bodies below are backbone-agnostic; only sampler
#: *construction* differs, in make_train_sampler/make_tuning_sampler.
StreamingSampler = PackedLaneSampler | PackedContextSampler


@dataclass
class TrainingConfig:
    """All paths and hyperparameters for one training run."""

    train_shard_dir: str
    tuning_shard_dir: str
    output_dir: str
    max_train_shards: int | None = None
    max_tuning_shards: int | None = None
    resume_from: str | None = None

    model_kind: str = "bottleneck"
    """``"bottleneck"`` (ConceptBottleneckSequenceModel, the interpretable
    model this project is about) or ``"baseline"`` (BaselineSequenceModel:
    the same backbone and forecasting/time/event heads with no concept
    bottleneck and no concept supervision). Train both with identical
    settings to price the bottleneck: the README's "costs little" claim
    is measured, not assumed."""

    backbone: str = "hybrid"
    """``"hybrid"`` (EHRHybridBackbone, Mamba-2 + attention, this project's
    own architecture) or ``"transformer"``
    (odyssey.models.backbones.transformer.TransformerBackbone, the
    modern-vanilla decoder-only control -- roadmap Track A item 5). Both
    share every downstream head/loss; this prices the backbone choice the
    way model_kind prices the bottleneck. The transformer backbone is
    stateless, so this loop drives it with
    odyssey.data.packed_context.PackedContextSampler (whole/truncated
    patients packed per row, num_lanes rows per step, no carried state)
    instead of PackedLaneSampler's TBTT chunking (which "hybrid" still
    uses)."""

    max_context: int = 4096
    """Token budget per packed row for backbone="transformer" (see
    PackedContextSampler). Unused by the hybrid backbone."""

    # Backbone (EHRHybridBackbone). Defaults are modest, not the paper-scale
    # numbers -- see the training run's own README note on why.
    hidden_size: int = 256
    num_hidden_layers: int = 8
    value_embeddings: bool = False
    """Feed standardized numeric values (``aux.values``) into the token
    embeddings alongside the bin tokens (see
    :class:`~odyssey.models.embeddings.ClinicalEventEmbeddings`). Opt-in;
    an A/B against the bin-only input."""
    event_head_hidden: int = 0
    """Hidden width of the per-event hazard heads' MLP readout; 0 = the
    single linear layer every run before v8 used."""
    stream_shards: bool = False
    """Prepare the training split shard by shard (vocabulary, binner,
    concept labels, event times and the per-epoch token stream) instead
    of loading it whole: required at full-extraction scale (292 MIMIC-IV
    shards OOM-killed an 83 GB host in concept labeling). Tuning shards
    stay in memory. See :mod:`odyssey.training.shard_stream`."""
    task_set: str = "v1"
    """Which concept registry + alert-event set this run supervises
    (:data:`odyssey.data.concepts.TASK_SETS`,
    :func:`odyssey.data.alert_events.alert_events_for`). "v1" = the 15
    concepts / 4 alerts every run before Aug 23 2026 used; "v2" adds the
    Sepsis-3 concept + alert and 30-day readmission. Saved in config.json
    so evaluation rebuilds exactly this run's heads and labels; needs the
    microbiology sidecar next to the data for sepsis3 (see
    :mod:`odyssey.data.sidecars`)."""
    concept_global_pairs: bool = False
    bottleneck_kind: str = "mixture"
    """'mixture' (CEM-style per-concept embedding pairs) or 'decomposed'
    (Steerling: h splits into known concepts, unknown concepts and a
    residual). 'decomposed' ignores embedding_dim, unknown_dim and
    concept_global_pairs, and uses unknown_ratio/unknown_rank/
    residual_dropout instead."""
    unknown_ratio: int = 3
    """Decomposed bottleneck: m = unknown_ratio * n unknown concepts."""
    unknown_rank: int | None = None
    """Decomposed bottleneck: factorize U = A @ B at this rank, or None
    for the full matrix. Steerling factorizes because m is 101k; at our
    n a full matrix is a few tens of thousands of parameters."""
    residual_dropout: float = 0.1
    """Decomposed bottleneck: dropout on the unexplained residual eps
    during training (Steerling's p_eps). This is the direct counter-
    pressure to residual domination; they raise it to 0.3 when
    tightening a trained model."""
    """Leakage control: input-independent (w+, w-) per known concept, so a
    concept slot carries only its probability (see ConceptBottleneck)."""
    unknown_dim: int | None = None
    """Width of the unknown (residual) slot; None = embedding_dim."""
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
    vocab_backoff: str | None = "icd3"
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
    max_seq_len: int | None = None

    # Streaming TBTT
    num_lanes: int = 8
    chunk_size: int = 256
    reset_prob: float = 0.0

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_epochs: int = 3
    concept_weight: float = 1.0
    orthogonality_weight: float = 0.1
    observability_weight: float = 0.1
    teacher_known_start: float = 0.0
    teacher_known_end: float = 0.0
    teacher_unknown_start: float = 0.0
    teacher_unknown_end: float = 0.0
    teacher_anneal_steps: int = 4500
    """Steerling's concept teacher forcing, off by default so an existing
    config keeps its behaviour. Their known head anneals 1.0 -> 0.5
    (cosine) over the first 10% of steps. Their unknown head is described
    as annealing FROM 1 in the prose and as 0.0 -> 0.5 in Table 26; the
    two disagree, so both endpoints are explicit here and whichever is
    used gets recorded with the run. The ramp length is absolute because
    this loop never knows max_steps; 4,500 is their tenth at our run
    lengths."""
    reconstruction_weight: float = 1.0
    """Decomposed bottleneck only: Steerling's lambda_rec."""
    independence_weight: float = 1.0
    """Decomposed bottleneck only: Steerling's lambda_indep."""
    steering_phases: int = 0
    """Steerling's steering training (their 10.2.4), off by default: this
    many consecutive phases of ``steering_phase_steps`` steps, starting
    after ``steering_warmup_steps``, in which a concept's direction is
    injected at the positions its running label covers and the respond
    and express losses are added to the forecasting loss. Meant for a
    short mid-training run started with ``init_from``."""
    steering_phase_steps: int = 0
    steering_warmup_steps: int = 100
    steering_gamma: float = 1.0
    """Injection strength during steering phases (their Table 36: 1.0)."""
    steering_tau: float | None = None
    """If set, calibrate the strength per concept as tau / peak(e_c)
    (their Eq. 19) instead of using ``steering_gamma`` for every concept."""
    steering_layer_index: int | None = None
    """First backbone block whose output is pushed; ``None`` = the middle."""
    respond_weight: float = 1.0
    express_weight: float = 1.0
    steering_forecast_at_injected: bool = True
    """Score the forecasting losses on injected positions during steering
    phases (Steerling's Eq. 33 as written). ``False`` scores them on the
    other real positions only, so an injected position is trained to
    respond and express and not also to leave its forecast unchanged."""
    lifted_top_k: int = 25
    lifted_min_count: int = 20
    lifted_min_share: float = 0.005
    lifted_min_lift: float = 2.0
    lifted_patients: int = 2000
    """How many training patients the lifted token sets are counted over."""
    task_weight: float = 1.0
    """Weight of the forecasting (task) loss; see
    ConceptBottleneckLossWeights.task. 0.0, paired with init_from and
    an unweighted run to resume from, is stage one of independent
    (Koh et al. 2020) training: shape the bottleneck from concept
    supervision alone, no forecast gradient."""

    freeze_bottleneck: bool = False
    """Stop gradient at the bottleneck: only lm_head/time_head/event_heads
    are optimized, backbone and bottleneck parameters are frozen. Stage
    two of independent training, paired with init_from pointing at a
    task_weight=0.0 checkpoint and randint_prob=1.0 (so the task heads
    train against the ground-truth concept mixture, never the model's
    own predicted probability -- Koh et al.'s classical "train f on
    true c" step, adapted to CEM's embedding mixture)."""

    init_from: str | None = None
    """Load a checkpoint's model weights only (not optimizer/step/epoch
    state) before training starts, unlike resume_from which continues
    an interrupted run of the same config. Use with freeze_bottleneck
    to start stage two of independent training from a stage-one
    checkpoint."""

    unfreeze_top_backbone_layers: int = 0
    """With freeze_bottleneck, re-enable gradient on the last N blocks
    of the backbone's layer stack after the full freeze, so stage two
    can partially re-shape the backbone instead of only training the
    task heads. The bottleneck itself stays frozen regardless (this
    only ever affects backbone.layers). 0 (default) reproduces plain
    freeze_bottleneck exactly. A middle ground between "fully frozen"
    (independent training's Koh et al. purity) and "fully joint"
    (ordinary training): tests whether a little backbone re-shaping
    recovers forecasting power without losing the intervention gain."""

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

    auxiliary_event_names: tuple[str, ...] = ()
    """Extra events (from odyssey.data.alert_events.COUNTING_AUXILIARY_EVENTS_BY_NAME)
    trained by the SAME per-event hazard heads as event_hazards' curated
    events, but never scored as alerts (odyssey.inference.alerts iterates
    alert_events_for(task_set) directly, which never includes these) --
    an auxiliary training signal only. See hazard_events_for. Empty (the
    default) reproduces today's event_names exactly for every existing
    config/checkpoint."""

    value_head: bool = False
    """Add the next-event value-quantile head
    (odyssey.models.value_head.ValueQuantileHead): K=9 quantiles of the
    next event's standardized value, conditioned on the target token's
    own embedding, trained with pinball loss masked to positions whose
    target carries a value. Off by default -- every existing run and
    checkpoint stays unaffected; this is purely additive alongside the
    bin-token representation of value, which is unchanged either way."""

    value_tail_transform: str = CLIP_TAIL
    """How the standardized input value's tail is treated:
    ``"clip"`` (the default, and what every run before 2026-08-24 used)
    saturates at ``+-VALUE_Z_CLIP``, ``"symlog"`` compresses it while
    staying strictly monotone. The scale is robust (IQR / 1.349), so the
    clip threshold sits inside the clinically abnormal range for skewed
    labs; see :func:`odyssey.data.value_binning._tail_expr`. Saved on the
    run's ``quantile_binner.json``, so evaluation reproduces it without
    needing this flag."""

    value_head_hidden: int = 0
    """Hidden width of the value head's MLP readout; 0 = the single linear
    layer arm B ran, whose own calibration was poor (see
    :class:`~odyssey.models.value_head.ValueQuantileHead`)."""

    value_head_weight: float = 1.0

    value_fourier: bool = False
    """Only meaningful with value_embeddings=True: encode the standardized
    input value as Fourier features (odyssey.models.embeddings.value_features_fourier)
    instead of [z, z^2, has] before the value projection. Independent of
    value_head -- lets an A/B separate "better input encoding" from
    "better output objective"."""

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

    early_stopping_patience: int | None = None
    """Stop once the combined validation loss (the same task + weighted
    concept/orthogonality/observability combination compute_streaming_loss
    trains against, evaluated on the tuning split) hasn't improved for
    this many consecutive ``eval_every`` checks. ``None`` disables early
    stopping -- the run always did every configured epoch before this
    existed, so opt-in keeps that the default. Every improvement saves
    ``checkpoint_best.pt``, independent of ``checkpoint_every``."""

    seed: int = 0

    def __post_init__(self) -> None:
        """Validate fields whose bad values would otherwise fail deep into a run.

        ``checkpoint_every=0`` in particular used to raise
        ``ZeroDivisionError`` from ``global_step % config.checkpoint_every``
        the first time a checkpoint was due, hours into a real run rather
        than at config-load time.
        """
        if self.checkpoint_every < 1:
            raise ValueError(
                f"checkpoint_every must be >= 1, got {self.checkpoint_every}"
            )


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
    :class:`~odyssey.models.backbones.hybrid.HybridState`, the plain
    tuple-of-tensors state lighter backbones (e.g. ``TinyGRUBackbone``)
    return, and ``None`` for stateless backbones (``TransformerBackbone``:
    ``state`` is always ignored, so there is nothing to detach but
    ``prev_time_stamps``, itself unused since that backbone never carries
    a delta across calls -- detaching it anyway costs nothing and keeps
    this function's contract "safe to call unconditionally after every
    chunk" true for every backbone, not just the recurrent ones). A new
    backbone with a different state shape must be added here explicitly
    -- silently not detaching would leak the autograd graph across every
    chunk of an epoch.
    """
    from odyssey.models.backbones.hybrid import HybridState  # noqa: PLC0415

    recurrent = state.recurrent
    detached_recurrent: object
    if recurrent is None:
        detached_recurrent = None
    elif isinstance(recurrent, HybridState):
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


def _activate_run_sidecars(config: TrainingConfig) -> None:
    """Activate the label sidecars next to the training data; guard sepsis3.

    A run that supervises sepsis3 needs the microbiology sidecar:
    silently training with it absent would mask the concept everywhere
    and leave an untrained sepsis head, so refuse loudly with the fix
    (build it with scripts/build_mimic_sidecars.py and place it at
    ``<root>/sidecars/`` next to ``data/``). The guard is
    source-resolved: on a source where sepsis3 does not resolve (eICU),
    the same task_set trains without the sidecar.
    """
    names = activate_sidecars(config.train_shard_dir)
    if names:
        logger.info("[data] sidecars active: %s", ", ".join(names))
    resolved = {
        concept.name
        for concept in concepts_for_source(config.source, task_set=config.task_set)
    }
    if "sepsis3" in resolved and "microbiology" not in active_sidecar_names():
        raise FileNotFoundError(
            f"task_set={config.task_set!r} on source={config.source!r} "
            "supervises sepsis3, which needs the 'microbiology' sidecar, "
            f"but none was found next to {config.train_shard_dir} "
            "-- build it with scripts/build_mimic_sidecars.py and place it at "
            "<root>/sidecars/microbiology.parquet (sibling of data/)."
        )


def build_model(
    config: TrainingConfig, *, vocab_size: int, num_concepts: int
) -> SequenceModel:
    """Construct the real backbone + heads from ``config`` (see ``model_kind``)."""
    from odyssey.models.backbones.base import SequenceBackbone  # noqa: PLC0415
    from odyssey.models.backbones.hybrid import EHRHybridBackbone  # noqa: PLC0415
    from odyssey.models.backbones.transformer import (  # noqa: PLC0415
        TransformerBackbone,
    )

    kind = getattr(config, "model_kind", "bottleneck")
    if kind not in ("bottleneck", "baseline"):
        raise ValueError(f"model_kind must be 'bottleneck' or 'baseline', got {kind!r}")
    backbone_kind = getattr(config, "backbone", "hybrid")
    backbone: SequenceBackbone
    if backbone_kind == "hybrid":
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
            use_value_fourier=bool(getattr(config, "value_fourier", False)),
        )
    elif backbone_kind == "transformer":
        backbone = TransformerBackbone(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            padding_idx=PAD_ID,
            num_hidden_layers=config.num_hidden_layers,
            num_heads=config.attn_num_heads,
            use_values=bool(getattr(config, "value_embeddings", False)),
            use_value_fourier=bool(getattr(config, "value_fourier", False)),
        )
    else:
        raise ValueError(
            f"backbone must be 'hybrid' or 'transformer', got {backbone_kind!r}"
        )
    time_bin_edges = (
        DEFAULT_TIME_BIN_EDGES_HOURS
        if getattr(config, "time_to_event", False)
        else None
    )
    event_names = (
        [
            a.name
            for a in hazard_events_for(
                getattr(config, "task_set", "v1"),
                getattr(config, "auxiliary_event_names", ()),
                source=getattr(config, "source", None),
            )
        ]
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
            value_head=bool(getattr(config, "value_head", False)),
            value_head_hidden=int(getattr(config, "value_head_hidden", 0) or 0),
            source=getattr(config, "source", "mimic_iv"),
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
        unknown_dim=_checked_unknown_dim(config),
        bottleneck_kind=str(getattr(config, "bottleneck_kind", "mixture")),
        unknown_ratio=int(getattr(config, "unknown_ratio", 3)),
        unknown_rank=getattr(config, "unknown_rank", None),
        residual_dropout=float(getattr(config, "residual_dropout", 0.1)),
        value_head=bool(getattr(config, "value_head", False)),
        value_head_hidden=int(getattr(config, "value_head_hidden", 0) or 0),
        source=getattr(config, "source", "mimic_iv"),
    )


@dataclass
class _SteeringRuntime:
    """Everything a steering step needs, built once per run."""

    schedule: SteeringSchedule
    directions: torch.Tensor
    """(num_concepts, hidden) unit steering directions."""
    gammas: list[float]
    layer_index: int
    lifted: dict[int, torch.Tensor]
    labels: ConceptLabelDict
    masks: ConceptLabelDict
    first_times: ConceptLabelDict
    supervision: ConceptSupervision
    generator: torch.Generator

    def injection_for(self, chunk: StreamingChunk, step: int) -> Injection | None:
        """Return the chunk's target concept and positions; ``None`` outside a phase."""
        if not self.schedule.is_steering_step(step):
            return None
        return choose_injection(
            chunk,
            self.labels,
            self.masks,
            self.first_times,
            supervision=self.supervision,
            num_concepts=int(self.directions.shape[0]),
            generator=self.generator,
        )


def _prepare_steering(
    config: TrainingConfig,
    model: SequenceModel,
    corpus: "PreparedCorpus",
    device: str,
) -> "_SteeringRuntime | None":
    """Build the steering runtime when the config asks for steering phases."""
    schedule = SteeringSchedule(
        warmup_steps=config.steering_warmup_steps,
        phases=config.steering_phases,
        phase_steps=config.steering_phase_steps,
    )
    if not schedule.enabled:
        return None
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError("steering phases need a concept-bottleneck model")
    if not corpus.train_first_times:
        raise ValueError(
            "steering phases need concept first-trigger times (running labels); "
            "the corpus was prepared without them"
        )
    num_concepts = len(corpus.concepts)
    directions = torch.stack(
        [steering_direction(model, c) for c in range(num_concepts)]
    )
    if config.steering_tau is not None:
        gammas = [
            steering_gamma(model, directions[c], tau=config.steering_tau)
            for c in range(num_concepts)
        ]
    else:
        gammas = [config.steering_gamma] * num_concepts
    layer_index = (
        config.steering_layer_index
        if config.steering_layer_index is not None
        else middle_layer(model.backbone)
    )
    logger.info(
        "[steering] counting lifted tokens over %d training patients",
        config.lifted_patients,
    )
    lifted = lifted_token_sets(
        itertools.islice(
            corpus.make_train_patients(config.seed + 7919), config.lifted_patients
        ),
        vocab_size=len(corpus.vocab),
        num_concepts=num_concepts,
        concept_labels=corpus.train_labels,
        concept_mask=corpus.train_masks,
        concept_first_times=corpus.train_first_times,
        supervision=config.concept_supervision,  # type: ignore[arg-type]
        top_k=config.lifted_top_k,
        min_count=config.lifted_min_count,
        min_share=config.lifted_min_share,
        min_lift=config.lifted_min_lift,
        num_lanes=config.num_lanes,
        chunk_size=config.chunk_size,
        device=device,
    )
    names = [c.name for c in corpus.concepts]
    for c, ids in lifted.items():
        logger.info(
            "[steering] %s: %d lifted tokens, e.g. %s",
            names[c],
            len(ids),
            [corpus.vocab.id_to_token[i] for i in ids[:5]],
        )
    logger.info(
        "[steering] %d phases x %d steps after %d warmup steps; gamma %s; "
        "injecting from block %d",
        schedule.phases,
        schedule.phase_steps,
        schedule.warmup_steps,
        "calibrated" if config.steering_tau is not None else config.steering_gamma,
        layer_index,
    )
    return _SteeringRuntime(
        schedule=schedule,
        directions=directions.to(device),
        gammas=gammas,
        layer_index=layer_index,
        lifted={c: torch.tensor(ids, dtype=torch.long) for c, ids in lifted.items()},
        labels=corpus.train_labels,
        masks=corpus.train_masks,
        first_times=corpus.train_first_times,
        supervision=config.concept_supervision,  # type: ignore[arg-type]
        generator=torch.Generator().manual_seed(config.seed + 4243),
    )


def _needs_running_labels(config: TrainingConfig) -> bool:
    """Whether the corpus must carry concept first-trigger times.

    Both intervention-aware training (RandInt) and Steerling's steering
    phases substitute or inject at positions where a concept has already
    triggered, which only the running labels can say.
    """
    return config.randint_prob > 0.0 or config.steering_phases > 0


def optimizer_param_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    """Parameter groups for AdamW, with decay-exempt parameters at 0.0.

    A bottleneck may declare ``decay_exempt_parameters()``; the decomposed
    one exempts its concept embeddings, following Steerling's "weight
    decay ... excluding embeddings". Its known embeddings receive no task
    gradient in expectation, so decaying them is a steady pull toward an
    inert named channel rather than regularization. Every other parameter
    keeps ``weight_decay`` unchanged, so no other arm's behaviour moves.
    """
    exempt_fn = getattr(
        getattr(model, "bottleneck", None), "decay_exempt_parameters", None
    )
    exempt = {id(p) for p in (exempt_fn() if exempt_fn is not None else [])}
    trainable = [p for p in model.parameters() if p.requires_grad]
    decayed = [p for p in trainable if id(p) not in exempt]
    undecayed = [p for p in trainable if id(p) in exempt]
    groups: list[dict[str, object]] = [
        {"params": decayed, "weight_decay": weight_decay}
    ]
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
        logger.info(
            "[optim] %d parameter tensors (%.3fM values) exempt from weight decay",
            len(undecayed),
            sum(p.numel() for p in undecayed) / 1e6,
        )
    return groups


def _checked_unknown_dim(config: TrainingConfig) -> int | None:
    """Return ``config.unknown_dim``, warning if it silently voids the penalty.

    ``orthogonality_loss`` compares a known concept's embedding against the
    unknown slot's by cosine similarity, which is undefined when the two
    have different widths; it returns exactly zero in that case, by design
    (the width cap becomes the leakage control instead). The trap is that
    the term then vanishes *silently* while ``orthogonality_weight`` still
    reads as nonzero in the config and the run still logs an
    ``orthogonality_loss`` of 0.0 as though it were being optimized. Any
    sweep over ``unknown_dim`` therefore changes TWO things at once unless
    the weight is zeroed everywhere, which cost us a confounded comparison
    across the L1/L3/L4 arms. Warn loudly rather than let the next sweep
    inherit the same silence.
    """
    unknown_dim = getattr(config, "unknown_dim", None)
    weight = float(getattr(config, "orthogonality_weight", 0.0) or 0.0)
    if (
        unknown_dim is not None
        and int(unknown_dim) != int(config.embedding_dim)
        and weight > 0
    ):
        logger.warning(
            "[bottleneck] orthogonality_weight=%.3g is INERT: unknown_dim=%d "
            "!= embedding_dim=%d, so orthogonality_loss returns 0 by "
            "construction. This run is effectively unpenalized. For a clean "
            "unknown_dim sweep set orthogonality_weight=0.0 at EVERY point, "
            "including unknown_dim == embedding_dim, or the endpoint is the "
            "only penalized arm.",
            weight,
            int(unknown_dim),
            int(config.embedding_dim),
        )
    if (
        str(getattr(config, "bottleneck_kind", "mixture")) == "decomposed"
        and weight > 0
    ):
        # Same silent zero, different cause: combined_loss derives its
        # orthogonality term from the mixture's per-concept embedding
        # blocks, which the decomposition does not produce, so the term is
        # structurally 0 there too. independence_weight is the knob that
        # actually controls known/unknown redundancy in that design.
        logger.warning(
            "[bottleneck] orthogonality_weight=%.3g is INERT for "
            "bottleneck_kind='decomposed': that term is defined over the "
            "mixture's per-concept embedding blocks and returns 0 here. Use "
            "independence_weight, which penalizes known/unknown redundancy "
            "on activations, and set orthogonality_weight=0.0 so the logs "
            "stop implying an active penalty.",
            weight,
        )
    return unknown_dim


def build_objective(
    config: TrainingConfig,
    vocab: Vocabulary,
    train_events_binned: pl.DataFrame | None,
    device: str,
    *,
    code_counts: dict[str, int] | None = None,
) -> ForecastObjective:
    """Construct the :class:`ForecastObjective` a run trains and validates with.

    Family weights come from the binned training events, or from
    ``code_counts`` (the shard-streaming path) when no frame is held.
    """
    family_weights = None
    # token -> family is always needed: the bundle-invariant loss restricts
    # membership to the target's own family (see _bundle_log_likelihood).
    token_types = token_type_lookup(vocab).to(device)
    if config.family_balance_alpha > 0.0:
        n_families = int(token_types.max().item()) + 1
        if code_counts is not None:
            family_weights = family_loss_weights_from_counts(
                code_counts,
                alpha=config.family_balance_alpha,
                cap=config.family_weight_cap,
                n_families=n_families,
            ).to(device)
        else:
            assert train_events_binned is not None
            family_weights = family_loss_weights(
                train_events_binned,
                alpha=config.family_balance_alpha,
                cap=config.family_weight_cap,
                n_families=n_families,
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
        value_head_weight=(
            config.value_head_weight if getattr(config, "value_head", False) else 0.0
        ),
    )


def evaluate_streaming(
    model: SequenceModel,
    make_sampler: Callable[[], StreamingSampler],
    labels: ConceptLabelDict,
    masks: ConceptLabelDict,
    *,
    device: str,
    max_chunks: int | None = None,
    supervision: ConceptSupervision = "stay",
    objective: ForecastObjective | None = None,
    event_tables: EventTimeTables | None = None,
) -> dict[str, float]:
    """Average loss components over one (partial), gradient-free sampler pass."""
    model.eval()
    sampler = make_sampler()
    state = None
    totals: dict[str, float] = {}
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
    components: dict[str, float],
    weights: ConceptBottleneckLossWeights,
    time_weight: float = 0.0,
    event_hazard_weight: float = 0.0,
    value_head_weight: float = 0.0,
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
        + value_head_weight * components.get("value_loss", 0.0)
        + weights.concept * components.get("concept_loss", 0.0)
        + weights.orthogonality * components.get("orthogonality_loss", 0.0)
        + weights.observability * components.get("observability_loss", 0.0)
    )


def train(config: TrainingConfig) -> Path:  # noqa: PLR0912, PLR0915
    """Run one full training job; returns the output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume_from is not None:
        # The checkpoint is the authority on which heads exist: a run
        # started before time_to_event / event_hazards existed must not be
        # rebuilt with heads its weights do not have (mirrors load_run).
        # Runs BEFORE config.json is written and before the
        # stream_shards branch, so the streaming path adapts too and the
        # on-disk config.json records the heads this run actually trains.
        resume_keys = torch.load(config.resume_from, map_location="cpu")["model"].keys()
        config.time_to_event = any(k.startswith("time_head.") for k in resume_keys)
        config.event_hazards = any(k.startswith("event_heads.") for k in resume_keys)
        config.value_head = any(k.startswith("value_head.") for k in resume_keys)
        del resume_keys
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    _activate_run_sidecars(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)

    if config.stream_shards:
        return _train_streaming(config, output_dir, device)

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

    concepts = concepts_for_source(config.source, task_set=config.task_set)
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
        if _needs_running_labels(config):
            train_first_times = build_visit_concept_first_times(train_events, concepts)
    elif config.concept_supervision == "stay":
        train_labels, train_masks = build_concept_label_dicts(train_events, concepts)
        tuning_labels, tuning_masks = build_concept_label_dicts(tuning_events, concepts)
        if _needs_running_labels(config):
            train_first_times = build_concept_first_times(train_events, concepts)
    else:
        raise ValueError(
            f"concept_supervision must be 'visit' or 'stay', got "
            f"{config.concept_supervision!r}"
        )
    train_event_tables: EventTimeTables | None = None
    tuning_event_tables: EventTimeTables | None = None
    if config.event_hazards:
        logger.info("[data] computing alert-event onset/censoring times")
        alerts = hazard_events_for(
            config.task_set, config.auxiliary_event_names, source=config.source
        )
        event_names = [a.name for a in alerts]
        train_event_tables = EventTimeTables(
            all_event_times(
                train_events, alerts, config.source, task_set=config.task_set
            ),
            event_names,
        )
        tuning_event_tables = EventTimeTables(
            all_event_times(
                tuning_events, alerts, config.source, task_set=config.task_set
            ),
            event_names,
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
        train_events,
        n_bins=config.quantile_n_bins,
        min_count=config.quantile_min_count,
        tail_transform=config.value_tail_transform,
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

    objective = build_objective(config, vocab, train_events_binned, device)

    def make_train_patients(epoch: int) -> Iterator[PatientSequence]:
        return iter_patient_sequences(
            train_events_binned,
            vocab,
            max_seq_len=config.max_seq_len,
            shuffle_seed=config.seed + epoch,
        )

    corpus = PreparedCorpus(
        vocab=vocab,
        concepts=concepts,
        objective=objective,
        train_labels=train_labels,
        train_masks=train_masks,
        train_first_times=train_first_times,
        tuning_labels=tuning_labels,
        tuning_masks=tuning_masks,
        train_event_tables=train_event_tables,
        tuning_event_tables=tuning_event_tables,
        tuning_events_binned=tuning_events_binned,
        make_train_patients=make_train_patients,
    )
    return _run_training(config, output_dir, device, corpus)


@dataclass
class PreparedCorpus:
    """Everything the training loop needs, however the corpus was prepared."""

    vocab: Vocabulary
    concepts: Sequence[AnyConceptDefinition]
    objective: ForecastObjective
    train_labels: ConceptLabelDict
    train_masks: ConceptLabelDict
    train_first_times: ConceptLabelDict
    tuning_labels: ConceptLabelDict
    tuning_masks: ConceptLabelDict
    train_event_tables: EventTimeTables | None
    tuning_event_tables: EventTimeTables | None
    tuning_events_binned: pl.DataFrame
    make_train_patients: Callable[[int], Iterator[PatientSequence]]
    """epoch -> the training patient stream for that epoch."""


def _train_streaming(config: TrainingConfig, output_dir: Path, device: str) -> Path:
    """Shard-streaming corpus preparation (see :mod:`odyssey.training.shard_stream`)."""
    paths = shard_paths(config.train_shard_dir, config.max_train_shards)
    prepare = make_preparer(
        normalize_medications=config.normalize_medications,
        history_recap=config.history_recap,
        source=config.source,
    )
    concepts = concepts_for_source(config.source, task_set=config.task_set)
    logger.info("[stream] fitting quantile binner over %d train shards", len(paths))
    binner = fit_binner_streaming(
        paths,
        prepare,
        n_bins=config.quantile_n_bins,
        min_count=config.quantile_min_count,
        seed=config.seed,
        tail_transform=config.value_tail_transform,
    )
    binner.save(output_dir / "quantile_binner.json")
    logger.info("[stream] corpus statistics, concept labels and event times")
    stats = build_corpus_stats(
        paths,
        prepare,
        binner,
        source=config.source,
        concepts=concepts,
        concept_supervision=config.concept_supervision,
        with_first_times=_needs_running_labels(config),
        alerts=(
            hazard_events_for(
                config.task_set, config.auxiliary_event_names, source=config.source
            )
            if config.event_hazards
            else None
        ),
        task_set=config.task_set,
    )
    logger.info(
        "[data] train: %d subjects, %d events (streamed)",
        stats.n_subjects,
        stats.n_events,
    )
    vocab = Vocabulary.build_from_counts(
        stats.code_counts,
        min_count=config.vocab_min_count,
        max_size=config.vocab_max_size,
        backoff=config.vocab_backoff,
    )
    vocab.save(output_dir / "vocabulary.json")
    logger.info("[data] vocab size: %d", len(vocab))

    logger.info("[data] loading tuning shards from %s", config.tuning_shard_dir)
    tuning_events = prepare(
        load_meds_shards(config.tuning_shard_dir, max_shards=config.max_tuning_shards)
    )
    tuning_labels: ConceptLabelDict
    tuning_masks: ConceptLabelDict
    if config.concept_supervision == "visit":
        tuning_labels, tuning_masks = build_visit_concept_label_dicts(
            tuning_events, concepts
        )
    else:
        tuning_labels, tuning_masks = build_concept_label_dicts(tuning_events, concepts)
    alerts = hazard_events_for(
        config.task_set, config.auxiliary_event_names, source=config.source
    )
    event_names = [a.name for a in alerts]
    train_event_tables = (
        EventTimeTables(stats.event_times, event_names)
        if config.event_hazards
        else None
    )
    tuning_event_tables = (
        EventTimeTables(
            all_event_times(
                tuning_events, alerts, config.source, task_set=config.task_set
            ),
            event_names,
        )
        if config.event_hazards
        else None
    )
    tuning_events_binned = add_value_tokens(tuning_events, binner, source=config.source)
    del tuning_events

    objective = build_objective(
        config, vocab, None, device, code_counts=stats.code_counts
    )

    def make_train_patients(epoch: int) -> Iterator[PatientSequence]:
        return iter_patients_streaming(
            paths,
            prepare,
            binner,
            vocab,
            source=config.source,
            max_seq_len=config.max_seq_len,
            shuffle_seed=config.seed + epoch,
        )

    corpus = PreparedCorpus(
        vocab=vocab,
        concepts=concepts,
        objective=objective,
        train_labels=_labels_to_device(stats.labels, device),
        train_masks=_labels_to_device(stats.masks, device),
        train_first_times=_labels_to_device(stats.first_times, device),
        tuning_labels=_labels_to_device(tuning_labels, device),
        tuning_masks=_labels_to_device(tuning_masks, device),
        train_event_tables=train_event_tables,
        tuning_event_tables=tuning_event_tables,
        tuning_events_binned=tuning_events_binned,
        make_train_patients=make_train_patients,
    )
    return _run_training(config, output_dir, device, corpus)


def _batch_config_fields(cfg: TrainingConfig) -> dict[str, object]:
    """Fields that determine what a resume's data stream actually replays.

    What ``PackedLaneSampler.next_chunk()``/``PackedContextSampler`` produce
    at a given position, for a given epoch's seed -- saved alongside every
    periodic checkpoint so a resume can tell whether fast-forwarding to the
    checkpoint's ``steps_into_epoch`` would land on the same position it was
    taken at, or a different one (e.g. this run manually restarted with a
    different ``num_lanes``/``chunk_size``, which is exactly what happened
    partway through the run that motivated this: batch size was tuned up
    for GPU utilization mid-training). Deliberately excludes model/
    optimizer hyperparameters (``learning_rate`` etc.), which don't affect
    the data stream.

    Module-level (not a closure inside :func:`_run_training`) specifically
    so it can be unit-tested on its own -- see
    ``test_batch_config_fields_covers_every_resume_relevant_field`` in
    ``tests/odyssey/training/test_train.py``, which fails if a
    resume-relevant field is ever added to :class:`TrainingConfig` without
    also being added here.
    """
    return {
        "backbone": cfg.backbone,
        "num_lanes": cfg.num_lanes,
        "chunk_size": cfg.chunk_size,
        "reset_prob": cfg.reset_prob,
        "max_context": cfg.max_context,
        "seed": cfg.seed,
    }


def _run_training(  # noqa: PLR0912, PLR0915
    config: TrainingConfig, output_dir: Path, device: str, corpus: PreparedCorpus
) -> Path:
    """Run the training loop proper over a :class:`PreparedCorpus`."""
    vocab = corpus.vocab
    concepts = corpus.concepts
    objective = corpus.objective
    train_labels, train_masks = corpus.train_labels, corpus.train_masks
    train_first_times = corpus.train_first_times
    tuning_labels, tuning_masks = corpus.tuning_labels, corpus.tuning_masks
    train_event_tables = corpus.train_event_tables
    tuning_event_tables = corpus.tuning_event_tables
    tuning_events_binned = corpus.tuning_events_binned

    model = build_model(config, vocab_size=len(vocab), num_concepts=len(concepts)).to(
        device
    )
    if config.init_from is not None:
        if config.resume_from is not None:
            raise ValueError(
                "init_from and resume_from are mutually exclusive: init_from "
                "starts a fresh run seeded with another checkpoint's weights, "
                "resume_from continues an interrupted run of this same config"
            )
        logger.info("[init] loading weights only from %s", config.init_from)
        init_checkpoint = torch.load(config.init_from, map_location=device)
        model.load_state_dict(init_checkpoint["model"])
    if config.freeze_bottleneck:
        if not isinstance(model, ConceptBottleneckSequenceModel):
            raise ValueError(
                "freeze_bottleneck requires model_kind='bottleneck' "
                f"(got {config.model_kind!r}, no bottleneck to freeze)"
            )
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.bottleneck.parameters():
            p.requires_grad_(False)
        n_unfreeze = config.unfreeze_top_backbone_layers
        if n_unfreeze > 0:
            backbone_layers = getattr(model.backbone, "layers", None)
            if not isinstance(backbone_layers, torch.nn.ModuleList):
                raise ValueError(
                    "unfreeze_top_backbone_layers requires a backbone with a "
                    f"'layers' ModuleList (got {type(model.backbone).__name__})"
                )
            for layer in backbone_layers[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(
            "[freeze] backbone + bottleneck frozen (%.1fM parameters, top %d "
            "backbone layer(s) re-unfrozen); training the task heads%s",
            n_frozen / 1e6,
            n_unfreeze,
            " and the re-unfrozen backbone layers" if n_unfreeze > 0 else "",
        )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_run_provenance(
        output_dir, model, len(vocab), device=device
    )  # fingerprint only
    logger.info(
        "[model] %.1fM parameters (%.1fM trainable) on %s",
        n_params / 1e6,
        n_trainable / 1e6,
        device,
    )
    for name, module in model.named_children():
        m_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        m_total = sum(p.numel() for p in module.parameters())
        logger.info(
            "[model] %s: %.2fM / %.2fM trainable",
            name,
            m_trainable / 1e6,
            m_total / 1e6,
        )

    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, config.weight_decay),
        lr=config.learning_rate,
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
    steering = _prepare_steering(config, model, corpus, device)
    loss_weights = ConceptBottleneckLossWeights(
        concept=config.concept_weight,
        orthogonality=config.orthogonality_weight,
        observability=config.observability_weight,
        task=config.task_weight,
        reconstruction=config.reconstruction_weight,
        independence=config.independence_weight,
        concept_pos_weight=pos_weight,
    )

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

    def make_train_sampler(epoch: int) -> StreamingSampler:
        patients = corpus.make_train_patients(epoch)
        if config.backbone == "transformer":
            return PackedContextSampler(
                patients, batch_size=config.num_lanes, max_context=config.max_context
            )
        return PackedLaneSampler(
            patients,
            num_lanes=config.num_lanes,
            chunk_size=config.chunk_size,
            reset_prob=config.reset_prob,
            seed=config.seed + epoch,
        )

    def make_tuning_sampler() -> StreamingSampler:
        patients = iter_patient_sequences(
            tuning_events_binned,
            vocab,
            max_seq_len=config.max_seq_len,
        )
        if config.backbone == "transformer":
            return PackedContextSampler(
                patients, batch_size=config.num_lanes, max_context=config.max_context
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
            elif (
                steering is not None
                and (injection := steering.injection_for(chunk, global_step))
                is not None
            ):
                # Steerling's steering phase: forecasting loss plus respond
                # and express at the injected positions; the interpretability
                # losses are off for these steps, as in their recipe.
                total, components, state = model.compute_steering_loss(
                    chunk,
                    state=state,
                    concept_index=injection.concept_index,
                    injected=injection.positions,
                    direction=steering.directions[injection.concept_index],
                    gamma=steering.gammas[injection.concept_index],
                    layer_index=steering.layer_index,
                    lifted_ids=steering.lifted[injection.concept_index],
                    objective=objective,
                    event_targets=event_targets,
                    respond_weight=config.respond_weight,
                    express_weight=config.express_weight,
                    forecast_at_injected=config.steering_forecast_at_injected,
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
                    teacher_alpha_known=annealed_alpha(
                        global_step,
                        config.teacher_anneal_steps,
                        start=config.teacher_known_start,
                        end=config.teacher_known_end,
                        cosine=True,
                    ),
                    teacher_alpha_unknown=annealed_alpha(
                        global_step,
                        config.teacher_anneal_steps,
                        start=config.teacher_unknown_start,
                        end=config.teacher_unknown_end,
                    ),
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
                    objective.value_head_weight,
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
                    write_run_provenance(
                        output_dir,
                        model,
                        len(vocab),
                        device=device,
                        checkpoint_name="checkpoint_best.pt",
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
                write_run_provenance(
                    output_dir,
                    model,
                    len(vocab),
                    device=device,
                    checkpoint_name=f"checkpoint_{global_step}.pt",
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
        write_run_provenance(
            output_dir,
            model,
            len(vocab),
            device=device,
            checkpoint_name=f"checkpoint_epoch_{epoch}.pt",
        )

    _atomic_torch_save(
        {"model": model.state_dict(), "step": global_step, "config": asdict(config)},
        output_dir / "checkpoint_final.pt",
    )
    write_run_provenance(
        output_dir,
        model,
        len(vocab),
        device=device,
        checkpoint_name="checkpoint_final.pt",
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


def _load_config_overrides(config_json: str | None) -> dict[str, Any]:
    """Parse ``--config-json``: inline JSON first, then a path to a JSON file.

    The module docstring has always promised both forms; only the path
    form was implemented until now, so the docstring's own usage example
    (inline JSON) died with ``FileNotFoundError``. Inline is tried first
    -- a JSON object literal is never a valid path -- and the path
    fallback keeps every existing invocation working unchanged.
    """
    if not config_json:
        return {}
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError:
        parsed = json.loads(Path(config_json).read_text())
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--config-json must hold a JSON object of TrainingConfig field "
            f"overrides, got {type(parsed).__name__}"
        )
    return parsed


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
        help=(
            "Inline JSON, or a path to a JSON file, with any subset of "
            "TrainingConfig field overrides."
        ),
    )
    args = parser.parse_args()

    overrides = _load_config_overrides(args.config_json)
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
