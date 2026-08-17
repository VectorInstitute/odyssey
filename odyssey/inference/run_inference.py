"""Run a trained sequence model (bottleneck or baseline) over held-out data.

Streams held-out patients through the model exactly the way training did
(:class:`~odyssey.data.streaming.PackedLaneSampler`, carried recurrent
state across chunks), rather than the whole-sequence-per-row path: a
held-out patient's full stay can be far longer than any single training
chunk, and streaming is the only path that scales to that without an
enormous padded batch. ``reset_prob=0.0`` here, unlike training -- at
inference time we want the model to see a patient's true full history,
not synthetic missing-history resets.

Produces one :class:`InferenceResults`, covering all three eval
questions from ``odyssey/training/metrics.py``: forecasting quality,
concept quality, and (via :func:`orthogonality_diagnostic`) whether the
known/unknown concept split held on data the model never trained on.
Concept usefulness (completeness) is intentionally not computed here --
see ``research_journal`` for why a binary task-outcome label for that
probe still needs a real design decision, not implemented yet.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import AnyConceptDefinition, concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.streaming import PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import QuantileBinner, add_value_tokens
from odyssey.data.vocabulary import Vocabulary, code_type
from odyssey.models.concept_bottleneck import ConceptBottleneckOutput
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
    SequenceModel,
    _gather_by_subject,
    _gather_by_visit,
    _pool_patient_ends,
)
from odyssey.models.time_to_event import (
    gap_survival_valid_mask,
    gap_to_bin,
    hazard_log_likelihood,
    probability_within,
)
from odyssey.training.data import (
    build_concept_label_dicts,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.metrics import (
    ConceptMetrics,
    ObservabilityMetrics,
    TaskMetrics,
    TimeMetrics,
    compute_concept_metrics,
    compute_observability_metrics,
    orthogonality_diagnostic,
)
from odyssey.training.train import TrainingConfig, _move_chunk_to_device, build_model


logger = logging.getLogger(__name__)

_CODE_TYPE_NAMES = {
    1: "diagnosis",
    2: "medication",
    3: "procedure",
    4: "lab",
    5: "visit",
    6: "demographic",
    7: "billing",
    8: "other",
}


@dataclass(frozen=True)
class InferenceResults:
    """Everything scored from one streaming pass over a held-out split."""

    task_metrics: TaskMetrics
    task_metrics_by_code_type: Dict[str, TaskMetrics]
    concept_metrics: List[ConceptMetrics]
    observability_metrics: List[ObservabilityMetrics]
    orthogonality: float
    n_patient_ends_scored: int
    time_metrics: Optional[TimeMetrics] = None
    """Time-to-next-event scoring; None for models without a time head."""


# Horizons are bin edges of DEFAULT_TIME_BIN_EDGES_HOURS so P(within h) is exact.
_TIME_HORIZONS_HOURS: Dict[str, float] = {"1h": 1.0, "8h": 8.0, "24h": 24.0}


class _RunningTimeMetrics:
    """Streaming accumulator for :class:`TimeMetrics` (see that docstring)."""

    def __init__(self, edges: Sequence[float]) -> None:
        self.edges = list(edges)
        self.nll_sum = 0.0
        self.n = 0
        self.same_correct = 0
        self.same_observed = 0
        self.pred_within = dict.fromkeys(_TIME_HORIZONS_HOURS, 0.0)
        self.obs_within = dict.fromkeys(_TIME_HORIZONS_HOURS, 0)

    def update(
        self, hazard_logits: torch.Tensor, gap_hours: torch.Tensor, valid: torch.Tensor
    ) -> None:
        if not bool(valid.any()):
            return
        logits = hazard_logits[valid]
        gaps = gap_hours[valid]
        target_bin = gap_to_bin(gaps.clamp_min(0.0), self.edges)
        ll = hazard_log_likelihood(logits, target_bin)
        self.nll_sum += float(-ll.sum().item())
        self.n += int(gaps.numel())
        p_same = torch.sigmoid(logits[:, 0])
        same = gaps <= 0
        self.same_correct += int(((p_same > 0.5) == same).sum().item())
        self.same_observed += int(same.sum().item())
        for label, horizon in _TIME_HORIZONS_HOURS.items():
            self.pred_within[label] += float(
                probability_within(logits, self.edges, horizon).sum().item()
            )
            self.obs_within[label] += int((gaps <= horizon).sum().item())

    def finalize(self) -> Optional[TimeMetrics]:
        if self.n == 0:
            return None
        return TimeMetrics(
            nll=self.nll_sum / self.n,
            n_positions=self.n,
            same_instant_accuracy=self.same_correct / self.n,
            same_instant_rate=self.same_observed / self.n,
            calibration={
                label: {
                    "predicted": self.pred_within[label] / self.n,
                    "observed": self.obs_within[label] / self.n,
                }
                for label in _TIME_HORIZONS_HOURS
            },
        )


def _latest_checkpoint(run_dir: Path) -> Path:
    """Return ``checkpoint_final.pt`` if present, else the highest-step periodic one.

    Lets evaluation run against an in-progress training run (e.g. to
    sanity-check the pipeline before a long run finishes), not only a
    fully completed one.
    """
    final = run_dir / "checkpoint_final.pt"
    if final.exists():
        return final
    candidates = list(run_dir.glob("checkpoint_[0-9]*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_*.pt found in {run_dir}")
    return max(candidates, key=lambda p: int(p.stem.split("_")[-1]))


def load_run(
    run_dir: Union[str, Path],
    *,
    device: str = "cuda",
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Tuple[SequenceModel, Vocabulary, QuantileBinner, TrainingConfig]:
    """Reconstruct a trained model and its tokenization artifacts from a run dir.

    ``run_dir`` is a :func:`~odyssey.training.train.train` output
    directory: reads ``config.json`` (architecture hyperparameters --
    the training-only fields it also contains, e.g. ``learning_rate``,
    are simply unused by :func:`~odyssey.training.train.build_model`),
    ``vocabulary.json``, ``quantile_binner.json``, and a checkpoint --
    ``checkpoint_path`` if given (e.g. ``run_dir / "checkpoint_best.pt"``
    to evaluate the lowest-val-loss checkpoint rather than wherever
    training happened to stop), else the latest available one (see
    :func:`_latest_checkpoint`).
    """
    run_dir = Path(run_dir)
    config = TrainingConfig(**json.loads((run_dir / "config.json").read_text()))
    vocab = Vocabulary.load(run_dir / "vocabulary.json")
    binner = QuantileBinner.load(run_dir / "quantile_binner.json")

    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path else _latest_checkpoint(run_dir)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # A run predating a config field gets that field's *current* default
    # from the dataclass, which can disagree with the checkpoint's actual
    # architecture; the checkpoint is the authority on what heads exist.
    config.time_to_event = any(k.startswith("time_head.") for k in checkpoint["model"])
    config.event_hazards = any(
        k.startswith("event_heads.") for k in checkpoint["model"]
    )

    concepts = concepts_for_source(getattr(config, "source", "mimic_iv"))
    model = build_model(config, vocab_size=len(vocab), num_concepts=len(concepts))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model, vocab, binner, config


def load_and_bin_held_out(
    shard_dir: Union[str, Path],
    binner: QuantileBinner,
    *,
    max_shards: Optional[int] = None,
    source: str = "mimic_iv",
) -> pl.DataFrame:
    """Load a held-out MEDS split and apply the *train-fit* binner to it.

    Never re-fits the binner here -- using the train split's own
    quantile boundaries on held-out data is the whole point of
    evaluating on genuinely unseen data. ``source`` must match the
    training run's (it picks the curated clinical bin prefixes).
    """
    events = load_meds_shards(shard_dir, max_shards=max_shards)
    return add_value_tokens(events, binner, source=source)


def _build_type_lookup(vocab: Vocabulary, device: str) -> torch.Tensor:
    """``(vocab_size,)`` token id -> code-type id, for a vectorized per-chunk lookup.

    Precomputed once rather than decoding each target token on every
    chunk (real held-out passes have hundreds of thousands of real
    positions).
    """
    lookup = torch.zeros(len(vocab), dtype=torch.long)
    for token_id, token in vocab.id_to_token.items():
        lookup[token_id] = code_type(token)
    return lookup.to(device)


def _build_category_lookup(vocab: Vocabulary, device: str) -> Tuple[torch.Tensor, int]:
    """``(vocab_size,)`` token id -> ICD 3-character category id, or -1.

    Every ICD-coded diagnosis/procedure token (a full code or an ``icd3``
    backoff category token alike) maps to the integer id of its 3-char
    category, so ``I5023`` and ``I50`` share one id; every other token
    gets -1. Returns the lookup and the number of category ids.
    """
    lookup = torch.full((len(vocab),), -1, dtype=torch.long)
    categories: Dict[str, int] = {}
    for token_id, token in vocab.id_to_token.items():
        parts = token.split("//")
        if (
            len(parts) == 4
            and parts[0] in ("DIAGNOSIS", "PROCEDURE")
            and parts[1] == "ICD"
        ):
            key = "//".join([*parts[:3], parts[3][:3]])
            lookup[token_id] = categories.setdefault(key, len(categories))
    return lookup.to(device), len(categories)


class BlockSetHits(NamedTuple):
    """Per-position set-based scoring flags, all ``(lanes, chunk)`` bool."""

    set_valid: torch.Tensor
    """Position has an in-chunk target time block (everything but the
    final lane position)."""

    set_hit: torch.Tensor
    """Top-1 names some event of the *target's own family* recorded at
    the same instant as the true next event."""

    category_valid: torch.Tensor
    """``set_valid`` and the target is an ICD-coded token."""

    category_hit: torch.Tensor
    """Top-1's ICD 3-character category matches some same-family event
    in the target's block."""


@dataclass
class _RunningBucket:
    """Cross-entropy/top-k sums for one slice of targets, updated chunk by chunk.

    Sums (not means) so weighted-averaging across chunks of different
    sizes reduces to a single division at the end -- exactly what
    ``F.cross_entropy(..., reduction="sum")`` plus a running count gives.
    """

    ce_sum: float = 0.0
    hit_sums: Dict[int, int] = field(default_factory=dict)
    n: int = 0
    set_hit_sum: int = 0
    n_set: int = 0
    category_hit_sum: int = 0
    n_category: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        top_k: Sequence[int],
        set_hits: Optional[BlockSetHits] = None,
    ) -> None:
        """``logits`` is ``(n, vocab_size)``, ``targets`` is ``(n,)``: one chunk.

        ``set_hits`` carries the matching ``(n,)``-shaped slices of the
        chunk's :class:`BlockSetHits`.
        """
        if targets.numel() == 0:
            return
        self.n += int(targets.numel())
        if set_hits is not None:
            self.n_set += int(set_hits.set_valid.sum().item())
            self.set_hit_sum += int(set_hits.set_hit.sum().item())
            self.n_category += int(set_hits.category_valid.sum().item())
            self.category_hit_sum += int(set_hits.category_hit.sum().item())
        self.ce_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        top_k_preds = logits.topk(max(top_k), dim=-1).indices
        hits = top_k_preds == targets.unsqueeze(-1)
        for k in top_k:
            self.hit_sums[k] = self.hit_sums.get(k, 0) + int(
                hits[:, :k].any(dim=-1).sum().item()
            )

    def finalize(self) -> TaskMetrics:
        """Combine the running sums into one :class:`TaskMetrics`."""
        if self.n == 0:
            raise ValueError("no non-ignored predictions to compute metrics over")
        cross_entropy = self.ce_sum / self.n
        return TaskMetrics(
            cross_entropy=cross_entropy,
            perplexity=float(torch.exp(torch.tensor(cross_entropy))),
            top1_accuracy=self.hit_sums.get(1, 0) / self.n,
            top5_accuracy=self.hit_sums.get(5, 0) / self.n,
            n_predictions=self.n,
            set_top1_accuracy=(self.set_hit_sum / self.n_set if self.n_set else None),
            n_set_predictions=self.n_set or None,
            category_set_top1_accuracy=(
                self.category_hit_sum / self.n_category if self.n_category else None
            ),
            n_category_predictions=self.n_category or None,
        )


def _block_set_hits(
    top1: torch.Tensor,
    targets: torch.Tensor,
    *,
    times: torch.Tensor,
    subject_ids: torch.Tensor,
    real_mask: torch.Tensor,
    vocab_size: int,
    type_lookup: torch.Tensor,
    category_lookup: Optional[torch.Tensor] = None,
    n_categories: int = 0,
) -> BlockSetHits:
    """Per-position "top-1 named some event in the target's time block".

    Sequences are time-sorted per subject, so a same-timestamp event block
    is a contiguous run; block membership is recoverable from the chunk's
    own input timestamps (the target at position ``j`` is the input token
    at ``j+1``). Fully vectorized: each position gets a composite
    ``(block_id, family) * vocab_size + token`` key, and ``torch.isin`` of
    predicted keys against target keys answers membership with no Python
    loops. The final position of each lane has no in-chunk target
    timestamp and is excluded (`n_set_predictions` counts what remains);
    blocks never span subjects because a subject change starts a new
    block.

    Membership is restricted to the *target's own code family*: a
    discharge block holds the ~22 diagnosis codes together with the
    discharge event and DRG billing at the same instant, and without the
    restriction a diagnosis target could be "set-hit" by predicting the
    discharge token -- crediting the diagnosis family with a prediction
    that says nothing about diagnoses. With it, a diagnosis set-hit means
    the model named a diagnosis that is in the block.

    ``category_lookup`` (see :func:`_build_category_lookup`) adds a second,
    coarser flag for ICD-coded targets: the top-1's 3-character category
    matches some same-family block member's. Under the ``icd3`` vocabulary
    backoff a frequent full code and its category token coexist, so
    probability mass splits between ``I5023`` and ``I50``; category
    scoring asks whether the model knows the *kind* of diagnosis or
    procedure coming, independent of that split.
    """
    lanes, chunk = targets.shape
    tgt_t = times[:, 1:]
    tgt_s = subject_ids[:, 1:]
    tgt = targets[:, : chunk - 1]
    pred = top1[:, : chunk - 1]
    valid = real_mask[:, : chunk - 1]

    new_block = torch.ones_like(tgt_s, dtype=torch.bool)
    new_block[:, 1:] = (tgt_t[:, 1:] != tgt_t[:, :-1]) | (tgt_s[:, 1:] != tgt_s[:, :-1])
    lane_offset = torch.arange(lanes, device=targets.device).unsqueeze(1) * (chunk + 1)
    block_id = new_block.long().cumsum(dim=1) + lane_offset

    # Blocks are keyed by (block, target family): a member only counts for
    # queries whose target shares its family, and a query token matches a
    # member only if it *is* that member, so a hit implies the top-1 is
    # in the block and of the target's family.
    tgt_family = type_lookup[tgt]
    n_families = int(type_lookup.max().item()) + 1
    block_family = block_id * n_families + tgt_family
    member_keys = torch.where(
        valid, block_family * vocab_size + tgt, torch.full_like(tgt, -1)
    )
    query_keys = block_family * vocab_size + pred
    hit = torch.isin(query_keys, member_keys) & valid

    set_valid = torch.zeros_like(real_mask)
    set_hit = torch.zeros_like(real_mask)
    set_valid[:, : chunk - 1] = valid
    set_hit[:, : chunk - 1] = hit

    category_valid = torch.zeros_like(real_mask)
    category_hit = torch.zeros_like(real_mask)
    if category_lookup is not None and n_categories > 0:
        tgt_cat = category_lookup[tgt]
        pred_cat = category_lookup[pred]
        cat_valid = valid & (tgt_cat >= 0)
        cat_members = torch.where(
            cat_valid,
            block_family * n_categories + tgt_cat,
            torch.full_like(tgt, -1),
        )
        # An out-of-hierarchy top-1 (pred_cat == -1) can never match.
        cat_query = block_family * n_categories + pred_cat.clamp_min(0)
        cat_hit = torch.isin(cat_query, cat_members) & cat_valid & (pred_cat >= 0)
        category_valid[:, : chunk - 1] = cat_valid
        category_hit[:, : chunk - 1] = cat_hit

    return BlockSetHits(set_valid, set_hit, category_valid, category_hit)


class _RunningTaskMetrics:
    """Streaming equivalent of ``compute_task_metrics``/``..._by_code_type``.

    Those two functions need the full ``(N, vocab_size)`` logits tensor
    materialized at once -- fine for one training batch, but for a real
    held-out split with hundreds of thousands of real positions, holding
    onto every chunk's logits until the very end doesn't scale.
    Confirmed the hard way: exactly this accumulation pattern (an
    earlier version of :func:`run_streaming_inference`) OOM-killed the
    actual training job it happened to be running alongside, evaluating
    against only 5 real held-out shards. This accumulates the same
    quantities incrementally instead, holding only running scalars
    (never more than one chunk's logits at a time -- the same transient
    cost the model's own forward pass already pays).
    """

    def __init__(
        self, vocab: Vocabulary, *, device: str, top_k: Sequence[int] = (1, 5)
    ) -> None:
        self._top_k = top_k
        self.type_lookup = _build_type_lookup(vocab, device)
        self.category_lookup, self.n_categories = _build_category_lookup(vocab, device)
        self.overall = _RunningBucket()
        self.by_type: Dict[str, _RunningBucket] = {}

    def block_set_hits(
        self,
        top1: torch.Tensor,
        targets: torch.Tensor,
        *,
        times: torch.Tensor,
        subject_ids: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> BlockSetHits:
        """:func:`_block_set_hits` with this evaluation's own lookups."""
        return _block_set_hits(
            top1,
            targets,
            times=times,
            subject_ids=subject_ids,
            real_mask=real_mask,
            vocab_size=int(self.type_lookup.shape[0]),
            type_lookup=self.type_lookup,
            category_lookup=self.category_lookup,
            n_categories=self.n_categories,
        )

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        set_hits: Optional[BlockSetHits] = None,
    ) -> None:
        """Fold in one chunk's real-position ``(logits, targets)``.

        ``set_hits`` holds the same real-position slice of the chunk's
        :class:`BlockSetHits`.
        """
        if targets.numel() == 0:
            return
        self.overall.update(logits, targets, self._top_k, set_hits)
        target_types = self.type_lookup[targets]
        for type_id, name in _CODE_TYPE_NAMES.items():
            type_mask = target_types == type_id
            if type_mask.any():
                self.by_type.setdefault(name, _RunningBucket()).update(
                    logits[type_mask],
                    targets[type_mask],
                    self._top_k,
                    BlockSetHits(*(flag[type_mask] for flag in set_hits))
                    if set_hits is not None
                    else None,
                )

    def finalize(self) -> Tuple[TaskMetrics, Dict[str, TaskMetrics]]:
        return self.overall.finalize(), {
            name: bucket.finalize() for name, bucket in self.by_type.items()
        }


@dataclass
class _PooledEnds:
    """Per-chunk pooled bottleneck readouts at supervision positions, on CPU."""

    subject_ids: List[torch.Tensor] = field(default_factory=list)
    visit_ids: List[torch.Tensor] = field(default_factory=list)
    concept_probs: List[torch.Tensor] = field(default_factory=list)
    observability_probs: List[torch.Tensor] = field(default_factory=list)
    concept_embeddings: List[torch.Tensor] = field(default_factory=list)
    unknown_embedding: List[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        chunk: StreamingChunk,
        out: ConceptBottleneckOutput,
        pool_mask: torch.Tensor,
    ) -> None:
        self.subject_ids.append(_pool_patient_ends(chunk.subject_ids, pool_mask).cpu())
        self.visit_ids.append(_pool_patient_ends(chunk.visit_ids, pool_mask).cpu())
        self.concept_probs.append(
            _pool_patient_ends(out.concept_probs, pool_mask).cpu()
        )
        self.observability_probs.append(
            _pool_patient_ends(out.observability_probs, pool_mask).cpu()
        )
        self.concept_embeddings.append(
            _pool_patient_ends(out.concept_embeddings, pool_mask).cpu()
        )
        self.unknown_embedding.append(
            _pool_patient_ends(out.unknown_embedding, pool_mask).cpu()
        )


def run_streaming_inference(
    model: SequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    supervision: ConceptSupervision = "stay",
    concepts: Optional[Sequence[AnyConceptDefinition]] = None,
) -> InferenceResults:
    """Stream held-out patients through ``model`` and score every eval question.

    ``concept_labels``/``concept_mask`` follow
    :func:`~odyssey.training.data.build_concept_label_dicts`'s
    ``subject_id -> (num_concepts,)`` shape, built from the *unbinned*
    held-out events (concept labeling never looks at value tokens).
    ``concepts`` must be the same definitions those labels were built
    from (defaults to the MIMIC-IV expansion of the canonical registry,
    matching the training default).
    """
    if concepts is None:
        concepts = concepts_for_source("mimic_iv")
    model.eval()
    patients = iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )

    task_stats = _RunningTaskMetrics(vocab, device=device)
    time_head = getattr(model, "time_head", None)
    time_stats = _RunningTimeMetrics(time_head.edges) if time_head is not None else None
    pooled = _PooledEnds()

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            fwd = model.forward_with_features(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            logits, state = fwd.logits, fwd.state

            real = chunk.real_mask
            if time_stats is not None and time_head is not None:
                hazard_logits = time_head(fwd.features)
                gap, gap_valid = gap_survival_valid_mask(
                    chunk.batch.aux.time_stamps, real
                )
                time_stats.update(hazard_logits, gap, gap_valid)
            if real.any():
                set_hits = task_stats.block_set_hits(
                    logits.argmax(dim=-1),
                    chunk.targets,
                    times=chunk.batch.aux.time_stamps,
                    subject_ids=chunk.subject_ids,
                    real_mask=real,
                )
                task_stats.update(
                    logits[real],
                    chunk.targets[real],
                    BlockSetHits(*(flag[real] for flag in set_hits)),
                )

            pool_mask = chunk.patient_end if supervision == "stay" else chunk.visit_end
            if fwd.bottleneck is not None and pool_mask.any():
                pooled.append(chunk, fwd.bottleneck, pool_mask)

    task_metrics, task_metrics_by_code_type = task_stats.finalize()

    if not pooled.subject_ids:
        # A baseline model has no bottleneck to pool, or no chunk ever had
        # a real pool_mask position -- e.g. supervision
        # is "visit" but nothing in this split has a real hadm_id, so
        # chunk.visit_end never fires. Forecasting quality (task_metrics
        # above) is still valid, since it never depends on pooling; only
        # the pooled concept/observability/orthogonality questions have
        # nothing to score.
        if isinstance(model, BaselineSequenceModel):
            logger.info("[inference] baseline model: no concept metrics to score")
        else:
            logger.warning(
                "[inference] no %s-scoped pool positions were ever produced -- "
                "skipping concept/observability/orthogonality metrics",
                supervision,
            )
        return InferenceResults(
            task_metrics=task_metrics,
            task_metrics_by_code_type=task_metrics_by_code_type,
            concept_metrics=[],
            observability_metrics=[],
            orthogonality=float("nan"),
            n_patient_ends_scored=0,
            time_metrics=time_stats.finalize() if time_stats is not None else None,
        )

    subject_ids = torch.cat(pooled.subject_ids)
    concept_probs = torch.cat(pooled.concept_probs)
    observability_probs = torch.cat(pooled.observability_probs)
    concept_embeddings = torch.cat(pooled.concept_embeddings)
    unknown_embedding = torch.cat(pooled.unknown_embedding)

    concept_names = [c.name for c in concepts]
    if supervision == "visit":
        visit_ids = torch.cat(pooled.visit_ids)
        labels = _gather_by_visit(subject_ids, visit_ids, concept_labels)  # type: ignore[arg-type]
        masks = _gather_by_visit(subject_ids, visit_ids, concept_mask)  # type: ignore[arg-type]
    else:
        labels = _gather_by_subject(subject_ids, concept_labels)  # type: ignore[arg-type]
        masks = _gather_by_subject(subject_ids, concept_mask)  # type: ignore[arg-type]
    observed_mask = masks > 0

    concept_metrics = compute_concept_metrics(
        concept_probs, labels, masks, concept_names
    )
    observability_metrics = compute_observability_metrics(
        observability_probs, observed_mask.float(), concept_names
    )
    orthogonality = orthogonality_diagnostic(concept_embeddings, unknown_embedding)

    return InferenceResults(
        task_metrics=task_metrics,
        task_metrics_by_code_type=task_metrics_by_code_type,
        concept_metrics=concept_metrics,
        observability_metrics=observability_metrics,
        orthogonality=orthogonality,
        n_patient_ends_scored=int(subject_ids.shape[0]),
        time_metrics=time_stats.finalize() if time_stats is not None else None,
    )


def evaluate_run(
    run_dir: Union[str, Path],
    held_out_shard_dir: Union[str, Path],
    *,
    max_shards: Optional[int] = None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> InferenceResults:
    """End-to-end: load a trained run, score it against a held-out split."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )

    logger.info("[inference] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events, enabled=getattr(config, "normalize_medications", False)
    )
    raw_events = maybe_history_recap(
        raw_events, enabled=getattr(config, "history_recap", False)
    )
    source = getattr(config, "source", "mimic_iv")
    concepts = concepts_for_source(source)
    events_binned = add_value_tokens(raw_events, binner, source=source)

    supervision = getattr(config, "concept_supervision", "stay")
    logger.info(
        "[inference] labeling concepts (%s-scoped, source=%s)", supervision, source
    )
    concept_labels: ConceptLabelDict
    concept_mask: ConceptLabelDict
    if supervision == "visit":
        concept_labels, concept_mask = build_visit_concept_label_dicts(
            raw_events, concepts
        )
    else:
        concept_labels, concept_mask = build_concept_label_dicts(raw_events, concepts)
    del raw_events

    logger.info("[inference] running streaming inference")
    return run_streaming_inference(
        model,
        events_binned,
        vocab,
        concept_labels,
        concept_mask,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
        supervision=supervision,  # type: ignore[arg-type]
        concepts=concepts,
    )


def results_to_dict(results: InferenceResults) -> Dict[str, object]:
    """Plain-JSON-able view of :class:`InferenceResults` (already all plain types)."""
    from dataclasses import asdict  # noqa: PLC0415

    return asdict(results)


@dataclass(frozen=True)
class _CliArgs:
    """Parsed CLI args for :func:`evaluate_run`, mirroring ``training.train``'s CLI."""

    run_dir: Path
    held_out_shard_dir: str
    output_json: Path
    checkpoint_path: Path
    max_shards: Optional[int]
    num_lanes: int
    chunk_size: int


def _parse_args() -> _CliArgs:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename within --run-dir (default: checkpoint_best.pt).",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    return _CliArgs(
        run_dir=run_dir,
        held_out_shard_dir=args.held_out_shard_dir,
        output_json=Path(args.output_json),
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        max_shards=args.max_shards,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli_args = _parse_args()
    results = evaluate_run(
        cli_args.run_dir,
        cli_args.held_out_shard_dir,
        max_shards=cli_args.max_shards,
        num_lanes=cli_args.num_lanes,
        chunk_size=cli_args.chunk_size,
        checkpoint_path=cli_args.checkpoint_path,
    )
    cli_args.output_json.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output_json.write_text(json.dumps(results_to_dict(results), indent=2))
    logger.info("[inference] wrote results to %s", cli_args.output_json)


__all__ = [
    "InferenceResults",
    "load_run",
    "load_and_bin_held_out",
    "run_streaming_inference",
    "evaluate_run",
    "results_to_dict",
]
