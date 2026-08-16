"""Real MEDS shards -> training-ready patient sequences and concept labels.

Bridges the standard MEDS extraction layout (a split directory of numbered
``.parquet`` shards, exactly what ``meds-extract-run`` itself produces --
see ``.../data/{train,tuning,held_out}`` in a real extraction) to the
per-subject :class:`~odyssey.data.sequences.PatientSequence` objects
:class:`~odyssey.data.streaming.PackedLaneSampler` consumes, and to the
``subject_id -> (num_concepts,) tensor`` dicts
:meth:`~odyssey.models.sequence_model.ConceptBottleneckSequenceModel.compute_streaming_loss`
expects for concept labels/masks.
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import polars as pl
import torch

from odyssey.data.concepts import (
    AnyConceptDefinition,
    label_concepts,
    label_concepts_by_visit,
)
from odyssey.data.sequences import BIRTH_CODE, PatientSequence, build_patient_sequence
from odyssey.data.vocabulary import Vocabulary, code_type


_T = TypeVar("_T")


# The only columns anything in this pipeline (concept labeling, value
# binning, sequence building, vocab building) ever reads -- confirmed by
# grepping every raw-events consumer in odyssey/. A real MIMIC-IV MEDS
# shard carries 21 columns (drg_severity, emar_id, icustay_id, order_id,
# route, ... plus this one's used set): loading all of them, not just
# these 5, was the dominant real memory cost, not shard count. Measured
# on one real train shard (2.41M rows): 325.5MB total vs. 122MB for just
# these 5 columns -- the other 16 alone cost more than the 5 that matter.
_MEDS_EVENT_COLUMNS = ["subject_id", "time", "code", "numeric_value", "hadm_id"]


def load_meds_shards(
    shard_dir: Union[str, Path], *, max_shards: Optional[int] = None
) -> pl.DataFrame:
    """Load and concatenate MEDS parquet shards from one split directory.

    Shards are read in filename-numeric order for determinism.
    ``max_shards`` bounds how many are read, so a training run can use a
    real but time-bounded subset of a full extraction (e.g. the real
    MIMIC-IV 3.1 extraction's 292 train shards) rather than requiring the
    whole thing every time.

    Projects down to :data:`_MEDS_EVENT_COLUMNS` *before* collecting, so
    Parquet's columnar layout lets Polars skip reading the other ~16
    columns' data at all, not just drop them after loading -- confirmed
    the hard way that this, not the read+concat strategy, was the real
    OOM cause at full-extraction scale (292 shards, 706M rows): switching
    concat to the lazy/streaming engine alone (avoiding the double
    materialization of ``pl.concat([pl.read_parquet(p) for p in paths])``)
    still OOM-killed at the same ~85GB peak RSS on an 83GB host, since the
    21-column frame was too large on its own regardless of how it was
    assembled. Falls back to whichever of these columns actually exist
    (``hadm_id`` is already optional downstream, see
    :func:`~odyssey.data.sequences.build_patient_sequence`) rather than
    hardcoding the select and failing on a shard schema that lacks one.
    """
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("*.parquet"), key=lambda p: int(p.stem))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"no .parquet shards found in {shard_dir}")
    lf = pl.scan_parquet(paths)
    available = set(lf.collect_schema().names())
    columns = [c for c in _MEDS_EVENT_COLUMNS if c in available]
    return lf.select(columns).collect(engine="streaming")


def _shuffle_buffered(
    items: Iterator[_T], *, buffer_size: int, rng: random.Random
) -> Iterator[_T]:
    """Approximate shuffle over a streaming iterator, bounded to ``buffer_size``.

    Standard reservoir-style streaming shuffle (as used by e.g.
    ``tf.data.Dataset.shuffle``): fills a buffer, then on every further
    item swaps in a uniformly-random buffered item and yields the one
    displaced, draining the buffer in random order once ``items`` is
    exhausted. Not a perfect global shuffle -- an item can never move
    more than roughly ``buffer_size`` positions from its original spot
    -- but bounded to O(buffer_size) memory regardless of how many
    total items there are, unlike collecting everything to shuffle it
    exactly. Generic (not hardcoded to ``PatientSequence``) purely so
    its own tests can exercise the buffering/ordering behavior with
    plain, trivially-comparable ints instead of constructing real
    sequences.
    """
    buffer: List[_T] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        idx = rng.randrange(buffer_size)
        yield buffer[idx]
        buffer[idx] = item
    rng.shuffle(buffer)
    yield from buffer


def iter_patient_sequences(
    events: pl.DataFrame,
    vocab: Vocabulary,
    *,
    max_seq_len: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 4096,
) -> Iterator[PatientSequence]:
    """Yield one :class:`PatientSequence` per subject in ``events``.

    Subjects are shuffled before tokenizing when ``shuffle_seed`` is
    given, matching :class:`~odyssey.data.streaming.PackedLaneSampler`'s
    expectation that its input iterator already arrives in the order
    lanes should see patients (that class does not shuffle itself -- see
    its docstring). Subjects with an empty tokenized sequence (e.g. every
    event was a static, timeless fact) are skipped rather than yielded
    as zero-length.

    Groups subjects via ``group_by(..., maintain_order=True)`` rather
    than ``partition_by(..., as_dict=True)``: the latter eagerly builds
    one sub-DataFrame per subject and holds *all* of them alive in a
    dict simultaneously, which for a real split's ~100K subjects was a
    large, unnecessary cost confirmed on the training VM. An earlier
    version of this function shuffled by sorting the *whole* events
    table into shuffled-subject order first, which fixed that but
    reintroduced the same class of cost one step later (a second full
    copy of the table, alongside the original the caller still holds
    for the next epoch) -- confirmed on the VM to give back essentially
    all of the memory this was meant to save. Shuffling here instead
    uses a bounded streaming buffer (see :func:`_shuffle_buffered`) over
    naturally-ordered groups, at the cost of an approximate rather than
    exact global shuffle.
    """

    def _sequences() -> Iterator[PatientSequence]:
        for _, frame in events.group_by("subject_id", maintain_order=True):
            if frame.height == 0:
                continue
            seq = build_patient_sequence(frame, vocab, max_seq_len=max_seq_len)
            if len(seq) > 0:
                yield seq

    sequences = _sequences()
    if shuffle_seed is None:
        yield from sequences
    else:
        yield from _shuffle_buffered(
            sequences, buffer_size=shuffle_buffer_size, rng=random.Random(shuffle_seed)
        )


def build_concept_label_dicts(
    events: pl.DataFrame, concepts: Sequence[AnyConceptDefinition]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Run label_concepts and reshape its output to per-subject dicts.

    Returns ``(labels, masks)``, each ``subject_id -> (num_concepts,)``
    float tensors in ``concepts`` order -- exactly the shape
    :meth:`~odyssey.models.sequence_model.ConceptBottleneckSequenceModel.compute_streaming_loss`
    expects for ``concept_labels``/``concept_mask``.
    """
    labeled = label_concepts(events, list(concepts))
    names = [c.name for c in concepts]
    label_cols = labeled.select(names).to_numpy()
    mask_cols = labeled.select([f"{name}_observed" for name in names]).to_numpy()
    subject_ids = labeled["subject_id"].to_list()

    labels: Dict[int, torch.Tensor] = {}
    masks: Dict[int, torch.Tensor] = {}
    for i, subject_id in enumerate(subject_ids):
        labels[subject_id] = torch.tensor(label_cols[i], dtype=torch.float32)
        masks[subject_id] = torch.tensor(mask_cols[i], dtype=torch.float32)
    return labels, masks


def build_visit_concept_label_dicts(
    events: pl.DataFrame, concepts: Sequence[AnyConceptDefinition]
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
    """Visit-scoped labels: ``(subject_id, visit_id) -> (num_concepts,)`` dicts.

    The visit-mode counterpart of :func:`build_concept_label_dicts`,
    consuming :func:`~odyssey.data.concepts.label_concepts_by_visit` --
    exactly the keying
    :meth:`~odyssey.models.sequence_model.ConceptBottleneckSequenceModel.compute_streaming_loss`
    expects with ``supervision="visit"``. Only real (``hadm_id``-bearing)
    visits get entries; solo events carry no visit supervision.
    """
    labeled = label_concepts_by_visit(events, list(concepts))
    names = [c.name for c in concepts]
    label_cols = labeled.select(names).to_numpy()
    mask_cols = labeled.select([f"{name}_observed" for name in names]).to_numpy()
    subject_ids = labeled["subject_id"].to_list()
    visit_ids = labeled["hadm_id"].to_list()

    labels: Dict[Tuple[int, int], torch.Tensor] = {}
    masks: Dict[Tuple[int, int], torch.Tensor] = {}
    for i, (subject_id, visit_id) in enumerate(zip(subject_ids, visit_ids)):
        key = (int(subject_id), int(visit_id))
        labels[key] = torch.tensor(label_cols[i], dtype=torch.float32)
        masks[key] = torch.tensor(mask_cols[i], dtype=torch.float32)
    return labels, masks


# Hours before a patient's first event: a first-trigger time strictly earlier
# than any position, so a running label "true from first_time on" is true
# everywhere -- used only for concepts whose label is 0 (never triggered),
# where the running label is false everywhere; the sentinel is +inf there.
NEVER_TRIGGERED = float("inf")


def _first_event_hours(events: pl.DataFrame) -> Dict[int, datetime]:
    """Each subject's sequence time origin: its first non-birth, timed event.

    Must match :func:`~odyssey.data.sequences.build_patient_sequence`,
    which sets ``time_stamps`` as hours since exactly this event, so
    first-trigger times converted against it line up with chunk
    ``time_stamps`` position-for-position.
    """
    origins = (
        events.filter(pl.col("time").is_not_null() & (pl.col("code") != BIRTH_CODE))
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("_origin"))
    )
    return dict(zip(origins["subject_id"].to_list(), origins["_origin"].to_list()))


def build_visit_concept_first_times(
    events: pl.DataFrame, concepts: Sequence[AnyConceptDefinition]
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Per-visit first-trigger times, in hours since the subject's first event.

    Keyed ``(subject_id, visit_id) -> (num_concepts,)``, on the same time
    origin as sequence ``time_stamps``. ``inf`` where the concept never
    triggered in that visit. Together with the visit labels this yields a
    per-position *running* label: concept ``k`` is true at a position
    with time stamp ``t`` iff ``t >= first_times[k]`` -- the label that
    is actually true as of that moment, unlike the whole-visit label,
    which is retrospective.
    """
    labeled = label_concepts_by_visit(events, list(concepts), include_first_time=True)
    origins = _first_event_hours(events)
    names = [c.name for c in concepts]
    subject_ids = labeled["subject_id"].to_list()
    visit_ids = labeled["hadm_id"].to_list()
    first_cols = [labeled[f"{name}_first_time"].to_list() for name in names]

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for i, (subject_id, visit_id) in enumerate(zip(subject_ids, visit_ids)):
        origin = origins[subject_id]
        hours = [
            NEVER_TRIGGERED
            if col[i] is None
            else (col[i] - origin).total_seconds() / 3600.0
            for col in first_cols
        ]
        out[(int(subject_id), int(visit_id))] = torch.tensor(hours, dtype=torch.float32)
    return out


def build_concept_first_times(
    events: pl.DataFrame, concepts: Sequence[AnyConceptDefinition]
) -> Dict[int, torch.Tensor]:
    """Stay-scoped counterpart of :func:`build_visit_concept_first_times`."""
    labeled = label_concepts(events, list(concepts), include_first_time=True)
    origins = _first_event_hours(events)
    names = [c.name for c in concepts]
    subject_ids = labeled["subject_id"].to_list()
    first_cols = [labeled[f"{name}_first_time"].to_list() for name in names]

    out: Dict[int, torch.Tensor] = {}
    for i, subject_id in enumerate(subject_ids):
        origin = origins.get(subject_id)
        if origin is None:
            continue
        hours = [
            NEVER_TRIGGERED
            if col[i] is None
            else (col[i] - origin).total_seconds() / 3600.0
            for col in first_cols
        ]
        out[int(subject_id)] = torch.tensor(hours, dtype=torch.float32)
    return out


def token_type_lookup(vocab: Vocabulary) -> torch.Tensor:
    """``(vocab_size,)`` token id -> code-family id (see :func:`code_type`)."""
    lookup = torch.zeros(len(vocab), dtype=torch.long)
    for token_id, token in vocab.id_to_token.items():
        lookup[token_id] = code_type(token)
    return lookup


def family_loss_weights(
    events: pl.DataFrame, *, alpha: float, cap: float = 20.0, code_col: str = "code"
) -> torch.Tensor:
    """Per-family loss weights ``(share of events) ** -alpha``, mean-1 normalized.

    ``share`` is each code family's fraction of ``events`` (the training
    split's tokenized events, a proxy for its share of forecast targets),
    so with ``alpha=1`` every family contributes equally to the loss and
    with ``alpha=0`` the weights are uniform. Normalized so the weighted
    mean over events is 1 (the loss keeps its scale), then capped at
    ``cap`` so a family that is 0.1% of positions cannot dominate a
    batch on its own. Indexed by family id; families absent from
    ``events`` get weight 0.
    """
    unique_codes = events[code_col].unique()
    families = pl.Series([code_type(c) for c in unique_codes.to_list()], dtype=pl.Int64)
    fam_of_code = pl.DataFrame({code_col: unique_codes, "_family": families})
    counts = (
        events.select(code_col)
        .join(fam_of_code, on=code_col, how="left")
        .group_by("_family")
        .len()
    )
    fam_ids = [int(f) for f in counts["_family"].to_list()]
    n_families = max(fam_ids) + 1
    n = torch.zeros(n_families, dtype=torch.float64)
    for fam, cnt in zip(fam_ids, counts["len"].to_list()):
        n[fam] = float(cnt)
    share = n / n.sum()
    raw = torch.where(
        share > 0, share.clamp_min(1e-12) ** (-alpha), torch.zeros_like(share)
    )
    # normalize: sum_f share_f * w_f = 1
    scale = (share * raw).sum()
    weights = torch.where(scale > 0, raw / scale, raw)
    return weights.clamp_max(cap).to(torch.float32)


def count_subjects(events: pl.DataFrame) -> int:
    """Count distinct subjects in ``events`` -- for logging/progress only."""
    return int(events["subject_id"].n_unique())


def build_vocabulary(
    train_events: pl.DataFrame,
    *,
    min_count: int,
    max_size: int,
    backoff: Optional[str] = None,
) -> Vocabulary:
    """Build a :class:`Vocabulary` from the training split's real code frequencies.

    Never call this on tuning/held_out events -- the same leakage
    discipline :class:`~odyssey.data.value_binning.QuantileBinner` and
    :meth:`Vocabulary.build` are both already documented to require.

    Counts codes with Polars' vectorized ``group_by`` rather than going
    through :meth:`Vocabulary.build`'s plain ``Counter(codes)`` path: a
    real train split has tens of millions of event rows, and
    materializing that column as a Python list of ``str`` objects (one
    Python object per *row*, not per unique code) was a large,
    unnecessary memory cost -- confirmed as a real contributor to an OOM
    on a 50-shard MIMIC-IV run. ``group_by`` is bounded by vocabulary
    cardinality instead.
    """
    counts = dict(train_events.group_by("code").len().iter_rows())
    return Vocabulary.build_from_counts(
        counts, min_count=min_count, max_size=max_size, backoff=backoff
    )


__all__ = [
    "load_meds_shards",
    "iter_patient_sequences",
    "build_concept_label_dicts",
    "build_visit_concept_label_dicts",
    "count_subjects",
    "build_vocabulary",
]
