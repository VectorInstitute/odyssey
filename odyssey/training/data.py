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
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import polars as pl
import torch

from odyssey.data.concepts import (
    AnyConceptDefinition,
    label_concepts,
    label_concepts_by_visit,
)
from odyssey.data.sequences import PatientSequence, build_patient_sequence
from odyssey.data.vocabulary import Vocabulary


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


def count_subjects(events: pl.DataFrame) -> int:
    """Count distinct subjects in ``events`` -- for logging/progress only."""
    return int(events["subject_id"].n_unique())


def build_vocabulary(
    train_events: pl.DataFrame, *, min_count: int, max_size: int
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
    return Vocabulary.build_from_counts(counts, min_count=min_count, max_size=max_size)


__all__ = [
    "load_meds_shards",
    "iter_patient_sequences",
    "build_concept_label_dicts",
    "build_visit_concept_label_dicts",
    "count_subjects",
    "build_vocabulary",
]
