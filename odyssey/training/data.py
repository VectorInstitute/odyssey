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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import polars as pl
import torch

from odyssey.data.concepts import AnyConceptDefinition, label_concepts
from odyssey.data.sequences import PatientSequence, build_patient_sequence
from odyssey.data.vocabulary import Vocabulary


def load_meds_shards(
    shard_dir: Union[str, Path], *, max_shards: Optional[int] = None
) -> pl.DataFrame:
    """Load and concatenate MEDS parquet shards from one split directory.

    Shards are read in filename-numeric order for determinism.
    ``max_shards`` bounds how many are read, so a training run can use a
    real but time-bounded subset of a full extraction (e.g. the real
    MIMIC-IV 3.1 extraction's 292 train shards) rather than requiring the
    whole thing every time.
    """
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("*.parquet"), key=lambda p: int(p.stem))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"no .parquet shards found in {shard_dir}")
    return pl.concat([pl.read_parquet(p) for p in paths])


def _shuffle_buffered(
    items: Iterator[PatientSequence], *, buffer_size: int, rng: random.Random
) -> Iterator[PatientSequence]:
    """Approximate shuffle over a streaming iterator, bounded to ``buffer_size``.

    Standard reservoir-style streaming shuffle (as used by e.g.
    ``tf.data.Dataset.shuffle``): fills a buffer, then on every further
    item swaps in a uniformly-random buffered item and yields the one
    displaced, draining the buffer in random order once ``items`` is
    exhausted. Not a perfect global shuffle -- an item can never move
    more than roughly ``buffer_size`` positions from its original spot
    -- but bounded to O(buffer_size) memory regardless of how many
    total items there are, unlike collecting everything to shuffle it
    exactly.
    """
    buffer: List[PatientSequence] = []
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
    "count_subjects",
    "build_vocabulary",
]
