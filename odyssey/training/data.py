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
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

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


def iter_patient_sequences(
    events: pl.DataFrame,
    vocab: Vocabulary,
    *,
    max_seq_len: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> Iterator[PatientSequence]:
    """Yield one :class:`PatientSequence` per subject in ``events``.

    Subjects are shuffled before tokenizing when ``shuffle_seed`` is
    given, matching :class:`~odyssey.data.streaming.PackedLaneSampler`'s
    expectation that its input iterator already arrives in the order
    lanes should see patients (that class does not shuffle itself -- see
    its docstring). Subjects with an empty tokenized sequence (e.g. every
    event was a static, timeless fact) are skipped rather than yielded
    as zero-length.
    """
    subject_ids = events["subject_id"].unique().to_list()
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(subject_ids)

    partitions = events.partition_by("subject_id", as_dict=True)
    for subject_id in subject_ids:
        frame = partitions.get((subject_id,))
        if frame is None or frame.height == 0:
            continue
        seq = build_patient_sequence(frame, vocab, max_seq_len=max_seq_len)
        if len(seq) > 0:
            yield seq


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


def build_vocabulary(train_events: pl.DataFrame, *, min_count: int, max_size: int) -> Vocabulary:
    """Build a :class:`Vocabulary` from the training split's real code frequencies.

    Never call this on tuning/held_out events -- the same leakage
    discipline :class:`~odyssey.data.value_binning.QuantileBinner` and
    :meth:`Vocabulary.build` are both already documented to require.
    """
    return Vocabulary.build(
        train_events["code"].to_list(), min_count=min_count, max_size=max_size
    )


__all__ = [
    "load_meds_shards",
    "iter_patient_sequences",
    "build_concept_label_dicts",
    "count_subjects",
    "build_vocabulary",
]
