"""Convert raw MEDS events into patient token sequences.

Turns one subject's raw MEDS event stream (``subject_id``, ``time``,
``code``, and optionally ``hadm_id``) into the token/type/time/age/visit
arrays :class:`odyssey.models.embeddings.ClinicalEventEmbeddings` consumes,
then pads/collates many subjects into a
:class:`odyssey.data.types.ClinicalSequenceBatch`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.vocabulary import PAD_ID, Vocabulary, code_type


BIRTH_CODE = "MEDS_BIRTH"
HOURS_PER_YEAR = 24.0 * 365.25


@dataclass
class PatientSequence:
    """One subject's tokenized event sequence, ready for padding/batching."""

    subject_id: int
    concept_ids: List[int]
    type_ids: List[int]
    time_stamps: List[float]
    """Hours since this sequence's first event (not since epoch) -- absolute
    values, not deltas; :class:`~odyssey.models.embeddings.TimeEmbeddingLayer`
    computes deltas internally."""
    ages: List[float]
    """Age in years at each event; 0.0 for every event if no MEDS_BIRTH
    event was present to compute a real age from."""
    visit_orders: List[int]
    visit_segments: List[int]
    """0 = first event of a visit, 1 = middle, 2 = last (matches
    ClinicalEventEmbeddings' default visit_order_size=3)."""

    def __len__(self) -> int:
        """Return the number of events in this sequence."""
        return len(self.concept_ids)


def _assign_visits(
    hadm_ids: List[Optional[int]], max_num_visits: int
) -> Tuple[List[int], List[int]]:
    """Derive (visit_order, visit_segment) from admission ids.

    Events sharing an ``hadm_id`` belong to the same visit. Events without
    one (e.g. outpatient labs) each get their own single-event visit --
    a v1 simplification; a real outpatient-visit grouping (e.g. by day)
    is a reasonable follow-up but not implemented here.
    """
    keys: List[Tuple[str, object]] = []
    solo_counter = 0
    for hadm_id in hadm_ids:
        if hadm_id is not None:
            keys.append(("hadm", hadm_id))
        else:
            keys.append(("solo", solo_counter))
            solo_counter += 1

    order_by_key: Dict[Tuple[str, object], int] = {}
    visit_orders = []
    for key in keys:
        if key not in order_by_key:
            order_by_key[key] = min(len(order_by_key), max_num_visits - 1)
        visit_orders.append(order_by_key[key])

    visit_segments = [1] * len(keys)
    i = 0
    while i < len(keys):
        j = i
        while j < len(keys) and keys[j] == keys[i]:
            j += 1
        for k in range(i, j):
            if k == i:
                visit_segments[k] = 0
            elif k == j - 1:
                visit_segments[k] = 2
        i = j
    return visit_orders, visit_segments


def build_patient_sequence(
    events: pl.DataFrame,
    vocabulary: Vocabulary,
    *,
    max_seq_len: Optional[int] = None,
    max_num_visits: int = 512,
) -> PatientSequence:
    """Build one subject's tokenized sequence from their raw MEDS events.

    ``events`` must contain a single ``subject_id``. Static, timeless facts
    (``time`` is null, e.g. ``GENDER//...``) are dropped, since every event
    needs a real timestamp; ``MEDS_BIRTH`` is consumed to compute ages, not
    included as a sequence token. If ``max_seq_len`` truncates, the most
    recent events are kept (older history is less relevant to near-term
    forecasting).
    """
    events = events.filter(pl.col("time").is_not_null())
    birth_rows = events.filter(pl.col("code") == BIRTH_CODE)
    birth_time = birth_rows["time"][0] if birth_rows.height > 0 else None
    events = events.filter(pl.col("code") != BIRTH_CODE).sort("time")

    subject_id = int(events["subject_id"][0]) if events.height > 0 else -1
    codes = events["code"].to_list()
    times = events["time"].to_list()
    hadm_ids = (
        events["hadm_id"].to_list()
        if "hadm_id" in events.columns
        else [None] * len(codes)
    )

    if not times:
        return PatientSequence(subject_id, [], [], [], [], [], [])

    first_time = times[0]
    time_stamps = [(t - first_time).total_seconds() / 3600.0 for t in times]
    if birth_time is not None:
        ages = [
            (t - birth_time).total_seconds() / 3600.0 / HOURS_PER_YEAR for t in times
        ]
    else:
        ages = [0.0] * len(times)

    concept_ids = [vocabulary.encode(c) for c in codes]
    type_ids = [code_type(c) for c in codes]
    visit_orders, visit_segments = _assign_visits(hadm_ids, max_num_visits)

    if max_seq_len is not None and len(concept_ids) > max_seq_len:
        concept_ids = concept_ids[-max_seq_len:]
        type_ids = type_ids[-max_seq_len:]
        time_stamps = time_stamps[-max_seq_len:]
        ages = ages[-max_seq_len:]
        visit_orders = visit_orders[-max_seq_len:]
        visit_segments = visit_segments[-max_seq_len:]

    return PatientSequence(
        subject_id=subject_id,
        concept_ids=concept_ids,
        type_ids=type_ids,
        time_stamps=time_stamps,
        ages=ages,
        visit_orders=visit_orders,
        visit_segments=visit_segments,
    )


def collate_patient_sequences(
    sequences: List[PatientSequence], *, padding_idx: int = PAD_ID
) -> ClinicalSequenceBatch:
    """Right-pad a list of :class:`PatientSequence` into one batched tensor."""
    max_len = max((len(s) for s in sequences), default=0)
    batch = len(sequences)

    concept_ids = torch.full((batch, max_len), padding_idx, dtype=torch.long)
    type_ids = torch.zeros((batch, max_len), dtype=torch.long)
    time_stamps = torch.zeros((batch, max_len), dtype=torch.float)
    ages = torch.zeros((batch, max_len), dtype=torch.float)
    visit_orders = torch.zeros((batch, max_len), dtype=torch.long)
    visit_segments = torch.zeros((batch, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        n = len(seq)
        if n == 0:
            continue
        concept_ids[i, :n] = torch.tensor(seq.concept_ids, dtype=torch.long)
        type_ids[i, :n] = torch.tensor(seq.type_ids, dtype=torch.long)
        time_stamps[i, :n] = torch.tensor(seq.time_stamps, dtype=torch.float)
        ages[i, :n] = torch.tensor(seq.ages, dtype=torch.float)
        visit_orders[i, :n] = torch.tensor(seq.visit_orders, dtype=torch.long)
        visit_segments[i, :n] = torch.tensor(seq.visit_segments, dtype=torch.long)

    return ClinicalSequenceBatch(
        concept_ids=concept_ids,
        aux=AuxiliaryInputs(
            type_ids=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        ),
    )
