"""Convert raw MEDS events into patient token sequences.

Turns one subject's raw MEDS event stream (``subject_id``, ``time``,
``code``, and optionally ``hadm_id``) into the token/type/time/age/visit
arrays :class:`odyssey.models.embeddings.ClinicalEventEmbeddings` consumes,
then pads/collates many subjects into a
:class:`odyssey.data.types.ClinicalSequenceBatch`.
"""

from dataclasses import dataclass, field

import polars as pl
import torch

from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.data.value_binning import VALUE_Z_COL
from odyssey.data.vocabulary import PAD_ID, Vocabulary, code_type


BIRTH_CODE = "MEDS_BIRTH"
HOURS_PER_YEAR = 24.0 * 365.25

# Sentinel visit id for events without a real hadm_id (solo/outpatient
# events); such positions never carry visit-scoped concept supervision.
NO_VISIT = -1


@dataclass
class PatientSequence:
    """One subject's tokenized event sequence, ready for padding/batching."""

    subject_id: int
    concept_ids: list[int]
    type_ids: list[int]
    time_stamps: list[float]
    """Hours since this sequence's first event (not since epoch) -- absolute
    values, not deltas; :class:`~odyssey.models.embeddings.TimeEmbeddingLayer`
    computes deltas internally."""
    ages: list[float]
    """Age in years at each event; 0.0 for every event if no MEDS_BIRTH
    event was present to compute a real age from."""
    visit_orders: list[int]
    visit_segments: list[int]
    """0 = first event of a visit, 1 = middle, 2 = last (matches
    ClinicalEventEmbeddings' default visit_order_size=3)."""

    visit_ids: list[int] = field(default_factory=list)
    """Per-token raw visit identifier (``hadm_id``; eICU's unit-stay id
    plays the same role), or ``NO_VISIT`` for events without one. Keys
    visit-scoped concept labels
    (:func:`odyssey.data.concepts.label_concepts_by_visit`); empty on
    sequences built before this field existed (treated as all
    ``NO_VISIT``)."""

    visit_ends: list[bool] = field(default_factory=list)
    """True at the last token of each *real* (``hadm_id``-bearing) visit
    -- the position where visit-scoped concept supervision applies. The
    last occurrence across the whole sequence, not per contiguous run,
    since a visit's events can be interleaved with solo events."""

    static_mask: list[bool] = field(default_factory=list)
    """True for the timeless facts placed at the sequence start (GENDER,
    RACE, ...). They are inputs only: the streaming sampler never makes
    them prediction targets (predicting race from sex is not forecasting).
    Empty on sequences built before this field existed (treated as all
    False)."""

    values: list[float] = field(default_factory=list)
    """Standardized numeric value per token (the ``numeric_z`` column of
    :func:`~odyssey.data.value_binning.add_value_tokens`), ``nan`` where
    the event carries none; empty on sequences built without that column
    (treated as all-``nan``). Input-side only: targets stay bin tokens."""

    def __len__(self) -> int:
        """Return the number of events in this sequence."""
        return len(self.concept_ids)

    def tail(self, n: int, *, rebase_times: bool = False) -> "PatientSequence":
        """Return the most recent ``n`` tokens (the whole sequence if shorter).

        Optional per-token fields that were never populated stay empty.
        ``rebase_times`` subtracts the kept window's first timestamp so the
        result again obeys the "hours since this sequence's first event"
        convention (the packed-context path needs that; the streaming path
        keeps absolute offsets).
        """
        total = len(self)
        if total <= n:
            return self

        def _tail(values: list[object]) -> list[object]:
            return values[-n:] if len(values) == total else []

        times = self.time_stamps[-n:]
        if rebase_times:
            first = times[0]
            times = [t - first for t in times]
        return PatientSequence(
            subject_id=self.subject_id,
            concept_ids=self.concept_ids[-n:],
            type_ids=self.type_ids[-n:],
            time_stamps=times,
            ages=self.ages[-n:],
            visit_orders=self.visit_orders[-n:],
            visit_segments=self.visit_segments[-n:],
            visit_ids=_tail(list(self.visit_ids)),  # type: ignore[arg-type]
            visit_ends=_tail(list(self.visit_ends)),  # type: ignore[arg-type]
            static_mask=_tail(list(self.static_mask)),  # type: ignore[arg-type]
            values=_tail(list(self.values)),  # type: ignore[arg-type]
        )

    def head(self, n: int) -> "PatientSequence":
        """Return the EARLIEST ``n`` tokens (the whole sequence if shorter).

        The causal-prefix counterpart of :meth:`tail`: everything strictly
        after position ``n - 1`` is dropped, so the result contains no
        future information relative to that position. Times keep the
        original origin (the first event is the sequence's first event, so
        there is nothing to rebase). Optional per-token fields that were
        never populated stay empty.
        """
        total = len(self)
        if total <= n:
            return self

        def _head(values: list[object]) -> list[object]:
            return values[:n] if len(values) == total else []

        return PatientSequence(
            subject_id=self.subject_id,
            concept_ids=self.concept_ids[:n],
            type_ids=self.type_ids[:n],
            time_stamps=self.time_stamps[:n],
            ages=self.ages[:n],
            visit_orders=self.visit_orders[:n],
            visit_segments=self.visit_segments[:n],
            visit_ids=_head(list(self.visit_ids)),  # type: ignore[arg-type]
            visit_ends=_head(list(self.visit_ends)),  # type: ignore[arg-type]
            static_mask=_head(list(self.static_mask)),  # type: ignore[arg-type]
            values=_head(list(self.values)),  # type: ignore[arg-type]
        )


def _assign_visits(
    hadm_ids: list[int | None], max_num_visits: int
) -> tuple[list[int], list[int]]:
    """Derive (visit_order, visit_segment) from admission ids.

    Events sharing an ``hadm_id`` belong to the same visit. Events without
    one (e.g. outpatient labs) each get their own single-event visit --
    a v1 simplification; a real outpatient-visit grouping (e.g. by day)
    is a reasonable follow-up but not implemented here.
    """
    keys: list[tuple[str, object]] = []
    solo_counter = 0
    for hadm_id in hadm_ids:
        if hadm_id is not None:
            keys.append(("hadm", hadm_id))
        else:
            keys.append(("solo", solo_counter))
            solo_counter += 1

    order_by_key: dict[tuple[str, object], int] = {}
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
    # FROZEN, DO NOT CHANGE (review finding 14): when a visit has exactly
    # one event, k == i and k == j - 1 are the same index, so only the
    # `if k == i` branch above ever runs -- a single-event visit is always
    # coded segment 0 ("first"), never 2 ("last"). Pinned as intentional:
    # see test_events_without_hadm_id_each_get_their_own_visit. Changing
    # this would alter tokenization for every source (MIMIC-IV, eICU,
    # GEMINI) mid-program.
    return visit_orders, visit_segments


def build_patient_sequence(
    events: pl.DataFrame,
    vocabulary: Vocabulary,
    *,
    max_seq_len: int | None = None,
    max_num_visits: int = 512,
) -> PatientSequence:
    """Build one subject's tokenized sequence from their raw MEDS events.

    ``events`` must contain a single ``subject_id``. Static, timeless facts
    (``time`` is null, e.g. ``GENDER//...``) are placed as the first tokens
    of the sequence, stamped with the first timed event's time and visit,
    so the model sees them before anything else (before this change they
    were dropped, and MIMIC-IV's sex never reached the model);
    ``MEDS_BIRTH`` is consumed to compute ages, not included as a sequence
    token. If ``max_seq_len`` truncates, the most recent events are kept
    (older history is less relevant to near-term forecasting).

    Several events commonly share the exact same timestamp (e.g. a panel
    of labs drawn together) -- there's no true order between them, but
    the tokenized sequence still needs *some* fixed, reproducible order.
    ``sort(..., maintain_order=True)`` makes that order deterministic
    (same-timestamp events keep their relative order from ``events``,
    which traces back to the source shard's own row order): Polars'
    default (``maintain_order=False``, the faster path) does not
    guarantee this, which would otherwise make the exact tokenization of
    a patient with simultaneous events not reliably reproducible across
    runs/environments -- a real problem for auditable eval results, even
    though it hasn't been observed to actually vary in practice yet.

    Raises
    ------
    ValueError
        If ``events`` contains more than one distinct ``subject_id`` --
        silently keying off the first row would otherwise merge two
        patients' histories into one token stream.
    """
    n_subjects = events["subject_id"].n_unique() if events.height > 0 else 0
    if n_subjects > 1:
        raise ValueError(
            f"build_patient_sequence expects a single subject_id, got "
            f"{n_subjects} distinct subject_ids"
        )

    static = events.filter(pl.col("time").is_null() & (pl.col("code") != BIRTH_CODE))
    events = events.filter(pl.col("time").is_not_null())
    birth_rows = events.filter(pl.col("code") == BIRTH_CODE)
    birth_time = birth_rows["time"][0] if birth_rows.height > 0 else None
    events = events.filter(pl.col("code") != BIRTH_CODE).sort(
        "time", maintain_order=True
    )
    n_static = 0
    if static.height > 0 and events.height > 0:
        # Static facts lead the sequence at the first timed event's instant
        # and visit; a static-only subject has no timeline and yields nothing.
        first = events.head(1)
        static = static.with_columns(
            pl.lit(first["time"][0]).alias("time"),
            *(
                [pl.lit(first["hadm_id"][0]).alias("hadm_id")]
                if "hadm_id" in events.columns
                else []
            ),
        ).select(events.columns)
        events = pl.concat([static, events], how="vertical_relaxed")
        n_static = static.height

    subject_id = int(events["subject_id"][0]) if events.height > 0 else -1
    codes = events["code"].to_list()
    times = events["time"].to_list()
    hadm_ids = (
        events["hadm_id"].to_list()
        if "hadm_id" in events.columns
        else [None] * len(codes)
    )
    values = (
        [float("nan") if v is None else float(v) for v in events[VALUE_Z_COL].to_list()]
        if VALUE_Z_COL in events.columns
        else []
    )

    if not times:
        return PatientSequence(subject_id, [], [], [], [], [], [])

    first_time = times[0]
    # Computed as a polars expression, not Python's native
    # timedelta.total_seconds() -- the two are conceptually equivalent but
    # not bit-identical (different internal rounding), which showed up as
    # a real bug: a landmark bucket computed from this sequence's own
    # time_stamps (torch path) disagreeing with one computed from
    # _index_rows_from_events' polars pipeline (ground truth) on the exact
    # same real-world event, off by ~1e-13 hours -- just enough to flip a
    # floor() right at a bucket boundary (root-caused 2026-08-23; see
    # odyssey.inference.alerts.verify_packed_landmark_rows). Matching the
    # exact same expression removes the second, independently-rounded
    # derivation rather than tolerating the disagreement with an epsilon.
    time_stamps = events.select(
        ((pl.col("time") - pl.lit(first_time)).dt.total_seconds() / 3600.0).alias("_h")
    )["_h"].to_list()
    if birth_time is not None:
        ages = events.select(
            (
                (pl.col("time") - pl.lit(birth_time)).dt.total_seconds()
                / 3600.0
                / HOURS_PER_YEAR
            ).alias("_a")
        )["_a"].to_list()
    else:
        ages = [0.0] * len(times)

    concept_ids = [vocabulary.encode(c) for c in codes]
    type_ids = [code_type(c) for c in codes]
    visit_orders, visit_segments = _assign_visits(hadm_ids, max_num_visits)

    visit_ids = [NO_VISIT if h is None else int(h) for h in hadm_ids]
    last_pos: dict[int, int] = {}
    for i, vid in enumerate(visit_ids):
        if vid != NO_VISIT:
            last_pos[vid] = i
    visit_ends = [
        vid != NO_VISIT and last_pos[vid] == i for i, vid in enumerate(visit_ids)
    ]

    static_mask = [i < n_static for i in range(len(concept_ids))]

    seq = PatientSequence(
        subject_id=subject_id,
        concept_ids=concept_ids,
        type_ids=type_ids,
        time_stamps=time_stamps,
        ages=ages,
        visit_orders=visit_orders,
        visit_segments=visit_segments,
        visit_ids=visit_ids,
        visit_ends=visit_ends,
        static_mask=static_mask,
        values=values,
    )
    if max_seq_len is not None:
        seq = seq.tail(max_seq_len)
    return seq


def collate_patient_sequences(
    sequences: list[PatientSequence], *, padding_idx: int = PAD_ID
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
    values = torch.full((batch, max_len), float("nan"), dtype=torch.float)

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
        if len(seq.values) == n:
            values[i, :n] = torch.tensor(seq.values, dtype=torch.float)

    return ClinicalSequenceBatch(
        concept_ids=concept_ids,
        aux=AuxiliaryInputs(
            type_ids=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
            values=values,
        ),
    )
