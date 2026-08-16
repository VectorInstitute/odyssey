"""End-to-end tests: PackedLaneSampler + a stateful backbone + segment-aware loss.

Uses TinyGRUBackbone as the only backbone that can actually execute here
(no CUDA). This validates the sampler/interface/pooling plumbing, which is
backbone-agnostic by construction -- it says nothing about whether the
real EHRHybridBackbone behaves the same way; that needs
``tests/odyssey/models/backbones/test_hybrid_gpu.py`` on a CUDA host.
"""

from typing import Dict, Iterator, List

import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import PackedLaneSampler, StreamingChunk
from odyssey.data.types import AuxiliaryInputs, ClinicalSequenceBatch
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights, concept_loss
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)
from odyssey.training.running_labels import randint_intervention


VOCAB_SIZE = 60
HIDDEN_SIZE = 16
NUM_CONCEPTS = 3
EMBEDDING_DIM = 4
PADDING_IDX = 0


def _seq(subject_id: int, n: int) -> PatientSequence:
    return PatientSequence(
        subject_id=subject_id,
        concept_ids=[1 + ((subject_id * 37 + i) % (VOCAB_SIZE - 1)) for i in range(n)],
        type_ids=[1] * n,
        time_stamps=[float(i) for i in range(n)],
        ages=[40.0] * n,
        visit_orders=[0] * n,
        visit_segments=[0] * n,
    )


def _patients(seqs: List[PatientSequence]) -> Iterator[PatientSequence]:
    return iter(seqs)


def _make_backbone() -> TinyGRUBackbone:
    return TinyGRUBackbone(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        padding_idx=PADDING_IDX,
    )


def _make_model() -> ConceptBottleneckSequenceModel:
    return ConceptBottleneckSequenceModel(
        backbone=_make_backbone(),
        vocab_size=VOCAB_SIZE,
        num_concepts=NUM_CONCEPTS,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=PADDING_IDX,
    )


def _labels(subject_ids: List[int]) -> Dict[int, torch.Tensor]:
    return {sid: torch.randint(0, 2, (NUM_CONCEPTS,)).float() for sid in subject_ids}


def _detach_state(state: TimeAwareState) -> TimeAwareState:
    """Truncate BPTT across chunks.

    Detach both the recurrent state and the carried timestamp before
    reusing them as the next chunk's state.
    """
    return TimeAwareState(
        recurrent=tuple(h.detach() for h in state.recurrent),
        prev_time_stamps=state.prev_time_stamps.detach(),
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_streaming_loss_is_finite_when_a_patient_ends_in_chunk() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 4)]), num_lanes=1, chunk_size=4)
    chunk = sampler.next_chunk()
    assert chunk.patient_end[0].any()

    total, components, new_state = model.compute_streaming_loss(chunk, _labels([1]))

    assert torch.isfinite(total)
    assert set(components) == {
        "task_loss",
        "concept_loss",
        "orthogonality_loss",
        "observability_loss",
    }
    assert new_state is not None


def test_streaming_loss_concept_terms_are_zero_when_no_patient_ends() -> None:
    # a single long patient spanning multiple chunks: the first chunk has
    # no patient_end at all.
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 20)]), num_lanes=1, chunk_size=4)
    chunk = sampler.next_chunk()
    assert not chunk.patient_end.any()

    total, components, _ = model.compute_streaming_loss(chunk, _labels([1]))

    assert components["concept_loss"].item() == 0.0
    assert components["orthogonality_loss"].item() == 0.0
    assert torch.isfinite(total)
    assert total.item() == components["task_loss"].item()


def test_streaming_loss_backward_works_with_no_patient_end() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 20)]), num_lanes=1, chunk_size=4)
    chunk = sampler.next_chunk()

    total, _, _ = model.compute_streaming_loss(chunk, {})
    total.backward()

    assert model.backbone.embeddings.embeddings.word_embeddings.weight.grad is not None


def test_streaming_task_loss_is_zero_not_nan_when_chunk_has_no_real_target() -> None:
    # a single-event patient with chunk_size=1: its one token has no
    # successor to predict, so real_mask is False everywhere -- F.cross_
    # entropy's mean reduction would otherwise divide 0 valid elements by
    # 0, giving NaN (see _streaming_next_token_loss).
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 1)]), num_lanes=1, chunk_size=1)
    chunk = sampler.next_chunk()
    assert not chunk.real_mask.any()

    total, components, _ = model.compute_streaming_loss(chunk, _labels([1]))

    assert components["task_loss"].item() == 0.0
    assert torch.isfinite(total)
    total.backward()  # must not crash even though every loss term is zero


def _make_chunk_with_three_single_token_patients() -> StreamingChunk:
    """Build a chunk with 3 patient_ends directly, bypassing PackedLaneSampler.

    PackedLaneSampler itself never packs a second patient's reset into the
    same chunk (see ``tests/odyssey/data/test_streaming.py`` --
    ``EHRHybridBackbone`` can't resume mid-chunk state for more than one
    segment). But ``compute_streaming_loss``'s pooling logic is
    backbone-agnostic and must still correctly supervise every
    ``patient_end`` in a chunk for a backbone (like ``TinyGRUBackbone``
    here) that *can* handle mid-chunk resets -- so this builds that chunk
    shape by hand rather than deriving it from the sampler.
    """
    return StreamingChunk(
        batch=ClinicalSequenceBatch(
            concept_ids=torch.tensor([[10, 20, 30]]),
            aux=AuxiliaryInputs(
                type_ids=torch.ones(1, 3, dtype=torch.long),
                time_stamps=torch.tensor([[0.0, 1.0, 2.0]]),
                ages=torch.full((1, 3), 40.0),
                visit_orders=torch.zeros(1, 3, dtype=torch.long),
                visit_segments=torch.zeros(1, 3, dtype=torch.long),
            ),
        ),
        targets=torch.tensor([[11, 21, 31]]),
        reset_mask=torch.tensor([[True, True, True]]),
        real_mask=torch.tensor([[True, True, True]]),
        subject_ids=torch.tensor([[1, 2, 3]]),
        patient_end=torch.tensor([[True, True, True]]),
        visit_ids=torch.tensor([[-1, -1, -1]]),
        visit_end=torch.tensor([[False, False, False]]),
    )


def test_multiple_patient_ends_in_one_chunk_are_all_supervised() -> None:
    model = _make_model()
    chunk = _make_chunk_with_three_single_token_patients()
    assert chunk.patient_end[0].tolist() == [True, True, True]

    # concept_loss should reflect all 3 subjects, not just one -- compare
    # against manually computing the same BCE over all 3. eval() mode: the
    # bottleneck's dropout would otherwise make two separate forward calls
    # differ for reasons unrelated to what's being tested here.
    model.eval()
    labels = _labels([1, 2, 3])
    with torch.no_grad():
        _, components, _ = model.compute_streaming_loss(
            chunk,
            labels,
            loss_weights=ConceptBottleneckLossWeights(concept=1.0, orthogonality=0.0),
        )
        _, bottleneck_out, _ = model(chunk.batch, reset_mask=chunk.reset_mask)

    expected_concept_loss = F.binary_cross_entropy_with_logits(
        bottleneck_out.concept_logits[0], torch.stack([labels[1], labels[2], labels[3]])
    )
    assert torch.allclose(components["concept_loss"], expected_concept_loss, atol=1e-5)


def test_missing_subject_label_raises_clear_error() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 4)]), num_lanes=1, chunk_size=4)
    chunk = sampler.next_chunk()

    try:
        model.compute_streaming_loss(chunk, {})  # subject 1's label missing
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "1" in str(exc)


# ---------------------------------------------------------------------------
# State carries across chunks
# ---------------------------------------------------------------------------


def test_state_from_previous_chunk_changes_next_chunk_output() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 20)]), num_lanes=1, chunk_size=4)
    chunk1 = sampler.next_chunk()
    chunk2 = sampler.next_chunk()

    model.eval()
    with torch.no_grad():
        _, _, state1 = model(chunk1.batch)
        _, out_with_state, _ = model(chunk2.batch, state=state1)
        _, out_fresh, _ = model(chunk2.batch, state=None)

    assert not torch.allclose(
        out_with_state.concept_logits, out_fresh.concept_logits, atol=1e-6
    )


def test_reset_row_matches_fresh_state_other_row_keeps_carried_state() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(
        _patients([_seq(1, 4), _seq(2, 4)]), num_lanes=2, chunk_size=4
    )
    chunk1 = sampler.next_chunk()
    chunk2 = sampler.next_chunk()  # both patients continue for 4 more events each

    model.eval()
    with torch.no_grad():
        _, _, state1 = model(chunk1.batch)
        reset_mask = torch.zeros(2, 4, dtype=torch.bool)
        reset_mask[0, 0] = True  # reset lane 0 only, at the chunk boundary
        _, out_reset, _ = model(chunk2.batch, state=state1, reset_mask=reset_mask)
        _, out_fresh, _ = model(chunk2.batch, state=None)

    assert torch.allclose(
        out_reset.concept_logits[0], out_fresh.concept_logits[0], atol=1e-6
    )
    assert not torch.allclose(
        out_reset.concept_logits[1], out_fresh.concept_logits[1], atol=1e-6
    )


# ---------------------------------------------------------------------------
# Full training loop: sampler -> streaming loss -> backward -> optimizer,
# across multiple chunks with persistent state, reduces loss.
# ---------------------------------------------------------------------------


def test_streaming_training_loop_reduces_loss() -> None:
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    weights = ConceptBottleneckLossWeights(concept=1.0, orthogonality=0.05)

    patients = [_seq(sid, 12) for sid in range(1, 9)]
    labels = _labels([p.subject_id for p in patients])

    losses = []
    for _ in range(30):
        sampler = PackedLaneSampler(
            _patients(list(patients)), num_lanes=2, chunk_size=6, seed=0
        )
        state = None
        epoch_losses = []
        for chunk in sampler:
            total, components, state = model.compute_streaming_loss(
                chunk, labels, state=state, loss_weights=weights
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            state = _detach_state(state)  # truncate BPTT across chunks
            epoch_losses.append(components["task_loss"].item())
        losses.append(sum(epoch_losses) / len(epoch_losses))

    assert losses[-1] < losses[0] * 0.8


def test_baseline_model_streaming_loss_and_training_loop() -> None:
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=_make_backbone(), vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    patients = [_seq(sid, 12) for sid in range(1, 5)]

    losses = []
    for _ in range(30):
        sampler = PackedLaneSampler(
            _patients(list(patients)), num_lanes=2, chunk_size=6, seed=0
        )
        state = None
        epoch_losses = []
        for chunk in sampler:
            loss, components, state = model.compute_streaming_loss(chunk, state=state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = _detach_state(state)
            epoch_losses.append(components["task_loss"].item())
        losses.append(sum(epoch_losses) / len(epoch_losses))

    assert losses[-1] < losses[0] * 0.8


# ---------------------------------------------------------------------------
# Visit-scoped concept supervision
# ---------------------------------------------------------------------------


def _seq_visits(subject_id: int, visit_ids: List[int]) -> PatientSequence:
    n = len(visit_ids)
    last = {}
    for i, v in enumerate(visit_ids):
        if v != -1:
            last[v] = i
    seq = _seq(subject_id, n)
    seq.visit_ids.extend(visit_ids)
    seq.visit_ends.extend(v != -1 and last[v] == i for i, v in enumerate(visit_ids))
    return seq


def _visit_labels(keys: List[tuple]) -> Dict[tuple, torch.Tensor]:
    return {k: torch.randint(0, 2, (NUM_CONCEPTS,)).float() for k in keys}


def test_visit_supervision_pools_at_every_visit_end() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(
        _patients([_seq_visits(1, [10, 10, 11, 11])]), num_lanes=1, chunk_size=4
    )
    chunk = sampler.next_chunk()
    labels = _visit_labels([(1, 10), (1, 11)])
    masks = {k: torch.ones(NUM_CONCEPTS) for k in labels}
    total, components, _ = model.compute_streaming_loss(
        chunk, labels, masks, supervision="visit"
    )
    assert torch.isfinite(total)
    assert components["concept_loss"] > 0


def test_visit_supervision_ignores_solo_events() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(
        _patients([_seq_visits(1, [-1, -1, -1])]), num_lanes=1, chunk_size=3
    )
    chunk = sampler.next_chunk()
    total, components, _ = model.compute_streaming_loss(
        chunk, {}, {}, supervision="visit"
    )
    # no visit ends -> auxiliary terms are exactly zero, task loss remains
    assert components["concept_loss"].item() == 0.0
    assert torch.isfinite(total)


def test_visit_supervision_missing_label_raises_clear_error() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(
        _patients([_seq_visits(1, [10, 10])]), num_lanes=1, chunk_size=2
    )
    chunk = sampler.next_chunk()
    try:
        model.compute_streaming_loss(chunk, {}, None, supervision="visit")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "visit" in str(exc)


def test_stay_supervision_remains_the_default_and_unchanged() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(_patients([_seq(1, 3)]), num_lanes=1, chunk_size=4)
    chunk = sampler.next_chunk()
    labels = _labels([1])
    total, components, _ = model.compute_streaming_loss(chunk, labels)
    assert torch.isfinite(total)
    assert components["concept_loss"] > 0


def test_concept_pos_weight_scales_positive_term() -> None:

    logits = torch.zeros(2, NUM_CONCEPTS)
    labels = torch.ones(2, NUM_CONCEPTS)  # all positive
    base = concept_loss(logits, labels)
    weighted = concept_loss(logits, labels, pos_weight=torch.full((NUM_CONCEPTS,), 2.0))
    assert torch.isclose(weighted, base * 2.0)

    # negatives are unaffected by pos_weight
    neg = torch.zeros(2, NUM_CONCEPTS)
    assert torch.isclose(
        concept_loss(logits, neg, pos_weight=torch.full((NUM_CONCEPTS,), 2.0)),
        concept_loss(logits, neg),
    )


# ---------------------------------------------------------------------------
# Intervention-aware training (RandInt)
# ---------------------------------------------------------------------------


def _first_times(subject_ids: List[int]) -> Dict[int, torch.Tensor]:
    # every concept triggers at hour 0: running label == stay label
    return {sid: torch.zeros(NUM_CONCEPTS) for sid in subject_ids}


def test_randint_intervention_respects_prob_and_observed_mask() -> None:
    torch.manual_seed(0)
    labels = _labels([1, 2])
    masks = {1: torch.ones(NUM_CONCEPTS), 2: torch.zeros(NUM_CONCEPTS)}
    first = _first_times([1, 2])

    def _build(chunk, prob):
        return randint_intervention(
            chunk,
            labels,
            masks,
            first,
            supervision="stay",
            num_concepts=NUM_CONCEPTS,
            prob=prob,
        )

    # A mid-chunk patient boundary truncates the window, so each patient
    # arrives in its own chunk: chunk 1 is subject 1, chunk 2 subject 2.
    sampler = PackedLaneSampler(
        _patients([_seq(1, 4), _seq(2, 4)]), num_lanes=1, chunk_size=8
    )
    chunk_1 = sampler.next_chunk()
    chunk_2 = sampler.next_chunk()
    assert (chunk_1.subject_ids[0] == 1).any()
    assert (chunk_2.subject_ids[0] == 2).any()

    assert _build(chunk_1, 0.0) is None

    always_1 = _build(chunk_1, 1.0)
    assert always_1 is not None and always_1.probs_mask is not None
    is_1 = chunk_1.subject_ids[0] == 1
    # subject 1 observed: substituted at every real position with prob 1
    assert always_1.probs_mask[0][is_1].all()
    assert torch.equal(always_1.probs[0][is_1][0], labels[1])
    # padding positions (no label entry) are never substituted
    assert not always_1.probs_mask[0][~is_1].any()

    always_2 = _build(chunk_2, 1.0)
    assert always_2 is not None and always_2.probs_mask is not None
    # subject 2 is unobserved everywhere: never substituted
    assert not always_2.probs_mask.any()


def test_randint_changes_task_logits_but_not_concept_readouts() -> None:
    model = _make_model()
    model.eval()
    sampler = PackedLaneSampler(_patients([_seq(1, 8)]), num_lanes=1, chunk_size=8)
    chunk = sampler.next_chunk()
    labels = _labels([1])
    masks = {1: torch.ones(NUM_CONCEPTS)}
    intervention = randint_intervention(
        chunk,
        labels,
        masks,
        _first_times([1]),
        supervision="stay",
        num_concepts=NUM_CONCEPTS,
        prob=1.0,
    )
    plain_logits, plain_out, _ = model(chunk.batch, reset_mask=chunk.reset_mask)
    iv_logits, iv_out, _ = model(
        chunk.batch, reset_mask=chunk.reset_mask, intervention=intervention
    )
    assert not torch.allclose(plain_logits, iv_logits)
    assert torch.equal(plain_out.concept_logits, iv_out.concept_logits)
    assert torch.equal(plain_out.observability_logits, iv_out.observability_logits)


def test_streaming_loss_accepts_randint_intervention_and_trains() -> None:
    torch.manual_seed(0)
    model = _make_model()
    labels = _labels([1, 2, 3])
    masks = {sid: torch.ones(NUM_CONCEPTS) for sid in labels}
    first = _first_times([1, 2, 3])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(6):
        sampler = PackedLaneSampler(
            _patients([_seq(1, 12), _seq(2, 12), _seq(3, 12)]),
            num_lanes=1,
            chunk_size=6,
        )
        state = None
        for chunk in sampler:
            intervention = randint_intervention(
                chunk,
                labels,
                masks,
                first,
                supervision="stay",
                num_concepts=NUM_CONCEPTS,
                prob=0.5,
            )
            total, components, state = model.compute_streaming_loss(
                chunk, labels, masks, state=state, intervention=intervention
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            state = _detach_state(state)
            losses.append(float(components["task_loss"]))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-3:]) < sum(losses[:3])
