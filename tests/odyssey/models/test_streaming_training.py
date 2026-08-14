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
from odyssey.data.streaming import PackedLaneSampler
from odyssey.models.backbones.base import TimeAwareState
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.concept_bottleneck import ConceptBottleneckLossWeights
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
)


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
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, padding_idx=PADDING_IDX
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
    return {
        sid: torch.randint(0, 2, (NUM_CONCEPTS,)).float() for sid in subject_ids
    }


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
    assert set(components) == {"task_loss", "concept_loss", "orthogonality_loss"}
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


def test_multiple_patient_ends_in_one_chunk_are_all_supervised() -> None:
    model = _make_model()
    sampler = PackedLaneSampler(
        _patients([_seq(1, 1), _seq(2, 1), _seq(3, 1)]), num_lanes=1, chunk_size=3
    )
    chunk = sampler.next_chunk()
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
        sampler = PackedLaneSampler(_patients(list(patients)), num_lanes=2, chunk_size=6, seed=0)
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
        sampler = PackedLaneSampler(_patients(list(patients)), num_lanes=2, chunk_size=6, seed=0)
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
