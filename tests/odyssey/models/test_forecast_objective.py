"""Tests for the bundle-invariant forecasting objective and the time head."""

import math

import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.sequences import PatientSequence
from odyssey.data.streaming import PackedLaneSampler
from odyssey.models.backbones.tiny_gru import TinyGRUBackbone
from odyssey.models.sequence_model import (
    BaselineSequenceModel,
    ConceptBottleneckSequenceModel,
    ForecastObjective,
    _bundle_log_likelihood,
)
from odyssey.models.time_to_event import (
    DEFAULT_TIME_BIN_EDGES_HOURS,
    gap_to_bin,
    hazard_log_likelihood,
    probability_within,
    survival_curve,
)


VOCAB = 12


def _seq(subject_id: int, codes: list[int], times: list[float]) -> PatientSequence:
    n = len(codes)
    return PatientSequence(
        subject_id=subject_id,
        concept_ids=codes,
        type_ids=[1] * n,
        time_stamps=times,
        ages=[50.0] * n,
        visit_orders=[0] * n,
        visit_segments=[0] * n,
    )


def _chunk(seqs, chunk_size=16, num_lanes=1):
    sampler = PackedLaneSampler(
        iter(seqs), num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    return sampler.next_chunk()


def _slow_bundle_ll(logp, targets, times, subject_ids, real):
    """Compute the reference bundle log-likelihood with plain Python loops."""
    lanes, chunk = targets.shape
    out = torch.zeros(lanes, chunk)
    for lane in range(lanes):
        for i in range(chunk):
            if not real[lane, i]:
                continue
            if i == chunk - 1:
                out[lane, i] = logp[lane, i, targets[lane, i]]
                continue
            key = (subject_ids[lane, i + 1].item(), times[lane, i + 1].item())
            seen = set()
            total = 0.0
            for j in range(i, chunk - 1):
                if not real[lane, j]:
                    continue
                if (subject_ids[lane, j + 1].item(), times[lane, j + 1].item()) != key:
                    continue
                tok = targets[lane, j].item()
                if tok in seen:
                    continue
                seen.add(tok)
                total += math.exp(logp[lane, i, tok].item())
            out[lane, i] = math.log(total)
    return out


def test_bundle_ll_matches_slow_reference_with_duplicates_and_two_lanes() -> None:
    torch.manual_seed(0)
    # subject 1: bundle at t=1 (codes 3,4,3 duplicate), t=2 (5), t=3 (6,7)
    s1 = _seq(1, [2, 3, 4, 3, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    s2 = _seq(2, [8, 9, 9, 10], [0.0, 0.0, 0.0, 5.0])
    chunk = _chunk([s1, s2], chunk_size=8, num_lanes=2)
    logits = torch.randn(*chunk.targets.shape, VOCAB)
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    fast = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    slow = _slow_bundle_ll(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    assert torch.allclose(fast, slow, atol=1e-5), (fast, slow)
    # never exceeds log 1 (dedup keeps credited mass <= 1)
    assert (fast <= 1e-6).all()


def test_bundle_loss_equals_cross_entropy_on_singleton_bundles() -> None:
    torch.manual_seed(0)
    s = _seq(1, [2, 3, 4, 5, 6], [0.0, 1.0, 2.0, 3.0, 4.0])
    chunk = _chunk([s], chunk_size=8)
    logits = torch.randn(*chunk.targets.shape, VOCAB)
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    ll = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    ce = logp.gather(-1, chunk.targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(ll[real], ce[real], atol=1e-6)


def test_bundle_loss_lower_bounds_cross_entropy() -> None:
    torch.manual_seed(1)
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chunk = _chunk([s], chunk_size=8)
    logits = torch.randn(*chunk.targets.shape, VOCAB)
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    ll = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    ce = logp.gather(-1, chunk.targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    assert (ll[real] >= ce[real] - 1e-6).all()
    assert (ll[real] > ce[real] + 1e-6).any()


def _model(time: bool = False) -> ConceptBottleneckSequenceModel:
    torch.manual_seed(0)
    return ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS if time else None,
    )


def test_default_objective_reproduces_original_loss() -> None:
    model = _model()
    model.eval()  # concept dropout would otherwise differ between passes
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chunk = _chunk([s], chunk_size=8)
    labels = {1: torch.tensor([1.0, 0.0])}
    a, comp_a, _ = model.compute_streaming_loss(chunk, labels)
    b, comp_b, _ = model.compute_streaming_loss(
        chunk, labels, objective=ForecastObjective()
    )
    assert torch.allclose(a, b)
    assert comp_a["time_loss"].item() == 0.0  # no time head


def test_family_weights_reweight_and_are_normalized() -> None:
    model = _model()
    model.eval()
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    chunk = _chunk([s], chunk_size=8)
    labels = {1: torch.tensor([1.0, 0.0])}
    token_types = torch.zeros(VOCAB, dtype=torch.long)
    token_types[[3, 5]] = 1  # family 1
    uniform = ForecastObjective(
        family_weights=torch.tensor([1.0, 1.0]), token_types=token_types
    )
    boosted = ForecastObjective(
        family_weights=torch.tensor([1.0, 5.0]), token_types=token_types
    )
    plain, _, _ = model.compute_streaming_loss(chunk, labels)
    u, _, _ = model.compute_streaming_loss(chunk, labels, objective=uniform)
    b, cb, _ = model.compute_streaming_loss(chunk, labels, objective=boosted)
    assert torch.allclose(plain, u, atol=1e-6)
    assert not torch.allclose(plain, b)
    assert torch.isfinite(cb["task_loss"])


def test_time_head_trains_and_survival_is_well_formed() -> None:
    model = _model(time=True)
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 0.0, 0.5, 6.0, 6.0, 30.0])
    chunk = _chunk([s], chunk_size=8)
    labels = {1: torch.tensor([1.0, 0.0])}
    obj = ForecastObjective(time_weight=1.0)
    total, comp, _ = model.compute_streaming_loss(chunk, labels, objective=obj)
    assert comp["time_loss"].item() > 0.0
    total.backward()
    assert model.time_head is not None
    assert model.time_head.proj.weight.grad is not None
    logits = torch.randn(4, model.time_head.num_bins)
    surv = survival_curve(logits)
    assert (surv[:, 1:] <= surv[:, :-1] + 1e-6).all()
    p_bins = torch.stack(
        [
            hazard_log_likelihood(logits, torch.full((4,), b)).exp()
            for b in range(model.time_head.num_bins)
        ],
        -1,
    )
    assert torch.allclose(p_bins.sum(-1) + surv[:, -1], torch.ones(4), atol=1e-5)
    assert (probability_within(logits, DEFAULT_TIME_BIN_EDGES_HOURS, 24.0) <= 1.0).all()


def test_gap_to_bin_edges() -> None:
    edges = DEFAULT_TIME_BIN_EDGES_HOURS
    g = torch.tensor([0.0, 1 / 60, 0.9 / 60, 24.0, 24.01, 1e6])
    assert gap_to_bin(g, edges).tolist() == [0, 1, 1, 10, 11, len(edges) + 1]


def test_baseline_model_shares_objective_and_time_head() -> None:
    torch.manual_seed(0)
    model = BaselineSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
    )
    s = _seq(1, [2, 3, 4, 5, 6, 7], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chunk = _chunk([s], chunk_size=8)
    total, comp, _ = model.compute_streaming_loss(
        chunk, objective=ForecastObjective(bundle_invariant=True, time_weight=0.5)
    )
    assert set(comp) == {"task_loss", "time_loss", "event_loss", "value_loss"}
    total.backward()


# ---------------------------------------------------------------------------
# Per-event hazard heads
# ---------------------------------------------------------------------------


def test_censored_likelihood_is_survival_through_earlier_bins() -> None:
    from odyssey.models.time_to_event import (  # noqa: PLC0415
        censored_hazard_log_likelihood,
    )

    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    target = torch.tensor([0, 2, 5])
    observed = torch.tensor([True, False, False])
    ll = censored_hazard_log_likelihood(logits, target, observed)
    full = hazard_log_likelihood(logits, target)
    log_1mh = F.logsigmoid(-logits)
    assert torch.allclose(ll[0], full[0])
    # censored in bin 2: survived bins 0 and 1
    assert torch.allclose(ll[1], log_1mh[1, :2].sum())
    assert torch.allclose(ll[2], log_1mh[2, :5].sum())
    # a censored contribution never exceeds the observed one
    assert (ll[1:] >= full[1:] - 1e-6).all()


def test_event_hazard_targets_from_tables() -> None:
    from odyssey.data.alert_events import EventTimes  # noqa: PLC0415
    from odyssey.training.event_targets import (  # noqa: PLC0415
        EventTimeTables,
        event_hazard_targets,
    )

    seq = PatientSequence(
        subject_id=1,
        concept_ids=[2, 3, 4, 5],
        type_ids=[1] * 4,
        time_stamps=[0.0, 5.0, 10.0, 20.0],
        ages=[50.0] * 4,
        visit_orders=[0] * 4,
        visit_segments=[0] * 4,
        visit_ids=[7, 7, 7, 7],
        visit_ends=[False, False, False, True],
    )
    chunk = _chunk([seq], chunk_size=8)
    tables = EventTimeTables(
        {
            "vaso": EventTimes(
                onset={(1, 7): 12.0}, censor={(1, 7): 20.0}, subject_scoped=False
            ),
            "death": EventTimes(onset={}, censor={(1, -1): 20.0}, subject_scoped=True),
        },
        ["vaso", "death"],
    )
    tg = event_hazard_targets(chunk, tables)
    real = chunk.subject_ids[0] == 1
    # vaso: onset 12 -> observed with gaps 12, 7, 2 at t=0,5,10; at t=20 not at risk
    assert tg.observed[0, real, 0].tolist() == [True, True, True, False]
    assert tg.at_risk[0, real, 0].tolist() == [True, True, True, False]
    assert tg.gap_hours[0, real, 0].tolist()[:3] == [12.0, 7.0, 2.0]
    # death: never; censored at 20 -> gaps 20, 15, 10; at t=20 (c == t) not at risk
    assert tg.observed[0, real, 1].tolist() == [False] * 4
    assert tg.at_risk[0, real, 1].tolist() == [True, True, True, False]
    assert tg.gap_hours[0, real, 1].tolist()[:3] == [20.0, 15.0, 10.0]
    # padding positions are never at risk
    assert not tg.at_risk[0, ~real].any()


def test_event_heads_train_and_survival_within_is_a_probability() -> None:
    from odyssey.data.alert_events import EventTimes  # noqa: PLC0415
    from odyssey.training.event_targets import (  # noqa: PLC0415
        EventTimeTables,
        event_hazard_targets,
    )

    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=["vaso", "death"],
    )
    seq = PatientSequence(
        subject_id=1,
        concept_ids=[2, 3, 4, 5, 6],
        type_ids=[1] * 5,
        time_stamps=[0.0, 5.0, 10.0, 20.0, 30.0],
        ages=[50.0] * 5,
        visit_orders=[0] * 5,
        visit_segments=[0] * 5,
        visit_ids=[7] * 5,
        visit_ends=[False] * 4 + [True],
    )
    chunk = _chunk([seq], chunk_size=8)
    tables = EventTimeTables(
        {
            "vaso": EventTimes(
                onset={(1, 7): 12.0}, censor={(1, 7): 30.0}, subject_scoped=False
            ),
            "death": EventTimes(onset={}, censor={(1, -1): 30.0}, subject_scoped=True),
        },
        ["vaso", "death"],
    )
    targets = event_hazard_targets(chunk, tables)
    labels = {1: torch.tensor([1.0, 0.0])}
    obj = ForecastObjective(event_hazard_weight=1.0)
    total, comp, _ = model.compute_streaming_loss(
        chunk, labels, objective=obj, event_targets=targets
    )
    assert comp["event_loss"].item() > 0.0
    total.backward()
    assert model.event_heads is not None
    assert model.event_heads.proj.weight.grad is not None
    hz = model.event_heads(torch.randn(3, model.event_heads.proj.in_features))
    assert hz.shape == (3, 2, model.event_heads.num_bins)
    p8 = probability_within(hz[:, 0], DEFAULT_TIME_BIN_EDGES_HOURS, 8.0)
    assert ((p8 >= 0) & (p8 <= 1)).all()


def test_task_weight_zero_gates_time_and_event_loss_too_not_just_next_token() -> None:
    """task_weight=0 must zero ALL of next-token + time + event gradient.

    ConceptBottleneckSequenceModel.compute_streaming_loss bundles
    next_token_loss + time_weight*time_loss + event_hazard_weight*event_loss
    into one forecast_loss *before* handing it to combined_loss as its
    single task_loss argument, so ConceptBottleneckLossWeights.task=0.0
    scales that whole bundle at once, not just the next-token term. This
    pins that down directly (not just by reading the code): with a time
    head and event heads both active and weighted 1.0, after
    task_weight=0.0's backward pass, their own parameters -- reachable
    only through time_loss/event_loss -- must show exactly zero
    gradient, even though the logged (unweighted, for-visibility)
    time_loss/event_loss values are themselves nonzero and would change
    step to step as concept/orthogonality training moves the shared
    bottleneck output under them.
    """
    from odyssey.data.alert_events import EventTimes  # noqa: PLC0415
    from odyssey.training.event_targets import (  # noqa: PLC0415
        EventTimeTables,
        event_hazard_targets,
    )

    torch.manual_seed(0)
    model = ConceptBottleneckSequenceModel(
        backbone=TinyGRUBackbone(
            vocab_size=VOCAB, hidden_size=8, num_layers=1, padding_idx=0
        ),
        vocab_size=VOCAB,
        num_concepts=2,
        embedding_dim=4,
        padding_idx=0,
        time_bin_edges=DEFAULT_TIME_BIN_EDGES_HOURS,
        event_names=["vaso", "death"],
    )
    seq = PatientSequence(
        subject_id=1,
        concept_ids=[2, 3, 4, 5, 6],
        type_ids=[1] * 5,
        time_stamps=[0.0, 5.0, 10.0, 20.0, 30.0],
        ages=[50.0] * 5,
        visit_orders=[0] * 5,
        visit_segments=[0] * 5,
        visit_ids=[7] * 5,
        visit_ends=[False] * 4 + [True],
    )
    chunk = _chunk([seq], chunk_size=8)
    tables = EventTimeTables(
        {
            "vaso": EventTimes(
                onset={(1, 7): 12.0}, censor={(1, 7): 30.0}, subject_scoped=False
            ),
            "death": EventTimes(onset={}, censor={(1, -1): 30.0}, subject_scoped=True),
        },
        ["vaso", "death"],
    )
    event_targets = event_hazard_targets(chunk, tables)
    labels = {1: torch.tensor([1.0, 0.0])}

    from odyssey.models.concept_bottleneck import (  # noqa: PLC0415
        ConceptBottleneckLossWeights,
    )

    total, comp, _ = model.compute_streaming_loss(
        chunk,
        labels,
        loss_weights=ConceptBottleneckLossWeights(task=0.0),
        objective=ForecastObjective(time_weight=1.0, event_hazard_weight=1.0),
        event_targets=event_targets,
    )
    # the logged values are the boring, expected source of odyssey-6e's
    # "moving every step" observation: nonzero and free to change as the
    # shared bottleneck output moves under concept/orthogonality training,
    # with no bearing on whether they contribute gradient (checked below).
    assert comp["time_loss"].item() > 0.0
    assert comp["event_loss"].item() > 0.0

    total.backward()

    assert model.time_head is not None
    assert model.event_heads is not None
    for name, p in list(model.time_head.named_parameters()) + list(
        model.event_heads.named_parameters()
    ):
        assert p.grad is None or torch.all(p.grad == 0), (
            f"{name} received nonzero gradient under task_weight=0.0"
        )
    for name, p in model.lm_head.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"lm_head.{name} received nonzero gradient under task_weight=0.0"
        )
    # contrast: the bottleneck itself DOES move (concept/orthogonality
    # supervision is unaffected by task_weight), confirming this isn't a
    # trivially-disconnected graph where nothing gets gradient at all.
    assert model.bottleneck.context_proj.weight.grad is not None
    assert torch.any(model.bottleneck.context_proj.weight.grad != 0)


def test_bundle_loss_credits_only_the_targets_own_family() -> None:
    """A discharge bundle mixes diagnoses with the discharge/DRG tokens.

    At a diagnosis position the credited mass must be the remaining
    diagnoses only, never the co-timed discharge or billing tokens --
    otherwise the model learns to predict the always-present discharge
    token instead of the diagnoses (the v5 subset run did exactly this).
    """
    torch.manual_seed(0)
    # codes 2,3 = diagnoses; 8 = discharge (visit); 9 = DRG (billing), all at t=1
    s = _seq(1, [5, 2, 3, 8, 9], [0.0, 1.0, 1.0, 1.0, 1.0])
    chunk = _chunk([s], chunk_size=8)
    logits = torch.zeros(*chunk.targets.shape, VOCAB)
    # put all mass on the discharge token at every position
    logits[..., 8] = 20.0
    logp = F.log_softmax(logits, -1)
    real = chunk.real_mask & (chunk.targets != 0)
    types = torch.zeros(VOCAB, dtype=torch.long)
    types[[2, 3]] = 1  # diagnosis
    types[8] = 5  # visit
    types[9] = 7  # billing
    unrestricted = _bundle_log_likelihood(
        logp, chunk.targets, chunk.batch.aux.time_stamps, chunk.subject_ids, real
    )
    restricted = _bundle_log_likelihood(
        logp,
        chunk.targets,
        chunk.batch.aux.time_stamps,
        chunk.subject_ids,
        real,
        token_types=types,
    )
    # position 0 targets diagnosis 2 (bundle at t=1): unrestricted credits the
    # discharge token (near log 1); restricted credits diagnoses only (~-20)
    assert unrestricted[0, 0].item() > -0.01
    assert restricted[0, 0].item() < -15.0
    # position 2 targets the discharge token itself: fully credited either way
    assert restricted[0, 2].item() > -0.01
