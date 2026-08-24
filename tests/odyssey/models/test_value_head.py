"""The value-quantile head: pinball loss, monotonicity, masking, metrics."""

import torch

from odyssey.models.value_head import (
    DEFAULT_QUANTILE_LEVELS,
    ValueQuantileHead,
    crps_from_quantiles,
    median_absolute_error,
    pinball_loss,
    quantile_coverage,
    value_quantile_loss,
    value_target_valid_mask,
)


torch.manual_seed(0)


def test_pinball_loss_hand_computed() -> None:
    # One quantile level 0.1 (under-prediction favored), one 0.9
    # (over-prediction favored); target=2.0 in both cases, prediction=1.0.
    # tau=0.1, y=2.0, q=1.0, y>q: loss = tau*(y-q) = 0.1*1.0 = 0.1
    # tau=0.9, y=2.0, q=1.0, y>q: loss = tau*(y-q) = 0.9*1.0 = 0.9
    quantiles = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([2.0])
    loss = pinball_loss(quantiles, target, [0.1, 0.9])
    assert torch.allclose(loss, torch.tensor([(0.1 + 0.9) / 2]))

    # Same target/prediction but tau=0.1 with prediction ABOVE target:
    # y=1.0, q=2.0, y<q: loss = (1-tau)*(q-y) = 0.9*1.0 = 0.9
    quantiles2 = torch.tensor([[2.0]])
    target2 = torch.tensor([1.0])
    loss2 = pinball_loss(quantiles2, target2, [0.1])
    assert torch.allclose(loss2, torch.tensor([0.9]))


def test_head_output_is_monotone_for_random_inputs() -> None:
    head = ValueQuantileHead(16, 8, DEFAULT_QUANTILE_LEVELS)
    features = torch.randn(100, 16)
    target_embedding = torch.randn(100, 8)
    quantiles = head(features, target_embedding)
    assert quantiles.shape == (100, len(DEFAULT_QUANTILE_LEVELS))
    diffs = quantiles[:, 1:] - quantiles[:, :-1]
    assert bool((diffs >= 0).all())


def test_head_output_monotone_with_leading_batch_dims() -> None:
    head = ValueQuantileHead(4, 4, (0.1, 0.5, 0.9))
    features = torch.randn(3, 5, 4)
    target_embedding = torch.randn(3, 5, 4)
    quantiles = head(features, target_embedding)
    assert quantiles.shape == (3, 5, 3)
    diffs = quantiles[..., 1:] - quantiles[..., :-1]
    assert bool((diffs >= 0).all())


def test_value_target_valid_mask_shift_and_nan_handling() -> None:
    # (lanes=1, T=4); position i's target value is values[i+1].
    values = torch.tensor([[1.0, float("nan"), 3.0, 4.0]])
    real_mask = torch.tensor([[True, True, True, True]])
    target_value, valid = value_target_valid_mask(values, real_mask)
    assert target_value.shape == values.shape
    # target_value[0] = values[1] = nan; target_value[1] = values[2] = 3.0;
    # target_value[2] = values[3] = 4.0; target_value[3] = nan (last col, unset).
    assert torch.isnan(target_value[0, 0])
    assert target_value[0, 1] == 3.0
    assert target_value[0, 2] == 4.0
    assert torch.isnan(target_value[0, 3])
    assert valid.tolist() == [[False, True, True, False]]


def test_value_target_valid_mask_respects_real_mask() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]])
    real_mask = torch.tensor([[True, False, True]])
    _, valid = value_target_valid_mask(values, real_mask)
    # position 1's real_mask is False even though values[2]=3.0 is non-NaN.
    assert valid[0, 1].item() is False


def test_value_quantile_loss_masking_no_nan_and_zero_graph_when_no_valid() -> None:
    quantiles = torch.randn(2, 5, 9, requires_grad=True)
    target = torch.full((2, 5), float("nan"))
    target[:, 2] = 1.5  # exactly one valid position per lane
    valid = ~torch.isnan(target)
    loss = value_quantile_loss(quantiles, target, valid, DEFAULT_QUANTILE_LEVELS)
    assert torch.isfinite(loss)
    loss.backward()
    assert quantiles.grad is not None
    assert torch.isfinite(quantiles.grad).all()
    # Gradient at invalid (NaN-target) positions must be exactly zero --
    # the mask must not merely down-weight them, it must exclude them.
    grad_at_invalid = quantiles.grad[~valid]
    assert torch.all(grad_at_invalid == 0.0)

    # No valid positions anywhere: zero-graph fallback, not NaN.
    all_invalid = torch.zeros(2, 5, dtype=torch.bool)
    quantiles2 = torch.randn(2, 5, 9, requires_grad=True)
    target2 = torch.full((2, 5), float("nan"))
    loss2 = value_quantile_loss(
        quantiles2, target2, all_invalid, DEFAULT_QUANTILE_LEVELS
    )
    assert loss2.item() == 0.0
    loss2.backward()
    assert torch.all(quantiles2.grad == 0.0)


def test_masked_positions_get_no_gradient_from_random_nan_pattern() -> None:
    """Same check as above, generalized: any NaN-target position is grad-zero."""
    torch.manual_seed(1)
    quantiles = torch.randn(4, 20, 9, requires_grad=True)
    target = torch.randn(4, 20)
    drop = torch.rand(4, 20) < 0.4
    target = target.masked_fill(drop, float("nan"))
    valid = ~torch.isnan(target)
    loss = value_quantile_loss(quantiles, target, valid, DEFAULT_QUANTILE_LEVELS)
    loss.backward()
    assert torch.isfinite(quantiles.grad).all()
    assert torch.all(quantiles.grad[drop] == 0.0)
    assert torch.any(quantiles.grad[~drop] != 0.0)


def test_recovers_planted_conditional_normal_distribution() -> None:
    """Fit against value ~ N(mu(code), sigma) in isolation; check quantiles."""
    torch.manual_seed(2)
    code_means = {0: -2.0, 1: 3.0}
    sigma = 1.0
    n = 4000
    codes = torch.randint(0, 2, (n,))
    mu = torch.tensor([code_means[int(c)] for c in codes.tolist()])
    target = mu + sigma * torch.randn(n)
    # Trivial conditioning signal: a one-hot of the code as the "target
    # embedding", features are a constant bias term.
    target_embedding = torch.nn.functional.one_hot(codes, num_classes=2).float()
    features = torch.ones(n, 1)

    head = ValueQuantileHead(1, 2, DEFAULT_QUANTILE_LEVELS)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        quantiles = head(features, target_embedding)
        loss = pinball_loss(quantiles, target, DEFAULT_QUANTILE_LEVELS).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for code, true_mu in code_means.items():
            emb = torch.nn.functional.one_hot(
                torch.tensor([code]), num_classes=2
            ).float()
            q = head(torch.ones(1, 1), emb)[0]
            true_quantiles = torch.tensor(
                [
                    true_mu
                    + sigma * torch.distributions.Normal(0, 1).icdf(torch.tensor(lvl))
                    for lvl in DEFAULT_QUANTILE_LEVELS
                ]
            )
            assert torch.allclose(q, true_quantiles, atol=0.35)

    # Coverage near nominal on a fresh sample from the same distribution.
    codes_test = torch.randint(0, 2, (2000,))
    mu_test = torch.tensor([code_means[int(c)] for c in codes_test.tolist()])
    target_test = mu_test + sigma * torch.randn(2000)
    emb_test = torch.nn.functional.one_hot(codes_test, num_classes=2).float()
    with torch.no_grad():
        q_test = head(torch.ones(2000, 1), emb_test)
    coverage = quantile_coverage(q_test, target_test)
    for level, cov in zip(DEFAULT_QUANTILE_LEVELS, coverage.tolist()):
        assert abs(cov - level) < 0.06


def test_crps_and_coverage_correctness_on_known_distribution() -> None:
    """Quantiles set to the true distribution's: coverage must equal the levels."""
    torch.manual_seed(3)
    n = 5000
    target = torch.randn(n)  # standard normal
    normal = torch.distributions.Normal(0.0, 1.0)
    true_quantiles = torch.stack(
        [normal.icdf(torch.tensor(lvl)).expand(n) for lvl in DEFAULT_QUANTILE_LEVELS],
        dim=-1,
    )
    coverage = quantile_coverage(true_quantiles, target)
    for level, cov in zip(DEFAULT_QUANTILE_LEVELS, coverage.tolist()):
        assert abs(cov - level) < 0.03

    mae = median_absolute_error(true_quantiles, target, DEFAULT_QUANTILE_LEVELS)
    # median_absolute_error is MEAN(|median_pred - target|); the median
    # prediction here is exactly 0 (normal.icdf(0.5) == 0), so this equals
    # E[|X|] for X ~ N(0,1) = sqrt(2/pi) ~= 0.7979 (the half-normal mean,
    # not its median).
    assert abs(mae.item() - (2 / 3.14159265) ** 0.5) < 0.05

    crps = crps_from_quantiles(true_quantiles, target, DEFAULT_QUANTILE_LEVELS)
    assert torch.isfinite(crps).all()
    assert bool((crps >= 0).all())
    # CRPS should be non-trivially smaller for the true quantiles than for
    # a badly miscalibrated constant prediction far from the data.
    bad_quantiles = torch.full_like(true_quantiles, 10.0)
    crps_bad = crps_from_quantiles(bad_quantiles, target, DEFAULT_QUANTILE_LEVELS)
    assert crps.mean().item() < crps_bad.mean().item()


def test_value_head_hidden_gives_an_mlp_readout_and_stays_monotone() -> None:
    """hidden>0 swaps the single linear layer for an MLP, quantiles still sorted.

    Arm B (2026-08-24) ran the linear default and fit the value distribution
    poorly (mid-quantile coverage 0.243 vs a nominal 0.3), so the capacity
    knob exists to separate "distributional value prediction does not help"
    from "a linear head could not fit it".
    """
    from torch import nn  # noqa: PLC0415

    from odyssey.models.value_head import ValueQuantileHead  # noqa: PLC0415

    linear = ValueQuantileHead(8, 4)
    mlp = ValueQuantileHead(8, 4, hidden=32)
    assert isinstance(linear.proj, nn.Linear)
    assert isinstance(mlp.proj, nn.Sequential)
    assert sum(p.numel() for p in mlp.parameters()) > sum(
        p.numel() for p in linear.parameters()
    )
    q = mlp(torch.randn(5, 8), torch.randn(5, 4))
    assert q.shape == (5, len(mlp.quantile_levels))
    assert (q[:, 1:] >= q[:, :-1]).all()  # monotone by construction, MLP or not
