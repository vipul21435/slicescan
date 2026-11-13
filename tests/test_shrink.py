import math

import numpy as np
import pytest
from scipy import stats as sps

from slicescan.shrink import fit_beta_prior, shrink_ranking, spearman


def test_hand_worked_prior():
    # Four slices of 20 with 2, 8, 12, 18 successes. mu = 40/80 = 0.5.
    # S = 20 * (0.16 + 0.01 + 0.01 + 0.16) = 6.8, S / (mu (1 - mu)) = 27.2.
    # Denominator: 80 - 4 + 1 - 1600 / 80 = 57.  rho = (27.2 - 3) / 57 = 24.2 / 57.
    prior = fit_beta_prior([2, 8, 12, 18], [20, 20, 20, 20])
    assert prior.mu == pytest.approx(0.5)
    assert prior.rho == pytest.approx(24.2 / 57)
    assert prior.strength == pytest.approx(57 / 24.2 - 1)
    a = prior.strength / 2
    assert prior.posterior_mean(2, 20) == pytest.approx((2 + a) / (20 + 2 * a))
    assert prior.posterior_mean(2, 20) == pytest.approx(0.125386, abs=1e-6)
    assert prior.posterior_mean(18, 20) == pytest.approx(1 - 0.125386, abs=1e-6)


def test_no_extra_spread_shrinks_to_the_mean():
    prior = fit_beta_prior([10, 10, 10], [20, 20, 20])
    assert math.isinf(prior.strength)
    assert prior.posterior_mean(3, 4) == pytest.approx(0.5)
    assert prior.posterior_sd(3, 4) == 0.0
    assert math.isinf(fit_beta_prior([3], [9]).strength)
    assert math.isinf(fit_beta_prior([0, 0], [5, 9]).strength)


def test_small_slices_move_more():
    labels = ["big", "small"]
    prior, rows = shrink_ranking(labels, [90, 1], [100, 2])
    by = {r.label: r for r in rows}
    assert abs(by["small"].shrunk - by["small"].raw) > abs(by["big"].shrunk - by["big"].raw)
    for r in rows:
        assert min(r.raw, prior.mu) - 1e-12 <= r.shrunk <= max(r.raw, prior.mu) + 1e-12


def test_posterior_sd_matches_scipy_beta():
    prior = fit_beta_prior([2, 8, 12, 18], [20, 20, 20, 20])
    sd = prior.posterior_sd(8, 20)
    assert sd == pytest.approx(sps.beta(8 + prior.a, 12 + prior.b).std(), rel=1e-9)


def test_moment_estimate_recovers_simulated_prior():
    rng = np.random.default_rng(12)
    a, b = 6.0, 4.0  # mu 0.6, rho 1/11
    estimates = []
    for _ in range(200):
        n = rng.integers(10, 80, size=40)
        p = rng.beta(a, b, size=40)
        k = rng.binomial(n, p)
        estimates.append(fit_beta_prior([int(x) for x in k], [int(x) for x in n]).rho)
    assert np.mean(estimates) == pytest.approx(1 / 11, abs=0.02)


def test_shrunk_ranking_reorders_tiny_extremes():
    labels = ["a", "b", "c", "d", "e"]
    _, rows = shrink_ranking(labels, [3, 65, 60, 55, 50], [3, 100, 100, 100, 100])
    by = {r.label: r for r in rows}
    assert by["a"].raw_rank == 1
    assert by["a"].shrunk_rank > 1


def test_spearman_against_scipy():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 1, 4, 3, 5]
    assert spearman(xs, ys) == pytest.approx(sps.spearmanr(xs, ys).statistic)
    tied = [0.75, 0.75, 0.5, 0.2, 0.2, 0.1]
    other = [0.6, 0.7, 0.5, 0.45, 0.3, 0.4]
    assert spearman(tied, other) == pytest.approx(sps.spearmanr(tied, other).statistic)
    with pytest.raises(ValueError):
        spearman([1], [1])
    with pytest.raises(ValueError):
        spearman([1, 1], [1, 2])


def test_tied_raw_rates_share_an_average_rank():
    _, rows = shrink_ranking(["p", "q", "r"], [6, 24, 1], [8, 32, 4])
    by = {r.label: r for r in rows}
    assert by["p"].raw_rank == pytest.approx(1.5)
    assert by["q"].raw_rank == pytest.approx(1.5)
    assert by["q"].shrunk_rank == pytest.approx(1.0)


def test_bad_input():
    with pytest.raises(ValueError):
        fit_beta_prior([1], [1, 2])
    with pytest.raises(ValueError):
        fit_beta_prior([], [])
    with pytest.raises(ValueError):
        fit_beta_prior([3], [2])
