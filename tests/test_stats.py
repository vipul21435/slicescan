import random

import pytest
from scipy import stats as sps
from statsmodels.stats.proportion import proportion_confint

from slicescan.stats import binomial_test, fisher_exact, normal_quantile, wilson_interval


def test_normal_quantile_95():
    assert normal_quantile(0.95) == pytest.approx(1.959963984540054, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_normal_quantile_rejects_bad_confidence(bad):
    with pytest.raises(ValueError):
        normal_quantile(bad)


@pytest.mark.parametrize(
    "k,n,conf",
    [(0, 10, 0.95), (3, 10, 0.95), (10, 10, 0.95), (17, 40, 0.9), (250, 500, 0.99), (1, 1, 0.95)],
)
def test_wilson_matches_statsmodels(k, n, conf):
    low, high = wilson_interval(k, n, conf)
    ref_low, ref_high = proportion_confint(k, n, alpha=1 - conf, method="wilson")
    assert low == pytest.approx(ref_low, abs=1e-12)
    assert high == pytest.approx(ref_high, abs=1e-12)


def test_wilson_ends_are_exact_at_the_boundary():
    # These ends are assigned, not computed, so they hold on every platform.
    assert wilson_interval(0, 25)[0] == 0.0
    assert wilson_interval(25, 25)[1] == 1.0


@pytest.mark.parametrize("k,n", [(-1, 5), (6, 5), (0, 0)])
def test_wilson_rejects_bad_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_fisher_matches_scipy_on_random_tables():
    rng = random.Random(1)
    for _ in range(300):
        table = [rng.randint(0, 40) for _ in range(4)]
        a, b, c, d = table
        ours = fisher_exact(a, b, c, d)
        ref = sps.fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue
        assert ours == pytest.approx(ref, rel=1e-9, abs=1e-15), table


def test_fisher_matches_scipy_on_unbalanced_large_tables():
    for a, b, c, d in [(2, 38, 400, 60), (0, 30, 350, 120), (90, 10, 900, 1000)]:
        ref = sps.fisher_exact([[a, b], [c, d]]).pvalue
        assert fisher_exact(a, b, c, d) == pytest.approx(ref, rel=1e-8, abs=1e-300)


def test_fisher_degenerate_margin_gives_one():
    assert fisher_exact(0, 0, 5, 7) == 1.0
    assert fisher_exact(3, 0, 4, 0) == 1.0


def test_fisher_rejects_negative():
    with pytest.raises(ValueError):
        fisher_exact(-1, 2, 3, 4)


def test_binomial_matches_scipy():
    rng = random.Random(2)
    for _ in range(200):
        n = rng.randint(0, 60)
        k = rng.randint(0, n)
        p = rng.choice([0.5, 0.3, 0.71])
        ours = binomial_test(k, n, p)
        ref = sps.binomtest(k, n, p).pvalue if n > 0 else 1.0
        assert ours == pytest.approx(ref, rel=1e-9, abs=1e-15), (k, n, p)


def test_binomial_rejects_bad_input():
    with pytest.raises(ValueError):
        binomial_test(4, 3)
    with pytest.raises(ValueError):
        binomial_test(1, 3, 1.0)
