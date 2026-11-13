import random

import numpy as np
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


def test_mcnemar_matches_statsmodels_and_scipy():
    from statsmodels.stats.contingency_tables import mcnemar

    from slicescan.stats import mcnemar_exact

    rng = np.random.default_rng(21)
    for _ in range(40):
        b, c = (int(x) for x in rng.integers(0, 40, size=2))
        a, d = 10, 7
        expected = mcnemar([[a, b], [c, d]], exact=True).pvalue
        assert mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-9, abs=1e-15)
        if b + c:
            assert mcnemar_exact(b, c) == pytest.approx(
                sps.binomtest(b, b + c, 0.5).pvalue, rel=1e-9
            )
    assert mcnemar_exact(0, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_paired_interval_hand_worked():
    from slicescan.stats import paired_diff_interval

    # b=12, c=4, n=50: cells +0.5 give diff 8/52, var = (17/52 - (8/52)^2) / 52.
    low, high = paired_diff_interval(12, 4, 50)
    diff = 8 / 52
    half = 1.959963984540054 * ((17 / 52 - diff**2) / 52) ** 0.5
    assert low == pytest.approx(diff - half, rel=1e-9)
    assert high == pytest.approx(diff + half, rel=1e-9)
    assert paired_diff_interval(0, 0, 3)[0] < 0 < paired_diff_interval(0, 0, 3)[1]
    with pytest.raises(ValueError):
        paired_diff_interval(5, 6, 10)


def test_paired_interval_coverage_by_simulation():
    from slicescan.stats import paired_diff_interval

    rng = np.random.default_rng(5)
    probs = [0.55, 0.15, 0.05, 0.25]  # both right, first only, second only, both wrong
    true = probs[1] - probs[2]
    n, reps, cover = 60, 2000, 0
    for _ in range(reps):
        counts = rng.multinomial(n, probs)
        low, high = paired_diff_interval(int(counts[1]), int(counts[2]), n)
        cover += low <= true <= high
    assert 0.93 <= cover / reps <= 0.975


def test_mean_interval_against_normal_formula():
    from slicescan.stats import mean_interval

    values = [0.2, 0.4, 0.4, 0.9, 1.0]
    low, high = mean_interval(values)
    half = 1.959963984540054 * np.std(values, ddof=1) / np.sqrt(5)
    assert low == pytest.approx(np.mean(values) - half)
    assert high == pytest.approx(np.mean(values) + half)
    assert mean_interval([0.3]) == pytest.approx((0.3, 0.3))
    with pytest.raises(ValueError):
        mean_interval([])


def test_permutation_test_against_scipy():
    from slicescan.stats import permutation_test

    rng = np.random.default_rng(8)
    x = list(rng.normal(0.4, 0.2, size=15))
    y = list(rng.normal(0.6, 0.2, size=40))
    ours = permutation_test(x, y, resamples=4000, seed=1)
    ref = sps.permutation_test(
        (x, y), lambda a, b: np.mean(a) - np.mean(b), n_resamples=4000, random_state=2
    ).pvalue
    assert ours == pytest.approx(ref, abs=0.01)
    same = permutation_test([0.5] * 5, [0.5] * 5, resamples=50)
    assert same == 1.0
    with pytest.raises(ValueError):
        permutation_test([], [1.0])


def test_bootstrap_interval_close_to_scipy():
    from slicescan.stats import bootstrap_mean_interval

    rng = np.random.default_rng(3)
    values = list(rng.uniform(0, 1, size=60))
    low, high = bootstrap_mean_interval(values, resamples=4000, seed=4)
    ref = sps.bootstrap(
        (values,), np.mean, n_resamples=4000, method="percentile", random_state=9
    ).confidence_interval
    assert low == pytest.approx(ref.low, abs=0.01)
    assert high == pytest.approx(ref.high, abs=0.01)
    assert low < float(np.mean(values)) < high
    with pytest.raises(ValueError):
        bootstrap_mean_interval([])
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1.0], resamples=0)


def test_sign_flip_test_against_scipy():
    from slicescan.stats import sign_flip_test

    rng = np.random.default_rng(14)
    d = list(rng.normal(0.08, 0.3, size=40))
    ours = sign_flip_test(d, resamples=4000, seed=2)
    ref = sps.permutation_test(
        (d,), np.mean, permutation_type="samples", n_resamples=4000, random_state=3
    ).pvalue
    assert ours == pytest.approx(ref, abs=0.015)
    assert sign_flip_test([0.0, 0.0]) == 1.0
    with pytest.raises(ValueError):
        sign_flip_test([])
