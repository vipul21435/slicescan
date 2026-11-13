"""Intervals and exact tests for proportions, written with the standard library only.

The exact tests follow the usual "sum of probabilities no larger than the observed one" rule for
two-sided p-values, with the same small relative tolerance scipy uses so that ties in probability
caused by rounding are counted together.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# Relative tolerance when comparing a table's probability with the observed one.
_REL_TOL = 1 + 1e-7


def normal_quantile(confidence: float) -> float:
    """Two-sided critical value z for a central interval of the given confidence."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (Wilson 1927).

    The lower end is exactly 0 when there are no successes and the upper end exactly 1 when every
    trial succeeded, instead of whatever rounding leaves there.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = normal_quantile(confidence)
    z2 = z * z
    p = successes / n
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_sum_exp(values: list[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p-value of Fisher's exact test for the 2x2 table [[a, b], [c, d]].

    With margins fixed, ``a`` follows a hypergeometric distribution. The p-value is the total
    probability of all tables whose probability does not exceed that of the observed table.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("table entries must be non-negative")
    row1 = a + b
    col1 = a + c
    total = a + b + c + d
    low = max(0, row1 - (total - col1))
    high = min(row1, col1)
    if low == high:
        return 1.0

    def logpmf(x: int) -> float:
        return _log_comb(col1, x) + _log_comb(total - col1, row1 - x) - _log_comb(total, row1)

    observed = logpmf(a) + math.log(_REL_TOL)
    kept = [lp for lp in (logpmf(x) for x in range(low, high + 1)) if lp <= observed]
    return min(1.0, math.exp(_log_sum_exp(kept)))


def binomial_test(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial test of ``k`` successes in ``n`` trials against ``p``."""
    if n < 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if n == 0:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)

    def logpmf(x: int) -> float:
        return _log_comb(n, x) + x * log_p + (n - x) * log_q

    observed = logpmf(k) + math.log(_REL_TOL)
    kept = [lp for lp in (logpmf(x) for x in range(n + 1)) if lp <= observed]
    return min(1.0, math.exp(_log_sum_exp(kept)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (conditional) McNemar test on the discordant counts of a paired 2x2 table.

    ``b`` counts items the first system got right and the second wrong, ``c`` the reverse.
    Under the null each discordant item is equally likely to fall either way (McNemar 1947), so
    the p-value is a two-sided binomial test of ``b`` out of ``b + c`` against one half.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    return binomial_test(b, b + c, 0.5)


def paired_diff_interval(
    b: int, c: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Interval for the difference of two matched proportions, ``(b - c) / n``.

    Agresti and Min (2005): add one half to each of the four cells of the paired table, then
    use the Wald interval. It keeps close to nominal coverage for small ``n`` where the plain
    Wald interval does not.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if b < 0 or c < 0 or b + c > n:
        raise ValueError("need 0 <= b, c and b + c <= n")
    z = normal_quantile(confidence)
    b_, c_, n_ = b + 0.5, c + 0.5, n + 2.0
    diff = (b_ - c_) / n_
    var = ((b_ + c_) / n_ - diff * diff) / n_
    half = z * math.sqrt(max(var, 0.0))
    return max(-1.0, diff - half), min(1.0, diff + half)


def mean_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation interval for a mean, from the sample standard deviation."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = math.fsum(values) / n
    if n == 1:
        return mean, mean
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    half = normal_quantile(confidence) * math.sqrt(var / n)
    return mean - half, mean + half


def permutation_test(
    slice_values: list[float],
    rest_values: list[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> float:
    """Two-sided permutation test for a difference in means between a slice and the rest.

    Random relabellings keep the slice size fixed. The p-value counts the observed split as one
    of the relabellings, ``(1 + hits) / (1 + resamples)``, so it is never zero.
    """
    import random

    n, m = len(slice_values), len(rest_values)
    if n == 0 or m == 0:
        raise ValueError("both groups need at least one value")
    pooled = list(slice_values) + list(rest_values)
    total = math.fsum(pooled)
    observed = abs(math.fsum(slice_values) / n - (total - math.fsum(slice_values)) / m)
    rng = random.Random(seed)
    hits = 0
    tol = 1e-12 * max(1.0, observed)
    for _ in range(resamples):
        s = math.fsum(rng.sample(pooled, n))
        if abs(s / n - (total - s) / m) >= observed - tol:
            hits += 1
    return (1 + hits) / (1 + resamples)


def bootstrap_mean_interval(
    values: list[float],
    confidence: float = 0.95,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean (Efron 1979), seeded so runs repeat."""
    import random

    n = len(values)
    if n == 0:
        raise ValueError("no values")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    lo_i = min(resamples - 1, max(0, math.floor(tail * resamples)))
    hi_i = min(resamples - 1, max(0, math.ceil((1.0 - tail) * resamples) - 1))
    return means[lo_i], means[hi_i]
