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
