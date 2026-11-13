"""Empirical-Bayes shrinkage of many slice proportions toward their common mean.

Each slice ``i`` has ``k_i`` successes out of ``n_i``. Its true rate is taken to be drawn from a
Beta(a, b) prior shared by the slices of one column; the prior is estimated from the slices
themselves and each raw rate is replaced by its posterior mean

    (k_i + a) / (n_i + a + b)  =  w_i * k_i / n_i + (1 - w_i) * mu,   w_i = n_i / (n_i + s),

with ``mu = a / (a + b)`` and ``s = a + b``. Small slices (small ``n_i``) are pulled hardest. This
is the beta-binomial form of the idea Efron and Morris (1975) illustrate on batting averages.

The prior is fitted by the method of moments. ``mu`` is the pooled rate. The intra-class
correlation ``rho = 1 / (s + 1)`` comes from matching the weighted spread of the raw rates,
``S = sum_i n_i (p_i - mu)^2``, to its expectation under the model,

    E[S] = mu (1 - mu) [ (m - 1) + rho (N - m + 1 - sum_i n_i^2 / N) ],

where ``m`` is the number of slices and ``N = sum_i n_i``. When the raw rates spread no more
than binomial noise alone would make them (``rho <= 0``), every slice is shrunk to ``mu``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BetaPrior:
    """Fitted Beta prior: mean ``mu`` and prior sample size ``strength`` (a + b, may be inf)."""

    mu: float
    strength: float
    rho: float

    @property
    def a(self) -> float:
        return self.mu * self.strength

    @property
    def b(self) -> float:
        return (1.0 - self.mu) * self.strength

    def posterior_mean(self, k: int, n: int) -> float:
        if math.isinf(self.strength):
            return self.mu
        return (k + self.a) / (n + self.strength)

    def posterior_sd(self, k: int, n: int) -> float:
        if math.isinf(self.strength):
            return 0.0
        a, b = k + self.a, n - k + self.b
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))


def fit_beta_prior(successes: Sequence[int], sizes: Sequence[int]) -> BetaPrior:
    """Method-of-moments Beta prior from slice counts (see the module docstring)."""
    if len(successes) != len(sizes):
        raise ValueError("successes and sizes differ in length")
    if not sizes:
        raise ValueError("no slices")
    for k, n in zip(successes, sizes, strict=True):
        if n <= 0 or not 0 <= k <= n:
            raise ValueError(f"bad slice counts k={k}, n={n}")
    m = len(sizes)
    total = sum(sizes)
    mu = sum(successes) / total
    if m == 1 or mu in (0.0, 1.0):
        return BetaPrior(mu, math.inf, 0.0)
    spread = math.fsum(n * (k / n - mu) ** 2 for k, n in zip(successes, sizes, strict=True))
    denom = total - m + 1 - math.fsum(n * n for n in sizes) / total
    if denom <= 0:
        return BetaPrior(mu, math.inf, 0.0)
    rho = (spread / (mu * (1.0 - mu)) - (m - 1)) / denom
    if rho <= 0:
        return BetaPrior(mu, math.inf, 0.0)
    rho = min(rho, 1.0 - 1e-9)
    return BetaPrior(mu, 1.0 / rho - 1.0, rho)


@dataclass
class RankRow:
    """Raw and shrunk rate of one slice; ranks are 1 for the highest, tied values share the
    average of the ranks they span (two slices tied for second both get 2.5)."""

    label: str
    n: int
    successes: int
    raw: float
    shrunk: float
    raw_rank: float
    shrunk_rank: float


def _ranks(values: Sequence[float], tol: float = 1e-12) -> list[float]:
    """Average ranks, 1 = highest value."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and abs(values[order[end + 1]] - values[order[start]]) <= tol:
            end += 1
        average = (start + end) / 2.0 + 1.0
        for k in range(start, end + 1):
            ranks[order[k]] = average
        start = end + 1
    return ranks


def shrink_ranking(
    labels: Sequence[str], successes: Sequence[int], sizes: Sequence[int]
) -> tuple[BetaPrior, list[RankRow]]:
    """Raw and shrunk rates of a set of slices with both rankings, sorted by shrunk rank."""
    prior = fit_beta_prior(successes, sizes)
    raw = [k / n for k, n in zip(successes, sizes, strict=True)]
    shrunk = [prior.posterior_mean(k, n) for k, n in zip(successes, sizes, strict=True)]
    raw_rank = _ranks(raw)
    shrunk_rank = _ranks(shrunk)
    rows = [
        RankRow(labels[i], sizes[i], successes[i], raw[i], shrunk[i], raw_rank[i], shrunk_rank[i])
        for i in range(len(labels))
    ]
    rows.sort(key=lambda r: (r.shrunk_rank, r.raw_rank, r.label))
    return prior, rows


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman correlation: the Pearson correlation of the average ranks (ties allowed)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        raise ValueError("need two sequences of the same length, at least two values")
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = math.fsum(rx) / n, math.fsum(ry) / n
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = math.fsum((a - mx) ** 2 for a in rx)
    syy = math.fsum((b - my) ** 2 for b in ry)
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("a constant sequence has no rank correlation")
    return sxy / math.sqrt(sxx * syy)
