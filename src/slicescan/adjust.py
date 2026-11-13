"""Adjusted p-values for a family of tests.

Holm (1979) controls the family-wise error rate under any dependence between the tests.
Benjamini and Hochberg (1995) control the false discovery rate for independent tests and for
positively dependent ones; slices of the same data overlap, so their tests are dependent.
"""

from __future__ import annotations

from collections.abc import Sequence


def _check(pvalues: Sequence[float]) -> list[float]:
    values = [float(p) for p in pvalues]
    for p in values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must lie in [0, 1], got {p}")
    return values


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    values = _check(pvalues)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[i]))
        adjusted[i] = running
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), in the input order."""
    values = _check(pvalues)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, min(1.0, m / rank * values[i]))
        adjusted[i] = running
    return adjusted
