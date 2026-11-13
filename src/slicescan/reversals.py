"""Aggregation reversals: the overall winner loses inside most or all slices of a partition.

This is the pattern Simpson (1951) described and Bickel, Hammel and O'Connell (1975) found in
graduate admissions. Each system is scored on its own items here, so a full reversal can occur
when the systems were run on different items (different coverage of the slices). When both
systems cover exactly the same items, the overall difference is the size-weighted mean of the
slice differences, so the winner cannot lose in every slice; it can still lose in most of them,
by winning big where the slices are large.

Only columns that partition the items are checked (categorical and numeric). Tag columns are
skipped: an item in several slices breaks the weighted-mean identity the check rests on. Items
with no value in a column form their own slice, "(missing)", so the slices cover everything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .data import Dataset
from .slices import slice_column

MISSING = "(missing)"


@dataclass
class PartSlice:
    label: str
    n_a: int
    n_b: int
    score_a: float | None
    score_b: float | None
    winner: str  # "a", "b", "tie" or "" when one system has no items here


@dataclass
class PartitionCheck:
    column: str
    kind: str
    overall_a: float
    overall_b: float
    winner: str
    slices: list[PartSlice]
    wins: int
    losses: int
    ties: int
    lost_share: float
    verdict: str
    same_items: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReversalReport:
    model_a: str
    model_b: str
    n_a: int
    n_b: int
    shared: int
    checks: list[PartitionCheck]
    skipped: list[str]

    @property
    def reversals(self) -> list[PartitionCheck]:
        return [c for c in self.checks if c.verdict != "none"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "items_a": self.n_a,
            "items_b": self.n_b,
            "items_shared": self.shared,
            "checks": [c.to_dict() for c in self.checks],
            "skipped_columns": self.skipped,
        }


def _mean(scores: Mapping[str, float], items: Sequence[str]) -> float | None:
    if not items:
        return None
    return math.fsum(scores[i] for i in items) / len(items)


def _winner(x: float | None, y: float | None, tol: float = 1e-12) -> str:
    if x is None or y is None:
        return ""
    if abs(x - y) <= tol:
        return "tie"
    return "a" if x > y else "b"


def check_partition(
    dataset: Dataset,
    model_a: str,
    model_b: str,
    column: str,
    *,
    kind: str | None = None,
    bins: int = 4,
) -> PartitionCheck | None:
    """Check one column; ``None`` when the column is a tag column."""
    a = dataset.model_scores(model_a)
    b = dataset.model_scores(model_b)
    union = sorted(set(a) | set(b))
    col = slice_column(dataset.meta, union, column, kind=kind, bins=bins)
    if col.kind == "tag":
        return None
    groups: list[tuple[str, set[str]]] = [(s.label, set(s.items)) for s in col.slices]
    covered = set().union(*(g for _, g in groups)) if groups else set()
    missing = set(union) - covered
    if missing:
        groups.append((f"{column}={MISSING}", missing))
    overall_a = _mean(a, sorted(a))
    overall_b = _mean(b, sorted(b))
    if overall_a is None or overall_b is None:
        raise ValueError("both systems need scored items")
    winner = _winner(overall_a, overall_b)
    parts: list[PartSlice] = []
    for label, items in groups:
        ia = sorted(i for i in items if i in a)
        ib = sorted(i for i in items if i in b)
        sa, sb = _mean(a, ia), _mean(b, ib)
        parts.append(PartSlice(label, len(ia), len(ib), sa, sb, _winner(sa, sb)))
    decided = [p for p in parts if p.winner in ("a", "b", "tie")]
    wins = losses = ties = 0
    lost_items = 0
    total_items = sum(p.n_a + p.n_b for p in decided)
    if winner in ("a", "b"):
        for p in decided:
            if p.winner == "tie":
                ties += 1
            elif p.winner == winner:
                wins += 1
            else:
                losses += 1
                lost_items += p.n_a + p.n_b
    if winner == "tie" or losses == 0:
        verdict = "none"
    elif wins == 0 and losses >= 2:
        verdict = "full"
    elif losses > len(decided) / 2:
        verdict = "majority"
    else:
        verdict = "none"
    same = set(a) == set(b)
    notes = []
    if same:
        notes.append("same items for both systems: a full reversal is arithmetically impossible")
    return PartitionCheck(
        column=column,
        kind=col.kind,
        overall_a=overall_a,
        overall_b=overall_b,
        winner=winner,
        slices=parts,
        wins=wins,
        losses=losses,
        ties=ties,
        lost_share=lost_items / total_items if total_items else 0.0,
        verdict=verdict,
        same_items=same,
        notes=notes,
    )


def reversals(
    dataset: Dataset,
    model_a: str,
    model_b: str,
    columns: Sequence[str] | None = None,
    *,
    kinds: Mapping[str, str] | None = None,
    bins: int = 4,
) -> ReversalReport:
    """Check each partitioning column for an aggregation reversal between two systems."""
    if model_a == model_b:
        raise ValueError("pick two different systems")
    a = dataset.model_scores(model_a)
    b = dataset.model_scores(model_b)
    if columns is None:
        columns = dataset.meta_columns()
    kinds = dict(kinds or {})
    checks, skipped = [], []
    for column in columns:
        check = check_partition(
            dataset, model_a, model_b, column, kind=kinds.get(column), bins=bins
        )
        if check is None:
            skipped.append(column)
        else:
            checks.append(check)
    return ReversalReport(
        model_a, model_b, len(a), len(b), len(set(a) & set(b)), checks, skipped
    )
