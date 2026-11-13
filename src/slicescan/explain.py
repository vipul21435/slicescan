"""A closer look at one slice: its 2x2 table, where it sits in the family, what it holds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .data import Dataset
from .slices import Slice, one_way_slices
from .table import SliceResult, SliceTable, score_slice, slice_table


def parse_label(label: str) -> list[tuple[str, str]]:
    """``"repo=x & size=[1, 5)"`` -> ``[("repo", "x"), ("size", "[1, 5)")]``."""
    parts = []
    for chunk in label.split(" & "):
        column, sep, value = chunk.partition("=")
        if not sep or not column.strip():
            raise ValueError(f"cannot read slice {label!r}; write COLUMN=VALUE[ & COLUMN=VALUE]")
        parts.append((column.strip(), value.strip()))
    return parts


@dataclass
class Breakdown:
    column: str
    rows: list[tuple[str, int, float]]  # value, items in the slice, score inside the slice


@dataclass
class Explanation:
    model: str
    label: str
    row: SliceResult
    tested: bool
    table: SliceTable
    rank: int | None
    failed: list[str]
    passed: int
    breakdowns: list[Breakdown] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "label": self.label,
            "row": self.row.to_dict(),
            "tested": self.tested,
            "rank_by_p": self.rank,
            "family": self.table.family.to_dict(),
            "failed_items": self.failed,
            "passed_items": self.passed,
            "breakdowns": [
                {"column": b.column, "rows": [list(r) for r in b.rows]} for b in self.breakdowns
            ],
        }


def find_slice(
    dataset: Dataset,
    items: Sequence[str],
    label: str,
    *,
    kinds: Mapping[str, str] | None = None,
    bins: int = 4,
    tag_sep: str | None = "|",
) -> Slice:
    parts = parse_label(label)
    columns = [c for c, _ in parts]
    cols = one_way_slices(dataset.meta, items, columns, kinds=kinds, bins=bins, tag_sep=tag_sep)
    found: list[Slice] = []
    for (column, value), col in zip(parts, cols, strict=True):
        match = [s for s in col.slices if s.values[0] == value]
        if not match:
            known = ", ".join(s.values[0] for s in col.slices[:12])
            raise KeyError(f"no slice {column}={value}; values of {column!r}: {known}")
        found.append(match[0])
    items_in = frozenset.intersection(*(s.items for s in found))
    return Slice(
        tuple(columns),
        tuple(v for _, v in parts),
        tuple(s.kinds[0] for s in found),
        items_in,
    )


def _find_row(table: SliceTable, slice_: Slice) -> SliceResult:
    """The tested row for this slice, whatever order its columns were written in."""
    wanted = dict(zip(slice_.columns, slice_.values, strict=True))
    for row in table.rows:
        if dict(zip(row.columns, row.values, strict=True)) == wanted:
            return row
    raise KeyError(f"no tested slice {slice_.label!r}")


def explain(
    dataset: Dataset,
    model: str,
    label: str,
    columns: Sequence[str] | None = None,
    *,
    kinds: Mapping[str, str] | None = None,
    bins: int = 4,
    tag_sep: str | None = "|",
    min_size: int = 10,
    confidence: float = 0.95,
    two_way: bool = False,
    two_way_cap: int | None = 100,
    resamples: int = 2000,
    seed: int = 0,
    max_items: int = 20,
) -> Explanation:
    """Explain one slice within the same family ``slice_table`` would build with these options."""
    table = slice_table(
        dataset,
        model,
        columns,
        kinds=kinds,
        bins=bins,
        tag_sep=tag_sep,
        min_size=min_size,
        confidence=confidence,
        two_way=two_way,
        two_way_cap=two_way_cap,
        resamples=resamples,
        seed=seed,
    )
    scores = dataset.model_scores(model)
    slice_ = find_slice(dataset, list(scores), label, kinds=kinds, bins=bins, tag_sep=tag_sep)
    tested = True
    try:
        row = _find_row(table, slice_)
        rank: int | None = table.rows.index(row) + 1
    except KeyError:
        tested = False
        rank = None
        row = score_slice(
            slice_, scores, confidence, binary=table.binary, resamples=resamples, seed=seed
        )
        row.p_holm = row.p_bh = math.nan
    members = sorted(i for i in slice_.items if i in scores)
    failed = [i for i in members if scores[i] < 0.5]
    passed = len(members) - len(failed)
    used = set(slice_.columns)
    other = [c for c in (columns or dataset.meta_columns()) if c not in used]
    breakdowns = []
    if other:
        member_set = set(members)
        for col in one_way_slices(
            dataset.meta, list(scores), other, kinds=kinds, bins=bins, tag_sep=tag_sep
        ):
            rows = []
            for s in col.slices:
                inside = [i for i in s.items if i in member_set]
                if inside:
                    mean = math.fsum(scores[i] for i in inside) / len(inside)
                    rows.append((s.values[0], len(inside), mean))
            rows.sort(key=lambda r: (-r[1], r[0]))
            breakdowns.append(Breakdown(col.column, rows))
    return Explanation(
        model=model,
        label=slice_.label,
        row=row,
        tested=tested,
        table=table,
        rank=rank,
        failed=failed[:max_items],
        passed=passed,
        breakdowns=breakdowns,
    )
