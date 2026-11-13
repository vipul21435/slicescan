"""Turn item metadata columns into slices: sets of items that share a value.

Three kinds of column are handled:

* ``category``: each distinct value is a slice; every item is in at most one slice.
* ``numeric``: values are cut into quantile bins; every item is in at most one bin.
* ``tag``: a cell holds several values (a JSON list, or a string split on a separator); an item is
  in one slice per tag it carries, so the slices of a tag column overlap.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .data import is_missing

KINDS = ("category", "numeric", "tag")


@dataclass(frozen=True)
class Slice:
    """A set of items defined by one value per column (one column for a one-way slice)."""

    columns: tuple[str, ...]
    values: tuple[str, ...]
    kinds: tuple[str, ...]
    items: frozenset[str]

    @property
    def label(self) -> str:
        return " & ".join(f"{c}={v}" for c, v in zip(self.columns, self.values, strict=True))

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass
class ColumnSlices:
    """The slices made from one column, with how the column was read."""

    column: str
    kind: str
    slices: list[Slice]
    missing: int
    edges: list[float] | None = None
    notes: list[str] = field(default_factory=list)


def format_number(x: float) -> str:
    """Short, stable text for a number: integers without a decimal point."""
    if math.isfinite(x) and x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.4g}"


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Quantile by linear interpolation between order statistics (numpy's default rule)."""
    if not sorted_values:
        raise ValueError("no values")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    h = (len(sorted_values) - 1) * q
    lo = math.floor(h)
    frac = h - lo
    if frac == 0.0 or lo + 1 >= len(sorted_values):
        return sorted_values[lo]
    return sorted_values[lo] + frac * (sorted_values[lo + 1] - sorted_values[lo])


def quantile_edges(values: Iterable[float], bins: int) -> list[float]:
    """Edges of ``bins`` quantile bins; repeated edges (from ties) are merged."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    xs = sorted(values)
    edges: list[float] = []
    for k in range(bins + 1):
        edge = quantile(xs, k / bins)
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def bin_labels(edges: Sequence[float]) -> list[str]:
    if len(edges) == 1:
        return [format_number(edges[0])]
    labels = []
    for i in range(len(edges) - 1):
        close = "]" if i == len(edges) - 2 else ")"
        labels.append(f"[{format_number(edges[i])}, {format_number(edges[i + 1])}{close}")
    return labels


def bin_index(edges: Sequence[float], value: float) -> int:
    """Index of the bin holding ``value``: bins are [e_i, e_i+1), the last one closed."""
    if len(edges) == 1:
        return 0
    return min(max(bisect_right(edges, value) - 1, 0), len(edges) - 2)


def _split_tags(value: Any, sep: str | None) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = [str(v) for v in value if not is_missing(v)]
    elif sep is not None:
        raw = str(value).split(sep)
    else:
        raw = [str(value)]
    tags = []
    for tag in raw:
        tag = tag.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def detect_kind(values: Iterable[Any], bins: int) -> str:
    """Guess a column's kind from its non-missing values."""
    present = [v for v in values if not is_missing(v)]
    if any(isinstance(v, (list, tuple, set)) for v in present):
        return "tag"
    numbers = [_as_number(v) for v in present]
    if present and all(n is not None for n in numbers) and len(set(numbers)) > bins:
        return "numeric"
    return "category"


def _category_label(value: Any) -> str:
    if isinstance(value, float):
        return format_number(value)
    return str(value)


def slice_column(
    meta: Mapping[str, Mapping[str, Any]],
    items: Iterable[str],
    column: str,
    *,
    kind: str | None = None,
    bins: int = 4,
    tag_sep: str | None = "|",
) -> ColumnSlices:
    """Build the one-way slices of ``column`` over ``items``."""
    items = list(items)
    cells = {item: meta.get(item, {}).get(column) for item in items}
    present = {item: v for item, v in cells.items() if not is_missing(v)}
    missing = len(items) - len(present)
    if kind is None:
        kind = detect_kind(present.values(), bins)
    if kind not in KINDS:
        raise ValueError(f"unknown column kind {kind!r}; use one of {', '.join(KINDS)}")

    groups: dict[str, set[str]] = {}
    edges: list[float] | None = None
    notes: list[str] = []
    if kind == "tag":
        for item, value in present.items():
            for tag in _split_tags(value, tag_sep):
                groups.setdefault(tag, set()).add(item)
        order = sorted(groups)
        notes.append("tag column: an item can be in several of its slices, so they overlap")
    elif kind == "numeric":
        numbers = {}
        for item, value in present.items():
            number = _as_number(value)
            if number is None:
                raise ValueError(f"column {column!r} is numeric but item {item!r} has {value!r}")
            numbers[item] = number
        edges = quantile_edges(numbers.values(), bins) if numbers else []
        labels = bin_labels(edges) if edges else []
        for item, number in numbers.items():
            groups.setdefault(labels[bin_index(edges, number)], set()).add(item)
        order = [label for label in labels if label in groups]
        if edges and len(edges) - 1 < bins:
            notes.append(f"ties left {max(len(edges) - 1, 1)} of {bins} quantile bins")
    else:
        for item, value in present.items():
            groups.setdefault(_category_label(value), set()).add(item)
        order = sorted(groups)

    slices = [
        Slice((column,), (label,), (kind,), frozenset(groups[label])) for label in order
    ]
    return ColumnSlices(column, kind, slices, missing, edges, notes)


def one_way_slices(
    meta: Mapping[str, Mapping[str, Any]],
    items: Iterable[str],
    columns: Sequence[str],
    *,
    kinds: Mapping[str, str] | None = None,
    bins: int = 4,
    tag_sep: str | None = "|",
) -> list[ColumnSlices]:
    """Slice each column in turn."""
    items = list(items)
    kinds = dict(kinds or {})
    return [
        slice_column(meta, items, col, kind=kinds.get(col), bins=bins, tag_sep=tag_sep)
        for col in columns
    ]
