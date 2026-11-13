"""The slice table for one model: each slice's score, its interval, and a test against the rest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .adjust import benjamini_hochberg, holm
from .data import Dataset
from .slices import ColumnSlices, Slice, one_way_slices
from .stats import fisher_exact, wilson_interval


@dataclass
class SliceResult:
    """One row of the slice table."""

    label: str
    columns: tuple[str, ...]
    values: tuple[str, ...]
    kinds: tuple[str, ...]
    n: int
    successes: int
    score: float
    low: float
    high: float
    rest_n: int
    rest_score: float
    diff: float
    p_value: float
    p_holm: float = 1.0
    p_bh: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("columns", "values", "kinds"):
            out[key] = list(out[key])
        return out


@dataclass
class Family:
    """The set of tests the adjusted p-values are computed over."""

    size: int
    description: str
    methods: tuple[str, ...] = ("holm", "benjamini-hochberg")

    def to_dict(self) -> dict[str, Any]:
        return {"size": self.size, "description": self.description, "methods": list(self.methods)}


@dataclass
class SliceTable:
    model: str
    n_items: int
    successes: int
    score: float
    confidence: float
    min_size: int
    family: Family
    rows: list[SliceResult]
    columns: list[dict[str, Any]] = field(default_factory=list)
    skipped_small: int = 0
    skipped_whole: int = 0

    def significant(self, alpha: float = 0.05, method: str = "bh") -> list[SliceResult]:
        key = {"bh": "p_bh", "holm": "p_holm", "raw": "p_value"}[method]
        return [row for row in self.rows if getattr(row, key) < alpha]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_items": self.n_items,
            "successes": self.successes,
            "score": self.score,
            "confidence": self.confidence,
            "min_size": self.min_size,
            "family": self.family.to_dict(),
            "columns": self.columns,
            "skipped_small": self.skipped_small,
            "skipped_whole": self.skipped_whole,
            "rows": [row.to_dict() for row in self.rows],
        }


def _binary_counts(scores: Mapping[str, float]) -> None:
    bad = [item for item, s in scores.items() if s not in (0.0, 1.0)]
    if bad:
        raise NotImplementedError(
            f"slice tables need 0/1 scores for now; item {bad[0]!r} has {scores[bad[0]]!r}"
        )


def score_slice(
    slice_: Slice, scores: Mapping[str, float], total_successes: int, confidence: float
) -> SliceResult:
    """Compare one slice with every other scored item using Fisher's exact test."""
    members = [item for item in slice_.items if item in scores]
    n = len(members)
    successes = int(round(sum(scores[item] for item in members)))
    rest_n = len(scores) - n
    rest_successes = total_successes - successes
    low, high = wilson_interval(successes, n, confidence)
    score = successes / n
    rest_score = rest_successes / rest_n
    p = fisher_exact(successes, n - successes, rest_successes, rest_n - rest_successes)
    return SliceResult(
        label=slice_.label,
        columns=slice_.columns,
        values=slice_.values,
        kinds=slice_.kinds,
        n=n,
        successes=successes,
        score=score,
        low=low,
        high=high,
        rest_n=rest_n,
        rest_score=rest_score,
        diff=score - rest_score,
        p_value=p,
    )


def _column_summary(col: ColumnSlices) -> dict[str, Any]:
    out: dict[str, Any] = {
        "column": col.column,
        "kind": col.kind,
        "slices": len(col.slices),
        "missing": col.missing,
    }
    if col.edges is not None:
        out["edges"] = col.edges
    if col.notes:
        out["notes"] = col.notes
    return out


def slice_table(
    dataset: Dataset,
    model: str,
    columns: Sequence[str] | None = None,
    *,
    kinds: Mapping[str, str] | None = None,
    bins: int = 4,
    tag_sep: str | None = "|",
    min_size: int = 10,
    confidence: float = 0.95,
) -> SliceTable:
    """Test every one-way slice of ``columns`` for ``model`` against the rest of its items.

    "The rest" is every other item scored for the model, including items with no value in the
    column. Slices smaller than ``min_size`` and slices that hold every item are not tested and
    do not count toward the family. Holm and Benjamini-Hochberg adjustments run over all
    remaining slices of all columns together.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    scores = dataset.model_scores(model)
    if not scores:
        raise ValueError(f"model {model!r} has no scored items")
    _binary_counts(scores)
    if columns is None:
        columns = dataset.meta_columns()
    if not columns:
        raise ValueError("no metadata columns to slice by")
    column_slices = one_way_slices(
        dataset.meta, scores, columns, kinds=kinds, bins=bins, tag_sep=tag_sep
    )
    total_successes = int(round(sum(scores.values())))
    rows: list[SliceResult] = []
    skipped_small = skipped_whole = 0
    for col in column_slices:
        for slice_ in col.slices:
            if slice_.size < min_size:
                skipped_small += 1
            elif slice_.size >= len(scores):
                skipped_whole += 1
            else:
                rows.append(score_slice(slice_, scores, total_successes, confidence))
    for row, ph, pb in zip(
        rows,
        holm([r.p_value for r in rows]),
        benjamini_hochberg([r.p_value for r in rows]),
        strict=True,
    ):
        row.p_holm, row.p_bh = ph, pb
    rows.sort(key=lambda r: (r.p_value, r.label))
    family = Family(
        size=len(rows),
        description=f"one-way slices of {', '.join(columns)} with at least {min_size} items",
    )
    return SliceTable(
        model=model,
        n_items=len(scores),
        successes=total_successes,
        score=total_successes / len(scores),
        confidence=confidence,
        min_size=min_size,
        family=family,
        rows=rows,
        columns=[_column_summary(col) for col in column_slices],
        skipped_small=skipped_small,
        skipped_whole=skipped_whole,
    )
