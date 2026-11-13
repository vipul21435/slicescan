"""The slice table for one model: each slice's score, its interval, and a test against the rest."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .adjust import benjamini_hochberg, holm
from .data import Dataset
from .shrink import BetaPrior, RankRow, shrink_ranking
from .slices import ColumnSlices, Slice, one_way_slices, two_way_slices
from .stats import bootstrap_mean_interval, fisher_exact, permutation_test, wilson_interval


@dataclass
class SliceResult:
    """One row of the slice table."""

    label: str
    columns: tuple[str, ...]
    values: tuple[str, ...]
    kinds: tuple[str, ...]
    n: int
    successes: int | None
    score: float
    low: float
    high: float
    rest_n: int
    rest_score: float
    diff: float
    p_value: float
    p_holm: float = 1.0
    p_bh: float = 1.0
    shrunk: float | None = None

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
    successes: int | None
    score: float
    binary: bool
    confidence: float
    min_size: int
    family: Family
    rows: list[SliceResult]
    columns: list[dict[str, Any]] = field(default_factory=list)
    skipped_small: int = 0
    skipped_whole: int = 0
    interval: str = "wilson"
    test: str = "fisher"
    priors: dict[str, BetaPrior] = field(default_factory=dict)
    rankings: dict[str, list[RankRow]] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05, method: str = "bh") -> list[SliceResult]:
        key = {"bh": "p_bh", "holm": "p_holm", "raw": "p_value"}[method]
        return [row for row in self.rows if getattr(row, key) < alpha]

    def find(self, label: str) -> SliceResult:
        for row in self.rows:
            if row.label == label:
                return row
        raise KeyError(f"no tested slice {label!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_items": self.n_items,
            "successes": self.successes,
            "score": self.score,
            "binary": self.binary,
            "confidence": self.confidence,
            "interval": self.interval,
            "test": self.test,
            "min_size": self.min_size,
            "family": self.family.to_dict(),
            "columns": self.columns,
            "skipped_small": self.skipped_small,
            "skipped_whole": self.skipped_whole,
            "rows": [row.to_dict() for row in self.rows],
            "shrinkage": {
                col: {
                    "prior_mean": prior.mu,
                    "prior_strength": None if math.isinf(prior.strength) else prior.strength,
                    "ranking": [asdict(r) for r in self.rankings[col]],
                }
                for col, prior in self.priors.items()
            },
        }


def is_binary(scores: Mapping[str, float]) -> bool:
    return all(s in (0.0, 1.0) for s in scores.values())


def score_slice(
    slice_: Slice,
    scores: Mapping[str, float],
    confidence: float,
    *,
    binary: bool | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> SliceResult:
    """Compare one slice with every other scored item.

    0/1 scores get a Wilson interval and Fisher's exact test against the rest; real-valued
    scores get a percentile bootstrap interval and a permutation test of the difference in
    means.
    """
    if binary is None:
        binary = is_binary(scores)
    members = sorted(item for item in slice_.items if item in scores)
    inside = set(members)
    rest = [scores[item] for item in scores if item not in inside]
    values = [scores[item] for item in members]
    n, rest_n = len(values), len(rest)
    if n == 0 or rest_n == 0:
        raise ValueError(f"slice {slice_.label!r} needs items both inside and outside it")
    score = math.fsum(values) / n
    rest_score = math.fsum(rest) / rest_n
    successes: int | None
    if binary:
        successes = int(round(math.fsum(values)))
        rest_successes = int(round(math.fsum(rest)))
        low, high = wilson_interval(successes, n, confidence)
        p = fisher_exact(successes, n - successes, rest_successes, rest_n - rest_successes)
    else:
        successes = None
        low, high = bootstrap_mean_interval(values, confidence, resamples=resamples, seed=seed)
        p = permutation_test(values, rest, resamples=resamples, seed=seed)
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


def _counts(slice_: Slice, scores: Mapping[str, float]) -> tuple[int, int]:
    members = [item for item in slice_.items if item in scores]
    return int(round(math.fsum(scores[i] for i in members))), len(members)


def _shrink_columns(
    column_slices: Sequence[ColumnSlices], scores: Mapping[str, float]
) -> tuple[dict[str, BetaPrior], dict[str, list[RankRow]], dict[str, float]]:
    """Fit one prior per column over all its slices (small ones too) and shrink each slice."""
    priors: dict[str, BetaPrior] = {}
    rankings: dict[str, list[RankRow]] = {}
    shrunk: dict[str, float] = {}
    for col in column_slices:
        usable = [s for s in col.slices if _counts(s, scores)[1] > 0]
        if len(usable) < 2:
            continue
        counts = [_counts(s, scores) for s in usable]
        prior, rows = shrink_ranking(
            [s.label for s in usable], [k for k, _ in counts], [n for _, n in counts]
        )
        priors[col.column] = prior
        rankings[col.column] = rows
        for row in rows:
            shrunk[row.label] = row.shrunk
    return priors, rankings, shrunk


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
    two_way: bool = False,
    two_way_cap: int | None = 100,
    resamples: int = 2000,
    seed: int = 0,
) -> SliceTable:
    """Test every slice of ``columns`` for ``model`` against the rest of its items.

    "The rest" is every other item scored for the model, including items with no value in the
    column. Slices smaller than ``min_size`` and slices that hold every item are not tested and
    do not count toward the family. With ``two_way``, intersections of slices from two
    different columns are tested too, at most ``two_way_cap`` of them (the largest). Holm and
    Benjamini-Hochberg adjustments run over every slice tested, one-way and two-way together.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    scores = dataset.model_scores(model)
    if not scores:
        raise ValueError(f"model {model!r} has no scored items")
    binary = is_binary(scores)
    if columns is None:
        columns = dataset.meta_columns()
    columns = list(columns)
    if not columns:
        raise ValueError("no metadata columns to slice by")
    column_slices = one_way_slices(
        dataset.meta, scores, columns, kinds=kinds, bins=bins, tag_sep=tag_sep
    )
    candidates: list[Slice] = []
    skipped_small = skipped_whole = 0
    for col in column_slices:
        for slice_ in col.slices:
            if slice_.size < min_size:
                skipped_small += 1
            elif slice_.size >= len(scores):
                skipped_whole += 1
            else:
                candidates.append(slice_)
    description = f"{len(candidates)} one-way slices of {', '.join(columns)}"
    if two_way:
        extra = two_way_slices(column_slices, len(scores), min_size=min_size, cap=two_way_cap)
        candidates.extend(extra.slices)
        description += f" + {len(extra.slices)} two-way slices of {len(extra.pairs)} column pairs"
        if extra.dropped:
            description += (
                f" (cap {extra.cap}, largest kept; {extra.dropped} smaller ones not tested)"
            )
    description += f", each with at least {min_size} items"
    rows = [
        score_slice(s, scores, confidence, binary=binary, resamples=resamples, seed=seed + i)
        for i, s in enumerate(candidates)
    ]
    for row, ph, pb in zip(
        rows,
        holm([r.p_value for r in rows]),
        benjamini_hochberg([r.p_value for r in rows]),
        strict=True,
    ):
        row.p_holm, row.p_bh = ph, pb
    priors: dict[str, BetaPrior] = {}
    rankings: dict[str, list[RankRow]] = {}
    if binary:
        priors, rankings, shrunk = _shrink_columns(column_slices, scores)
        for row in rows:
            if len(row.columns) == 1:
                row.shrunk = shrunk.get(row.label)
    rows.sort(key=lambda r: (r.p_value, r.label))
    total = math.fsum(scores.values())
    return SliceTable(
        model=model,
        n_items=len(scores),
        successes=int(round(total)) if binary else None,
        score=total / len(scores),
        binary=binary,
        confidence=confidence,
        min_size=min_size,
        family=Family(size=len(rows), description=description),
        rows=rows,
        columns=[_column_summary(col) for col in column_slices],
        skipped_small=skipped_small,
        skipped_whole=skipped_whole,
        interval="wilson" if binary else "percentile bootstrap",
        test="fisher exact" if binary else "permutation",
        priors=priors,
        rankings=rankings,
    )
