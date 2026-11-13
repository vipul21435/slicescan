"""Two systems scored on the same items: paired differences per slice and sign flips."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .adjust import benjamini_hochberg, holm
from .data import Dataset
from .slices import Slice, one_way_slices, two_way_slices
from .stats import mcnemar_exact, mean_interval, paired_diff_interval, sign_flip_test
from .table import Family, _column_summary, is_binary


@dataclass
class PairedResult:
    """Paired comparison of systems A and B on one set of items (a slice or all of them)."""

    label: str
    columns: tuple[str, ...]
    values: tuple[str, ...]
    n: int
    score_a: float
    score_b: float
    diff: float
    low: float
    high: float
    a_only: int | None
    b_only: int | None
    p_value: float
    p_holm: float = 1.0
    p_bh: float = 1.0
    flips: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["columns"] = list(out["columns"])
        out["values"] = list(out["values"])
        return out


def paired_result(
    label: str,
    items: Sequence[str],
    a: Mapping[str, float],
    b: Mapping[str, float],
    confidence: float,
    *,
    binary: bool,
    columns: tuple[str, ...] = (),
    values: tuple[str, ...] = (),
    resamples: int = 2000,
    seed: int = 0,
) -> PairedResult:
    """Mean of A - B over ``items`` with an interval and a paired test.

    0/1 scores: the interval of Agresti and Min (2005) and the exact McNemar test on the
    discordant items. Real scores: a normal interval of the mean difference and a sign-flip
    permutation test.
    """
    n = len(items)
    if n == 0:
        raise ValueError(f"{label!r} has no items scored by both systems")
    score_a = math.fsum(a[i] for i in items) / n
    score_b = math.fsum(b[i] for i in items) / n
    if binary:
        a_only = sum(1 for i in items if a[i] == 1.0 and b[i] == 0.0)
        b_only = sum(1 for i in items if a[i] == 0.0 and b[i] == 1.0)
        diff = (a_only - b_only) / n
        low, high = paired_diff_interval(a_only, b_only, n, confidence)
        p = mcnemar_exact(a_only, b_only)
    else:
        a_only = b_only = None
        d = [a[i] - b[i] for i in items]
        diff = math.fsum(d) / n
        low, high = mean_interval(d, confidence)
        p = sign_flip_test(d, resamples=resamples, seed=seed)
    return PairedResult(
        label, columns, values, n, score_a, score_b, diff, low, high, a_only, b_only, p
    )


@dataclass
class Comparison:
    model_a: str
    model_b: str
    binary: bool
    confidence: float
    min_size: int
    overall: PairedResult
    family: Family
    rows: list[PairedResult]
    only_a: int = 0
    only_b: int = 0
    columns: list[dict[str, Any]] = field(default_factory=list)
    skipped_small: int = 0

    @property
    def flips(self) -> list[PairedResult]:
        """Slices whose difference has the opposite sign to the overall difference."""
        return [row for row in self.rows if row.flips]

    def significant(self, alpha: float = 0.05, method: str = "bh") -> list[PairedResult]:
        key = {"bh": "p_bh", "holm": "p_holm", "raw": "p_value"}[method]
        return [row for row in self.rows if getattr(row, key) < alpha]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "binary": self.binary,
            "confidence": self.confidence,
            "min_size": self.min_size,
            "items_only_a": self.only_a,
            "items_only_b": self.only_b,
            "overall": self.overall.to_dict(),
            "family": self.family.to_dict(),
            "columns": self.columns,
            "skipped_small": self.skipped_small,
            "rows": [row.to_dict() for row in self.rows],
            "flips": [row.label for row in self.flips],
        }


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def compare(
    dataset: Dataset,
    model_a: str,
    model_b: str,
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
) -> Comparison:
    """Paired comparison of two systems on the items both were scored on, slice by slice.

    The family for the adjusted p-values is every slice tested (one-way, and two-way when
    asked). A slice "flips" when its difference is non-zero and of the opposite sign to the
    overall difference; flips are reported whether or not they are significant.
    """
    if model_a == model_b:
        raise ValueError("pick two different systems")
    a = dataset.model_scores(model_a)
    b = dataset.model_scores(model_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(f"{model_a!r} and {model_b!r} share no items")
    binary = is_binary({i: a[i] for i in common}) and is_binary({i: b[i] for i in common})
    overall = paired_result(
        "all items", common, a, b, confidence, binary=binary, resamples=resamples, seed=seed
    )
    if columns is None:
        columns = dataset.meta_columns()
    columns = list(columns)
    if not columns:
        raise ValueError("no metadata columns to slice by")
    dataset.check_columns(columns, common)
    column_slices = one_way_slices(
        dataset.meta, common, columns, kinds=kinds, bins=bins, tag_sep=tag_sep
    )
    candidates: list[Slice] = []
    skipped_small = 0
    for col in column_slices:
        for slice_ in col.slices:
            if slice_.size < min_size:
                skipped_small += 1
            elif slice_.size < len(common):
                candidates.append(slice_)
    description = f"{len(candidates)} one-way slices of {', '.join(columns)}"
    if two_way:
        extra = two_way_slices(column_slices, len(common), min_size=min_size, cap=two_way_cap)
        candidates.extend(extra.slices)
        pairs = f"{len(extra.pairs)} column pair" + ("" if len(extra.pairs) == 1 else "s")
        description += f" + {len(extra.slices)} two-way slices of {pairs}"
        if extra.dropped:
            description += (
                f" (cap {extra.cap}, largest kept; {extra.dropped} smaller ones not tested)"
            )
    description += f", each with at least {min_size} shared items"
    rows = [
        paired_result(
            s.label,
            sorted(s.items),
            a,
            b,
            confidence,
            binary=binary,
            columns=s.columns,
            values=s.values,
            resamples=resamples,
            seed=seed + 1 + k,
        )
        for k, s in enumerate(candidates)
    ]
    for row, ph, pb in zip(
        rows,
        holm([r.p_value for r in rows]),
        benjamini_hochberg([r.p_value for r in rows]),
        strict=True,
    ):
        row.p_holm, row.p_bh = ph, pb
        overall_sign = _sign(overall.diff)
        row.flips = overall_sign != 0 and _sign(row.diff) == -overall_sign
    rows.sort(key=lambda r: (r.p_value, r.label))
    return Comparison(
        model_a=model_a,
        model_b=model_b,
        binary=binary,
        confidence=confidence,
        min_size=min_size,
        overall=overall,
        family=Family(size=len(rows), description=description),
        rows=rows,
        only_a=len(set(a) - set(b)),
        only_b=len(set(b) - set(a)),
        columns=[_column_summary(col) for col in column_slices],
        skipped_small=skipped_small,
    )
