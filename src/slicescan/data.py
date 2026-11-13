"""Read per-item results and item metadata from JSONL or CSV files."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TRUE = {"true", "yes"}
_FALSE = {"false", "no"}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read records from JSON lines (``.jsonl``, ``.ndjson``, ``.json``) or ``.csv``/``.tsv``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            return [dict(row) for row in reader]
    if suffix in {".jsonl", ".ndjson", ".json"}:
        rows = []
        with path.open(encoding="utf-8-sig") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from None
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{lineno}: expected a JSON object per line")
                rows.append(record)
        return rows
    raise ValueError(f"{path}: unsupported file type {suffix!r} (use .jsonl, .csv or .tsv)")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def parse_score(value: Any) -> float:
    """Turn a score cell into a float: booleans and 'true'/'false' become 1 and 0."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return 1.0
        if text in _FALSE:
            return 0.0
        try:
            score = float(text)
        except ValueError:
            raise ValueError(f"cannot read a score from {value!r}") from None
    else:
        raise ValueError(f"cannot read a score from {value!r}")
    if not math.isfinite(score):
        raise ValueError(f"score must be a finite number, got {value!r}")
    return score


@dataclass
class Dataset:
    """Scores per model and item, and metadata per item."""

    scores: dict[str, dict[str, float]]
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_model_columns: list[str] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        return sorted(self.scores)

    def meta_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.meta.values():
            for key in row:
                seen.setdefault(key, None)
        return list(seen)

    def check_columns(self, columns: list[str], items: Iterable[str]) -> None:
        """Raise if a column is unknown or if none of ``items`` has any metadata."""
        known = self.meta_columns()
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ValueError(
                f"unknown column {unknown[0]!r}; metadata columns: {', '.join(known) or 'none'}"
            )
        if not any(self.meta.get(item) for item in items):
            raise ValueError("no scored item has metadata; check the item ids in both files")

    def model_scores(self, model: str) -> dict[str, float]:
        if model not in self.scores:
            raise KeyError(f"model {model!r} not found; models: {', '.join(self.models)}")
        return self.scores[model]


def _merge_meta(meta: dict[str, dict[str, Any]], item: str, row: dict[str, Any]) -> None:
    current = meta.setdefault(item, {})
    for key, value in row.items():
        if is_missing(value):
            continue
        if key in current and current[key] != value:
            raise ValueError(
                f"item {item!r} has two values for column {key!r}: {current[key]!r} and {value!r}"
            )
        current[key] = value


def build_dataset(
    results: Iterable[dict[str, Any]],
    meta_rows: Iterable[dict[str, Any]] | None = None,
    *,
    item_col: str = "item",
    model_col: str = "model",
    score_col: str = "score",
    meta_item_col: str | None = None,
) -> Dataset:
    """Join result rows (item, model, score, ...) with optional metadata rows keyed by item.

    Columns of the result rows other than item, model and score are taken as item metadata when
    every model gives an item the same value; a column whose value differs between models for
    some item (a cost or a run time, say) is a per-model column, left out of the metadata and
    listed in ``per_model_columns``. An item scored twice for the same model is an error, as is
    an item with two different values for the same column of the metadata file.
    """
    scores: dict[str, dict[str, float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    from_results: dict[str, dict[str, Any]] = {}
    per_model: set[str] = set()
    for n, row in enumerate(results, start=1):
        for col in (item_col, model_col, score_col):
            if col not in row or is_missing(row[col]):
                raise ValueError(f"result row {n} has no {col!r} value")
        item = str(row[item_col])
        model = str(row[model_col])
        model_scores = scores.setdefault(model, {})
        if item in model_scores:
            raise ValueError(f"item {item!r} is scored twice for model {model!r}")
        try:
            model_scores[item] = parse_score(row[score_col])
        except ValueError as exc:
            raise ValueError(f"result row {n}: {exc}") from None
        current = from_results.setdefault(item, {})
        for key, value in row.items():
            if key in (item_col, model_col, score_col) or is_missing(value):
                continue
            if key in current and current[key] != value:
                per_model.add(key)
            current.setdefault(key, value)
    for item, row in from_results.items():
        kept = {k: v for k, v in row.items() if k not in per_model}
        if kept:
            meta[item] = kept
    if meta_rows is not None:
        key = meta_item_col or item_col
        for n, row in enumerate(meta_rows, start=1):
            if key not in row or is_missing(row[key]):
                raise ValueError(f"metadata row {n} has no {key!r} value")
            item = str(row[key])
            _merge_meta(meta, item, {k: v for k, v in row.items() if k != key})
    return Dataset(scores=scores, meta=meta, per_model_columns=sorted(per_model))


def load_dataset(
    results_path: str | Path,
    meta_path: str | Path | None = None,
    **columns: str | None,
) -> Dataset:
    """Read the files and join them with :func:`build_dataset`."""
    meta_rows = read_rows(meta_path) if meta_path is not None else None
    return build_dataset(read_rows(results_path), meta_rows, **columns)
