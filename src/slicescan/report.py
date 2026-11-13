"""Markdown and JSON renderings of the result objects."""

from __future__ import annotations

import json
from typing import Any

from .table import SliceTable


def fmt_p(p: float) -> str:
    """p-values: three significant digits, scientific notation below 0.001."""
    if p >= 0.001:
        return f"{p:.3f}"
    return f"{p:.2e}"


def fmt_score(x: float) -> str:
    return f"{x:.3f}"


def fmt_diff(x: float) -> str:
    return f"{x:+.3f}"


def markdown_rows(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _column_lines(columns: list[dict[str, Any]]) -> list[str]:
    lines = []
    for col in columns:
        text = f"- `{col['column']}`: {col['kind']}, {col['slices']} slices"
        if col["missing"]:
            text += f", {col['missing']} items with no value"
        for note in col.get("notes", []):
            text += f"; {note}"
        lines.append(text)
    return lines


def table_markdown(table: SliceTable, *, alpha: float = 0.05, top: int | None = None) -> str:
    conf = round(table.confidence * 100)
    out = [
        f"## Slices for `{table.model}`",
        "",
        f"Overall: {table.successes}/{table.n_items} = {fmt_score(table.score)}.",
        "",
        f"Family: {table.family.size} tests ({table.family.description}); "
        "adjusted with Holm (family-wise error) and Benjamini-Hochberg (false discovery rate).",
    ]
    if table.skipped_small or table.skipped_whole:
        out.append(
            f"Not tested: {table.skipped_small} slices under {table.min_size} items, "
            f"{table.skipped_whole} slices holding every item."
        )
    out += ["", "Columns:", *_column_lines(table.columns), ""]
    rows = table.rows if top is None else table.rows[:top]
    body = [
        [
            row.label,
            str(row.n),
            fmt_score(row.score),
            f"[{fmt_score(row.low)}, {fmt_score(row.high)}]",
            fmt_score(row.rest_score),
            fmt_diff(row.diff),
            fmt_p(row.p_value),
            fmt_p(row.p_holm),
            fmt_p(row.p_bh),
        ]
        for row in rows
    ]
    header = ["slice", "n", "score", f"{conf}% CI", "rest", "diff", "p", "p Holm", "p BH"]
    out.append(markdown_rows(header, body))
    if top is not None and len(table.rows) > top:
        out.append(f"\n({len(table.rows) - top} more rows not shown; all count in the family.)")
    raw = len(table.significant(alpha, "raw"))
    bh = len(table.significant(alpha, "bh"))
    hm = len(table.significant(alpha, "holm"))
    out += [
        "",
        f"Slices with p < {alpha}: {raw} before adjustment, {bh} after Benjamini-Hochberg, "
        f"{hm} after Holm.",
    ]
    return "\n".join(out) + "\n"


def to_json(obj: Any) -> str:
    data = obj.to_dict() if hasattr(obj, "to_dict") else obj
    return json.dumps(data, indent=2, sort_keys=False) + "\n"
