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


def _overall(table: SliceTable) -> str:
    if table.successes is not None:
        return f"{table.successes}/{table.n_items} = {fmt_score(table.score)}"
    return f"mean {fmt_score(table.score)} over {table.n_items} items"


def ranking_markdown(table: SliceTable, column: str) -> str:
    """Raw and shrunk ranking of one column's slices side by side."""
    if column not in table.rankings:
        raise KeyError(f"no shrinkage for column {column!r} (0/1 scores and two slices needed)")
    prior = table.priors[column]
    strength = "infinite" if prior.strength == float("inf") else f"{prior.strength:.1f}"
    out = [
        f"### Raw and shrunk ranking of `{column}`",
        "",
        f"Beta prior fitted to the {len(table.rankings[column])} slices: mean "
        f"{fmt_score(prior.mu)}, prior size a + b = {strength} items.",
        "",
    ]
    body = [
        [
            str(r.shrunk_rank),
            str(r.raw_rank),
            r.label,
            f"{r.successes}/{r.n}",
            fmt_score(r.raw),
            fmt_score(r.shrunk),
        ]
        for r in table.rankings[column]
    ]
    out.append(markdown_rows(["shrunk rank", "raw rank", "slice", "k/n", "raw", "shrunk"], body))
    return "\n".join(out) + "\n"


def table_markdown(
    table: SliceTable,
    *,
    alpha: float = 0.05,
    top: int | None = None,
    rank: list[str] | None = None,
) -> str:
    conf = round(table.confidence * 100)
    out = [
        f"## Slices for `{table.model}`",
        "",
        f"Overall: {_overall(table)}. Interval: {table.interval}; test against the rest: "
        f"{table.test}.",
        "",
        f"Family: {table.family.size} tests ({table.family.description}); "
        "adjusted with Holm (family-wise error) and Benjamini-Hochberg (false discovery rate).",
    ]
    if table.skipped_small or table.skipped_whole:
        out.append(
            f"Not tested: {table.skipped_small} one-way slices under {table.min_size} items, "
            f"{table.skipped_whole} holding every item."
        )
    out += ["", "Columns:", *_column_lines(table.columns), ""]
    rows = table.rows if top is None else table.rows[:top]
    shrink = bool(table.priors)
    body = []
    for row in rows:
        cells = [
            row.label,
            str(row.n),
            fmt_score(row.score),
            f"[{fmt_score(row.low)}, {fmt_score(row.high)}]",
        ]
        if shrink:
            cells.append("" if row.shrunk is None else fmt_score(row.shrunk))
        cells += [
            fmt_score(row.rest_score),
            fmt_diff(row.diff),
            fmt_p(row.p_value),
            fmt_p(row.p_holm),
            fmt_p(row.p_bh),
        ]
        body.append(cells)
    header = ["slice", "n", "score", f"{conf}% CI"]
    if shrink:
        header.append("shrunk")
    header += ["rest", "diff", "p", "p Holm", "p BH"]
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
    text = "\n".join(out) + "\n"
    for column in rank or []:
        text += "\n" + ranking_markdown(table, column)
    return text


def to_json(obj: Any) -> str:
    data = obj.to_dict() if hasattr(obj, "to_dict") else obj
    return json.dumps(data, indent=2, sort_keys=False) + "\n"
