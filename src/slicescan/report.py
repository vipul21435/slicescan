"""Markdown and JSON renderings of the result objects."""

from __future__ import annotations

import json
import math
from typing import Any

from .compare import Comparison
from .explain import Explanation
from .reversals import ReversalReport
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
            f"{r.shrunk_rank:g}",
            f"{r.raw_rank:g}",
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


def compare_markdown(cmp: Comparison, *, alpha: float = 0.05, top: int | None = None) -> str:
    conf = round(cmp.confidence * 100)
    o = cmp.overall
    test = "exact McNemar" if cmp.binary else "sign-flip permutation"
    interval = "Agresti-Min" if cmp.binary else "normal"
    out = [
        f"## `{cmp.model_a}` minus `{cmp.model_b}`",
        "",
        f"Shared items: {o.n} ({cmp.only_a} scored only for `{cmp.model_a}`, {cmp.only_b} only "
        f"for `{cmp.model_b}`; those are left out).",
        f"Overall: {fmt_score(o.score_a)} vs {fmt_score(o.score_b)}, difference "
        f"{fmt_diff(o.diff)} [{fmt_diff(o.low)}, {fmt_diff(o.high)}], "
        f"{test} p = {fmt_p(o.p_value)}.",
        "",
        f"Family: {cmp.family.size} tests ({cmp.family.description}); adjusted with Holm and "
        f"Benjamini-Hochberg. Interval: {interval}, {conf}%.",
        "",
    ]
    header = ["slice", "n", "A", "B", "A - B", f"{conf}% CI"]
    if cmp.binary:
        header += ["A only", "B only"]
    header += ["p", "p Holm", "p BH", "flips"]
    body = []
    rows = cmp.rows if top is None else cmp.rows[:top]
    for row in rows:
        cells = [
            row.label,
            str(row.n),
            fmt_score(row.score_a),
            fmt_score(row.score_b),
            fmt_diff(row.diff),
            f"[{fmt_diff(row.low)}, {fmt_diff(row.high)}]",
        ]
        if cmp.binary:
            cells += [str(row.a_only), str(row.b_only)]
        cells += [
            fmt_p(row.p_value),
            fmt_p(row.p_holm),
            fmt_p(row.p_bh),
            "yes" if row.flips else "",
        ]
        body.append(cells)
    out.append(markdown_rows(header, body))
    if top is not None and len(cmp.rows) > top:
        out.append(f"\n({len(cmp.rows) - top} more rows not shown; all count in the family.)")
    flips = cmp.flips
    sig_flips = [r for r in flips if r.p_bh < alpha]
    out += [
        "",
        f"Slices with p < {alpha}: {len(cmp.significant(alpha, 'raw'))} before adjustment, "
        f"{len(cmp.significant(alpha, 'bh'))} after Benjamini-Hochberg, "
        f"{len(cmp.significant(alpha, 'holm'))} after Holm.",
        f"Sign flips against the overall difference: {len(flips)} slices, {len(sig_flips)} of "
        f"them with BH-adjusted p < {alpha}.",
    ]
    return "\n".join(out) + "\n"


def _opt(x: float | None) -> str:
    return "" if x is None else fmt_score(x)


def reversals_markdown(report: ReversalReport) -> str:
    out = [
        f"## Aggregation reversals: `{report.model_a}` vs `{report.model_b}`",
        "",
        f"Items: {report.n_a} for `{report.model_a}`, {report.n_b} for `{report.model_b}`, "
        f"{report.shared} shared. Each system is scored on its own items.",
        "",
    ]
    names = {"a": report.model_a, "b": report.model_b, "tie": "tie", "": "-"}
    for check in report.checks:
        out.append(
            f"### `{check.column}` ({check.kind}): overall winner {names[check.winner]}, "
            f"wins {check.wins}, loses {check.losses}, ties {check.ties}; verdict: "
            f"**{check.verdict}**"
        )
        for note in check.notes:
            out.append(f"\nNote: {note}.")
        out.append("")
        body = [
            [p.label, str(p.n_a), _opt(p.score_a), str(p.n_b), _opt(p.score_b), names[p.winner]]
            for p in check.slices
        ]
        out.append(markdown_rows(["slice", "n A", "A", "n B", "B", "better"], body))
        out.append("")
    if report.skipped:
        out.append(f"Not checked (tag columns, not a partition): {', '.join(report.skipped)}.")
        out.append("")
    found = report.reversals
    out.append(
        "Reversals found: "
        + (", ".join(f"{c.column} ({c.verdict})" for c in found) if found else "none")
        + "."
    )
    return "\n".join(out) + "\n"


def explain_markdown(exp: Explanation) -> str:
    row = exp.row
    conf = round(exp.table.confidence * 100)
    out = [f"## `{exp.label}` for `{exp.model}`", ""]
    if row.successes is not None:
        rest_k = round(row.rest_score * row.rest_n)
        out += [
            markdown_rows(
                ["", "solved", "not solved", "total"],
                [
                    ["slice", str(row.successes), str(row.n - row.successes), str(row.n)],
                    ["rest", str(rest_k), str(row.rest_n - rest_k), str(row.rest_n)],
                ],
            ),
            "",
        ]
    out.append(
        f"Score {fmt_score(row.score)} [{fmt_score(row.low)}, {fmt_score(row.high)}] ({conf}%), "
        f"rest {fmt_score(row.rest_score)}, difference {fmt_diff(row.diff)}, "
        f"{exp.table.test} p = {fmt_p(row.p_value)}."
    )
    if row.shrunk is not None:
        out.append(f"Shrunk score (beta-binomial, this column's slices): {fmt_score(row.shrunk)}.")
    if exp.tested:
        out.append(
            f"In the family of {exp.table.family.size} tests it ranks {exp.rank} by p; "
            f"Holm p = {fmt_p(row.p_holm)}, BH p = {fmt_p(row.p_bh)}."
        )
    else:
        out.append(
            f"Not in the tested family (under {exp.table.min_size} items or holding every item), "
            "so it has no adjusted p-value; the raw p-value above has not paid for the other "
            "slices looked at."
        )
    for b in exp.breakdowns:
        out += ["", f"Inside the slice, by `{b.column}`:", ""]
        out.append(
            markdown_rows(
                ["value", "n", "score"], [[v, str(n), fmt_score(m)] for v, n, m in b.rows]
            )
        )
    shown = ", ".join(exp.failed) if exp.failed else "none"
    out += ["", f"Items scored below 0.5 (first {len(exp.failed)}): {shown}."]
    return "\n".join(out) + "\n"


def table_chart(table: SliceTable, path: str, *, top: int = 20) -> None:
    """Dot plot of the slices with the smallest p-values: score, interval and shrunk score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table.rows[:top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.4))
    ys = list(range(len(rows)))
    ax.hlines(ys, [r.low for r in rows], [r.high for r in rows], color="#4c72b0", lw=2)
    ax.plot([r.score for r in rows], ys, "o", color="#4c72b0", label="score with interval")
    shrunk = [(r.shrunk, y) for r, y in zip(rows, ys, strict=True) if r.shrunk is not None]
    if shrunk:
        ax.plot(
            [s for s, _ in shrunk],
            [y for _, y in shrunk],
            "x",
            color="#dd8452",
            label="shrunk score",
        )
    ax.axvline(table.score, color="grey", ls="--", lw=1, label="overall")
    for r, y in zip(rows, ys, strict=True):
        if r.p_bh < 0.05:
            ax.text(1.01, y, "BH<0.05", va="center", fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.label} (n={r.n})" for r in rows], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title(f"{table.model}: {len(rows)} slices with the smallest p", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def to_json(obj: Any) -> str:
    """JSON text; NaN and infinite values become null so the output is valid JSON."""
    data = obj.to_dict() if hasattr(obj, "to_dict") else obj
    return json.dumps(_clean(data), indent=2, allow_nan=False) + "\n"
