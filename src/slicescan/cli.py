"""Command-line interface: ``slicescan table | compare | reversals | explain``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .compare import compare
from .data import load_dataset
from .explain import explain
from .report import (
    compare_markdown,
    explain_markdown,
    reversals_markdown,
    table_chart,
    table_markdown,
    to_json,
)
from .reversals import reversals
from .slices import KINDS
from .table import slice_table


def _kinds(pairs: Sequence[str]) -> dict[str, str]:
    kinds = {}
    for pair in pairs:
        column, sep, kind = pair.partition("=")
        if not sep or kind not in KINDS:
            raise ValueError(f"--kind wants COLUMN=KIND with KIND one of {', '.join(KINDS)}")
        kinds[column] = kind
    return kinds


def _columns(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [c.strip() for c in text.split(",") if c.strip()]


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results", help="per-item results, .jsonl or .csv (item, model, score)")
    parser.add_argument("--meta", help="item metadata, .jsonl or .csv, joined on the item column")
    parser.add_argument("--item-col", default="item")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--meta-item-col", help="item column of the metadata file if different")
    parser.add_argument("--columns", help="comma-separated metadata columns (default: all)")
    parser.add_argument(
        "--kind",
        action="append",
        default=[],
        metavar="COLUMN=KIND",
        help="force a column's kind: category, numeric or tag (repeatable)",
    )
    parser.add_argument("--bins", type=int, default=4, help="quantile bins for numeric columns")
    parser.add_argument("--tag-sep", default="|", help="separator inside tag cells (default '|')")
    parser.add_argument("--min-size", type=int, default=10, help="smallest slice that is tested")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--top", type=int, help="show only the first N rows")
    parser.add_argument(
        "--two-way", action="store_true", help="also test intersections of two columns"
    )
    parser.add_argument(
        "--two-way-cap", type=int, default=100, help="most two-way slices to test (largest first)"
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="bootstrap and permutation draws (real scores)"
    )
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slicescan",
        description="Find metadata slices where a model's score differs from the rest.",
    )
    parser.add_argument("--version", action="version", version=f"slicescan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    table = sub.add_parser("table", help="slice table for one model")
    _add_input(table)
    table.add_argument("--model", help="model to analyse (default: the only one in the file)")
    table.add_argument(
        "--rank",
        action="append",
        default=[],
        metavar="COLUMN",
        help="print the raw and shrunk ranking of this column's slices (repeatable)",
    )
    table.add_argument("--chart", metavar="PNG", help="write a dot plot (needs matplotlib)")

    cmp = sub.add_parser("compare", help="paired comparison of two systems per slice")
    _add_input(cmp)
    cmp.add_argument("model_a", metavar="A")
    cmp.add_argument("model_b", metavar="B")

    rev = sub.add_parser("reversals", help="does the overall winner lose inside the slices?")
    _add_input(rev)
    rev.add_argument("model_a", metavar="A")
    rev.add_argument("model_b", metavar="B")

    exp = sub.add_parser("explain", help="one slice in detail, within the table's family")
    _add_input(exp)
    exp.add_argument("--model", help="model to analyse (default: the only one in the file)")
    exp.add_argument(
        "--slice",
        required=True,
        dest="slice_label",
        help='slice label, e.g. "repo=x" or "repo=x & size=[1, 5)"',
    )
    exp.add_argument("--max-items", type=int, default=20)
    return parser


def _pick_model(models: list[str], wanted: str | None) -> str:
    if wanted is not None:
        return wanted
    if len(models) == 1:
        return models[0]
    raise ValueError(f"several models in the results; pick one with --model: {', '.join(models)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; user errors print one line to stderr and return 2."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, KeyError, OSError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        sys.stderr.write(f"slicescan: error: {message}\n")
        return 2


def _run(args: argparse.Namespace) -> int:
    dataset = load_dataset(
        args.results,
        args.meta,
        item_col=args.item_col,
        model_col=args.model_col,
        score_col=args.score_col,
        meta_item_col=args.meta_item_col,
    )
    if not dataset.scores:
        raise ValueError(f"no result rows in {args.results}")
    common = {
        "kinds": _kinds(args.kind),
        "bins": args.bins,
        "tag_sep": args.tag_sep,
    }
    tested = {
        "min_size": args.min_size,
        "confidence": args.confidence,
        "two_way": args.two_way,
        "two_way_cap": args.two_way_cap,
        "resamples": args.resamples,
        "seed": args.seed,
    }
    columns = _columns(args.columns)
    if args.command == "table":
        result = slice_table(
            dataset, _pick_model(dataset.models, args.model), columns, **common, **tested
        )
        if args.chart:
            table_chart(result, args.chart, top=args.top or 20)
        text = table_markdown(result, alpha=args.alpha, top=args.top, rank=args.rank)
    elif args.command == "compare":
        result = compare(dataset, args.model_a, args.model_b, columns, **common, **tested)
        text = compare_markdown(result, alpha=args.alpha, top=args.top)
    elif args.command == "reversals":
        result = reversals(dataset, args.model_a, args.model_b, columns, **common)
        text = reversals_markdown(result)
    else:
        result = explain(
            dataset,
            _pick_model(dataset.models, args.model),
            args.slice_label,
            columns,
            **common,
            **tested,
            max_items=args.max_items,
        )
        text = explain_markdown(result)
    sys.stdout.write(to_json(result) if args.format == "json" else text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
