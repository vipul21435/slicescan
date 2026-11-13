"""Command-line interface: ``slicescan table``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .data import load_dataset
from .report import table_markdown, to_json
from .slices import KINDS
from .table import slice_table


def _kinds(pairs: Sequence[str]) -> dict[str, str]:
    kinds = {}
    for pair in pairs:
        column, sep, kind = pair.partition("=")
        if not sep or kind not in KINDS:
            raise SystemExit(f"--kind wants COLUMN=KIND with KIND one of {', '.join(KINDS)}")
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
    return parser


def _pick_model(models: list[str], wanted: str | None) -> str:
    if wanted is not None:
        return wanted
    if len(models) == 1:
        return models[0]
    raise SystemExit(f"several models in the results; pick one with --model: {', '.join(models)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(
        args.results,
        args.meta,
        item_col=args.item_col,
        model_col=args.model_col,
        score_col=args.score_col,
        meta_item_col=args.meta_item_col,
    )
    if args.command == "table":
        table = slice_table(
            dataset,
            _pick_model(dataset.models, args.model),
            _columns(args.columns),
            kinds=_kinds(args.kind),
            bins=args.bins,
            tag_sep=args.tag_sep,
            min_size=args.min_size,
            confidence=args.confidence,
        )
        if args.format == "json":
            sys.stdout.write(to_json(table))
        else:
            sys.stdout.write(table_markdown(table, alpha=args.alpha, top=args.top))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
