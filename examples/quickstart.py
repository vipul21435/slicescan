"""The README's quickstart from Python, on the synthetic example data.

python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from slicescan.compare import compare
from slicescan.data import load_dataset
from slicescan.report import compare_markdown, ranking_markdown, table_markdown
from slicescan.reversals import reversals
from slicescan.table import slice_table

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    ds = load_dataset(DATA / "results.jsonl", DATA / "meta.csv")
    kinds = {"tags": "tag"}
    table = slice_table(ds, "alpha", ["project", "lines", "tags"], kinds=kinds)
    print(table_markdown(table, top=5))
    print(ranking_markdown(table, "project"))
    cmp = compare(ds, "alpha", "beta", ["project", "lines", "tags"], kinds=kinds)
    print(compare_markdown(cmp, top=5))
    report = reversals(ds, "alpha", "beta", ["project", "lines"])
    for check in report.checks:
        print(f"{check.column}: winner {check.winner}, loses in {check.losses}, {check.verdict}")


if __name__ == "__main__":
    main()
