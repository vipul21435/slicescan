"""Summarise slicescan's output over every submission and every pair in a fetched folder.

    python examples/swebench_verified/fetch.py DATA
    python examples/swebench_verified/summarise.py DATA

Prints, per submission, how many slices have p < 0.05 before and after adjustment and how far
the shrunk ranking of repositories moves from the raw one; then, per pair of submissions, the
overall paired difference, the sign flips and the reversal verdicts.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from slicescan.compare import compare
from slicescan.data import load_dataset
from slicescan.reversals import reversals
from slicescan.shrink import spearman
from slicescan.table import slice_table

COLUMNS = ["repo", "difficulty", "year", "patch_lines"]


def short(name: str) -> str:
    return name.split("_", 2)[-1]


def main(folder: Path) -> None:
    ds = load_dataset(folder / "results.jsonl", folder / "meta.csv")
    print("| submission | solved | tests | p<0.05 raw | BH | Holm | repo ranks moved | Spearman |")
    print("|---|---|---|---|---|---|---|---|")
    for model in ds.models:
        t = slice_table(ds, model, COLUMNS, min_size=10)
        ranks = t.rankings["repo"]
        moved = sum(1 for r in ranks if r.raw_rank != r.shrunk_rank)
        rho = spearman([r.raw_rank for r in ranks], [r.shrunk_rank for r in ranks])
        print(
            f"| {short(model)} | {t.successes}/{t.n_items} | {t.family.size} "
            f"| {len(t.significant(0.05, 'raw'))} | {len(t.significant(0.05, 'bh'))} "
            f"| {len(t.significant(0.05, 'holm'))} | {moved} of {len(ranks)} | {rho:.3f} |"
        )
    print()
    print("| A | B | A - B | flips | flips BH<0.05 | slices BH<0.05 | reversal verdicts |")
    print("|---|---|---|---|---|---|---|")
    for a, b in itertools.combinations(ds.models, 2):
        c = compare(ds, a, b, COLUMNS, min_size=10)
        r = reversals(ds, a, b, COLUMNS)
        verdicts = ", ".join(f"{x.column}: {x.verdict}" for x in r.checks if x.verdict != "none")
        print(
            f"| {short(a)} | {short(b)} | {c.overall.diff:+.3f} | {len(c.flips)} "
            f"| {sum(1 for x in c.flips if x.p_bh < 0.05)} | {len(c.significant(0.05, 'bh'))} "
            f"| {verdicts or 'none'} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
