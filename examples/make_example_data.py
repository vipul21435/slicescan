"""Write the small synthetic example used in the README and the tests of the CLI.

240 items from six made-up projects, each with a numeric size and a few tags. Two systems are
scored on every item; "alpha" is planted to fail more often on the project "kiwi".

    python examples/make_example_data.py examples/data
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

PROJECTS = ["apple", "fig", "kiwi", "lime", "pear", "plum"]
TAGS = ["io", "parsing", "numeric", "ui"]


def make(out_dir: Path, seed: int = 7, n: int = 240) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    results = []
    for i in range(n):
        item = f"item-{i:03d}"
        project = PROJECTS[i % len(PROJECTS)]
        lines = int(rng.lognormvariate(3.0, 1.0)) + 1
        tags = sorted(rng.sample(TAGS, rng.randint(1, 2)))
        meta_rows.append({"item": item, "project": project, "lines": lines, "tags": "|".join(tags)})
        hard = min(0.5, lines / 400)
        alpha_rate = (0.25 if project == "kiwi" else 0.65) - hard
        beta_rate = 0.55 - hard
        results.append({"item": item, "model": "alpha", "score": int(rng.random() < alpha_rate)})
        results.append({"item": item, "model": "beta", "score": int(rng.random() < beta_rate)})
    with (out_dir / "meta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "project", "lines", "tags"])
        writer.writeheader()
        writer.writerows(meta_rows)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/data"))
