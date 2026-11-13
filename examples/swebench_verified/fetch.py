"""Fetch public SWE-bench Verified per-instance results and instance metadata into a folder.

Neither source states a licence, so nothing they hold is committed to this repository; this
script downloads them for a local run and writes two derived files:

* ``results.jsonl``: one row per (instance, submission) with ``score`` 1 if resolved, else 0;
* ``meta.csv``: per instance, the repository, the difficulty annotation, the year the issue was
  created, the number of changed lines in the reference patch and the number of
  FAIL_TO_PASS tests.

Usage:

    python examples/swebench_verified/fetch.py OUT_DIR [SUBMISSION ...]

SUBMISSION is a folder name under ``evaluation/verified`` of the SWE-bench/experiments
repository; the default list is the six used in the README. Files are read at a fixed commit.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

EXPERIMENTS_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/{commit}/evaluation/verified/{name}/per_instance_details.json"
ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Verified"
    "&config=default&split=test&offset={offset}&length=100"
)
DEFAULT = [
    "20260217_mini-v2.0.0_claude-4-5-haiku-high",
    "20260217_mini-v2.0.0_deepseek-3-2-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_glm-5-high",
    "20260217_mini-v2.0.0_gpt-5-mini",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
]


def _get(url: str, cache: Path) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "slicescan-example"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(body)
    return body


def changed_lines(patch: str) -> int:
    """Added plus removed lines of a unified diff, not counting the file headers."""
    count = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count


def fetch(out: Path, submissions: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(0, 500, 100):
        page = json.loads(_get(ROWS.format(offset=offset), out / "hf" / f"rows_{offset}.json"))
        rows += [r["row"] for r in page["rows"]]
    with (out / "meta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "repo", "difficulty", "year", "patch_lines", "f2p_tests"])
        for r in rows:
            writer.writerow(
                [
                    r["instance_id"],
                    r["repo"],
                    r["difficulty"],
                    r["created_at"][:4],
                    changed_lines(r["patch"]),
                    len(json.loads(r["FAIL_TO_PASS"])),
                ]
            )
    ids = {r["instance_id"] for r in rows}
    with (out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for name in submissions:
            url = RAW.format(commit=EXPERIMENTS_COMMIT, name=name)
            details = json.loads(_get(url, out / "raw" / f"{name}.per_instance_details.json"))
            for instance, info in sorted(details.items()):
                if instance in ids:
                    score = 1 if info.get("resolved") is True else 0
                    handle.write(json.dumps({"item": instance, "model": name, "score": score}))
                    handle.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    fetch(Path(sys.argv[1]), sys.argv[2:] or DEFAULT)
