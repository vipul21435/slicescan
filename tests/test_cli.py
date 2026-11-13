import json
from pathlib import Path

import pytest

from slicescan.cli import main

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def test_table_markdown(capsys):
    assert (
        main(
            [
                "table",
                str(DATA / "results.jsonl"),
                "--meta",
                str(DATA / "meta.csv"),
                "--model",
                "alpha",
                "--kind",
                "tags=tag",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "## Slices for `alpha`" in out
    assert "Family:" in out and "project=kiwi" in out
    assert "overlap" in out


def test_table_json(capsys):
    main(
        [
            "table",
            str(DATA / "results.jsonl"),
            "--meta",
            str(DATA / "meta.csv"),
            "--model",
            "alpha",
            "--format",
            "json",
            "--columns",
            "project",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "alpha"
    assert data["family"]["size"] == len(data["rows"]) == 6
    assert data["rows"][0]["label"] == "project=kiwi"


def test_cli_errors():
    with pytest.raises(SystemExit):
        main(["table", str(DATA / "results.jsonl"), "--meta", str(DATA / "meta.csv")])
    with pytest.raises(SystemExit):
        main(["table", str(DATA / "results.jsonl"), "--model", "alpha", "--kind", "tags"])
