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


ARGS = [str(DATA / "results.jsonl"), "--meta", str(DATA / "meta.csv"), "--kind", "tags=tag"]


def test_compare_reversals_explain(capsys):
    main(["compare", *ARGS, "alpha", "beta", "--two-way", "--two-way-cap", "5"])
    out = capsys.readouterr().out
    assert "`alpha` minus `beta`" in out and "Sign flips" in out and "two-way" in out
    main(["reversals", *ARGS, "alpha", "beta"])
    out = capsys.readouterr().out
    assert "Not checked (tag columns" in out and "Reversals found: none." in out
    main(["explain", *ARGS, "--model", "alpha", "--slice", "project=kiwi"])
    out = capsys.readouterr().out
    assert "ranks 1 by p" in out and "Inside the slice, by `lines`" in out


def test_explain_untested_slice_and_json(capsys):
    label = "project=kiwi & tags=io"
    main(["explain", *ARGS, "--model", "alpha", "--slice", label, "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["tested"] is False and data["row"]["p_holm"] is None
    main(["explain", *ARGS, "--model", "alpha", "--slice", label])
    assert "Not in the tested family" in capsys.readouterr().out
    with pytest.raises(KeyError):
        main(["explain", *ARGS, "--model", "alpha", "--slice", "project=none"])
    with pytest.raises(ValueError):
        main(["explain", *ARGS, "--model", "alpha", "--slice", "kiwi"])


def test_json_outputs_parse(capsys):
    for cmd in (["compare", *ARGS, "alpha", "beta"], ["reversals", *ARGS, "alpha", "beta"]):
        main([*cmd, "--format", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["model_a"] == "alpha"
    main(["table", *ARGS, "--model", "alpha", "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert "project" in data["shrinkage"]


def test_rank_and_chart(tmp_path, capsys):
    png = tmp_path / "chart.png"
    main(["table", *ARGS, "--model", "alpha", "--rank", "project", "--chart", str(png)])
    out = capsys.readouterr().out
    assert "Raw and shrunk ranking of `project`" in out
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_real_scores_through_the_cli(tmp_path, capsys):
    rows = [
        json.dumps({"item": f"i{k}", "model": m, "score": (k % 7) / 7 + (0.1 if m == "x" else 0)})
        for k in range(40)
        for m in ("x", "y")
    ]
    res = tmp_path / "r.jsonl"
    res.write_text("\n".join(rows) + "\n")
    meta = tmp_path / "m.csv"
    meta.write_text("item,g\n" + "".join(f"i{k},{'a' if k < 20 else 'b'}\n" for k in range(40)))
    main(["table", str(res), "--meta", str(meta), "--model", "x", "--resamples", "100"])
    assert "mean " in capsys.readouterr().out
    main(["compare", str(res), "--meta", str(meta), "x", "y", "--resamples", "100"])
    out = capsys.readouterr().out
    assert "sign-flip" in out and "A only" not in out
    with pytest.raises(KeyError):
        main(["table", str(res), "--meta", str(meta), "--model", "x", "--rank", "g"])


def test_explain_two_way_in_either_order(capsys):
    for label in ("project=kiwi & tags=io", "tags=io & project=kiwi"):
        main(
            [
                "explain",
                *ARGS,
                "--model",
                "alpha",
                "--slice",
                label,
                "--two-way",
                "--min-size",
                "5",
                "--format",
                "json",
            ]
        )
        data = json.loads(capsys.readouterr().out)
        assert data["tested"] is True and data["row"]["n"] == 14
