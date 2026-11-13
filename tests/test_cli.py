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


def test_cli_errors(capsys):
    assert main(["table", str(DATA / "results.jsonl"), "--meta", str(DATA / "meta.csv")]) == 2
    assert "several models" in capsys.readouterr().err
    args = ["table", str(DATA / "results.jsonl"), "--model", "alpha", "--kind", "tags"]
    assert main(args) == 2
    assert "--kind wants" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["table"])


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
    assert main(["explain", *ARGS, "--model", "alpha", "--slice", "project=none"]) == 2
    assert main(["explain", *ARGS, "--model", "alpha", "--slice", "kiwi"]) == 2


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
    assert main(["table", str(res), "--meta", str(meta), "--model", "x", "--rank", "g"]) == 2
    assert "no shrinkage" in capsys.readouterr().err


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


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


GOOD_RESULTS = "".join(
    json.dumps({"item": f"i{k}", "model": m, "score": (k + j) % 2}) + "\n"
    for k in range(30)
    for j, m in enumerate(("m", "n"))
)
GOOD_META = "item,g,t\n" + "".join(f"i{k},{'a' if k < 15 else 'b'},x{k}\n" for k in range(30))


@pytest.mark.parametrize(
    ("results", "meta", "extra", "message"),
    [
        (None, GOOD_META, ["--model", "m"], "No such file"),
        ("r.txt:" + GOOD_RESULTS, GOOD_META, ["--model", "m"], "unsupported file type"),
        ('r.jsonl:{"item": "a", "model": "m", "score": 1}\n{oops\n', None, [], "r.jsonl:2"),
        ('r.jsonl:{"item": "a", "model": "m"}\n', None, [], "no 'score' value"),
        ('r.jsonl:{"item": "a", "model": "m", "score": "abc"}\n', None, [], "result row 1"),
        ('r.jsonl:{"item": "a", "model": "m", "score": "nan"}\n', None, [], "finite"),
        ('r.jsonl:{"item": "a", "model": "m", "score": "inf"}\n', None, [], "finite"),
        (
            'r.jsonl:{"item": "a", "model": "m", "score": 1}\n'
            '{"item": "a", "model": "m", "score": 0}\n',
            None,
            [],
            "scored twice",
        ),
        ('r.jsonl:{"item": "a", "model": "m", "score": 1}\n', None, [], "no metadata columns"),
        ("r.jsonl:", None, [], "no result rows"),
        ("r.csv:item,model,score\n", None, [], "no result rows"),
        (GOOD_RESULTS, GOOD_META, ["--model", "zz"], "model 'zz' not found"),
        (GOOD_RESULTS, GOOD_META, [], "several models"),
        (GOOD_RESULTS, GOOD_META, ["--model", "m", "--min-size", "0"], "min_size"),
        (GOOD_RESULTS, GOOD_META, ["--model", "m", "--bins", "0", "--columns", "t"], "bins"),
        (GOOD_RESULTS, GOOD_META, ["--model", "m", "--confidence", "1.5"], "confidence"),
        (GOOD_RESULTS, GOOD_META, ["--model", "m", "--kind", "g=numeric"], "numeric"),
        (GOOD_RESULTS, GOOD_META, ["--model", "m", "--kind", "g"], "--kind"),
        (GOOD_RESULTS, GOOD_META, ["--model", "m", "--columns", "nope"], "unknown column"),
        (GOOD_RESULTS, "item,g\nother,a\n", ["--model", "m"], "no scored item has metadata"),
        (GOOD_RESULTS, "item,g\ni1,a\ni1,b\n", ["--model", "m"], "two values"),
    ],
)
def test_table_user_errors(tmp_path, capsys, results, meta, extra, message):
    if results is None:
        rpath = str(tmp_path / "missing.jsonl")
    else:
        if results.startswith("r."):
            name, _, text = results.partition(":")
        else:
            name, text = "r.jsonl", results
        rpath = _write(tmp_path, name, text)
    args = ["table", rpath]
    if meta is not None:
        args += ["--meta", _write(tmp_path, "m.csv", meta)]
    assert main(args + extra) == 2
    err = capsys.readouterr().err
    assert err.startswith("slicescan: error: ") and message in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["compare", "m", "m"], "two different"),
        (["compare", "m", "zz"], "not found"),
        (["reversals", "m", "m"], "two different"),
        (["reversals", "m", "n", "--columns", "nope"], "unknown column"),
        (["compare", "m", "n", "--columns", "nope"], "unknown column"),
        (["explain", "--model", "m", "--slice", "g=zz"], "no slice g=zz"),
        (["explain", "--model", "m", "--slice", "nonsense"], "cannot read slice"),
        (
            ["explain", "--model", "m", "--slice", "t=x1", "--kind", "t=tag", "--min-size", "1"],
            None,
        ),
    ],
)
def test_other_commands_user_errors(tmp_path, capsys, argv, message):
    r = _write(tmp_path, "r.jsonl", GOOD_RESULTS)
    m = _write(tmp_path, "m.csv", GOOD_META)
    command, rest = argv[0], argv[1:]
    code = main([command, r, "--meta", m, *rest])
    err = capsys.readouterr().err
    if message is None:
        assert code == 0
    else:
        assert code == 2 and message in err


def test_explain_slice_holding_every_item(tmp_path, capsys):
    r = _write(tmp_path, "r.jsonl", GOOD_RESULTS)
    m = _write(tmp_path, "m.csv", "item,all\n" + "".join(f"i{k},yes\n" for k in range(30)))
    assert main(["explain", r, "--meta", m, "--model", "m", "--slice", "all=yes"]) == 2
    assert "both inside and outside" in capsys.readouterr().err
