import json
import random

import numpy as np
import pytest

from slicescan.data import build_dataset, load_dataset, parse_score, read_rows
from slicescan.slices import (
    bin_index,
    bin_labels,
    detect_kind,
    format_number,
    one_way_slices,
    quantile,
    quantile_edges,
    slice_column,
)


def test_quantile_matches_numpy():
    rng = random.Random(4)
    for n in [1, 2, 7, 50]:
        xs = sorted(rng.uniform(-5, 5) for _ in range(n))
        for q in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0]:
            assert quantile(xs, q) == pytest.approx(float(np.quantile(xs, q)), abs=1e-12)


def test_quantile_edges_merge_ties():
    edges = quantile_edges([1, 1, 1, 1, 1, 1, 2, 3], 4)
    assert edges == pytest.approx([1.0, 1.25, 3.0])
    assert quantile_edges([5, 5, 5], 4) == [5]


def test_bins_cover_every_value_once():
    rng = random.Random(5)
    values = [rng.randint(0, 30) for _ in range(200)]
    edges = quantile_edges(values, 4)
    labels = bin_labels(edges)
    assert labels[-1].endswith("]") and labels[0].endswith(")")
    counts = [0] * len(labels)
    for v in values:
        counts[bin_index(edges, v)] += 1
    assert sum(counts) == len(values)
    assert min(values) >= edges[0] and max(values) <= edges[-1]


def test_detect_kind():
    assert detect_kind(["a", "b", None], 4) == "category"
    assert detect_kind([1, 2, 3], 4) == "category"  # few distinct numbers read as categories
    assert detect_kind([str(i) for i in range(10)], 4) == "numeric"
    assert detect_kind([["x"], ["x", "y"]], 4) == "tag"
    assert detect_kind([True, False, True, False, True, False], 1) == "category"


def test_format_number():
    assert format_number(3.0) == "3"
    assert format_number(2.5) == "2.5"
    assert format_number(1 / 3) == "0.3333"


def test_category_tag_and_numeric_slices():
    meta = {
        "a": {"repo": "x", "tags": ["io", "net"], "size": 1},
        "b": {"repo": "x", "tags": "io", "size": 2},
        "c": {"repo": "y", "tags": "net|db", "size": 3},
        "d": {"repo": "y", "size": 4},
        "e": {"size": 5},
    }
    items = list(meta)
    repo = slice_column(meta, items, "repo")
    assert repo.kind == "category" and repo.missing == 1
    assert {s.label: set(s.items) for s in repo.slices} == {
        "repo=x": {"a", "b"},
        "repo=y": {"c", "d"},
    }
    tags = slice_column(meta, items, "tags", kind="tag")
    assert {s.values[0]: set(s.items) for s in tags.slices} == {
        "db": {"c"},
        "io": {"a", "b"},
        "net": {"a", "c"},
    }
    assert tags.notes and "overlap" in tags.notes[0]
    size = slice_column(meta, items, "size", bins=2)
    assert size.kind == "numeric"
    assert [s.values[0] for s in size.slices] == ["[1, 3)", "[3, 5]"]
    assert [len(s.items) for s in size.slices] == [2, 3]


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        slice_column({"a": {"c": 1}}, ["a"], "c", kind="ordinal")


def test_one_way_slices_respects_kinds():
    meta = {"a": {"t": "p|q"}, "b": {"t": "q"}}
    cols = one_way_slices(meta, ["a", "b"], ["t"], kinds={"t": "tag"})
    assert [s.label for s in cols[0].slices] == ["t=p", "t=q"]


def test_parse_score():
    assert parse_score(True) == 1.0
    assert parse_score("false") == 0.0
    assert parse_score("0.25") == pytest.approx(0.25)
    with pytest.raises(ValueError):
        parse_score("nan")
    with pytest.raises(ValueError):
        parse_score([1])


def test_build_dataset_joins_and_checks():
    results = [
        {"item": "a", "model": "m", "score": 1, "lang": "py"},
        {"item": "b", "model": "m", "score": 0},
    ]
    meta = [{"item": "a", "repo": "x"}, {"item": "b", "repo": "y", "lang": "go"}]
    ds = build_dataset(results, meta)
    assert ds.models == ["m"]
    assert ds.meta["a"] == {"lang": "py", "repo": "x"}
    assert ds.meta_columns() == ["lang", "repo"]
    with pytest.raises(ValueError, match="scored twice"):
        build_dataset(results + [results[0]])
    with pytest.raises(ValueError, match="two values"):
        build_dataset(results, [{"item": "a", "lang": "rs"}])
    with pytest.raises(ValueError, match="no 'score'"):
        build_dataset([{"item": "a", "model": "m"}])
    with pytest.raises(KeyError):
        ds.model_scores("other")


def test_read_rows_jsonl_and_csv(tmp_path):
    jl = tmp_path / "r.jsonl"
    jl.write_text(json.dumps({"item": "a", "model": "m", "score": 1}) + "\n\n")
    csvf = tmp_path / "m.csv"
    csvf.write_text("item,repo\na,x\n")
    ds = load_dataset(jl, csvf)
    assert ds.scores == {"m": {"a": 1.0}} and ds.meta == {"a": {"repo": "x"}}
    bad = tmp_path / "r.txt"
    bad.write_text("")
    with pytest.raises(ValueError, match="unsupported"):
        read_rows(bad)
    notobj = tmp_path / "n.jsonl"
    notobj.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="JSON object"):
        read_rows(notobj)


def test_two_way_slices_cap_keeps_largest():
    from slicescan.slices import two_way_slices

    meta = {f"i{k}": {"a": "x" if k < 30 else "y", "b": "p" if k % 3 else "q"} for k in range(40)}
    cols = one_way_slices(meta, list(meta), ["a", "b"])
    all_two = two_way_slices(cols, 40, min_size=1, cap=None)
    assert all_two.candidates == 4 and all_two.dropped == 0
    sizes = {s.label: s.size for s in all_two.slices}
    assert sizes == {"a=x & b=p": 20, "a=x & b=q": 10, "a=y & b=p": 6, "a=y & b=q": 4}
    capped = two_way_slices(cols, 40, min_size=5, cap=2)
    assert [s.label for s in capped.slices] == ["a=x & b=p", "a=x & b=q"]
    assert capped.candidates == 3 and capped.dropped == 1
    with pytest.raises(ValueError):
        two_way_slices(cols, 40, min_size=1, pairs=[("a", "a")])
    with pytest.raises(ValueError):
        two_way_slices(cols, 40, min_size=1, cap=-1)
