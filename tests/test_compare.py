import random

import pytest
from statsmodels.stats.contingency_tables import mcnemar

from slicescan.compare import compare
from slicescan.data import build_dataset
from slicescan.reversals import reversals


def _paired(seed: int = 6, n: int = 400):
    """B beats A overall, but A beats B on the area 'docs'."""
    rng = random.Random(seed)
    results, meta = [], []
    for i in range(n):
        item = f"p{i:03d}"
        area = ["core", "docs", "cli", "web"][i % 4]
        rate_a, rate_b = (0.7, 0.3) if area == "docs" else (0.4, 0.65)
        results.append({"item": item, "model": "A", "score": int(rng.random() < rate_a)})
        results.append({"item": item, "model": "B", "score": int(rng.random() < rate_b)})
        meta.append({"item": item, "area": area, "size": rng.randint(1, 900)})
    return build_dataset(results, meta)


def test_paired_rows_match_statsmodels_mcnemar():
    ds = _paired()
    cmp = compare(ds, "A", "B", ["area"])
    a, b = ds.scores["A"], ds.scores["B"]
    for row in cmp.rows:
        items = [i for i, m in ds.meta.items() if f"area={m['area']}" == row.label]
        both = sum(1 for i in items if a[i] == 1 and b[i] == 1)
        neither = sum(1 for i in items if a[i] == 0 and b[i] == 0)
        ref = mcnemar([[both, row.a_only], [row.b_only, neither]], exact=True).pvalue
        assert row.p_value == pytest.approx(ref, rel=1e-9)
        assert row.diff == pytest.approx(row.score_a - row.score_b)
        assert row.low <= row.high
    assert cmp.family.size == 4


def test_sign_flip_found():
    cmp = compare(_paired(), "A", "B", ["area", "size"])
    assert cmp.overall.diff < 0
    assert [row.label for row in cmp.flips] == ["area=docs"]
    flip = cmp.flips[0]
    assert flip.diff > 0 and flip.p_holm < 0.05


def test_two_way_compare_and_errors():
    ds = _paired()
    cmp = compare(ds, "A", "B", ["area", "size"], two_way=True, two_way_cap=3)
    assert cmp.family.size == 4 + 4 + 3
    assert "cap 3" in cmp.family.description
    with pytest.raises(ValueError):
        compare(ds, "A", "A")


def test_real_valued_paired_compare():
    rng = random.Random(1)
    results, meta = [], []
    for i in range(80):
        x = rng.random()
        results.append({"item": f"r{i}", "model": "A", "score": x})
        results.append({"item": f"r{i}", "model": "B", "score": x - 0.1})
        meta.append({"item": f"r{i}", "half": "one" if i < 40 else "two"})
    cmp = compare(build_dataset(results, meta), "A", "B", ["half"], resamples=200)
    assert not cmp.binary and cmp.overall.a_only is None
    assert cmp.overall.diff == pytest.approx(0.1)
    assert cmp.overall.p_value == pytest.approx(1 / 201)


def _kidney_stones():
    """The textbook case: A is better on small and on large stones, B is better overall."""
    results, meta = [], []

    def add(model, size, n, k):
        for j in range(n):
            item = f"{model}-{size}-{j}"
            results.append({"item": item, "model": model, "score": int(j < k)})
            meta.append({"item": item, "stone": size})

    add("A", "small", 87, 81)
    add("A", "large", 263, 192)
    add("B", "small", 270, 234)
    add("B", "large", 80, 55)
    return build_dataset(results, meta)


def test_constructed_reversal_is_detected():
    report = reversals(_kidney_stones(), "A", "B", ["stone"])
    check = report.checks[0]
    assert check.winner == "b"
    assert check.overall_a == pytest.approx(273 / 350)
    assert check.overall_b == pytest.approx(289 / 350)
    assert check.losses == 2 and check.wins == 0
    assert check.verdict == "full"
    assert report.reversals == [check]
    assert not check.same_items


def test_paired_data_has_no_full_reversal():
    report = reversals(_paired(), "A", "B", ["area", "size"])
    for check in report.checks:
        assert check.same_items and check.verdict != "full"
    assert report.shared == 400


def test_tag_columns_are_skipped_and_missing_kept():
    results = [
        {"item": "x", "model": "A", "score": 1},
        {"item": "x", "model": "B", "score": 0},
        {"item": "y", "model": "A", "score": 0},
        {"item": "y", "model": "B", "score": 1},
    ]
    meta = [{"item": "x", "tags": ["t", "u"], "g": "k"}, {"item": "y", "tags": ["t"]}]
    report = reversals(build_dataset(results, meta), "A", "B")
    assert report.skipped == ["tags"]
    labels = [p.label for p in report.checks[0].slices]
    assert labels == ["g=k", "g=(missing)"]
    assert report.checks[0].winner == "tie" and report.checks[0].verdict == "none"
