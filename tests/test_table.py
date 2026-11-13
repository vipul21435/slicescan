import random

import pytest
from scipy import stats as sps

from slicescan.data import build_dataset
from slicescan.table import slice_table


def _planted(seed: int = 11, n: int = 400):
    """Six repositories at 60 % solved, one of them planted at 20 %."""
    rng = random.Random(seed)
    results, meta = [], []
    for i in range(n):
        item = f"item-{i:04d}"
        repo = f"r{i % 6}"
        rate = 0.2 if repo == "r3" else 0.6
        results.append({"item": item, "model": "m", "score": int(rng.random() < rate)})
        meta.append({"item": item, "repo": repo, "length": rng.randint(1, 500)})
    return build_dataset(results, meta)


def test_planted_bad_slice_is_found():
    ds = _planted()
    table = slice_table(ds, "m", ["repo", "length"], min_size=10)
    found = {row.label: row for row in table.significant(0.05, "holm")}
    assert "repo=r3" in found
    # The planted slice also pulls "the rest" down for every other repository, so a healthy one
    # can come out above the rest; nothing may come out below it except the planted slice.
    assert all(row.diff > 0 for label, row in found.items() if label != "repo=r3")
    worst = table.rows[0]
    assert worst.label == "repo=r3" and worst.diff < 0
    assert worst.low <= worst.score <= worst.high
    assert table.family.size == 6 + 4


def test_row_matches_scipy_fisher():
    ds = _planted(seed=3)
    table = slice_table(ds, "m", ["repo"])
    assert table.binary and table.test == "fisher exact"
    for row in table.rows:
        a, b = row.successes, row.n - row.successes
        rest_succ = round(row.rest_score * row.rest_n)
        c, d = rest_succ, row.rest_n - rest_succ
        expected = sps.fisher_exact([[a, b], [c, d]]).pvalue
        assert row.p_value == pytest.approx(expected, rel=1e-9)
        assert row.n + row.rest_n == table.n_items


def test_min_size_and_whole_slices_leave_the_family():
    results = [{"item": f"i{k}", "model": "m", "score": k % 2} for k in range(30)]
    meta = [{"item": f"i{k}", "g": "big" if k < 25 else "small", "all": "x"} for k in range(30)]
    table = slice_table(build_dataset(results, meta), "m", ["g", "all"], min_size=10)
    assert [row.label for row in table.rows] == ["g=big"]
    assert table.skipped_small == 1 and table.skipped_whole == 1
    assert table.family.size == 1


def test_bad_arguments():
    ds = _planted()
    with pytest.raises(ValueError):
        slice_table(ds, "m", ["repo"], min_size=0)
    with pytest.raises(KeyError):
        slice_table(ds, "nobody", ["repo"])


def test_two_way_slices_join_the_family():
    ds = _planted()
    one = slice_table(ds, "m", ["repo", "length"], min_size=10)
    both = slice_table(ds, "m", ["repo", "length"], min_size=10, two_way=True, two_way_cap=5)
    assert both.family.size == one.family.size + 5
    assert "two-way" in both.family.description and "cap 5" in both.family.description
    two = [row for row in both.rows if len(row.columns) == 2]
    assert len(two) == 5 and all(row.n >= 10 for row in two)
    assert all(row.shrunk is None for row in two)


def test_shrunk_column_and_ranking():
    ds = _planted()
    table = slice_table(ds, "m", ["repo"], min_size=10)
    assert set(table.rankings) == {"repo"}
    r3 = table.find("repo=r3")
    assert r3.shrunk is not None and r3.score < r3.shrunk < table.score
    with pytest.raises(KeyError):
        table.find("repo=nothing")


def test_real_valued_scores_use_bootstrap_and_permutation():
    rng = random.Random(2)
    results, meta = [], []
    for i in range(120):
        group = "slow" if i < 30 else "fast"
        base = 0.3 if group == "slow" else 0.6
        results.append({"item": f"q{i}", "model": "m", "score": base + rng.uniform(-0.1, 0.1)})
        meta.append({"item": f"q{i}", "speed": group})
    table = slice_table(build_dataset(results, meta), "m", ["speed"], resamples=300)
    assert not table.binary and table.successes is None
    assert table.interval == "percentile bootstrap" and not table.priors
    slow = table.find("speed=slow")
    assert slow.low < slow.score < slow.high
    assert slow.p_value == pytest.approx(1 / 301)
    assert slow.diff == pytest.approx(-0.3, abs=0.05)


def test_false_discoveries_under_a_global_null():
    """No slice differs; BH and Holm should flag something in at most ~5 % of datasets."""
    reps, any_bh, any_holm, any_raw = 150, 0, 0, 0
    for rep in range(reps):
        rng = random.Random(1000 + rep)
        results, meta = [], []
        for i in range(300):
            item = f"n{i}"
            results.append({"item": item, "model": "m", "score": int(rng.random() < 0.5)})
            meta.append({"item": item, "a": rng.choice("abcdef"), "b": rng.choice("pqrs"),
                         "c": rng.random()})
        table = slice_table(build_dataset(results, meta), "m", ["a", "b", "c"], min_size=10)
        any_bh += bool(table.significant(0.05, "bh"))
        any_holm += bool(table.significant(0.05, "holm"))
        any_raw += bool(table.significant(0.05, "raw"))
    # Under the global null the false discovery rate equals the chance of any discovery.
    assert any_bh / reps <= 0.08
    assert any_holm / reps <= 0.08
    # Unadjusted, 14 tests at 5 % flag something far more often.
    assert any_raw / reps > 0.2
