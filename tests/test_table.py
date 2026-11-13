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
