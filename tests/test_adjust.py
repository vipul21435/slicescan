import random

import pytest
from statsmodels.stats.multitest import multipletests

from slicescan.adjust import benjamini_hochberg, holm


def _random_pvalues(rng: random.Random, m: int) -> list[float]:
    values = [rng.random() ** 3 for _ in range(m)]
    # repeat a few values to exercise ties
    for _ in range(m // 4):
        values[rng.randrange(m)] = values[rng.randrange(m)]
    return values


@pytest.mark.parametrize("method,ours", [("holm", holm), ("fdr_bh", benjamini_hochberg)])
def test_matches_statsmodels(method, ours):
    rng = random.Random(3)
    for m in [1, 2, 5, 17, 60]:
        for _ in range(20):
            p = _random_pvalues(rng, m)
            ref = multipletests(p, method=method)[1]
            assert ours(p) == pytest.approx(list(ref), abs=1e-12)


def test_hand_worked_case():
    p = [0.01, 0.04, 0.03, 0.005]
    # Holm: sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # BH: m/rank * p = 0.02, 0.02, 0.04, 0.04 after the running minimum from the top
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_empty_family():
    assert holm([]) == []
    assert benjamini_hochberg([]) == []


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        holm([0.2, 1.3])
    with pytest.raises(ValueError):
        benjamini_hochberg([-0.1])
