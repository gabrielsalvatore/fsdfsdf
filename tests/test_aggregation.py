"""Aggregation, group weights, and the aggregate-first identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pie.aggregation import basket_effect, group_averaging_matrix
from pie.groups import income_quintile, label_groups
from pie.indexes import household_inflation
from pie.weights import democratic_shares, plutocratic_shares, weighted_quantile


def test_averaging_matrix_by_hand() -> None:
    levels, a = group_averaging_matrix(np.array(["b", "a", "b"]), np.array([1.0, 2.0, 3.0]))
    assert levels == ["a", "b"]
    np.testing.assert_allclose(a, [[0, 1, 0], [0.25, 0, 0.75]])


def test_aggregate_first_matches_household_first(rng: np.random.Generator) -> None:
    """Group mean of household Laspeyres = Laspeyres on the group's mean shares.

    This is the linear case where aggregating first is exact: (A W) R = A (W R).
    """
    h, i, t = 50, 7, 12
    shares = rng.dirichlet(np.ones(i), size=h)
    relatives = rng.uniform(0.9, 1.3, size=(i, t))
    labels = rng.choice(["x", "y", "z"], size=h)
    weights = rng.uniform(100, 5000, size=h)
    _, a = group_averaging_matrix(labels, weights)
    np.testing.assert_allclose(
        household_inflation(a @ shares, relatives), a @ household_inflation(shares, relatives)
    )


def test_basket_effect_by_hand() -> None:
    shares = np.array([[0.8, 0.2], [0.2, 0.8]])
    in_a, in_b = np.array([True, False]), np.array([False, True])
    sa, sb, contrib = basket_effect(shares, in_a, in_b, np.ones(2), np.array([1.5, 1.0]))
    np.testing.assert_allclose(sa, [0.8, 0.2])
    np.testing.assert_allclose(sb, [0.2, 0.8])
    np.testing.assert_allclose(contrib, [0.3, 0.0])


def test_basket_effect_sums_to_gap(rng: np.random.Generator) -> None:
    shares = rng.dirichlet(np.ones(5), size=40)
    rel = rng.uniform(0.95, 1.6, 5)
    w = rng.uniform(1, 10, 40)
    in_a = np.arange(40) < 15
    _, _, contrib = basket_effect(shares, in_a, ~in_a, w, rel)
    pi = household_inflation(shares, rel)
    gap = np.average(pi[in_a], weights=w[in_a]) - np.average(pi[~in_a], weights=w[~in_a])
    assert contrib.sum() == pytest.approx(gap)


def test_plutocratic_vs_democratic_shares() -> None:
    shares = np.array([[1.0, 0.0], [0.0, 1.0]])
    pop = np.array([1.0, 1.0])
    np.testing.assert_allclose(democratic_shares(shares, pop), [0.5, 0.5])
    # The second household spends three times as much, so it gets 3/4 of the vote.
    np.testing.assert_allclose(
        plutocratic_shares(shares, pop, np.array([100.0, 300.0])), [0.25, 0.75]
    )


def test_weighted_quantile_by_hand() -> None:
    x = np.array([3.0, 1.0, 2.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert weighted_quantile(x, w, 0.5) == 2.0
    assert weighted_quantile(x, w, 0.9) == 4.0
    assert weighted_quantile(x, np.array([0.0, 0.0, 0.0, 1.0]), 0.1) == 4.0


def households(n: int, **cols: object) -> pd.DataFrame:
    base = {
        "FINCBTXM": np.arange(1, n + 1) * 10_000.0,
        "FAM_SIZE": 1,
        "FINLWT21": 1.0,
        "BLS_URBN": 1,
        "FAM_TYPE": 1,
        "SEX_REF": 1,
        "AGE_REF": 40,
        "CUTENURE": 1,
    }
    base.update(cols)
    return pd.DataFrame(base, index=range(100, 100 + n))


def test_income_quintiles_split_evenly() -> None:
    q = income_quintile(households(10))
    assert q.value_counts().to_dict() == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert q.loc[100] == 1 and q.loc[109] == 5


def test_quintiles_use_equivalized_income() -> None:
    # Same income, but a family of four is poorer per person than a single.
    hh = households(2, FINCBTXM=[40_000.0, 40_000.0], FAM_SIZE=[4, 1])
    q = income_quintile(hh)
    assert q.loc[100] < q.loc[101]


def test_household_type_labels() -> None:
    hh = households(5, FAM_TYPE=[1, 7, 6, 8, 8], SEX_REF=[1, 2, 1, 2, 1])
    assert list(label_groups(hh)["Household type"]) == [
        "Married couple",
        "Single mother",
        "Single father",
        "Single woman",
        "Single man",
    ]


def test_area_age_and_tenure_labels() -> None:
    hh = households(3, BLS_URBN=[1, 2, 1], AGE_REF=[25, 50, 80], CUTENURE=[1, 2, 4])
    g = label_groups(hh)
    assert list(g["Area"]) == ["Urban", "Rural", "Urban"]
    assert list(g["Age of reference person"]) == ["Under 35", "35-64", "65+"]
    assert list(g["Housing tenure"]) == ["Owner, mortgage", "Owner, no mortgage", "Renter"]
