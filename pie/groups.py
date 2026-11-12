"""Assign households to the comparison groups (area, income, household type, ...)."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_QUINTILES = 5


def income_quintile(hh: pd.DataFrame) -> pd.Series:
    """Population-weighted quintile (1-5) of income per sqrt(family size)."""
    eq_income = hh["FINCBTXM"] / np.sqrt(hh["FAM_SIZE"].clip(lower=1))
    order = eq_income.sort_values().index
    cum = hh.loc[order, "FINLWT21"].cumsum() / hh["FINLWT21"].sum()
    # A household belongs to quintile k if the weight up to and including it is
    # within the first k/5 of the population. The tolerance keeps a household that
    # lands exactly on a boundary (cum = 0.2) in the lower quintile despite
    # floating-point round-off.
    q = np.clip(np.ceil(cum * N_QUINTILES - 1e-9), 1, N_QUINTILES).astype(int)
    return q.reindex(hh.index)


def _quintile_label(q: int) -> str:
    suffix = " (lowest)" if q == 1 else " (highest)" if q == N_QUINTILES else ""
    return f"Q{q}{suffix}"


def label_groups(hh: pd.DataFrame) -> pd.DataFrame:
    """One column per grouping dimension, one row per household."""
    g = pd.DataFrame(index=hh.index)
    g["Area"] = hh["BLS_URBN"].map({1: "Urban", 2: "Rural"})
    g["Income quintile"] = income_quintile(hh).map(_quintile_label)
    ft = hh["FAM_TYPE"]
    sex = hh["SEX_REF"]
    g["Household type"] = np.select(
        [ft.eq(7), ft.eq(6), ft.eq(8) & sex.eq(2), ft.eq(8) & sex.eq(1), ft.isin([1, 2, 3, 4, 5])],
        ["Single mother", "Single father", "Single woman", "Single man", "Married couple"],
        default=None,
    )
    g["Age of reference person"] = pd.cut(
        hh["AGE_REF"], [0, 34, 64, 200], labels=["Under 35", "35-64", "65+"]
    ).astype(str)
    g["Housing tenure"] = hh["CUTENURE"].map(
        {1: "Owner, mortgage", 2: "Owner, no mortgage", 4: "Renter"}
    )
    return g
