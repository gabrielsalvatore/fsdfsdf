"""Load CE Interview household files (FMLI) and build household spending shares."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pie.categories import CATEGORIES, CE_VARS, QUARTER_SUFFIXES
from pie.variance import replicate_columns

SHELTER_VARS = [
    "RENDWECQ",
    "RENDWEPQ",
    "OTHLODCQ",
    "OTHLODPQ",
    "OWNDWECQ",
    "OWNDWEPQ",
    "RENTEQVX",
]
DEMOGRAPHIC_VARS = ["FAM_SIZE", "AGE_REF", "SEX_REF", "FAM_TYPE", "BLS_URBN", "CUTENURE", "REGION"]
OWNER_TENURE_CODES = (1, 2, 3)
MONTHS_PER_INTERVIEW = 3  # RENTEQVX is monthly; one interview covers 3 months


class MissingCEVariableError(KeyError):
    """Raised when a CE file lacks variables the pipeline needs."""

    def __init__(self, missing: Iterable[str]):
        self.missing = sorted(set(missing))
        super().__init__(f"{len(self.missing)} required CE variables not found: {self.missing}")

    def __str__(self) -> str:  # KeyError would otherwise wrap the message in quotes
        return str(self.args[0])


def spending_variables() -> list[str]:
    return [f"{s}{q}" for stems in CE_VARS.values() for s in stems for q in QUARTER_SUFFIXES]


def required_variables() -> list[str]:
    return (
        ["NEWID", "FINLWT21", "FINCBTXM"]
        + replicate_columns()
        + DEMOGRAPHIC_VARS
        + spending_variables()
        + SHELTER_VARS
    )


def check_required(df: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    required = required_variables() if required is None else required
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingCEVariableError(missing)


def read_fmli(root: str | Path) -> pd.DataFrame:
    """Concatenate every fmli*.csv under root, with upper-case column names."""
    files = sorted(Path(root).rglob("fmli*.csv"))
    if not files:
        raise FileNotFoundError(f"No fmli*.csv files under {root}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.upper() for c in df.columns]
    return df


def _amount(df: pd.DataFrame, name: str) -> pd.Series:
    """A spending variable, blank values as 0 and negative refunds clipped to 0."""
    return df[name].fillna(0).clip(lower=0)


def interview_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Spending per interview (row) for each CPI category."""
    check_required(df, spending_variables() + SHELTER_VARS + ["CUTENURE"])
    spend = pd.DataFrame(index=df.index)
    for cat, stems in CE_VARS.items():
        spend[cat] = sum(_amount(df, f"{s}{q}") for s in stems for q in QUARTER_SUFFIXES)

    # Shelter, valued the way the CPI does: rent for renters, rental
    # equivalence for owners (their own outlays if it is not reported),
    # plus other lodging for everyone.
    owner = df["CUTENURE"].isin(OWNER_TENURE_CODES)
    rent = _amount(df, "RENDWECQ") + _amount(df, "RENDWEPQ")
    other_lodging = _amount(df, "OTHLODCQ") + _amount(df, "OTHLODPQ")
    own_outlays = _amount(df, "OWNDWECQ") + _amount(df, "OWNDWEPQ")
    rent_eq = MONTHS_PER_INTERVIEW * _amount(df, "RENTEQVX")
    owner_cost = np.where(rent_eq > 0, rent_eq, own_outlays)
    spend["Shelter"] = np.where(owner, owner_cost, rent) + other_lodging
    return spend[CATEGORIES]


def build_households(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse interviews to households (CUID = NEWID // 10).

    Returns (hh, shares): household attributes and weights, and each household's
    spending shares over CATEGORIES (rows sum to 1). Households with no spending
    are dropped.
    """
    check_required(df)
    df = df.copy()
    df["CUID"] = df["NEWID"] // 10
    spend = interview_spending(df)
    spend["CUID"] = df["CUID"]
    hh_spend = spend.groupby("CUID").sum()

    rep_cols = replicate_columns()
    df[rep_cols] = df[rep_cols].fillna(0)  # blank = not in that BRR half-sample
    agg = df.groupby("CUID")[["FINLWT21", *rep_cols, "FINCBTXM"]].mean()
    latest = df.sort_values("NEWID").groupby("CUID")[DEMOGRAPHIC_VARS].last()
    hh = agg.join(latest)

    total = hh_spend.sum(axis=1)
    ok = total > 0
    shares = hh_spend[ok].div(total[ok], axis=0)
    hh = hh.loc[shares.index]
    hh["TOTAL_SPEND"] = total[ok]
    return hh, shares


def load_households(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read FMLI files under root. Returns (interviews, households, shares)."""
    df = read_fmli(root)
    hh, shares = build_households(df)
    return df, hh, shares
