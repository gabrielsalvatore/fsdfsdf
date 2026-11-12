"""Load CPI-U category indexes from a CSV file and turn them into price relatives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pie.categories import ALL_ITEMS

DEFAULT_CPI_PATH = Path(__file__).resolve().parent.parent / "data" / "cpi_u.csv"
CPI_COLUMNS = ("category", "series_id", "date", "value")


def load_cpi(path: str | Path = DEFAULT_CPI_PATH) -> pd.DataFrame:
    """Read a tidy CPI file (category, series_id, date, value).

    Returns a table with one row per category and one column per date ("YYYY-MM").
    """
    raw = pd.read_csv(path, dtype={"category": str, "series_id": str, "date": str})
    missing = [c for c in CPI_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"CPI file {path} is missing columns: {missing}")
    if raw.duplicated(["category", "date"]).any():
        dupes = raw.loc[raw.duplicated(["category", "date"]), ["category", "date"]]
        raise ValueError(f"CPI file {path} has duplicate rows: {dupes.to_dict('records')}")
    return raw.pivot(index="category", columns="date", values="value")


def price_relatives(cpi: pd.DataFrame, start: str, end: str, categories: list[str]) -> np.ndarray:
    """Price relative P_end / P_start for each category, in the order given."""
    for date in (start, end):
        if date not in cpi.columns:
            raise KeyError(f"CPI data has no observations for {date}")
    absent = [c for c in categories if c not in cpi.index]
    if absent:
        raise KeyError(f"CPI data has no series for: {absent}")
    sub = cpi.loc[categories, [start, end]]
    if sub.isna().any().any():
        raise ValueError(f"CPI data has gaps between {start} and {end} for some categories")
    return (sub[end] / sub[start]).to_numpy(dtype=float)


def all_items_inflation(cpi: pd.DataFrame, start: str, end: str) -> float:
    """Headline CPI-U inflation between two dates."""
    return float(price_relatives(cpi, start, end, [ALL_ITEMS])[0] - 1)
