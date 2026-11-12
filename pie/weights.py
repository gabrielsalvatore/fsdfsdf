"""Population weights: weighted quantiles and CPI-style aggregate weights."""

from __future__ import annotations

import numpy as np


def weighted_quantile(x: np.ndarray, w: np.ndarray, p: float) -> float:
    """Smallest x such that the weight at or below it is more than share p of the total."""
    order = np.argsort(x)
    cum = np.cumsum(w[order]) / w.sum()
    return float(x[order][np.searchsorted(cum, p)])


def plutocratic_shares(shares: np.ndarray, pop_weight: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """Aggregate spending shares, weighting each household by its total spending.

    This is how the official CPI weights households: big spenders count more.
    """
    agg = (shares * (pop_weight * spend)[:, None]).sum(axis=0)
    return agg / agg.sum()


def democratic_shares(shares: np.ndarray, pop_weight: np.ndarray) -> np.ndarray:
    """Population-weighted mean of household spending shares (one household, one vote)."""
    return np.average(shares, axis=0, weights=pop_weight)
