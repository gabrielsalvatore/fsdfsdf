"""Price index formulas.

Two views of the same objects:

* Household indexes from spending shares and price relatives:
  Laspeyres inflation pi_h = sum_i w_hi * r_i - 1, computed for all households at
  once as W @ r.
* Bilateral indexes from prices and quantities (p0, p1, q0, q1). Passing
  quantities in place of prices, fisher(q0, q1, p0, p1), gives the matching
  quantity index, which the factor-reversal tests use.
"""

from __future__ import annotations

import numpy as np


def household_inflation(shares: np.ndarray, relatives: np.ndarray) -> np.ndarray:
    """Laspeyres inflation for each household: shares (H x I) @ relatives (I or I x T) - 1."""
    return shares @ relatives - 1


def laspeyres(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Cost of the base-period basket at new prices, relative to base prices."""
    return float(np.dot(p1, q0) / np.dot(p0, q0))


def paasche(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Cost of the current-period basket at new prices, relative to base prices."""
    return float(np.dot(p1, q1) / np.dot(p0, q1))


def fisher(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Geometric mean of Laspeyres and Paasche."""
    return float(np.sqrt(laspeyres(p0, p1, q0, q1) * paasche(p0, p1, q0, q1)))


def tornqvist(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Weighted geometric mean of price relatives, weights = average of the two periods' shares."""
    s0 = p0 * q0 / np.dot(p0, q0)
    s1 = p1 * q1 / np.dot(p1, q1)
    return float(np.exp(np.dot((s0 + s1) / 2, np.log(p1 / p0))))
