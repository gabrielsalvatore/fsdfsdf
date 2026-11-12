"""Aggregate household results into group results."""

from __future__ import annotations

import numpy as np


def group_averaging_matrix(labels: np.ndarray, weights: np.ndarray) -> tuple[list, np.ndarray]:
    """Dense G x H matrix A whose row g averages households in group g with the given weights.

    Returns (levels, A). A @ X gives the weighted mean of X's rows within each group.
    """
    levels = sorted(set(labels))
    a = np.array([(labels == level) * weights for level in levels], dtype=float)
    return levels, a / a.sum(axis=1, keepdims=True)


def basket_effect(
    shares: np.ndarray,
    in_a: np.ndarray,
    in_b: np.ndarray,
    weights: np.ndarray,
    relatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-category contribution to group A's minus group B's mean inflation.

    Returns (shares_a, shares_b, contributions) with
    contributions_i = (s_ai - s_bi) * (r_i - 1); they sum to the gap in mean inflation.
    """
    sa = np.average(shares[in_a], axis=0, weights=weights[in_a])
    sb = np.average(shares[in_b], axis=0, weights=weights[in_b])
    return sa, sb, (sa - sb) * (relatives - 1)
