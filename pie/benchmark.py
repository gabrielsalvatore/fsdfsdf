"""Time algorithms that compute the same household and group inflation numbers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from pie.aggregation import group_averaging_matrix

T = TypeVar("T")


@dataclass
class BenchmarkResult:
    a1_seconds: float
    a2_seconds: float
    a4_seconds: float
    n_groups: int
    max_diff_a1_a2: float
    max_diff_a4_a2: float


def best_of(fn: Callable[[], T], n: int = 7) -> tuple[T, float]:
    """Warm up once, then return the result and the fastest of n timed runs."""
    res = fn()
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t)
    return res, best


def naive_loops(shares: np.ndarray, relatives: np.ndarray) -> np.ndarray:
    """A1: triple loop in pure Python."""
    w, r = shares.tolist(), relatives.tolist()
    n_items, n_months = len(r), len(r[0])
    return np.array(
        [[sum(row[i] * r[i][m] for i in range(n_items)) - 1 for m in range(n_months)] for row in w]
    )


def synthetic_relatives(n_items: int, n_months: int, seed: int = 0) -> np.ndarray:
    """Random-walk monthly price relatives (I x T)."""
    rng = np.random.default_rng(seed)
    return 1 + rng.normal(0.005, 0.01, size=(n_items, n_months)).cumsum(axis=1)


def run(
    shares: np.ndarray, labels: np.ndarray, weights: np.ndarray, n_months: int = 12
) -> BenchmarkResult:
    relatives = synthetic_relatives(shares.shape[1], n_months)

    t = time.perf_counter()
    out1 = naive_loops(shares, relatives)
    a1 = time.perf_counter() - t

    out2, a2 = best_of(lambda: shares @ relatives - 1)

    levels, a = group_averaging_matrix(labels, weights)
    g4, a4 = best_of(lambda: (a @ shares) @ relatives - 1)
    g2 = a @ out2

    return BenchmarkResult(
        a1_seconds=a1,
        a2_seconds=a2,
        a4_seconds=a4,
        n_groups=len(levels),
        max_diff_a1_a2=float(np.abs(out1 - out2).max()),
        max_diff_a4_a2=float(np.abs(g4 - g2).max()),
    )
