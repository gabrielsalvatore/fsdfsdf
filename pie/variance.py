"""Standard errors from the CE's balanced repeated replication (BRR) weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_REP = 44
# Keeps a replicate's weighted mean defined when every household in a small
# group has replicate weight 0 in that half-sample.
_EPS = 1e-12


def replicate_columns(n: int = N_REP) -> list[str]:
    return [f"WTREP{i:02d}" for i in range(1, n + 1)]


def brr_mean(values: np.ndarray, hh: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    """Weighted mean and its BRR standard error.

    BLS formula: var = sum_r (t_r - t)^2 / 44, where t uses FINLWT21 and t_r uses
    replicate weight r.
    """
    v = values[mask]
    est = float(np.average(v, weights=hh.loc[mask, "FINLWT21"]))
    reps = np.array(
        [
            np.average(v, weights=hh.loc[mask, col].clip(lower=0) + _EPS)
            for col in replicate_columns()
        ]
    )
    se = float(np.sqrt(((reps - est) ** 2).sum() / N_REP))
    return est, se
