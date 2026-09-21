"""Slow, obviously-correct per-series (pandas rolling) implementation of the history features.

Used only by tests to prove that the vectorised production implementation is identical.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ml.src.config import LAGS, MIN_LAG, ROLLING_WINDOWS, STD_WINDOWS

MAX_DAYS_SINCE_SALE = 90


def reference_history_block(q: np.ndarray, price_obs: np.ndarray, promo_flag: np.ndarray) -> dict:
    """History features for one contiguous item-store series; every input is shifted >= MIN_LAG."""
    n = len(q)
    out: dict[str, np.ndarray] = {}
    qs = pd.Series(q)
    for k in LAGS:
        out[f"lag_{k}"] = qs.shift(k).to_numpy()
    s = qs.shift(MIN_LAG)
    for w in ROLLING_WINDOWS:
        out[f"rolling_mean_{w}"] = s.rolling(w, min_periods=(w + 1) // 2).mean().to_numpy()
    for w in STD_WINDOWS:
        out[f"rolling_std_{w}"] = s.rolling(w, min_periods=(w + 1) // 2).std().to_numpy()
    lag_stack = np.column_stack([out[f"lag_{k}"] for k in LAGS])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Mean of empty slice" on cold-start rows
        out["dow_mean_4w"] = np.nanmean(lag_stack, axis=1)
        out["ratio_mean_7_28"] = out["rolling_mean_7"] / np.where(
            out["rolling_mean_28"] > 0, out["rolling_mean_28"], np.nan
        )
    nz = pd.Series(np.where(np.isnan(q), np.nan, (q > 0).astype(float))).shift(MIN_LAG)
    out["sale_rate_28"] = nz.rolling(28, min_periods=14).mean().to_numpy()

    shifted = s.to_numpy()
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(shifted > 0, idx, -1))
    dsl = np.where(last >= 0, idx - last, np.nan).astype(float)
    # Cap ("not sold within the window") so short serving histories stay consistent with training.
    dsl = np.where(dsl > MAX_DAYS_SINCE_SALE, MAX_DAYS_SINCE_SALE, dsl)
    dsl = np.where((last < 0) & ~np.isnan(shifted), MAX_DAYS_SINCE_SALE, dsl)
    out["days_since_last_sale"] = dsl

    out["last_price"] = pd.Series(price_obs).ffill().shift(MIN_LAG).to_numpy()
    out["promo_days_28"] = (
        pd.Series(promo_flag.astype(float))
        .shift(MIN_LAG)
        .rolling(28, min_periods=1)
        .sum()
        .to_numpy()
    )
    return out
