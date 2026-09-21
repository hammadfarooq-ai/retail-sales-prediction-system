"""Forecast accuracy metrics. All functions take array-likes and return plain floats.

Zero handling (daily item-store demand contains many true zeros):
  * MAPE is computed on rows with actual > 0 only (undefined otherwise) and reported alongside
    the share of rows it covers.
  * SMAPE treats 0/0 as a perfect forecast (error 0).
  * WAPE (= sum|e| / sum|y|) is the robust headline percentage error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def mae(y, p) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(p)) ** 2)))


def mape(y, p) -> float:
    y, p = _arr(y), _arr(p)
    m = y > 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan")


def smape(y, p) -> float:
    y, p = _arr(y), _arr(p)
    denom = np.abs(y) + np.abs(p)
    with np.errstate(all="ignore"):
        v = np.where(denom == 0, 0.0, 2 * np.abs(y - p) / denom)
    return float(np.mean(v) * 100)


def wape(y, p) -> float:
    y, p = _arr(y), _arr(p)
    s = np.abs(y).sum()
    return float(np.abs(y - p).sum() / s * 100) if s > 0 else float("nan")


def r2(y, p) -> float:
    y, p = _arr(y), _arr(p)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def bias(y, p) -> float:
    """Mean signed error (prediction - actual); positive = over-forecasting."""
    return float(np.mean(_arr(p) - _arr(y)))


def all_metrics(y, p) -> dict[str, float]:
    y = _arr(y)
    return {
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "mape": mape(y, p),
        "smape": smape(y, p),
        "wape": wape(y, p),
        "r2": r2(y, p),
        "bias": bias(y, p),
        "n_rows": int(len(y)),
        "mape_coverage_pct": float((y > 0).mean() * 100),
    }


def aggregated_metrics(df: pd.DataFrame, by: list[str], y: str = "actual", p: str = "predicted"):
    """Metrics after summing item-level forecasts to a coarser level (e.g. store x day)."""
    g = df.groupby(by, observed=True)[[y, p]].sum()
    return all_metrics(g[y], g[p])
