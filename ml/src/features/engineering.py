"""Leakage-safe feature engineering, shared verbatim by training and the serving layer.

Forecast framing (direct, horizon H = MIN_LAG = 7 days)
    A forecast for day *t* is made at origin *t - 7*. Therefore:
      * every history-derived feature (lags, rolling stats, last price, days-since-sale, ...)
        is computed from data at or before day t-7  -> no target leakage;
      * "known in advance" covariates for day t (calendar, planned promotion) are used as-is.
    The realised price of day t (``price_obs``) is NEVER a feature: it only exists when a sale
    happened, so using it would leak the target. Price enters through the last observed price
    (lagged) and the *planned* promotional price.

The input frame must be sorted by (store_id, item_id, date) with a contiguous daily index per
pair; ``quantity`` may be NaN for days that are not yet observed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ml.src.config import LAGS, MIN_HISTORY_DAYS, MIN_LAG, ROLLING_WINDOWS, STD_WINDOWS

KEYS = ["store_id", "item_id"]

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "quarter",
    "is_weekend",
    "day_of_year",
]
LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLLING_MEAN_FEATURES = [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
ROLLING_STD_FEATURES = [f"rolling_std_{w}" for w in STD_WINDOWS]
HISTORY_EXTRA = ["dow_mean_4w", "ratio_mean_7_28", "sale_rate_28", "days_since_last_sale"]
PRICE_PROMO_FEATURES = [
    "last_price",
    "planned_price",
    "price_ratio",
    "promo_flag",
    "discount_depth",
    "promo_type",
    "promo_days_28",
]
STATIC_FEATURES = ["store_id", "store_area", "store_format_code", "dept_code", "class_code"]

FEATURE_COLUMNS: list[str] = (
    CALENDAR_FEATURES
    + LAG_FEATURES
    + ROLLING_MEAN_FEATURES
    + ROLLING_STD_FEATURES
    + HISTORY_EXTRA
    + PRICE_PROMO_FEATURES
    + STATIC_FEATURES
)
CATEGORICAL_FEATURES = ["store_id", "store_format_code", "dept_code"]
REQUIRED_INPUT_COLUMNS = [
    "store_id",
    "item_id",
    "date",
    "quantity",
    "price_obs",
    "promo_price",
    "promo_before_price",
    "promo_type",
    "dept_code",
    "class_code",
]
MAX_DAYS_SINCE_SALE = 90


def store_static_table(stores: pd.DataFrame) -> pd.DataFrame:
    """store_id -> (store_area, store_format_code). Format code is a stable sorted factorisation."""
    s = stores.sort_values("store_id").copy()
    fmt_codes = {f: i for i, f in enumerate(sorted(s["format"].unique()))}
    return pd.DataFrame(
        {
            "store_id": s["store_id"].astype("int8").to_numpy(),
            "store_area": s["area"].astype("float32").to_numpy(),
            "store_format_code": s["format"].map(fmt_codes).astype("int8").to_numpy(),
        }
    )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["day_of_week"] = d.dayofweek.astype("int8")
    df["day_of_month"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["week_of_year"] = d.isocalendar().week.astype("int16").to_numpy()
    df["quarter"] = d.quarter.astype("int8")
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["day_of_year"] = d.dayofyear.astype("int16")
    return df


def _shift(x: np.ndarray, k: int, pos: np.ndarray) -> np.ndarray:
    """Group-aware shift: value from k rows earlier inside the same series, else NaN."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    out[pos < k] = np.nan
    return out


def _window_stats(
    x: np.ndarray, start: np.ndarray, w: int, min_periods: int, *, want_std: bool = False
) -> np.ndarray:
    """Rolling mean (or sample std) over the last ``w`` rows of each series, ignoring NaN.

    Uses prefix sums so the whole panel is processed at once; identical semantics to
    ``Series.rolling(w, min_periods=min_periods)`` applied per series.
    """
    n = len(x)
    valid = ~np.isnan(x)
    idx = np.arange(n)
    lo = np.maximum(idx - w + 1, start)
    ccnt = np.concatenate([[0.0], np.cumsum(valid.astype("float64"))])
    cnt = ccnt[idx + 1] - ccnt[lo]
    with np.errstate(all="ignore"):
        if not want_std:
            # Uncentred prefix sums: a window of zeros sums to exactly 0 (matters for `mean > 0`).
            csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
            out = (csum[idx + 1] - csum[lo]) / cnt
            return np.where(cnt >= min_periods, out, np.nan)
        # Centre by the global mean to keep prefix sums of squares numerically small.
        centre = float(np.nanmean(x)) if valid.any() else 0.0
        xc = np.where(valid, x - centre, 0.0)
        csum = np.concatenate([[0.0], np.cumsum(xc)])
        csq = np.concatenate([[0.0], np.cumsum(xc * xc)])
        total = csum[idx + 1] - csum[lo]
        sq = csq[idx + 1] - csq[lo]
        var = (sq - total * total / cnt) / (cnt - 1)
        out = np.sqrt(np.clip(var, 0.0, None))
        return np.where((cnt >= min_periods) & (cnt >= 2), out, np.nan)


def _history_features(
    q: np.ndarray, price_obs: np.ndarray, promo_flag: np.ndarray, start: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorised history features for ALL series at once (rows sorted by series then date).

    ``start[i]`` is the row index where row i's series begins. Every quantity is shifted by
    >= MIN_LAG rows *within its series*, so no feature can see the target day or the blind window.
    """
    n = len(q)
    idx = np.arange(n)
    pos = idx - start
    out: dict[str, np.ndarray] = {}
    for k in LAGS:
        out[f"lag_{k}"] = _shift(q, k, pos)
    s = _shift(q, MIN_LAG, pos)
    for w in ROLLING_WINDOWS:
        out[f"rolling_mean_{w}"] = _window_stats(s, start, w, (w + 1) // 2)
    for w in STD_WINDOWS:
        out[f"rolling_std_{w}"] = _window_stats(s, start, w, (w + 1) // 2, want_std=True)
    lag_stack = np.column_stack([out[f"lag_{k}"] for k in LAGS])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Mean of empty slice" on cold-start rows
        out["dow_mean_4w"] = np.nanmean(lag_stack, axis=1)
        out["ratio_mean_7_28"] = out["rolling_mean_7"] / np.where(
            out["rolling_mean_28"] > 0, out["rolling_mean_28"], np.nan
        )
    nz = np.where(np.isnan(q), np.nan, (q > 0).astype(float))
    out["sale_rate_28"] = _window_stats(_shift(nz, MIN_LAG, pos), start, 28, 14)

    with np.errstate(invalid="ignore"):
        sold = s > 0
    last = np.maximum.accumulate(np.where(sold, idx, -1))
    none = last < start  # no sale yet inside this series
    dsl = np.where(none, np.nan, idx - last).astype(float)
    # Cap ("not sold within the window") so short serving histories stay consistent with training.
    dsl = np.where(dsl > MAX_DAYS_SINCE_SALE, MAX_DAYS_SINCE_SALE, dsl)
    dsl = np.where(none & ~np.isnan(s), MAX_DAYS_SINCE_SALE, dsl)
    out["days_since_last_sale"] = dsl

    seen = np.maximum.accumulate(np.where(~np.isnan(price_obs), idx, -1))
    ffilled = np.where(seen >= start, price_obs[np.clip(seen, 0, None)], np.nan)
    out["last_price"] = _shift(ffilled, MIN_LAG, pos)
    promo_s = _shift(promo_flag.astype(float), MIN_LAG, pos)
    ps = _window_stats(promo_s, start, 28, 1)
    cnt_valid = ~np.isnan(promo_s)
    # rolling(28, min_periods=1).sum(): sum of available values (mean * count)
    lo = np.maximum(idx - 28 + 1, start)
    ccnt = np.concatenate([[0.0], np.cumsum(cnt_valid.astype("float64"))])
    out["promo_days_28"] = ps * (ccnt[idx + 1] - ccnt[lo])
    return out


def build_features(
    panel: pd.DataFrame,
    stores: pd.DataFrame,
    *,
    min_history_days: int | None = MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    """Return ``panel`` keys + target + FEATURE_COLUMNS (+ ``age_days`` used for filtering).

    ``min_history_days=None`` keeps every row (used at serving time on short histories).
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    df = panel.sort_values([*KEYS, "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    promo_flag = df["promo_price"].notna().to_numpy().astype("int8")

    boundaries = np.flatnonzero(
        np.r_[True, (df["store_id"].to_numpy()[1:] != df["store_id"].to_numpy()[:-1])]
        | np.r_[True, (df["item_id"].to_numpy()[1:] != df["item_id"].to_numpy()[:-1])]
    )
    ends = np.r_[boundaries[1:], n]
    start = np.repeat(boundaries, ends - boundaries)
    age = (np.arange(n) - start).astype("int32")

    feats = _history_features(
        df["quantity"].to_numpy(dtype="float64"),
        df["price_obs"].to_numpy(dtype="float64"),
        promo_flag,
        start,
    )
    for name, arr in feats.items():
        df[name] = arr.astype("float32")
    df["age_days"] = age

    df = add_calendar_features(df)
    df["promo_flag"] = promo_flag
    df["promo_type"] = df["promo_type"].fillna(0).astype("float32")
    promo_price = df["promo_price"].to_numpy(dtype="float64")
    before = df["promo_before_price"].to_numpy(dtype="float64")
    with np.errstate(all="ignore"):
        depth = np.where((promo_flag == 1) & (before > 0), 1.0 - promo_price / before, 0.0)
    df["discount_depth"] = np.clip(depth, 0.0, 1.0).astype("float32")
    last_price = df["last_price"].to_numpy(dtype="float64")
    planned = np.where(promo_flag == 1, promo_price, last_price)
    df["planned_price"] = planned.astype("float32")
    with np.errstate(all="ignore"):
        ratio = np.where(last_price > 0, planned / last_price, np.nan)
    df["price_ratio"] = np.clip(ratio, 0.0, 3.0).astype("float32")

    df = df.merge(store_static_table(stores), on="store_id", how="left")
    for col in ("dept_code", "class_code"):
        df[col] = df[col].fillna(-1).astype("int16")

    if min_history_days is not None:
        df = df[df["age_days"] >= min_history_days].reset_index(drop=True)
    keep = list(dict.fromkeys([*KEYS, "date", "quantity", "age_days", *FEATURE_COLUMNS]))
    return df[keep]


def chronological_masks(dates: pd.Series, split) -> dict[str, np.ndarray]:
    """Boolean masks for train / val / test given SplitDates (inclusive bounds)."""
    d = pd.to_datetime(dates)
    return {
        "train": (d <= pd.Timestamp(split.train_end)).to_numpy(),
        "val": (
            (d >= pd.Timestamp(split.val_start)) & (d <= pd.Timestamp(split.val_end))
        ).to_numpy(),
        "test": (
            (d >= pd.Timestamp(split.test_start)) & (d <= pd.Timestamp(split.test_end))
        ).to_numpy(),
    }
