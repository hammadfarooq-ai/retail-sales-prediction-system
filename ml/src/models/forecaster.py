"""Serving-side forecaster: loads artifacts once and produces point forecasts (+ intervals).

Strategy
    * The model is a *direct 7-day* model: all history features are lagged >= 7 days.
    * For dates up to the last observed day (backtest mode) or up to 7 days beyond it, features
      are computed straight from observed history and the 80% quantile interval is available.
    * Beyond 7 days the forecast is *recursive* in 7-day blocks: predicted quantities are fed back
      as pseudo-history for the next block. Point forecasts are returned but the interval is
      omitted because the quantile models were calibrated for the direct 7-day setting only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.src.config import ARTIFACTS_DIR, MIN_LAG
from ml.src.features.engineering import FEATURE_COLUMNS, build_features

log = logging.getLogger(__name__)

# Median discount depth of scheduled promotions in the data (see docs/figures/eda/eda_facts.json).
DEFAULT_DISCOUNT_DEPTH = 0.143


class ModelNotAvailableError(RuntimeError):
    """Artifacts are missing or unreadable."""


@dataclass(frozen=True)
class Override:
    """Optional what-if inputs for one target day."""

    price: float | None = None  # planned shelf price on the target day
    promotion: bool | None = None  # force promo on/off (None = use the promo calendar)
    discount_pct: float | None = None  # 0-100, used when promotion is on


class Forecaster:
    REQUIRED_FILES = ("model.joblib", "feature_config.json", "model_metadata.json")

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        d = Path(artifacts_dir or ARTIFACTS_DIR)
        missing = [f for f in self.REQUIRED_FILES if not (d / f).exists()]
        if missing:
            raise ModelNotAvailableError(f"Model artifacts missing in {d}: {missing}")
        try:
            self.model = joblib.load(d / "model.joblib")
            self.model_q10 = (
                joblib.load(d / "model_q10.joblib") if (d / "model_q10.joblib").exists() else None
            )
            self.model_q90 = (
                joblib.load(d / "model_q90.joblib") if (d / "model_q90.joblib").exists() else None
            )
            self.feature_config: dict[str, Any] = json.loads(
                (d / "feature_config.json").read_text()
            )
            self.metadata: dict[str, Any] = json.loads((d / "model_metadata.json").read_text())
        except Exception as exc:  # corrupted / incompatible artifact
            raise ModelNotAvailableError(f"Could not load model artifacts from {d}: {exc}") from exc
        self.artifacts_dir = d
        self.feature_columns: list[str] = self.feature_config["feature_columns"]
        if self.feature_columns != FEATURE_COLUMNS:
            raise ModelNotAvailableError(
                "feature_config.json does not match the current feature code; retrain the model"
            )
        self.version: str = self.metadata["version"]
        self.has_interval = self.model_q10 is not None and self.model_q90 is not None
        log.info("Loaded model %s (%d features)", self.version, len(self.feature_columns))

    # ------------------------------------------------------------------ prediction core
    def _point(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[self.feature_columns]), 0, None)

    def _interval(self, X: pd.DataFrame, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.clip(self.model_q10.predict(X[self.feature_columns]), 0, None)  # type: ignore[union-attr]
        hi = np.clip(self.model_q90.predict(X[self.feature_columns]), 0, None)  # type: ignore[union-attr]
        return np.minimum(lo, point), np.maximum(hi, point)

    @staticmethod
    def _apply_overrides(
        feats: pd.DataFrame, overrides: dict[tuple[int, str, date], Override]
    ) -> pd.DataFrame:
        """Apply what-if price/promotion inputs to the matching target-day feature rows."""
        if not overrides:
            return feats
        feats = feats.copy()
        keys = list(
            zip(
                feats["store_id"].astype(int).tolist(),
                feats["item_id"].tolist(),
                feats["date"].dt.date.tolist(),
                strict=True,
            )
        )
        last_price = feats["last_price"].to_numpy(dtype="float64")
        promo_flag = feats["promo_flag"].to_numpy(dtype="int64").copy()
        depth_arr = feats["discount_depth"].to_numpy(dtype="float64").copy()
        planned_arr = feats["planned_price"].to_numpy(dtype="float64").copy()
        ratio_arr = feats["price_ratio"].to_numpy(dtype="float64").copy()
        for i, key in enumerate(keys):
            ov = overrides.get(key)
            if ov is None:
                continue
            lp = float(last_price[i])
            promo = bool(promo_flag[i]) if ov.promotion is None else ov.promotion
            if not promo:
                depth = 0.0
            elif ov.discount_pct is not None:
                depth = ov.discount_pct / 100.0
            elif depth_arr[i] > 0:
                depth = float(depth_arr[i])
            else:
                depth = DEFAULT_DISCOUNT_DEPTH
            if ov.price is not None:
                planned = float(ov.price)
            elif promo and not np.isnan(lp):
                planned = lp * (1 - depth)
            else:
                planned = lp
            promo_flag[i] = int(promo)
            depth_arr[i] = depth
            planned_arr[i] = planned
            ratio_arr[i] = float(np.clip(planned / lp, 0, 3)) if lp > 0 else np.nan
        feats["promo_flag"] = promo_flag
        feats["discount_depth"] = depth_arr
        feats["planned_price"] = planned_arr
        feats["price_ratio"] = ratio_arr
        return feats

    def _predict_rows(
        self,
        feats: pd.DataFrame,
        overrides: dict[tuple[int, str, date], Override],
        with_interval: bool,
    ) -> pd.DataFrame:
        feats = self._apply_overrides(feats, overrides)
        point = self._point(feats)
        out = feats[["store_id", "item_id", "date"]].copy()
        out["predicted"] = point
        out["lower"], out["upper"] = np.nan, np.nan
        if with_interval and self.has_interval:
            out["lower"], out["upper"] = self._interval(feats, point)
        # echo the effective business inputs so callers can show an input summary
        out["last_price"] = feats["last_price"].to_numpy()
        out["effective_price"] = feats["planned_price"].to_numpy()
        out["effective_promo"] = feats["promo_flag"].to_numpy().astype(bool)
        out["effective_discount_pct"] = feats["discount_depth"].to_numpy() * 100
        return out

    # ------------------------------------------------------------------ public API
    def predict_history_dates(
        self,
        history: pd.DataFrame,
        stores: pd.DataFrame,
        dates: set[date] | None = None,
        overrides: dict[tuple[int, str, date], Override] | None = None,
    ) -> pd.DataFrame:
        """Backtest-mode predictions for days already inside `history` (features use t-7 info)."""
        feats = build_features(history, stores, min_history_days=None)
        if dates is not None:
            feats = feats[feats["date"].dt.date.isin(dates)]
        res = self._predict_rows(feats.reset_index(drop=True), overrides or {}, True)
        res["is_recursive"] = False
        return res

    def _forecast_chunk(
        self,
        history: pd.DataFrame,
        promo_future: pd.DataFrame,
        stores: pd.DataFrame,
        end_date: date,
        overrides: dict[tuple[int, str, date], Override] | None = None,
    ) -> pd.DataFrame:
        """Forecast every series in `history` for last_history_day+1 .. end_date."""
        overrides = overrides or {}
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"])
        last = history["date"].max()
        end = pd.Timestamp(end_date)
        if end <= last:
            raise ValueError("end_date must be after the last observed day")

        pairs = history[["store_id", "item_id", "dept_code", "class_code"]].drop_duplicates()
        future_dates = pd.date_range(last + pd.Timedelta(days=1), end)
        fut = pairs.merge(pd.DataFrame({"date": future_dates}), how="cross")
        fut["quantity"] = np.nan
        fut["price_obs"] = np.nan
        promos = promo_future.copy()
        if len(promos):
            promos["date"] = pd.to_datetime(promos["date"])
            fut = fut.merge(
                promos[
                    [
                        "store_id",
                        "item_id",
                        "date",
                        "promo_price",
                        "promo_before_price",
                        "promo_type",
                    ]
                ],
                on=["store_id", "item_id", "date"],
                how="left",
            )
        else:
            for c in ("promo_price", "promo_before_price", "promo_type"):
                fut[c] = np.nan
        keep = list(history.columns)
        ext = pd.concat([history, fut[keep]], ignore_index=True)
        ext = ext.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        results = []
        block_start = last + pd.Timedelta(days=1)
        while block_start <= end:
            block_end = min(block_start + pd.Timedelta(days=MIN_LAG - 1), end)
            feats = build_features(ext, stores, min_history_days=None)
            rows = feats[(feats["date"] >= block_start) & (feats["date"] <= block_end)]
            recursive = block_start > last + pd.Timedelta(days=MIN_LAG)
            block = self._predict_rows(rows.reset_index(drop=True), overrides, not recursive)
            block["is_recursive"] = recursive
            results.append(block)
            # feed predictions back as pseudo-actuals for the next block's lag features
            key = block.set_index(["store_id", "item_id", "date"])["predicted"]
            idx = pd.MultiIndex.from_frame(ext[["store_id", "item_id", "date"]])
            m = idx.isin(key.index)
            ext.loc[m, "quantity"] = key.reindex(idx[m]).to_numpy()
            block_start = block_end + pd.Timedelta(days=1)
        return pd.concat(results, ignore_index=True)

    def forecast(
        self,
        history: pd.DataFrame,
        promo_future: pd.DataFrame,
        stores: pd.DataFrame,
        end_date: date,
        overrides: dict[tuple[int, str, date], Override] | None = None,
        chunk_pairs: int = 400,
    ) -> pd.DataFrame:
        """Forecast every series in `history` for last_history_day+1 .. end_date.

        Series are independent, so they are processed in chunks of ``chunk_pairs`` to bound peak
        memory (a whole store is ~2,400 series); results are identical to a single pass.
        """
        pairs = history[["store_id", "item_id"]].drop_duplicates().reset_index(drop=True)
        if len(pairs) <= chunk_pairs:
            return self._forecast_chunk(history, promo_future, stores, end_date, overrides)
        parts = []
        for i in range(0, len(pairs), chunk_pairs):
            keys = pairs.iloc[i : i + chunk_pairs]
            h = history.merge(keys, on=["store_id", "item_id"])
            p = (
                promo_future.merge(keys, on=["store_id", "item_id"])
                if len(promo_future)
                else promo_future
            )
            parts.append(self._forecast_chunk(h, p, stores, end_date, overrides))
        return pd.concat(parts, ignore_index=True)
