"""Synthetic panel + tiny model artifacts, shared by ML and backend tests (no dataset needed)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.config import compute_split
from ml.src.features.engineering import FEATURE_COLUMNS, build_features
from ml.src.models.candidates import lightgbm_params, make_identity_preprocessor

STORES = pd.DataFrame(
    {
        "store_id": [1, 2],
        "division": ["Div1", "Div2"],
        "format": ["Format-1", "Format-6"],
        "city": ["City1", "City1"],
        "area": [1500, 210],
    }
)
ITEMS = ["aaa111", "bbb222", "ccc333"]
DATA_START = pd.Timestamp("2024-01-01")
N_DAYS = 150
LAST_DATE = DATA_START + pd.Timedelta(days=N_DAYS - 1)


def make_panel(n_days: int = N_DAYS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(DATA_START, periods=n_days, freq="D")
    frames = []
    for store in (1, 2):
        for i, item in enumerate(ITEMS):
            base = 4 + 3 * i
            q = rng.poisson(base, n_days).astype("float32") + (dates.dayofweek == 4) * 3
            promo = rng.random(n_days) < 0.15
            q = q * np.where(promo, 1.5, 1.0)
            price = np.where(q > 0, 100 + 10 * i + rng.normal(0, 0.5, n_days), np.nan)
            frames.append(
                pd.DataFrame(
                    {
                        "store_id": np.int8(store),
                        "item_id": item,
                        "date": dates,
                        "quantity": q.astype("float32"),
                        "price_obs": price.astype("float32"),
                        "promo_price": np.where(promo, 80.0, np.nan).astype("float32"),
                        "promo_before_price": np.where(promo, 100.0, np.nan).astype("float32"),
                        "promo_type": np.where(promo, 5.0, np.nan).astype("float32"),
                        "dept_code": np.int16(i),
                        "class_code": np.int16(i),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def write_tiny_artifacts(out: Path, panel: pd.DataFrame | None = None) -> Path:
    """Train a small LightGBM (point + two quantile models) and write real artifact files."""
    import lightgbm as lgb

    panel = panel if panel is not None else make_panel()
    feats = build_features(panel, STORES)
    X, y = feats[FEATURE_COLUMNS], feats["quantity"].to_numpy()
    prep = make_identity_preprocessor().fit(X)

    def fit(**kw) -> Pipeline:
        est = lgb.LGBMRegressor(
            **lightgbm_params(n_estimators=40, num_leaves=15, min_child_samples=5, **kw)
        )
        est.fit(prep.transform(X), y)
        return Pipeline([("preprocess", prep), ("model", est)])

    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit(), out / "model.joblib")
    joblib.dump(fit(objective="quantile", alpha=0.1), out / "model_q10.joblib")
    joblib.dump(fit(objective="quantile", alpha=0.9), out / "model_q90.joblib")
    split = compute_split(DATA_START, LAST_DATE)
    test_metrics = {
        "mae": 1.0,
        "rmse": 2.0,
        "mape": 30.0,
        "smape": 25.0,
        "wape": 20.0,
        "r2": 0.5,
        "bias": 0.0,
        "n_rows": 10,
        "mape_coverage_pct": 90.0,
    }
    (out / "feature_config.json").write_text(
        json.dumps(
            {
                "target": "quantity",
                "horizon_days": 7,
                "min_lag": 7,
                "feature_columns": FEATURE_COLUMNS,
                "n_features": len(FEATURE_COLUMNS),
            }
        )
    )
    interval = {"level": 0.8, "empirical_test_coverage_pct": 75.0}
    (out / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "lightgbm_tiny",
                "version": "lightgbm_tiny-202601010000",
                "trained_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                "split": split.as_dict(),
                "n_features": len(FEATURE_COLUMNS),
                "n_series": 6,
                "n_train_rows": len(X),
                "params": {},
                "prediction_interval": interval,
                "test_metrics": test_metrics,
            }
        )
    )
    (out / "feature_importance.json").write_text(json.dumps({"lag_7": 0.4, "rolling_mean_28": 0.3}))
    metrics = {
        "selection_metric": "validation MAE (ML models only)",
        "selected_model": "lightgbm_tiny",
        "comparison": {
            "naive_seasonal_lag7": {
                "description": "naive",
                "is_baseline": True,
                "validation": test_metrics,
                "test": test_metrics,
                "fit_seconds": 0,
                "best_iteration": None,
                "params": {},
            },
            "lightgbm_tiny": {
                "description": "tiny",
                "is_baseline": False,
                "validation": test_metrics,
                "test": test_metrics,
                "fit_seconds": 1,
                "best_iteration": 40,
                "params": {},
            },
        },
        "final_model_test": test_metrics,
        "final_model_test_by_store": {"1": test_metrics},
        "final_model_test_aggregated": {"day_total": test_metrics},
        "improvement_vs_lag7_baseline_pct": {"mae": 10.0, "rmse": 5.0},
        "prediction_interval": interval,
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out
