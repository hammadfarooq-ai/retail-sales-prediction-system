"""Candidate forecasting models behind one small interface.

Every candidate implements ``fit(X_train, y_train, X_val, y_val)`` and ``predict(X)`` where X is a
DataFrame containing ``FEATURE_COLUMNS``. Tree models use NaN natively; the linear model gets
median imputation, scaling and one-hot store encoding inside its own pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src.features.engineering import FEATURE_COLUMNS

RANDOM_STATE = 42
N_JOBS = 4


class NaiveLag7:
    """Seasonal naive: same weekday last week (lag_7); falls back to the 28-day mean."""

    name = "naive_seasonal_lag7"
    description = "Prediction = quantity sold 7 days earlier (same weekday)"

    def fit(self, X, y, Xv=None, yv=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_7"].fillna(X["rolling_mean_28"]).fillna(0).to_numpy()


class NaiveMean28:
    name = "naive_moving_avg_28"
    description = "Prediction = mean daily quantity over the 28 days ending at the forecast origin"

    def fit(self, X, y, Xv=None, yv=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["rolling_mean_28"].fillna(X["lag_7"]).fillna(0).to_numpy()


def make_identity_preprocessor() -> ColumnTransformer:
    """Column selection/ordering only (tree models). Keeps NaN; output is a float32 DataFrame."""
    return ColumnTransformer(
        [("cols", "passthrough", FEATURE_COLUMNS)], verbose_feature_names_out=False
    ).set_output(transform="pandas")


def make_linear_preprocessor() -> ColumnTransformer:
    cat = ["store_id"]
    drop = {
        "dept_code",
        "class_code",
        "store_format_code",
        "store_id",
    }  # ordinal codes are meaningless
    num = [c for c in FEATURE_COLUMNS if c not in drop]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


@dataclass
class SklearnLike:
    """Wraps (preprocessor, estimator) so val-set early stopping uses the fitted preprocessor."""

    name: str
    description: str
    preprocessor: Any
    estimator: Any
    early_stopping: str | None = None  # "lightgbm" | "xgboost" | None
    fit_rows: int | None = None  # optional random subsample of training rows
    params: dict = field(default_factory=dict)
    fit_seconds: float = 0.0
    best_iteration: int | None = None

    def fit(self, X, y, Xv=None, yv=None):
        t0 = time.time()
        if self.fit_rows and len(X) > self.fit_rows:
            idx = np.random.default_rng(RANDOM_STATE).choice(len(X), self.fit_rows, replace=False)
            X, y = X.iloc[idx], y[idx]
        Xt = self.preprocessor.fit_transform(X)
        if self.early_stopping == "lightgbm":
            import lightgbm as lgb

            self.estimator.fit(
                Xt,
                y,
                eval_set=[(self.preprocessor.transform(Xv), yv)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            self.best_iteration = int(self.estimator.best_iteration_ or self.estimator.n_estimators)
        elif self.early_stopping == "xgboost":
            self.estimator.fit(
                Xt, y, eval_set=[(self.preprocessor.transform(Xv), yv)], verbose=False
            )
            self.best_iteration = int(self.estimator.best_iteration + 1)
        else:
            self.estimator.fit(Xt, y)
        self.fit_seconds = time.time() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.estimator.predict(self.preprocessor.transform(X)), 0, None)


def lightgbm_params(**overrides) -> dict:
    p = {
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }
    p.update(overrides)
    return p


def build_candidates() -> list:
    import lightgbm as lgb
    import xgboost as xgb

    return [
        NaiveLag7(),
        NaiveMean28(),
        SklearnLike(
            "ridge",
            "Ridge regression (median-imputed, standardised features, one-hot store)",
            make_linear_preprocessor(),
            Ridge(alpha=10.0, random_state=RANDOM_STATE),
            params={"alpha": 10.0},
        ),
        SklearnLike(
            "random_forest",
            "Random Forest (trained on a 600k-row random sample of the training period)",
            make_identity_preprocessor(),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=30,
                max_features=0.5,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            ),
            fit_rows=600_000,
            params={
                "n_estimators": 100,
                "max_depth": 16,
                "min_samples_leaf": 30,
                "max_features": 0.5,
            },
        ),
        SklearnLike(
            "xgboost",
            "XGBoost (histogram trees, squared error, early stopping on validation)",
            make_identity_preprocessor(),
            xgb.XGBRegressor(
                n_estimators=3000,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                early_stopping_rounds=50,
                eval_metric="rmse",
            ),
            early_stopping="xgboost",
            params={"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 20},
        ),
        SklearnLike(
            "lightgbm",
            "LightGBM (squared error, early stopping on validation)",
            make_identity_preprocessor(),
            lgb.LGBMRegressor(**lightgbm_params()),
            early_stopping="lightgbm",
            params={"learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 50},
        ),
        SklearnLike(
            "lightgbm_tweedie",
            "LightGBM with Tweedie objective (non-negative, zero-inflated demand)",
            make_identity_preprocessor(),
            lgb.LGBMRegressor(**lightgbm_params(objective="tweedie", tweedie_variance_power=1.3)),
            early_stopping="lightgbm",
            params={"objective": "tweedie", "tweedie_variance_power": 1.3, "num_leaves": 127},
        ),
    ]
