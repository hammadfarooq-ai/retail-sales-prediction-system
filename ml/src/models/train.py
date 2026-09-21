"""Train every candidate, compare on chronological validation/test, refit the winner, persist.

Protocol
    1. Fit each candidate on TRAIN only; score on VALIDATION and TEST (fair comparison table).
    2. Select the best ML model by validation MAE (the test period is never used for selection).
    3. Refit the selected model on TRAIN+VALIDATION (fixed iteration count from step 1) and report
       its TEST metrics as the headline figure; the TEST period remains unseen by that fit.
    4. Fit 10%/90% LightGBM quantile models (train+val) for an 80% prediction interval and report
       their empirical coverage on TEST.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from ml.src.config import (
    ARTIFACTS_DIR,
    LAGS,
    MIN_HISTORY_DAYS,
    MIN_LAG,
    PROCESSED_DIR,
    ROLLING_WINDOWS,
    STD_WINDOWS,
    TARGET,
    UNIVERSE_MIN_DENSITY,
    UNIVERSE_MIN_SALE_DAYS,
    SplitDates,
)
from ml.src.evaluation.metrics import aggregated_metrics, all_metrics
from ml.src.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
    chronological_masks,
)
from ml.src.models.candidates import (
    N_JOBS,
    RANDOM_STATE,
    SklearnLike,
    build_candidates,
    lightgbm_params,
    make_identity_preprocessor,
)

log = logging.getLogger(__name__)
BASELINES = {"naive_seasonal_lag7", "naive_moving_avg_28"}


def load_feature_frame() -> tuple[pd.DataFrame, SplitDates, pd.DataFrame]:
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    stores = pd.read_parquet(PROCESSED_DIR / "stores.parquet")
    meta = json.loads((PROCESSED_DIR / "prepare_meta.json").read_text())
    split = SplitDates(**meta["split"])
    log.info("Building features for %d panel rows", len(panel))
    feats = build_features(panel, stores)
    log.info("Feature frame: %s", feats.shape)
    return feats, split, stores


def _final_estimator_pipeline(model: SklearnLike) -> Pipeline:
    return Pipeline([("preprocess", model.preprocessor), ("model", model.estimator)])


def _refit_on_train_val(name: str, best_iteration: int | None, X, y):
    """Rebuild the winner with a fixed iteration budget and fit on train+val (no early stopping)."""
    fresh = next(c for c in build_candidates() if c.name == name)
    assert isinstance(fresh, SklearnLike)
    if best_iteration is not None:
        n = int(best_iteration * 1.1)
        fresh.estimator.set_params(n_estimators=n)
        if fresh.early_stopping == "xgboost":
            fresh.estimator.set_params(early_stopping_rounds=None)
        fresh.early_stopping = None
    fresh.fit(X, y)
    return fresh


def _feature_importance(model: SklearnLike) -> dict[str, float]:
    est = model.estimator
    names = list(model.preprocessor.get_feature_names_out())
    if hasattr(est, "booster_"):
        imp = est.booster_.feature_importance(importance_type="gain")
    elif hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_)
    else:
        return {}
    imp = np.asarray(imp, dtype="float64")
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    return dict(sorted(zip(names, map(float, imp), strict=True), key=lambda kv: -kv[1]))


def run_training(out_dir: Path | None = None) -> dict:
    out = out_dir or ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    feats, split, stores = load_feature_frame()
    masks = chronological_masks(feats["date"], split)
    X = feats[FEATURE_COLUMNS]
    y = feats[TARGET].to_numpy(dtype="float32")
    Xtr, ytr = X[masks["train"]], y[masks["train"]]
    Xva, yva = X[masks["val"]], y[masks["val"]]
    Xte, yte = X[masks["test"]], y[masks["test"]]
    log.info("rows train=%d val=%d test=%d", len(Xtr), len(Xva), len(Xte))
    assert feats.loc[masks["train"], "date"].max() < feats.loc[masks["val"], "date"].min()
    assert feats.loc[masks["val"], "date"].max() < feats.loc[masks["test"], "date"].min()

    comparison: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for cand in build_candidates():
        log.info("Fitting %s", cand.name)
        cand.fit(Xtr, ytr, Xva, yva)
        pv, pt = cand.predict(Xva), cand.predict(Xte)
        comparison[cand.name] = {
            "description": cand.description,
            "is_baseline": cand.name in BASELINES,
            "validation": all_metrics(yva, pv),
            "test": all_metrics(yte, pt),
            "fit_seconds": round(float(getattr(cand, "fit_seconds", 0.0)), 1),
            "best_iteration": getattr(cand, "best_iteration", None),
            "params": getattr(cand, "params", {}),
        }
        fitted[cand.name] = cand
        v = comparison[cand.name]["validation"]
        log.info("%s val MAE=%.4f RMSE=%.4f WAPE=%.2f%%", cand.name, v["mae"], v["rmse"], v["wape"])

    ml_names = [n for n in comparison if n not in BASELINES]
    selected = min(ml_names, key=lambda n: comparison[n]["validation"]["mae"])
    log.info("Selected model (lowest validation MAE): %s", selected)

    # ---- refit winner on train+val, evaluate on the untouched test period ----------------
    trval = masks["train"] | masks["val"]
    final = _refit_on_train_val(
        selected, comparison[selected]["best_iteration"], X[trval], y[trval]
    )
    pred_te = final.predict(Xte)
    final_test = all_metrics(yte, pred_te)

    # ---- quantile models for an 80% interval -------------------------------------------
    import lightgbm as lgb

    q_models: dict[str, SklearnLike] = {}
    for tag, alpha in (("q10", 0.1), ("q90", 0.9)):
        qm = SklearnLike(
            f"lightgbm_{tag}",
            f"LightGBM quantile {alpha}",
            make_identity_preprocessor(),
            lgb.LGBMRegressor(
                **lightgbm_params(
                    objective="quantile", alpha=alpha, n_estimators=400, num_leaves=63
                )
            ),
        )
        qm.fit(X[trval], y[trval])
        q_models[tag] = qm
    lo, hi = q_models["q10"].predict(Xte), q_models["q90"].predict(Xte)
    lo, hi = np.minimum(lo, pred_te), np.maximum(hi, pred_te)
    coverage = float(np.mean((yte >= lo) & (yte <= hi)) * 100)

    # ---- backtest frame (test period) ---------------------------------------------------
    bt = feats.loc[masks["test"], ["store_id", "item_id", "date"]].copy()
    bt["actual"] = yte
    bt["predicted"] = pred_te
    bt["lower"] = lo
    bt["upper"] = hi
    bt["baseline_lag7"] = fitted["naive_seasonal_lag7"].predict(Xte)  # type: ignore[attr-defined]
    per_store = {
        int(str(s)): all_metrics(g["actual"], g["predicted"]) for s, g in bt.groupby("store_id")
    }
    agg = {
        "store_day": aggregated_metrics(bt, ["store_id", "date"]),
        "day_total": aggregated_metrics(bt, ["date"]),
        "item_store_test_week": aggregated_metrics(
            bt.assign(week=bt["date"].dt.isocalendar().week), ["store_id", "item_id", "week"]
        ),
    }
    baseline_lag7_test = comparison["naive_seasonal_lag7"]["test"]
    importance = _feature_importance(final)

    trained_at = datetime.now(UTC)
    version = f"{selected}-{trained_at:%Y%m%d%H%M}"
    joblib.dump(_final_estimator_pipeline(final), out / "model.joblib", compress=3)
    joblib.dump(final.preprocessor, out / "preprocessor.joblib", compress=3)
    for tag, qm in q_models.items():
        joblib.dump(_final_estimator_pipeline(qm), out / f"model_{tag}.joblib", compress=3)

    feature_config = {
        "target": TARGET,
        "horizon_days": MIN_LAG,
        "min_lag": MIN_LAG,
        "lags": list(LAGS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "std_windows": list(STD_WINDOWS),
        "min_history_days": MIN_HISTORY_DAYS,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_features": len(FEATURE_COLUMNS),
        "universe": {"min_density": UNIVERSE_MIN_DENSITY, "min_sale_days": UNIVERSE_MIN_SALE_DAYS},
    }
    metrics_doc = {
        "selection_metric": "validation MAE (ML models only)",
        "selected_model": selected,
        "comparison": comparison,
        "final_model_test": final_test,
        "final_model_test_by_store": per_store,
        "final_model_test_aggregated": agg,
        "baseline_lag7_test": baseline_lag7_test,
        "improvement_vs_lag7_baseline_pct": {
            "mae": 100 * (1 - final_test["mae"] / baseline_lag7_test["mae"]),
            "rmse": 100 * (1 - final_test["rmse"] / baseline_lag7_test["rmse"]),
        },
        "prediction_interval": {"level": 0.8, "empirical_test_coverage_pct": coverage},
    }
    metadata = {
        "model_name": selected,
        "version": version,
        "trained_at": trained_at.isoformat(),
        "framework_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "split": split.as_dict(),
        "n_features": len(FEATURE_COLUMNS),
        "n_train_rows": int(masks["train"].sum()),
        "n_train_val_rows_final_fit": int(trval.sum()),
        "n_test_rows": int(masks["test"].sum()),
        "n_series": int(feats.groupby(["store_id", "item_id"]).ngroups),
        "params": comparison[selected]["params"],
        "final_iterations": getattr(final.estimator, "n_estimators", None),
        "prediction_interval": metrics_doc["prediction_interval"],
        "test_metrics": final_test,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    }
    for fname, doc in {
        "feature_config.json": feature_config,
        "metrics.json": metrics_doc,
        "model_metadata.json": metadata,
        "feature_importance.json": importance,
    }.items():
        (out / fname).write_text(json.dumps(doc, indent=2))
    bt.to_parquet(PROCESSED_DIR / "backtest_predictions.parquet", index=False)

    # residual/plot inputs for the visualisation step
    bt_val = feats.loc[masks["val"], ["store_id", "item_id", "date"]].copy()
    bt_val["actual"] = yva
    log.info("Artifacts written to %s (version %s)", out, version)
    return {"metrics": metrics_doc, "metadata": metadata}
