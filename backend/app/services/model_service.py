"""Model metadata, performance and backtest views (all read from the DB / loaded artifacts)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ModelUnavailableError, UnknownProductError, UnknownStoreError
from app.ml.registry import ModelRegistry
from app.models import BacktestPrediction, ItemStoreDaily, ModelVersion, Product, Store
from app.schemas.analytics import (
    BacktestPoint,
    BacktestResponse,
    HistogramBin,
    ModelInfo,
    ModelPerformance,
)


def active_version(db: Session) -> ModelVersion:
    mv = db.scalar(
        select(ModelVersion).where(ModelVersion.is_active).order_by(ModelVersion.id.desc())
    )
    if mv is None:
        raise ModelUnavailableError("No model version is registered. Run the seed step.")
    return mv


def model_info(db: Session, registry: ModelRegistry) -> ModelInfo:
    fc = registry.require()
    md = fc.metadata
    last = db.scalar(select(func.max(ItemStoreDaily.date)))
    return ModelInfo(
        version=fc.version,
        model_name=md["model_name"],
        trained_at=md["trained_at"],
        n_features=md["n_features"],
        feature_names=fc.feature_columns,
        horizon_days=fc.feature_config["horizon_days"],
        target=fc.feature_config["target"],
        split=md["split"],
        n_series=md.get("n_series"),
        n_train_rows=md.get("n_train_rows"),
        has_prediction_interval=fc.has_interval,
        prediction_interval=md.get("prediction_interval") if fc.has_interval else None,
        test_metrics=md["test_metrics"],
        last_observed_date=last,
    )


def performance(db: Session, registry: ModelRegistry) -> ModelPerformance:
    mv = active_version(db)
    m = mv.metrics
    return ModelPerformance(
        version=mv.version,
        model_name=mv.model_name,
        trained_at=mv.trained_at,
        selection_metric=m["selection_metric"],
        split=mv.split,
        n_features=mv.n_features,
        comparison=m["comparison"],
        final_model_test=m["final_model_test"],
        final_model_test_by_store=m["final_model_test_by_store"],
        final_model_test_aggregated=m["final_model_test_aggregated"],
        improvement_vs_lag7_baseline_pct=m["improvement_vs_lag7_baseline_pct"],
        feature_importance=dict(list(registry.feature_importance.items())[:20]),
        prediction_interval=m["prediction_interval"],
    )


def backtest(
    db: Session, store_id: int | None, item_id: str | None, dept_name: str | None = None
) -> BacktestResponse:
    mv = active_version(db)
    t = BacktestPrediction
    if store_id is not None and db.get(Store, store_id) is None:
        raise UnknownStoreError(f"Store {store_id} does not exist")
    if item_id is not None and db.get(Product, item_id) is None:
        raise UnknownProductError(f"Product '{item_id}' does not exist")

    where = [t.model_version == mv.version]
    if store_id is not None:
        where.append(t.store_id == store_id)
    if item_id is not None:
        where.append(t.item_id == item_id)

    daily = db.execute(
        select(
            t.date,
            func.sum(t.actual),
            func.sum(t.predicted),
            func.sum(t.baseline_lag7),
        )
        .where(*where)
        .group_by(t.date)
        .order_by(t.date)
    ).all()
    points = [
        BacktestPoint(
            date=d,
            actual=float(a),
            predicted=float(p),
            baseline_lag7=None if b is None else float(b),
        )
        for d, a, p, b in daily
    ]
    arr = pd.read_sql_query(select(t.actual, t.predicted).where(*where), db.connection()).to_numpy(
        dtype="float64"
    )
    scope = "item" if item_id else ("store" if store_id is not None else "all")
    if len(arr) == 0:
        return BacktestResponse(
            model_version=mv.version,
            scope=scope,
            points=[],
            error_histogram=[],
            residual_summary={},
            scatter_sample=[],
        )
    resid = arr[:, 1] - arr[:, 0]  # predicted - actual
    lo, hi = np.percentile(resid, [1, 99])
    edge = max(abs(lo), abs(hi), 1e-6)
    counts, edges = np.histogram(np.clip(resid, -edge, edge), bins=40, range=(-edge, edge))
    hist = [
        HistogramBin(lower=float(edges[i]), upper=float(edges[i + 1]), count=int(counts[i]))
        for i in range(len(counts))
    ]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(arr), size=min(600, len(arr)), replace=False)
    scatter = [{"actual": float(arr[i, 0]), "predicted": float(arr[i, 1])} for i in idx]
    return BacktestResponse(
        model_version=mv.version,
        scope=scope,
        points=points,
        error_histogram=hist,
        residual_summary={
            "mean": float(resid.mean()),
            "std": float(resid.std()),
            "median": float(np.median(resid)),
            "p1": float(lo),
            "p99": float(hi),
            "n": int(len(resid)),
        },
        scatter_sample=scatter,
    )


def data_end(db: Session) -> date | None:
    return db.scalar(select(func.max(ItemStoreDaily.date)))
