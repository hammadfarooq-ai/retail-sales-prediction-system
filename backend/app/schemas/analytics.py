"""Analytics / model / forecast response schemas."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class SalesSummary(BaseModel):
    start_date: dt.date
    end_date: dt.date
    n_days: int
    total_quantity: float
    total_revenue: float
    avg_daily_quantity: float
    avg_daily_revenue: float
    n_stores: int
    n_products: int
    n_departments: int
    filters: dict[str, Any]


class TrendPoint(BaseModel):
    period: dt.date = Field(description="First day of the day/week/month bucket")
    quantity: float
    revenue: float
    store_id: int | None = None


class SalesTrends(BaseModel):
    granularity: str
    group_by: str | None = None
    points: list[TrendPoint]


class StoreSales(BaseModel):
    store_id: int
    format: str
    city: str
    division: str
    area: int
    quantity: float
    revenue: float
    active_days: int
    avg_daily_revenue: float
    avg_daily_revenue_per_m2: float
    n_products: int


class ProductSalesOut(BaseModel):
    store_id: int
    item_id: str
    dept_name: str
    class_name: str
    subclass_name: str
    quantity: float
    revenue: float
    sale_days: int
    avg_price: float
    first_sale: dt.date
    last_sale: dt.date
    forecastable: bool


class ProductSalesList(BaseModel):
    period: str = "lifetime"
    total: int
    items: list[ProductSalesOut]


class CategorySales(BaseModel):
    dept_name: str
    quantity: float
    revenue: float
    revenue_share_pct: float
    n_products: int


class ForecastPoint(BaseModel):
    date: dt.date
    predicted: float
    lower: float | None = None
    upper: float | None = None
    is_recursive: bool = False


class HistoryPoint(BaseModel):
    date: dt.date
    actual: float


class ForecastResponse(BaseModel):
    scope: str = Field(description="item | store")
    store_id: int
    item_id: str | None = None
    model_version: str
    last_observed_date: dt.date
    horizon_days: int
    n_series: int = Field(description="Number of item-store series summed (store scope) or 1")
    history: list[HistoryPoint]
    forecast: list[ForecastPoint]
    interval_level: float | None = None
    notes: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    version: str
    model_name: str
    trained_at: dt.datetime
    n_features: int
    feature_names: list[str]
    horizon_days: int
    target: str
    split: dict[str, str]
    n_series: int | None = None
    n_train_rows: int | None = None
    has_prediction_interval: bool
    prediction_interval: dict[str, Any] | None = None
    test_metrics: dict[str, Any]
    last_observed_date: dt.date | None = None


class BacktestPoint(BaseModel):
    date: dt.date
    actual: float
    predicted: float
    baseline_lag7: float | None = None


class HistogramBin(BaseModel):
    lower: float
    upper: float
    count: int


class ModelPerformance(BaseModel):
    version: str
    model_name: str
    trained_at: dt.datetime
    selection_metric: str
    split: dict[str, str]
    n_features: int
    comparison: dict[str, Any]
    final_model_test: dict[str, Any]
    final_model_test_by_store: dict[str, Any]
    final_model_test_aggregated: dict[str, Any]
    improvement_vs_lag7_baseline_pct: dict[str, float]
    feature_importance: dict[str, float]
    prediction_interval: dict[str, Any]


class BacktestResponse(BaseModel):
    model_version: str
    scope: str
    points: list[BacktestPoint]
    error_histogram: list[HistogramBin]
    residual_summary: dict[str, float]
    scatter_sample: list[dict[str, float]]
