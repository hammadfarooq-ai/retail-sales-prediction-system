"""v1 REST endpoints (thin controllers over the service layer)."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query

from app.api.deps import AppSettings, DbSession, Registry
from app.schemas.analytics import (
    BacktestResponse,
    CategorySales,
    ForecastResponse,
    ModelInfo,
    ModelPerformance,
    ProductSalesList,
    SalesSummary,
    SalesTrends,
    StoreSales,
)
from app.schemas.common import ErrorResponse, FiltersResponse, ProductOut, StoreOut
from app.schemas.predict import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionPage,
    PredictionRecord,
    PredictRequest,
    PredictResponse,
)
from app.services import (
    analytics_service,
    catalog_service,
    forecast_service,
    history_service,
    model_service,
    prediction_service,
)

router = APIRouter(prefix="/api/v1")
ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Unknown store / product"},
    422: {"model": ErrorResponse, "description": "Invalid input"},
    503: {"model": ErrorResponse, "description": "Model or database unavailable"},
}

StoreQ = Annotated[int | None, Query(ge=1, description="Filter by store")]
StartQ = Annotated[date | None, Query(description="Inclusive start date")]
EndQ = Annotated[date | None, Query(description="Inclusive end date")]
DeptQ = Annotated[str | None, Query(max_length=128, description="Department (category) name")]

# ----------------------------------------------------------------------------- model
model_router = APIRouter(tags=["model"])


@model_router.get("/model/info", response_model=ModelInfo, responses=ERRORS)
def model_info(db: DbSession, registry: Registry):
    return model_service.model_info(db, registry)


@model_router.get("/model/performance", response_model=ModelPerformance, responses=ERRORS)
def model_performance(db: DbSession, registry: Registry):
    """Model comparison table, final test metrics and feature importance (real training output)."""
    return model_service.performance(db, registry)


@model_router.get("/model/backtest", response_model=BacktestResponse, responses=ERRORS)
def model_backtest(
    db: DbSession,
    store_id: StoreQ = None,
    item_id: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_-]{1,32}$")] = None,
):
    """Actual vs predicted on the held-out test period, plus residual distribution."""
    return model_service.backtest(db, store_id, item_id)


# ----------------------------------------------------------------------------- predict
predict_router = APIRouter(tags=["predict"])


@predict_router.post("/predict", response_model=PredictResponse, responses=ERRORS)
def predict(body: PredictRequest, db: DbSession, registry: Registry, settings: AppSettings):
    """Predict daily quantity for one store x product x date (saved to prediction history)."""
    return prediction_service.PredictionService(db, registry, settings).predict(body)


@predict_router.post("/predict/batch", response_model=BatchPredictResponse, responses=ERRORS)
def predict_batch(
    body: BatchPredictRequest, db: DbSession, registry: Registry, settings: AppSettings
):
    """Predict many rows at once. Invalid rows go to `errors`; valid rows still succeed."""
    return prediction_service.PredictionService(db, registry, settings).predict_batch(body.items)


@predict_router.get(
    "/predictions", response_model=PredictionPage, tags=["history"], responses=ERRORS
)
def list_predictions(
    db: DbSession,
    store_id: StoreQ = None,
    item_id: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_-]{1,32}$")] = None,
    model_version: Annotated[str | None, Query(max_length=64)] = None,
    created_from: date | None = None,
    created_to: date | None = None,
    target_from: date | None = None,
    target_to: date | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return history_service.list_predictions(
        db,
        store_id=store_id,
        item_id=item_id,
        model_version=model_version,
        created_from=created_from,
        created_to=created_to,
        target_from=target_from,
        target_to=target_to,
        limit=limit,
        offset=offset,
    )


@predict_router.get("/predictions/latest", response_model=PredictionRecord | None, tags=["history"])
def latest_prediction(db: DbSession):
    return history_service.latest_prediction(db)


@predict_router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecord,
    tags=["history"],
    responses=ERRORS,
)
def get_prediction(prediction_id: int, db: DbSession):
    return history_service.get_prediction(db, prediction_id)


# ----------------------------------------------------------------------------- sales
sales_router = APIRouter(prefix="/sales", tags=["sales"])


@sales_router.get("/summary", response_model=SalesSummary, responses=ERRORS)
def sales_summary(
    db: DbSession,
    start_date: StartQ = None,
    end_date: EndQ = None,
    store_id: StoreQ = None,
    dept_name: DeptQ = None,
):
    return analytics_service.summary(db, start_date, end_date, store_id, dept_name)


@sales_router.get("/trends", response_model=SalesTrends, responses=ERRORS)
def sales_trends(
    db: DbSession,
    granularity: Annotated[Literal["day", "week", "month"], Query()] = "day",
    group_by: Annotated[Literal["store"] | None, Query()] = None,
    start_date: StartQ = None,
    end_date: EndQ = None,
    store_id: StoreQ = None,
    dept_name: DeptQ = None,
):
    return analytics_service.trends(
        db, granularity, start_date, end_date, store_id, dept_name, group_by
    )


@sales_router.get("/by-store", response_model=list[StoreSales], responses=ERRORS)
def sales_by_store(db: DbSession, start_date: StartQ = None, end_date: EndQ = None):
    return analytics_service.by_store(db, start_date, end_date)


@sales_router.get("/by-product", response_model=ProductSalesList, responses=ERRORS)
def sales_by_product(
    db: DbSession,
    store_id: StoreQ = None,
    dept_name: DeptQ = None,
    search: Annotated[str | None, Query(max_length=64)] = None,
    sort_by: Annotated[Literal["revenue", "quantity"], Query()] = "revenue",
    forecastable_only: bool = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Lifetime totals per store x product (date filters are not supported at product level)."""
    return analytics_service.by_product(
        db, store_id, dept_name, search, sort_by, limit, offset, forecastable_only
    )


@sales_router.get("/by-category", response_model=list[CategorySales], responses=ERRORS)
def sales_by_category(
    db: DbSession,
    start_date: StartQ = None,
    end_date: EndQ = None,
    store_id: StoreQ = None,
    limit: Annotated[int, Query(ge=1, le=300)] = 20,
):
    return analytics_service.by_category(db, start_date, end_date, store_id, limit)


# ----------------------------------------------------------------------------- forecasts / catalog
forecast_router = APIRouter(tags=["forecast"])


@forecast_router.get("/forecasts", response_model=ForecastResponse, responses=ERRORS)
def forecasts(
    db: DbSession,
    registry: Registry,
    settings: AppSettings,
    store_id: Annotated[int, Query(ge=1)],
    item_id: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_-]{1,32}$")] = None,
    horizon_days: Annotated[int, Query(ge=1, le=90)] = 14,
    history_days: Annotated[int, Query(ge=7, le=180)] = 60,
):
    """Historical + forecast series for one product (with 80% interval for days 1-7) or a store."""
    return forecast_service.ForecastService(db, registry, settings).forecast(
        store_id, item_id, horizon_days, history_days
    )


catalog_router = APIRouter(tags=["catalog"])


@catalog_router.get("/stores", response_model=list[StoreOut])
def stores(db: DbSession):
    return catalog_service.list_stores(db)


@catalog_router.get("/meta/filters", response_model=FiltersResponse)
def meta_filters(db: DbSession, settings: AppSettings):
    return catalog_service.filters(db, settings)


@catalog_router.get("/products", response_model=list[ProductOut])
def products(
    db: DbSession,
    q: Annotated[str | None, Query(max_length=64)] = None,
    store_id: StoreQ = None,
    dept_name: DeptQ = None,
    forecastable_only: bool = True,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
):
    return catalog_service.search_products(db, q, store_id, dept_name, forecastable_only, limit)


router.include_router(model_router)
router.include_router(predict_router)
router.include_router(sales_router)
router.include_router(forecast_router)
router.include_router(catalog_router)
