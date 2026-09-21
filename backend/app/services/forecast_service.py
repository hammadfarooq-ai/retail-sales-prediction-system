"""Multi-day forecasts for one item-store series or an entire store (sum of forecastable items)."""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import (
    NotForecastableError,
    PredictionError,
    UnknownProductError,
    UnknownStoreError,
    UnprocessableError,
)
from app.ml.registry import ModelRegistry
from app.models import Product, ProductSales, Store
from app.schemas.analytics import ForecastPoint, ForecastResponse, HistoryPoint
from app.services.prediction_service import INTERVAL_LEVEL, HistoryStore, stores_frame
from ml.src.config import MIN_LAG

log = logging.getLogger(__name__)

# Store-level forecasts sum thousands of series (seconds of compute) and only change when the
# data or model changes, so they are cached briefly in-process. Item-level ones are fast.
_CACHE_TTL_SECONDS = 600
_CACHE_MAX = 64
_cache: dict[tuple, tuple[float, ForecastResponse]] = {}


class ForecastService:
    def __init__(self, db: Session, registry: ModelRegistry, settings: Settings) -> None:
        self.db = db
        self.registry = registry
        self.settings = settings
        self.history = HistoryStore(db)

    def forecast(
        self, store_id: int, item_id: str | None, horizon_days: int, history_days: int
    ) -> ForecastResponse:
        fc = self.registry.require()
        if not 1 <= horizon_days <= self.settings.max_forecast_horizon_days:
            raise UnprocessableError(
                f"horizon_days must be between 1 and {self.settings.max_forecast_horizon_days}"
            )
        if self.db.get(Store, store_id) is None:
            raise UnknownStoreError(f"Store {store_id} does not exist")
        _, last = self.history.date_bounds()
        cache_key = (store_id, item_id, horizon_days, history_days, fc.version, last)
        hit = _cache.get(cache_key) if item_id is None else None
        if hit and time.monotonic() - hit[0] < _CACHE_TTL_SECONDS:
            return hit[1]
        since = last - timedelta(days=max(history_days, 100) - 1)

        if item_id:
            if self.db.get(Product, item_id) is None:
                raise UnknownProductError(f"Product '{item_id}' does not exist")
            flag = self.db.scalar(
                select(ProductSales.in_universe).where(
                    ProductSales.store_id == store_id, ProductSales.item_id == item_id
                )
            )
            if flag is None:
                raise NotForecastableError(f"Product '{item_id}' has no sales in store {store_id}")
            if not flag:
                raise NotForecastableError(
                    "This product-store pair is not forecastable (irregular / low demand)."
                )
            history = self.history.fetch({(store_id, item_id)}, since=since)
            promos = self.history.promo_future(
                {(store_id, item_id)}, None, last + timedelta(days=horizon_days)
            )
        else:
            history = self.history.fetch_store(store_id, since=since)
            promos = self.history.promo_future(None, store_id, last + timedelta(days=horizon_days))
        if history.empty:
            raise NotForecastableError("No history is available for the requested series")

        try:
            pred = fc.forecast(
                history, promos, stores_frame(self.db), last + timedelta(days=horizon_days)
            )
        except Exception as exc:
            log.exception("Forecast failed")
            raise PredictionError("Forecast failed due to an internal error") from exc

        n_series = int(history.groupby(["store_id", "item_id"]).ngroups)
        notes = [
            f"Horizon days 1-{MIN_LAG}: direct model forecast. Later days are recursive "
            "(predictions are fed back as lags) so error grows with horizon."
        ]
        if item_id:
            per_day = pred.sort_values("date")
            fpoints = [
                ForecastPoint(
                    date=r.date.date(),
                    predicted=round(float(r.predicted), 3),
                    lower=None if pd.isna(r.lower) else round(float(r.lower), 3),
                    upper=None if pd.isna(r.upper) else round(float(r.upper), 3),
                    is_recursive=bool(r.is_recursive),
                )
                for r in per_day.itertuples()
            ]
            hist = history.groupby("date")["quantity"].sum()
        else:
            g = pred.groupby("date").agg(
                predicted=("predicted", "sum"), rec=("is_recursive", "max")
            )
            fpoints = [
                ForecastPoint(
                    date=d.date(), predicted=round(float(r.predicted), 3), is_recursive=bool(r.rec)
                )
                for d, r in g.iterrows()
            ]
            hist = history.groupby("date")["quantity"].sum()
            notes.append(
                f"Store total = sum of {n_series} forecastable products only "
                "(not the whole store). "
                "No interval is shown for aggregates."
            )
        hist = hist[hist.index >= pd.Timestamp(last - timedelta(days=history_days - 1))]
        response = ForecastResponse(
            scope="item" if item_id else "store",
            store_id=store_id,
            item_id=item_id,
            model_version=fc.version,
            last_observed_date=last,
            horizon_days=horizon_days,
            n_series=n_series,
            history=[
                HistoryPoint(date=d.date(), actual=round(float(v), 3)) for d, v in hist.items()
            ],
            forecast=fpoints,
            interval_level=INTERVAL_LEVEL if item_id and fc.has_interval else None,
            notes=notes,
        )
        if item_id is None:
            if len(_cache) >= _CACHE_MAX:
                _cache.pop(next(iter(_cache)))
            _cache[cache_key] = (time.monotonic(), response)
        return response
