"""Response schemas shared by several endpoints."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict] | dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    version: str
    database: str = Field(description="up | down")
    model_loaded: bool
    model_version: str | None = None
    time: dt.datetime


class StoreOut(ORMModel):
    store_id: int
    division: str
    format: str
    city: str
    area: int


class ProductOut(ORMModel):
    item_id: str
    dept_name: str
    class_name: str
    subclass_name: str
    item_type: str | None = None
    forecastable_stores: list[int] = Field(default_factory=list)


class FiltersResponse(BaseModel):
    stores: list[StoreOut]
    departments: list[str]
    min_date: dt.date
    max_date: dt.date
    max_forecast_horizon_days: int
    forecastable_pairs: int
