"""Prediction request/response schemas."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import ORMModel

ITEM_ID_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"


class PredictRequest(BaseModel):
    store_id: int = Field(ge=1, le=32767, description="Store identifier", examples=[1])
    item_id: str = Field(
        pattern=ITEM_ID_PATTERN, description="Product identifier", examples=["63161948a95a"]
    )
    date: dt.date = Field(description="Day to predict (YYYY-MM-DD)", examples=["2024-09-30"])
    price: float | None = Field(
        default=None,
        gt=0,
        le=1_000_000,
        description="Planned shelf price on that day. Default: last observed price.",
    )
    promotion: bool | None = Field(
        default=None,
        description="Force promotion on/off. Default: use the promo calendar for that day.",
    )
    discount_pct: float | None = Field(
        default=None,
        ge=0,
        le=95,
        description="Discount depth in percent (only meaningful when promotion is on).",
    )

    @field_validator("date")
    @classmethod
    def _sane_year(cls, v: dt.date) -> dt.date:
        if v.year < 2000 or v.year > 2100:
            raise ValueError("date must be between years 2000 and 2100")
        return v

    @model_validator(mode="after")
    def _discount_needs_promo(self) -> PredictRequest:
        if self.discount_pct and self.promotion is False:
            raise ValueError("discount_pct cannot be set when promotion is false")
        return self


class EffectiveInputs(BaseModel):
    price: float | None = Field(description="Price used by the model")
    last_observed_price: float | None
    promotion: bool
    discount_pct: float
    promotion_source: Literal["user", "promo_calendar", "none"]


class PredictResponse(BaseModel):
    id: int | None = Field(default=None, description="Prediction history id")
    model_version: str
    store_id: int
    item_id: str
    dept_name: str | None = None
    subclass_name: str | None = None
    date: dt.date
    predicted_quantity: float
    lower: float | None = Field(default=None, description="10th percentile (80% interval)")
    upper: float | None = Field(default=None, description="90th percentile (80% interval)")
    interval_level: float | None = None
    mode: Literal["in_sample", "held_out_test", "forecast"] = Field(
        description="in_sample: model was trained on that day; held_out_test: unseen day with "
        "known actual; forecast: after the last observed day"
    )
    horizon_days: int = Field(description="Days after the last observed day (<=0 for past days)")
    is_recursive: bool = Field(description="True when horizon > 7 (predictions fed back as lags)")
    actual_quantity: float | None = None
    inputs: EffectiveInputs
    created_at: dt.datetime | None = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1)


class BatchError(BaseModel):
    index: int
    code: str
    message: str


class BatchPredictResponse(BaseModel):
    batch_id: str
    model_version: str
    n_requested: int
    n_succeeded: int
    results: list[PredictResponse]
    errors: list[BatchError]


class PredictionRecord(ORMModel):
    id: int
    created_at: dt.datetime
    model_version: str
    source: str
    batch_id: str | None
    store_id: int
    item_id: str
    target_date: dt.date
    inputs: dict
    predicted_quantity: float
    lower: float | None
    upper: float | None
    actual_quantity: float | None
    is_recursive: bool


class PredictionPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionRecord]
