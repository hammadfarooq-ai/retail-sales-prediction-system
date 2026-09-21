"""ORM schema.

Design notes
  * Only *derived* tables are stored; the 379 MB raw sales file is not duplicated. Aggregates
    (daily store / category / product) power the analytics endpoints; ``item_store_daily`` keeps
    the recent zero-filled history of the forecastable (store, product) universe, which is what
    the model needs to compute lag features at request time.
  * ``predictions`` is the append-only prediction history.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Store(Base):
    __tablename__ = "stores"
    store_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    division: Mapped[str] = mapped_column(String(32))
    format: Mapped[str] = mapped_column(String(64))
    city: Mapped[str] = mapped_column(String(64))
    area: Mapped[int] = mapped_column(Integer)


class Product(Base):
    __tablename__ = "products"
    item_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dept_name: Mapped[str] = mapped_column(String(128), index=True)
    class_name: Mapped[str] = mapped_column(String(128))
    subclass_name: Mapped[str] = mapped_column(String(128))
    item_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dept_code: Mapped[int] = mapped_column(SmallInteger)
    class_code: Mapped[int] = mapped_column(SmallInteger)


class DailyStoreSales(Base):
    __tablename__ = "daily_store_sales"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.store_id"), primary_key=True)
    quantity: Mapped[float] = mapped_column(Float)
    revenue: Mapped[float] = mapped_column(Float)
    n_products: Mapped[int] = mapped_column(Integer)


class DailyCategorySales(Base):
    __tablename__ = "daily_category_sales"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.store_id"), primary_key=True)
    dept_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    quantity: Mapped[float] = mapped_column(Float)
    revenue: Mapped[float] = mapped_column(Float)
    n_products: Mapped[int] = mapped_column(Integer)
    __table_args__ = (Index("ix_dcs_dept_date", "dept_name", "date"),)


class ProductSales(Base):
    """Lifetime totals per (store, product)."""

    __tablename__ = "product_sales"
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.store_id"), primary_key=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("products.item_id"), primary_key=True)
    quantity: Mapped[float] = mapped_column(Float)
    revenue: Mapped[float] = mapped_column(Float, index=True)
    sale_days: Mapped[int] = mapped_column(Integer)
    first_sale: Mapped[date] = mapped_column(Date)
    last_sale: Mapped[date] = mapped_column(Date)
    avg_price: Mapped[float] = mapped_column(Float)
    in_universe: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class ItemStoreDaily(Base):
    """Recent zero-filled daily history for the forecastable universe (model input)."""

    __tablename__ = "item_store_daily"
    store_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    quantity: Mapped[float] = mapped_column(Float)
    price_obs: Mapped[float | None] = mapped_column(Float, nullable=True)
    promo_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    promo_before_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    promo_type: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (Index("ix_isd_date", "date"),)


class PromoCalendar(Base):
    """Planned promotions dated after the last observed sales day."""

    __tablename__ = "promo_calendar"
    store_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    promo_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    promo_before_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    promo_type: Mapped[float | None] = mapped_column(Float, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    n_features: Mapped[int] = mapped_column(Integer)
    split: Mapped[dict] = mapped_column(JSONType)
    metrics: Mapped[dict] = mapped_column(JSONType)
    params: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BacktestPrediction(Base):
    """Model output on the held-out test period, for actual-vs-predicted charts."""

    __tablename__ = "backtest_predictions"
    store_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    actual: Mapped[float] = mapped_column(Float)
    predicted: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_lag7: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)


class Prediction(Base):
    """Prediction request/result history."""

    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16), default="single")  # single | batch
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    store_id: Mapped[int] = mapped_column(SmallInteger, index=True)
    item_id: Mapped[str] = mapped_column(String(32), index=True)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    inputs: Mapped[dict] = mapped_column(JSONType)
    predicted_quantity: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_recursive: Mapped[bool] = mapped_column(Boolean, default=False)
