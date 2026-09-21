"""Validated single / batch prediction with persistence to the prediction history."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import (
    AppError,
    InvalidDateError,
    NotForecastableError,
    PredictionError,
    UnknownProductError,
    UnknownStoreError,
    UnprocessableError,
)
from app.ml.registry import ModelRegistry
from app.models import ItemStoreDaily, Prediction, Product, ProductSales, PromoCalendar, Store
from app.schemas.predict import (
    BatchError,
    BatchPredictResponse,
    EffectiveInputs,
    PredictRequest,
    PredictResponse,
)
from ml.src.config import MIN_HISTORY_DAYS
from ml.src.models.forecaster import Forecaster, Override

log = logging.getLogger(__name__)
INTERVAL_LEVEL = 0.8


@dataclass
class _Valid:
    index: int
    req: PredictRequest
    product: Product


class HistoryStore:
    """Reads the zero-filled item-store history the model needs for lag features."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def date_bounds(self) -> tuple[date, date]:
        lo, hi = self.db.execute(
            select(func.min(ItemStoreDaily.date), func.max(ItemStoreDaily.date))
        ).one()
        if lo is None:
            raise InvalidDateError(
                "No item history loaded. Run the seed step (python -m app.db.seed)."
            )
        return lo, hi

    def fetch(self, pairs: set[tuple[int, str]], since: date | None = None) -> pd.DataFrame:
        t = ItemStoreDaily
        by_store: dict[int, list[str]] = {}
        for s, i in pairs:
            by_store.setdefault(s, []).append(i)
        cond = or_(*[and_(t.store_id == s, t.item_id.in_(items)) for s, items in by_store.items()])
        q = (
            select(
                t.store_id,
                t.item_id,
                t.date,
                t.quantity,
                t.price_obs,
                t.promo_price,
                t.promo_before_price,
                t.promo_type,
                Product.dept_code,
                Product.class_code,
            )
            .join(Product, Product.item_id == t.item_id)
            .where(cond)
        )
        if since:
            q = q.where(t.date >= since)
        return self._read(q.order_by(t.store_id, t.item_id, t.date))

    def fetch_store(self, store_id: int, since: date | None = None) -> pd.DataFrame:
        t = ItemStoreDaily
        q = (
            select(
                t.store_id,
                t.item_id,
                t.date,
                t.quantity,
                t.price_obs,
                t.promo_price,
                t.promo_before_price,
                t.promo_type,
                Product.dept_code,
                Product.class_code,
            )
            .join(Product, Product.item_id == t.item_id)
            .where(t.store_id == store_id)
        )
        if since:
            q = q.where(t.date >= since)
        return self._read(q.order_by(t.item_id, t.date))

    def _read(self, stmt) -> pd.DataFrame:
        """Bulk read into a DataFrame (much faster than materialising ORM rows)."""
        return self._frame(pd.read_sql_query(stmt, self.db.connection()))

    @staticmethod
    def _frame(raw: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "store_id",
            "item_id",
            "date",
            "quantity",
            "price_obs",
            "promo_price",
            "promo_before_price",
            "promo_type",
            "dept_code",
            "class_code",
        ]
        df = raw[cols].copy()
        df["date"] = pd.to_datetime(df["date"])
        for c in ("quantity", "price_obs", "promo_price", "promo_before_price", "promo_type"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df["store_id"] = df["store_id"].astype("int8")
        return df

    def promo_future(
        self, pairs: set[tuple[int, str]] | None, store_id: int | None, end: date
    ) -> pd.DataFrame:
        t = PromoCalendar
        q = select(t.store_id, t.item_id, t.date, t.promo_price, t.promo_before_price, t.promo_type)
        q = q.where(t.date <= end)
        if store_id is not None:
            q = q.where(t.store_id == store_id)
        elif pairs:
            by_store: dict[int, list[str]] = {}
            for s, i in pairs:
                by_store.setdefault(s, []).append(i)
            q = q.where(
                or_(*[and_(t.store_id == s, t.item_id.in_(v)) for s, v in by_store.items()])
            )
        df = pd.DataFrame(
            [tuple(r) for r in self.db.execute(q).all()],
            columns=[
                "store_id",
                "item_id",
                "date",
                "promo_price",
                "promo_before_price",
                "promo_type",
            ],
        )
        df["date"] = pd.to_datetime(df["date"])
        df["store_id"] = df["store_id"].astype("int8")
        for c in ("promo_price", "promo_before_price", "promo_type"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        return df


def stores_frame(db: Session) -> pd.DataFrame:
    rows = db.execute(
        select(Store.store_id, Store.division, Store.format, Store.city, Store.area)
    ).all()
    return pd.DataFrame(
        [tuple(r) for r in rows], columns=["store_id", "division", "format", "city", "area"]
    )


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


class PredictionService:
    def __init__(self, db: Session, registry: ModelRegistry, settings: Settings) -> None:
        self.db = db
        self.registry = registry
        self.settings = settings
        self.history = HistoryStore(db)

    # ------------------------------------------------------------------ validation
    def _validate(self, reqs: list[PredictRequest], hist_lo: date, hist_hi: date):
        store_ids = set(self.db.scalars(select(Store.store_id)).all())
        item_ids = {r.item_id for r in reqs}
        products = {
            p.item_id: p
            for p in self.db.scalars(select(Product).where(Product.item_id.in_(item_ids))).all()
        }
        pair_flags: dict[tuple[int, str], bool] = {}
        by_store: dict[int, set[str]] = {}
        for r in reqs:
            by_store.setdefault(r.store_id, set()).add(r.item_id)
        for s, items in by_store.items():
            if s not in store_ids:
                continue
            for row in self.db.execute(
                select(ProductSales.store_id, ProductSales.item_id, ProductSales.in_universe).where(
                    ProductSales.store_id == s, ProductSales.item_id.in_(items)
                )
            ).all():
                pair_flags[(row[0], row[1])] = bool(row[2])

        earliest = hist_lo + timedelta(days=MIN_HISTORY_DAYS)
        latest = hist_hi + timedelta(days=self.settings.max_forecast_horizon_days)
        valid: list[_Valid] = []
        errors: dict[int, AppError] = {}
        for idx, r in enumerate(reqs):
            if r.store_id not in store_ids:
                errors[idx] = UnknownStoreError(f"Store {r.store_id} does not exist")
            elif r.item_id not in products:
                errors[idx] = UnknownProductError(f"Product '{r.item_id}' does not exist")
            elif (r.store_id, r.item_id) not in pair_flags:
                errors[idx] = NotForecastableError(
                    f"Product '{r.item_id}' has no sales in store {r.store_id}"
                )
            elif not pair_flags[(r.store_id, r.item_id)]:
                errors[idx] = NotForecastableError(
                    "This product-store pair is not forecastable: the model covers items "
                    "with regular "
                    "demand only (sold on >=70% of days since first sale and >=90 sale days in the "
                    "training period)."
                )
            elif not (earliest <= r.date <= latest):
                errors[idx] = InvalidDateError(
                    f"Date {r.date} is outside the supported range {earliest} to {latest}",
                    details={"min_date": str(earliest), "max_date": str(latest)},
                )
            else:
                valid.append(_Valid(idx, r, products[r.item_id]))
        return valid, errors

    # ------------------------------------------------------------------ core
    def _run(self, reqs: list[PredictRequest], source: str, batch_id: str | None):
        fc: Forecaster = self.registry.require()
        hist_lo, hist_hi = self.history.date_bounds()
        valid, errors = self._validate(reqs, hist_lo, hist_hi)
        results: dict[int, PredictResponse] = {}
        if valid:
            try:
                results = self._predict_valid(fc, valid, hist_hi, errors)
            except AppError:
                raise
            except Exception as exc:
                log.exception("Prediction failed")
                raise PredictionError("Prediction failed due to an internal error") from exc
            self._persist(results, [v for v in valid if v.index in results], source, batch_id)
        return results, errors

    def _predict_valid(
        self, fc: Forecaster, valid: list[_Valid], hist_hi: date, errors: dict[int, AppError]
    ):
        pairs = {(v.req.store_id, v.req.item_id) for v in valid}
        history = self.history.fetch(
            pairs, since=hist_hi - timedelta(days=self.settings.history_days_in_db)
        )
        first = history.groupby(["store_id", "item_id"])["date"].min().dt.date.to_dict()
        ok: list[_Valid] = []
        for v in valid:
            f = first.get((v.req.store_id, v.req.item_id))
            if f is None:
                errors[v.index] = NotForecastableError("No recent history is stored for this pair")
            elif v.req.date <= hist_hi and v.req.date < f + timedelta(days=MIN_HISTORY_DAYS):
                errors[v.index] = InvalidDateError(
                    f"Not enough history before {v.req.date} for this pair "
                    f"(history starts {f}; earliest supported date "
                    f"{f + timedelta(days=MIN_HISTORY_DAYS)})"
                )
            else:
                ok.append(v)
        valid = ok
        if not valid:
            return {}
        pairs = {(v.req.store_id, v.req.item_id) for v in valid}
        history = history.merge(
            pd.DataFrame(list(pairs), columns=["store_id", "item_id"]).astype({"store_id": "int8"}),
            on=["store_id", "item_id"],
        )
        stores = stores_frame(self.db)
        overrides: dict[tuple[int, str, date], Override] = {}
        for v in valid:
            r = v.req
            if r.price is not None or r.promotion is not None or r.discount_pct is not None:
                promo = (
                    r.promotion if r.promotion is not None else (True if r.discount_pct else None)
                )
                overrides[(r.store_id, r.item_id, r.date)] = Override(
                    r.price, promo, r.discount_pct
                )

        past_dates = {v.req.date for v in valid if v.req.date <= hist_hi}
        future_end = max((v.req.date for v in valid if v.req.date > hist_hi), default=None)
        frames = []
        if past_dates:
            past_pairs = {(v.req.store_id, v.req.item_id) for v in valid if v.req.date <= hist_hi}
            h = history.merge(
                pd.DataFrame(list(past_pairs), columns=["store_id", "item_id"]).astype(
                    {"store_id": "int8"}
                ),
                on=["store_id", "item_id"],
            )
            frames.append(fc.predict_history_dates(h, stores, past_dates, overrides))
        if future_end is not None:
            fut_pairs = {(v.req.store_id, v.req.item_id) for v in valid if v.req.date > hist_hi}
            h = history.merge(
                pd.DataFrame(list(fut_pairs), columns=["store_id", "item_id"]).astype(
                    {"store_id": "int8"}
                ),
                on=["store_id", "item_id"],
            )
            promos = self.history.promo_future(fut_pairs, None, future_end)
            frames.append(fc.forecast(h, promos, stores, future_end, overrides))
        pred = pd.concat(frames, ignore_index=True)
        pred["d"] = pred["date"].dt.date
        pred = pred.set_index(["store_id", "item_id", "d"]).sort_index()
        actual = history.assign(d=history["date"].dt.date).set_index(["store_id", "item_id", "d"])[
            "quantity"
        ]

        split = fc.metadata["split"]
        val_end = date.fromisoformat(split["val_end"])
        out: dict[int, PredictResponse] = {}
        for v in valid:
            r = v.req
            key = (r.store_id, r.item_id, r.date)
            row = pred.loc[key]  # type: ignore[index]
            is_future = r.date > hist_hi
            mode = (
                "forecast" if is_future else ("in_sample" if r.date <= val_end else "held_out_test")
            )
            lo, hi = _f(row["lower"]), _f(row["upper"])
            user_set = (r.promotion is not None) or bool(r.discount_pct)
            promo_on = bool(row["effective_promo"])
            out[v.index] = PredictResponse(
                model_version=fc.version,
                store_id=r.store_id,
                item_id=r.item_id,
                dept_name=v.product.dept_name,
                subclass_name=v.product.subclass_name,
                date=r.date,
                predicted_quantity=round(float(row["predicted"]), 3),
                lower=None if lo is None else round(lo, 3),
                upper=None if hi is None else round(hi, 3),
                interval_level=INTERVAL_LEVEL if lo is not None else None,
                mode=mode,
                horizon_days=(r.date - hist_hi).days,
                is_recursive=bool(row["is_recursive"]),
                actual_quantity=None if is_future else _f(actual.get(key)),
                inputs=EffectiveInputs(
                    price=_f(row["effective_price"]),
                    last_observed_price=_f(row["last_price"]),
                    promotion=promo_on,
                    discount_pct=round(float(row["effective_discount_pct"]), 2),
                    promotion_source=(
                        "user" if user_set else ("promo_calendar" if promo_on else "none")
                    ),
                ),
            )
        return out

    def _persist(
        self,
        results: dict[int, PredictResponse],
        valid: list[_Valid],
        source: str,
        batch_id: str | None,
    ):
        rows: list[tuple[int, Prediction]] = []
        for v in valid:
            res = results[v.index]
            row = Prediction(
                model_version=res.model_version,
                source=source,
                batch_id=batch_id,
                store_id=res.store_id,
                item_id=res.item_id,
                target_date=res.date,
                inputs={
                    "request": v.req.model_dump(mode="json"),
                    "effective": res.inputs.model_dump(mode="json"),
                    "mode": res.mode,
                    "horizon_days": res.horizon_days,
                },
                predicted_quantity=res.predicted_quantity,
                lower=res.lower,
                upper=res.upper,
                actual_quantity=res.actual_quantity,
                is_recursive=res.is_recursive,
            )
            self.db.add(row)
            rows.append((v.index, row))
        self.db.commit()
        for idx, row in rows:
            results[idx].id = row.id
            results[idx].created_at = row.created_at

    # ------------------------------------------------------------------ public
    def predict(self, req: PredictRequest) -> PredictResponse:
        results, errors = self._run([req], "single", None)
        if errors:
            raise errors[0]
        return results[0]

    def predict_batch(self, reqs: list[PredictRequest]) -> BatchPredictResponse:
        if len(reqs) > self.settings.max_batch_size:
            raise UnprocessableError(
                f"Batch too large: {len(reqs)} items (max {self.settings.max_batch_size})"
            )
        batch_id = str(uuid.uuid4())
        results, errors = self._run(reqs, "batch", batch_id)
        fc = self.registry.require()
        return BatchPredictResponse(
            batch_id=batch_id,
            model_version=fc.version,
            n_requested=len(reqs),
            n_succeeded=len(results),
            results=[results[i] for i in sorted(results)],
            errors=[
                BatchError(index=i, code=e.code, message=e.message)
                for i, e in sorted(errors.items())
            ],
        )
