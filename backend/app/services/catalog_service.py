"""Stores, departments and product lookups for filters/dropdowns."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import DailyStoreSales, Product, ProductSales, Store
from app.schemas.common import FiltersResponse, ProductOut, StoreOut


def list_stores(db: Session) -> list[StoreOut]:
    return [StoreOut.model_validate(s) for s in db.scalars(select(Store).order_by(Store.store_id))]


def filters(db: Session, settings: Settings) -> FiltersResponse:
    lo, hi = db.execute(
        select(func.min(DailyStoreSales.date), func.max(DailyStoreSales.date))
    ).one()
    depts = db.scalars(select(Product.dept_name).distinct().order_by(Product.dept_name)).all()
    pairs = (
        db.scalar(select(func.count()).select_from(ProductSales).where(ProductSales.in_universe))
        or 0
    )
    return FiltersResponse(
        stores=list_stores(db),
        departments=list(depts),
        min_date=lo,
        max_date=hi,
        max_forecast_horizon_days=settings.max_forecast_horizon_days,
        forecastable_pairs=pairs,
    )


def search_products(
    db: Session,
    q: str | None,
    store_id: int | None,
    dept_name: str | None,
    forecastable_only: bool,
    limit: int,
) -> list[ProductOut]:
    """Products ranked by revenue, with the stores in which each is forecastable."""
    ps = ProductSales
    stmt = select(Product, func.sum(ps.revenue).label("rev")).join(
        ps, ps.item_id == Product.item_id
    )
    if store_id:
        stmt = stmt.where(ps.store_id == store_id)
    if forecastable_only:
        stmt = stmt.where(ps.in_universe.is_(True))
    if dept_name:
        stmt = stmt.where(Product.dept_name == dept_name)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(
            Product.item_id.ilike(like)
            | Product.subclass_name.ilike(like)
            | Product.class_name.ilike(like)
        )
    rows = db.execute(
        stmt.group_by(Product.item_id).order_by(func.sum(ps.revenue).desc()).limit(limit)
    ).all()
    ids = [r.Product.item_id for r in rows]
    stores_by_item: dict[str, list[int]] = {}
    if ids:
        f = select(ps.item_id, ps.store_id).where(ps.item_id.in_(ids))
        if forecastable_only:
            f = f.where(ps.in_universe.is_(True))
        for item_id, store in db.execute(f).all():
            stores_by_item.setdefault(item_id, []).append(store)
    return [
        ProductOut(
            item_id=r.Product.item_id,
            dept_name=r.Product.dept_name,
            class_name=r.Product.class_name,
            subclass_name=r.Product.subclass_name,
            item_type=r.Product.item_type,
            forecastable_stores=sorted(stores_by_item.get(r.Product.item_id, [])),
        )
        for r in rows
    ]
