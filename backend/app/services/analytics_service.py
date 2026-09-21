"""Sales analytics queries over the pre-aggregated tables."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Select, and_, func, select
from sqlalchemy.orm import Session

from app.core.errors import InvalidDateError, UnknownStoreError
from app.models import (
    DailyCategorySales,
    DailyStoreSales,
    Product,
    ProductSales,
    Store,
)
from app.schemas.analytics import (
    CategorySales,
    ProductSalesList,
    ProductSalesOut,
    SalesSummary,
    SalesTrends,
    StoreSales,
    TrendPoint,
)


def data_date_range(db: Session) -> tuple[date, date]:
    lo, hi = db.execute(
        select(func.min(DailyStoreSales.date), func.max(DailyStoreSales.date))
    ).one()
    if lo is None:
        raise InvalidDateError("No sales data loaded. Run the seed step (python -m app.db.seed).")
    return lo, hi


def _validate_range(lo: date, hi: date, start: date | None, end: date | None) -> tuple[date, date]:
    s, e = start or lo, end or hi
    if s > e:
        raise InvalidDateError("start_date must be on or before end_date")
    return s, e


def _check_store(db: Session, store_id: int | None) -> None:
    if store_id is not None and db.get(Store, store_id) is None:
        raise UnknownStoreError(f"Store {store_id} does not exist")


def summary(
    db: Session,
    start: date | None,
    end: date | None,
    store_id: int | None,
    dept_name: str | None,
) -> SalesSummary:
    lo, hi = data_date_range(db)
    s, e = _validate_range(lo, hi, start, end)
    _check_store(db, store_id)

    if dept_name:
        t = DailyCategorySales
        q = select(
            func.coalesce(func.sum(t.quantity), 0.0),
            func.coalesce(func.sum(t.revenue), 0.0),
            func.count(func.distinct(t.date)),
            func.count(func.distinct(t.store_id)),
        ).where(t.date.between(s, e), t.dept_name == dept_name)
        if store_id:
            q = q.where(t.store_id == store_id)
        qty, rev, n_days, n_stores = db.execute(q).one()
        n_dept = 1
    else:
        t2 = DailyStoreSales
        q = select(
            func.coalesce(func.sum(t2.quantity), 0.0),
            func.coalesce(func.sum(t2.revenue), 0.0),
            func.count(func.distinct(t2.date)),
            func.count(func.distinct(t2.store_id)),
        ).where(t2.date.between(s, e))
        if store_id:
            q = q.where(t2.store_id == store_id)
        qty, rev, n_days, n_stores = db.execute(q).one()
        d = select(func.count(func.distinct(DailyCategorySales.dept_name))).where(
            DailyCategorySales.date.between(s, e)
        )
        if store_id:
            d = d.where(DailyCategorySales.store_id == store_id)
        n_dept = db.scalar(d) or 0

    ps = select(func.count(func.distinct(ProductSales.item_id)))
    if store_id:
        ps = ps.where(ProductSales.store_id == store_id)
    if dept_name:
        ps = ps.join(Product, Product.item_id == ProductSales.item_id).where(
            Product.dept_name == dept_name
        )
    n_products = db.scalar(ps) or 0
    n_days = n_days or 0
    return SalesSummary(
        start_date=s,
        end_date=e,
        n_days=n_days,
        total_quantity=float(qty),
        total_revenue=float(rev),
        avg_daily_quantity=float(qty) / n_days if n_days else 0.0,
        avg_daily_revenue=float(rev) / n_days if n_days else 0.0,
        n_stores=n_stores,
        n_products=n_products,
        n_departments=n_dept,
        filters={"store_id": store_id, "dept_name": dept_name},
    )


def _bucket(dialect: str, col, granularity: str):
    """SQL expression that maps a date to the first day of its day/week/month bucket."""
    if granularity == "day":
        return col
    unit = "week" if granularity == "week" else "month"
    if dialect == "postgresql":
        return func.date_trunc(unit, col).cast(col.type)
    # sqlite (tests): week -> Monday, month -> first of month
    if unit == "month":
        return func.date(col, "start of month")
    return func.date(col, "weekday 0", "-6 days")


def trends(
    db: Session,
    granularity: str,
    start: date | None,
    end: date | None,
    store_id: int | None,
    dept_name: str | None,
    group_by: str | None,
) -> SalesTrends:
    lo, hi = data_date_range(db)
    s, e = _validate_range(lo, hi, start, end)
    _check_store(db, store_id)
    dialect = db.get_bind().dialect.name
    t = DailyCategorySales if dept_name else DailyStoreSales
    bucket = _bucket(dialect, t.date, granularity).label("period")
    cols = [bucket, func.sum(t.quantity).label("quantity"), func.sum(t.revenue).label("revenue")]
    q: Select = select(*cols).where(t.date.between(s, e))
    if dept_name:
        q = q.where(t.dept_name == dept_name)
    if store_id:
        q = q.where(t.store_id == store_id)
    if group_by == "store":
        q = q.add_columns(t.store_id).group_by(bucket, t.store_id).order_by(bucket, t.store_id)
    else:
        q = q.group_by(bucket).order_by(bucket)
    rows = db.execute(q).all()
    pts = []
    for r in rows:
        period = r.period if isinstance(r.period, date) else date.fromisoformat(str(r.period))
        pts.append(
            TrendPoint(
                period=period,
                quantity=float(r.quantity),
                revenue=float(r.revenue),
                store_id=getattr(r, "store_id", None) if group_by == "store" else None,
            )
        )
    return SalesTrends(granularity=granularity, group_by=group_by, points=pts)


def by_store(db: Session, start: date | None, end: date | None) -> list[StoreSales]:
    lo, hi = data_date_range(db)
    s, e = _validate_range(lo, hi, start, end)
    t = DailyStoreSales
    rows = db.execute(
        select(
            Store,
            func.coalesce(func.sum(t.quantity), 0.0),
            func.coalesce(func.sum(t.revenue), 0.0),
            func.count(t.date),
        )
        .join(t, and_(t.store_id == Store.store_id, t.date.between(s, e)), isouter=True)
        .group_by(Store.store_id)
        .order_by(Store.store_id)
    ).all()
    prod_counts: dict[int, int] = {
        r[0]: r[1]
        for r in db.execute(
            select(ProductSales.store_id, func.count(ProductSales.item_id)).group_by(
                ProductSales.store_id
            )
        ).all()
    }
    out = []
    for store, qty, rev, days in rows:
        avg = float(rev) / days if days else 0.0
        out.append(
            StoreSales(
                store_id=store.store_id,
                format=store.format,
                city=store.city,
                division=store.division,
                area=store.area,
                quantity=float(qty),
                revenue=float(rev),
                active_days=days,
                avg_daily_revenue=avg,
                avg_daily_revenue_per_m2=avg / store.area if store.area else 0.0,
                n_products=int(prod_counts.get(store.store_id, 0)),
            )
        )
    return out


def by_product(
    db: Session,
    store_id: int | None,
    dept_name: str | None,
    search: str | None,
    sort_by: str,
    limit: int,
    offset: int,
    forecastable_only: bool,
) -> ProductSalesList:
    _check_store(db, store_id)
    ps = ProductSales
    where = []
    if store_id:
        where.append(ps.store_id == store_id)
    if dept_name:
        where.append(Product.dept_name == dept_name)
    if forecastable_only:
        where.append(ps.in_universe.is_(True))
    if search:
        like = f"%{search.strip()}%"
        where.append(
            (Product.item_id.ilike(like))
            | (Product.subclass_name.ilike(like))
            | (Product.class_name.ilike(like))
        )
    base = select(ps, Product).join(Product, Product.item_id == ps.item_id).where(*where)
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    order = ps.quantity if sort_by == "quantity" else ps.revenue
    rows = db.execute(base.order_by(order.desc(), ps.item_id).limit(limit).offset(offset)).all()
    items = [
        ProductSalesOut(
            store_id=r.ProductSales.store_id,
            item_id=r.ProductSales.item_id,
            dept_name=r.Product.dept_name,
            class_name=r.Product.class_name,
            subclass_name=r.Product.subclass_name,
            quantity=r.ProductSales.quantity,
            revenue=r.ProductSales.revenue,
            sale_days=r.ProductSales.sale_days,
            avg_price=r.ProductSales.avg_price,
            first_sale=r.ProductSales.first_sale,
            last_sale=r.ProductSales.last_sale,
            forecastable=bool(r.ProductSales.in_universe),
        )
        for r in rows
    ]
    return ProductSalesList(total=total, items=items)


def by_category(
    db: Session, start: date | None, end: date | None, store_id: int | None, limit: int
) -> list[CategorySales]:
    lo, hi = data_date_range(db)
    s, e = _validate_range(lo, hi, start, end)
    _check_store(db, store_id)
    t = DailyCategorySales
    q = (
        select(
            t.dept_name,
            func.sum(t.quantity).label("quantity"),
            func.sum(t.revenue).label("revenue"),
        )
        .where(t.date.between(s, e))
        .group_by(t.dept_name)
    )
    if store_id:
        q = q.where(t.store_id == store_id)
    rows = db.execute(q.order_by(func.sum(t.revenue).desc()).limit(limit)).all()
    total_q = select(func.coalesce(func.sum(t.revenue), 0.0)).where(t.date.between(s, e))
    if store_id:
        total_q = total_q.where(t.store_id == store_id)
    total = float(db.scalar(total_q) or 0.0)
    pc = (
        select(Product.dept_name, func.count(func.distinct(ProductSales.item_id)))
        .join(Product, Product.item_id == ProductSales.item_id)
        .group_by(Product.dept_name)
    )
    if store_id:
        pc = pc.where(ProductSales.store_id == store_id)
    prod_counts: dict[str, int] = {r[0]: r[1] for r in db.execute(pc).all()}
    return [
        CategorySales(
            dept_name=r.dept_name,
            quantity=float(r.quantity),
            revenue=float(r.revenue),
            revenue_share_pct=float(r.revenue) / total * 100 if total else 0.0,
            n_products=int(prod_counts.get(r.dept_name, 0)),
        )
        for r in rows
    ]
