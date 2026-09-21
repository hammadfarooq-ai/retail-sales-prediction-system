"""Backend test fixtures: SQLite in-memory DB + a real tiny trained model (no PostgreSQL)."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "backend", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import json  # noqa: E402

import synthetic_data as sd  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_session, make_engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.models import (  # noqa: E402
    BacktestPrediction,
    DailyCategorySales,
    DailyStoreSales,
    ItemStoreDaily,
    ModelVersion,
    Product,
    ProductSales,
    PromoCalendar,
    Store,
)
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

LAST = sd.LAST_DATE.date()


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    return sd.write_tiny_artifacts(tmp_path_factory.mktemp("artifacts"))


def _populate(db: Session, artifacts_dir: Path) -> None:
    panel = sd.make_panel()
    for r in sd.STORES.itertuples():
        db.add(
            Store(
                store_id=r.store_id, division=r.division, format=r.format, city=r.city, area=r.area
            )
        )
    names = {
        "aaa111": ("DAIRY", "MILK"),
        "bbb222": ("DAIRY", "CHEESE"),
        "ccc333": ("BAKERY", "BREAD"),
        "zzz999": ("BAKERY", "RARE"),
    }
    for i, (item, (dept, sub)) in enumerate(names.items()):
        db.add(
            Product(
                item_id=item,
                dept_name=dept,
                class_name=dept + "-C",
                subclass_name=sub,
                item_type=None,
                dept_code=min(i, 2),
                class_code=min(i, 2),
            )
        )
    db.flush()
    for store in (1, 2):
        for item in [*sd.ITEMS, "zzz999"]:
            g = (
                panel[(panel.store_id == store) & (panel.item_id == item)]
                if item != "zzz999"
                else None
            )
            qty = float(g["quantity"].sum()) if g is not None else 5.0
            db.add(
                ProductSales(
                    store_id=store,
                    item_id=item,
                    quantity=qty,
                    revenue=qty * 100,
                    sale_days=100,
                    first_sale=date(2024, 1, 1),
                    last_sale=LAST,
                    avg_price=100.0,
                    in_universe=item != "zzz999",
                )
            )
    for r in panel.itertuples():
        db.add(
            ItemStoreDaily(
                store_id=int(r.store_id),
                item_id=r.item_id,
                date=r.date.date(),
                quantity=float(r.quantity),
                price_obs=None if pd.isna(r.price_obs) else float(r.price_obs),
                promo_price=None if pd.isna(r.promo_price) else float(r.promo_price),
                promo_before_price=(
                    None if pd.isna(r.promo_before_price) else float(r.promo_before_price)
                ),
                promo_type=None if pd.isna(r.promo_type) else float(r.promo_type),
            )
        )
    db.add(
        PromoCalendar(
            store_id=1,
            item_id="aaa111",
            date=date.fromordinal(LAST.toordinal() + 2),
            promo_price=70.0,
            promo_before_price=100.0,
            promo_type=5.0,
        )
    )
    daily = (
        panel.groupby(["date", "store_id"])
        .agg(quantity=("quantity", "sum"), n=("item_id", "nunique"))
        .reset_index()
    )
    for r in daily.itertuples():
        db.add(
            DailyStoreSales(
                date=r.date.date(),
                store_id=int(r.store_id),
                quantity=float(r.quantity),
                revenue=float(r.quantity) * 100,
                n_products=int(r.n),
            )
        )
    prod_dept = {"aaa111": "DAIRY", "bbb222": "DAIRY", "ccc333": "BAKERY"}
    cat = (
        panel.assign(dept=panel["item_id"].map(prod_dept))
        .groupby(["date", "store_id", "dept"])
        .agg(quantity=("quantity", "sum"), n=("item_id", "nunique"))
        .reset_index()
    )
    for r in cat.itertuples():
        db.add(
            DailyCategorySales(
                date=r.date.date(),
                store_id=int(r.store_id),
                dept_name=r.dept,
                quantity=float(r.quantity),
                revenue=float(r.quantity) * 100,
                n_products=int(r.n),
            )
        )
    meta = json.loads((artifacts_dir / "model_metadata.json").read_text())
    metrics = json.loads((artifacts_dir / "metrics.json").read_text())
    db.add(
        ModelVersion(
            version=meta["version"],
            model_name=meta["model_name"],
            trained_at=datetime.fromisoformat(meta["trained_at"]),
            is_active=True,
            n_features=meta["n_features"],
            split=meta["split"],
            metrics=metrics,
            params={},
        )
    )
    for d in pd.date_range(LAST - pd.Timedelta(days=5), LAST):
        db.add(
            BacktestPrediction(
                store_id=1,
                item_id="aaa111",
                date=d.date(),
                actual=5.0,
                predicted=6.0,
                lower=3.0,
                upper=9.0,
                baseline_lag7=4.0,
                model_version=meta["version"],
            )
        )
        db.add(
            BacktestPrediction(
                store_id=2,
                item_id="bbb222",
                date=d.date(),
                actual=8.0,
                predicted=7.0,
                lower=4.0,
                upper=10.0,
                baseline_lag7=9.0,
                model_version=meta["version"],
            )
        )
    db.commit()


@pytest.fixture()
def db_session_factory(artifacts_dir):
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory() as s:
        _populate(s, artifacts_dir)
    yield factory
    engine.dispose()


@pytest.fixture()
def db(db_session_factory):
    with db_session_factory() as s:
        yield s


@pytest.fixture()
def registry(artifacts_dir) -> ModelRegistry:
    reg = ModelRegistry()
    reg.load(artifacts_dir)
    assert reg.loaded, reg.error
    return reg


@pytest.fixture()
def client(db_session_factory, registry):
    app = create_app()
    app.state.registry = registry

    def _get_session():
        with db_session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _get_session
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.registry = registry  # lifespan would try to load the (missing) real artifacts
        yield c


@pytest.fixture()
def client_no_model(db_session_factory):
    app = create_app()

    def _get_session():
        with db_session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _get_session
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.registry = ModelRegistry()  # nothing loaded
        yield c


PAST = str(date.fromordinal(LAST.toordinal() - 10))
NEXT_DAY = str(date.fromordinal(LAST.toordinal() + 1))
IN_3 = str(date.fromordinal(LAST.toordinal() + 3))
IN_14 = str(date.fromordinal(LAST.toordinal() + 14))
_ = UTC
