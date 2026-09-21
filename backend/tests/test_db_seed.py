"""Database operations: seeding derived tables, idempotency, and preserved prediction history."""

from __future__ import annotations

import pandas as pd
import pytest
import synthetic_data as sd
from app.db.base import Base
from app.db.seed import seed_database
from app.db.session import make_engine
from app.models import ItemStoreDaily, ModelVersion, Prediction, Product, ProductSales, Store
from sqlalchemy import func, select
from sqlalchemy.orm import Session


@pytest.fixture()
def processed(tmp_path):
    panel = sd.make_panel()
    panel["dept_code"] = 0
    p = tmp_path
    sd.STORES.to_parquet(p / "stores.parquet", index=False)
    pd.DataFrame(
        {
            "item_id": sd.ITEMS,
            "dept_name": ["D1", "D1", "D2"],
            "class_name": ["C1", "C1", "C2"],
            "subclass_name": ["S1", "S2", "S3"],
            "item_type": [None, None, None],
            "dept_code": [0, 0, 1],
            "class_code": [0, 0, 1],
        }
    ).to_parquet(p / "products.parquet", index=False)
    daily = panel.groupby(["date", "store_id"]).agg(quantity=("quantity", "sum")).reset_index()
    daily["revenue"] = daily["quantity"] * 10
    daily["n_products"] = 3
    daily.to_parquet(p / "daily_store.parquet", index=False)
    cat = daily.assign(dept_name="D1")
    cat.to_parquet(p / "daily_category.parquet", index=False)
    ps = panel.groupby(["store_id", "item_id"]).agg(quantity=("quantity", "sum")).reset_index()
    ps["revenue"] = ps["quantity"] * 10
    ps["sale_days"] = 100
    ps["first_sale"] = pd.Timestamp("2024-01-01")
    ps["last_sale"] = sd.LAST_DATE
    ps["avg_price"] = 10.0
    ps["in_universe"] = True
    ps.to_parquet(p / "product_summary.parquet", index=False)
    panel.to_parquet(p / "panel.parquet", index=False)
    pd.DataFrame(
        columns=["store_id", "item_id", "date", "promo_price", "promo_before_price", "promo_type"]
    ).to_parquet(p / "promo_future.parquet", index=False)
    bt = panel[panel.date > sd.LAST_DATE - pd.Timedelta(days=3)][
        ["store_id", "item_id", "date"]
    ].copy()
    bt["actual"], bt["predicted"], bt["lower"], bt["upper"], bt["baseline_lag7"] = (
        1.0,
        1.5,
        0.5,
        2.5,
        1.0,
    )
    bt.to_parquet(p / "backtest_predictions.parquet", index=False)
    return p


def test_seed_loads_all_tables_and_limits_history_window(processed, artifacts_dir):
    engine = make_engine("sqlite://")
    assert seed_database(engine, processed, artifacts_dir, history_days=30) is True
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Store)) == 2
        assert s.scalar(select(func.count()).select_from(Product)) == 3
        assert s.scalar(select(func.count()).select_from(ProductSales)) == 6
        n_hist = s.scalar(select(func.count()).select_from(ItemStoreDaily))
        assert n_hist == 6 * 30  # only the last 30 days are kept
        assert (
            s.scalar(select(func.min(ItemStoreDaily.date)))
            == (sd.LAST_DATE - pd.Timedelta(days=29)).date()
        )
        mv = s.scalars(select(ModelVersion)).all()
        assert len(mv) == 1 and mv[0].is_active and mv[0].version == "lightgbm_tiny-202601010000"


def test_seed_is_idempotent_and_force_reloads(processed, artifacts_dir):
    engine = make_engine("sqlite://")
    assert seed_database(engine, processed, artifacts_dir, 30) is True
    assert seed_database(engine, processed, artifacts_dir, 30) is False  # already current
    assert seed_database(engine, processed, artifacts_dir, 30, force=True) is True
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(ModelVersion)) == 1


def test_reseed_preserves_prediction_history(processed, artifacts_dir):
    engine = make_engine("sqlite://")
    seed_database(engine, processed, artifacts_dir, 30)
    with Session(engine) as s:
        s.add(
            Prediction(
                model_version="v",
                store_id=1,
                item_id="aaa111",
                target_date=sd.LAST_DATE.date(),
                inputs={},
                predicted_quantity=3.2,
            )
        )
        s.commit()
    seed_database(engine, processed, artifacts_dir, 30, force=True)
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Prediction)) == 1


def test_seed_without_artifacts_fails_with_actionable_message(processed, tmp_path):
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    with pytest.raises(FileNotFoundError, match="pipeline"):
        seed_database(engine, processed, tmp_path / "nope", 30)


def test_prediction_crud_roundtrip(db_session_factory):
    with db_session_factory() as s:
        row = Prediction(
            model_version="v1",
            store_id=1,
            item_id="aaa111",
            target_date=sd.LAST_DATE.date(),
            inputs={"a": 1},
            predicted_quantity=4.5,
            lower=2.0,
            upper=7.0,
        )
        s.add(row)
        s.commit()
        pid = row.id
    with db_session_factory() as s:
        got = s.get(Prediction, pid)
        assert got is not None and got.predicted_quantity == 4.5 and got.inputs == {"a": 1}
        assert got.created_at is not None and got.source == "single"
