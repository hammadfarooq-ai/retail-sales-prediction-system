"""Load processed parquet tables + model registry row into the database.

    python -m app.db.seed            # load only if the DB is empty / model version changed
    python -m app.db.seed --force    # wipe derived tables and reload

Prediction history (``predictions``) is preserved across re-seeds.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import mask_url, setup_logging
from app.db.base import Base
from app.db.session import get_engine
from app.models import (
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

log = logging.getLogger(__name__)

# Deletion order respects foreign keys (children first).
DERIVED_TABLES = [
    BacktestPrediction,
    ItemStoreDaily,
    PromoCalendar,
    ProductSales,
    DailyCategorySales,
    DailyStoreSales,
    Product,
    Store,
]


def _bulk_insert(engine: Engine, table: str, df: pd.DataFrame, chunk: int = 200_000) -> None:
    """COPY on PostgreSQL (fast for millions of rows); executemany fallback elsewhere."""
    if df.empty:
        return
    if engine.dialect.name == "postgresql":
        cols = ", ".join(df.columns)
        with engine.begin() as conn:
            raw = conn.connection.driver_connection
            sql = f"COPY {table} ({cols}) FROM STDIN (FORMAT CSV, NULL '')"
            with raw.cursor() as cur, cur.copy(sql) as cp:  # type: ignore[union-attr]
                for i in range(0, len(df), chunk):
                    buf = io.StringIO()
                    df.iloc[i : i + chunk].to_csv(buf, index=False, header=False)
                    cp.write(buf.getvalue())
    else:
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=5000, method="multi")
    log.info("loaded %-22s %9d rows", table, len(df))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def _is_current(engine: Engine, version: str) -> bool:
    with Session(engine) as s:
        try:
            n_stores = s.scalar(select(func.count()).select_from(Store)) or 0
            n_isd = s.scalar(select(func.count()).select_from(ItemStoreDaily)) or 0
            active = s.scalar(select(ModelVersion.version).where(ModelVersion.is_active))
        except Exception:
            return False
    return n_stores > 0 and n_isd > 0 and active == version


def seed_database(
    engine: Engine,
    processed_dir: Path,
    artifacts_dir: Path,
    history_days: int,
    force: bool = False,
) -> bool:
    """Returns True if data was (re)loaded, False if the DB was already up to date."""
    meta_path = artifacts_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run the ML pipeline first (make pipeline)."
        )
    meta = json.loads(meta_path.read_text())
    metrics = json.loads((artifacts_dir / "metrics.json").read_text())

    Base.metadata.create_all(engine)
    if not force and _is_current(engine, meta["version"]):
        log.info("Database already seeded for model %s; skipping", meta["version"])
        return False

    p = processed_dir
    log.info("Seeding database from %s", p)
    with Session(engine) as s:
        for model in DERIVED_TABLES:
            s.execute(delete(model))
        s.execute(delete(ModelVersion))
        s.commit()

    _bulk_insert(engine, "stores", pd.read_parquet(p / "stores.parquet"))
    products = pd.read_parquet(p / "products.parquet")
    _bulk_insert(
        engine,
        "products",
        products[
            [
                "item_id",
                "dept_name",
                "class_name",
                "subclass_name",
                "item_type",
                "dept_code",
                "class_code",
            ]
        ],
    )
    _bulk_insert(engine, "daily_store_sales", _clean(pd.read_parquet(p / "daily_store.parquet")))
    _bulk_insert(
        engine, "daily_category_sales", _clean(pd.read_parquet(p / "daily_category.parquet"))
    )
    ps = pd.read_parquet(p / "product_summary.parquet")
    _bulk_insert(
        engine,
        "product_sales",
        _clean(
            ps[
                [
                    "store_id",
                    "item_id",
                    "quantity",
                    "revenue",
                    "sale_days",
                    "first_sale",
                    "last_sale",
                    "avg_price",
                    "in_universe",
                ]
            ]
        ),
    )

    panel = pd.read_parquet(
        p / "panel.parquet",
        columns=[
            "store_id",
            "item_id",
            "date",
            "quantity",
            "price_obs",
            "promo_price",
            "promo_before_price",
            "promo_type",
        ],
    )
    panel["date"] = pd.to_datetime(panel["date"])
    cutoff = panel["date"].max() - pd.Timedelta(days=history_days - 1)
    _bulk_insert(engine, "item_store_daily", _clean(panel[panel["date"] >= cutoff]))
    del panel

    promo = pd.read_parquet(p / "promo_future.parquet")
    promo = promo[
        ["store_id", "item_id", "date", "promo_price", "promo_before_price", "promo_type"]
    ]
    _bulk_insert(engine, "promo_calendar", _clean(promo))

    bt = pd.read_parquet(p / "backtest_predictions.parquet")
    bt["model_version"] = meta["version"]
    _bulk_insert(engine, "backtest_predictions", _clean(bt))

    with Session(engine) as s:
        s.add(
            ModelVersion(
                version=meta["version"],
                model_name=meta["model_name"],
                trained_at=datetime.fromisoformat(meta["trained_at"]),
                is_active=True,
                n_features=meta["n_features"],
                split=meta["split"],
                metrics=metrics,
                params={**meta.get("params", {}), "final_iterations": meta.get("final_iterations")},
            )
        )
        s.commit()
    if engine.dialect.name == "postgresql":
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("ANALYZE"))
    log.info("Seed complete (model %s)", meta["version"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="wipe and reload derived tables")
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    log.info("Database: %s", mask_url(settings.sqlalchemy_url))
    seed_database(
        get_engine(),
        settings.processed_data_dir,
        settings.model_artifacts_dir,
        settings.history_days_in_db,
        force=args.force,
    )


if __name__ == "__main__":
    main()
