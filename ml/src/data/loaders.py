"""Typed loaders for the raw Kaggle CSVs in ./data (the raw files are never modified)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.src.config import RAW_DIR

_TX_DTYPES = {"item_id": "string", "store_id": "int8"}


def _path(name: str, raw_dir: Path | None = None) -> Path:
    p = (raw_dir or RAW_DIR) / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download the dataset from "
            "https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data into ./data"
        )
    return p


def load_stores(raw_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_path("stores.csv", raw_dir), index_col=0).sort_values("store_id")


def load_catalog(raw_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_path("catalog.csv", raw_dir), index_col=0)


def load_sales(raw_dir: Path | None = None) -> pd.DataFrame:
    """Sales transactions aggregated per (date, item_id, store_id); positive-sale rows only."""
    df = pd.read_csv(
        _path("sales.csv", raw_dir),
        index_col=0,
        dtype={
            "quantity": "float32",
            "price_base": "float32",
            "sum_total": "float64",
            **_TX_DTYPES,
        },
        parse_dates=["date"],
    )
    return df


def load_discounts(raw_dir: Path | None = None) -> pd.DataFrame:
    """Promo calendar (includes future-dated planned promotions)."""
    return pd.read_csv(
        _path("discounts_history.csv", raw_dir),
        usecols=[
            "date",
            "item_id",
            "sale_price_before_promo",
            "sale_price_time_promo",
            "promo_type_code",
            "store_id",
        ],
        dtype={
            "sale_price_before_promo": "float32",
            "sale_price_time_promo": "float32",
            "promo_type_code": "float32",
            **_TX_DTYPES,
        },
        parse_dates=["date"],
    )
