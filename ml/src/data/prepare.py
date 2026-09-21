"""Clean the raw data and materialise every processed table used downstream.

Outputs (data/processed/*.parquet + data_profile.json):
    stores, products, daily_store, daily_category, product_summary,
    panel  (zero-filled item x store x day frame for the modelling universe),
    promo_future (planned promotions dated after the last observed sales day).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.config import (
    PROCESSED_DIR,
    UNIVERSE_MIN_DENSITY,
    UNIVERSE_MIN_SALE_DAYS,
    SplitDates,
    compute_split,
)
from ml.src.data.loaders import load_catalog, load_discounts, load_sales, load_stores

log = logging.getLogger(__name__)

FUTURE_PROMO_HORIZON_DAYS = 90


def clean_sales(sales: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop returns / zero-value lines. Returns cleaned frame + a dict of what was removed."""
    stats = {"rows_raw": len(sales)}
    bad_qty = sales["quantity"] <= 0
    bad_val = (sales["sum_total"] <= 0) | (sales["price_base"] <= 0)
    stats["removed_non_positive_quantity"] = int(bad_qty.sum())
    stats["removed_non_positive_value_or_price"] = int((~bad_qty & bad_val).sum())
    out = sales.loc[~(bad_qty | bad_val)].copy()
    stats["rows_clean"] = len(out)
    return out, stats


def select_universe(clean: pd.DataFrame, split: SplitDates) -> pd.DataFrame:
    """Pick (store, item) pairs with regular demand, using ONLY the training period."""
    train = clean.loc[clean["date"] <= split.train_end_ts]
    g = (
        train.groupby(["store_id", "item_id"], observed=True)
        .agg(
            sale_days=("date", "nunique"),
            first_sale=("date", "min"),
            train_rev=("sum_total", "sum"),
        )
        .reset_index()
    )
    g["window_days"] = (split.train_end_ts - g["first_sale"]).dt.days + 1
    g["density"] = g["sale_days"] / g["window_days"]
    sel = g[(g["density"] >= UNIVERSE_MIN_DENSITY) & (g["sale_days"] >= UNIVERSE_MIN_SALE_DAYS)]
    log.info(
        "Universe: %d of %d train pairs (%.1f%% of train revenue)",
        len(sel),
        len(g),
        100 * sel["train_rev"].sum() / g["train_rev"].sum(),
    )
    return sel.reset_index(drop=True)


def build_panel(
    clean: pd.DataFrame,
    universe: pd.DataFrame,
    discounts: pd.DataFrame,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Zero-filled daily panel from each pair's first sale to `data_end`.

    `price_obs` is the realised unit price on days with a sale (NaN otherwise); it is only ever
    used through lagged features. Promo columns come from the promo calendar for that day.
    """
    starts = universe["first_sale"].to_numpy("datetime64[D]")
    lengths = ((data_end.to_datetime64().astype("datetime64[D]") - starts).astype(int) + 1).astype(
        int
    )
    total = int(lengths.sum())
    pair_idx = np.repeat(np.arange(len(universe)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    dates = starts[pair_idx] + offsets.astype("timedelta64[D]")
    panel = pd.DataFrame(
        {
            "store_id": universe["store_id"].to_numpy()[pair_idx].astype("int8"),
            "item_id": universe["item_id"].astype(str).to_numpy()[pair_idx],
            "date": pd.to_datetime(dates),
        }
    )

    keys = universe[["store_id", "item_id"]].astype({"item_id": str})
    c = clean.copy()
    c["item_id"] = c["item_id"].astype(str)
    c = c.merge(keys, on=["store_id", "item_id"], how="inner")
    c = c.rename(columns={"price_base": "price_obs", "sum_total": "revenue"})
    panel["item_id"] = panel["item_id"].astype(str)
    panel = panel.merge(
        c[["store_id", "item_id", "date", "quantity", "price_obs", "revenue"]],
        on=["store_id", "item_id", "date"],
        how="left",
    )
    panel["quantity"] = panel["quantity"].fillna(0).astype("float32")
    panel["revenue"] = panel["revenue"].fillna(0).astype("float32")

    d = discounts.copy()
    d["item_id"] = d["item_id"].astype(str)
    d = d[d["date"] <= data_end].merge(keys, on=["store_id", "item_id"], how="inner")
    d = d.rename(
        columns={
            "sale_price_time_promo": "promo_price",
            "sale_price_before_promo": "promo_before_price",
            "promo_type_code": "promo_type",
        }
    )
    panel = panel.merge(
        d[["store_id", "item_id", "date", "promo_price", "promo_before_price", "promo_type"]],
        on=["store_id", "item_id", "date"],
        how="left",
    )
    panel["promo_flag"] = panel["promo_price"].notna().astype("int8")
    panel = panel.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return panel


def build_promo_future(
    discounts: pd.DataFrame, universe: pd.DataFrame, data_end: pd.Timestamp
) -> pd.DataFrame:
    """Planned promotions for the next FUTURE_PROMO_HORIZON_DAYS after the last sales day."""
    keys = universe[["store_id", "item_id"]].astype({"item_id": str})
    d = discounts.copy()
    d["item_id"] = d["item_id"].astype(str)
    horizon_end = data_end + pd.Timedelta(days=FUTURE_PROMO_HORIZON_DAYS)
    d = d[(d["date"] > data_end) & (d["date"] <= horizon_end)]
    d = d.merge(keys, on=["store_id", "item_id"], how="inner")
    return d.rename(
        columns={
            "sale_price_time_promo": "promo_price",
            "sale_price_before_promo": "promo_before_price",
            "promo_type_code": "promo_type",
        }
    ).reset_index(drop=True)


def build_products(clean: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    sold = pd.DataFrame({"item_id": clean["item_id"].astype(str).unique()})
    cat = catalog.assign(item_id=catalog["item_id"].astype(str))
    prod = sold.merge(cat, on="item_id", how="left")
    for col in ("dept_name", "class_name", "subclass_name"):
        prod[col] = prod[col].fillna("UNKNOWN")
    prod["dept_code"] = pd.factorize(prod["dept_name"], sort=True)[0].astype("int16")
    prod["class_code"] = pd.factorize(prod["class_name"], sort=True)[0].astype("int16")
    return prod.sort_values("item_id").reset_index(drop=True)


def prepare_all(out_dir: Path | None = None) -> dict:
    out = out_dir or PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw files")
    stores = load_stores()
    catalog = load_catalog()
    sales = load_sales()
    discounts = load_discounts()

    data_start, data_end = sales["date"].min(), sales["date"].max()
    split = compute_split(data_start, data_end)
    log.info("Split: %s", split.as_dict())

    clean, clean_stats = clean_sales(sales)
    del sales
    clean["item_id"] = clean["item_id"].astype(str)

    products = build_products(clean, catalog)
    dept = products.set_index("item_id")[["dept_name", "class_name", "subclass_name"]]
    clean = clean.join(dept, on="item_id")

    daily_store = (
        clean.groupby(["date", "store_id"])
        .agg(
            quantity=("quantity", "sum"),
            revenue=("sum_total", "sum"),
            n_products=("item_id", "nunique"),
        )
        .reset_index()
    )
    daily_category = (
        clean.groupby(["date", "store_id", "dept_name"])
        .agg(
            quantity=("quantity", "sum"),
            revenue=("sum_total", "sum"),
            n_products=("item_id", "nunique"),
        )
        .reset_index()
    )
    product_summary = (
        clean.groupby(["store_id", "item_id"])
        .agg(
            quantity=("quantity", "sum"),
            revenue=("sum_total", "sum"),
            sale_days=("date", "nunique"),
            first_sale=("date", "min"),
            last_sale=("date", "max"),
        )
        .reset_index()
    )
    product_summary["avg_price"] = product_summary["revenue"] / product_summary["quantity"]

    universe = select_universe(clean, split)
    panel = build_panel(clean, universe, discounts, data_end)
    promo_future = build_promo_future(discounts, universe, data_end)
    universe_keys = set(zip(universe["store_id"], universe["item_id"].astype(str), strict=True))
    product_summary["in_universe"] = [
        (s, i) in universe_keys
        for s, i in zip(product_summary["store_id"], product_summary["item_id"], strict=True)
    ]
    panel = panel.merge(products[["item_id", "dept_code", "class_code"]], on="item_id", how="left")

    for name, df in {
        "stores": stores,
        "products": products,
        "daily_store": daily_store,
        "daily_category": daily_category,
        "product_summary": product_summary,
        "universe": universe,
        "panel": panel,
        "promo_future": promo_future,
    }.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        log.info("wrote %s %s", name, df.shape)

    meta = {
        "split": split.as_dict(),
        "cleaning": clean_stats,
        "universe": {
            "pairs": int(len(universe)),
            "min_density": UNIVERSE_MIN_DENSITY,
            "min_sale_days": UNIVERSE_MIN_SALE_DAYS,
            "panel_rows": int(len(panel)),
            "revenue_share_of_clean_total": float(
                clean.loc[
                    pd.MultiIndex.from_frame(clean[["store_id", "item_id"]]).isin(
                        pd.MultiIndex.from_frame(universe[["store_id", "item_id"]])
                    ),
                    "sum_total",
                ].sum()
                / clean["sum_total"].sum()
            ),
        },
        "data_end": data_end.strftime("%Y-%m-%d"),
    }
    (out / "prepare_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
