"""Data preparation on tiny synthetic frames: cleaning, universe selection (train-only), panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.src.config import compute_split
from ml.src.data.prepare import build_panel, clean_sales, select_universe


def _sales(rows):
    df = pd.DataFrame(
        rows, columns=["date", "item_id", "quantity", "price_base", "sum_total", "store_id"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_clean_sales_drops_returns_and_zero_value_rows():
    raw = _sales(
        [
            ("2024-01-01", "a", 2.0, 10.0, 20.0, 1),
            ("2024-01-01", "b", -1.0, 10.0, -10.0, 1),  # return
            ("2024-01-01", "c", 1.0, 0.0, 0.0, 1),  # free item
            ("2024-01-02", "d", 0.0, 5.0, 0.0, 1),  # zero qty
        ]
    )
    clean, stats = clean_sales(raw)
    assert list(clean["item_id"]) == ["a"]
    assert stats["rows_raw"] == 4 and stats["rows_clean"] == 1
    assert stats["removed_non_positive_quantity"] == 2


def test_universe_uses_only_training_period_and_density_rule():
    dates = pd.date_range("2024-01-01", "2024-06-30")
    split = compute_split(dates[0], dates[-1])
    rows = []
    for d in dates:
        rows.append((d, "steady", 1.0, 10.0, 10.0, 1))  # sells every day
        if d.day % 5 == 0:
            rows.append((d, "sparse", 1.0, 10.0, 10.0, 1))  # ~20% density
        if d > pd.Timestamp(split.train_end):
            rows.append((d, "future_only", 1.0, 10.0, 10.0, 1))  # appears only after train
    clean, _ = clean_sales(_sales(rows))
    uni = select_universe(clean, split)
    assert set(uni["item_id"]) == {"steady"}


def test_panel_zero_fills_between_first_sale_and_end():
    dates = pd.date_range("2024-01-01", "2024-01-10")
    clean, _ = clean_sales(
        _sales(
            [
                (dates[0], "a", 3.0, 10.0, 30.0, 1),
                (dates[4], "a", 2.0, 11.0, 22.0, 1),
                (dates[9], "a", 1.0, 12.0, 12.0, 1),
            ]
        )
    )
    uni = pd.DataFrame({"store_id": [1], "item_id": ["a"], "first_sale": [dates[0]]})
    disc = pd.DataFrame(
        {
            "date": [dates[4]],
            "item_id": ["a"],
            "sale_price_before_promo": np.float32(12.0),
            "sale_price_time_promo": np.float32(11.0),
            "promo_type_code": np.float32(5.0),
            "store_id": np.int8(1),
        }
    )
    panel = build_panel(clean, uni, disc, dates[-1])
    assert len(panel) == 10 and panel["date"].is_monotonic_increasing
    assert panel["quantity"].tolist() == [3, 0, 0, 0, 2, 0, 0, 0, 0, 1]
    assert panel["price_obs"].notna().sum() == 3  # price only on sale days
    assert panel["promo_flag"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert not panel.duplicated(["store_id", "item_id", "date"]).any()
