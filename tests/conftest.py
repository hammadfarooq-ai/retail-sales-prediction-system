"""Shared fixtures for ML tests (small synthetic panels; no dataset needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": [1, 2],
            "division": ["Div1", "Div2"],
            "format": ["Format-1", "Format-6"],
            "city": ["City1", "City1"],
            "area": [1500, 210],
        }
    )


def make_panel(n_days: int = 120, n_items: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for store in (1, 2):
        for i in range(n_items):
            q = rng.poisson(5, n_days).astype("float32") + (dates.dayofweek == 4) * 3
            price = np.where(q > 0, 100 + i * 10 + rng.normal(0, 1, n_days), np.nan)
            promo = rng.random(n_days) < 0.15
            rows.append(
                pd.DataFrame(
                    {
                        "store_id": np.int8(store),
                        "item_id": f"item{i}",
                        "date": dates,
                        "quantity": q,
                        "price_obs": price.astype("float32"),
                        "promo_price": np.where(promo, 80.0, np.nan).astype("float32"),
                        "promo_before_price": np.where(promo, 100.0, np.nan).astype("float32"),
                        "promo_type": np.where(promo, 5.0, np.nan).astype("float32"),
                        "dept_code": np.int16(i),
                        "class_code": np.int16(i),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
