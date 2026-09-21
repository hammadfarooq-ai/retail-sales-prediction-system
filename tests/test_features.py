"""Feature engineering: correctness and, most importantly, absence of target leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.config import MIN_HISTORY_DAYS, MIN_LAG, compute_split
from ml.src.features.engineering import (
    FEATURE_COLUMNS,
    build_features,
    chronological_masks,
    store_static_table,
)

HISTORY_FEATURES = [
    c
    for c in FEATURE_COLUMNS
    if c.startswith(("lag_", "rolling_"))
    or c in {"dow_mean_4w", "ratio_mean_7_28", "sale_rate_28", "days_since_last_sale", "last_price"}
    or c == "promo_days_28"
]


def test_feature_columns_present_and_unique(panel, stores):
    feats = build_features(panel, stores)
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_lag_values_are_correct(panel, stores):
    feats = build_features(panel, stores, min_history_days=None)
    one = feats[(feats.store_id == 1) & (feats.item_id == "item0")].reset_index(drop=True)
    for k in (7, 14, 21, 28):
        expected = one["quantity"].shift(k)
        pd.testing.assert_series_equal(
            one[f"lag_{k}"], expected, check_names=False, check_dtype=False
        )


def test_rolling_mean_uses_only_data_at_or_before_origin(panel, stores):
    feats = build_features(panel, stores, min_history_days=None)
    one = feats[(feats.store_id == 1) & (feats.item_id == "item0")].reset_index(drop=True)
    t = 60
    window = one["quantity"].iloc[t - MIN_LAG - 27 : t - MIN_LAG + 1]  # 28 days ending at t-7
    assert one.loc[t, "rolling_mean_28"] == pytest.approx(window.mean(), rel=1e-5)


def test_no_target_leakage_when_future_values_change(panel, stores):
    """Perturbing y at/after day t (and within the 7-day blind window) must not move any
    history feature of day t."""
    base = build_features(panel, stores, min_history_days=None)
    t_date = pd.Timestamp("2024-03-01")
    perturbed = panel.copy()
    blind_start = t_date - pd.Timedelta(days=MIN_LAG - 1)  # days t-6 .. end are unknown at origin
    m = perturbed["date"] >= blind_start
    perturbed.loc[m, "quantity"] = perturbed.loc[m, "quantity"] * 100 + 999
    perturbed.loc[m, "price_obs"] = 12345.0
    pert = build_features(perturbed, stores, min_history_days=None)
    a = base[base["date"] == t_date].reset_index(drop=True)
    b = pert[pert["date"] == t_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[HISTORY_FEATURES], b[HISTORY_FEATURES])


def test_target_and_realised_price_are_not_features(panel, stores):
    assert "quantity" not in FEATURE_COLUMNS
    assert "price_obs" not in FEATURE_COLUMNS
    assert "revenue" not in FEATURE_COLUMNS


def test_min_history_filter(panel, stores):
    feats = build_features(panel, stores)
    assert feats["age_days"].min() >= MIN_HISTORY_DAYS
    assert len(build_features(panel, stores, min_history_days=None)) == len(panel)


def test_promo_features(panel, stores):
    feats = build_features(panel, stores, min_history_days=None)
    promo_rows = feats[feats["promo_flag"] == 1]
    assert len(promo_rows) > 0
    assert np.allclose(promo_rows["discount_depth"], 0.2, atol=1e-6)
    assert (promo_rows["planned_price"] == 80).all()
    assert (feats.loc[feats["promo_flag"] == 0, "discount_depth"] == 0).all()


def test_calendar_and_static_features(panel, stores):
    feats = build_features(panel, stores, min_history_days=None)
    row = feats[feats["date"] == pd.Timestamp("2024-01-05")].iloc[0]  # a Friday
    assert row["day_of_week"] == 4 and row["is_weekend"] == 0 and row["month"] == 1
    assert set(feats["store_area"].unique()) == {1500.0, 210.0}
    assert list(store_static_table(stores).columns) == [
        "store_id",
        "store_area",
        "store_format_code",
    ]


def test_missing_required_columns_raises(panel, stores):
    with pytest.raises(ValueError, match="missing required columns"):
        build_features(panel.drop(columns=["promo_price"]), stores)


def test_chronological_split_has_no_overlap():
    split = compute_split(pd.Timestamp("2022-08-28"), pd.Timestamp("2024-09-26"))
    dates = pd.Series(pd.date_range("2022-08-28", "2024-09-26"))
    m = chronological_masks(dates, split)
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    assert dates[m["train"]].max() < dates[m["val"]].min() < dates[m["val"]].max()
    assert dates[m["val"]].max() < dates[m["test"]].min()
    assert m["test"].sum() == 42 and m["val"].sum() == 42
    assert (m["train"] | m["val"] | m["test"]).all()
    assert np.array_equal(
        dates[m["test"]].iloc[[0, -1]].dt.strftime("%Y-%m-%d"), [split.test_start, split.test_end]
    )


# --------------------------------------------------------------------------------------
# Vectorised implementation == slow per-series reference implementation
# --------------------------------------------------------------------------------------
def _reference_frame(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    from reference_features import reference_history_block

    df = panel.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    promo = df["promo_price"].notna().to_numpy().astype("int8")
    parts: dict[str, list[np.ndarray]] = {}
    for _, g in df.groupby(["store_id", "item_id"], sort=False):
        i = g.index.to_numpy()
        blk = reference_history_block(
            df.loc[i, "quantity"].to_numpy("float64"),
            df.loc[i, "price_obs"].to_numpy("float64"),
            promo[i],
        )
        for k, v in blk.items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def _assert_equivalent(panel, stores):
    ref = _reference_frame(panel)
    got = build_features(panel, stores, min_history_days=None)
    for name, expected in ref.items():
        actual = got[name].to_numpy("float64")
        np.testing.assert_allclose(
            actual, expected.astype("float32").astype("float64"), rtol=2e-4, atol=1e-4, err_msg=name
        )


def test_vectorised_features_match_reference_synthetic(panel, stores):
    _assert_equivalent(panel, stores)


def test_vectorised_features_match_reference_with_gaps_and_nans(stores):
    """Unobserved (NaN) future days, all-zero series and a series shorter than the windows."""
    from synthetic_data import make_panel

    p = make_panel(n_days=90, seed=3)
    p.loc[(p["item_id"] == "item0") & (p["date"] > "2024-03-10"), "quantity"] = np.nan
    p.loc[p["item_id"] == "item1", "quantity"] = 0.0
    p = pd.concat([p, p[p["date"] < "2024-01-20"].assign(item_id="short")], ignore_index=True)
    _assert_equivalent(p, stores)


@pytest.mark.requires_data
def test_vectorised_features_match_reference_on_real_panel_slice(stores):
    from ml.src.config import PROCESSED_DIR

    path = PROCESSED_DIR / "panel.parquet"
    if not path.exists():
        pytest.skip("processed panel not available")
    real = pd.read_parquet(path)
    keys = real[["store_id", "item_id"]].drop_duplicates().sample(25, random_state=1)
    real = real.merge(keys, on=["store_id", "item_id"])
    real["date"] = pd.to_datetime(real["date"]).astype("datetime64[ns]")
    real_stores = pd.read_parquet(PROCESSED_DIR / "stores.parquet")
    _assert_equivalent(real, real_stores)
