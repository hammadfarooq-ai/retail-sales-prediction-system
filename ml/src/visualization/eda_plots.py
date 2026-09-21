"""EDA on the real dataset: machine-readable profile + figures (all computed from actual data)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ml.src.config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
DATE_COLS = {"date"}


def profile_file(name: str) -> dict:
    """Rows, columns, dtypes, nulls, duplicates, cardinality, ranges for one raw CSV."""
    path = RAW_DIR / name
    df = pd.read_csv(path, index_col=0)
    prof: dict = {
        "file": name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "rows": int(len(df)),
        "columns": {},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    for c in df.columns:
        s = df[c]
        info: dict = {
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean() * 100), 2),
            "nunique": int(s.nunique()),
        }
        if c in DATE_COLS:
            info["min"], info["max"] = str(s.min()), str(s.max())
        elif pd.api.types.is_numeric_dtype(s):
            info.update(
                min=float(s.min()),
                p01=float(s.quantile(0.01)),
                median=float(s.median()),
                p99=float(s.quantile(0.99)),
                max=float(s.max()),
            )
        prof["columns"][c] = info
    key = [c for c in ("date", "item_id", "store_id") if c in df.columns]
    if len(key) == 3:
        prof["duplicate_date_item_store_keys"] = int(df.duplicated(key).sum())
    return prof


def write_data_profile(out: Path) -> dict:
    files = sorted(p.name for p in RAW_DIR.glob("*.csv"))
    profile = {}
    for f in files:
        log.info("profiling %s", f)
        profile[f] = profile_file(f)
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    return profile


def _save(fig, name: str, out_dir: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=130)
    plt.close(fig)
    log.info("saved %s", name)


def make_figures(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(PROCESSED_DIR / "daily_store.parquet")
    dc = pd.read_parquet(PROCESSED_DIR / "daily_category.parquet")
    ps = pd.read_parquet(PROCESSED_DIR / "product_summary.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    stores = pd.read_parquet(PROCESSED_DIR / "stores.parquet")
    facts: dict = {}

    # 1. Total daily quantity/revenue with 7-day MA
    tot = ds.groupby("date")[["quantity", "revenue"]].sum()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for a, col in zip(ax, ("revenue", "quantity"), strict=True):
        a.plot(tot.index, tot[col], lw=0.7, alpha=0.5, label="daily")
        a.plot(tot.index, tot[col].rolling(7).mean(), lw=1.8, label="7-day mean")
        a.set_ylabel(col)
        a.legend()
    ax[0].set_title("Total daily sales, all stores")
    _save(fig, "01_daily_total_sales.png", out_dir)

    # 2. Monthly revenue by store
    m = ds.assign(month=ds["date"].dt.to_period("M").astype(str)).pivot_table(
        index="month", columns="store_id", values="revenue", aggfunc="sum"
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    m.plot.bar(stacked=True, ax=ax, width=0.85)
    ax.set_title("Monthly revenue by store (store 4 opens 2023-12-13; first/last months partial)")
    ax.set_xlabel("")
    _save(fig, "02_monthly_revenue_by_store.png", out_dir)
    first4 = ds.loc[ds["store_id"] == 4, "date"].min()
    facts["store_4_first_sale_date"] = str(first4.date())

    # 3. Day-of-week profile (normalised per store)
    dow = ds.assign(dow=ds["date"].dt.dayofweek)
    prof = dow.groupby(["store_id", "dow"])["revenue"].mean().unstack(0)
    prof = prof / prof.mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    prof.plot(marker="o", ax=ax)
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylabel("revenue index (mean = 1)")
    ax.set_title("Weekly seasonality by store")
    _save(fig, "03_weekday_profile.png", out_dir)
    facts["weekday_revenue_index"] = {
        str(k): round(float(v), 3) for k, v in prof.mean(axis=1).items()
    }

    # 4. Store performance incl. revenue per m2
    sp = ds.groupby("store_id").agg(revenue=("revenue", "sum"), days=("date", "nunique"))
    sp = sp.join(stores.set_index("store_id")[["area", "format"]])
    sp["revenue_per_day_per_m2"] = sp["revenue"] / sp["days"] / sp["area"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=sp.index.astype(str), y=sp["revenue"] / 1e9, ax=ax[0])
    ax[0].set_title("Total revenue by store (billions)")
    ax[0].set_xlabel("store_id")
    sns.barplot(x=sp.index.astype(str), y=sp["revenue_per_day_per_m2"], ax=ax[1])
    ax[1].set_title("Revenue per selling day per m²")
    ax[1].set_xlabel("store_id")
    _save(fig, "04_store_performance.png", out_dir)

    # 5. Top departments
    top = dc.groupby("dept_name")["revenue"].sum().nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    (top / 1e6).plot.barh(ax=ax, color=sns.color_palette("deep")[0])
    ax.set_xlabel("revenue (millions)")
    ax.set_ylabel("")
    ax.set_title("Top-15 departments (category level) by revenue")
    _save(fig, "05_top_departments.png", out_dir)
    facts["n_departments"] = int(dc["dept_name"].nunique())
    facts["top5_departments_revenue_share_pct"] = round(
        float(
            dc.groupby("dept_name")["revenue"].sum().nlargest(5).sum() / dc["revenue"].sum() * 100
        ),
        1,
    )

    # 6. Sparsity of item-store series & universe motivation
    ps = ps.assign(window=(pd.Timestamp("2024-09-26") - ps["first_sale"]).dt.days + 1)
    ps["density"] = ps["sale_days"] / ps["window"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(ps["density"], bins=40, ax=ax[0])
    ax[0].axvline(0.7, color="r", ls="--", label="universe threshold (train-period, 0.7)")
    ax[0].set_title("Share of days with a sale, per item-store pair")
    ax[0].legend()
    sns.histplot(np.log10(ps["revenue"].clip(lower=1)), bins=40, ax=ax[1])
    ax[1].set_title("log10 revenue per item-store pair (heavy tail)")
    _save(fig, "06_series_sparsity.png", out_dir)
    facts["item_store_pairs"] = int(len(ps))
    facts["pairs_density_ge_0_7_full_period"] = int((ps["density"] >= 0.7).sum())
    top_pairs = ps.nlargest(int(len(ps) * 0.1), "revenue")["revenue"].sum() / ps["revenue"].sum()
    facts["top10pct_pairs_revenue_share_pct"] = round(float(top_pairs * 100), 1)

    # 7. Quantity & price distributions from the panel (universe)
    sold = panel[panel["quantity"] > 0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(np.log10(sold["quantity"]), bins=60, ax=ax[0])
    ax[0].set_title("log10(quantity) on sale-days (universe)")
    sns.boxplot(x=sold["quantity"].clip(upper=float(sold["quantity"].quantile(0.999))), ax=ax[1])
    ax[1].set_title("quantity boxplot (clipped at p99.9)")
    sns.histplot(np.log10(sold["price_obs"].dropna().clip(lower=0.01)), bins=60, ax=ax[2])
    ax[2].set_title("log10(unit price) on sale-days")
    _save(fig, "07_quantity_price_distributions.png", out_dir)
    q = sold["quantity"]
    iqr_hi = q.quantile(0.75) + 3 * (q.quantile(0.75) - q.quantile(0.25))
    facts["universe_quantity_outliers_above_3iqr_pct"] = round(float((q > iqr_hi).mean() * 100), 2)
    facts["universe_zero_days_pct"] = round(float((panel["quantity"] == 0).mean() * 100), 2)
    facts["universe_promo_days_pct"] = round(float(panel["promo_flag"].mean() * 100), 2)

    # 8. Promotion effect: per-pair lift (mean qty on promo days / mean qty on non-promo days)
    g = panel.groupby(["store_id", "item_id", "promo_flag"])["quantity"].mean().unstack()
    g = g.dropna()
    g = g[g[0] > 0]
    lift = g[1] / g[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(lift.clip(upper=4), bins=60, ax=ax)
    ax.axvline(1, color="k", ls="--")
    ax.axvline(lift.median(), color="r", label=f"median lift = {lift.median():.2f}")
    ax.set_title("Promotion lift per item-store pair (mean qty promo / non-promo)")
    ax.legend()
    _save(fig, "08_promo_lift.png", out_dir)
    facts["promo_lift_median"] = round(float(lift.median()), 3)
    facts["promo_lift_pairs_compared"] = int(len(lift))
    facts["share_pairs_with_lift_gt_1_pct"] = round(float((lift > 1).mean() * 100), 1)

    # 9. Discount depth vs lift
    d = panel[panel["promo_flag"] == 1]
    depth = 1 - d["promo_price"] / d["promo_before_price"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(depth.clip(0, 1).dropna(), bins=50, ax=ax)
    ax.set_title("Discount depth on promo-days (1 - promo price / regular price)")
    _save(fig, "09_discount_depth.png", out_dir)
    facts["median_discount_depth"] = round(float(depth.median()), 3)

    # 10. Weekly seasonality across the two years (holiday peak)
    wk = tot["revenue"].resample("W").sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wk.index[1:-1], np.asarray(wk.values[1:-1], dtype=float) / 1e6)
    ax.set_title("Weekly revenue (millions), all stores")
    _save(fig, "10_weekly_revenue.png", out_dir)

    (out_dir / "eda_facts.json").write_text(json.dumps(facts, indent=2))
    return facts
