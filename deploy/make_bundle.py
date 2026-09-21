"""Build the slim, committed bundle the hosted backend needs (model artifacts + seed tables).

    python deploy/make_bundle.py

Reads ml/artifacts and data/processed (produced by `make pipeline`) and writes
deploy/artifacts and deploy/seed. The panel is cut to the last HISTORY_DAYS days, which is all the
API stores in PostgreSQL. No raw Kaggle data is included - only derived aggregates.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts"
PROCESSED = ROOT / "data" / "processed"
OUT_ART = ROOT / "deploy" / "artifacts"
OUT_SEED = ROOT / "deploy" / "seed"
HISTORY_DAYS = 180

ARTIFACT_FILES = [
    "model.joblib",
    "model_q10.joblib",
    "model_q90.joblib",
    "feature_config.json",
    "feature_importance.json",
    "metrics.json",
    "model_metadata.json",
]
SEED_FILES = [
    "stores",
    "products",
    "daily_store",
    "daily_category",
    "product_summary",
    "promo_future",
    "backtest_predictions",
]


def main() -> None:
    for d in (OUT_ART, OUT_SEED):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for name in ARTIFACT_FILES:
        shutil.copy2(ARTIFACTS / name, OUT_ART / name)
    for name in SEED_FILES:
        shutil.copy2(PROCESSED / f"{name}.parquet", OUT_SEED / f"{name}.parquet")

    cols = [
        "store_id",
        "item_id",
        "date",
        "quantity",
        "price_obs",
        "promo_price",
        "promo_before_price",
        "promo_type",
    ]
    panel = pd.read_parquet(PROCESSED / "panel.parquet", columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    cutoff = panel["date"].max() - pd.Timedelta(days=HISTORY_DAYS - 1)
    panel[panel["date"] >= cutoff].to_parquet(OUT_SEED / "panel.parquet", index=False)

    total = 0
    for f in sorted([*OUT_ART.iterdir(), *OUT_SEED.iterdir()]):
        size = f.stat().st_size
        total += size
        print(f"{size / 1e6:7.2f} MB  {f.relative_to(ROOT)}")
    print(f"{total / 1e6:7.2f} MB  total")


if __name__ == "__main__":
    main()
