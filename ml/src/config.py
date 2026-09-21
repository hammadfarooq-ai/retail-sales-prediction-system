"""Central configuration for the ML pipeline (paths, split logic, modelling constants).

Every date-dependent value is *derived from the data* rather than hard-coded, and is
persisted into the model metadata so the split is fully documented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data"  # the downloaded Kaggle CSVs live directly in ./data (left intact)
PROCESSED_DIR = RAW_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / "eda"

# --- Problem definition -------------------------------------------------------------
TARGET = "quantity"  # units sold (kg for weighted goods) per store x product x day
MIN_LAG = 7  # forecast horizon in days; every history feature is lagged >= MIN_LAG
LAGS = (7, 14, 21, 28)
ROLLING_WINDOWS = (7, 14, 28, 56)
STD_WINDOWS = (7, 28)
MIN_HISTORY_DAYS = 35  # rows with < 35 days of item-store history are dropped (cold start)

# --- Split (chronological, derived from the last observed date) ---------------------
TEST_DAYS = 42
VAL_DAYS = 42

# --- Modelling universe (chosen from TRAIN period only) -----------------------------
UNIVERSE_MIN_DENSITY = 0.7  # share of days with >= 1 sale since first sale
UNIVERSE_MIN_SALE_DAYS = 90


@dataclass(frozen=True)
class SplitDates:
    """Inclusive date boundaries of the chronological split."""

    data_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def train_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_end)


def compute_split(data_start: pd.Timestamp, data_end: pd.Timestamp) -> SplitDates:
    """Last TEST_DAYS -> test, the VAL_DAYS before that -> validation, the rest -> train."""
    test_start = data_end - pd.Timedelta(days=TEST_DAYS - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=VAL_DAYS - 1)
    train_end = val_start - pd.Timedelta(days=1)
    fmt = "%Y-%m-%d"
    return SplitDates(
        data_start=data_start.strftime(fmt),
        train_end=train_end.strftime(fmt),
        val_start=val_start.strftime(fmt),
        val_end=val_end.strftime(fmt),
        test_start=test_start.strftime(fmt),
        test_end=data_end.strftime(fmt),
    )
