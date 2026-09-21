"""Evaluation figures generated from the *saved* training outputs (metrics, backtest)."""

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

from ml.src.config import ARTIFACTS_DIR, PLOTS_DIR, PROCESSED_DIR  # noqa: E402

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _save(fig, name: str, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    log.info("saved %s", name)


def make_eval_plots(artifacts_dir: Path | None = None, out_dir: Path | None = None) -> None:
    art = artifacts_dir or ARTIFACTS_DIR
    out = out_dir or PLOTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((art / "metrics.json").read_text())
    importance = json.loads((art / "feature_importance.json").read_text())
    bt = pd.read_parquet(PROCESSED_DIR / "backtest_predictions.parquet")
    bt["date"] = pd.to_datetime(bt["date"])

    # 1. Model comparison (validation vs test MAE, RMSE)
    comp = metrics["comparison"]
    names = list(comp)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, key in zip(ax, ("mae", "rmse"), strict=True):
        a.bar(
            x - 0.2,
            [comp[n]["validation"][key] for n in names],
            0.4,
            label="validation",
            color=BLUE,
        )
        a.bar(x + 0.2, [comp[n]["test"][key] for n in names], 0.4, label="test", color=ORANGE)
        a.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
        a.set_title(key.upper())
        a.legend()
    _save(fig, "01_model_comparison.png", out)

    # 2. Actual vs predicted: daily total over the test period
    daily = bt.groupby("date")[["actual", "predicted", "baseline_lag7"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily["actual"], label="actual", color=BLUE, lw=2)
    ax.plot(daily.index, daily["predicted"], label="model", color=ORANGE, lw=2)
    ax.plot(daily.index, daily["baseline_lag7"], label="naive lag-7", color=AQUA, lw=1.4, ls="--")
    ax.set_title(
        f"Test period: daily total units, actual vs predicted ({metrics['selected_model']})"
    )
    ax.legend()
    _save(fig, "02_actual_vs_predicted_daily.png", out)

    # 3. Scatter (sample) actual vs predicted
    s = bt.sample(min(20000, len(bt)), random_state=0)
    lim = float(np.percentile(bt["actual"], 99.5))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s["actual"], s["predicted"], s=4, alpha=0.25, color=BLUE)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Actual vs predicted (20k-row sample, test period)")
    _save(fig, "03_scatter_actual_predicted.png", out)

    # 4. Error distribution + residual analysis
    resid = bt["predicted"] - bt["actual"]
    lo, hi = np.percentile(resid, [1, 99])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.histplot(resid.clip(lo, hi), bins=80, ax=ax[0], color=BLUE)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_title(
        "Error (pred - actual), clipped p1-p99\n"
        f"mean={resid.mean():.3f}, median={resid.median():.3f}"
    )
    ax[1].scatter(s["predicted"], (s["predicted"] - s["actual"]), s=4, alpha=0.25, color=ORANGE)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlim(0, lim)
    ax[1].set_ylim(lo * 2, hi * 2)
    ax[1].set_xlabel("predicted")
    ax[1].set_ylabel("residual")
    ax[1].set_title("Residuals vs predicted (heteroscedastic, as expected for counts)")
    by_dow = bt.assign(dow=bt["date"].dt.dayofweek, resid=resid).groupby("dow")["resid"].mean()
    ax[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_dow.values, color=BLUE)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set_title("Mean residual by weekday")
    _save(fig, "04_error_residuals.png", out)

    # 5. Feature importance
    top = dict(list(importance.items())[:15])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(list(top)[::-1], np.array(list(top.values()))[::-1] * 100, color=AQUA)
    ax.set_xlabel("share of total gain (%)")
    ax.set_title("Feature importance (top 15)")
    _save(fig, "05_feature_importance.png", out)

    # 6. Per-store error
    err = bt.assign(abs_err=(bt["predicted"] - bt["actual"]).abs())
    agg = err.groupby("store_id").agg(abs_err=("abs_err", "sum"), actual=("actual", "sum"))
    by_store = pd.DataFrame(
        {
            "MAE": err.groupby("store_id")["abs_err"].mean(),
            "WAPE %": agg["abs_err"] / agg["actual"] * 100,
        }
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    by_store["MAE"].plot.bar(ax=ax[0], color=BLUE)
    ax[0].set_title("MAE by store")
    by_store["WAPE %"].plot.bar(ax=ax[1], color=ORANGE)
    ax[1].set_title("WAPE % by store")
    _save(fig, "06_error_by_store.png", out)
