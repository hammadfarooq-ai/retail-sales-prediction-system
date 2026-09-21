"""Model loading + prediction (tiny LightGBM on synthetic data); real artifacts if present."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from synthetic_data import LAST_DATE, STORES, make_panel, write_tiny_artifacts

from ml.src.config import ARTIFACTS_DIR
from ml.src.features.engineering import FEATURE_COLUMNS
from ml.src.models.forecaster import Forecaster, ModelNotAvailableError, Override


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    return write_tiny_artifacts(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def fc(art):
    return Forecaster(art)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def test_loading_reads_metadata(fc):
    assert fc.version.startswith("lightgbm_tiny") and fc.has_interval
    assert fc.feature_columns == FEATURE_COLUMNS


def test_missing_artifacts_raise_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="missing"):
        Forecaster(tmp_path)


def test_corrupt_model_raises(art, tmp_path):
    for f in art.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    (tmp_path / "model.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelNotAvailableError, match="Could not load"):
        Forecaster(tmp_path)


def test_feature_config_mismatch_is_rejected(art, tmp_path):
    for f in art.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    cfg = json.loads((tmp_path / "feature_config.json").read_text())
    cfg["feature_columns"] = cfg["feature_columns"][:-1]
    (tmp_path / "feature_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ModelNotAvailableError, match="retrain"):
        Forecaster(tmp_path)


def test_forecast_shape_flags_and_bounds(fc, panel):
    out = fc.forecast(
        panel,
        pd.DataFrame(
            columns=[
                "store_id",
                "item_id",
                "date",
                "promo_price",
                "promo_before_price",
                "promo_type",
            ]
        ),
        STORES,
        (LAST_DATE + timedelta(days=15)).date(),
    )
    assert len(out) == 6 * 15  # 2 stores x 3 items x 15 days
    assert (out["predicted"] >= 0).all() and np.isfinite(out["predicted"]).all()
    direct, rec = out[~out["is_recursive"]], out[out["is_recursive"]]
    assert direct["date"].max() == LAST_DATE + timedelta(days=7)
    assert rec["date"].min() == LAST_DATE + timedelta(days=8)
    assert direct["lower"].notna().all() and rec["lower"].isna().all()
    assert (direct["lower"] <= direct["predicted"] + 1e-9).all() and (
        direct["predicted"] <= direct["upper"] + 1e-9
    ).all()


def test_forecast_requires_future_end_date(fc, panel):
    with pytest.raises(ValueError, match="after the last observed day"):
        fc.forecast(panel, pd.DataFrame(), STORES, LAST_DATE.date())


def test_history_prediction_is_reasonable_on_synthetic_data(fc, panel):
    res = fc.predict_history_dates(
        panel[panel.date >= LAST_DATE - pd.Timedelta(days=40)].pipe(lambda d: panel),
        STORES,
        {(LAST_DATE - timedelta(days=3)).date()},
    )
    assert len(res) == 6 and (res["predicted"] > 0).all()
    actual = panel[panel.date == LAST_DATE - pd.Timedelta(days=3)]["quantity"].to_numpy()
    assert (
        np.mean(np.abs(np.sort(res["predicted"].to_numpy()) - np.sort(actual))) < 6
    )  # loose sanity bound


def test_predictions_do_not_depend_on_unobservable_future_values(fc, panel):
    """Changing quantities inside the 7-day blind window must not change a day's prediction."""
    target = (LAST_DATE - timedelta(days=2)).date()
    a = fc.predict_history_dates(panel, STORES, {target})
    p2 = panel.copy()
    p2.loc[p2["date"] > LAST_DATE - pd.Timedelta(days=8), "quantity"] += 500
    b = fc.predict_history_dates(p2, STORES, {target})
    np.testing.assert_allclose(a["predicted"].to_numpy(), b["predicted"].to_numpy())


def test_overrides_are_applied_and_echoed(fc, panel):
    key = (1, "aaa111", (LAST_DATE + timedelta(days=1)).date())
    empty = pd.DataFrame(
        columns=["store_id", "item_id", "date", "promo_price", "promo_before_price", "promo_type"]
    )
    one = panel[(panel.store_id == 1) & (panel.item_id == "aaa111")]
    res = fc.forecast(
        one, empty, STORES, key[2], {key: Override(price=50.0, promotion=True, discount_pct=50)}
    )
    row = res.iloc[0]
    assert (
        row["effective_price"] == 50.0
        and bool(row["effective_promo"])
        and row["effective_discount_pct"] == 50.0
    )
    base = fc.forecast(one, empty, STORES, key[2]).iloc[0]
    assert not bool(base["effective_promo"]) or base["effective_discount_pct"] >= 0


@pytest.mark.requires_artifacts
def test_real_artifacts_load_and_predict():
    if not (ARTIFACTS_DIR / "model.joblib").exists():
        pytest.skip("real artifacts not trained yet (run `make pipeline`)")
    real = Forecaster(ARTIFACTS_DIR)
    X = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 1.0)])
    pred = real._point(X)
    assert pred.shape == (1,) and np.isfinite(pred).all() and pred[0] >= 0
    meta = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())
    assert meta["n_features"] == len(FEATURE_COLUMNS)
    assert (
        meta["split"]["train_end"]
        < meta["split"]["val_start"]
        <= meta["split"]["val_end"]
        < meta["split"]["test_start"]
    )


def test_chunked_forecast_equals_single_pass(fc, panel):
    empty = pd.DataFrame(
        columns=["store_id", "item_id", "date", "promo_price", "promo_before_price", "promo_type"]
    )
    end = (LAST_DATE + timedelta(days=12)).date()
    whole = fc.forecast(panel, empty, STORES, end, chunk_pairs=1000)
    chunked = fc.forecast(panel, empty, STORES, end, chunk_pairs=2)  # 6 series -> 3 chunks
    key = ["store_id", "item_id", "date"]
    a = whole.sort_values(key).reset_index(drop=True)
    b = chunked.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
