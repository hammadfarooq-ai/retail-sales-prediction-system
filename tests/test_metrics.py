"""Metrics: hand-computed expectations and zero handling."""

from __future__ import annotations

import numpy as np
import pytest

from ml.src.evaluation import metrics as m


def test_basic_metrics_hand_computed():
    y, p = [10, 20, 30], [12, 18, 33]
    assert m.mae(y, p) == pytest.approx((2 + 2 + 3) / 3)
    assert m.rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m.wape(y, p) == pytest.approx(7 / 60 * 100)
    assert m.bias(y, p) == pytest.approx((2 - 2 + 3) / 3)
    assert m.mape(y, p) == pytest.approx((0.2 + 0.1 + 0.1) / 3 * 100)


def test_perfect_forecast():
    y = [1.0, 2.0, 3.0]
    assert m.mae(y, y) == 0 and m.rmse(y, y) == 0 and m.smape(y, y) == 0 and m.r2(y, y) == 1


def test_mape_ignores_zero_actuals_and_smape_handles_zero_zero():
    assert m.mape([0, 10], [5, 5]) == pytest.approx(50.0)  # only the y=10 row counts
    assert np.isnan(m.mape([0, 0], [1, 1]))
    assert m.smape([0, 0], [0, 0]) == 0
    assert m.smape([0], [4]) == pytest.approx(200.0)


def test_r2_can_be_negative_and_constant_target_is_nan():
    assert m.r2([1, 2, 3], [3, 2, 1]) < 0
    assert np.isnan(m.r2([2, 2, 2], [1, 2, 3]))


def test_all_metrics_reports_coverage_and_rows():
    out = m.all_metrics([0, 1, 2, 0], [0, 1, 1, 1])
    assert out["n_rows"] == 4 and out["mape_coverage_pct"] == 50.0
    assert set(out) >= {"mae", "rmse", "mape", "smape", "wape", "r2", "bias"}
