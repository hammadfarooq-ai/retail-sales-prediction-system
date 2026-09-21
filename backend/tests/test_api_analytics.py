"""Health, model info, analytics and forecast endpoints."""

from __future__ import annotations

from datetime import timedelta

from conftest import LAST


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok" and b["database"] == "up" and b["model_loaded"] is True
    assert b["model_version"] == "lightgbm_tiny-202601010000"


def test_health_degraded_without_model(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded" and r.json()["model_loaded"] is False


def test_docs_and_openapi_available(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    for path in [
        "/health",
        "/api/v1/predict",
        "/api/v1/predict/batch",
        "/api/v1/model/info",
        "/api/v1/forecasts",
        "/api/v1/sales/summary",
    ]:
        assert path in spec["paths"]


def test_cors_headers_for_allowed_origin(client):
    r = client.options(
        "/api/v1/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options(
        "/api/v1/predict",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


def test_model_info_and_performance(client):
    info = client.get("/api/v1/model/info").json()
    assert info["model_name"] == "lightgbm_tiny" and info["n_features"] == len(
        info["feature_names"]
    )
    assert info["has_prediction_interval"] is True and info["horizon_days"] == 7
    perf = client.get("/api/v1/model/performance").json()
    assert "naive_seasonal_lag7" in perf["comparison"] and perf["final_model_test"]["mae"] == 1.0
    assert perf["feature_importance"]["lag_7"] == 0.4


def test_model_endpoints_503_without_model(client_no_model):
    assert client_no_model.get("/api/v1/model/info").status_code == 503


def test_backtest(client):
    b = client.get("/api/v1/model/backtest").json()
    assert len(b["points"]) == 6
    assert b["points"][0]["actual"] == 13.0 and b["points"][0]["predicted"] == 13.0
    assert sum(h["count"] for h in b["error_histogram"]) == 12
    only = client.get("/api/v1/model/backtest", params={"store_id": 1, "item_id": "aaa111"}).json()
    assert only["points"][0]["actual"] == 5.0 and only["scope"] == "item"
    assert client.get("/api/v1/model/backtest", params={"store_id": 42}).status_code == 404


def test_summary_matches_database(client, db):
    from app.models import DailyStoreSales
    from sqlalchemy import func, select

    total = db.scalar(select(func.sum(DailyStoreSales.quantity)))
    b = client.get("/api/v1/sales/summary").json()
    assert abs(b["total_quantity"] - total) < 1e-6
    assert b["n_stores"] == 2 and b["n_days"] == 150 and b["n_departments"] == 2
    assert abs(b["avg_daily_quantity"] - total / 150) < 1e-6


def test_summary_filters(client):
    all_ = client.get("/api/v1/sales/summary").json()
    s1 = client.get("/api/v1/sales/summary", params={"store_id": 1}).json()
    dairy = client.get("/api/v1/sales/summary", params={"dept_name": "DAIRY"}).json()
    assert s1["total_quantity"] < all_["total_quantity"] and s1["n_stores"] == 1
    assert dairy["total_quantity"] < all_["total_quantity"] and dairy["n_departments"] == 1
    ranged = client.get(
        "/api/v1/sales/summary", params={"start_date": "2024-01-01", "end_date": "2024-01-10"}
    ).json()
    assert ranged["n_days"] == 10


def test_summary_invalid_inputs(client):
    assert client.get("/api/v1/sales/summary", params={"store_id": 9}).status_code == 404
    r = client.get(
        "/api/v1/sales/summary", params={"start_date": "2024-03-01", "end_date": "2024-02-01"}
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_date"
    assert (
        client.get("/api/v1/sales/summary", params={"start_date": "not-a-date"}).status_code == 422
    )


def test_trends_granularities(client):
    day = client.get("/api/v1/sales/trends", params={"granularity": "day"}).json()
    week = client.get("/api/v1/sales/trends", params={"granularity": "week"}).json()
    month = client.get("/api/v1/sales/trends", params={"granularity": "month"}).json()
    assert len(day["points"]) == 150
    assert 20 <= len(week["points"]) <= 23
    assert len(month["points"]) == 5  # Jan-May 2024
    assert (
        abs(sum(p["quantity"] for p in day["points"]) - sum(p["quantity"] for p in month["points"]))
        < 1e-6
    )
    by_store = client.get(
        "/api/v1/sales/trends", params={"granularity": "month", "group_by": "store"}
    ).json()
    assert {p["store_id"] for p in by_store["points"]} == {1, 2}
    assert client.get("/api/v1/sales/trends", params={"granularity": "year"}).status_code == 422
    assert (
        len(
            client.get(
                "/api/v1/sales/trends", params={"dept_name": "BAKERY", "granularity": "month"}
            ).json()["points"]
        )
        == 5
    )


def test_by_store(client):
    rows = client.get("/api/v1/sales/by-store").json()
    assert [r["store_id"] for r in rows] == [1, 2]
    assert rows[0]["area"] == 1500 and rows[0]["n_products"] == 4
    assert abs(rows[0]["avg_daily_revenue_per_m2"] - rows[0]["avg_daily_revenue"] / 1500) < 1e-9


def test_by_product_filter_sort_paginate(client):
    b = client.get("/api/v1/sales/by-product", params={"limit": 3}).json()
    assert b["total"] == 8 and len(b["items"]) == 3
    revs = [i["revenue"] for i in b["items"]]
    assert revs == sorted(revs, reverse=True)
    only = client.get("/api/v1/sales/by-product", params={"forecastable_only": True}).json()
    assert only["total"] == 6 and all(i["forecastable"] for i in only["items"])
    dairy = client.get(
        "/api/v1/sales/by-product", params={"dept_name": "DAIRY", "store_id": 1}
    ).json()
    assert dairy["total"] == 2
    found = client.get("/api/v1/sales/by-product", params={"search": "chees"}).json()
    assert {i["item_id"] for i in found["items"]} == {"bbb222"}
    assert client.get("/api/v1/sales/by-product", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/sales/by-product", params={"sort_by": "colour"}).status_code == 422


def test_by_category(client):
    rows = client.get("/api/v1/sales/by-category").json()
    assert {r["dept_name"] for r in rows} == {"DAIRY", "BAKERY"}
    assert abs(sum(r["revenue_share_pct"] for r in rows) - 100) < 1e-6
    assert rows[0]["revenue"] >= rows[1]["revenue"]


def test_filters_and_catalog(client):
    f = client.get("/api/v1/meta/filters").json()
    assert len(f["stores"]) == 2 and set(f["departments"]) == {"DAIRY", "BAKERY"}
    assert f["max_date"] == str(LAST) and f["forecastable_pairs"] == 6
    prods = client.get("/api/v1/products", params={"store_id": 1}).json()
    assert {p["item_id"] for p in prods} == {
        "aaa111",
        "bbb222",
        "ccc333",
    }  # zzz999 not forecastable
    assert client.get("/api/v1/products", params={"forecastable_only": False}).json()
    assert client.get("/api/v1/stores").json()[0]["store_id"] == 1


def test_forecast_item(client):
    r = client.get(
        "/api/v1/forecasts",
        params={"store_id": 1, "item_id": "aaa111", "horizon_days": 14, "history_days": 30},
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["scope"] == "item" and len(b["forecast"]) == 14 and len(b["history"]) == 30
    assert b["forecast"][0]["date"] == str(LAST + timedelta(days=1))
    assert b["history"][-1]["date"] == str(LAST) and b["last_observed_date"] == str(LAST)
    assert all(p["is_recursive"] is False and p["lower"] is not None for p in b["forecast"][:7])
    assert all(p["is_recursive"] is True and p["lower"] is None for p in b["forecast"][7:])
    assert all(p["predicted"] >= 0 for p in b["forecast"])
    assert b["interval_level"] == 0.8


def test_forecast_store_total(client):
    b = client.get("/api/v1/forecasts", params={"store_id": 1, "horizon_days": 7}).json()
    assert b["scope"] == "store" and b["n_series"] == 3 and len(b["forecast"]) == 7
    assert b["interval_level"] is None and any("sum of 3" in n for n in b["notes"])


def test_forecast_errors(client):
    assert client.get("/api/v1/forecasts", params={"store_id": 9}).status_code == 404
    assert (
        client.get("/api/v1/forecasts", params={"store_id": 1, "item_id": "nope"}).status_code
        == 404
    )
    assert (
        client.get("/api/v1/forecasts", params={"store_id": 1, "item_id": "zzz999"}).status_code
        == 422
    )
    assert (
        client.get("/api/v1/forecasts", params={"store_id": 1, "horizon_days": 0}).status_code
        == 422
    )
    assert (
        client.get("/api/v1/forecasts", params={"store_id": 1, "horizon_days": 60}).status_code
        == 422
    )  # > configured max (28)
    assert client.get("/api/v1/forecasts").status_code == 422


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/api/v1/nothing-here")
    assert r.status_code == 404 and "error" in r.json()
