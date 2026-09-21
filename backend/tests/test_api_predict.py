"""Prediction endpoint: happy paths, validation, error handling, persistence."""

from __future__ import annotations

from datetime import date, timedelta

from conftest import IN_3, IN_14, NEXT_DAY, PAST

BASE = {"store_id": 1, "item_id": "aaa111"}


def post(client, **kw):
    return client.post("/api/v1/predict", json={**BASE, **kw})


def test_predict_next_day_has_interval_and_inputs(client):
    r = post(client, date=NEXT_DAY)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_quantity"] >= 0
    assert body["mode"] == "forecast" and body["horizon_days"] == 1
    assert body["is_recursive"] is False
    assert body["lower"] is not None and body["upper"] is not None
    assert body["lower"] <= body["predicted_quantity"] <= body["upper"]
    assert body["interval_level"] == 0.8
    assert body["inputs"]["last_observed_price"] is not None
    assert body["dept_name"] == "DAIRY" and body["id"] >= 1


def test_predict_uses_promo_calendar_for_scheduled_day(client):
    assert post(client, date=IN_3).json()["inputs"]["promotion_source"] in {
        "none",
        "promo_calendar",
    }
    # the fixture schedules a 30%-off promotion for LAST+2
    r2 = post(client, date=str(date.fromisoformat(NEXT_DAY) + timedelta(days=1)))
    assert r2.json()["inputs"]["promotion"] is True
    assert r2.json()["inputs"]["promotion_source"] == "promo_calendar"
    assert round(r2.json()["inputs"]["discount_pct"]) == 30


def test_predict_beyond_seven_days_is_recursive_without_interval(client):
    r = post(client, date=IN_14)
    assert r.status_code == 200
    b = r.json()
    assert b["is_recursive"] is True and b["lower"] is None and b["upper"] is None
    assert b["horizon_days"] == 14


def test_predict_past_date_returns_actual(client):
    r = post(client, date=PAST)
    assert r.status_code == 200
    b = r.json()
    assert b["actual_quantity"] is not None
    assert b["mode"] in {"in_sample", "held_out_test"}
    assert b["horizon_days"] < 0


def test_user_overrides_are_echoed(client):
    r = post(client, date=NEXT_DAY, price=55.5, promotion=True, discount_pct=40)
    assert r.status_code == 200
    inp = r.json()["inputs"]
    assert inp["price"] == 55.5 and inp["promotion"] is True
    assert inp["discount_pct"] == 40 and inp["promotion_source"] == "user"


def test_promotion_changes_the_prediction_input_not_crash(client):
    a = post(client, date=NEXT_DAY, promotion=False).json()
    b = post(client, date=NEXT_DAY, promotion=True, discount_pct=50).json()
    assert a["inputs"]["promotion"] is False and b["inputs"]["promotion"] is True


def test_prediction_is_deterministic(client):
    a = post(client, date=NEXT_DAY).json()["predicted_quantity"]
    b = post(client, date=NEXT_DAY).json()["predicted_quantity"]
    assert a == b


def test_unknown_store_404(client):
    r = post(client, store_id=99, date=NEXT_DAY)
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_store"


def test_unknown_product_404(client):
    r = post(client, item_id="doesnotexist", date=NEXT_DAY)
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_product"


def test_not_forecastable_product_422(client):
    r = post(client, item_id="zzz999", date=NEXT_DAY)
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_forecastable"


def test_product_without_sales_in_store_422(client):
    # zzz999 is not forecastable in any store
    r = post(client, item_id="zzz999", store_id=2, date=NEXT_DAY)
    assert r.status_code == 422


def test_date_outside_supported_range_422(client):
    far = post(client, date="2030-01-01")
    assert far.status_code == 422 and far.json()["error"]["code"] == "invalid_date"
    assert "max_date" in far.json()["error"]["details"]
    early = post(client, date="2023-01-01")
    assert early.status_code == 422 and early.json()["error"]["code"] == "invalid_date"


def test_invalid_date_format_422(client):
    r = post(client, date="2024-13-45")
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error"
    assert any(d["field"] == "date" for d in body["details"])


def test_invalid_inputs_rejected(client):
    assert post(client, date=NEXT_DAY, price=-5).status_code == 422
    assert post(client, date=NEXT_DAY, price=0).status_code == 422
    assert post(client, date=NEXT_DAY, discount_pct=120).status_code == 422
    assert post(client, date=NEXT_DAY, promotion=False, discount_pct=20).status_code == 422
    assert post(client, date=NEXT_DAY, store_id=0).status_code == 422
    assert post(client, date=NEXT_DAY, item_id="bad id; DROP TABLE").status_code == 422
    assert client.post("/api/v1/predict", json={}).status_code == 422


def test_sql_injection_like_input_is_harmless(client):
    r = post(client, date=NEXT_DAY, item_id="a'--")
    assert r.status_code == 422  # rejected by the ID pattern before reaching the DB
    assert client.get("/health").status_code == 200


def test_model_missing_returns_503_not_stack_trace(client_no_model):
    r = client_no_model.post("/api/v1/predict", json={**BASE, "date": NEXT_DAY})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_unavailable"
    assert "Traceback" not in r.text


def test_prediction_saved_and_listed(client):
    created = post(client, date=NEXT_DAY, price=99.0).json()
    got = client.get(f"/api/v1/predictions/{created['id']}")
    assert got.status_code == 200
    rec = got.json()
    assert rec["predicted_quantity"] == created["predicted_quantity"]
    assert rec["inputs"]["request"]["price"] == 99.0
    assert rec["model_version"] == created["model_version"]
    lst = client.get("/api/v1/predictions", params={"store_id": 1, "item_id": "aaa111"}).json()
    assert lst["total"] == 1 and lst["items"][0]["id"] == created["id"]
    assert client.get("/api/v1/predictions/latest").json()["id"] == created["id"]


def test_failed_predictions_are_not_saved(client):
    post(client, item_id="doesnotexist", date=NEXT_DAY)
    assert client.get("/api/v1/predictions").json()["total"] == 0


def test_history_pagination_and_filters(client):
    for _ in range(3):
        post(client, date=NEXT_DAY)
    post(client, store_id=2, item_id="bbb222", date=NEXT_DAY)
    page = client.get("/api/v1/predictions", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 4 and len(page["items"]) == 2
    page2 = client.get("/api/v1/predictions", params={"limit": 2, "offset": 2}).json()
    assert len(page2["items"]) == 2
    assert {i["id"] for i in page["items"]}.isdisjoint({i["id"] for i in page2["items"]})
    assert client.get("/api/v1/predictions", params={"store_id": 2}).json()["total"] == 1
    assert (
        client.get("/api/v1/predictions", params={"target_from": "2999-01-01"}).json()["total"] == 0
    )
    assert client.get("/api/v1/predictions", params={"limit": 1000}).status_code == 422
    assert client.get("/api/v1/predictions/999999").status_code == 404


def test_batch_partial_success(client):
    items = [
        {**BASE, "date": NEXT_DAY},
        {"store_id": 2, "item_id": "bbb222", "date": IN_3, "promotion": True, "discount_pct": 25},
        {"store_id": 1, "item_id": "nope", "date": NEXT_DAY},
        {**BASE, "date": "2030-01-01"},
    ]
    r = client.post("/api/v1/predict/batch", json={"items": items})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["n_requested"] == 4 and b["n_succeeded"] == 2
    assert {e["index"] for e in b["errors"]} == {2, 3}
    assert {e["code"] for e in b["errors"]} == {"unknown_product", "invalid_date"}
    assert client.get("/api/v1/predictions", params={"limit": 10}).json()["total"] == 2


def test_batch_validation(client):
    assert client.post("/api/v1/predict/batch", json={"items": []}).status_code == 422
    assert client.post("/api/v1/predict/batch", json={}).status_code == 422
    bad = client.post("/api/v1/predict/batch", json={"items": [{**BASE, "date": "x"}]})
    assert bad.status_code == 422


def test_batch_matches_single(client):
    single = post(client, date=IN_3).json()["predicted_quantity"]
    batch = client.post("/api/v1/predict/batch", json={"items": [{**BASE, "date": IN_3}]}).json()
    assert batch["results"][0]["predicted_quantity"] == single
