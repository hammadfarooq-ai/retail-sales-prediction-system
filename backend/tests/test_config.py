"""Settings: managed-Postgres URLs and CORS regex."""

from __future__ import annotations

from app.core.config import Settings, normalize_database_url
from app.main import create_app
from fastapi.testclient import TestClient


def test_managed_postgres_urls_are_normalised():
    assert normalize_database_url("postgres://u:p@h:5432/d") == "postgresql+psycopg://u:p@h:5432/d"
    assert normalize_database_url("postgresql://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert normalize_database_url("postgresql+psycopg://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert normalize_database_url("sqlite://") == "sqlite://"


def test_settings_uses_normalised_database_url():
    s = Settings(database_url="postgres://u:p@h:5432/d")
    assert s.sqlalchemy_url.startswith("postgresql+psycopg://")


def test_cors_regex_allows_matching_origin_only(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGIN_REGEX", r"^https://retail-sales-prediction[a-z0-9-]*\.vercel\.app$"
    )
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        hdr = {"Access-Control-Request-Method": "GET"}
        ok = client.options(
            "/health", headers={"Origin": "https://retail-sales-prediction-abc.vercel.app", **hdr}
        )
        bad = client.options("/health", headers={"Origin": "https://evil.vercel.app", **hdr})
        assert (
            ok.headers.get("access-control-allow-origin")
            == "https://retail-sales-prediction-abc.vercel.app"
        )
        assert "access-control-allow-origin" not in bad.headers
    finally:
        get_settings.cache_clear()
