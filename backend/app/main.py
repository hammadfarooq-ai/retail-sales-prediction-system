"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import health
from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import mask_url, setup_logging
from app.ml.registry import ModelRegistry

log = logging.getLogger("app")

DESCRIPTION = """
Retail sales forecasting API.

* **Predict** daily units sold for a *store x product x date* (`/api/v1/predict`, `/predict/batch`).
* **Forecast** multi-day series with history (`/api/v1/forecasts`).
* **Analytics** on the real sales data (`/api/v1/sales/*`).
* **Model** metadata, comparison and backtest results (`/api/v1/model/*`).

The model is trained offline (`ml/`) and loaded once at start-up; requests never retrain it.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment
        )
        log.info("Database: %s", mask_url(settings.sqlalchemy_url))
        registry = ModelRegistry()
        registry.load()  # the model is loaded ONCE here and reused by every request
        app.state.registry = registry
        yield
        log.info("Shutting down")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    # Tests (TestClient without lifespan) can still resolve a registry.
    app.state.registry = ModelRegistry()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        if request.url.path != "/health":
            log.info(
                "%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code, ms
            )
        return response

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(v1_router)
    return app


app = create_app()
