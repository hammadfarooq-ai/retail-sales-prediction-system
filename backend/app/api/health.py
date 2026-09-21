from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.api.deps import AppSettings, DbSession, Registry
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness/readiness probe")
def health(db: DbSession, registry: Registry, settings: AppSettings, response: Response):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    ok = db_ok and registry.loaded
    if not db_ok:
        response.status_code = 503
    return HealthResponse(
        status="ok" if ok else "degraded",
        version=settings.app_version,
        database="up" if db_ok else "down",
        model_loaded=registry.loaded,
        model_version=registry.forecaster.version if registry.forecaster else None,
        time=datetime.now(UTC),
    )
