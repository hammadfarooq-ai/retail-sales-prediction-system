"""Domain exceptions + centralized handlers. Internal details never reach the client."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class UnknownStoreError(NotFoundError):
    code = "unknown_store"


class UnknownProductError(NotFoundError):
    code = "unknown_product"


class UnprocessableError(AppError):
    status_code = 422
    code = "unprocessable"


class InvalidDateError(UnprocessableError):
    code = "invalid_date"


class NotForecastableError(UnprocessableError):
    code = "not_forecastable"


class ModelUnavailableError(AppError):
    status_code = 503
    code = "model_unavailable"


class PredictionError(AppError):
    status_code = 500
    code = "prediction_failed"


def _body(code: str, message: str, details: Any = None, request: Request | None = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"error": err}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("AppError %s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code, content=_body(exc.code, exc.message, exc.details)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_body("validation_error", "Request validation failed", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content=_body("http_error", str(exc.detail))
        )

    @app.exception_handler(SQLAlchemyError)
    async def _db(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        log.exception("Database error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503, content=_body("database_error", "Database temporarily unavailable")
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content=_body("internal_error", "An unexpected error occurred")
        )
