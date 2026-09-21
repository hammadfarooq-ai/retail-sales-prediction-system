"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.ml.registry import ModelRegistry

DbSession = Annotated[Session, Depends(get_session)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


Registry = Annotated[ModelRegistry, Depends(get_registry)]
