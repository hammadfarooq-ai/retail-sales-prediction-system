"""Prediction history queries."""

from __future__ import annotations

from datetime import date, datetime, time

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.models import Prediction
from app.schemas.predict import PredictionPage, PredictionRecord


def list_predictions(
    db: Session,
    *,
    store_id: int | None,
    item_id: str | None,
    model_version: str | None,
    created_from: date | None,
    created_to: date | None,
    target_from: date | None,
    target_to: date | None,
    limit: int,
    offset: int,
) -> PredictionPage:
    t = Prediction
    where = []
    if store_id is not None:
        where.append(t.store_id == store_id)
    if item_id:
        where.append(t.item_id == item_id)
    if model_version:
        where.append(t.model_version == model_version)
    if created_from:
        where.append(t.created_at >= datetime.combine(created_from, time.min))
    if created_to:
        where.append(t.created_at <= datetime.combine(created_to, time.max))
    if target_from:
        where.append(t.target_date >= target_from)
    if target_to:
        where.append(t.target_date <= target_to)
    total = db.scalar(select(func.count()).select_from(t).where(*where)) or 0
    rows = db.scalars(
        select(t)
        .where(*where)
        .order_by(t.created_at.desc(), t.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return PredictionPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[PredictionRecord.model_validate(r) for r in rows],
    )


def get_prediction(db: Session, prediction_id: int) -> PredictionRecord:
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise NotFoundError(f"Prediction {prediction_id} not found")
    return PredictionRecord.model_validate(row)


def latest_prediction(db: Session) -> PredictionRecord | None:
    row = db.scalar(
        select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(1)
    )
    return PredictionRecord.model_validate(row) if row else None
