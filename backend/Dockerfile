# Build context = repository root (the API imports the shared `ml` package).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# libgomp is required by LightGBM / scikit-learn; curl is used by the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY ml/__init__.py /app/ml/__init__.py
COPY ml/src /app/ml/src
COPY backend/app /app/backend/app
COPY backend/docker-entrypoint.sh /app/backend/docker-entrypoint.sh
RUN chmod +x /app/backend/docker-entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser /app
USER appuser

# The app package lives in /app/backend/app
WORKDIR /app/backend
ENV PYTHONPATH=/app:/app/backend

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=360s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/backend/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
