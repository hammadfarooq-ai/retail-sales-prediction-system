# Hosted backend image (Render, Railway etc.). Build context = repository root.
# This file is a root-level copy of deploy/Dockerfile so hosts that default to a
# root Dockerfile (e.g. Render) find it without extra configuration.
# Unlike backend/Dockerfile, the model artifacts and seed tables are baked in
# (deploy/make_bundle.py creates them), so no volume mounts are needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/backend \
    MODEL_ARTIFACTS_DIR=/app/ml/artifacts \
    PROCESSED_DATA_DIR=/app/data/processed

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
COPY deploy/artifacts /app/ml/artifacts
COPY deploy/seed /app/data/processed

RUN chmod +x /app/backend/docker-entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser /app
USER appuser
WORKDIR /app/backend

# The host (Railway) injects $PORT. The entrypoint seeds PostgreSQL on first start, then execs the command.
EXPOSE 10000
ENTRYPOINT ["/app/backend/docker-entrypoint.sh"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
