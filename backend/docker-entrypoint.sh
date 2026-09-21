#!/bin/sh
# Wait for PostgreSQL (compose also gates on its health check), seed on first run, then start the API.
set -e

echo "Seeding database if needed..."
python -m app.db.seed || { echo "Seeding failed (are ml/artifacts and data/processed mounted? run 'make pipeline' first)"; exit 1; }

exec "$@"
