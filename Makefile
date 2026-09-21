.PHONY: help install pipeline data eda train plots seed api frontend-install web test test-ml test-backend test-frontend lint format typecheck up down logs clean

PY ?= python

help:
	@echo "Targets:"
	@echo "  install          install Python (dev) + frontend dependencies"
	@echo "  pipeline         data prep -> EDA -> train/evaluate -> plots (needs ./data/*.csv)"
	@echo "  seed             load processed data + model registry into PostgreSQL"
	@echo "  api              run FastAPI locally on :8000"
	@echo "  web              run the Vite dev server on :5173"
	@echo "  test             backend + ML + frontend tests"
	@echo "  lint / format / typecheck"
	@echo "  up / down        docker compose up --build / down"

install:
	$(PY) -m pip install -r requirements-dev.txt
	cd frontend && npm install

data:
	$(PY) ml/scripts/01_prepare_data.py

eda:
	$(PY) ml/scripts/02_run_eda.py

train:
	$(PY) ml/scripts/03_train_models.py

plots:
	$(PY) ml/scripts/04_make_plots.py

pipeline: data eda train plots

seed:
	cd backend && PYTHONPATH=.. $(PY) -m app.db.seed

api:
	cd backend && PYTHONPATH=.. $(PY) -m uvicorn app.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test: test-backend test-frontend

test-ml:
	$(PY) -m pytest tests -q

test-backend:
	$(PY) -m pytest tests backend/tests -q

test-frontend:
	cd frontend && npm test

lint:
	ruff check .
	black --check .
	cd frontend && npm run lint

format:
	ruff check --fix .
	black .

typecheck:
	mypy ml/src backend/app
	cd frontend && npm run typecheck

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	rm -rf frontend/dist .pytest_cache .mypy_cache .ruff_cache
