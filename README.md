# Retail Sales Prediction System

End-to-end retail demand forecasting: a leakage-safe ML pipeline, a FastAPI service backed by
PostgreSQL, and a React + TypeScript analytics dashboard — built on the
[Kaggle *Retail Sales Forecasting* dataset](https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data).

**Dataset used:** [Retail Sales Forecasting Data (Kaggle, by svizor)](https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data) — <https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data>

> Every number in this README (data statistics, metrics, feature importances) was produced by running
> the code in this repository on the real dataset. Nothing is invented; where something is weak, it is
> said so in [Limitations](#limitations).

## Contents

[Overview](#overview) · [Features](#features) · [Architecture](#architecture) · [Tech stack](#tech-stack) ·
[Dataset](#dataset) · [ML problem](#ml-problem) · [Feature engineering](#feature-engineering) ·
[Split](#train--validation--test-strategy) · [Models & results](#models-evaluated-and-actual-metrics) ·
[API](#backend-api) · [Database](#database) · [Frontend](#frontend) · [Structure](#project-structure) ·
[Installation](#installation) · [Docker](#docker) · [Deploy](#deploy-vercel--railway) · [Env vars](#environment-variables) ·
[Testing](#testing) · [Limitations](#limitations) · [Future work](#future-improvements)

## Overview

The system predicts **daily units sold for a store × product**, 7 days ahead (longer horizons are
produced recursively), and serves those predictions through a REST API and a dashboard.

Design decisions come from an inspection of the real files (full write-up:
[`docs/ANALYSIS_AND_DESIGN.md`](docs/ANALYSIS_AND_DESIGN.md)).

## Features

* Complete EDA on the real data (profile of all 8 CSVs + 10 figures) — [`docs/`](docs)
* Leakage-safe feature engineering with tests that prove it
* 7 models compared on chronological validation/test data (2 naive baselines, Ridge, Random Forest, XGBoost, LightGBM, LightGBM-Tweedie)
* 80% prediction interval (quantile models) with measured test coverage
* FastAPI: single & batch prediction, multi-day forecasts, sales analytics, model performance, prediction history
* PostgreSQL storage of derived aggregates and the prediction history
* React dashboard: dashboard, analytics, prediction form, forecasts (CSV export), stores, products, model performance, history, about
* Docker Compose (postgres + backend + frontend) with health checks
* 81 Python tests + 21 frontend tests, Ruff / Black / mypy / ESLint / `tsc` all clean

## Architecture

```
 data/*.csv ──► ml/ (prepare → features → train → evaluate) ──► ml/artifacts/*.joblib, metrics.json
                     │                                                   │ loaded once at start-up
                     └─► data/processed/*.parquet ─► app.db.seed ─► PostgreSQL ◄──── FastAPI (backend/) ◄──── React (frontend/)
                                                                    (aggregates,      /api/v1/*             nginx in Docker,
                                                                     recent history,                        Vite proxy in dev
                                                                     prediction log)
```

* `ml/src/features/engineering.py` is the **single** feature implementation used by training *and*
  serving, so there is no train/serve skew.
* The API **never trains**. `ModelRegistry` loads the artifacts once in the FastAPI lifespan; if they are
  missing the app still starts and reports `degraded` on `/health` (predictions return `503`).
* PostgreSQL holds only *derived* data (the 379 MB raw sales file is not duplicated).

## Tech stack

| Layer | Tools |
|---|---|
| ML | Python 3.11+ (developed on 3.13), pandas, NumPy, scikit-learn, LightGBM, XGBoost, joblib, matplotlib, seaborn |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2, psycopg 3, Uvicorn |
| Database | PostgreSQL 16 |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts, Axios, React Router |
| Tests | pytest, FastAPI `TestClient`, Vitest + Testing Library |
| Quality | Ruff, Black, mypy, ESLint |
| DevOps | Docker, Docker Compose, nginx |

## Dataset

Source: [Retail Sales Forecasting Data on Kaggle](https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data) — <https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data> (place the CSVs in `./data/`,
see [`data/README.md`](data/README.md)). The raw files are never modified.

### Dataset structure (measured)

| File | Rows | Grain | Notes |
|---|---:|---|---|
| `sales.csv` | 7,432,685 | (date, item, store) with ≥1 sale | 2022-08-28 → 2024-09-26, 761 consecutive days, 28,182 items, 4 stores; **only days with a sale exist** |
| `stores.csv` | 4 | store attributes | format, city, division, area (109–1,887 m²) |
| `catalog.csv` | 219,810 | item hierarchy | 196 departments / 613 classes / 1,007 subclasses (Russian names) |
| `discounts_history.csv` | 3,746,744 | promo calendar | includes **future-dated planned promos** (to 2045) |
| `online.csv` | 1,123,412 | online sales | 81% of its keys already in `sales.csv` → not added |
| `price_history.csv` | 698,626 | sparse price events | corrupt values, unclear codes → not used |
| `markdowns.csv` | 8,979 | clearance | tiny → not used |
| `actual_matrix.csv` | 35,202 | assortment snapshots | undocumented → not used |

Highlights: `price_base` is the realised net unit price; 0.12% of sales rows are returns / zero-value and
are removed; store 4 opened 2023-12-13; Friday revenue is 1.25× the weekly mean, Sunday 0.82×.
Details, missing values, duplicates, cardinalities: [`docs/data_profile.json`](docs/data_profile.json).

## ML problem

* **Target:** `quantity` (units) — **unit:** store × product × day — **horizon:** 7 days (direct), 8–28 days recursive.
* **Universe:** 6,418 store-product pairs with regular demand (sold on ≥70% of days since first sale, ≥90 sale days; chosen on the *training period only*) = **62.0% of revenue**, 3.23 M zero-filled panel rows.
* Why not everything? Item-level daily demand is very sparse (58,000 pairs; top 10% of pairs = 74% of revenue), and zero-filling all pairs would need >40 M rows.

## Feature engineering

33 features, all supported by real columns ([details](docs/ANALYSIS_AND_DESIGN.md#3-features-33)):
calendar (7), lags 7/14/21/28, rolling mean 7/14/28/56 and std 7/28, weekday-mean, 7d/28d ratio,
sale rate, days-since-last-sale, last observed price, planned promo price, discount depth, promo type,
promo days in the last 28 days, store area/format and department/class codes.

**Leakage prevention:** every history feature is shifted ≥ 7 days *within each series*; the realised
price of the target day is never used (it exists only if a sale happened); promotions are used as
known-in-advance covariates. `tests/test_features.py` perturbs all values in the 7-day blind window and
asserts the features do not change, and proves the fast vectorised code equals a slow reference on real
data.

## Train / validation / test strategy

Chronological, derived from the last observed date (no shuffling):

| Set | Period | Rows |
|---|---|---:|
| Train | 2022-08-28 → 2024-07-04 (rows with ≥35 days history) | 2,469,488 |
| Validation | 2024-07-05 → 2024-08-15 (42 days) | 269,556 |
| Test | 2024-08-16 → 2024-09-26 (42 days) | 269,556 |

Protocol: fit each candidate on train → score on validation and test (fair comparison table) → select the
best ML model by **validation MAE** → refit it on train+validation with a fixed iteration budget → report
its **test** metrics (test never used for selection or fitting).

## Models evaluated and actual metrics

Fit on the training period only. MAE/RMSE in units per store-product-day (test period: 269,556 rows).
Produced by `make train`; raw numbers in [`ml/artifacts/metrics.json`](ml/artifacts/metrics.json).

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE | Test WAPE | Test R² | Fit time* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive: same weekday last week | 3.929 | 8.375 | 3.710 | 8.405 | 46.6% | 0.949 | – |
| Naive: 28-day moving average | 3.656 | 9.611 | 3.438 | 9.183 | 43.2% | 0.939 | – |
| Ridge | 3.530 | 7.919 | 3.343 | 7.216 | 42.0% | 0.962 | 23 s |
| Random Forest (600k-row sample) | 3.293 | 7.890 | 3.119 | 7.219 | 39.2% | 0.962 | 511 s |
| XGBoost | 3.304 | 7.950 | 3.146 | 8.036 | 39.5% | 0.953 | 229 s |
| LightGBM | 3.279 | 7.635 | 3.132 | 7.742 | 39.3% | 0.957 | 164 s |
| **LightGBM, Tweedie objective** | **3.180** | **7.498** | **3.008** | **6.904** | **37.8%** | **0.966** | 485 s |

\* 4-core laptop, 8 GB RAM.

### Selected model

**LightGBM with a Tweedie objective** (non-negative, zero-inflated counts) — best validation MAE. After
refitting on train+validation (1,156 trees) its **held-out test** metrics are:

| MAE | RMSE | WAPE | SMAPE | MAPE† | R² | Bias | 80% interval coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **2.993** | 8.161 | 37.6% | 80.1% | 64.6% | 0.952 | +0.124 | **81.0%** (nominal 80%) |

† MAPE is defined only where actual > 0 (77.9% of rows).

By store (test): store 1 MAE 3.12 / WAPE 35.3%, store 2 2.30 / 41.8%, store 3 3.28 / 40.1%, store 4 2.98 / 38.6%.

Accuracy improves sharply with aggregation (same model, test period): **store-day** totals WAPE 4.4%
(R² 0.991); **product-store-week** WAPE 21.5% (R² 0.975); all-store **day** totals WAPE 3.1%.

Top features (share of gain): same-weekday 4-week mean 59.4%, 56-day rolling mean 21.1%, 28-day rolling
mean 8.0%, lag-7 2.1%, 7-day mean 1.2%, discount depth 0.7%, promo days 0.6%.

**Honest reading of these results**

* Versus the "same weekday last week" baseline the final model has **19.3% lower MAE** but only **2.9% lower RMSE**; a plain Ridge is close on RMSE (7.22 vs 6.90 for the train-only Tweedie fit).
* The train+val refit has *better* MAE (2.99 vs 3.01) but *worse* test RMSE (8.16 vs 6.90) than the train-only Tweedie model — RMSE is dominated by a few very large-quantity products, so it is noisy. If RMSE is your priority, ship the train-only fit.
* Daily item-level MAPE/SMAPE are poor by nature (many small counts and zeros); WAPE and aggregated errors are the meaningful headline numbers.
* Beyond 7 days the forecast is recursive and **has not been backtested**.

Figures: [`ml/artifacts/plots/`](ml/artifacts/plots) (model comparison, actual vs predicted, scatter,
residuals, feature importance, per-store error) and [`docs/figures/eda/`](docs/figures/eda).

## Backend API

Interactive docs at **`/docs`** (Swagger) and `/redoc`. All errors use one envelope:
`{"error": {"code": "...", "message": "...", "details": ...}}` — no stack traces are ever returned.

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness + DB + model status (`ok` / `degraded`) |
| `GET /api/v1/model/info` | version, features, split, interval coverage |
| `GET /api/v1/model/performance` | comparison table, final metrics, feature importance |
| `GET /api/v1/model/backtest` | actual vs predicted (test period), residual histogram |
| `POST /api/v1/predict` | one store × product × date (saved to history) |
| `POST /api/v1/predict/batch` | up to 500 rows; invalid rows reported in `errors` |
| `GET /api/v1/forecasts` | history + forecast for a product or a store total |
| `GET /api/v1/predictions[/{id}\|/latest]` | prediction history (filters + pagination) |
| `GET /api/v1/sales/summary` `…/trends` `…/by-store` `…/by-product` `…/by-category` | analytics |
| `GET /api/v1/stores` `…/products` `…/meta/filters` | dropdown data |

Status codes: `200`, `404` (unknown store/product), `422` (invalid input / date out of range / product not
forecastable), `503` (model or database unavailable).

### Examples

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/api/v1/predict -H 'Content-Type: application/json' \
  -d '{"store_id": 1, "item_id": "63161948a95a", "date": "2024-09-30"}'

# what-if: 30% promotion at a custom price
curl -s -X POST localhost:8000/api/v1/predict -H 'Content-Type: application/json' \
  -d '{"store_id": 1, "item_id": "63161948a95a", "date": "2024-09-30", "price": 315, "promotion": true, "discount_pct": 30}'

curl -s "localhost:8000/api/v1/forecasts?store_id=1&item_id=63161948a95a&horizon_days=14"
```

Response of `/predict` (fields): `predicted_quantity`, `lower`/`upper` (80% interval, days 1–7 only),
`mode` (`forecast` | `held_out_test` | `in_sample`), `actual_quantity` for past days, `is_recursive`,
`horizon_days`, and `inputs` (price used, last observed price, promotion, discount, promotion source).

## Database

PostgreSQL tables (SQLAlchemy models in [`backend/app/models/tables.py`](backend/app/models/tables.py)):
`stores`, `products` (28,167), `daily_store_sales` (2,571), `daily_category_sales` (350,169),
`product_sales` (58,000), `item_store_daily` (last 180 days of the forecastable universe, 1.16 M rows),
`promo_calendar` (planned promos for the next 90 days), `model_versions`, `backtest_predictions`
(269,556) and **`predictions`** (history: timestamp, inputs JSONB, prediction, interval, model version).
Row counts are from the actual seeded database. `python -m app.db.seed` loads them with `COPY`
(~50 s) and preserves prediction history across re-seeds.

## Frontend

Pages: Dashboard · Sales Analytics · Sales Prediction · Forecasts · Stores · Products · Model
Performance · Prediction History · About. All figures are fetched from the API (nothing hard-coded);
loading skeletons, empty and error states, toasts, accessible forms, responsive from phone to desktop,
route-level code splitting.

## Project structure

```
retail-sales-prediction-system/
├── data/                 raw Kaggle CSVs (untouched) + processed/ (generated) + README.md
├── ml/
│   ├── src/{data,features,models,evaluation,visualization}/   pipeline + shared feature code
│   ├── scripts/          01_prepare_data · 02_run_eda · 03_train_models · 04_make_plots
│   └── artifacts/        model.joblib, q10/q90, metrics.json, feature_config.json, plots/
├── backend/app/{api,core,db,models,schemas,services,ml}/      FastAPI service
├── backend/tests/        API + DB tests (SQLite, tiny real model)
├── frontend/src/{components,pages,layouts,services,hooks,types,utils}/
├── tests/                ML tests (features, leakage, metrics, forecaster, data prep)
├── docs/                 analysis, data profile, figures
├── docker-compose.yml · Makefile · .env.example · pyproject.toml
```

(The raw CSVs stay in `data/` rather than `data/raw/` because the brief said not to move them.)

## Installation

Requirements: Python ≥ 3.11, Node ≥ 20, Docker (for PostgreSQL / the full stack), the dataset in `./data/`.

```bash
python -m pip install -r requirements-dev.txt
cd frontend && npm install && cd ..
cp .env.example .env            # then set POSTGRES_PASSWORD

make pipeline                   # data prep → EDA → train & evaluate → plots  (~45 min on a 4-core laptop; training dominates)
```

## Local development

```bash
docker compose up -d postgres   # PostgreSQL on localhost:55432
make seed                       # load processed data + model version
make api                        # FastAPI on :8000  (docs: http://localhost:8000/docs)
make web                        # Vite on :5173, proxies /api to :8000
```

## Docker

After `make pipeline` (artifacts + processed data are mounted read-only into the backend):

```bash
cp .env.example .env            # set POSTGRES_PASSWORD

# A) bundled PostgreSQL container
docker compose --profile bundled-db up --build

# B) your own PostgreSQL (e.g. a database named RetailDB): set COMPOSE_DATABASE_URL in .env, see .env.example
docker compose up --build
```

With (B) the backend connects to PostgreSQL on your host via `host.docker.internal`; the bundled
`postgres` service is not started. The host PostgreSQL must accept connections from Docker.

* Frontend: <http://localhost:3000> (nginx; proxies `/api` and `/docs` to the backend)
* API docs: <http://localhost:8000/docs> · Health: <http://localhost:8000/health>

The backend container seeds PostgreSQL on first start (skipped when already up to date). All three
services have health checks; the frontend waits for a healthy backend, the backend for a healthy database.

## Deploy (Vercel + Railway)

The React frontend goes on **Vercel**; the API and PostgreSQL go on **Railway** (the API needs
LightGBM, pandas and a database, which do not fit Vercel's serverless limits).

1. **Bundle** (already committed): `python deploy/make_bundle.py` copies the model artifacts and slim
   seed tables (~27 MB, derived aggregates only — no raw Kaggle data) into `deploy/`.
2. **Railway:** create a project with a **PostgreSQL** database and a service built from this repo
   ([`railway.json`](railway.json) → [`deploy/Dockerfile`](deploy/Dockerfile)). Set on the service:
   `DATABASE_URL=${{Postgres.DATABASE_URL}}`, `ENVIRONMENT=production` and
   `CORS_ORIGIN_REGEX=^https://retail-sales-prediction[a-z0-9-]*\.vercel\.app$`, then generate a domain.
   The first start seeds the database (a few minutes). CLI: `railway init`, `railway add --database postgres`,
   `railway add --service api`, `railway up`, `railway domain`.
3. **Vercel:** import the repo (or `cd frontend && npx vercel --prod`), **Root Directory = `frontend`**,
   project name `retail-sales-prediction`, and set `VITE_API_URL=https://<your-railway-domain>`.
   [`frontend/vercel.json`](frontend/vercel.json) adds the SPA fallback.
   The API allows `https://retail-sales-prediction*.vercel.app`; set `CORS_ORIGINS` for a custom domain.

## Environment variables

See [`.env.example`](.env.example). Secrets are read only from the environment (`.env` is git-ignored);
DB URLs are masked in logs.

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `retail` / *(required)* / `retail_sales` | database credentials |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `localhost` / `55432` | used when the API runs on your machine |
| `DATABASE_URL` | – | full SQLAlchemy URL (overrides the above) |
| `COMPOSE_DATABASE_URL` | – | database URL used by `docker compose` (e.g. your own `RetailDB` via `host.docker.internal`); unset = bundled Postgres |
| `MODEL_ARTIFACTS_DIR` / `PROCESSED_DATA_DIR` | `./ml/artifacts` / `./data/processed` | relative paths resolve against the project root |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | allowed browser origins |
| `CORS_ORIGIN_REGEX` | – | extra allowed origins by regex (e.g. Vercel domains) |
| `HISTORY_DAYS_IN_DB` · `MAX_FORECAST_HORIZON_DAYS` · `MAX_BATCH_SIZE` | `180` · `28` · `500` | limits |
| `LOG_LEVEL` · `LOG_JSON` | `INFO` · `false` | logging |

## Testing

```bash
make test-backend     # 81 Python tests: ML + API + DB (no PostgreSQL needed; SQLite + a tiny real LightGBM)
make test-frontend    # 21 Vitest tests
make lint             # ruff + black --check + eslint
make typecheck        # mypy + tsc
```

Coverage highlights: leakage perturbation test, vectorised-vs-reference feature equivalence (synthetic and
real panel slice), chronological split, metrics with zero handling, model loading (missing / corrupt /
incompatible artifacts), health, prediction (interval, recursion, overrides, invalid dates/inputs, unknown
store/product, non-forecastable, batch partial success, persistence), analytics endpoints, DB seeding
(idempotency, history preserved), forecast, CORS, and the frontend prediction flow.

## Limitations

* **Coverage:** only 6,418 regular-demand store-product pairs (62% of revenue) are forecastable; new or intermittent items are not.
* **Accuracy at daily item level is modest** (WAPE ≈ 38%); it improves substantially when aggregated (store-day WAPE ≈ 4%).
* **RMSE gain over baselines is small** (2.9% vs same-weekday naive; Ridge is competitive on RMSE) — see the honest reading above.
* **Promotion calendar assumed known in advance.** If promotions were recorded retroactively, real-world accuracy would be lower.
* **Recursive forecasts (>7 days) are not backtested**; the API omits the interval for them.
* Only ~2 years of history — yearly seasonality is weakly identified; test period is 6 weeks.
* `online`, `markdowns`, `price_history`, `actual_matrix` are not modelled (reasons in the analysis doc); stock-outs and inventory are unknown.
* Quantities mix units and kilograms (weighted goods); revenue has no currency in the source data.
* Hyper-parameters were set by hand and early stopping, not tuned; no authentication (single-tenant demo).
* Schema is created with `create_all`; there are no Alembic migrations.

## Future improvements

Hierarchical/global models for intermittent items, proper hyper-parameter search, conformal prediction
intervals, backtesting of the recursive horizon, exogenous holiday features, Alembic migrations, auth +
rate limiting, scheduled retraining with drift monitoring.
