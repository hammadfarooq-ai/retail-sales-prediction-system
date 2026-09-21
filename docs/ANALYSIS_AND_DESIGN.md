# Data inspection, problem definition and design decisions

Everything here was derived from the **actual local files** in `./data/` (see
[`data_profile.json`](data_profile.json) and [`figures/eda/`](figures/eda) for the machine-readable
profile and the charts). No value in this document comes from outside the dataset or from a model run
that was not executed.

## 1. What is in `./data/`

| File | Rows | Grain / meaning | Notes |
|---|---:|---|---|
| `sales.csv` | 7,432,685 | one row per **(date, item_id, store_id)** with a sale: `quantity`, `price_base`, `sum_total` | 2022-08-28 → 2024-09-26 (761 days, **no missing calendar day**). No duplicate keys. Only *positive-sale* rows exist – days with no sale are simply absent. |
| `stores.csv` | 4 | `store_id`, `division`, `format`, `city`, `area` | 4 stores, 4 formats, 3 cities, area 109–1887 m². |
| `catalog.csv` | 219,810 | item hierarchy: `dept_name` (196) > `class_name` (613) > `subclass_name` (1,007), `item_type`, weights | Names are Russian (Cyrillic). `item_type` 80% null, `weight_volume` 62%, `weight_netto` 78%, `fatness` 97% null. Covers 27,234 of the 28,182 sold items. |
| `discounts_history.csv` | 3,746,744 | promo calendar per (date, item, store): regular price, promo price, `promo_type_code` | **Contains future-dated promotions** (dates run to **2045-12-31**) – i.e. a planned-promo calendar. 8.5% null `promo_type_code`. |
| `price_history.csv` | 698,626 | sparse price events | 82,320 duplicate (date, item, store) keys (18,641 fully duplicated rows), 8.8% of prices are `0`, max price 1.2·10⁸ (corrupt), unclear `code` field → **not used**. |
| `online.csv` | 1,123,412 | online-channel sales (stores 1 and 4 only) | 906,603 of its keys also appear in `sales.csv`, so it is (partly) already inside `sales` → **not added** (would double count). |
| `markdowns.csv` | 8,979 | clearance markdowns (313 items) | 268 duplicate rows; tiny → **not used**. |
| `actual_matrix.csv` | 35,202 | (item, store, date) assortment snapshots since 2019 | Semantics undocumented (only 75 rows on the last day) → **not used**. |

Relationships: everything joins on `item_id` (12-char hash) and `store_id`; `date` links the time-series
tables. `catalog` is the dimension for `sales` (`item_id`), `stores` for `store_id`.

### Data-quality findings that shaped the design

* `price_base` **is the realised net unit price**: `sum_total / quantity == price_base` (ratio 1.000,
  p1–p99). It therefore already contains promotional discounts.
* 3,598 rows have `quantity <= 0` (returns) and 5,606 more have non-positive price/total; they are
  removed (0.12% of rows) – see `clean_sales`.
* 11% of quantities are fractional (weighted goods, kg). The target is *quantity in native units*.
* `quantity` is heavy-tailed (max 4,952 in one row; 5.9% of universe sale-days are above Q3 + 3·IQR).
* **Store 4 opened on 2023-12-13** – it has only 289 selling days; lifetime totals are not comparable.
* Strong weekly seasonality (Friday index 1.25 vs Sunday 0.82) and a December peak (figures 03, 10).
* Demand is **sparse at item level**: most of the 58,000 store×item pairs sell on few days; the top 10%
  of pairs generate 74% of revenue (figure 06).
* Promotions matter: for pairs with both promo and non-promo days, the median promo/non-promo mean-
  quantity ratio is **1.48** (87.8% of 4,502 pairs > 1) – an uncontrolled comparison, but a clear signal.

## 2. The ML problem

| Decision | Choice | Why |
|---|---|---|
| Target | `quantity` (units sold) | Directly what is observed; revenue = quantity × price is derivable. |
| Prediction unit | **store × product × day** | The finest grain the data supports and the one a replenishment planner needs; the UI asks for store + product + date. |
| Universe | 6,418 (store, product) pairs with *regular demand*: sold on ≥ 70% of days since first sale **and** ≥ 90 sale days, **judged on the training period only** | Daily forecasting of items that sell a few times a year is not meaningful, and zero-filling 58,000 pairs × 761 days (>40 M rows) does not fit an 8 GB machine. The universe covers **62.0%** of clean revenue and 3.23 M panel rows. |
| Zero handling | Panel is **zero-filled** from each pair's first sale to the end | Absent rows in `sales.csv` mean "no sale". Zero-filling is only valid inside the item's active window, which the density rule guarantees. |
| Horizon | **7 days**, direct strategy | Every history feature is lagged ≥ 7 days, so one model serves any forecast origin ≥ 7 days earlier. |
| Longer forecasts (8–28 days) | **Recursive** in 7-day blocks (predictions fed back as lags) | Documented as *not backtested*; the API flags it (`is_recursive`) and drops the interval. |
| Split | Chronological: **train ≤ 2024-07-04 · validation 2024-07-05 → 2024-08-15 · test 2024-08-16 → 2024-09-26** (42 days each for val/test) | Derived from the last observed date, never random. |
| Selection | Best **validation MAE** among ML models; test never used for selection | MAE is robust to the heavy tail and is in business units. |

### Leakage analysis

| Risk | Handling |
|---|---|
| Lags / rolling stats using the target day or the blind window | All shifted ≥ 7 days **within each series**; a unit test perturbs every value in the blind window and asserts identical features (`tests/test_features.py`). |
| Realised price of the target day (`price_base`) | **Never a feature** – it exists only if a sale happened, which would leak the target. Price enters as the *last observed price at the origin* plus the *planned* promo price. |
| Promotion calendar | Used as a **known-in-advance** covariate. Assumption: promotions dated for day *t* were scheduled before *t-7*. If some were recorded retroactively, results would be optimistic. |
| Universe selection | Computed from training-period data only. |
| Random splits / shuffling | None; a test asserts split boundaries do not overlap. |
| Cold start | Rows with < 35 days of history are dropped (`MIN_HISTORY_DAYS`). |
| Duplicate/second channel | `online.csv` not added, to avoid double counting. |

## 3. Features (33)

* **Calendar (7):** day_of_week, day_of_month, month, week_of_year, quarter, is_weekend, day_of_year.
* **Lags (4):** lag_7, 14, 21, 28.
* **Rolling (6):** mean over 7/14/28/56 days, std over 7/28 – all on the series shifted by 7.
* **History extras (4):** mean of the four same-weekday lags, 7d/28d mean ratio, share of days with a sale (28d), days since last sale (capped at 90).
* **Price & promotion (7):** last observed price, planned price (promo price or last price), planned/last ratio, promo flag, discount depth, promo type code, promo-days in the last 28d.
* **Static (5):** store_id, store area, store format, department code, class code.

The training and serving code path is **one function** (`ml/src/features/engineering.py`); a test proves
the vectorised implementation is numerically identical to a slow per-series pandas reference.

## 4. Implementation plan (as executed)

1. Inspect `./data/` and profile every file (`make eda` → `docs/data_profile.json`).
2. Clean, aggregate and build the zero-filled panel (`make data`).
3. Feature engineering with leakage tests.
4. Train baselines + Ridge + Random Forest + XGBoost + LightGBM (+ Tweedie objective); compare on
   validation and test; refit the winner on train+val; 10%/90% quantile models for an interval.
5. Persist artifacts; build the FastAPI service around a once-loaded `Forecaster`.
6. PostgreSQL schema of *derived* tables only (raw 379 MB file is not duplicated); seed with `COPY`.
7. React dashboard against real endpoints; Docker Compose; tests, linting, type-checking.
