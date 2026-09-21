# Data

The raw Kaggle files live **directly in this folder** (they were downloaded before the project was
built and are left untouched — nothing in the pipeline writes to or modifies them).

Source: <https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data>

They are ~850 MB in total and are **git-ignored**. To reproduce the project on a fresh clone, download
the dataset from Kaggle (for example `kaggle datasets download -d svizor/retail-sales-forecasting-data`)
and unzip the eight CSVs into this folder:

```
data/
├── actual_matrix.csv       1.1 MB      35,202 rows
├── catalog.csv            25.3 MB     219,810 rows
├── discounts_history.csv 341.5 MB   3,746,744 rows
├── markdowns.csv           0.4 MB       8,979 rows
├── online.csv             56.8 MB   1,123,412 rows
├── price_history.csv      29.3 MB     698,626 rows
├── sales.csv             379.2 MB   7,432,685 rows
└── stores.csv                158 B           4 rows
```

`data/processed/` is created by `make data` (`ml/scripts/01_prepare_data.py`) and holds derived
Parquet tables (zero-filled modelling panel, aggregates, backtest predictions). It is also git-ignored
and fully reproducible from the raw CSVs.

A machine-readable profile of every file (dtypes, nulls, duplicates, cardinality, ranges) is generated
into [`docs/data_profile.json`](../docs/data_profile.json) by `make eda`. The written analysis is in
[`docs/ANALYSIS_AND_DESIGN.md`](../docs/ANALYSIS_AND_DESIGN.md).
