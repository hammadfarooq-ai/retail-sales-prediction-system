// Types mirror the FastAPI schemas (backend/app/schemas).

export interface Store {
  store_id: number
  division: string
  format: string
  city: string
  area: number
}

export interface Filters {
  stores: Store[]
  departments: string[]
  min_date: string
  max_date: string
  max_forecast_horizon_days: number
  forecastable_pairs: number
}

export interface Product {
  item_id: string
  dept_name: string
  class_name: string
  subclass_name: string
  item_type: string | null
  forecastable_stores: number[]
}

export interface SalesSummary {
  start_date: string
  end_date: string
  n_days: number
  total_quantity: number
  total_revenue: number
  avg_daily_quantity: number
  avg_daily_revenue: number
  n_stores: number
  n_products: number
  n_departments: number
  filters: { store_id: number | null; dept_name: string | null }
}

export type Granularity = 'day' | 'week' | 'month'

export interface TrendPoint {
  period: string
  quantity: number
  revenue: number
  store_id: number | null
}

export interface SalesTrends {
  granularity: Granularity
  group_by: string | null
  points: TrendPoint[]
}

export interface StoreSales {
  store_id: number
  format: string
  city: string
  division: string
  area: number
  quantity: number
  revenue: number
  active_days: number
  avg_daily_revenue: number
  avg_daily_revenue_per_m2: number
  n_products: number
}

export interface ProductSales {
  store_id: number
  item_id: string
  dept_name: string
  class_name: string
  subclass_name: string
  quantity: number
  revenue: number
  sale_days: number
  avg_price: number
  first_sale: string
  last_sale: string
  forecastable: boolean
}

export interface ProductSalesList {
  period: string
  total: number
  items: ProductSales[]
}

export interface CategorySales {
  dept_name: string
  quantity: number
  revenue: number
  revenue_share_pct: number
  n_products: number
}

export interface PredictRequest {
  store_id: number
  item_id: string
  date: string
  price?: number | null
  promotion?: boolean | null
  discount_pct?: number | null
}

export interface EffectiveInputs {
  price: number | null
  last_observed_price: number | null
  promotion: boolean
  discount_pct: number
  promotion_source: 'user' | 'promo_calendar' | 'none'
}

export interface PredictResponse {
  id: number | null
  model_version: string
  store_id: number
  item_id: string
  dept_name: string | null
  subclass_name: string | null
  date: string
  predicted_quantity: number
  lower: number | null
  upper: number | null
  interval_level: number | null
  mode: 'in_sample' | 'held_out_test' | 'forecast'
  horizon_days: number
  is_recursive: boolean
  actual_quantity: number | null
  inputs: EffectiveInputs
  created_at: string | null
}

export interface PredictionRecord {
  id: number
  created_at: string
  model_version: string
  source: string
  batch_id: string | null
  store_id: number
  item_id: string
  target_date: string
  inputs: {
    request?: PredictRequest
    effective?: EffectiveInputs
    mode?: string
    horizon_days?: number
  }
  predicted_quantity: number
  lower: number | null
  upper: number | null
  actual_quantity: number | null
  is_recursive: boolean
}

export interface PredictionPage {
  total: number
  limit: number
  offset: number
  items: PredictionRecord[]
}

export interface ForecastPoint {
  date: string
  predicted: number
  lower: number | null
  upper: number | null
  is_recursive: boolean
}

export interface HistoryPoint {
  date: string
  actual: number
}

export interface ForecastResponse {
  scope: 'item' | 'store'
  store_id: number
  item_id: string | null
  model_version: string
  last_observed_date: string
  horizon_days: number
  n_series: number
  history: HistoryPoint[]
  forecast: ForecastPoint[]
  interval_level: number | null
  notes: string[]
}

export interface MetricSet {
  mae: number
  rmse: number
  mape: number
  smape: number
  wape: number
  r2: number
  bias: number
  n_rows: number
  mape_coverage_pct: number
}

export interface ComparisonRow {
  description: string
  is_baseline: boolean
  validation: MetricSet
  test: MetricSet
  fit_seconds: number
  best_iteration: number | null
  params: Record<string, unknown>
}

export interface ModelInfo {
  version: string
  model_name: string
  trained_at: string
  n_features: number
  feature_names: string[]
  horizon_days: number
  target: string
  split: Record<string, string>
  n_series: number | null
  n_train_rows: number | null
  has_prediction_interval: boolean
  prediction_interval: { level: number; empirical_test_coverage_pct: number } | null
  test_metrics: MetricSet
  last_observed_date: string | null
}

export interface ModelPerformance {
  version: string
  model_name: string
  trained_at: string
  selection_metric: string
  split: Record<string, string>
  n_features: number
  comparison: Record<string, ComparisonRow>
  final_model_test: MetricSet
  final_model_test_by_store: Record<string, MetricSet>
  final_model_test_aggregated: Record<string, MetricSet>
  improvement_vs_lag7_baseline_pct: { mae: number; rmse: number }
  feature_importance: Record<string, number>
  prediction_interval: { level: number; empirical_test_coverage_pct: number }
}

export interface BacktestPoint {
  date: string
  actual: number
  predicted: number
  baseline_lag7: number | null
}

export interface BacktestResponse {
  model_version: string
  scope: string
  points: BacktestPoint[]
  error_histogram: { lower: number; upper: number; count: number }[]
  residual_summary: Record<string, number>
  scatter_sample: { actual: number; predicted: number }[]
}

export interface Health {
  status: 'ok' | 'degraded'
  version: string
  database: 'up' | 'down'
  model_loaded: boolean
  model_version: string | null
  time: string
}
