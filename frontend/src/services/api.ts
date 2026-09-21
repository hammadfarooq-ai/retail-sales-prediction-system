import axios, { AxiosError } from 'axios'
import type {
  BacktestResponse,
  CategorySales,
  Filters,
  ForecastResponse,
  Granularity,
  Health,
  ModelInfo,
  ModelPerformance,
  PredictRequest,
  PredictResponse,
  PredictionPage,
  PredictionRecord,
  Product,
  ProductSalesList,
  SalesSummary,
  SalesTrends,
  StoreSales,
} from '../types/api'

const baseURL = import.meta.env.VITE_API_URL ?? ''

export const http = axios.create({ baseURL, timeout: 60_000 })

/** Error with a human-readable message extracted from the backend's error envelope. */
export class ApiError extends Error {
  status?: number
  code?: string
  details?: unknown
  constructor(message: string, status?: number, code?: string, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err
  if (axios.isAxiosError(err)) {
    const e = err as AxiosError<{ error?: { code?: string; message?: string; details?: unknown } }>
    if (!e.response) {
      return new ApiError('Cannot reach the server. Check that the backend is running.', undefined, 'network_error')
    }
    const body = e.response.data?.error
    let message = body?.message ?? `Request failed (${e.response.status})`
    if (Array.isArray(body?.details) && body.code === 'validation_error') {
      const first = (body.details as { field: string; message: string }[])[0]
      if (first) message = `${first.field}: ${first.message}`
    }
    return new ApiError(message, e.response.status, body?.code, body?.details)
  }
  return new ApiError(err instanceof Error ? err.message : 'Unexpected error')
}

type Params = Record<string, string | number | boolean | null | undefined>

function clean(params?: Params): Params | undefined {
  if (!params) return undefined
  return Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''))
}

async function get<T>(url: string, params?: Params, signal?: AbortSignal): Promise<T> {
  try {
    const res = await http.get<T>(url, { params: clean(params), signal })
    return res.data
  } catch (err) {
    throw toApiError(err)
  }
}

async function post<T>(url: string, body: unknown): Promise<T> {
  try {
    const res = await http.post<T>(url, body)
    return res.data
  } catch (err) {
    throw toApiError(err)
  }
}

export interface RangeFilter {
  start_date?: string
  end_date?: string
  store_id?: number | null
  dept_name?: string | null
}

export const api = {
  health: () => get<Health>('/health'),
  filters: () => get<Filters>('/api/v1/meta/filters'),
  summary: (f: RangeFilter) => get<SalesSummary>('/api/v1/sales/summary', { ...f }),
  trends: (f: RangeFilter & { granularity: Granularity; group_by?: 'store' }) =>
    get<SalesTrends>('/api/v1/sales/trends', { ...f }),
  byStore: (f: Pick<RangeFilter, 'start_date' | 'end_date'>) => get<StoreSales[]>('/api/v1/sales/by-store', { ...f }),
  byCategory: (f: RangeFilter & { limit?: number }) => get<CategorySales[]>('/api/v1/sales/by-category', { ...f }),
  byProduct: (p: {
    store_id?: number | null
    dept_name?: string | null
    search?: string
    sort_by?: 'revenue' | 'quantity'
    forecastable_only?: boolean
    limit?: number
    offset?: number
  }) => get<ProductSalesList>('/api/v1/sales/by-product', { ...p }),
  products: (p: { q?: string; store_id?: number | null; dept_name?: string | null; forecastable_only?: boolean; limit?: number }) =>
    get<Product[]>('/api/v1/products', { ...p }),
  modelInfo: () => get<ModelInfo>('/api/v1/model/info'),
  modelPerformance: () => get<ModelPerformance>('/api/v1/model/performance'),
  backtest: (p: { store_id?: number | null; item_id?: string | null }) =>
    get<BacktestResponse>('/api/v1/model/backtest', { ...p }),
  predict: (body: PredictRequest) => post<PredictResponse>('/api/v1/predict', body),
  forecast: (p: { store_id: number; item_id?: string | null; horizon_days: number; history_days?: number }) =>
    get<ForecastResponse>('/api/v1/forecasts', { ...p }),
  predictions: (p: {
    store_id?: number | null
    item_id?: string
    target_from?: string
    target_to?: string
    created_from?: string
    created_to?: string
    limit?: number
    offset?: number
  }) => get<PredictionPage>('/api/v1/predictions', { ...p }),
  latestPrediction: () => get<PredictionRecord | null>('/api/v1/predictions/latest'),
}
