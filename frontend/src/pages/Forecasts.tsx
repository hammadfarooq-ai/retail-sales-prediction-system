import { useMemo, useState } from 'react'
import { api, toApiError } from '../services/api'
import { useFilters } from '../hooks/FiltersContext'
import { useToast } from '../components/Toast'
import { Badge, Card, ChartSkeleton, EmptyState, ErrorState, Field, PageHeader, COLOR_ACTUAL, COLOR_PRED } from '../components/ui'
import { ProductPicker } from '../components/ProductPicker'
import { TimeSeriesChart } from '../components/charts'
import { downloadCsv, toCsv } from '../utils/csv'
import { fmtDate, fmtDec, fmtShortDate } from '../utils/format'
import type { ForecastResponse, Product } from '../types/api'

const HORIZONS = [7, 14, 21, 28]

export default function Forecasts() {
  const filters = useFilters()
  const toast = useToast()
  const [scope, setScope] = useState<'item' | 'store'>('item')
  const [storeId, setStoreId] = useState(filters.stores[0].store_id)
  const [product, setProduct] = useState<Product | null>(null)
  const [horizon, setHorizon] = useState(14)
  const [historyDays, setHistoryDays] = useState(60)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<ForecastResponse | null>(null)
  const [formError, setFormError] = useState<string | null>(null)

  async function generate() {
    setFormError(null)
    if (scope === 'item' && !product) {
      setFormError('Select a product to forecast')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await api.forecast({ store_id: storeId, item_id: scope === 'item' ? product!.item_id : null, horizon_days: horizon, history_days: historyDays })
      setData(res)
      toast.success('Forecast generated')
    } catch (e) {
      const msg = toApiError(e).message
      setError(msg)
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  const rows = useMemo(() => {
    if (!data) return []
    const last = data.history[data.history.length - 1]
    return [
      ...data.history.map((h) => ({ date: h.date, actual: h.actual, predicted: null as number | null, lower: null as number | null, upper: null as number | null })),
      // connect the forecast line to the last observed point
      ...(last ? [] : []),
      ...data.forecast.map((p) => ({ date: p.date, actual: null as number | null, predicted: p.predicted, lower: p.lower, upper: p.upper })),
    ]
  }, [data])

  function exportCsv() {
    if (!data) return
    const csv = toCsv(
      [
        ...data.history.map((h) => ({ date: h.date, type: 'actual', quantity: h.actual, lower: '', upper: '', recursive: '' })),
        ...data.forecast.map((p) => ({ date: p.date, type: 'forecast', quantity: p.predicted, lower: p.lower ?? '', upper: p.upper ?? '', recursive: p.is_recursive })),
      ],
      ['date', 'type', 'quantity', 'lower', 'upper', 'recursive'],
    )
    downloadCsv(`forecast_store${data.store_id}_${data.item_id ?? 'store-total'}.csv`, csv)
  }

  const hasBand = data?.forecast.some((p) => p.lower != null)
  return (
    <>
      <PageHeader title="Forecasts" description="Generate a multi-day forecast for a product or for a store's forecastable products, alongside recent history." />
      <div className="grid gap-6 lg:grid-cols-4">
        <Card title="Options" className="lg:col-span-1">
          <div className="space-y-4">
            <Field label="Scope" htmlFor="fc-scope">
              <select id="fc-scope" className="input" value={scope} onChange={(e) => setScope(e.target.value as 'item' | 'store')}>
                <option value="item">Single product</option>
                <option value="store">Store total</option>
              </select>
            </Field>
            <Field label="Store" htmlFor="fc-store">
              <select
                id="fc-store"
                className="input"
                value={storeId}
                onChange={(e) => {
                  setStoreId(Number(e.target.value))
                  setProduct(null)
                }}
              >
                {filters.stores.map((s) => (
                  <option key={s.store_id} value={s.store_id}>
                    Store {s.store_id} · {s.format}
                  </option>
                ))}
              </select>
            </Field>
            {scope === 'item' && (
              <Field label="Product" htmlFor="fc-product" error={formError}>
                <ProductPicker id="fc-product" storeId={storeId} value={product} onChange={setProduct} error={formError} required />
              </Field>
            )}
            <Field label="Forecast period" htmlFor="fc-h">
              <select id="fc-h" className="input" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
                {HORIZONS.filter((h) => h <= filters.max_forecast_horizon_days).map((h) => (
                  <option key={h} value={h}>
                    Next {h} days
                  </option>
                ))}
              </select>
            </Field>
            <Field label="History shown" htmlFor="fc-hist">
              <select id="fc-hist" className="input" value={historyDays} onChange={(e) => setHistoryDays(Number(e.target.value))}>
                {[30, 60, 90, 180].map((d) => (
                  <option key={d} value={d}>
                    Last {d} days
                  </option>
                ))}
              </select>
            </Field>
            <button type="button" className="btn-primary w-full" onClick={generate} disabled={loading}>
              {loading ? 'Generating…' : 'Generate forecast'}
            </button>
          </div>
        </Card>

        <Card
          className="lg:col-span-3"
          title={data ? (data.scope === 'item' ? `Product ${product?.subclass_name ?? data.item_id} · Store ${data.store_id}` : `Store ${data.store_id} total (${data.n_series} products)`) : 'Forecast'}
          subtitle={data ? `Last observed day ${fmtDate(data.last_observed_date)} · model ${data.model_version}` : undefined}
          actions={
            data && (
              <button type="button" className="btn-secondary" onClick={exportCsv}>
                Export CSV
              </button>
            )
          }
        >
          {loading ? (
            <ChartSkeleton height={380} />
          ) : error ? (
            <ErrorState message={error} onRetry={generate} />
          ) : !data ? (
            <EmptyState title="No forecast generated" description="Choose options on the left and press “Generate forecast”." />
          ) : (
            <>
              <TimeSeriesChart
                data={rows}
                height={380}
                xFormatter={fmtShortDate}
                splitAt={data.last_observed_date}
                splitLabel="forecast →"
                band={hasBand ? { lower: 'lower', upper: 'upper', label: `${Math.round((data.interval_level ?? 0.8) * 100)}% interval (days 1–7)`, color: COLOR_PRED } : undefined}
                series={[
                  { key: 'actual', label: 'Historical sales', color: COLOR_ACTUAL },
                  { key: 'predicted', label: 'Forecast', color: COLOR_PRED, dashed: true },
                ]}
              />
              <ul className="mt-3 space-y-1 text-xs text-slate-500">
                {data.notes.map((n) => (
                  <li key={n}>• {n}</li>
                ))}
              </ul>
              <div className="mt-4 flex flex-wrap gap-2">
                <Badge tone="blue">Total forecast: {fmtDec(data.forecast.reduce((a, p) => a + p.predicted, 0), 1)} units</Badge>
                <Badge>Days 1–7 direct · {data.forecast.filter((p) => p.is_recursive).length} recursive days</Badge>
              </div>
            </>
          )}
        </Card>
      </div>
    </>
  )
}
