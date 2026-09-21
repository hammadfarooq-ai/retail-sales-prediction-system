import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { Card, ChartSkeleton, EmptyState, ErrorState, PageHeader, SERIES, StatCard, COLOR_ACTUAL, COLOR_PRED, COLOR_BASE } from '../components/ui'
import { BarsChart, TimeSeriesChart } from '../components/charts'
import { FilterBar, defaultFilterState, type FilterState } from '../components/FilterBar'
import { fmtCompact, fmtDate, fmtDec, fmtInt, fmtMonth, fmtPct, fmtShortDate } from '../utils/format'
import type { StoreSales } from '../types/api'

function useRange(f: FilterState) {
  return { start_date: f.start_date, end_date: f.end_date, store_id: f.store_id, dept_name: f.dept_name }
}

export default function Dashboard() {
  const filters = useFilters()
  const [f, setF] = useState<FilterState>(() => defaultFilterState(filters))
  const range = useRange(f)
  const key = [f.store_id, f.dept_name, f.start_date, f.end_date]

  const summary = useApi(() => api.summary(range), key)
  const daily = useApi(() => api.trends({ ...range, granularity: 'day' }), key)
  const monthly = useApi(() => api.trends({ ...range, granularity: 'month', group_by: 'store' }), key)
  const stores = useApi(() => api.byStore({ start_date: f.start_date, end_date: f.end_date }), [f.start_date, f.end_date])
  const products = useApi(() => api.byProduct({ store_id: f.store_id, dept_name: f.dept_name, sort_by: 'revenue', limit: 10 }), [f.store_id, f.dept_name])
  const perf = useApi(() => api.modelInfo(), [])
  const latest = useApi(() => api.latestPrediction(), [])
  const backtest = useApi(() => api.backtest({ store_id: f.store_id }), [f.store_id])
  const storeForecast = useApi(() => api.forecast({ store_id: f.store_id ?? filters.stores[0].store_id, horizon_days: 14, history_days: 42 }), [f.store_id])

  const dailyRows = useMemo(() => (daily.data?.points ?? []).map((p) => ({ date: p.period, quantity: p.quantity, revenue: p.revenue })), [daily.data])
  const monthlyRows = useMemo(() => {
    const byMonth = new Map<string, Record<string, string | number>>()
    for (const p of monthly.data?.points ?? []) {
      const row = byMonth.get(p.period) ?? { month: p.period }
      row[`s${p.store_id}`] = p.revenue
      byMonth.set(p.period, row)
    }
    return [...byMonth.values()]
  }, [monthly.data])
  const storeIds = useMemo(() => [...new Set((monthly.data?.points ?? []).map((p) => p.store_id!))].sort(), [monthly.data])
  const storeColor = (id: number) => SERIES[(filters.stores.findIndex((s) => s.store_id === id) + SERIES.length) % SERIES.length]

  const s = summary.data
  const m = perf.data?.test_metrics
  const lp = latest.data

  return (
    <>
      <PageHeader title="Dashboard" description="Live overview computed from the sales data in PostgreSQL and the trained forecasting model." />
      <FilterBar filters={filters} value={f} onChange={setF} />
      {summary.error && <div className="mb-4"><ErrorState message={summary.error} onRetry={summary.refetch} /></div>}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Total sales (units)" value={s ? fmtCompact(s.total_quantity) : ''} loading={summary.loading} hint={s ? `${fmtDate(s.start_date)} – ${fmtDate(s.end_date)}` : undefined} />
        <StatCard label="Total revenue" value={s ? fmtCompact(s.total_revenue) : ''} loading={summary.loading} hint="Sum of receipt totals" />
        <StatCard label="Avg daily sales (units)" value={s ? fmtCompact(s.avg_daily_quantity) : ''} loading={summary.loading} hint={s ? `${fmtInt(s.n_days)} trading days` : undefined} />
        <StatCard label="Stores · Products" value={s ? `${s.n_stores} · ${fmtInt(s.n_products)}` : ''} loading={summary.loading} hint={s ? `${s.n_departments} categories` : undefined} />
        <StatCard
          label="Latest prediction"
          loading={latest.loading}
          value={lp ? `${fmtDec(lp.predicted_quantity, 1)} units` : '—'}
          hint={lp ? `Store ${lp.store_id} · ${fmtDate(lp.target_date)}` : <Link className="text-brand-600 underline" to="/predict">Make your first prediction</Link>}
        />
        <StatCard label="Model MAE (test)" loading={perf.loading} value={m ? `${fmtDec(m.mae, 2)} units` : '—'} hint="Mean abs. error per product-store-day" />
        <StatCard label="Model WAPE (test)" loading={perf.loading} value={m ? fmtPct(m.wape) : '—'} hint="Σ|error| / Σ actual" />
        <StatCard label="Model R² (test)" loading={perf.loading} value={m ? fmtDec(m.r2, 3) : '—'} hint={perf.data ? `v: ${perf.data.model_name}` : undefined} />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Sales over time" subtitle="Daily units sold (filters applied)">
          {daily.loading ? <ChartSkeleton /> : daily.error ? <ErrorState message={daily.error} onRetry={daily.refetch} /> : dailyRows.length === 0 ? <EmptyState title="No sales in this selection" /> : (
            <TimeSeriesChart data={dailyRows} series={[{ key: 'quantity', label: 'Units sold', color: COLOR_ACTUAL }]} />
          )}
        </Card>
        <Card title="Monthly revenue by store" subtitle="First and last months of the dataset are partial">
          {monthly.loading ? <ChartSkeleton /> : monthly.error ? <ErrorState message={monthly.error} onRetry={monthly.refetch} /> : monthlyRows.length === 0 ? <EmptyState title="No sales in this selection" /> : (
            <BarsChart data={monthlyRows} categoryKey="month" categoryFormatter={fmtMonth} stacked bars={storeIds.map((id) => ({ key: `s${id}`, label: `Store ${id}`, color: storeColor(id) }))} />
          )}
        </Card>
        <Card title="Store performance" subtitle="Revenue in the selected period">
          {stores.loading ? <ChartSkeleton /> : stores.error ? <ErrorState message={stores.error} onRetry={stores.refetch} /> : (
            <BarsChart data={(stores.data ?? []).map((x: StoreSales) => ({ store: `Store ${x.store_id}`, revenue: x.revenue }))} categoryKey="store" bars={[{ key: 'revenue', label: 'Revenue', color: COLOR_ACTUAL }]} />
          )}
        </Card>
        <Card title="Top products by revenue" subtitle="Lifetime totals for the selected store / category">
          {products.loading ? <ChartSkeleton /> : products.error ? <ErrorState message={products.error} onRetry={products.refetch} /> : (products.data?.items.length ?? 0) === 0 ? <EmptyState title="No products match" /> : (
            <BarsChart layout="horizontal" categoryWidth={190} data={products.data!.items.map((p) => ({ name: `${p.subclass_name.slice(0, 16)} · ${p.item_id.slice(0, 4)} (S${p.store_id})`, revenue: p.revenue }))} categoryKey="name" bars={[{ key: 'revenue', label: 'Revenue', color: SERIES[2] }]} />
          )}
        </Card>
        <Card title="Actual vs predicted (held-out test period)" subtitle={f.store_id ? `Store ${f.store_id}` : 'All stores'} >
          {backtest.loading ? <ChartSkeleton /> : backtest.error ? <ErrorState message={backtest.error} onRetry={backtest.refetch} /> : (backtest.data?.points.length ?? 0) === 0 ? <EmptyState title="No backtest rows" /> : (
            <TimeSeriesChart
              data={backtest.data!.points.map((p) => ({ date: p.date, actual: p.actual, predicted: p.predicted, baseline: p.baseline_lag7 }))}
              xFormatter={fmtShortDate}
              series={[
                { key: 'actual', label: 'Actual', color: COLOR_ACTUAL },
                { key: 'predicted', label: 'Model', color: COLOR_PRED },
                { key: 'baseline', label: 'Naive (same weekday last week)', color: COLOR_BASE, dashed: true, width: 1.5 },
              ]}
            />
          )}
        </Card>
        <Card title="Forecast trend" subtitle={`Next 14 days for store ${f.store_id ?? filters.stores[0].store_id} (forecastable products only)`} actions={<Link className="text-sm text-brand-600 hover:underline" to="/forecasts">Open forecasts →</Link>}>
          {storeForecast.loading ? <ChartSkeleton /> : storeForecast.error ? <ErrorState message={storeForecast.error} onRetry={storeForecast.refetch} /> : (
            <TimeSeriesChart
              data={[
                ...storeForecast.data!.history.map((h) => ({ date: h.date, actual: h.actual, predicted: null as number | null })),
                ...storeForecast.data!.forecast.map((p) => ({ date: p.date, actual: null as number | null, predicted: p.predicted })),
              ]}
              xFormatter={fmtShortDate}
              splitAt={storeForecast.data!.last_observed_date}
              splitLabel="last observed"
              series={[
                { key: 'actual', label: 'Historical', color: COLOR_ACTUAL },
                { key: 'predicted', label: 'Forecast', color: COLOR_PRED, dashed: true },
              ]}
            />
          )}
        </Card>
      </div>
    </>
  )
}
