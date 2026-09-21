import { useMemo, useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { Card, ChartSkeleton, EmptyState, ErrorState, PageHeader, StatCard, Table, COLOR_ACTUAL, COLOR_PRED, SERIES } from '../components/ui'
import { BarsChart, TimeSeriesChart } from '../components/charts'
import { FilterBar, defaultFilterState, type FilterState } from '../components/FilterBar'
import { fmtCompact, fmtDate, fmtDec, fmtInt, fmtMonth, fmtPct, fmtShortDate } from '../utils/format'
import type { Granularity } from '../types/api'

export default function SalesAnalytics() {
  const filters = useFilters()
  const [f, setF] = useState<FilterState>(() => defaultFilterState(filters))
  const [gran, setGran] = useState<Granularity>('week')
  const [metric, setMetric] = useState<'quantity' | 'revenue'>('revenue')
  const range = { start_date: f.start_date, end_date: f.end_date, store_id: f.store_id, dept_name: f.dept_name }
  const key = [f.store_id, f.dept_name, f.start_date, f.end_date]

  const summary = useApi(() => api.summary(range), key)
  const trend = useApi(() => api.trends({ ...range, granularity: gran, group_by: f.store_id ? undefined : 'store' }), [...key, gran])
  const cats = useApi(() => api.byCategory({ ...range, limit: 15 }), [f.store_id, f.start_date, f.end_date])

  const xfmt = gran === 'month' ? fmtMonth : fmtShortDate
  const { rows, storeIds } = useMemo(() => {
    const by = new Map<string, Record<string, string | number>>()
    const ids = new Set<number>()
    for (const p of trend.data?.points ?? []) {
      const r = by.get(p.period) ?? { date: p.period }
      const k = p.store_id ? `s${p.store_id}` : 'total'
      r[k] = p[metric]
      if (p.store_id) ids.add(p.store_id)
      by.set(p.period, r)
    }
    return { rows: [...by.values()], storeIds: [...ids].sort() }
  }, [trend.data, metric])
  const colorFor = (id: number) => SERIES[filters.stores.findIndex((s) => s.store_id === id) % SERIES.length]
  const series = storeIds.length
    ? storeIds.map((id) => ({ key: `s${id}`, label: `Store ${id}`, color: colorFor(id) }))
    : [{ key: 'total', label: metric === 'revenue' ? 'Revenue' : 'Units sold', color: metric === 'revenue' ? COLOR_ACTUAL : COLOR_PRED }]
  const s = summary.data

  return (
    <>
      <PageHeader title="Sales Analytics" description="Explore sales trends and category mix. Category filters apply at department level (the dataset's top-level product hierarchy)." />
      <FilterBar filters={filters} value={f} onChange={setF} />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Revenue" value={s ? fmtCompact(s.total_revenue) : ''} loading={summary.loading} />
        <StatCard label="Units sold" value={s ? fmtCompact(s.total_quantity) : ''} loading={summary.loading} />
        <StatCard label="Avg daily revenue" value={s ? fmtCompact(s.avg_daily_revenue) : ''} loading={summary.loading} />
        <StatCard label="Avg revenue / unit" value={s && s.total_quantity ? fmtDec(s.total_revenue / s.total_quantity, 2) : '–'} loading={summary.loading} />
      </div>

      <Card
        className="mt-6"
        title="Sales trend"
        subtitle={f.store_id ? `Store ${f.store_id}` : 'Split by store'}
        actions={
          <div className="flex flex-wrap gap-2">
            <select aria-label="Granularity" className="input !w-auto" value={gran} onChange={(e) => setGran(e.target.value as Granularity)}>
              <option value="day">Daily</option>
              <option value="week">Weekly</option>
              <option value="month">Monthly</option>
            </select>
            <select aria-label="Metric" className="input !w-auto" value={metric} onChange={(e) => setMetric(e.target.value as 'quantity' | 'revenue')}>
              <option value="revenue">Revenue</option>
              <option value="quantity">Units</option>
            </select>
          </div>
        }
      >
        {trend.loading ? <ChartSkeleton height={340} /> : trend.error ? <ErrorState message={trend.error} onRetry={trend.refetch} /> : rows.length === 0 ? <EmptyState title="No sales in this selection" description="Try widening the date range or clearing filters." /> : (
          <TimeSeriesChart data={rows} series={series} xFormatter={xfmt} height={340} />
        )}
        {gran !== 'day' && <p className="mt-2 text-xs text-slate-500">Partial first/last {gran}s can look lower than full periods.</p>}
      </Card>

      <Card className="mt-6" title="Category performance" subtitle="Top 15 departments by revenue in the selected period">
        {cats.loading ? <ChartSkeleton height={380} /> : cats.error ? <ErrorState message={cats.error} onRetry={cats.refetch} /> : (cats.data?.length ?? 0) === 0 ? <EmptyState title="No categories" /> : (
          <>
            <BarsChart layout="horizontal" height={420} categoryWidth={190} categoryKey="dept" data={cats.data!.map((c) => ({ dept: c.dept_name, revenue: c.revenue }))} bars={[{ key: 'revenue', label: 'Revenue', color: COLOR_ACTUAL }]} />
            <div className="mt-4">
              <Table caption="Category revenue table">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="th">Category</th>
                    <th className="th text-right">Revenue</th>
                    <th className="th text-right">Share</th>
                    <th className="th text-right">Units</th>
                    <th className="th text-right">Products sold</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {cats.data!.map((c) => (
                    <tr key={c.dept_name}>
                      <td className="td max-w-xs truncate">{c.dept_name}</td>
                      <td className="td text-right tabular-nums">{fmtInt(c.revenue)}</td>
                      <td className="td text-right tabular-nums">{fmtPct(c.revenue_share_pct)}</td>
                      <td className="td text-right tabular-nums">{fmtInt(c.quantity)}</td>
                      <td className="td text-right tabular-nums">{fmtInt(c.n_products)}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          </>
        )}
      </Card>
      {s && <p className="mt-4 text-xs text-slate-500">Period {fmtDate(s.start_date)} – {fmtDate(s.end_date)}.</p>}
    </>
  )
}
