import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { Card, ChartSkeleton, ErrorState, PageHeader, Table, COLOR_ACTUAL, SERIES } from '../components/ui'
import { BarsChart } from '../components/charts'
import { FilterBar, defaultFilterState, type FilterState } from '../components/FilterBar'
import { fmtCompact, fmtDate, fmtDec, fmtInt } from '../utils/format'

export default function Stores() {
  const filters = useFilters()
  const [f, setF] = useState<FilterState>(() => defaultFilterState(filters))
  const { data, loading, error, refetch } = useApi(() => api.byStore({ start_date: f.start_date, end_date: f.end_date }), [f.start_date, f.end_date])

  return (
    <>
      <PageHeader title="Stores" description="Compare the stores in the dataset. Store 4 opened on 2023-12-13, so lifetime totals are not comparable without the active-days column." />
      <FilterBar filters={filters} value={f} onChange={setF} showDept={false} showStore={false} />
      {error && <ErrorState message={error} onRetry={refetch} />}
      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Revenue by store">
          {loading ? <ChartSkeleton /> : <BarsChart categoryKey="store" data={(data ?? []).map((s) => ({ store: `Store ${s.store_id}`, revenue: s.revenue }))} bars={[{ key: 'revenue', label: 'Revenue', color: COLOR_ACTUAL }]} />}
        </Card>
        <Card title="Average daily revenue per m²" subtitle="Normalises for store size and trading days">
          {loading ? <ChartSkeleton /> : <BarsChart categoryKey="store" yFormatter={(v) => fmtDec(v, 1)} data={(data ?? []).map((s) => ({ store: `Store ${s.store_id}`, value: s.avg_daily_revenue_per_m2 }))} bars={[{ key: 'value', label: 'Revenue / day / m²', color: SERIES[1] }]} />}
        </Card>
      </div>
      <Card className="mt-6" title="Store details" subtitle={`${fmtDate(f.start_date)} – ${fmtDate(f.end_date)}`}>
        <Table caption="Stores">
          <thead className="bg-slate-50">
            <tr>
              <th className="th">Store</th>
              <th className="th">Format</th>
              <th className="th">City</th>
              <th className="th">Division</th>
              <th className="th text-right">Area (m²)</th>
              <th className="th text-right">Active days</th>
              <th className="th text-right">Revenue</th>
              <th className="th text-right">Units</th>
              <th className="th text-right">Avg daily revenue</th>
              <th className="th text-right">Products sold</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {(data ?? []).map((s) => (
              <tr key={s.store_id}>
                <td className="td font-medium">Store {s.store_id}</td>
                <td className="td">{s.format}</td>
                <td className="td">{s.city}</td>
                <td className="td">{s.division}</td>
                <td className="td text-right tabular-nums">{fmtInt(s.area)}</td>
                <td className="td text-right tabular-nums">{fmtInt(s.active_days)}</td>
                <td className="td text-right tabular-nums">{fmtCompact(s.revenue)}</td>
                <td className="td text-right tabular-nums">{fmtCompact(s.quantity)}</td>
                <td className="td text-right tabular-nums">{fmtCompact(s.avg_daily_revenue)}</td>
                <td className="td text-right tabular-nums">{fmtInt(s.n_products)}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </>
  )
}
