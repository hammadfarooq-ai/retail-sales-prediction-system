import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useDebounce } from '../hooks/useDebounce'
import { useFilters } from '../hooks/FiltersContext'
import { Badge, Card, EmptyState, ErrorState, PageHeader, Pagination, Spinner, Table } from '../components/ui'
import { fmtCompact, fmtDate, fmtDec, fmtInt } from '../utils/format'

const LIMIT = 20

export default function Products() {
  const filters = useFilters()
  const [store, setStore] = useState<number | null>(null)
  const [dept, setDept] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<'revenue' | 'quantity'>('revenue')
  const [forecastable, setForecastable] = useState(false)
  const [offset, setOffset] = useState(0)
  const q = useDebounce(search, 300)

  const { data, loading, error, refetch } = useApi(
    () => api.byProduct({ store_id: store, dept_name: dept, search: q || undefined, sort_by: sortBy, forecastable_only: forecastable, limit: LIMIT, offset }),
    [store, dept, q, sortBy, forecastable, offset],
  )
  const reset = () => setOffset(0)

  return (
    <>
      <PageHeader title="Products" description="Best sellers per store. Totals cover the whole dataset period. “Forecastable” products have regular enough demand for the model." />
      <div className="card mb-6 flex flex-wrap items-end gap-3 p-4">
        <div className="w-full sm:w-64">
          <label htmlFor="pr-search" className="label">Search</label>
          <input id="pr-search" className="input" placeholder="Product ID or name" value={search} onChange={(e) => { setSearch(e.target.value); reset() }} />
        </div>
        <div className="w-full sm:w-44">
          <label htmlFor="pr-store" className="label">Store</label>
          <select id="pr-store" className="input" value={store ?? ''} onChange={(e) => { setStore(e.target.value ? Number(e.target.value) : null); reset() }}>
            <option value="">All stores</option>
            {filters.stores.map((s) => <option key={s.store_id} value={s.store_id}>Store {s.store_id}</option>)}
          </select>
        </div>
        <div className="w-full sm:w-64">
          <label htmlFor="pr-dept" className="label">Category</label>
          <select id="pr-dept" className="input" value={dept ?? ''} onChange={(e) => { setDept(e.target.value || null); reset() }}>
            <option value="">All categories</option>
            {filters.departments.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="w-full sm:w-36">
          <label htmlFor="pr-sort" className="label">Sort by</label>
          <select id="pr-sort" className="input" value={sortBy} onChange={(e) => { setSortBy(e.target.value as 'revenue' | 'quantity'); reset() }}>
            <option value="revenue">Revenue</option>
            <option value="quantity">Units</option>
          </select>
        </div>
        <label className="flex items-center gap-2 pb-2 text-sm text-slate-700">
          <input type="checkbox" checked={forecastable} onChange={(e) => { setForecastable(e.target.checked); reset() }} className="h-4 w-4 rounded border-slate-300" />
          Forecastable only
        </label>
      </div>

      <Card>
        {error ? <ErrorState message={error} onRetry={refetch} /> : loading && !data ? <Spinner /> : !data || data.items.length === 0 ? (
          <EmptyState title="No products match" description="Adjust the search or filters." />
        ) : (
          <>
            <Table caption="Products">
              <thead className="bg-slate-50">
                <tr>
                  <th className="th">Product</th>
                  <th className="th">Category</th>
                  <th className="th">Store</th>
                  <th className="th text-right">Revenue</th>
                  <th className="th text-right">Units</th>
                  <th className="th text-right">Avg price</th>
                  <th className="th text-right">Sale days</th>
                  <th className="th">Last sale</th>
                  <th className="th">Model</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.items.map((p) => (
                  <tr key={`${p.store_id}-${p.item_id}`}>
                    <td className="td max-w-xs">
                      <div className="truncate font-medium text-slate-800">{p.subclass_name}</div>
                      <div className="text-xs text-slate-500">{p.item_id}</div>
                    </td>
                    <td className="td max-w-[14rem] truncate">{p.dept_name}</td>
                    <td className="td">{p.store_id}</td>
                    <td className="td text-right tabular-nums">{fmtCompact(p.revenue)}</td>
                    <td className="td text-right tabular-nums">{fmtInt(p.quantity)}</td>
                    <td className="td text-right tabular-nums">{fmtDec(p.avg_price, 2)}</td>
                    <td className="td text-right tabular-nums">{fmtInt(p.sale_days)}</td>
                    <td className="td">{fmtDate(p.last_sale)}</td>
                    <td className="td">
                      {p.forecastable ? <Link to="/forecasts" className="inline-flex"><Badge tone="green">Forecastable</Badge></Link> : <Badge>Not covered</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
          </>
        )}
      </Card>
    </>
  )
}
