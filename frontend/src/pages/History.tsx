import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { Badge, Card, EmptyState, ErrorState, PageHeader, Pagination, Spinner, Table } from '../components/ui'
import { fmtDate, fmtDateTime, fmtDec, shortId } from '../utils/format'

const LIMIT = 15

export default function History() {
  const filters = useFilters()
  const [store, setStore] = useState<number | null>(null)
  const [item, setItem] = useState('')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [offset, setOffset] = useState(0)
  const itemOk = item === '' || /^[A-Za-z0-9_-]{1,32}$/.test(item)

  const { data, loading, error, refetch } = useApi(
    () => api.predictions({ store_id: store, item_id: item || undefined, target_from: from || undefined, target_to: to || undefined, limit: LIMIT, offset }),
    [store, item, from, to, offset],
    itemOk,
  )
  const reset = () => setOffset(0)

  return (
    <>
      <PageHeader title="Prediction History" description="Every prediction made through the API or this UI, stored in PostgreSQL with its inputs and model version." />
      <div className="card mb-6 flex flex-wrap items-end gap-3 p-4">
        <div className="w-full sm:w-40">
          <label htmlFor="h-store" className="label">Store</label>
          <select id="h-store" className="input" value={store ?? ''} onChange={(e) => { setStore(e.target.value ? Number(e.target.value) : null); reset() }}>
            <option value="">All stores</option>
            {filters.stores.map((s) => <option key={s.store_id} value={s.store_id}>Store {s.store_id}</option>)}
          </select>
        </div>
        <div className="w-full sm:w-56">
          <label htmlFor="h-item" className="label">Product ID</label>
          <input id="h-item" className="input" value={item} onChange={(e) => { setItem(e.target.value.trim()); reset() }} placeholder="exact product id" aria-invalid={!itemOk} />
          {!itemOk && <p className="mt-1 text-xs text-red-600" role="alert">Letters, digits, - and _ only</p>}
        </div>
        <div className="w-[calc(50%-0.375rem)] sm:w-40">
          <label htmlFor="h-from" className="label">Target date from</label>
          <input id="h-from" type="date" className="input" value={from} onChange={(e) => { setFrom(e.target.value); reset() }} />
        </div>
        <div className="w-[calc(50%-0.375rem)] sm:w-40">
          <label htmlFor="h-to" className="label">Target date to</label>
          <input id="h-to" type="date" className="input" value={to} onChange={(e) => { setTo(e.target.value); reset() }} />
        </div>
        <button type="button" className="btn-secondary" onClick={() => { setStore(null); setItem(''); setFrom(''); setTo(''); reset() }}>Clear</button>
      </div>

      <Card>
        {error ? <ErrorState message={error} onRetry={refetch} /> : loading && !data ? <Spinner /> : !data || data.total === 0 ? (
          <EmptyState title="No predictions yet" description="Predictions you make on the Sales Prediction page (or through the API) will show up here." />
        ) : (
          <>
            <Table caption="Prediction history">
              <thead className="bg-slate-50">
                <tr>
                  <th className="th">Created</th>
                  <th className="th">Store</th>
                  <th className="th">Product</th>
                  <th className="th">Target date</th>
                  <th className="th">Inputs</th>
                  <th className="th text-right">Predicted</th>
                  <th className="th text-right">Interval</th>
                  <th className="th text-right">Actual</th>
                  <th className="th">Model</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.items.map((p) => {
                  const req = p.inputs.request
                  const eff = p.inputs.effective
                  return (
                    <tr key={p.id}>
                      <td className="td">{fmtDateTime(p.created_at)}<div className="text-xs text-slate-400">#{p.id} · {p.source}</div></td>
                      <td className="td">{p.store_id}</td>
                      <td className="td" title={p.item_id}>{shortId(p.item_id)}</td>
                      <td className="td">{fmtDate(p.target_date)}</td>
                      <td className="td text-xs text-slate-600">
                        price {eff?.price != null ? fmtDec(eff.price, 2) : '–'} · {eff?.promotion ? `promo ${fmtDec(eff.discount_pct, 0)}%` : 'no promo'}
                        {req?.price != null && <Badge tone="blue">custom price</Badge>}
                      </td>
                      <td className="td text-right font-semibold tabular-nums">{fmtDec(p.predicted_quantity, 2)}</td>
                      <td className="td text-right tabular-nums text-slate-600">{p.lower != null ? `${fmtDec(p.lower, 1)}–${fmtDec(p.upper, 1)}` : '–'}</td>
                      <td className="td text-right tabular-nums">{p.actual_quantity != null ? fmtDec(p.actual_quantity, 2) : '–'}</td>
                      <td className="td text-xs">{p.model_version}</td>
                    </tr>
                  )
                })}
              </tbody>
            </Table>
            <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
          </>
        )}
      </Card>
    </>
  )
}
