import { useState, type FormEvent } from 'react'
import { api, toApiError } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { useToast } from '../components/Toast'
import { Badge, Card, EmptyState, ErrorState, Field, PageHeader } from '../components/ui'
import { ProductPicker } from '../components/ProductPicker'
import { addDays, fmtDate, fmtDec, fmtPct } from '../utils/format'
import type { PredictResponse, Product } from '../types/api'

interface FormErrors {
  store?: string
  product?: string
  date?: string
  price?: string
  discount?: string
}

const MODE_LABEL: Record<PredictResponse['mode'], { text: string; tone: 'blue' | 'green' | 'amber'; help: string }> = {
  forecast: { text: 'Future forecast', tone: 'blue', help: 'Date is after the last observed sales day.' },
  held_out_test: { text: 'Held-out test day', tone: 'green', help: 'The model never saw this day, so the actual value is a fair accuracy check.' },
  in_sample: { text: 'In-sample day', tone: 'amber', help: 'The final model was trained on this day; the prediction is optimistic.' },
}

export default function Predict() {
  const filters = useFilters()
  const toast = useToast()
  const info = useApi(() => api.modelInfo(), [])
  const last = info.data?.last_observed_date ?? filters.max_date
  const maxDate = addDays(last, filters.max_forecast_horizon_days)

  const [storeId, setStoreId] = useState<number>(filters.stores[0].store_id)
  const [product, setProduct] = useState<Product | null>(null)
  const [date, setDate] = useState<string>(addDays(filters.max_date, 1))
  const [price, setPrice] = useState('')
  const [promo, setPromo] = useState<'calendar' | 'yes' | 'no'>('calendar')
  const [discount, setDiscount] = useState('')
  const [errors, setErrors] = useState<FormErrors>({})
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)

  function validate(): FormErrors {
    const e: FormErrors = {}
    if (!product) e.product = 'Select a product'
    else if (!product.forecastable_stores.includes(storeId)) e.product = `This product is not forecastable in store ${storeId}`
    if (!date) e.date = 'Choose a date'
    else if (date > maxDate) e.date = `Latest supported date is ${fmtDate(maxDate)}`
    if (price && !(Number(price) > 0)) e.price = 'Price must be greater than 0'
    if (discount && !(Number(discount) >= 0 && Number(discount) <= 95)) e.discount = 'Discount must be between 0 and 95'
    return e
  }

  async function onSubmit(ev: FormEvent) {
    ev.preventDefault()
    const e = validate()
    setErrors(e)
    setApiError(null)
    if (Object.keys(e).length) return
    setSubmitting(true)
    try {
      const res = await api.predict({
        store_id: storeId,
        item_id: product!.item_id,
        date,
        price: price ? Number(price) : null,
        promotion: promo === 'calendar' ? null : promo === 'yes',
        discount_pct: promo === 'yes' && discount ? Number(discount) : null,
      })
      setResult(res)
      toast.success(`Prediction saved (#${res.id})`)
    } catch (err) {
      const msg = toApiError(err).message
      setApiError(msg)
      toast.error(msg)
    } finally {
      setSubmitting(false)
    }
  }

  const modelBadge = info.data ? `${info.data.model_name}` : ''

  return (
    <>
      <PageHeader
        title="Sales Prediction"
        description={`Predict units sold for one product in one store on a given day. Predictions are stored in history. Model: ${modelBadge || 'loading…'}; every input maps to a real model feature.`}
      />
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2" title="Inputs">
          <form onSubmit={onSubmit} noValidate className="space-y-4" aria-label="Prediction form">
            <Field label="Store" htmlFor="p-store">
              <select
                id="p-store"
                className="input"
                value={storeId}
                onChange={(e) => {
                  setStoreId(Number(e.target.value))
                  setProduct(null)
                }}
              >
                {filters.stores.map((s) => (
                  <option key={s.store_id} value={s.store_id}>
                    Store {s.store_id} · {s.format} · {s.city}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Product" htmlFor="p-product" error={errors.product} hint={`Only products with regular demand can be forecast (${filters.forecastable_pairs.toLocaleString()} store-product pairs).`}>
              <ProductPicker id="p-product" storeId={storeId} value={product} onChange={setProduct} error={errors.product} required />
            </Field>
            {product && (
              <p className="-mt-2 text-xs text-slate-500">
                Category: <span className="font-medium text-slate-700">{product.dept_name}</span>
              </p>
            )}
            <Field label="Date" htmlFor="p-date" error={errors.date} hint={`Data ends ${fmtDate(last)}. Past dates show a backtest with the actual value.`}>
              <input id="p-date" type="date" className="input" value={date} min={addDays(last, -140)} max={maxDate} onChange={(e) => setDate(e.target.value)} aria-invalid={!!errors.date} aria-describedby={errors.date ? 'p-date-error' : undefined} />
            </Field>
            <Field label="Planned price (optional)" htmlFor="p-price" error={errors.price} hint="Leave empty to use the product's last observed price.">
              <input id="p-price" type="number" step="0.01" min="0" inputMode="decimal" className="input" value={price} onChange={(e) => setPrice(e.target.value)} placeholder="e.g. 129.90" aria-invalid={!!errors.price} />
            </Field>
            <Field label="Promotion" htmlFor="p-promo" hint="“Use calendar” applies the promotion scheduled for that day, if any.">
              <select id="p-promo" className="input" value={promo} onChange={(e) => setPromo(e.target.value as typeof promo)}>
                <option value="calendar">Use promo calendar</option>
                <option value="yes">Promotion on</option>
                <option value="no">No promotion</option>
              </select>
            </Field>
            {promo === 'yes' && (
              <Field label="Discount depth (%)" htmlFor="p-disc" error={errors.discount} hint="Empty = scheduled depth, else the dataset median (14%).">
                <input id="p-disc" type="number" min="0" max="95" step="1" className="input" value={discount} onChange={(e) => setDiscount(e.target.value)} placeholder="e.g. 20" aria-invalid={!!errors.discount} />
              </Field>
            )}
            <button type="submit" className="btn-primary w-full" disabled={submitting}>
              {submitting ? 'Predicting…' : 'Predict sales'}
            </button>
          </form>
        </Card>

        <div className="space-y-4 lg:col-span-3">
          {apiError && <ErrorState message={apiError} />}
          {!result && !apiError && <EmptyState title="No prediction yet" description="Fill in the form and press “Predict sales”. The result and an input summary appear here." />}
          {result && <ResultCard r={result} />}
        </div>
      </div>
    </>
  )
}

function ResultCard({ r }: { r: PredictResponse }) {
  const mode = MODE_LABEL[r.mode]
  const err = r.actual_quantity != null ? r.predicted_quantity - r.actual_quantity : null
  return (
    <Card title="Prediction result" actions={<Badge tone={mode.tone}>{mode.text}</Badge>}>
      <div className="flex flex-wrap items-end gap-x-8 gap-y-2">
        <div>
          <p className="text-sm text-slate-500">Predicted units sold</p>
          <p className="text-5xl font-semibold tabular-nums text-slate-900" data-testid="predicted-value">
            {fmtDec(r.predicted_quantity, 2)}
          </p>
        </div>
        {r.lower != null && r.upper != null && (
          <div>
            <p className="text-sm text-slate-500">{Math.round((r.interval_level ?? 0.8) * 100)}% prediction interval</p>
            <p className="text-xl font-medium tabular-nums text-slate-700">
              {fmtDec(r.lower, 2)} – {fmtDec(r.upper, 2)}
            </p>
          </div>
        )}
        {r.actual_quantity != null && (
          <div>
            <p className="text-sm text-slate-500">Actual (from data)</p>
            <p className="text-xl font-medium tabular-nums text-slate-700">
              {fmtDec(r.actual_quantity, 2)} <span className="text-sm text-slate-500">(error {err! >= 0 ? '+' : ''}{fmtDec(err, 2)})</span>
            </p>
          </div>
        )}
      </div>
      <p className="mt-3 text-xs text-slate-500">{mode.help}</p>
      {r.lower == null && r.is_recursive && (
        <p className="mt-2 rounded bg-amber-50 p-2 text-xs text-amber-800">
          Horizon is {r.horizon_days} days after the last observed day, beyond the 7-day direct model, so this value is a recursive forecast and no interval is shown.
        </p>
      )}
      <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-slate-100 pt-4 text-sm sm:grid-cols-3">
        <Item k="Prediction date" v={fmtDate(r.date)} />
        <Item k="Store" v={`Store ${r.store_id}`} />
        <Item k="Product" v={r.subclass_name ?? r.item_id} />
        <Item k="Category" v={r.dept_name ?? '–'} />
        <Item k="Price used" v={r.inputs.price != null ? fmtDec(r.inputs.price, 2) : 'n/a'} />
        <Item k="Last observed price" v={r.inputs.last_observed_price != null ? fmtDec(r.inputs.last_observed_price, 2) : 'n/a'} />
        <Item k="Promotion" v={r.inputs.promotion ? `On (${fmtPct(r.inputs.discount_pct, 0)} off)` : 'Off'} />
        <Item k="Promotion source" v={r.inputs.promotion_source.replace('_', ' ')} />
        <Item k="Horizon" v={r.horizon_days > 0 ? `${r.horizon_days} day(s) ahead` : 'past day (backtest)'} />
        <Item k="Model version" v={r.model_version} />
        <Item k="History id" v={r.id != null ? `#${r.id}` : '–'} />
      </dl>
    </Card>
  )
}

function Item({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
      <dd className="mt-0.5 break-words font-medium text-slate-800">{v}</dd>
    </div>
  )
}
