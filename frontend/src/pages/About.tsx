import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { Card, PageHeader } from '../components/ui'
import { fmtDate, fmtInt } from '../utils/format'

export default function About() {
  const filters = useFilters()
  const info = useApi(() => api.modelInfo(), [])
  const m = info.data
  return (
    <>
      <PageHeader title="About" description="What this system does and where its limits are." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="The problem">
          <p className="text-sm leading-relaxed text-slate-700">
            The model predicts <strong>daily units sold</strong> for a <strong>store × product</strong> pair, {m?.horizon_days ?? 7} days ahead. It covers
            {' '}{fmtInt(filters.forecastable_pairs)} store-product pairs with regular demand (sold on ≥70% of days since first sale), which is the
            portion of the assortment where daily forecasting is meaningful. Longer horizons are produced recursively and are less accurate.
          </p>
        </Card>
        <Card title="Data">
          <p className="text-sm leading-relaxed text-slate-700">
            Kaggle dataset{' '}
            <a className="text-brand-600 underline" href="https://www.kaggle.com/datasets/svizor/retail-sales-forecasting-data" target="_blank" rel="noreferrer">
              svizor/retail-sales-forecasting-data
            </a>
            : {filters.stores.length} stores, sales from {fmtDate(filters.min_date)} to {fmtDate(filters.max_date)}, a product catalog and a promotion calendar. Revenue has no
            currency in the source data, so values are shown as plain numbers.
          </p>
        </Card>
        <Card title="Model">
          {m ? (
            <ul className="space-y-1 text-sm text-slate-700">
              <li>Algorithm: <strong>{m.model_name}</strong> ({m.version})</li>
              <li>Features: {m.n_features} (calendar, lags ≥ 7 days, rolling stats, price & promotion, store/category)</li>
              <li>Split: train ≤ {m.split.train_end}, validation {m.split.val_start} → {m.split.val_end}, test {m.split.test_start} → {m.split.test_end}</li>
              <li>Prediction interval: {m.has_prediction_interval ? '80% (quantile models), days 1–7 only' : 'not available'}</li>
            </ul>
          ) : <p className="text-sm text-slate-500">Loading…</p>}
        </Card>
        <Card title="Known limitations">
          <ul className="list-inside list-disc space-y-1 text-sm text-slate-700">
            <li>Only regular-demand products are forecastable; new or intermittent items are not covered.</li>
            <li>The promotion calendar is treated as known in advance; retroactively recorded promotions would make results optimistic.</li>
            <li>Online orders, markdown clearance and stock levels are not modelled.</li>
            <li>Only ~2 years of history: yearly seasonality is weakly identified.</li>
          </ul>
        </Card>
      </div>
    </>
  )
}
