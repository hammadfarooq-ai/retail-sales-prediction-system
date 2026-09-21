import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useFilters } from '../hooks/FiltersContext'
import { Badge, Card, ChartSkeleton, ErrorState, PageHeader, StatCard, Table, COLOR_ACTUAL, COLOR_BASE, COLOR_PRED, SERIES } from '../components/ui'
import { ActualVsPredictedScatter, BarsChart, TimeSeriesChart } from '../components/charts'
import { fmtDateTime, fmtDec, fmtInt, fmtPct, fmtShortDate } from '../utils/format'

export default function ModelPerformance() {
  const filters = useFilters()
  const perf = useApi(() => api.modelPerformance(), [])
  const info = useApi(() => api.modelInfo(), [])
  const [store, setStore] = useState<number | null>(null)
  const bt = useApi(() => api.backtest({ store_id: store }), [store])

  if (perf.error) return (<><PageHeader title="Model Performance" /><ErrorState message={perf.error} onRetry={perf.refetch} /></>)
  const p = perf.data
  const t = p?.final_model_test

  const rows = p ? Object.entries(p.comparison).map(([name, r]) => ({ name, ...r })) : []
  const best = p?.model_name
  const importance = p ? Object.entries(p.feature_importance).slice(0, 12).map(([k, v]) => ({ feature: k, share: v * 100 })) : []

  return (
    <>
      <PageHeader title="Model Performance" description="All numbers below are real outputs of the training pipeline (chronological split; the test period was never used for model selection)." />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Selected model" loading={perf.loading} value={p?.model_name ?? ''} hint={p ? `Version ${p.version}` : undefined} />
        <StatCard label="MAE (test)" loading={perf.loading} value={t ? fmtDec(t.mae, 3) : ''} hint="units / product-store-day" />
        <StatCard label="RMSE (test)" loading={perf.loading} value={t ? fmtDec(t.rmse, 3) : ''} />
        <StatCard label="WAPE (test)" loading={perf.loading} value={t ? fmtPct(t.wape) : ''} hint={p ? `${fmtPct(p.improvement_vs_lag7_baseline_pct.mae)} lower MAE than naive` : undefined} />
        <StatCard label="SMAPE (test)" loading={perf.loading} value={t ? fmtPct(t.smape) : ''} />
        <StatCard label="MAPE (test)" loading={perf.loading} value={t ? fmtPct(t.mape) : ''} hint={t ? `on ${fmtPct(t.mape_coverage_pct, 0)} of rows with sales > 0` : undefined} />
        <StatCard label="R² (test)" loading={perf.loading} value={t ? fmtDec(t.r2, 3) : ''} />
        <StatCard label="Features" loading={perf.loading} value={p ? fmtInt(p.n_features) : ''} hint={info.data ? `${fmtInt(info.data.n_series)} series` : undefined} />
      </div>

      {p && info.data && (
        <Card className="mt-6" title="Training details">
          <dl className="grid grid-cols-1 gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
            <D k="Trained at" v={fmtDateTime(p.trained_at)} />
            <D k="Training period" v={`${p.split.data_start} → ${p.split.train_end}`} />
            <D k="Validation period" v={`${p.split.val_start} → ${p.split.val_end}`} />
            <D k="Test period" v={`${p.split.test_start} → ${p.split.test_end}`} />
            <D k="Forecast horizon" v={`${info.data.horizon_days} days (direct); longer = recursive`} />
            <D k="Selection metric" v={p.selection_metric} />
            <D k="Prediction interval" v={`${Math.round(p.prediction_interval.level * 100)}% · empirical test coverage ${fmtPct(p.prediction_interval.empirical_test_coverage_pct)}`} />
            <D k="Training rows" v={fmtInt(info.data.n_train_rows)} />
            <D k="Final fit" v="Train + validation (test untouched)" />
          </dl>
        </Card>
      )}

      <Card className="mt-6" title="Model comparison" subtitle="Each model fit on the training period only; scored on validation and test">
        {perf.loading ? <ChartSkeleton height={200} /> : (
          <>
            <Table caption="Model comparison">
              <thead className="bg-slate-50">
                <tr>
                  <th className="th">Model</th>
                  <th className="th text-right">Val MAE</th>
                  <th className="th text-right">Val RMSE</th>
                  <th className="th text-right">Test MAE</th>
                  <th className="th text-right">Test RMSE</th>
                  <th className="th text-right">Test SMAPE</th>
                  <th className="th text-right">Test WAPE</th>
                  <th className="th text-right">Test R²</th>
                  <th className="th text-right">Fit (s)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {rows.map((r) => (
                  <tr key={r.name} className={r.name === best ? 'bg-brand-50/60' : ''}>
                    <td className="td font-medium">
                      {r.name} {r.name === best && <Badge tone="blue">selected</Badge>} {r.is_baseline && <Badge>baseline</Badge>}
                      <div className="text-xs font-normal text-slate-500">{r.description}</div>
                    </td>
                    <td className="td text-right tabular-nums">{fmtDec(r.validation.mae, 3)}</td>
                    <td className="td text-right tabular-nums">{fmtDec(r.validation.rmse, 3)}</td>
                    <td className="td text-right tabular-nums">{fmtDec(r.test.mae, 3)}</td>
                    <td className="td text-right tabular-nums">{fmtDec(r.test.rmse, 3)}</td>
                    <td className="td text-right tabular-nums">{fmtPct(r.test.smape)}</td>
                    <td className="td text-right tabular-nums">{fmtPct(r.test.wape)}</td>
                    <td className="td text-right tabular-nums">{fmtDec(r.test.r2, 3)}</td>
                    <td className="td text-right tabular-nums">{r.is_baseline ? '–' : fmtDec(r.fit_seconds, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <div className="mt-6">
              <BarsChart
                height={280}
                categoryKey="name"
                data={rows.map((r) => ({ name: r.name.replace('naive_', 'naive ').replace('_', ' '), val: r.validation.mae, test: r.test.mae }))}
                bars={[
                  { key: 'val', label: 'Validation MAE', color: COLOR_ACTUAL },
                  { key: 'test', label: 'Test MAE', color: COLOR_PRED },
                ]}
                yFormatter={(v) => fmtDec(v, 1)}
              />
            </div>
            <p className="mt-2 text-xs text-slate-500">
              The final selected model is refit on train+validation; its test metrics (cards above) can differ slightly from its train-only row here.
            </p>
          </>
        )}
      </Card>

      <div className="mt-6 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Backtest on the held-out test period</h2>
        <select aria-label="Backtest store" className="input !w-auto" value={store ?? ''} onChange={(e) => setStore(e.target.value ? Number(e.target.value) : null)}>
          <option value="">All stores</option>
          {filters.stores.map((s) => <option key={s.store_id} value={s.store_id}>Store {s.store_id}</option>)}
        </select>
      </div>
      <div className="mt-4 grid gap-6 xl:grid-cols-2">
        <Card title="Actual vs predicted (daily total)">
          {bt.loading ? <ChartSkeleton /> : bt.error ? <ErrorState message={bt.error} onRetry={bt.refetch} /> : (
            <TimeSeriesChart
              xFormatter={fmtShortDate}
              data={bt.data!.points.map((x) => ({ date: x.date, actual: x.actual, predicted: x.predicted, naive: x.baseline_lag7 }))}
              series={[
                { key: 'actual', label: 'Actual', color: COLOR_ACTUAL },
                { key: 'predicted', label: 'Model', color: COLOR_PRED },
                { key: 'naive', label: 'Naive lag-7', color: COLOR_BASE, dashed: true, width: 1.5 },
              ]}
            />
          )}
        </Card>
        <Card title="Actual vs predicted (sample of 600 rows)" subtitle="Each dot is a product-store-day; dashed line = perfect prediction">
          {bt.loading ? <ChartSkeleton /> : bt.error ? null : <ActualVsPredictedScatter data={bt.data!.scatter_sample} />}
        </Card>
        <Card title="Prediction error distribution" subtitle={bt.data ? `predicted − actual · mean ${fmtDec(bt.data.residual_summary.mean, 3)}, std ${fmtDec(bt.data.residual_summary.std, 2)} (clipped to 1–99th percentile)` : undefined}>
          {bt.loading ? <ChartSkeleton /> : bt.error ? null : (
            <BarsChart
              categoryKey="bin"
              data={bt.data!.error_histogram.map((b) => ({ bin: fmtDec((b.lower + b.upper) / 2, 1), count: b.count }))}
              bars={[{ key: 'count', label: 'Rows', color: SERIES[3] }]}
              yFormatter={(v) => fmtInt(v)}
            />
          )}
        </Card>
        <Card title="Feature importance" subtitle="Share of total gain (top 12)">
          {perf.loading ? <ChartSkeleton /> : importance.length === 0 ? <p className="text-sm text-slate-500">Not available for this model type.</p> : (
            <BarsChart layout="horizontal" categoryKey="feature" categoryWidth={140} data={importance} bars={[{ key: 'share', label: 'Importance %', color: SERIES[2] }]} yFormatter={(v) => `${fmtDec(v, 0)}%`} />
          )}
        </Card>
      </div>

      {p && (
        <Card className="mt-6" title="Test metrics by store & aggregation level">
          <Table caption="Metrics by store">
            <thead className="bg-slate-50">
              <tr><th className="th">Level</th><th className="th text-right">MAE</th><th className="th text-right">RMSE</th><th className="th text-right">WAPE</th><th className="th text-right">R²</th><th className="th text-right">Rows</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {[...Object.entries(p.final_model_test_by_store).map(([k, v]) => [`Store ${k} (product-day)`, v] as const), ...Object.entries(p.final_model_test_aggregated).map(([k, v]) => [`Aggregated: ${k.replace(/_/g, ' ')}`, v] as const)].map(([label, m]) => (
                <tr key={label}>
                  <td className="td">{label}</td>
                  <td className="td text-right tabular-nums">{fmtDec(m.mae, 3)}</td>
                  <td className="td text-right tabular-nums">{fmtDec(m.rmse, 3)}</td>
                  <td className="td text-right tabular-nums">{fmtPct(m.wape)}</td>
                  <td className="td text-right tabular-nums">{fmtDec(m.r2, 3)}</td>
                  <td className="td text-right tabular-nums">{fmtInt(m.n_rows)}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card>
      )}
    </>
  )
}

function D({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
      <dd className="mt-0.5 font-medium text-slate-800">{v}</dd>
    </div>
  )
}
