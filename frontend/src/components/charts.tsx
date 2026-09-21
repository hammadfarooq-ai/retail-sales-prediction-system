import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from 'recharts'
import { AXIS, GRID } from './ui'
import { fmtCompact, fmtDate, fmtDec } from '../utils/format'

export interface LineSeries {
  key: string
  label: string
  color: string
  dashed?: boolean
  width?: number
}

const tickStyle = { fontSize: 12, fill: AXIS }

function tooltipStyle() {
  return {
    contentStyle: { borderRadius: 8, border: '1px solid #e2e8f0', boxShadow: '0 4px 12px rgba(15,23,42,.08)', fontSize: 12 },
    labelStyle: { fontWeight: 600, color: '#0f172a' },
  }
}

/** Multi-series line chart with optional shaded interval band (lowerKey/upperKey). */
export function TimeSeriesChart({
  data,
  series,
  xKey = 'date',
  height = 300,
  xFormatter = (v: string) => fmtDate(v),
  yFormatter = fmtCompact,
  band,
  splitAt,
  splitLabel,
}: {
  data: Record<string, string | number | null>[]
  series: LineSeries[]
  xKey?: string
  height?: number
  xFormatter?: (v: string) => string
  yFormatter?: (v: number) => string
  band?: { lower: string; upper: string; label: string; color: string }
  splitAt?: string
  splitLabel?: string
}) {
  // Recharts stacks two areas to draw a band: base = lower (transparent), range = upper - lower.
  const rows = band
    ? data.map((d) => ({
        ...d,
        __base: d[band.lower],
        __range: d[band.upper] != null && d[band.lower] != null ? Number(d[band.upper]) - Number(d[band.lower]) : null,
      }))
    : data
  return (
    <div role="img" aria-label="Time series chart" style={{ height }} className="w-full">
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey={xKey} tickFormatter={xFormatter} tick={tickStyle} tickLine={false} axisLine={{ stroke: GRID }} minTickGap={32} />
          <YAxis tickFormatter={(v) => yFormatter(v as number)} tick={tickStyle} tickLine={false} axisLine={false} width={56} />
          <Tooltip
            {...tooltipStyle()}
            labelFormatter={(v) => xFormatter(String(v))}
            formatter={(v, name) => [typeof v === 'number' ? fmtDec(v, 2) : v, name]}
          />
          <Legend verticalAlign="top" height={28} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
          {band && (
            <>
              <Area dataKey="__base" stackId="band" stroke="none" fill="transparent" legendType="none" isAnimationActive={false} tooltipType="none" />
              <Area dataKey="__range" stackId="band" stroke="none" fill={band.color} fillOpacity={0.16} name={band.label} isAnimationActive={false} tooltipType="none" />
            </>
          )}
          {splitAt && <ReferenceLine x={splitAt} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: splitLabel, fill: AXIS, fontSize: 11, position: 'insideTopRight' }} />}
          {series.map((s) => (
            <Line
              key={s.key}
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={s.width ?? 2}
              strokeDasharray={s.dashed ? '6 4' : undefined}
              dot={false}
              activeDot={{ r: 4, stroke: '#fff', strokeWidth: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Vertical or horizontal bars; `stackKeys` renders stacked segments with a 2px surface gap. */
export function BarsChart({
  data,
  categoryKey,
  bars,
  layout = 'vertical',
  height = 300,
  stacked = false,
  yFormatter = fmtCompact,
  categoryFormatter,
  categoryWidth = 120,
}: {
  data: Record<string, string | number | null>[]
  categoryKey: string
  bars: LineSeries[]
  layout?: 'vertical' | 'horizontal'
  height?: number
  stacked?: boolean
  yFormatter?: (v: number) => string
  categoryFormatter?: (v: string) => string
  categoryWidth?: number
}) {
  const horizontalBars = layout === 'horizontal' // categories on Y axis
  return (
    <div role="img" aria-label="Bar chart" style={{ height }} className="w-full">
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout={horizontalBars ? 'vertical' : 'horizontal'}
          margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
          barCategoryGap="22%"
        >
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={!horizontalBars} vertical={horizontalBars} />
          {horizontalBars ? (
            <>
              <XAxis type="number" tickFormatter={(v) => yFormatter(v as number)} tick={tickStyle} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey={categoryKey} tick={tickStyle} tickLine={false} axisLine={false} width={categoryWidth} tickFormatter={categoryFormatter} />
            </>
          ) : (
            <>
              <XAxis dataKey={categoryKey} tick={tickStyle} tickLine={false} axisLine={{ stroke: GRID }} tickFormatter={categoryFormatter} minTickGap={12} />
              <YAxis tickFormatter={(v) => yFormatter(v as number)} tick={tickStyle} tickLine={false} axisLine={false} width={56} />
            </>
          )}
          <Tooltip {...tooltipStyle()} cursor={{ fill: 'rgba(148,163,184,.12)' }} formatter={(v, name) => [typeof v === 'number' ? fmtDec(v, 0) : v, name]} />
          {bars.length > 1 && <Legend verticalAlign="top" height={28} iconType="square" wrapperStyle={{ fontSize: 12 }} />}
          {bars.map((b, i) => (
            <Bar
              key={b.key}
              dataKey={b.key}
              name={b.label}
              fill={b.color}
              stackId={stacked ? 'a' : undefined}
              stroke={stacked ? '#ffffff' : undefined}
              strokeWidth={stacked ? 2 : 0}
              radius={stacked ? (i === bars.length - 1 ? [4, 4, 0, 0] : 0) : horizontalBars ? [0, 4, 4, 0] : [4, 4, 0, 0]}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Actual (x) vs predicted (y) scatter with a y = x reference line. */
export function ActualVsPredictedScatter({ data, height = 300 }: { data: { actual: number; predicted: number }[]; height?: number }) {
  const max = Math.max(1, ...data.map((d) => Math.max(d.actual, d.predicted)))
  return (
    <div role="img" aria-label="Scatter chart of actual versus predicted quantity" style={{ height }} className="w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis type="number" dataKey="actual" name="Actual" domain={[0, max]} tick={tickStyle} tickLine={false} label={{ value: 'Actual units', position: 'insideBottom', offset: -8, fontSize: 12, fill: AXIS }} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[0, max]} tick={tickStyle} tickLine={false} axisLine={false} width={48} />
          <Tooltip {...tooltipStyle()} cursor={{ strokeDasharray: '3 3' }} formatter={(v) => (typeof v === 'number' ? fmtDec(v, 1) : v)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="#94a3b8" strokeDasharray="4 4" />
          <Scatter data={data} fill="#2a78d6" fillOpacity={0.45} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
