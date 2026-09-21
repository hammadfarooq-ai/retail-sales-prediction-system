import type { Filters } from '../types/api'
import { addDays } from '../utils/format'

export interface FilterState {
  store_id: number | null
  dept_name: string | null
  start_date: string
  end_date: string
}

const PRESETS = [
  { label: 'Last 30 days', days: 30 },
  { label: 'Last 90 days', days: 90 },
  { label: 'Last 365 days', days: 365 },
  { label: 'All time', days: 0 },
]

/** Filter row: store, department, date range and presets (dates are bounded by the real data range). */
export function FilterBar({
  filters,
  value,
  onChange,
  showDept = true,
  showStore = true,
}: {
  filters: Filters
  value: FilterState
  onChange: (v: FilterState) => void
  showDept?: boolean
  showStore?: boolean
}) {
  const set = (patch: Partial<FilterState>) => onChange({ ...value, ...patch })
  return (
    <div className="card mb-6 flex flex-wrap items-end gap-3 p-4" role="search" aria-label="Filters">
      {showStore && (
        <div className="w-full sm:w-44">
          <label htmlFor="f-store" className="label">
            Store
          </label>
          <select id="f-store" className="input" value={value.store_id ?? ''} onChange={(e) => set({ store_id: e.target.value ? Number(e.target.value) : null })}>
            <option value="">All stores</option>
            {filters.stores.map((s) => (
              <option key={s.store_id} value={s.store_id}>
                Store {s.store_id} · {s.format}
              </option>
            ))}
          </select>
        </div>
      )}
      {showDept && (
        <div className="w-full sm:w-64">
          <label htmlFor="f-dept" className="label">
            Category (department)
          </label>
          <select id="f-dept" className="input" value={value.dept_name ?? ''} onChange={(e) => set({ dept_name: e.target.value || null })}>
            <option value="">All categories</option>
            {filters.departments.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="w-[calc(50%-0.375rem)] sm:w-40">
        <label htmlFor="f-start" className="label">
          From
        </label>
        <input id="f-start" type="date" className="input" min={filters.min_date} max={value.end_date} value={value.start_date} onChange={(e) => e.target.value && set({ start_date: e.target.value })} />
      </div>
      <div className="w-[calc(50%-0.375rem)] sm:w-40">
        <label htmlFor="f-end" className="label">
          To
        </label>
        <input id="f-end" type="date" className="input" min={value.start_date} max={filters.max_date} value={value.end_date} onChange={(e) => e.target.value && set({ end_date: e.target.value })} />
      </div>
      <div className="flex flex-wrap gap-2" role="group" aria-label="Date presets">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            className="btn-secondary !px-3 !py-1.5"
            onClick={() =>
              set({
                end_date: filters.max_date,
                start_date: p.days === 0 ? filters.min_date : maxIso(addDays(filters.max_date, -(p.days - 1)), filters.min_date),
              })
            }
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  )
}

const maxIso = (a: string, b: string) => (a > b ? a : b)

export function defaultFilterState(filters: Filters): FilterState {
  return { store_id: null, dept_name: null, start_date: filters.min_date, end_date: filters.max_date }
}
