import { useEffect, useId, useState } from 'react'
import { api } from '../services/api'
import { useDebounce } from '../hooks/useDebounce'
import type { Product } from '../types/api'
import { shortId } from '../utils/format'

/**
 * Searchable product dropdown limited to forecastable products of the chosen store
 * (the model only covers items with regular demand).
 */
export function ProductPicker({
  storeId,
  value,
  onChange,
  id,
  error,
  required = false,
}: {
  storeId: number | null
  value: Product | null
  onChange: (p: Product | null) => void
  id: string
  error?: string | null
  required?: boolean
}) {
  const [query, setQuery] = useState('')
  const [options, setOptions] = useState<Product[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const debounced = useDebounce(query, 250)
  const listId = useId()

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setLoading(true)
    setFailed(false)
    api
      .products({ q: debounced || undefined, store_id: storeId, forecastable_only: true, limit: 25 })
      .then((r) => !cancelled && setOptions(r))
      .catch(() => !cancelled && setFailed(true))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [debounced, storeId, open])

  const label = value ? `${value.subclass_name} · ${shortId(value.item_id)}` : ''
  return (
    <div className="relative">
      <input
        id={id}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-required={required}
        aria-invalid={!!error}
        aria-describedby={error ? `${id}-error` : undefined}
        autoComplete="off"
        className="input"
        placeholder="Search by name or ID (top sellers shown)"
        value={open ? query : label}
        onFocus={() => {
          setOpen(true)
          setQuery('')
        }}
        onChange={(e) => {
          setQuery(e.target.value)
          setOpen(true)
        }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
      />
      {open && (
        <ul id={listId} role="listbox" className="absolute z-20 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-slate-200 bg-white py-1 text-sm shadow-lg">
          {loading && <li className="px-3 py-2 text-slate-500">Searching…</li>}
          {failed && <li className="px-3 py-2 text-red-600">Could not load products</li>}
          {!loading && !failed && options.length === 0 && <li className="px-3 py-2 text-slate-500">No forecastable products match</li>}
          {options.map((p) => (
            <li
              key={p.item_id}
              role="option"
              aria-selected={value?.item_id === p.item_id}
              className="cursor-pointer px-3 py-2 hover:bg-brand-50"
              onMouseDown={(e) => {
                e.preventDefault()
                onChange(p)
                setOpen(false)
              }}
            >
              <div className="font-medium text-slate-800">{p.subclass_name}</div>
              <div className="text-xs text-slate-500">
                {p.dept_name} · {p.item_id}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
