import { createContext, useContext, type ReactNode } from 'react'
import { api } from '../services/api'
import { useApi } from './useApi'
import type { Filters } from '../types/api'
import { ErrorState, Spinner } from '../components/ui'

interface Ctx {
  filters: Filters
}
const FiltersContext = createContext<Ctx | null>(null)

/** Loads stores / departments / date bounds once from the backend and gates the app on them. */
export function FiltersProvider({ children }: { children: ReactNode }) {
  const { data, loading, error, refetch } = useApi(() => api.filters(), [])
  if (loading && !data) return <Spinner label="Connecting to the API" />
  if (error || !data) {
    return (
      <div className="mx-auto max-w-xl p-8">
        <ErrorState message={error ?? 'No data returned. Has the database been seeded?'} onRetry={refetch} />
      </div>
    )
  }
  return <FiltersContext.Provider value={{ filters: data }}>{children}</FiltersContext.Provider>
}

export function useFilters(): Filters {
  const ctx = useContext(FiltersContext)
  if (!ctx) throw new Error('useFilters must be used inside <FiltersProvider>')
  return ctx.filters
}
