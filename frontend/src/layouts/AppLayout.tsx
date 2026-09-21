import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge } from '../components/ui'

const NAV = [
  { to: '/', label: 'Dashboard', icon: 'M3 12l9-9 9 9M5 10v10h5v-6h4v6h5V10' },
  { to: '/analytics', label: 'Sales Analytics', icon: 'M4 20V10m6 10V4m6 16v-7m4 7H2' },
  { to: '/predict', label: 'Sales Prediction', icon: 'M13 2L3 14h8l-1 8 10-12h-8l1-8z' },
  { to: '/forecasts', label: 'Forecasts', icon: 'M3 17l6-6 4 4 8-8M15 7h6v6' },
  { to: '/stores', label: 'Stores', icon: 'M3 9l1-5h16l1 5M4 9v11h16V9M9 20v-6h6v6' },
  { to: '/products', label: 'Products', icon: 'M21 8l-9-5-9 5v8l9 5 9-5V8zM3 8l9 5 9-5M12 13v8' },
  { to: '/model', label: 'Model Performance', icon: 'M12 2a10 10 0 100 20 10 10 0 000-20zm0 5v5l3 3' },
  { to: '/history', label: 'Prediction History', icon: 'M12 8v4l3 2M3 12a9 9 0 109-9 9 9 0 00-7 3.3L3 8m0-5v5h5' },
  { to: '/about', label: 'About', icon: 'M12 16v-4m0-4h.01M22 12a10 10 0 11-20 0 10 10 0 0120 0z' },
]

function Icon({ d }: { d: string }) {
  return (
    <svg aria-hidden viewBox="0 0 24 24" className="h-5 w-5 flex-none" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  )
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="flex h-full flex-col">
      <div className="flex items-center gap-2 px-5 py-5">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-brand-500 text-white" aria-hidden>
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 17l5-5 4 3 7-8" />
          </svg>
        </span>
        <span className="text-base font-semibold tracking-tight text-slate-900">Retail Predictor</span>
      </div>
      <ul className="flex-1 space-y-1 px-3">
        {NAV.map((n) => (
          <li key={n.to}>
            <NavLink
              to={n.to}
              end={n.to === '/'}
              onClick={onNavigate}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`
              }
            >
              <Icon d={n.icon} />
              {n.label}
            </NavLink>
          </li>
        ))}
      </ul>
      <p className="px-5 py-4 text-xs text-slate-400">Data: Kaggle · svizor/retail-sales-forecasting-data</p>
    </nav>
  )
}

function HealthBadge() {
  const { data, error } = useApi(() => api.health(), [])
  if (error) return <Badge tone="red">API offline</Badge>
  if (!data) return <Badge>Checking…</Badge>
  return data.status === 'ok' ? <Badge tone="green">Model {data.model_version?.split('-').slice(0, -1).join('-')} · online</Badge> : <Badge tone="amber">Degraded</Badge>
}

export function AppLayout() {
  const [open, setOpen] = useState(false)
  const location = useLocation()
  useEffect(() => setOpen(false), [location.pathname])

  return (
    <div className="min-h-screen lg:pl-64">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:p-2">
        Skip to content
      </a>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r border-slate-200 bg-white lg:block">
        <Sidebar />
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation menu">
          <div className="absolute inset-0 bg-slate-900/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 bg-white shadow-xl">
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur sm:px-6">
        <button type="button" className="btn-secondary !px-2.5 lg:hidden" aria-label="Open navigation menu" onClick={() => setOpen(true)}>
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <div className="hidden text-sm text-slate-500 lg:block">Retail sales forecasting</div>
        <HealthBadge />
      </header>

      <main id="main" className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <Outlet />
      </main>
    </div>
  )
}
