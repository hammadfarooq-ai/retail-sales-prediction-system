import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Spinner } from './components/ui'
import { FiltersProvider } from './hooks/FiltersContext'
import { AppLayout } from './layouts/AppLayout'
import NotFound from './pages/NotFound'

// Route-level code splitting keeps the initial bundle small (Recharts is heavy).
const Dashboard = lazy(() => import('./pages/Dashboard'))
const SalesAnalytics = lazy(() => import('./pages/SalesAnalytics'))
const Predict = lazy(() => import('./pages/Predict'))
const Forecasts = lazy(() => import('./pages/Forecasts'))
const Stores = lazy(() => import('./pages/Stores'))
const Products = lazy(() => import('./pages/Products'))
const ModelPerformance = lazy(() => import('./pages/ModelPerformance'))
const History = lazy(() => import('./pages/History'))
const About = lazy(() => import('./pages/About'))

export default function App() {
  return (
    <FiltersProvider>
      <Suspense fallback={<Spinner />}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="analytics" element={<SalesAnalytics />} />
            <Route path="predict" element={<Predict />} />
            <Route path="forecasts" element={<Forecasts />} />
            <Route path="stores" element={<Stores />} />
            <Route path="products" element={<Products />} />
            <Route path="model" element={<ModelPerformance />} />
            <Route path="history" element={<History />} />
            <Route path="about" element={<About />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Suspense>
    </FiltersProvider>
  )
}
