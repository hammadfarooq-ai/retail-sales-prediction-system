import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import Predict from '../pages/Predict'
import { ToastProvider } from '../components/Toast'
import { FiltersProvider } from '../hooks/FiltersContext'
import { api } from '../services/api'

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>('../services/api')
  return {
    ...actual,
    api: {
      filters: vi.fn(),
      modelInfo: vi.fn(),
      products: vi.fn(),
      predict: vi.fn(),
    },
  }
})

const filters = {
  stores: [{ store_id: 1, division: 'Div1', format: 'Format-1', city: 'City1', area: 1500 }],
  departments: ['A'],
  min_date: '2022-08-28',
  max_date: '2024-09-26',
  max_forecast_horizon_days: 28,
  forecastable_pairs: 100,
}
const product = { item_id: 'abc123', dept_name: 'DEPT', class_name: 'C', subclass_name: 'Milk', item_type: null, forecastable_stores: [1] }

function setup() {
  vi.mocked(api.filters).mockResolvedValue(filters)
  vi.mocked(api.modelInfo).mockResolvedValue({ model_name: 'lightgbm', last_observed_date: '2024-09-26' } as never)
  vi.mocked(api.products).mockResolvedValue([product])
  return render(
    <MemoryRouter>
      <ToastProvider>
        <FiltersProvider>
          <Predict />
        </FiltersProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('Predict page', () => {
  it('shows a validation error and does not call the API when no product is selected', async () => {
    setup()
    const btn = await screen.findByRole('button', { name: /predict sales/i })
    await userEvent.click(btn)
    expect(await screen.findByText('Select a product')).toBeInTheDocument()
    expect(api.predict).not.toHaveBeenCalled()
  })

  it('submits the selected inputs and renders the result with an input summary', async () => {
    vi.mocked(api.predict).mockResolvedValue({
      id: 7,
      model_version: 'lightgbm-x',
      store_id: 1,
      item_id: 'abc123',
      dept_name: 'DEPT',
      subclass_name: 'Milk',
      date: '2024-09-27',
      predicted_quantity: 12.345,
      lower: 8,
      upper: 17,
      interval_level: 0.8,
      mode: 'forecast',
      horizon_days: 1,
      is_recursive: false,
      actual_quantity: null,
      inputs: { price: 99.9, last_observed_price: 99.9, promotion: false, discount_pct: 0, promotion_source: 'none' },
      created_at: null,
    })
    setup()
    await userEvent.click(await screen.findByPlaceholderText(/search by name/i))
    fireEvent.mouseDown(await screen.findByRole('option', { name: /milk/i }))
    await userEvent.click(screen.getByRole('button', { name: /predict sales/i }))
    await waitFor(() => expect(api.predict).toHaveBeenCalled())
    expect(vi.mocked(api.predict).mock.calls[0][0]).toMatchObject({ store_id: 1, item_id: 'abc123', date: '2024-09-27', promotion: null })
    expect(await screen.findByTestId('predicted-value')).toHaveTextContent('12.35')
    expect(screen.getByText(/80% prediction interval/i)).toBeInTheDocument()
    expect(screen.getByText('Future forecast')).toBeInTheDocument()
  })

  it('surfaces API errors to the user', async () => {
    vi.mocked(api.predict).mockRejectedValue(new Error('x'))
    setup()
    await userEvent.click(await screen.findByPlaceholderText(/search by name/i))
    fireEvent.mouseDown(await screen.findByRole('option', { name: /milk/i }))
    await userEvent.click(screen.getByRole('button', { name: /predict sales/i }))
    expect((await screen.findAllByRole('alert')).length).toBeGreaterThan(0)
  })
})
