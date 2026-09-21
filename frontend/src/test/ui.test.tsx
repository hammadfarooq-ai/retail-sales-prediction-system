import { render, screen, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { ToastProvider, useToast } from '../components/Toast'
import { EmptyState, ErrorState, Pagination, StatCard } from '../components/ui'

function Trigger() {
  const t = useToast()
  return <button onClick={() => t.error('Boom happened')}>go</button>
}

describe('ui components', () => {
  it('shows and auto-dismisses toasts', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    )
    await userEvent.click(screen.getByText('go'))
    expect(screen.getByRole('alert')).toHaveTextContent('Boom happened')
    act(() => {
      vi.advanceTimersByTime(8000)
    })
    expect(screen.queryByText('Boom happened')).not.toBeInTheDocument()
    vi.useRealTimers()
  })

  it('renders empty and error states, retry fires', async () => {
    const retry = vi.fn()
    render(
      <>
        <EmptyState title="Nothing here" />
        <ErrorState message="bad" onRetry={retry} />
      </>,
    )
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(retry).toHaveBeenCalledOnce()
  })

  it('StatCard shows a loading placeholder instead of the value', () => {
    render(<StatCard label="Total" value="123" loading />)
    expect(screen.getByLabelText('Loading')).toBeInTheDocument()
    expect(screen.queryByText('123')).not.toBeInTheDocument()
  })

  it('Pagination moves between pages and disables edges', async () => {
    const onChange = vi.fn()
    render(<Pagination total={50} limit={20} offset={0} onChange={onChange} />)
    expect(screen.getByText('1–20 of 50')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(onChange).toHaveBeenCalledWith(20)
  })
})
