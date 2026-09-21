import { AxiosError, type AxiosResponse } from 'axios'
import { ApiError, toApiError } from './api'

function axiosErr(status: number, data: unknown): AxiosError {
  return new AxiosError('fail', 'ERR_BAD_REQUEST', undefined, undefined, {
    status,
    data,
    statusText: '',
    headers: {},
    config: {} as never,
  } as AxiosResponse)
}

describe('toApiError', () => {
  it('extracts the backend error envelope message', () => {
    const e = toApiError(axiosErr(404, { error: { code: 'unknown_store', message: 'Store 9 does not exist' } }))
    expect(e).toBeInstanceOf(ApiError)
    expect(e.message).toBe('Store 9 does not exist')
    expect(e.status).toBe(404)
    expect(e.code).toBe('unknown_store')
  })
  it('summarises validation errors by first field', () => {
    const e = toApiError(
      axiosErr(422, {
        error: { code: 'validation_error', message: 'Request validation failed', details: [{ field: 'date', message: 'Input should be a valid date' }] },
      }),
    )
    expect(e.message).toBe('date: Input should be a valid date')
  })
  it('reports network failures', () => {
    const e = toApiError(new AxiosError('Network Error'))
    expect(e.code).toBe('network_error')
    expect(e.message).toMatch(/Cannot reach the server/)
  })
  it('falls back for unknown errors', () => {
    expect(toApiError('boom').message).toBe('Unexpected error')
  })
})
