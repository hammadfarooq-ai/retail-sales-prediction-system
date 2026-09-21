import { addDays, fmtCompact, fmtDec, fmtInt, fmtPct, shortId, toISODate } from './format'

describe('format helpers', () => {
  it('formats integers and handles missing values', () => {
    expect(fmtInt(1234567)).toBe('1,234,567')
    expect(fmtInt(null)).toBe('–')
    expect(fmtInt(Number.NaN)).toBe('–')
  })
  it('formats decimals', () => {
    expect(fmtDec(3.14159, 2)).toBe('3.14')
    expect(fmtDec(undefined)).toBe('–')
  })
  it('compacts large numbers', () => {
    expect(fmtCompact(1_500)).toBe('1.5K')
    expect(fmtCompact(2_340_000)).toBe('2.34M')
    expect(fmtCompact(5_600_000_000)).toBe('5.60B')
    expect(fmtCompact(12)).toBe('12')
  })
  it('formats percentages', () => {
    expect(fmtPct(40.236)).toBe('40.2%')
    expect(fmtPct(null)).toBe('–')
  })
  it('does date arithmetic without timezone drift', () => {
    expect(addDays('2024-09-26', 1)).toBe('2024-09-27')
    expect(addDays('2024-03-01', -1)).toBe('2024-02-29')
    expect(toISODate(new Date(2024, 0, 5))).toBe('2024-01-05')
  })
  it('shortens ids', () => {
    expect(shortId('abcdef123456')).toBe('abcdef12…')
    expect(shortId('abc')).toBe('abc')
  })
})
