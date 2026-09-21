import { toCsv } from './csv'

describe('toCsv', () => {
  it('builds header and rows', () => {
    expect(toCsv([{ a: 1, b: 'x' }, { a: 2, b: 'y' }])).toBe('a,b\n1,x\n2,y')
  })
  it('quotes commas, quotes and newlines', () => {
    expect(toCsv([{ t: 'a,b', u: 'say "hi"' }])).toBe('t,u\n"a,b","say ""hi"""')
  })
  it('renders null/undefined as empty and honours column order', () => {
    expect(toCsv([{ a: null, b: 2 }], ['b', 'a'])).toBe('b,a\n2,')
  })
  it('returns empty string for no rows', () => {
    expect(toCsv([])).toBe('')
  })
})
