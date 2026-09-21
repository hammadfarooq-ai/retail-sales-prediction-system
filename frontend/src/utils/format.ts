/** Number/date formatting helpers. The dataset has no currency, so revenue is shown as plain units. */

const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const nf1 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 })
const nf2 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 })

export function fmtInt(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? '–' : nf0.format(n)
}

export function fmtDec(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '–'
  return digits === 1 ? nf1.format(n) : digits === 2 ? nf2.format(n) : n.toFixed(digits)
}

/** 1.2K / 3.4M / 5.6B */
export function fmtCompact(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '–'
  const abs = Math.abs(n)
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return nf1.format(n)
}

export function fmtPct(n: number | null | undefined, digits = 1): string {
  return n == null || Number.isNaN(n) ? '–' : `${n.toFixed(digits)}%`
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '–'
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00` : iso)
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '–'
  return new Date(iso).toLocaleString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function fmtShortDate(iso: string): string {
  const d = new Date(`${iso.slice(0, 10)}T00:00:00`)
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
}

export function fmtMonth(iso: string): string {
  const d = new Date(`${iso.slice(0, 10)}T00:00:00`)
  return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit' })
}

export function toISODate(d: Date): string {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

export function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00`)
  d.setDate(d.getDate() + days)
  return toISODate(d)
}

export function shortId(id: string): string {
  return id.length > 8 ? `${id.slice(0, 8)}…` : id
}
