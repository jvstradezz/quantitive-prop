export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`
export const usd = (v: number, d = 0) => `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('ru-RU', { minimumFractionDigits: d, maximumFractionDigits: d })}`
export const signed = (v: number, d = 2) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`
