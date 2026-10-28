export type Rgb = [number, number, number]

export function parseColor(input: string): Rgb {
  const s = input.trim()
  if (s.startsWith('#')) {
    const h = s.length === 4 ? [...s.slice(1)].map((c) => c + c).join('') : s.slice(1)
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
  }
  const m = s.match(/rgba?\(([^)]+)\)/)
  if (m) {
    const p = m[1].split(/[ ,/]+/).map(Number)
    return [p[0], p[1], p[2]]
  }
  throw new Error(`Неизвестный формат цвета: ${input}`)
}

const hex = (n: number) => Math.round(n).toString(16).padStart(2, '0')

/** Смешивает `a` с `b` в доле `t` (0 — чистый `a`, 1 — чистый `b`), возвращает #rrggbb. */
export function mix(a: string, b: string, t: number): string {
  const ca = parseColor(a)
  const cb = parseColor(b)
  const [r, g, bl] = ca.map((v, i) => v + (cb[i] - v) * t)
  return `#${hex(r)}${hex(g)}${hex(bl)}`
}

export function withAlpha(color: string, alpha: number): string {
  const [r, g, b] = parseColor(color)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
