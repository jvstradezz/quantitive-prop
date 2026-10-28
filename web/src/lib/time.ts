const fmt = (tz: string) =>
  new Intl.DateTimeFormat('ru-RU', { timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: false })

const et = fmt('America/New_York')
const msk = fmt('Europe/Moscow')

const etOffset = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', timeZoneName: 'shortOffset' })

/** Смещение Нью-Йорка от UTC в часах (−4 летом, −5 зимой). */
export function etOffsetHours(d: Date): number {
  const part = etOffset.formatToParts(d).find((p) => p.type === 'timeZoneName')?.value ?? 'GMT-5'
  const m = part.match(/GMT([+-]\d+)/)
  return m ? Number(m[1]) : -5
}

/** Время сервера FundedNext. Допущение (O-10 в docs/DECISIONS.md): сервер = ET + 7 часов. */
export function serverFn(d: Date): string {
  const off = etOffsetHours(d) + 7
  const shifted = new Date(d.getTime() + off * 3600_000)
  const h = String(shifted.getUTCHours()).padStart(2, '0')
  const m = String(shifted.getUTCMinutes()).padStart(2, '0')
  return `${h}:${m}`
}

export const timeEt = (d: Date) => et.format(d)
export const timeMsk = (d: Date) => msk.format(d)

export function dateShort(d: Date): string {
  return new Intl.DateTimeFormat('ru-RU', { timeZone: 'America/New_York', day: '2-digit', month: '2-digit' }).format(d)
}
