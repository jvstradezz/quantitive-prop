/**
 * ФЕЙКОВЫЕ ДАННЫЕ для статичного макета (M0/T3). Ни одна цифра здесь не получена
 * из движка или из бэктеста — всё сгенерировано детерминированным PRNG или написано руками.
 * В M5 страницы переключаются на API (docs/ARCHITECTURE.md).
 */
import { rng } from '../lib/rng'

// ───────────── Идеи ─────────────

export type IdeaStatus = 'Новая' | 'Формализована' | 'Кодится' | 'Тест' | 'Результат' | 'Демо' | 'Отклонена'

export const IDEA_STATUSES: IdeaStatus[] = ['Новая', 'Формализована', 'Кодится', 'Тест', 'Результат', 'Демо', 'Отклонена']

export interface IdeaSession {
  name: string
  et: string
  msk: string
  server: string
}

export interface Idea {
  id: string
  title: string
  status: IdeaStatus
  created: string
  summary: string
  instruments: string[]
  sessions: IdeaSession[]
  assumptions: string[]
  result?: string
  verdict?: string
  demo?: boolean
}

export const ideas: Idea[] = [
  {
    id: 'IDEA-0001',
    title: 'Range bot по мотивам Sniper Range Bot V2',
    status: 'Новая',
    created: '2026-10-28',
    summary:
      'Чужой бот на NQ торгует диапазон, стоп и тейк 1:1, направление зависит от дня недели. Проверяем на NAS100 и других индексах CFD, главный вопрос — держится ли фильтр дня на OOS.',
    instruments: ['NAS100', 'US30', 'SPX500', 'GER40'],
    sessions: [
      { name: 'Первые 30 мин NY', et: '09:30–10:00', msk: '16:30–17:00', server: '16:30–17:00' },
      { name: 'Лондон', et: '03:00–04:00', msk: '10:00–11:00', server: '10:00–11:00' },
    ],
    assumptions: ['Диапазон = high/low окна', 'Выход 1:1 / 1:2 / трейлинг — отдельные варианты', 'Фильтр дня: пн/ср лонг, чт шорт, вт/пт оба'],
  },
  {
    id: 'IDEA-0002',
    title: 'Пример: возврат к VWAP после новостного шипа',
    status: 'Формализована',
    created: '2026-10-29',
    summary: 'Вход против импульса после выхода данных, если цена вернулась к VWAP за 15 минут. Стоп за экстремум шипа.',
    instruments: ['EURUSD', 'XAUUSD', 'NAS100'],
    sessions: [{ name: 'Окно данных США', et: '08:30–09:15', msk: '15:30–16:15', server: '15:30–16:15' }],
    assumptions: ['Календарь новостей — исторический, из бесплатного источника', 'Вне окна ±5 мин сделок нет'],
  },
  {
    id: 'IDEA-0003',
    title: 'Пример: пробой азиатского диапазона на золоте',
    status: 'Тест',
    created: '2026-10-30',
    summary: 'Пробой high/low азиатской сессии с подтверждением закрытием M15. Выходы: RR 1:2 и трейлинг по ATR.',
    instruments: ['XAUUSD', 'XAGUSD'],
    sessions: [{ name: 'Азия', et: '19:00–03:00', msk: '02:00–10:00', server: '02:00–10:00' }],
    assumptions: ['Время сервера FN = ET + 7 (O-10)', 'Своп учитывается по спецификации'],
  },
  {
    id: 'IDEA-0004',
    title: 'Пример: тренд на H1 с фильтром волатильности',
    status: 'Результат',
    created: '2026-10-25',
    summary: 'Пересечение каналов Дончиана на H1, вход только при ATR выше медианы.',
    instruments: ['GER40', 'NAS100', 'USDJPY'],
    sessions: [{ name: 'Лондон + NY', et: '03:00–16:00', msk: '10:00–23:00', server: '10:00–23:00' }],
    assumptions: ['Выходы по времени — закрытие в 22:00 сервера'],
    result: 'PF OOS 1.18, DSR 0.71, издержки 41% валовой.',
    verdict: 'Отклонить: DSR ниже порога, издержки выше 30%.',
  },
  {
    id: 'IDEA-0005',
    title: 'Пример: ночной откат на индексах',
    status: 'Демо',
    created: '2026-10-20',
    summary: 'Покупка слабости перед закрытием NY с выходом на открытии Лондона.',
    instruments: ['GER40', 'UK100'],
    sessions: [{ name: 'Закрытие NY → Лондон', et: '15:30–03:00', msk: '22:30–10:00', server: '22:30–10:00' }],
    assumptions: ['Своп и тройной своп в пятницу — по спецификации'],
    result: 'PF OOS 1.46, DSR 0.97, P(1-я выплата) 38%.',
    verdict: 'В демо-форвард (S4).',
    demo: true,
  },
  {
    id: 'IDEA-0006',
    title: 'Пример: мартингейл по сетке на EURUSD',
    status: 'Отклонена',
    created: '2026-10-22',
    summary: 'Усреднение против позиции с шагом 15 пунктов.',
    instruments: ['EURUSD'],
    sessions: [],
    assumptions: [],
    verdict: 'Автоотказ: усреднение против позиции (docs/VALIDATION.md).',
  },
]

// ───────────── Leaderboard ─────────────

export type Stage = 'S0' | 'S1' | 'S2' | 'S3' | 'S4' | 'S5' | 'S6'
export type Market = 'Индексы' | 'Металлы' | 'Форекс' | 'Сырьё'
export type Family = 'Пробой диапазона' | 'Возврат к среднему' | 'Тренд' | 'Сезонность'

export interface StrategyRow {
  id: string
  name: string
  market: Market
  instrument: string
  family: Family
  program: string
  pPayout: number
  ev: number
  pfOos: number
  dsr: number
  trades: number
  stage: Stage
  costShare: number
  status: 'Ок' | 'Демо' | 'Пауза' | 'Отклонена'
}

export const strategies: StrategyRow[] = [
  { id: 'STR-014', name: 'Asia range breakout', market: 'Металлы', instrument: 'XAUUSD', family: 'Пробой диапазона', program: 'FN Stellar 2-Step 25K', pPayout: 0.41, ev: 38, pfOos: 1.42, dsr: 0.96, trades: 312, stage: 'S4', costShare: 0.24, status: 'Демо' },
  { id: 'STR-021', name: 'NY close fade', market: 'Индексы', instrument: 'GER40', family: 'Возврат к среднему', program: 'FN Stellar Lite 25K', pPayout: 0.37, ev: 29, pfOos: 1.38, dsr: 0.97, trades: 276, stage: 'S4', costShare: 0.19, status: 'Демо' },
  { id: 'STR-009', name: 'London open range', market: 'Индексы', instrument: 'NAS100', family: 'Пробой диапазона', program: 'FN Stellar 1-Step 25K', pPayout: 0.33, ev: 17, pfOos: 1.31, dsr: 0.95, trades: 244, stage: 'S3', costShare: 0.27, status: 'Ок' },
  { id: 'STR-033', name: 'H1 Donchian + ATR filter', market: 'Форекс', instrument: 'USDJPY', family: 'Тренд', program: 'FP 2-Step Pro 25K', pPayout: 0.31, ev: 12, pfOos: 1.26, dsr: 0.94, trades: 408, stage: 'S3', costShare: 0.33, status: 'Ок' },
  { id: 'STR-027', name: 'VWAP reversion after news', market: 'Индексы', instrument: 'NAS100', family: 'Возврат к среднему', program: 'FN Stellar 2-Step 25K', pPayout: 0.28, ev: 6, pfOos: 1.24, dsr: 0.93, trades: 203, stage: 'S3', costShare: 0.36, status: 'Ок' },
  { id: 'STR-018', name: 'Silver weekly seasonality', market: 'Металлы', instrument: 'XAGUSD', family: 'Сезонность', program: 'FN Stellar Lite 25K', pPayout: 0.24, ev: -4, pfOos: 1.19, dsr: 0.88, trades: 156, stage: 'S2', costShare: 0.29, status: 'Ок' },
  { id: 'STR-040', name: 'Brent EIA drift', market: 'Сырьё', instrument: 'UKOIL', family: 'Сезонность', program: 'FP 2-Step Standard 25K', pPayout: 0.22, ev: -9, pfOos: 1.17, dsr: 0.81, trades: 129, stage: 'S2', costShare: 0.31, status: 'Пауза' },
  { id: 'STR-005', name: 'EURUSD Bollinger fade', market: 'Форекс', instrument: 'EURUSD', family: 'Возврат к среднему', program: 'FP 2-Step Pro 25K', pPayout: 0.19, ev: -15, pfOos: 1.12, dsr: 0.74, trades: 521, stage: 'S2', costShare: 0.44, status: 'Ок' },
  { id: 'STR-036', name: 'US30 opening drive', market: 'Индексы', instrument: 'US30', family: 'Пробой диапазона', program: 'FN Stellar 1-Step 25K', pPayout: 0.17, ev: -18, pfOos: 1.1, dsr: 0.69, trades: 287, stage: 'S1', costShare: 0.39, status: 'Ок' },
  { id: 'STR-012', name: 'GBPUSD London breakout', market: 'Форекс', instrument: 'GBPUSD', family: 'Пробой диапазона', program: 'FP 2-Step Standard 25K', pPayout: 0.12, ev: -27, pfOos: 1.05, dsr: 0.52, trades: 340, stage: 'S1', costShare: 0.47, status: 'Отклонена' },
]

// ───────────── Страница стратегии ─────────────

export const strategyMeta = {
  id: 'STR-014',
  name: 'Asia range breakout · XAUUSD',
  stage: 'S3' as Stage,
  account: 25_000,
  riskPct: 0.5,
  verdict: 'Фейковый вердикт для макета: в демо-форвард, пока обязательно сверить спред в азиатской сессии с выгрузкой платформы.',
}

export interface SeriesPoint {
  time: string
  value: number
}

function businessDays(start: string, count: number): string[] {
  const out: string[] = []
  const d = new Date(`${start}T00:00:00Z`)
  while (out.length < count) {
    const wd = d.getUTCDay()
    if (wd !== 0 && wd !== 6) out.push(d.toISOString().slice(0, 10))
    d.setUTCDate(d.getUTCDate() + 1)
  }
  return out
}

export interface Trade {
  id: number
  instrument: string
  side: 'Лонг' | 'Шорт'
  openUtc: string
  closeUtc: string
  rGross: number
  cost: number
  rNet: number
  pnl: number
}

const COST_R = { spread: 0.062, commission: 0.021, swap: 0.006, slippage: 0.034 }
const COST_TOTAL_R = COST_R.spread + COST_R.commission + COST_R.swap + COST_R.slippage

function buildTrades(): { days: string[]; trades: Trade[] } {
  const r = rng(14)
  const days = businessDays('2024-01-02', 520)
  const trades: Trade[] = []
  const risk = (strategyMeta.account * strategyMeta.riskPct) / 100
  let id = 1
  days.forEach((day, i) => {
    if (r.next() > 0.52 || i < 3) return
    const win = r.next() < 0.5
    const rGross = win ? 1.7 + Math.abs(r.normal()) * 0.25 : -1
    const rNet = rGross - COST_TOTAL_R
    const openH = 2 + Math.floor(r.next() * 6)
    const openM = Math.floor(r.next() * 60)
    const holdMin = 20 + Math.floor(r.next() * 180)
    const open = new Date(`${day}T${String(openH).padStart(2, '0')}:${String(openM).padStart(2, '0')}:00Z`)
    const close = new Date(open.getTime() + holdMin * 60_000)
    trades.push({
      id: id++,
      instrument: 'XAUUSD',
      side: r.next() < 0.55 ? 'Лонг' : 'Шорт',
      openUtc: open.toISOString(),
      closeUtc: close.toISOString(),
      rGross,
      cost: COST_TOTAL_R,
      rNet,
      pnl: rNet * risk,
    })
  })
  return { days, trades }
}

const built = buildTrades()
export const trades = built.trades

function buildEquity() {
  const equity: SeriesPoint[] = []
  const drawdown: SeriesPoint[] = []
  const byDay = new Map<string, number>()
  for (const t of trades) byDay.set(t.openUtc.slice(0, 10), (byDay.get(t.openUtc.slice(0, 10)) ?? 0) + t.pnl)
  let eq = strategyMeta.account
  let peak = eq
  for (const day of built.days) {
    eq += byDay.get(day) ?? 0
    peak = Math.max(peak, eq)
    equity.push({ time: day, value: Math.round(eq * 100) / 100 })
    drawdown.push({ time: day, value: -((peak - eq) / peak) * 100 })
  }
  return { equity, drawdown }
}

const eqBuilt = buildEquity()
export const equityCurve = eqBuilt.equity
export const drawdownCurve = eqBuilt.drawdown

/** Граница IS / OOS: ~70% истории. */
export const oosStart = built.days[Math.floor(built.days.length * 0.7)]

export interface FanPoint {
  time: string
  p5: number
  p25: number
  p50: number
  p75: number
  p95: number
}

/** Веер MC: бутстрэп сделок на 120 торговых дней вперёд, 1 500 путей. */
function buildFan(): FanPoint[] {
  const r = rng(77)
  const horizon = 120
  const paths = 1500
  const pool = trades.map((t) => t.pnl)
  const start = equityCurve[equityCurve.length - 1]
  const steps = new Array<Float64Array>(horizon)
  for (let s = 0; s < horizon; s++) steps[s] = new Float64Array(paths)
  for (let p = 0; p < paths; p++) {
    let eq = start.value
    for (let s = 0; s < horizon; s++) {
      if (r.next() < 0.52) eq += pool[Math.floor(r.next() * pool.length)]
      steps[s][p] = eq
    }
  }
  const days = businessDays(nextDay(start.time), horizon)
  const q = (arr: Float64Array, f: number) => {
    const sorted = Float64Array.from(arr).sort()
    return sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))]
  }
  return days.map((time, s) => ({
    time,
    p5: q(steps[s], 0.05),
    p25: q(steps[s], 0.25),
    p50: q(steps[s], 0.5),
    p75: q(steps[s], 0.75),
    p95: q(steps[s], 0.95),
  }))
}

function nextDay(day: string): string {
  const d = new Date(`${day}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + 1)
  return d.toISOString().slice(0, 10)
}

export const fan = buildFan()
export const fanStart = equityCurve[equityCurve.length - 1]

export const walkForward = [
  { window: 'W1', isPf: 1.52, oosPf: 1.34 },
  { window: 'W2', isPf: 1.48, oosPf: 1.41 },
  { window: 'W3', isPf: 1.61, oosPf: 1.22 },
  { window: 'W4', isPf: 1.45, oosPf: 1.38 },
  { window: 'W5', isPf: 1.57, oosPf: 1.12 },
  { window: 'W6', isPf: 1.5, oosPf: 1.47 },
]

export const heatmap = (() => {
  const rows = [15, 30, 45, 60, 90, 120] // окно диапазона, мин
  const cols = [0.5, 1, 1.5, 2, 2.5] // цель в R
  const r = rng(5)
  const values = rows.map((m, i) =>
    cols.map((_, j) => {
      const ridge = 1.42 - 0.035 * Math.abs(i - 2.2) ** 1.4 - 0.05 * Math.abs(j - 2) ** 1.3
      return Math.round((ridge + r.normal() * 0.012 + (m === 45 ? 0.01 : 0)) * 100) / 100
    }),
  )
  return { rows, cols, values, current: { row: 2, col: 2 } }
})()

const sum = (f: (t: Trade) => number) => trades.reduce((a, t) => a + f(t), 0)
const riskUsd = (strategyMeta.account * strategyMeta.riskPct) / 100
const grossUsd = sum((t) => t.rGross) * riskUsd

function costRow(name: string, rPerTrade: number) {
  const usd = rPerTrade * trades.length * riskUsd
  return { name, usd, pctAccount: (usd / strategyMeta.account) * 100, r: rPerTrade * trades.length, shareOfGross: usd / grossUsd }
}

export const costs = (() => {
  const rows = [
    costRow('Спред', COST_R.spread),
    costRow('Комиссия', COST_R.commission),
    costRow('Своп', COST_R.swap),
    costRow('Проскальзывание', COST_R.slippage),
  ]
  const totalUsd = rows.reduce((a, c) => a + c.usd, 0)
  return {
    rows,
    grossUsd,
    totalUsd,
    totalShare: totalUsd / grossUsd,
    netUsd: grossUsd - totalUsd,
    grossR: sum((t) => t.rGross),
    netR: sum((t) => t.rNet),
  }
})()

export const strategyStats = (() => {
  const wins = trades.filter((t) => t.rNet > 0)
  const losses = trades.filter((t) => t.rNet <= 0)
  const pf = wins.reduce((a, t) => a + t.pnl, 0) / Math.abs(losses.reduce((a, t) => a + t.pnl, 0))
  const maxDd = Math.min(...drawdownCurve.map((p) => p.value))
  return {
    trades: trades.length,
    winRate: wins.length / trades.length,
    pf,
    expectancyR: costs.netR / trades.length,
    netUsd: costs.netUsd,
    maxDd,
  }
})()

// ───────────── Prop simulator ─────────────

export interface PropProgram {
  id: string
  firm: 'FundedNext' | 'FundingPips'
  name: string
  fee: number
  phases: string
  pPhase1: number | null
  pPhase2: number | null
  pFunded: number
  pPayout: number
  ciLo: number
  ciHi: number
  daysMedian: number
  daysP90: number
  ev: number
  risk: { phase1: number; phase2: number | null; funded: number }
  fails: { daily: number; total: number; time: number; rule: number }
}

export const programs: PropProgram[] = [
  { id: 'fn-2s', firm: 'FundedNext', name: 'Stellar 2-Step 25K', fee: 204.99, phases: '8% / 5%', pPhase1: 0.63, pPhase2: 0.77, pFunded: 0.49, pPayout: 0.41, ciLo: 0.37, ciHi: 0.45, daysMedian: 43, daysP90: 88, ev: 38, risk: { phase1: 0.9, phase2: 0.6, funded: 0.4 }, fails: { daily: 0.34, total: 0.52, time: 0, rule: 0.14 } },
  { id: 'fn-1s', firm: 'FundedNext', name: 'Stellar 1-Step 25K', fee: 224.99, phases: '10%', pPhase1: 0.46, pPhase2: null, pFunded: 0.46, pPayout: 0.38, ciLo: 0.34, ciHi: 0.42, daysMedian: 38, daysP90: 81, ev: 21, risk: { phase1: 0.7, phase2: null, funded: 0.4 }, fails: { daily: 0.41, total: 0.46, time: 0, rule: 0.13 } },
  { id: 'fn-lite', firm: 'FundedNext', name: 'Stellar Lite 25K', fee: 144.99, phases: '8% / 4%', pPhase1: 0.57, pPhase2: 0.81, pFunded: 0.46, pPayout: 0.35, ciLo: 0.31, ciHi: 0.39, daysMedian: 47, daysP90: 96, ev: 29, risk: { phase1: 0.8, phase2: 0.5, funded: 0.4 }, fails: { daily: 0.31, total: 0.57, time: 0, rule: 0.12 } },
  { id: 'fn-inst', firm: 'FundedNext', name: 'Stellar Instant 10K', fee: 304.99, phases: 'нет', pPhase1: null, pPhase2: null, pFunded: 1, pPayout: 0.27, ciLo: 0.23, ciHi: 0.31, daysMedian: 29, daysP90: 74, ev: -12, risk: { phase1: 0.3, phase2: null, funded: 0.3 }, fails: { daily: 0, total: 0.78, time: 0, rule: 0.22 } },
  { id: 'fp-std', firm: 'FundingPips', name: '2-Step Standard 25K', fee: 168, phases: '8% / 5%', pPhase1: 0.61, pPhase2: 0.74, pFunded: 0.45, pPayout: 0.33, ciLo: 0.29, ciHi: 0.37, daysMedian: 41, daysP90: 90, ev: 14, risk: { phase1: 0.9, phase2: 0.6, funded: 0.4 }, fails: { daily: 0.37, total: 0.5, time: 0, rule: 0.13 } },
  { id: 'fp-pro', firm: 'FundingPips', name: '2-Step Pro 25K', fee: 134, phases: '6% / 6%', pPhase1: 0.68, pPhase2: 0.7, pFunded: 0.48, pPayout: 0.31, ciLo: 0.27, ciHi: 0.35, daysMedian: 36, daysP90: 79, ev: 17, risk: { phase1: 0.7, phase2: 0.7, funded: 0.3 }, fails: { daily: 0.46, total: 0.4, time: 0, rule: 0.14 } },
  { id: 'fp-zero', firm: 'FundingPips', name: 'Zero 25K', fee: 188, phases: 'нет', pPhase1: null, pPhase2: null, pFunded: 1, pPayout: 0.22, ciLo: 0.18, ciHi: 0.26, daysMedian: 24, daysP90: 61, ev: -33, risk: { phase1: 0.3, phase2: null, funded: 0.3 }, fails: { daily: 0.2, total: 0.52, time: 0, rule: 0.28 } },
]

export interface Stress {
  id: string
  label: string
  /** Множитель к P(1-я выплата) — фейковый, для макета. */
  factor: number
}

export const stresses: Stress[] = [
  { id: 'spread', label: 'Шок спреда ×1.5', factor: 0.88 },
  { id: 'skip', label: 'Пропуск 10% сделок', factor: 0.93 },
  { id: 'wr5', label: 'Винрейт −5 п.п.', factor: 0.72 },
  { id: 'wr10', label: 'Винрейт −10 п.п.', factor: 0.46 },
  { id: 'years', label: 'Худшие годы', factor: 0.81 },
]

/** Кривая P(1-я выплата) от риска фазы 1 — лог-нормальный горб с пиком в оптимуме (фейк). */
export function pOfRisk(risk: number, optimum: number, pMax: number): number {
  const z = Math.log(risk / optimum) / 0.55
  return pMax * Math.exp(-0.5 * z * z)
}

// ───────────── Bots ─────────────

export interface Bot {
  id: string
  name: string
  stage: 'S4 демо' | 'S5 челлендж'
  account: string
  status: 'Работает' | 'Пауза' | 'Остановлен'
  heartbeatSec: number
  openRiskPct: number
  dayPnlPct: number
  firmDailyLimitPct: number
  firmTotalLimitPct: number
  totalDdPct: number
  tradesToday: number
  tradesLimit: number
}

export const bots: Bot[] = [
  { id: 'BOT-01', name: 'Asia range breakout · XAUUSD', stage: 'S4 демо', account: 'Демо FN-like 25K', status: 'Работает', heartbeatSec: 4, openRiskPct: 0.5, dayPnlPct: -0.9, firmDailyLimitPct: 5, firmTotalLimitPct: 10, totalDdPct: 2.1, tradesToday: 2, tradesLimit: 6 },
  { id: 'BOT-02', name: 'NY close fade · GER40', stage: 'S4 демо', account: 'Демо FN-like 25K', status: 'Пауза', heartbeatSec: 7, openRiskPct: 0, dayPnlPct: -2.6, firmDailyLimitPct: 5, firmTotalLimitPct: 10, totalDdPct: 4.6, tradesToday: 3, tradesLimit: 6 },
]
