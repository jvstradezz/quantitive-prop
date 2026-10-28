import { AreaSeries, BaselineSeries, LineSeries, LineStyle, type IChartApi, type Time } from 'lightweight-charts'
import { useCallback, useEffect, useRef, useState } from 'react'
import { drawdownCurve, equityCurve, fan, fanStart, oosStart, pOfRisk, walkForward, heatmap } from '../data/fake'
import { mix, parseColor, withAlpha } from '../lib/color'
import { useTokens, type Tokens } from '../lib/tokens'
import { ChartFrame } from './ChartFrame'

const money = (p: number) => `$${Math.round(p).toLocaleString('ru-RU')}`

export function EquityChart() {
  const setup = useCallback((chart: IChartApi, t: Tokens) => {
    const fmt = { type: 'custom' as const, formatter: money, minMove: 1 }
    const is = chart.addSeries(LineSeries, { color: t.text2, lineWidth: 1, priceFormat: fmt, lastValueVisible: false, priceLineVisible: false })
    is.setData(equityCurve.filter((p) => p.time <= oosStart).map((p) => ({ time: p.time as Time, value: p.value })))
    const oos = chart.addSeries(LineSeries, { color: t.text, lineWidth: 2, priceFormat: fmt, priceLineVisible: false })
    oos.setData(equityCurve.filter((p) => p.time >= oosStart).map((p) => ({ time: p.time as Time, value: p.value })))
  }, [])
  return <ChartFrame height={300} label="Кривая капитала: in-sample серым, out-of-sample тёмным" setup={setup} />
}

export function DrawdownChart() {
  const setup = useCallback((chart: IChartApi, t: Tokens) => {
    const s = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 0 },
      topLineColor: t.neg,
      bottomLineColor: t.neg,
      topFillColor1: 'transparent',
      topFillColor2: 'transparent',
      bottomFillColor1: withAlpha(t.neg, 0.28),
      bottomFillColor2: withAlpha(t.neg, 0.28),
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      priceFormat: { type: 'custom', formatter: (p: number) => `${p.toFixed(1)}%`, minMove: 0.1 },
    })
    s.setData(drawdownCurve.map((p) => ({ time: p.time as Time, value: p.value })))
  }, [])
  return <ChartFrame height={180} label="Просадка в процентах от пика" setup={setup} />
}

/**
 * Веер MC: полосы рисуются непрозрачными заливками «сверху вниз» (p95 → p75 → p25 → p5 цветом поверхности),
 * потому что Lightweight Charts не умеет заливку между двумя линиями. Ровные цвета, без градиентов.
 */
export function FanChart() {
  const setup = useCallback((chart: IChartApi, t: Tokens) => {
    const outer = mix(t.surface, t.accent, 0.12)
    const inner = mix(t.surface, t.accent, 0.24)
    const fmt = { type: 'custom' as const, formatter: money, minMove: 1 }
    const hist = chart.addSeries(LineSeries, { color: t.text, lineWidth: 2, priceFormat: fmt, lastValueVisible: false, priceLineVisible: false })
    hist.setData(equityCurve.slice(-90).map((p) => ({ time: p.time as Time, value: p.value })))
    const band = (color: string, key: 'p95' | 'p75' | 'p25' | 'p5') => {
      const s = chart.addSeries(AreaSeries, { lineVisible: false, topColor: color, bottomColor: color, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, priceFormat: fmt })
      s.setData([{ time: fanStart.time as Time, value: fanStart.value }, ...fan.map((f) => ({ time: f.time as Time, value: f[key] }))])
    }
    band(outer, 'p95')
    band(inner, 'p75')
    band(outer, 'p25')
    band(t.surface, 'p5')
    const med = chart.addSeries(LineSeries, { color: t.accent, lineWidth: 2, priceFormat: fmt, lastValueVisible: false, priceLineVisible: false })
    med.setData([{ time: fanStart.time as Time, value: fanStart.value }, ...fan.map((f) => ({ time: f.time as Time, value: f.p50 }))])
    const start = chart.addSeries(LineSeries, { color: t.text2, lineWidth: 1, lineStyle: LineStyle.Dashed, priceFormat: fmt, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false })
    start.setData(fan.map((f) => ({ time: f.time as Time, value: fanStart.value })))
  }, [])
  return <ChartFrame height={300} grid={false} label="Веер Monte Carlo: медиана, полосы 25–75 и 5–95 процентов на 120 торговых дней вперёд" setup={setup} />
}

/** Гистограмма PF по окнам walk-forward — на CSS, чтобы подписи не масштабировались вместе с картинкой. */
export function WalkForwardBars() {
  const max = 1.8
  return (
    <div>
      <div className="relative flex h-48 items-end gap-2 border-b border-line sm:gap-4">
        <div className="absolute inset-x-0 border-t border-dashed border-ink-2" style={{ bottom: `${(1.3 / max) * 100}%` }} aria-hidden="true" />
        <span className="num absolute left-0 text-[12px] leading-4 text-ink-2" style={{ bottom: `calc(${(1.3 / max) * 100}% + 2px)` }}>PF 1.3</span>
        {walkForward.map((w) => (
          <div key={w.window} className="flex h-full flex-1 items-end justify-center gap-1">
            <div className="w-1/3 rounded-t-[4px] bg-ink-2/50" style={{ height: `${(w.isPf / max) * 100}%` }} title={`IS ${w.isPf}`} />
            <div className="w-1/3 rounded-t-[4px] bg-accent" style={{ height: `${(w.oosPf / max) * 100}%` }} title={`OOS ${w.oosPf}`} />
          </div>
        ))}
      </div>
      <div className="mt-2 flex gap-2 sm:gap-4">
        {walkForward.map((w) => (
          <div key={w.window} className="flex-1 text-center">
            <div className="text-[12px] leading-4 text-ink-2">{w.window}</div>
            <div className="num text-[12px] leading-4">{w.oosPf.toFixed(2)}</div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-[13px] leading-5 text-ink-2">
        <span><i className="mr-2 inline-block h-2 w-3 rounded-sm bg-ink-2/50" />in-sample</span>
        <span><i className="mr-2 inline-block h-2 w-3 rounded-sm bg-accent" />out-of-sample</span>
        <span>WFE <span className="num text-ink">81%</span></span>
      </div>
    </div>
  )
}

function luminance(c: string): number {
  const [r, g, b] = parseColor(c).map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/** Тепловая карта PF по параметрам: последовательная шкала surface → accent, текущая точка обведена. */
export function ParamHeatmap() {
  const t = useTokens()
  const flat = heatmap.values.flat()
  const lo = Math.min(...flat)
  const hi = Math.max(...flat)
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="num w-full min-w-[320px] border-separate border-spacing-1 text-[13px] leading-5">
          <thead>
            <tr>
              <th className="text-left text-[12px] font-normal text-ink-2">окно, мин ╲ цель, R</th>
              {heatmap.cols.map((c) => (
                <th key={c} className="text-[12px] font-normal text-ink-2">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {heatmap.values.map((row, i) => (
              <tr key={heatmap.rows[i]}>
                <th className="text-left text-[12px] font-normal text-ink-2">{heatmap.rows[i]}</th>
                {row.map((v, j) => {
                  const bg = mix(t.surface, t.accent, 0.08 + 0.82 * ((v - lo) / (hi - lo)))
                  const current = heatmap.current.row === i && heatmap.current.col === j
                  return (
                    <td
                      key={j}
                      className="h-10 rounded-md text-center"
                      style={{ background: bg, color: luminance(bg) > 0.3 ? '#1F1E1C' : '#FBFAF7', outline: current ? `2px solid ${t.text}` : 'none', outlineOffset: 1 }}
                      title={current ? 'текущие параметры' : undefined}
                    >
                      {v.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[13px] leading-5 text-ink-2">PF после издержек × 1.5. Обведена точка текущих параметров; соседи ±20% держат PF ≥ 1.15 — гейт S2.</p>
    </div>
  )
}

/** Ширина контейнера, чтобы SVG-график рисовался в пикселях и подписи были чёткими. */
function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(480)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Кривая P(1-я выплата) от риска фазы 1 (фейковая лог-нормальная форма). */
export function RiskCurve({ optimum, pMax, current }: { optimum: number; pMax: number; current: number }) {
  const t = useTokens()
  const [ref, w] = useWidth()
  const h = 200
  const m = { l: 40, r: 8, t: 8, b: 28 }
  const xs = Array.from({ length: 60 }, (_, i) => 0.1 + (i * (2 - 0.1)) / 59)
  const X = (r: number) => m.l + ((r - 0.1) / 1.9) * (w - m.l - m.r)
  const Y = (p: number) => m.t + (1 - p / 0.6) * (h - m.t - m.b)
  const path = xs.map((r, i) => `${i ? 'L' : 'M'}${X(r).toFixed(1)},${Y(pOfRisk(r, optimum, pMax)).toFixed(1)}`).join(' ')
  return (
    <div ref={ref}>
      <svg width={w} height={h} role="img" aria-label="Вероятность первой выплаты в зависимости от риска на сделку в первой фазе" className="block">
        {[0, 0.2, 0.4, 0.6].map((p) => (
          <g key={p}>
            <line x1={m.l} x2={w - m.r} y1={Y(p)} y2={Y(p)} stroke={t.line} />
            <text x={m.l - 8} y={Y(p) + 4} textAnchor="end" fontSize="12" fill={t.text2} fontFamily={t.mono}>{Math.round(p * 100)}%</text>
          </g>
        ))}
        {[0.25, 0.5, 1, 1.5, 2].map((r) => (
          <text key={r} x={X(r)} y={h - 8} textAnchor="middle" fontSize="12" fill={t.text2} fontFamily={t.mono}>{r}%</text>
        ))}
        <path d={path} fill="none" stroke={t.accent} strokeWidth="1.5" />
        <line x1={X(optimum)} x2={X(optimum)} y1={m.t} y2={h - m.b} stroke={t.text2} strokeDasharray="3 3" />
        <circle cx={X(current)} cy={Y(pOfRisk(current, optimum, pMax))} r="4" fill={t.surface} stroke={t.text} strokeWidth="1.5" />
      </svg>
      <p className="mt-2 text-[13px] leading-5 text-ink-2">Пунктир — оптимум риска, кружок — выбранное значение. Ось X — риск на сделку, % счёта.</p>
    </div>
  )
}
