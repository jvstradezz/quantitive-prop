import { ColorType, createChart, CrosshairMode, type IChartApi } from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import { useTokens, type Tokens } from '../lib/tokens'

interface Props {
  height?: number
  label: string
  grid?: boolean
  /** Должна быть стабильной (useCallback): смена ссылки пересоздаёт график. */
  setup: (chart: IChartApi, t: Tokens) => void
}

/** Обёртка Lightweight Charts: тема из токенов, линии тонкие, сетка — `--line` (docs/DESIGN.md §5). */
export function ChartFrame({ height = 280, label, grid = true, setup }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const t = useTokens()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = createChart(el, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: t.text2, fontFamily: t.mono, fontSize: 12, attributionLogo: true },
      grid: { vertLines: { visible: false }, horzLines: { color: t.line, visible: grid } },
      rightPriceScale: { borderColor: t.line },
      timeScale: { borderColor: t.line, timeVisible: false },
      crosshair: { mode: CrosshairMode.Magnet, vertLine: { color: t.text2, labelBackgroundColor: t.text }, horzLine: { color: t.text2, labelBackgroundColor: t.text } },
      handleScroll: { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false },
      handleScale: { mouseWheel: false, pinch: false, axisPressedMouseMove: false, axisDoubleClickReset: false },
    })
    setup(chart, t)
    chart.timeScale().fitContent()
    return () => chart.remove()
  }, [t, setup, grid])

  return <div ref={ref} style={{ height }} role="img" aria-label={label} />
}
