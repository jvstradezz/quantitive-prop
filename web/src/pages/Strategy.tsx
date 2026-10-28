import { useState } from 'react'
import { costs, strategyMeta, strategyStats, trades } from '../data/fake'
import { DrawdownChart, EquityChart, FanChart, ParamHeatmap, WalkForwardBars } from '../components/charts'
import { Card, PageHeader, Stat, TableWrap } from '../components/ui'
import { pct, signed, usd } from '../lib/format'
import { dateShort, serverFn, timeEt, timeMsk } from '../lib/time'

export default function Strategy() {
  const [shown, setShown] = useState(12)
  const last = [...trades].reverse().slice(0, shown)
  const s = strategyStats
  return (
    <>
      <PageHeader
        title={strategyMeta.name}
        lead={
          <>
            <span className="num">{strategyMeta.id}</span> · счёт <span className="num">{usd(strategyMeta.account)}</span> · риск <span className="num">{strategyMeta.riskPct}%</span> на сделку · стадия <span className="num">{strategyMeta.stage}</span>
          </>
        }
      />
      <div className="card mb-8 border-l-4 border-l-accent p-4 sm:p-6">
        <div className="label mb-1">Вердикт</div>
        <p className="m-0">{strategyMeta.verdict}</p>
      </div>

      <div className="mb-8 grid grid-cols-2 gap-4 lg:grid-cols-6">
        <Stat label="Сделок" value={s.trades} />
        <Stat label="Винрейт" value={pct(s.winRate)} />
        <Stat label="PF (чистый)" value={s.pf.toFixed(2)} />
        <Stat label="Матожидание" value={`${signed(s.expectancyR)} R`} tone="pos" />
        <Stat label="Чистый результат" value={usd(s.netUsd)} tone={s.netUsd >= 0 ? 'pos' : 'neg'} />
        <Stat label="Макс. просадка" value={`${s.maxDd.toFixed(1)}%`} tone="neg" />
      </div>

      <div className="grid grid-cols-1 gap-6">
        <Card title="Эквити" note="серым — in-sample (~70%), тёмным — out-of-sample">
          <EquityChart />
          <div className="mt-4">
            <div className="label mb-1">Просадка</div>
            <DrawdownChart />
          </div>
        </Card>

        <Card title="Веер Monte Carlo" note="1 500 путей · бутстрэп блоками по дням · seed 42 · 120 торговых дней">
          <FanChart />
          <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-[13px] leading-5 text-ink-2">
            <span><i className="mr-2 inline-block h-2 w-3 rounded-sm bg-accent" />медиана</span>
            <span><i className="mr-2 inline-block h-2 w-3 rounded-sm bg-accent/25" />25–75%</span>
            <span><i className="mr-2 inline-block h-2 w-3 rounded-sm bg-accent/12" />5–95%</span>
          </div>
        </Card>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card title="Walk-forward" note="PF по окнам, OOS против IS">
            <WalkForwardBars />
          </Card>
          <Card title="Тепловая карта параметров">
            <ParamHeatmap />
          </Card>
        </div>

        <Card title="Издержки" note={<>валовая <span className="num">{usd(costs.grossUsd)}</span></>} pad={false}>
          <div className="px-4 pb-2 pt-4 sm:px-6">
            <div className="flex h-3 overflow-hidden rounded-full bg-line" role="img" aria-label="Доли издержек от валовой прибыли">
              {costs.rows.map((c, i) => (
                <i key={c.name} style={{ width: `${c.shareOfGross * 100}%`, opacity: 1 - i * 0.2 }} className="bg-accent" />
              ))}
            </div>
          </div>
          <TableWrap>
            <table className="tbl min-w-[520px]">
              <thead>
                <tr><th>Статья</th><th className="r">$</th><th className="r">% счёта</th><th className="r">R</th><th className="r">Доля валовой</th></tr>
              </thead>
              <tbody>
                {costs.rows.map((c) => (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td className="num r">{usd(-c.usd)}</td>
                    <td className="num r">{(-c.pctAccount).toFixed(2)}%</td>
                    <td className="num r">{signed(-c.r, 1)}</td>
                    <td className="num r">{pct(c.shareOfGross, 1)}</td>
                  </tr>
                ))}
                <tr className="font-medium">
                  <td>Издержки всего</td>
                  <td className="num r">{usd(-costs.totalUsd)}</td>
                  <td className="num r">{(-(costs.totalUsd / strategyMeta.account) * 100).toFixed(2)}%</td>
                  <td className="num r">{signed(costs.netR - costs.grossR, 1)}</td>
                  <td className={`num r ${costs.totalShare > 0.3 ? 'text-neg' : ''}`}>{costs.totalShare > 0.3 && '⚑ '}{pct(costs.totalShare, 1)}</td>
                </tr>
                <tr className="font-medium">
                  <td>Чистый результат</td>
                  <td className="num r">{usd(costs.netUsd)}</td>
                  <td className="num r">{((costs.netUsd / strategyMeta.account) * 100).toFixed(2)}%</td>
                  <td className="num r">{signed(costs.netR, 1)}</td>
                  <td />
                </tr>
              </tbody>
            </table>
          </TableWrap>
        </Card>

        <Card title="Сделки" note="время: ET · МСК · сервер FN" pad={false}>
          <TableWrap>
            <table className="tbl min-w-[860px]">
              <thead>
                <tr><th>№</th><th>Дата (ET)</th><th>Вход ET</th><th>Вход МСК</th><th>Сервер FN</th><th>Сторона</th><th className="r">R валовая</th><th className="r">R чистая</th><th className="r">P&amp;L</th></tr>
              </thead>
              <tbody>
                {last.map((t) => {
                  const d = new Date(t.openUtc)
                  return (
                    <tr key={t.id}>
                      <td className="num text-ink-2">{t.id}</td>
                      <td className="num">{dateShort(d)}</td>
                      <td className="num">{timeEt(d)}</td>
                      <td className="num">{timeMsk(d)}</td>
                      <td className="num">{serverFn(d)}</td>
                      <td>{t.side}</td>
                      <td className="num r">{signed(t.rGross)}</td>
                      <td className={`num r ${t.rNet >= 0 ? 'text-pos' : 'text-neg'}`}>{signed(t.rNet)}</td>
                      <td className={`num r ${t.pnl >= 0 ? 'text-pos' : 'text-neg'}`}>{usd(t.pnl, 2)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </TableWrap>
          {shown < trades.length && (
            <div className="border-t border-line p-4 text-center">
              <button type="button" className="btn btn-sm" onClick={() => setShown((n) => n + 12)}>Показать ещё</button>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}
