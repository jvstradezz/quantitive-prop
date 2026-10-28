import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { strategies, type Family, type Market, type StrategyRow } from '../data/fake'
import { Card, PageHeader, TableWrap } from '../components/ui'
import { pct, usd } from '../lib/format'

const markets: Market[] = ['Индексы', 'Металлы', 'Форекс', 'Сырьё']
const families: Family[] = ['Пробой диапазона', 'Возврат к среднему', 'Тренд', 'Сезонность']
const statuses: StrategyRow['status'][] = ['Ок', 'Демо', 'Пауза', 'Отклонена']

function Group<T extends string>({ label, items, value, onChange }: { label: string; items: T[]; value: T | null; onChange: (v: T | null) => void }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="label">{label}</span>
      <div className="flex gap-2 overflow-x-auto pb-1" role="group" aria-label={label}>
        {items.map((i) => (
          <button key={i} type="button" className="chip" aria-pressed={value === i} onClick={() => onChange(value === i ? null : i)}>{i}</button>
        ))}
      </div>
    </div>
  )
}

export default function Leaderboard() {
  const [market, setMarket] = useState<Market | null>(null)
  const [family, setFamily] = useState<Family | null>(null)
  const [status, setStatus] = useState<StrategyRow['status'] | null>(null)

  const rows = useMemo(
    () => strategies.filter((s) => (!market || s.market === market) && (!family || s.family === family) && (!status || s.status === status)).sort((a, b) => b.pPayout - a.pPayout),
    [market, family, status],
  )

  return (
    <>
      <PageHeader title="Leaderboard" lead="Стратегии по вероятности дойти до первой выплаты — на OOS и в демо, не по бэктесту." />
      <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Group label="Рынок" items={markets} value={market} onChange={setMarket} />
        <Group label="Семейство" items={families} value={family} onChange={setFamily} />
        <Group label="Статус" items={statuses} value={status} onChange={setStatus} />
      </div>
      <Card pad={false}>
        <TableWrap>
          <table className="tbl min-w-[960px]">
            <thead>
              <tr>
                <th className="r">№</th>
                <th>Стратегия</th>
                <th>Лучшая программа</th>
                <th className="r">P(1-я выплата)</th>
                <th className="r">EV на взнос</th>
                <th className="r">PF OOS</th>
                <th className="r">DSR</th>
                <th className="r">Сделок</th>
                <th>Стадия</th>
                <th className="r">Издержки / валовая</th>
                <th>Статус</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s, i) => (
                <tr key={s.id}>
                  <td className="num r text-ink-2">{i + 1}</td>
                  <td>
                    <Link to="/strategy" className="font-medium text-ink hover:text-accent">{s.name}</Link>
                    <div className="num text-[12px] leading-4 text-ink-2">{s.id} · {s.instrument}</div>
                  </td>
                  <td>{s.program}</td>
                  <td className="num r">{pct(s.pPayout)}</td>
                  <td className={`num r ${s.ev >= 0 ? 'text-pos' : 'text-neg'}`}>{usd(s.ev)}</td>
                  <td className="num r">{s.pfOos.toFixed(2)}</td>
                  <td className={`num r ${s.dsr >= 0.95 ? '' : 'text-ink-2'}`}>{s.dsr.toFixed(2)}</td>
                  <td className="num r">{s.trades}</td>
                  <td><span className="num rounded-md border border-line px-2 text-[12px] leading-5">{s.stage}</span></td>
                  <td className={`num r ${s.costShare > 0.3 ? 'font-medium text-neg' : ''}`}>{s.costShare > 0.3 && <span aria-label="флаг: издержки больше 30%">⚑ </span>}{pct(s.costShare)}</td>
                  <td className={s.status === 'Демо' ? 'text-pos' : s.status === 'Отклонена' ? 'text-neg' : s.status === 'Пауза' ? 'text-accent' : 'text-ink-2'}>{s.status}</td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr><td colSpan={11} className="!py-8 text-center text-ink-2">Под фильтр ничего не попало.</td></tr>
              )}
            </tbody>
          </table>
        </TableWrap>
      </Card>
      <p className="mt-4 text-[13px] leading-5 text-ink-2">⚑ — издержки больше 30% валовой прибыли. DSR ниже 0.95 приглушён: гейт S2 не пройден.</p>
    </>
  )
}
