import { useMemo, useState } from 'react'
import { pOfRisk, programs, stresses, type PropProgram } from '../data/fake'
import { RiskCurve } from '../components/charts'
import { Card, PageHeader, TableWrap } from '../components/ui'
import { pct, usd } from '../lib/format'

const reasons = [
  { key: 'daily', label: 'Дневной лимит', opacity: 1 },
  { key: 'total', label: 'Общий лимит', opacity: 0.7 },
  { key: 'time', label: 'Срок', opacity: 0.45 },
  { key: 'rule', label: 'Правило фирмы', opacity: 0.25 },
] as const

function adjust(p: PropProgram, f: number) {
  const clamp = (v: number) => Math.min(1, v)
  return {
    ...p,
    pPhase1: p.pPhase1 === null ? null : clamp(p.pPhase1 * f ** 0.4),
    pPhase2: p.pPhase2 === null ? null : clamp(p.pPhase2 * f ** 0.4),
    pFunded: clamp(p.pFunded * f ** 0.6),
    pPayout: p.pPayout * f,
    ciLo: p.ciLo * f,
    ciHi: p.ciHi * f,
    ev: (p.ev + p.fee) * f - p.fee,
  }
}

function PBar({ label, value }: { label: string; value: number | null }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-[14px] leading-5">
        <span>{label}</span>
        <span className="num">{value === null ? '—' : pct(value)}</span>
      </div>
      <div className="h-2 rounded-full bg-line">
        <div className="h-2 rounded-full bg-accent" style={{ width: `${(value ?? 0) * 100}%` }} />
      </div>
    </div>
  )
}

export default function PropSimulator() {
  const [active, setActive] = useState<string[]>([])
  const [selected, setSelected] = useState('fn-2s')

  const f = useMemo(() => stresses.filter((s) => active.includes(s.id)).reduce((a, s) => a * s.factor, 1), [active])
  const rows = useMemo(() => programs.map((p) => adjust(p, f)).sort((a, b) => b.pPayout - a.pPayout), [f])
  const best = rows[0]
  const sel = rows.find((r) => r.id === selected) ?? best

  const toggle = (id: string) => setActive((a) => (a.includes(id) ? a.filter((x) => x !== id) : [...a, id]))

  return (
    <>
      <PageHeader title="Prop simulator" lead="Одна стратегия против всех программ фирм: вероятность каждой фазы, причины провала и риск, который максимизирует шанс первой выплаты." />

      <div className="card mb-6 border-l-4 border-l-accent p-4 sm:p-6">
        <div className="label mb-1">Рекомендация · Asia range breakout · XAUUSD</div>
        <p className="m-0">
          Лучше всего — {best.firm} {best.name}: P(1-я выплата) <span className="num">{pct(best.pPayout)}</span> (<span className="num">{pct(best.ciLo)}–{pct(best.ciHi)}</span>), EV на взнос <span className="num">{usd(best.ev)}</span>, риск <span className="num">{best.risk.phase1}%</span>
          {best.risk.phase2 !== null && <> / <span className="num">{best.risk.phase2}%</span></>} / <span className="num">{best.risk.funded}%</span>.
        </p>
      </div>

      <div className="mb-6">
        <div className="label mb-2">Стресс-тесты</div>
        <div className="flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Переключатели стрессов">
          {stresses.map((s) => (
            <button key={s.id} type="button" className="chip" aria-pressed={active.includes(s.id)} onClick={() => toggle(s.id)}>{s.label}</button>
          ))}
        </div>
        <p className="mt-2 text-[13px] leading-5 text-ink-2">{active.length === 0 ? 'База: без стресса.' : `Совокупный множитель к P(1-я выплата): ×${f.toFixed(2)}.`} В макете множители условные.</p>
      </div>

      <Card title="Типы пропов" note="P по фазам, дни до funded и EV на один взнос" pad={false}>
        <TableWrap>
          <table className="tbl min-w-[980px]">
            <thead>
              <tr>
                <th>Программа</th>
                <th>Цели</th>
                <th className="r">Взнос</th>
                <th className="r">P фаза 1</th>
                <th className="r">P фаза 2</th>
                <th className="r">P funded</th>
                <th className="r">P 1-я выплата</th>
                <th className="r">Дни до funded (мед. / p90)</th>
                <th className="r">EV на взнос</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p, i) => (
                <tr key={p.id} className={p.id === sel.id ? 'bg-bg' : ''}>
                  <td>
                    <button type="button" onClick={() => setSelected(p.id)} aria-pressed={p.id === sel.id} className="cursor-pointer border-0 bg-transparent p-0 text-left font-medium text-ink hover:text-accent">
                      {i === 0 && <span className="mr-2 text-accent" aria-label="лучшая">★</span>}{p.name}
                    </button>
                    <div className="text-[12px] leading-4 text-ink-2">{p.firm}</div>
                  </td>
                  <td className="num">{p.phases}</td>
                  <td className="num r">{usd(p.fee, 2)}</td>
                  <td className="num r">{p.pPhase1 === null ? '—' : pct(p.pPhase1)}</td>
                  <td className="num r">{p.pPhase2 === null ? '—' : pct(p.pPhase2)}</td>
                  <td className="num r">{pct(p.pFunded)}</td>
                  <td className="num r font-medium">{pct(p.pPayout)}</td>
                  <td className="num r">{p.daysMedian} / {p.daysP90}</td>
                  <td className={`num r ${p.ev >= 0 ? 'text-pos' : 'text-neg'}`}>{usd(p.ev)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
      </Card>

      <h2 className="mb-6 mt-12">{sel.firm} · {sel.name}</h2>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Вероятности по фазам" note={<>95% ДИ первой выплаты: <span className="num">{pct(sel.ciLo)}–{pct(sel.ciHi)}</span></>}>
          <div className="flex flex-col gap-4">
            <PBar label="Фаза 1" value={sel.pPhase1} />
            <PBar label="Фаза 2" value={sel.pPhase2} />
            <PBar label="Funded" value={sel.pFunded} />
            <PBar label="1-я выплата" value={sel.pPayout} />
          </div>
        </Card>

        <Card title="Причины провалов" note="доли среди всех провалов">
          <div className="flex h-4 overflow-hidden rounded-full bg-line" role="img" aria-label="Распределение причин провала">
            {reasons.map((r) => (
              <i key={r.key} className="bg-accent" style={{ width: `${sel.fails[r.key] * 100}%`, opacity: r.opacity }} />
            ))}
          </div>
          <ul className="m-0 mt-4 list-none p-0">
            {reasons.map((r) => (
              <li key={r.key} className="flex items-center justify-between border-b border-line py-2 text-[14px] leading-5 last:border-0">
                <span><i className="mr-3 inline-block h-2 w-3 rounded-sm bg-accent" style={{ opacity: r.opacity }} />{r.label}</span>
                <span className="num">{pct(sel.fails[r.key])}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Риск по фазам" note="оптимум при ограничениях фирмы">
          <table className="tbl">
            <thead><tr><th className="!pl-0">Фаза</th><th className="r">Риск на сделку, % счёта</th></tr></thead>
            <tbody>
              <tr><td className="!pl-0">Фаза 1</td><td className="num r">{sel.risk.phase1.toFixed(1)}%</td></tr>
              {sel.risk.phase2 !== null && <tr><td className="!pl-0">Фаза 2</td><td className="num r">{sel.risk.phase2.toFixed(1)}%</td></tr>}
              <tr><td className="!pl-0">Funded</td><td className="num r">{sel.risk.funded.toFixed(1)}%</td></tr>
            </tbody>
          </table>
          <p className="mt-4 text-[13px] leading-5 text-ink-2">Правило снижения: после −50% дневного лимита фирмы бот режет риск вдвое и стоит до конца дня (предохранитель §10).</p>
        </Card>

        <Card title="P(1-я выплата) от риска" note="фаза 1">
          <RiskCurve optimum={sel.risk.phase1} pMax={Math.max(sel.pPayout, 0.05)} current={sel.risk.phase1} />
          <p className="sr-only">Значение в оптимуме: {pct(pOfRisk(sel.risk.phase1, sel.risk.phase1, sel.pPayout))}</p>
        </Card>
      </div>
    </>
  )
}
