import { useMemo, useState } from 'react'
import { Card, PageHeader, TableWrap } from '../components/ui'
import { SiteGate } from '../components/SiteGate'
import { pct, usd } from '../lib/format'
import type { MdTable, ProgramRow, PropReport } from '../lib/site'

const FIRMS = [
  { id: 'all', label: 'Все' },
  { id: 'fundednext', label: 'FundedNext' },
  { id: 'fundingpips', label: 'FundingPips' },
] as const
const COLLAPSED_ROWS = 15

const reasons = [
  { key: 'fail_guard_total', label: 'Предохранитель бота: общий лимит' },
  { key: 'fail_daily', label: 'Дневной лимит фирмы' },
  { key: 'fail_max', label: 'Общий лимит фирмы' },
  { key: 'fail_timeout', label: 'Срок' },
  { key: 'fail_inactivity', label: 'Неактивность' },
  { key: 'fail_rule', label: 'Правило фирмы' },
] as const

const firmName = (id: string) => FIRMS.find((f) => f.id === id)?.label ?? id
const num = (v: number | null, f: (x: number) => string) => (v === null ? '—' : f(v))
const evTone = (v: number | null) => (v === null ? '' : v >= 0 ? 'text-pos' : 'text-neg')

function PBar({ label, value }: { label: string; value: number | null }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-[14px] leading-5">
        <span>{label}</span>
        <span className="num">{num(value, (v) => pct(v, 1))}</span>
      </div>
      <div className="h-2 rounded-full bg-line">
        <div className="h-2 rounded-full bg-accent" style={{ width: `${(value ?? 0) * 100}%` }} />
      </div>
    </div>
  )
}

function MdTableView({ table, minWidth }: { table: MdTable; minWidth: number }) {
  return (
    <TableWrap>
      <table className="tbl" style={{ minWidth }}>
        <thead><tr>{table.header.map((h, i) => <th key={h} className={i === 0 ? '' : 'r'}>{h}</th>)}</tr></thead>
        <tbody>
          {table.rows.map((r) => (
            <tr key={r[0]}>{r.map((c, i) => <td key={i} className={i === 0 ? '' : 'num r'}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </TableWrap>
  )
}

function Report({ report }: { report: PropReport }) {
  const [firm, setFirm] = useState<(typeof FIRMS)[number]['id']>('all')
  const [expanded, setExpanded] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)

  const rows = useMemo(
    () =>
      report.programs
        .filter((p) => p.p1 !== null && (firm === 'all' || p.firm === firm))
        .sort((a, b) => Number((b.ev1 ?? 0) > 0) - Number((a.ev1 ?? 0) > 0) || (b.p1 ?? 0) - (a.p1 ?? 0)),
    [report, firm],
  )
  const templates = report.programs.filter((p) => p.p1 === null)
  const shown = expanded ? rows : rows.slice(0, COLLAPSED_ROWS)
  const sel: ProgramRow | undefined = rows.find((r) => r.key === selected) ?? rows[0]
  const synthetic = report.id.includes('synthetic')

  return (
    <>
      {synthetic && (
        <div className="card mb-6 p-4 text-[14px] leading-5 sm:p-6">
          <span className="font-medium text-accent">Это не рынок.</span> Отчёт посчитан на синтетических сделках с известным перевесом: так проверяют сам симулятор. Реальные стратегии появятся после движка бэктеста (M2).
        </div>
      )}
      {report.recommendation && (
        <div className="card mb-6 border-l-4 border-l-accent p-4 sm:p-6">
          <div className="label mb-1">Рекомендация · {report.title}</div>
          <p className="m-0">{report.recommendation}</p>
          {report.sample && <p className="mb-0 mt-2 text-[13px] leading-5 text-ink-2">{report.sample}</p>}
        </div>
      )}

      <div className="mb-4 flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Фильтр по фирме">
        {FIRMS.map((f) => (
          <button key={f.id} type="button" className="chip" aria-pressed={firm === f.id} onClick={() => { setFirm(f.id); setExpanded(false) }}>{f.label}</button>
        ))}
      </div>

      <Card title="Типы пропов" note={<>строк: <span className="num">{rows.length}</span>, сначала EV больше 0, затем по P(1-я выплата)</>} pad={false}>
        <TableWrap>
          <table className="tbl min-w-[1040px]">
            <thead>
              <tr>
                <th>Программа</th>
                <th>Риск Ф1 / Ф2 / funded</th>
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
              {shown.map((p) => (
                <tr key={p.key} className={p.key === sel?.key ? 'bg-bg' : ''}>
                  <td>
                    <button type="button" onClick={() => setSelected(p.key)} aria-pressed={p.key === sel?.key} className="cursor-pointer border-0 bg-transparent p-0 text-left font-medium text-ink hover:text-accent">
                      {p.label}
                    </button>
                    <div className="text-[12px] leading-4 text-ink-2">{firmName(p.firm)}</div>
                  </td>
                  <td className="num text-[13px]">{p.risk}</td>
                  <td className="num r">{num(p.cost_usd, (v) => usd(v, 2))}</td>
                  <td className="num r">{num(p.p_phase1, (v) => pct(v, 1))}</td>
                  <td className="num r">{num(p.p_phase2, (v) => pct(v, 1))}</td>
                  <td className="num r">{num(p.p_funded, (v) => pct(v, 1))}</td>
                  <td className="num r font-medium">{num(p.p1, (v) => pct(v, 1))}</td>
                  <td className="num r">{p.days_funded_median === null ? '—' : `${p.days_funded_median} / ${p.days_funded_p90}`}</td>
                  <td className={`num r ${evTone(p.ev1)}`}>{num(p.ev1, (v) => usd(v))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableWrap>
        {rows.length > COLLAPSED_ROWS && (
          <div className="border-t border-line p-3 text-center">
            <button type="button" className="btn btn-sm" onClick={() => setExpanded((e) => !e)}>{expanded ? 'Свернуть' : `Показать все ${rows.length}`}</button>
          </div>
        )}
      </Card>

      {sel && (
        <>
          <h2 className="mb-6 mt-12">{sel.label}</h2>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card
              title="Вероятности по фазам"
              note={sel.p1_lo_sample !== null && sel.p1_hi_sample !== null
                ? <>интервал по выборке: <span className="num">{pct(sel.p1_lo_sample, 1)}–{pct(sel.p1_hi_sample, 1)}</span></>
                : sel.p1_lo_mc !== null && sel.p1_hi_mc !== null
                  ? <>шум MC: <span className="num">{pct(sel.p1_lo_mc, 1)}–{pct(sel.p1_hi_mc, 1)}</span></>
                  : undefined}
            >
              <div className="flex flex-col gap-4">
                <PBar label="Фаза 1" value={sel.p_phase1} />
                <PBar label="Фаза 2" value={sel.p_phase2} />
                <PBar label="Funded" value={sel.p_funded} />
                <PBar label="Выплата после funded" value={sel.p_payout_given_funded} />
                <PBar label="1-я выплата" value={sel.p1} />
              </div>
            </Card>

            <Card title="Причины провалов" note="доли всех путей">
              <ul className="m-0 list-none p-0">
                {reasons.map((r) => (
                  <li key={r.key} className="border-b border-line py-2 text-[14px] leading-5 last:border-0">
                    <div className="flex items-center justify-between">
                      <span>{r.label}</span>
                      <span className="num">{num(sel[r.key], (v) => pct(v, 1))}</span>
                    </div>
                    <div className="mt-1 h-1 rounded-full bg-line"><div className="h-1 rounded-full bg-accent" style={{ width: `${(sel[r.key] ?? 0) * 100}%` }} /></div>
                  </li>
                ))}
              </ul>
            </Card>

            <Card title="Деньги" note="на один взнос">
              <table className="tbl">
                <tbody>
                  <tr><td className="!pl-0">Взнос</td><td className="num r">{num(sel.cost_usd, (v) => usd(v, 2))}</td></tr>
                  <tr><td className="!pl-0">EV на взнос</td><td className={`num r ${evTone(sel.ev1)}`}>{num(sel.ev1, (v) => usd(v))}</td></tr>
                  <tr><td className="!pl-0">EV за 180 дней</td><td className={`num r ${evTone(sel.ev180)}`}>{num(sel.ev180, (v) => usd(v))}</td></tr>
                  <tr><td className="!pl-0">P безубыточности</td><td className="num r">{num(sel.breakeven_p !== null && sel.breakeven_p <= 1 ? sel.breakeven_p : null, (v) => pct(v, 1))}</td></tr>
                  <tr><td className="!pl-0">Дни до 1-й выплаты (мед.)</td><td className="num r">{num(sel.days_payout_median, (v) => String(v))}</td></tr>
                </tbody>
              </table>
            </Card>
          </div>
        </>
      )}

      {report.costs && (
        <div className="mt-12">
          <Card title="Издержки стратегии" note={report.costs_note ?? undefined} pad={false}>
            <MdTableView table={report.costs} minWidth={480} />
          </Card>
        </div>
      )}

      {report.stresses && (
        <div className="mt-6">
          <Card title="Стрессы" note="P(1-я выплата) · EV на взнос; риск найден на базовом сценарии" pad={false}>
            <MdTableView table={report.stresses} minWidth={960} />
          </Card>
        </div>
      )}

      {templates.length > 0 && (
        <p className="mt-6 text-[13px] leading-5 text-ink-2">
          Шаблоны типов пропов (без выплат, для сравнения): {templates.map((t) => t.label).join('; ')}. Их числа условны (P-23 в docs/DECISIONS.md), в таблицу не входят.
        </p>
      )}
    </>
  )
}

function Body({ reports }: { reports: PropReport[] }) {
  const [id, setId] = useState(reports[0]?.id)
  const report = reports.find((r) => r.id === id) ?? reports[0]
  if (!report) return <div className="card p-8 text-center text-ink-2">Отчётов проп-MC в reports/ пока нет.</div>
  return (
    <>
      {reports.length > 1 && (
        <div className="mb-6 flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Отчёт">
          {reports.map((r) => (
            <button key={r.id} type="button" className="chip" aria-pressed={r.id === report.id} onClick={() => setId(r.id)}>{r.title}</button>
          ))}
        </div>
      )}
      <Report key={report.id} report={report} />
    </>
  )
}

export default function PropSimulator() {
  return (
    <>
      <PageHeader title="Prop simulator" lead="Одна стратегия против всех программ фирм: вероятность каждой фазы, причины провала и риск, который максимизирует шанс первой выплаты." />
      <SiteGate>{(data) => <Body reports={data.reports.prop} />}</SiteGate>
    </>
  )
}
