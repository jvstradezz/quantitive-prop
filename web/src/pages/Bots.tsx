import { useState } from 'react'
import { bots, type Bot } from '../data/fake'
import { PageHeader } from '../components/ui'

/** Шкала дневного убытка: 100% ширины = дневной лимит фирмы; риски — 50% (внутренний стоп) и 100% (лимит фирмы). */
function LimitGauge({ label, loss, limit, internalShare }: { label: string; loss: number; limit: number; internalShare: number }) {
  const used = Math.min(1, Math.max(0, loss / limit))
  const danger = used >= internalShare
  return (
    <div>
      <div className="mb-1 flex justify-between text-[14px] leading-5">
        <span>{label}</span>
        <span className="num">{loss.toFixed(1)}% из {limit}%</span>
      </div>
      <div className="relative h-3 rounded-full bg-line">
        <div className={`h-3 rounded-full ${danger ? 'bg-neg' : 'bg-accent'}`} style={{ width: `${used * 100}%` }} />
        <i className="absolute -top-1 h-5 w-px bg-ink" style={{ left: `${internalShare * 100}%` }} title="внутренний стоп" />
      </div>
      <div className="mt-1 flex justify-between text-[12px] leading-4 text-ink-2">
        <span>0</span>
        <span className="num">стоп бота {(limit * internalShare).toFixed(1)}%</span>
        <span className="num">лимит фирмы {limit}%</span>
      </div>
    </div>
  )
}

function BotCard({ bot }: { bot: Bot }) {
  const [armed, setArmed] = useState(false)
  const [killed, setKilled] = useState(false)
  const status = killed ? 'Остановлен' : bot.status
  const dot = status === 'Работает' ? 'bg-pos' : status === 'Пауза' ? 'bg-accent' : 'bg-neg'
  return (
    <article className="card flex flex-col gap-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="num text-[13px] leading-5 text-ink-2">{bot.id} · {bot.stage}</div>
          <h3 className="mt-1">{bot.name}</h3>
          <div className="text-[14px] leading-5 text-ink-2">{bot.account}</div>
        </div>
        <span className="inline-flex items-center gap-2 rounded-full border border-line px-3 text-[14px] leading-8">
          <i className={`inline-block h-2 w-2 rounded-full ${dot}`} />{status}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 text-[14px] leading-5">
        <div><div className="label">Heartbeat</div><div className="num mt-1">{killed ? '—' : `${bot.heartbeatSec} с назад`}</div></div>
        <div><div className="label">Открытый риск</div><div className="num mt-1">{killed ? '0.0' : bot.openRiskPct.toFixed(1)}% / 3%</div></div>
        <div><div className="label">Сделок сегодня</div><div className="num mt-1">{bot.tradesToday} / {bot.tradesLimit}</div></div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex justify-between text-[14px] leading-5">
          <span>P&amp;L дня</span>
          <span className={`num font-medium ${bot.dayPnlPct >= 0 ? 'text-pos' : 'text-neg'}`}>{bot.dayPnlPct > 0 ? '+' : bot.dayPnlPct < 0 ? '−' : ''}{Math.abs(bot.dayPnlPct).toFixed(1)}%</span>
        </div>
        <LimitGauge label="Дневной убыток против лимита фирмы" loss={Math.max(0, -bot.dayPnlPct)} limit={bot.firmDailyLimitPct} internalShare={0.5} />
        <LimitGauge label="Общая просадка против лимита фирмы" loss={bot.totalDdPct} limit={bot.firmTotalLimitPct} internalShare={0.6} />
      </div>

      <div className="mt-auto border-t border-line pt-4">
        {killed ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-[14px] leading-5 text-ink-2">Позиции закрыты, новые входы запрещены.</span>
            <button type="button" className="btn btn-sm" onClick={() => { setKilled(false); setArmed(false) }}>Снять остановку</button>
          </div>
        ) : armed ? (
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-[14px] leading-5">Закрыть позиции и остановить бота?</span>
            <button type="button" className="btn btn-sm !border-neg !bg-neg !text-surface" onClick={() => setKilled(true)}>Да, остановить</button>
            <button type="button" className="btn btn-sm" onClick={() => setArmed(false)}>Отмена</button>
          </div>
        ) : (
          <button type="button" className="btn !border-neg !text-neg" onClick={() => setArmed(true)}>Kill switch</button>
        )}
      </div>
    </article>
  )
}

export default function Bots() {
  return (
    <>
      <PageHeader title="Bots" lead="Статус, P&L дня против лимитов фирмы и внутренних предохранителей. Kill switch в макете ничего не отправляет." />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {bots.map((b) => <BotCard key={b.id} bot={b} />)}
      </div>
      <p className="mt-6 text-[13px] leading-5 text-ink-2">Внутренний дневной стоп — 50% дневного лимита фирмы (максимум 70%), общий — 60% общего лимита (инструкции проекта §10). Предохранители не отключаются.</p>
    </>
  )
}
