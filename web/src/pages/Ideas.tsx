import { useMemo, useState } from 'react'
import { IDEA_STATUSES, type IdeaStatus } from '../data/fake'
import { PageHeader } from '../components/ui'
import { SiteGate } from '../components/SiteGate'
import type { SiteIdea } from '../lib/site'

const tone = (s: IdeaStatus) => (s === 'Демо' ? 'text-pos' : s === 'Отклонена' ? 'text-neg' : s === 'Результат' ? 'text-accent' : 'text-ink-2')
const barTone = (s: IdeaStatus) => (s === 'Отклонена' ? 'bg-neg' : s === 'Демо' ? 'bg-pos' : 'bg-accent')

function Progress({ status }: { status: IdeaStatus }) {
  const idx = IDEA_STATUSES.indexOf(status)
  return (
    <div className="flex gap-1" role="img" aria-label={`Этап ${idx + 1} из ${IDEA_STATUSES.length}: ${status}`}>
      {IDEA_STATUSES.map((s, i) => (
        <i key={s} className={`h-1 flex-1 rounded-full ${i <= idx ? barTone(status) : 'bg-line'}`} />
      ))}
    </div>
  )
}

function IdeaCard({ idea }: { idea: SiteIdea }) {
  return (
    <article className="card flex flex-col gap-4 p-4 sm:p-6">
      <div className="flex items-center justify-between gap-4">
        <span className="num text-[13px] leading-5 text-ink-2">{idea.id} · {idea.created}</span>
        <span className={`rounded-full border border-line px-3 text-[13px] leading-6 font-medium ${tone(idea.status)}`}>{idea.status}</span>
      </div>
      <Progress status={idea.status} />
      <h3>{idea.title}</h3>
      <p className="m-0 text-ink-2">{idea.summary}</p>
      <div className="flex flex-wrap gap-2">
        {idea.markets.instruments.map((i) => (
          <span key={i} className="num rounded-md border border-line px-2 text-[12px] leading-5">{i}</span>
        ))}
        {idea.timeframes.map((t) => (
          <span key={t} className="num rounded-md border border-line px-2 text-[12px] leading-5 text-ink-2">{t}</span>
        ))}
      </div>
      <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[14px] leading-5">
        <dt className="label">Вход</dt><dd className="m-0">{idea.entry}</dd>
        <dt className="label">Стоп</dt><dd className="m-0">{idea.stop}</dd>
        <dt className="label">Тейк</dt><dd className="m-0">{idea.take}</dd>
      </dl>
      {idea.sessions.length > 0 && (
        <div className="overflow-x-auto">
          <table className="tbl min-w-[420px]">
            <thead>
              <tr><th className="!pl-0">Сессия</th><th>ET</th><th>МСК</th><th>Сервер FN</th></tr>
            </thead>
            <tbody>
              {idea.sessions.map((s) => (
                <tr key={s.name}>
                  <td className="!pl-0">{s.name}</td>
                  <td className="num">{s.et}</td>
                  <td className="num">{s.msk}</td>
                  <td className="num">{s.server}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {idea.assumptions.length > 0 && (
        <div>
          <div className="label mb-1">Допущения</div>
          <ul className="m-0 list-disc pl-5 text-[14px] leading-5 text-ink-2">
            {idea.assumptions.map((a) => <li key={a}>{a}</li>)}
          </ul>
        </div>
      )}
      <div className="mt-auto rounded-lg border border-line p-3 text-[14px] leading-5">
        <div className="num text-ink-2">Вариантов проверено: {idea.variants_tested}</div>
        <div className={`mt-1 font-medium ${idea.verdict ? tone(idea.status) : 'text-ink-2'}`}>{idea.verdict ?? 'Вердикта нет: тест начнётся после движка бэктеста (M2).'}</div>
      </div>
    </article>
  )
}

function Body({ ideas }: { ideas: SiteIdea[] }) {
  const [filter, setFilter] = useState<IdeaStatus | 'Все'>('Все')
  const counts = useMemo(() => {
    const c = new Map<string, number>()
    for (const i of ideas) c.set(i.status, (c.get(i.status) ?? 0) + 1)
    return c
  }, [ideas])
  const visible = filter === 'Все' ? ideas : ideas.filter((i) => i.status === filter)

  return (
    <>
      <div className="mb-6 flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Фильтр по статусу">
        {(['Все', ...IDEA_STATUSES] as const).map((s) => (
          <button key={s} type="button" className="chip" aria-pressed={filter === s} onClick={() => setFilter(s)}>
            {s} <span className="num ml-2 opacity-70">{s === 'Все' ? ideas.length : (counts.get(s) ?? 0)}</span>
          </button>
        ))}
      </div>
      {visible.length === 0 ? (
        <div className="card p-8 text-center text-ink-2">В этом статусе пока нет идей.</div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {visible.map((i) => <IdeaCard key={i.id} idea={i} />)}
        </div>
      )}
    </>
  )
}

export default function Ideas() {
  return (
    <>
      <PageHeader title="Идеи" lead="Реестр идей: карточки из research/registry/ и очередь из docs/IDEAS.md. Новую идею пишите в чат проекта: «Идея: …». Форма на сайте с GitHub Issue появится в M5." />
      <SiteGate>{(data) => <Body ideas={data.ideas} />}</SiteGate>
    </>
  )
}
