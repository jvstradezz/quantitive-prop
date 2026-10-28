import { useMemo, useState, type FormEvent } from 'react'
import { IDEA_STATUSES, ideas as seed, type Idea, type IdeaStatus } from '../data/fake'
import { PageHeader } from '../components/ui'

const tone = (s: IdeaStatus) => (s === 'Демо' ? 'text-pos' : s === 'Отклонена' ? 'text-neg' : s === 'Результат' ? 'text-accent' : 'text-ink-2')
const barTone = (s: IdeaStatus) => (s === 'Отклонена' ? 'bg-neg' : s === 'Демо' ? 'bg-pos' : 'bg-accent')

const attach = [
  { label: 'Скриншот', d: 'M4 5h16v14H4zM4 15l5-5 4 4 3-3 4 4' },
  { label: 'Голос', d: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v4' },
  { label: 'Ссылка', d: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1' },
  { label: 'Видео', d: 'M3 6h12v12H3zM15 10l6-3v10l-6-3' },
]

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

function IdeaCard({ idea }: { idea: Idea }) {
  const [rework, setRework] = useState(false)
  return (
    <article className="card flex flex-col gap-4 p-4 sm:p-6">
      <div className="flex items-center justify-between gap-4">
        <span className="num text-[13px] leading-5 text-ink-2">{idea.id} · {idea.created}</span>
        <span className={`rounded-full border border-line px-3 text-[13px] leading-6 font-medium ${tone(idea.status)}`}>{idea.status}</span>
      </div>
      <Progress status={idea.status} />
      <h3>{idea.title}</h3>
      <p className="text-ink-2">{idea.summary}</p>
      <div className="flex flex-wrap gap-2">
        {idea.instruments.map((i) => (
          <span key={i} className="num rounded-md border border-line px-2 text-[12px] leading-5">{i}</span>
        ))}
      </div>
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
      {(idea.result || idea.verdict) && (
        <div className="rounded-lg border border-line p-3 text-[14px] leading-5">
          {idea.result && <div className="num text-ink-2">{idea.result}</div>}
          {idea.verdict && <div className={`font-medium ${tone(idea.status)} ${idea.result ? 'mt-1' : ''}`}>{idea.verdict}</div>}
        </div>
      )}
      <div className="mt-auto">
        {rework ? (
          <form className="flex flex-col gap-2" onSubmit={(e) => { e.preventDefault(); setRework(false) }}>
            <label className="sr-only" htmlFor={`rw-${idea.id}`}>Что доработать</label>
            <textarea id={`rw-${idea.id}`} className="field min-h-20 w-full" placeholder="Что изменить: фильтр, выход, инструменты…" />
            <div className="flex gap-2">
              <button className="btn btn-accent btn-sm" type="submit">Отправить</button>
              <button className="btn btn-sm" type="button" onClick={() => setRework(false)}>Отмена</button>
            </div>
          </form>
        ) : (
          <button className="btn btn-sm" type="button" onClick={() => setRework(true)}>Доработать</button>
        )}
      </div>
    </article>
  )
}

export default function Ideas() {
  const [list, setList] = useState<Idea[]>(seed)
  const [filter, setFilter] = useState<IdeaStatus | 'Все'>('Все')
  const [text, setText] = useState('')

  const counts = useMemo(() => {
    const c = new Map<string, number>()
    for (const i of list) c.set(i.status, (c.get(i.status) ?? 0) + 1)
    return c
  }, [list])

  const visible = filter === 'Все' ? list : list.filter((i) => i.status === filter)

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const body = text.trim()
    if (!body) return
    const title = body.replace(/^Идея:\s*/i, '').split('\n')[0].slice(0, 80)
    setList([
      {
        id: `IDEA-${String(list.length + 1).padStart(4, '0')}`,
        title,
        status: 'Новая',
        created: '2026-10-28',
        summary: body.replace(/^Идея:\s*/i, ''),
        instruments: [],
        sessions: [],
        assumptions: [],
      },
      ...list,
    ])
    setText('')
    setFilter('Все')
  }

  return (
    <>
      <PageHeader title="Идеи" lead="Опишите идею как в чате — текстом, голосом, скрином, ссылкой или видео. Она станет карточкой и пойдёт по конвейеру до вердикта." />

      <form onSubmit={submit} className="card mb-8 p-4 sm:p-6">
        <label htmlFor="idea-text" className="sr-only">Текст идеи</label>
        <textarea id="idea-text" value={text} onChange={(e) => setText(e.target.value)} className="field min-h-28 w-full resize-y text-[16px] leading-6" placeholder="Идея: например, торговать пробой азиатского диапазона на золоте, стоп за противоположную границу…" />
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex gap-2">
            {attach.map((a) => (
              <button key={a.label} type="button" className="btn btn-sm !px-2" title={`${a.label} (в макете не работает)`} aria-label={a.label}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={a.d} /></svg>
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-[13px] leading-5 text-ink-2 sm:inline">Создаст GitHub Issue с меткой idea (M5)</span>
            <button type="submit" className="btn btn-accent" disabled={!text.trim()}>Отправить идею</button>
          </div>
        </div>
      </form>

      <div className="mb-6 flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Фильтр по статусу">
        {(['Все', ...IDEA_STATUSES] as const).map((s) => (
          <button key={s} type="button" className="chip" aria-pressed={filter === s} onClick={() => setFilter(s)}>
            {s} <span className="num ml-2 opacity-70">{s === 'Все' ? list.length : (counts.get(s) ?? 0)}</span>
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
