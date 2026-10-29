import { Card, PageHeader, Stat, TableWrap } from '../components/ui'
import { SiteGate } from '../components/SiteGate'
import { repoFileUrl, type Milestone, type SiteData } from '../lib/site'

const tone = (s: Milestone['status']) => (s === 'готово' ? 'text-pos' : s === 'в работе' ? 'text-accent' : 'text-ink-2')

function MilestoneCard({ m }: { m: Milestone }) {
  return (
    <article className="card flex flex-col gap-3 p-4 sm:p-6">
      <div className="flex items-center justify-between gap-4">
        <span className="num text-[13px] leading-5 text-ink-2">{m.id}{m.depends !== '—' && <> · после {m.depends}</>}</span>
        <span className={`rounded-full border border-line px-3 text-[13px] leading-6 font-medium ${tone(m.status)}`}>{m.status}</span>
      </div>
      <h3>{m.title.charAt(0).toUpperCase() + m.title.slice(1)}</h3>
      {m.summary.toLowerCase() !== m.title.toLowerCase() && <p className="m-0 text-ink-2">{m.summary}</p>}
      {m.tasks && (
        <details>
          <summary className="cursor-pointer text-[14px] leading-5 font-medium text-accent">Задачи ({m.tasks.rows.length})</summary>
          <TableWrap>
            <table className="tbl mt-3 min-w-[480px]">
              <thead><tr>{m.tasks.header.map((h) => <th key={h}>{h}</th>)}</tr></thead>
              <tbody>
                {m.tasks.rows.map((r) => (
                  <tr key={r[0]}>{r.map((c, i) => <td key={i} className="align-top">{c}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </TableWrap>
        </details>
      )}
    </article>
  )
}

function Body({ data }: { data: SiteData }) {
  const done = data.milestones.filter((m) => m.status === 'готово').length
  const active = data.milestones.filter((m) => m.status === 'в работе').map((m) => m.id)
  const built = data.meta.built_at ? new Date(data.meta.built_at).toLocaleString('ru-RU', { timeZone: 'Europe/Moscow' }) : null
  return (
    <>
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Stat label="Вехи готовы" value={`${done} из ${data.milestones.length}`} sub={active.length > 0 ? `сейчас: ${active.join(', ')}` : undefined} />
        <Stat label="Идей в реестре" value={data.ideas.length} sub="ни одна ещё не тестировалась: движка нет (M2)" />
        <Stat label="Отчётов проп-MC" value={data.reports.prop.length} sub="пока на синтетических сделках" />
      </div>

      <h2 className="mb-4">Вехи</h2>
      <div className="mb-12 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {data.milestones.map((m) => <MilestoneCard key={m.id} m={m} />)}
      </div>

      <Card title="Отчёты в репозитории" note="файлы reports/">
        <ul className="m-0 list-none p-0">
          {data.reports.documents.map((d) => {
            const href = repoFileUrl(data.meta, d.path)
            return (
              <li key={d.path} className="flex flex-wrap items-baseline justify-between gap-x-4 border-b border-line py-2 text-[14px] leading-5 last:border-0">
                <span>{href ? <a href={href}>{d.title}</a> : d.title}</span>
                <span className="num text-[12px] text-ink-2">{d.path}</span>
              </li>
            )
          })}
        </ul>
      </Card>

      <p className="mt-8 text-[13px] leading-5 text-ink-2">
        Данные собраны из репозитория при сборке сайта{built && <>: <span className="num">{built} МСК</span></>}
        {data.meta.sha && <>, коммит <span className="num">{data.meta.sha.slice(0, 7)}</span></>}. Сайт обновляется после каждого merge в main.
      </p>
    </>
  )
}

export default function Status() {
  return (
    <>
      <PageHeader title="Статус проекта" lead="Где мы сейчас по плану, что в реестре идей и какие отчёты уже посчитаны. Всё читается из репозитория, ничего не вымышлено." />
      <SiteGate>{(data) => <Body data={data} />}</SiteGate>
    </>
  )
}
